"""Public mode must carry no SAE jargon or raw values; Advanced must
render the exact same canonical ResolvedControlState Public used, never a
second, separately-computed technical view."""

from __future__ import annotations

from sae_concept_lab.canonical.concept_bundle import resolve_control
from sae_concept_lab.core.logic import advanced_output_details, public_output_summary
from sae_concept_lab.fixtures.loader import load_entries

GEMMA_ENTRIES = load_entries("gemma")


def _resolved():
    entry = GEMMA_ENTRIES[0]
    return resolve_control(entry, direction=entry.calibrated_directions[0], strength="high")


RAW_TECHNICAL_TERMS = ("feature_idx", "sae_id", "unit_source", "layer", "targets", "operation")


def test_public_summary_never_mentions_raw_technical_field_names():
    cfg = _resolved()
    summary = public_output_summary(cfg, "en").lower()
    for term in RAW_TECHNICAL_TERMS:
        assert term not in summary, f"public summary leaked technical term {term!r}"


def test_public_summary_never_contains_the_raw_feature_or_sae_values():
    cfg = _resolved()
    summary = public_output_summary(cfg, "en")
    for target in cfg.targets:
        assert str(target.feature_idx) not in summary
        assert target.sae_id not in summary


def test_public_summary_does_state_model_concept_direction_strength():
    from sae_concept_lab.fixtures.labels import concept_label, pairing_label

    cfg = _resolved()
    summary_en = public_output_summary(cfg, "en")
    assert pairing_label(cfg.pairing_id, "en") in summary_en
    assert concept_label(cfg.concept_id, "en") in summary_en
    assert "Amplify" in summary_en
    assert "High" in summary_en

    summary_fr = public_output_summary(cfg, "fr")
    assert concept_label(cfg.concept_id, "fr") in summary_fr
    assert "Amplifier" in summary_fr
    assert "Élevée" in summary_fr


def test_advanced_details_do_contain_every_raw_technical_field():
    cfg = _resolved()
    details = advanced_output_details(cfg)
    assert details["targets"] == [t.as_dict() for t in cfg.targets]
    assert details["positions"] == cfg.positions.value
    assert details["operation"] == cfg.operation.value
    assert details["execution_payload"] == cfg.execution_dict()
    assert details["state_fingerprint"] == cfg.state_fingerprint()
    assert details["execution_fingerprint"] == cfg.execution_fingerprint()


def test_advanced_details_is_the_same_object_public_summarized_not_a_recomputation():
    """The literal 'one shared state object' acceptance check: build ONE
    ResolvedControlState, pass THAT SAME instance to both renderers, and
    assert advanced's dict is exactly its own advanced_view() plus
    execution_dict() -- there is no second construction path in this
    codebase that could drift from either."""
    cfg = _resolved()
    _public_text = public_output_summary(cfg, "en")  # must not mutate cfg
    advanced = advanced_output_details(cfg)
    assert advanced == {**cfg.advanced_view(), "execution_payload": cfg.execution_dict()}
