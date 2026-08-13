"""Public mode must carry no SAE jargon or raw values; Advanced must
render the exact same canonical ResolvedControlState Public used, never a
second, separately-computed technical view."""

from __future__ import annotations

import json

from sae_concept_lab.canonical.concept_bundle import decode_entry, resolve_control
from sae_concept_lab.core.logic import (
    GENERATED_ONLY_POSITIONS_DISCLOSURE,
    advanced_output_details,
    advanced_positions_text,
    public_output_summary,
)
from sae_concept_lab.fixtures.loader import load_entries

GEMMA_ENTRIES = load_entries("gemma")
QWEN_ENTRIES = load_entries("qwen")


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


# ---------------------------------------------------------------------------
# Positions: public default is ALL (2026-08-13 researcher ruling); an
# ATTESTED entry's own ratified position remains authoritative;
# GENERATED_ONLY surfaces the exact disclosure sentence in Advanced.
# ---------------------------------------------------------------------------


def _generated_only_entry():
    document = json.dumps({
        "schema_version": "1.0",
        "concept_id": "test-only-generated-only-positions-check",
        "pairing_id": "test-only-pairing",
        "positions": "generated_only",
        "provenance": "fake",
        "calibration_provenance": None,
        "directions": {
            "amplify": {
                "targets": [{"sae_id": "test-sae", "layer": 0, "feature_idx": 0, "weight": 1.0}],
                "specs": {
                    strength: {"operation": "ablate", "value": None, "unit": None, "unit_source": None}
                    for strength in ("low", "medium", "high")
                },
            },
            "suppress": None,
        },
    })
    return decode_entry(document, where="generated-only-positions-check")


def test_every_shipped_fake_fixture_defaults_to_all_positions():
    """Public default: ALL (2026-08-13 researcher ruling). None of this
    repository's own shipped FAKE fixtures carries ATTESTED-level
    ratification for GENERATED_ONLY, so every one of them uses the public
    default -- a fixture-authoring fact, not something any code path
    enforces at resolution time."""
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        assert entry.positions.value == "all", f"{entry.concept_id} does not use the public default (all)"


def test_advanced_positions_text_shows_all_with_no_disclosure_for_shipped_fixtures():
    entry = GEMMA_ENTRIES[0]
    text = advanced_positions_text(entry, "en")
    assert "all" in text
    assert GENERATED_ONLY_POSITIONS_DISCLOSURE not in text


def test_advanced_positions_text_shows_generated_only_disclosure_when_ratified():
    """An entry that DOES carry a ratified GENERATED_ONLY position (an
    ATTESTED entry's own choice is exactly this case) still renders
    correctly and unconditionally surfaces the exact disclosure sentence
    -- GENERATED_ONLY remains fully available, never hidden or degraded."""
    entry = _generated_only_entry()
    text = advanced_positions_text(entry, "en")
    assert "generated_only" in text
    assert GENERATED_ONLY_POSITIONS_DISCLOSURE in text


def test_advanced_positions_text_disclosure_is_identical_across_languages():
    """The disclosure is fixed, precise, canonical prose -- like
    _capability_notice's PROHIBITED/CAPABILITY_LIMIT reasons, it is never
    paraphrased per language; only the label prefix is localized."""
    entry = _generated_only_entry()
    text_en = advanced_positions_text(entry, "en")
    text_fr = advanced_positions_text(entry, "fr")
    assert GENERATED_ONLY_POSITIONS_DISCLOSURE in text_en
    assert GENERATED_ONLY_POSITIONS_DISCLOSURE in text_fr
