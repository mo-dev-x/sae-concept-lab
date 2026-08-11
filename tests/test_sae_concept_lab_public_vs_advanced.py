"""Public mode must carry no SAE jargon or raw values; Advanced must
render the exact same ResolvedConfig Public used, never a second,
separately-computed technical view."""

from __future__ import annotations

from sae_concept_lab.core.config import resolve_config
from sae_concept_lab.core.logic import advanced_output_details, public_output_summary
from sae_concept_lab.fixtures.loader import default_bundle_path, load_bundle

GEMMA_BUNDLE = load_bundle(default_bundle_path("gemma"))


def _resolved():
    return resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="high",
        seed=12345,
    )


RAW_TECHNICAL_TERMS = ("seed", "feature_id", "sae_id", "positions", "hook_point", "coefficient", "layer")


def test_public_summary_never_mentions_raw_technical_field_names():
    cfg = _resolved()
    summary = public_output_summary(cfg, "en").lower()
    for term in RAW_TECHNICAL_TERMS:
        assert term not in summary, f"public summary leaked technical term {term!r}"


def test_public_summary_never_contains_the_raw_seed_or_feature_id_values():
    cfg = _resolved()
    summary = public_output_summary(cfg, "en")
    assert str(cfg.seed) not in summary
    assert cfg.feature_id not in summary
    assert cfg.sae_id not in summary


def test_public_summary_does_state_model_concept_direction_strength():
    cfg = _resolved()
    summary_en = public_output_summary(cfg, "en")
    assert cfg.model_label in summary_en
    assert cfg.concept_label_i18n["en"] in summary_en
    assert "Amplify" in summary_en
    assert "High" in summary_en

    summary_fr = public_output_summary(cfg, "fr")
    assert cfg.concept_label_i18n["fr"] in summary_fr
    assert "Amplifier" in summary_fr
    assert "Élevée" in summary_fr


def test_advanced_details_do_contain_every_raw_technical_field():
    cfg = _resolved()
    details = advanced_output_details(cfg)
    assert details["seed"] == cfg.seed
    assert details["feature_id"] == cfg.feature_id
    assert details["sae_id"] == cfg.sae_id
    assert details["positions"] == cfg.positions
    assert details["hook_point"] == cfg.hook_point
    assert details["strength_coefficient"] == cfg.strength_coefficient
    assert details["layer"] == cfg.layer
    assert details["random_feature_control_id"] == cfg.random_feature_control_id


def test_advanced_details_is_the_same_object_public_summarized_not_a_recomputation():
    """The literal 'one shared state object' acceptance check: build ONE
    ResolvedConfig, pass THAT SAME instance to both renderers, and assert
    advanced's dict is exactly dataclasses.asdict() of it -- there is no
    second construction path in this codebase that could drift from it."""
    import dataclasses

    cfg = _resolved()
    _public_text = public_output_summary(cfg, "en")  # must not mutate cfg
    advanced = advanced_output_details(cfg)
    assert advanced == dataclasses.asdict(cfg)
