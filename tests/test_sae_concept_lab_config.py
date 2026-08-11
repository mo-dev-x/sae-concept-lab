"""resolve_config(): the one function that builds the shared ResolvedConfig
both Public and Advanced read. Also covers the "positions sourced from
configuration, never hardcoded" requirement directly."""

from __future__ import annotations

import pytest

from sae_concept_lab.core.config import resolve_config
from sae_concept_lab.fixtures.loader import default_bundle_path, load_bundle

GEMMA_BUNDLE = load_bundle(default_bundle_path("gemma"))
QWEN_BUNDLE = load_bundle(default_bundle_path("qwen"))


def test_resolve_config_uses_bundles_own_positions_default_when_not_overridden():
    cfg = resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="low",
    )
    assert cfg.positions == GEMMA_BUNDLE["positions_default"]


def test_resolve_config_positions_override_is_representable():
    cfg = resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="low",
        positions="all",
    )
    assert cfg.positions == "all"


def test_resolve_config_uses_bundles_own_seed_default_when_not_overridden():
    cfg = resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="low",
    )
    assert cfg.seed == GEMMA_BUNDLE["seed_default"]


def test_resolve_config_seed_override():
    cfg = resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="low",
        seed=999,
    )
    assert cfg.seed == 999


def test_resolve_config_unknown_concept_raises():
    with pytest.raises(KeyError):
        resolve_config(
            bundle=GEMMA_BUNDLE, concept_id="not-a-real-concept", direction="amplify", strength_level="low"
        )


def test_resolve_config_is_marked_synthetic_for_every_shipped_bundle():
    for bundle in (GEMMA_BUNDLE, QWEN_BUNDLE):
        cfg = resolve_config(
            bundle=bundle, concept_id=bundle["concepts"][0]["concept_id"], direction="amplify", strength_level="low"
        )
        assert cfg.is_synthetic is True


def test_strength_levels_resolve_to_distinct_coefficients():
    concept_id = GEMMA_BUNDLE["concepts"][0]["concept_id"]
    coefficients = {
        level: resolve_config(
            bundle=GEMMA_BUNDLE, concept_id=concept_id, direction="amplify", strength_level=level
        ).strength_coefficient
        for level in ("low", "medium", "high")
    }
    assert len(set(coefficients.values())) == 3


def test_feature_ids_are_all_prefixed_fake_never_a_real_looking_index():
    for bundle in (GEMMA_BUNDLE, QWEN_BUNDLE):
        for concept in bundle["concepts"]:
            assert concept["feature_id"].startswith("FAKE-")


def test_diagnostics_synthetic_flag_matches_resolved_configs_is_synthetic_not_a_separate_literal():
    """P0 release-safety correction: diagnostics['synthetic'] used to be a
    hardcoded True regardless of the bundle's own is_synthetic flag -- a
    future real bundle/adapter setting is_synthetic=False would silently
    have kept reporting synthetic=True forever. Both fields must now be
    the exact same derived value, checked here against BOTH a synthetic
    and a (hypothetically) non-synthetic bundle so the assertion cannot
    pass vacuously just because every shipped bundle happens to be True."""
    cfg_synthetic = resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="low",
    )
    assert cfg_synthetic.is_synthetic is True
    assert cfg_synthetic.diagnostics["synthetic"] is True

    hypothetically_real_bundle = dict(GEMMA_BUNDLE, is_synthetic=False)
    cfg_real = resolve_config(
        bundle=hypothetically_real_bundle,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="low",
    )
    assert cfg_real.is_synthetic is False
    assert cfg_real.diagnostics["synthetic"] is False
