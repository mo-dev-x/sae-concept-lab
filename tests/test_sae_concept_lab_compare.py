"""Compare invariant: Original and Modified must share serialized
history, prompt, model_key, decoding, and seed exactly -- differing ONLY
in apply_intervention and resolved_config."""

from __future__ import annotations

import dataclasses

import pytest

from sae_concept_lab.core.config import resolve_config
from sae_concept_lab.core.logic import assert_compare_invariant, run_compare
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import default_bundle_path, load_bundle

GEMMA_BUNDLE = load_bundle(default_bundle_path("gemma"))


def _resolved(strength="medium"):
    return resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level=strength,
    )


def test_compare_requests_are_identical_except_intervention_fields():
    compare = run_compare(
        backend=StubConceptLabBackend(),
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "[FAKE] hello"}],
        prompt="tell me about your day",
        model_key="gemma",
        decoding={"temperature": 0.0, "max_new_tokens": 64},
        seed=42,
        resolved_config=_resolved(),
    )
    assert_compare_invariant(compare)  # must not raise

    o, m = compare.original_request, compare.modified_request
    assert o.history == m.history
    assert o.prompt == m.prompt
    assert o.model_key == m.model_key
    assert o.decoding == m.decoding
    assert o.seed == m.seed
    assert o.apply_intervention is False
    assert m.apply_intervention is True
    assert o.resolved_config is None
    assert m.resolved_config is not None


def test_compare_invariant_detects_a_real_divergence():
    """Adversarial check: assert_compare_invariant must actually be able
    to fail, not vacuously pass on anything. A hand-built pair that
    diverges in seed (a bug this exact function exists to catch) must
    raise."""
    compare = run_compare(
        backend=StubConceptLabBackend(),
        history=[],
        prompt="hi",
        model_key="gemma",
        decoding={"temperature": 0.0},
        seed=1,
        resolved_config=_resolved(),
    )
    broken_modified = dataclasses.replace(compare.modified_request, seed=999)
    broken = dataclasses.replace(compare, modified_request=broken_modified)
    with pytest.raises(AssertionError):
        assert_compare_invariant(broken)


def test_compare_original_and_modified_texts_differ():
    """The whole point of Compare: the two arms must actually read
    differently (baseline vs concept-applied), not just structurally --
    the stub backend must encode apply_intervention into its output."""
    compare = run_compare(
        backend=StubConceptLabBackend(),
        history=[],
        prompt="hi",
        model_key="gemma",
        decoding={"temperature": 0.0},
        seed=1,
        resolved_config=_resolved(),
    )
    assert compare.original_text != compare.modified_text


def test_compare_uses_the_same_seed_across_different_strength_levels():
    """Same seed/context regardless of which strength was selected --
    only the resolved_config's coefficient should vary."""
    for strength in ("low", "medium", "high"):
        compare = run_compare(
            backend=StubConceptLabBackend(),
            history=[],
            prompt="hi",
            model_key="gemma",
            decoding={"temperature": 0.0},
            seed=7,
            resolved_config=_resolved(strength),
        )
        assert compare.original_request.seed == 7
        assert compare.modified_request.seed == 7
