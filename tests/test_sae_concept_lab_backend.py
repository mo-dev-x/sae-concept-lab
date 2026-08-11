"""StubConceptLabBackend: determinism, FAKE tagging, and the wiring
contract between apply_intervention and resolved_config."""

from __future__ import annotations

import pytest

from sae_concept_lab.core.config import resolve_config
from sae_concept_lab.core.protocol import GenerationRequest
from sae_concept_lab.core.stub_backend import FAKE_TAG, StubConceptLabBackend
from sae_concept_lab.fixtures.loader import default_bundle_path, load_bundle

GEMMA_BUNDLE = load_bundle(default_bundle_path("gemma"))


def _resolved():
    return resolve_config(
        bundle=GEMMA_BUNDLE,
        concept_id=GEMMA_BUNDLE["concepts"][0]["concept_id"],
        direction="amplify",
        strength_level="medium",
    )


def _request(**overrides):
    defaults = dict(
        history=(),
        prompt="hello",
        model_key="gemma",
        decoding={"temperature": 0.0},
        seed=0,
        apply_intervention=True,
        resolved_config=_resolved(),
    )
    defaults.update(overrides)
    return GenerationRequest(**defaults)


def test_response_is_visibly_tagged_as_fake():
    result = StubConceptLabBackend().generate(_request())
    assert result.text.startswith(FAKE_TAG)
    assert result.is_synthetic is True


def test_deterministic_same_request_same_output():
    backend = StubConceptLabBackend()
    r1 = backend.generate(_request())
    r2 = backend.generate(_request())
    assert r1.text == r2.text


def test_different_seed_changes_output():
    backend = StubConceptLabBackend()
    r1 = backend.generate(_request(seed=0))
    r2 = backend.generate(_request(seed=1))
    assert r1.text != r2.text


def test_different_history_changes_output():
    backend = StubConceptLabBackend()
    r1 = backend.generate(_request(history=()))
    r2 = backend.generate(_request(history=(("user", "prior turn"), ("assistant", "reply"))))
    assert r1.text != r2.text


def test_baseline_arm_requires_resolved_config_none():
    backend = StubConceptLabBackend()
    with pytest.raises(ValueError):
        backend.generate(_request(apply_intervention=False, resolved_config=_resolved()))


def test_modified_arm_requires_resolved_config_present():
    backend = StubConceptLabBackend()
    with pytest.raises(ValueError):
        backend.generate(_request(apply_intervention=True, resolved_config=None))


def test_output_never_contains_plausible_prose_only_the_template():
    """Guards against someone "improving" the stub into something that
    reads like a real reply -- the whole point is that it must not."""
    result = StubConceptLabBackend().generate(_request())
    assert "digest=" in result.text
    assert result.text.count("|") >= 2  # the pipe-delimited template, not free prose
