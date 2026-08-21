"""Fake-loader end-to-end tests for QwenRuntimeBackend: canonical resolved
control state -> extracted-loader/hook call arguments -> GenerationResult,
without torch/transformers installed (see tests/_fake_runtime.py for what
is faked and why). Also covers same-layer/multi-SAE/cross-layer defensive
enforcement, Compare, and the release gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sae_concept_lab.canonical.concept_bundle import Operation, decode_entry, resolve_control
from sae_concept_lab.canonical.concept_bundle.codec import load_entry_file
from sae_concept_lab.core.logic import assert_compare_invariant, run_compare, send_message
from sae_concept_lab.core.protocol import GenerationRequest
from sae_concept_lab.core.qwen_backend import MECHANICALLY_UNVERIFIED_TAG, QwenRuntimeBackend
from sae_concept_lab.core.runtime_acceptance import ACCEPTANCE_REGISTRY
from tests._fake_runtime import (
    FakeQwenHfModel,
    FakeQwenTextDecoder,
    fake_make_clamp_hook,
    fake_register_qwen_raw_hook,
    install_fake_qwen_transformers,
    install_fake_torch,
    make_fake_wrap_hook_with_diagnostics,
)

# Test-OWNED fixtures (tests/fixtures/), not product fixtures. These tests need
# an entry with known directions, strengths and layers to exercise resolution
# and hook dispatch; that is a property of the tests, not of what ships.
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "qwen"
CURIOSITY = load_entry_file(_FIXTURE_DIR / "curiosity.json")
DIRECTNESS = load_entry_file(_FIXTURE_DIR / "directness.json")


def _install_fakes(monkeypatch, *, num_layers=8):
    install_fake_torch(monkeypatch)
    install_fake_qwen_transformers(monkeypatch)

    text_decoder = FakeQwenTextDecoder(num_layers=num_layers)
    hf_model = FakeQwenHfModel(text_decoder)

    def fake_load_qwen_target(model_path, sae_layer_file_path, *, layer, k=None, device="cuda", dtype="bfloat16",
                               expected_model_revision=None, expected_sae_revision=None):
        # engineering_layer is derived from the `layer` this loader was
        # actually called with, exactly as the real
        # qwen_loader.load_qwen_target does. A fixed literal here would make
        # the fake report a layer it was not asked to load -- which is the
        # precise defect core/scientific_identity.py exists to refuse, so a
        # fake that carried it could not be run through the gate at all.
        hook_identifier = f"resid_post:layer_{layer}"
        provenance = {
            "target": "qwen-3.5-27b",
            "model": {"repository": "Qwen/Qwen3.5-27B", "actual_class": "Qwen3_5ForCausalLM"},
            "sae": {
                "repository": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50", "release": None, "sae_id": None,
                "d_in": 5120, "d_sae": 81920, "k": 50,
            },
            "layer": {
                "engineering_layer": layer, "engineering_only": True, "hook_name": hook_identifier,
            },
        }
        return hf_model, text_decoder, object(), hook_identifier, provenance

    import sae_concept_lab.extracted_runtime.hooks as hooks_module
    import sae_concept_lab.extracted_runtime.qwen_loader as qwen_loader_module

    monkeypatch.setattr(qwen_loader_module, "load_qwen_target", fake_load_qwen_target)
    monkeypatch.setattr(qwen_loader_module, "register_qwen_raw_hook", fake_register_qwen_raw_hook)
    monkeypatch.setattr(hooks_module, "_make_clamp_hook", fake_make_clamp_hook)

    fake_wrap, wrap_calls = make_fake_wrap_hook_with_diagnostics()
    import sae_concept_lab.extracted_runtime.diagnostics as diagnostics_module

    monkeypatch.setattr(diagnostics_module, "wrap_hook_with_diagnostics", fake_wrap)
    return wrap_calls


def _backend(**overrides):
    # CURIOSITY/DIRECTNESS's shipped fixture targets are both layer 7.
    kwargs = dict(model_path="/fake/model", sae_path="/fake/layer7.sae.pt", qwen_layer=7)
    kwargs.update(overrides)
    return QwenRuntimeBackend(**kwargs)


def test_clamp_request_end_to_end(monkeypatch):
    wrap_calls = _install_fakes(monkeypatch)
    resolved = resolve_control(CURIOSITY, direction="amplify", strength="medium")
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hello there", model_key="qwen", decoding={"max_new_tokens": 4},
        seed=0, apply_intervention=True, resolved_config=resolved,
    )

    result = backend.generate(request)

    assert result.is_synthetic is False
    assert result.resolved_config is resolved
    assert result.diagnostics is not None
    assert result.diagnostics["requested"]["operation"] == "clamp"
    assert result.diagnostics["resolved_absolute_target"] == resolved.value
    assert result.diagnostics["backend_received_value"] == resolved.value
    assert result.diagnostics["provenance"]["layer"]["engineering_only"] is True
    assert result.diagnostics["mechanically_accepted"] is True
    assert result.diagnostics["verdict"]["hook_invocation_count"] == 4
    assert result.diagnostics["verdict"]["prefill_call_count"] == 1
    assert result.diagnostics["verdict"]["decode_call_count"] == 3
    assert len(wrap_calls) == 1
    assert wrap_calls[0]["resolved_absolute_target"] == resolved.value
    assert MECHANICALLY_UNVERIFIED_TAG not in result.text


def test_ablate_request_resolves_to_exactly_zero(monkeypatch):
    """resolve_target_value's own rule (final_pairing_harness.py, e63b08e):
    'ABLATE is always exactly 0.0 and is never subject to this check' --
    this backend's translation must reproduce that exact value for an
    ABLATE-operation resolved state, not merely something close to zero."""
    wrap_calls = _install_fakes(monkeypatch)
    document = json.dumps({
        "schema_version": "1.0",
        "concept_id": "FAKE-qwen-ablate-check",
        "pairing_id": "fake-qwen-demo-pairing",
        "positions": "all",
        "provenance": "fake",
        "calibration_provenance": None,
        "directions": {
            "amplify": {
                "targets": [{"sae_id": "fake-sae-demo-qwen-000", "layer": 7, "feature_idx": 4096, "weight": 1.0}],
                "specs": {
                    strength: {"operation": "ablate", "value": None, "unit": None, "unit_source": None}
                    for strength in ("low", "medium", "high")
                },
            },
            "suppress": None,
        },
    })
    entry = decode_entry(document, where="ablate-check")
    resolved = resolve_control(entry, direction="amplify", strength="medium")
    # Canonical's own ResolvedControlState carries no "absolute value" for
    # ABLATE at all (schema.py: an ABLATE Spec permits none of
    # value/unit/unit_source) -- resolved.value is None, not 0.0. The
    # backend's OWN translation (clamp_value = 0.0 if ABLATE else
    # resolved.value) is what produces the concrete 0.0 asserted below.
    assert resolved.operation is Operation.ABLATE
    assert resolved.value is None

    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hi", model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    result = backend.generate(request)
    assert result.diagnostics["resolved_absolute_target"] == 0.0
    assert result.diagnostics["backend_received_value"] == 0.0
    assert wrap_calls[0]["resolved_absolute_target"] == 0.0


def test_baseline_arm_never_registers_a_hook(monkeypatch):
    _install_fakes(monkeypatch)
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hello", model_key="qwen", decoding={}, seed=0,
        apply_intervention=False, resolved_config=None,
    )
    result = backend.generate(request)
    assert result.diagnostics is None
    assert result.resolved_config is None
    assert result.is_synthetic is False


def test_compare_invariant_holds_and_only_modified_arm_carries_diagnostics(monkeypatch):
    _install_fakes(monkeypatch)
    resolved = resolve_control(CURIOSITY, direction="amplify", strength="low")
    backend = _backend()
    compare = run_compare(
        backend=backend, history=[], prompt="tell me something", model_key="qwen",
        decoding={"max_new_tokens": 2}, seed=1, resolved_config=resolved,
    )
    assert_compare_invariant(compare)
    assert compare.modified_result is not None
    assert compare.modified_result.diagnostics is not None
    assert compare.modified_result.resolved_config is resolved


def test_send_message_returns_history_and_result(monkeypatch):
    _install_fakes(monkeypatch)
    resolved = resolve_control(CURIOSITY, direction="suppress", strength="high")
    backend = _backend()
    new_history, result = send_message(
        backend=backend, history=[], prompt="hi", model_key="qwen", decoding={},
        seed=0, resolved_config=resolved,
    )
    assert len(new_history) == 2
    assert new_history[0] == {"role": "user", "content": "hi"}
    assert result.diagnostics["requested"]["direction"] == "suppress"


def test_cross_layer_mismatch_between_resolved_target_and_backend_config_is_refused(monkeypatch):
    _install_fakes(monkeypatch)
    resolved = resolve_control(CURIOSITY, direction="amplify", strength="medium")
    backend = _backend(qwen_layer=99)  # curiosity's target is layer 7
    request = GenerationRequest(
        history=(), prompt="hi", model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    with pytest.raises(ValueError, match="does not match this backend's configured qwen_layer"):
        backend.generate(request)


def test_multi_sae_at_same_layer_is_refused_as_prohibited(monkeypatch):
    """directness.json's amplify direction names two SAEs at layer 7 --
    schema-valid, but PROHIBITED at runtime. Reaching the backend directly
    (bypassing ui/tab.py's own check_direction_executable pre-flight)
    proves execution_guard's defensive check independently catches it."""
    from sae_concept_lab.canonical.concept_bundle import MultipleSaeIdentitiesAtLayerError

    _install_fakes(monkeypatch)
    resolved = resolve_control(DIRECTNESS, direction="amplify", strength="medium")
    backend = _backend(qwen_layer=7)
    request = GenerationRequest(
        history=(), prompt="hi", model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    with pytest.raises(MultipleSaeIdentitiesAtLayerError):
        backend.generate(request)


def test_dev_mode_tags_responses_when_pairing_is_not_accepted(monkeypatch):
    """A defensive check on the honesty tag itself: simulate an
    unaccepted pairing (never actually true for 'qwen' as of this commit --
    see core/runtime_acceptance.py) and confirm the tag appears."""
    _install_fakes(monkeypatch)
    monkeypatch.setitem(ACCEPTANCE_REGISTRY, "qwen", None)
    resolved = resolve_control(CURIOSITY, direction="amplify", strength="medium")
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hi", model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    result = backend.generate(request)
    assert result.text.startswith(MECHANICALLY_UNVERIFIED_TAG)
    assert result.diagnostics["mechanically_accepted"] is False


def test_qwen_pairing_is_mechanically_accepted_as_of_this_commit():
    from sae_concept_lab.core.runtime_acceptance import is_mechanically_accepted

    assert is_mechanically_accepted("qwen") is True


def test_lazy_import_never_touches_torch_at_module_or_construction_time():
    """Constructing QwenRuntimeBackend, and importing this module, must
    never require torch/transformers -- proven simply by the fact that
    this whole test file runs in this venv, which has neither installed."""
    backend = QwenRuntimeBackend(model_path="x", sae_path="y", qwen_layer=0)
    assert backend is not None


def test_generated_only_disclosure_appears_only_for_generated_only_positions(monkeypatch):
    """2026-08-13 researcher ruling: GENERATED_ONLY remains fully
    available and must display the first-token disclosure whenever it is
    the resolved position mode. This repository's own shipped fixtures
    now default to ALL (the public default), so this uses a locally
    constructed entry to exercise the GENERATED_ONLY case directly."""
    from sae_concept_lab.core.qwen_backend import GENERATED_ONLY_FIRST_TOKEN_DISCLOSURE

    _install_fakes(monkeypatch)
    document = json.dumps({
        "schema_version": "1.0",
        "concept_id": "FAKE-qwen-generated-only-disclosure-check",
        "pairing_id": "fake-qwen-demo-pairing",
        "positions": "generated_only",
        "provenance": "fake",
        "calibration_provenance": None,
        "directions": {
            "amplify": {
                "targets": [{"sae_id": "fake-sae-demo-qwen-000", "layer": 7, "feature_idx": 4096, "weight": 1.0}],
                "specs": {
                    strength: {"operation": "clamp", "value": 5.0, "unit": "absolute_activation", "unit_source": None}
                    for strength in ("low", "medium", "high")
                },
            },
            "suppress": None,
        },
    })
    entry = decode_entry(document, where="generated-only-disclosure-check")
    resolved = resolve_control(entry, direction="amplify", strength="medium")
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hi", model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    result = backend.generate(request)
    assert result.diagnostics["generated_only_first_token_disclosure"] == GENERATED_ONLY_FIRST_TOKEN_DISCLOSURE


def test_generated_only_disclosure_is_absent_for_all_positions(monkeypatch):
    _install_fakes(monkeypatch)
    resolved = resolve_control(CURIOSITY, direction="amplify", strength="medium")
    assert resolved.positions.value == "all"
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hi", model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    result = backend.generate(request)
    assert "generated_only_first_token_disclosure" not in result.diagnostics
