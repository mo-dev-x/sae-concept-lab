"""Fake-loader end-to-end tests for GemmaRuntimeBackend: canonical resolved
control state -> extracted-loader/hook call arguments -> GenerationResult,
without torch/transformer_lens/sae_lens installed (see
tests/_fake_runtime.py for what is faked and why). Also covers same-layer/
multi-SAE/cross-layer defensive enforcement and Compare."""

from __future__ import annotations

import pytest

from sae_concept_lab.canonical.concept_bundle import (
    MultipleExecutionGroupsError,
    Operation,
    resolve_control,
)
from sae_concept_lab.core.gemma_backend import MECHANICALLY_UNVERIFIED_TAG, GemmaRuntimeBackend
from sae_concept_lab.core.logic import assert_compare_invariant, run_compare, send_message
from sae_concept_lab.core.protocol import GenerationRequest
from sae_concept_lab.core.runtime_acceptance import ACCEPTANCE_REGISTRY
from sae_concept_lab.fixtures.loader import load_entries
from tests._fake_runtime import (
    FakeGemmaModel,
    install_fake_torch,
    make_fake_wrap_hook_with_diagnostics,
)

GEMMA_ENTRIES = load_entries("gemma")
WARMTH = next(e for e in GEMMA_ENTRIES if e.concept_id == "FAKE-gemma-warmth")
ENTHUSIASM = next(e for e in GEMMA_ENTRIES if e.concept_id == "FAKE-gemma-enthusiasm")


def _install_fakes(monkeypatch):
    install_fake_torch(monkeypatch)

    model = FakeGemmaModel()
    provenance = {
        "target": "gemma-3-12b-it",
        "model": {"repository": "google/gemma-3-12b-it", "actual_class": "HookedTransformer"},
        "sae": {"repository": "google/gemma-scope-2-12b-it-res", "d_in": 3840, "d_sae": 16384},
        "layer": {"engineering_layer": 11, "hook_name": "blocks.11.hook_resid_post"},
    }

    def fake_load_gemma_it_target(model_path, sae_path, *, device="cuda", dtype="bfloat16",
                                   expected_model_revision=None, expected_sae_revision=None):
        return model, object(), "blocks.11.hook_resid_post", provenance

    import sae_concept_lab.extracted_runtime.gemma_loader as gemma_loader_module
    import sae_concept_lab.extracted_runtime.hooks as hooks_module

    monkeypatch.setattr(gemma_loader_module, "load_gemma_it_target", fake_load_gemma_it_target)

    def fake_make_clamp_hook(sae_fp32, feature_index, clamp_value, positions, prompt_lengths, stats):
        def hook_fn(resid, hook):
            return resid

        return hook_fn

    monkeypatch.setattr(hooks_module, "_make_clamp_hook", fake_make_clamp_hook)

    fake_wrap, wrap_calls = make_fake_wrap_hook_with_diagnostics()
    import sae_concept_lab.extracted_runtime.diagnostics as diagnostics_module

    monkeypatch.setattr(diagnostics_module, "wrap_hook_with_diagnostics", fake_wrap)
    return wrap_calls


def _backend(**overrides):
    kwargs = dict(model_path="/fake/model", sae_path="/fake/sae_snapshot_root")
    kwargs.update(overrides)
    return GemmaRuntimeBackend(**kwargs)


def test_clamp_request_end_to_end(monkeypatch):
    wrap_calls = _install_fakes(monkeypatch)
    resolved = resolve_control(WARMTH, direction="amplify", strength="high")
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="tell me about your day", model_key="gemma", decoding={"max_new_tokens": 4},
        seed=0, apply_intervention=True, resolved_config=resolved,
    )

    result = backend.generate(request)

    assert result.is_synthetic is False
    assert result.resolved_config is resolved
    assert result.diagnostics["requested"]["operation"] == "clamp"
    assert result.diagnostics["requested"]["layer"] == 11
    assert result.diagnostics["resolved_absolute_target"] == resolved.value
    assert result.diagnostics["backend_received_value"] == resolved.value
    assert result.diagnostics["provenance"]["layer"]["engineering_layer"] == 11
    assert result.diagnostics["mechanically_accepted"] is True
    assert result.diagnostics["verdict"]["hook_invocation_count"] == 4
    assert len(wrap_calls) == 1
    assert MECHANICALLY_UNVERIFIED_TAG not in result.text


def test_ablate_operation_resolves_backend_received_value_to_zero(monkeypatch):
    """Same contract as Qwen's: canonical leaves ResolvedControlState.value
    as None for ABLATE (schema.py permits no value/unit/unit_source on an
    ABLATE Spec) -- the backend's own translation is what turns this into
    the concrete 0.0 _make_clamp_hook/wrap_hook_with_diagnostics receive."""
    import json

    from sae_concept_lab.canonical.concept_bundle import decode_entry

    wrap_calls = _install_fakes(monkeypatch)
    document = json.dumps({
        "schema_version": "1.0",
        "concept_id": "FAKE-gemma-ablate-check",
        "pairing_id": "fake-gemma-demo-pairing",
        "positions": "all",
        "provenance": "fake",
        "calibration_provenance": None,
        "directions": {
            "amplify": {
                "targets": [{"sae_id": "fake-sae-demo-gemma-000", "layer": 11, "feature_idx": 1001, "weight": 1.0}],
                "specs": {
                    strength: {"operation": "ablate", "value": None, "unit": None, "unit_source": None}
                    for strength in ("low", "medium", "high")
                },
            },
            "suppress": None,
        },
    })
    entry = decode_entry(document, where="gemma-ablate-check")
    resolved = resolve_control(entry, direction="amplify", strength="low")
    assert resolved.operation is Operation.ABLATE
    assert resolved.value is None

    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hi", model_key="gemma", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    result = backend.generate(request)
    assert result.diagnostics["resolved_absolute_target"] == 0.0
    assert result.diagnostics["backend_received_value"] == 0.0
    assert wrap_calls[0]["resolved_absolute_target"] == 0.0


def test_baseline_arm_never_attaches_a_hook(monkeypatch):
    _install_fakes(monkeypatch)
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hello", model_key="gemma", decoding={}, seed=0,
        apply_intervention=False, resolved_config=None,
    )
    result = backend.generate(request)
    assert result.diagnostics is None
    assert result.resolved_config is None


def test_compare_invariant_holds(monkeypatch):
    _install_fakes(monkeypatch)
    resolved = resolve_control(WARMTH, direction="suppress", strength="medium")
    backend = _backend()
    compare = run_compare(
        backend=backend, history=[], prompt="tell me something", model_key="gemma",
        decoding={"max_new_tokens": 2}, seed=1, resolved_config=resolved,
    )
    assert_compare_invariant(compare)
    assert compare.modified_result.diagnostics is not None


def test_send_message_returns_history_and_result(monkeypatch):
    _install_fakes(monkeypatch)
    resolved = resolve_control(WARMTH, direction="amplify", strength="low")
    backend = _backend()
    new_history, result = send_message(
        backend=backend, history=[], prompt="hi", model_key="gemma", decoding={},
        seed=0, resolved_config=resolved,
    )
    assert len(new_history) == 2
    assert result.diagnostics["requested"]["direction"] == "amplify"


def test_cross_layer_execution_is_refused_as_capability_limit(monkeypatch):
    """enthusiasm.json's amplify direction targets layers 11 AND 12 --
    schema-valid, but runtime v1 executes one (sae_id, layer) group per
    pass. Reaching the backend directly (bypassing ui/tab.py's own
    check_direction_executable pre-flight) proves execution_guard's
    defensive check independently catches it."""
    _install_fakes(monkeypatch)
    resolved = resolve_control(ENTHUSIASM, direction="amplify", strength="medium")
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hi", model_key="gemma", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    with pytest.raises(MultipleExecutionGroupsError):
        backend.generate(request)


def test_dev_mode_tags_responses_when_pairing_is_not_accepted(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setitem(ACCEPTANCE_REGISTRY, "gemma", None)
    resolved = resolve_control(WARMTH, direction="amplify", strength="medium")
    backend = _backend()
    request = GenerationRequest(
        history=(), prompt="hi", model_key="gemma", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    result = backend.generate(request)
    assert result.text.startswith(MECHANICALLY_UNVERIFIED_TAG)
    assert result.diagnostics["mechanically_accepted"] is False


def test_gemma_pairing_is_mechanically_accepted_as_of_this_commit():
    from sae_concept_lab.core.runtime_acceptance import is_mechanically_accepted

    assert is_mechanically_accepted("gemma") is True


def test_lazy_construction_never_touches_torch():
    backend = GemmaRuntimeBackend(model_path="x", sae_path="y")
    assert backend is not None
