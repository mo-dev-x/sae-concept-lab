"""CPU/fake-loader tests for sae_concept_lab.smoke.tamia_smoke: the
orchestration, failure aggregation, and every defensive assertion that
never needs real torch/transformers/transformer_lens/sae_lens weights --
every real-weight scenario in this file goes through tests._fake_runtime's
monkeypatched loaders instead of touching a real model or SAE.

What this file deliberately does NOT prove: real prefill-masking numerics
for GENERATED_ONLY (tests._fake_runtime's shared hook fake is explicitly
NOT masking-aware -- see its own docstring -- so those scenarios are
EXPECTED to report a failure against it here; the real contract is
covered by tests/test_runtime_hooks_differential.py and this packet's own
tests/test_tamia_smoke_torch.py, torch-gated for Tamia)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys

import pytest

from sae_concept_lab.canonical.concept_bundle import PositionMode
from sae_concept_lab.core.protocol import GenerationResult
from sae_concept_lab.core.runtime_acceptance import ACCEPTANCE_REGISTRY, RuntimeAcceptanceError
from sae_concept_lab.fixtures.loader import load_entries
from sae_concept_lab.smoke import entries, tamia_smoke
from tests._fake_runtime import (
    FakeGemmaModel,
    FakeQwenHfModel,
    FakeQwenTextDecoder,
    fake_make_clamp_hook,
    fake_register_qwen_raw_hook,
    install_fake_qwen_transformers,
    install_fake_torch,
    make_fake_wrap_hook_with_diagnostics,
)


def _install_qwen_fakes(monkeypatch):
    install_fake_torch(monkeypatch)
    install_fake_qwen_transformers(monkeypatch)
    text_decoder = FakeQwenTextDecoder(num_layers=1)
    hf_model = FakeQwenHfModel(text_decoder)
    provenance = {
        "target": "qwen-3.5-27b",
        "model": {"repository": "Qwen/Qwen3.5-27B", "actual_class": "Qwen3_5ForCausalLM"},
        "sae": {"repository": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50", "d_in": 5120, "d_sae": 81920, "k": 50},
        "layer": {
            "engineering_layer": entries.QWEN_SMOKE_LAYER, "engineering_only": True,
            "hook_name": f"resid_post:layer_{entries.QWEN_SMOKE_LAYER}",
        },
    }

    def fake_load_qwen_target(
        model_path, sae_layer_file_path, *, layer, k=None, device="cuda", dtype="bfloat16",
        expected_model_revision=None, expected_sae_revision=None,
    ):
        return hf_model, text_decoder, object(), f"resid_post:layer_{layer}", provenance

    import sae_concept_lab.extracted_runtime.diagnostics as diagnostics_module
    import sae_concept_lab.extracted_runtime.hooks as hooks_module
    import sae_concept_lab.extracted_runtime.qwen_loader as qwen_loader_module

    monkeypatch.setattr(qwen_loader_module, "load_qwen_target", fake_load_qwen_target)
    monkeypatch.setattr(qwen_loader_module, "register_qwen_raw_hook", fake_register_qwen_raw_hook)
    monkeypatch.setattr(hooks_module, "_make_clamp_hook", fake_make_clamp_hook)
    fake_wrap, wrap_calls = make_fake_wrap_hook_with_diagnostics()
    monkeypatch.setattr(diagnostics_module, "wrap_hook_with_diagnostics", fake_wrap)
    return wrap_calls


def _install_gemma_fakes(monkeypatch):
    install_fake_torch(monkeypatch)
    model = FakeGemmaModel()
    provenance = {
        "target": "gemma-3-12b-it",
        "model": {"repository": "google/gemma-3-12b-it", "actual_class": "HookedTransformer"},
        "sae": {"repository": "google/gemma-scope-2-12b-it-res", "d_in": 3840, "d_sae": 16384},
        "layer": {
            "engineering_layer": entries.GEMMA_SMOKE_LAYER,
            "hook_name": f"blocks.{entries.GEMMA_SMOKE_LAYER}.hook_resid_post",
        },
    }

    def fake_load_gemma_it_target(
        model_path, sae_path, *, device="cuda", dtype="bfloat16", expected_model_revision=None,
        expected_sae_revision=None,
    ):
        return model, object(), f"blocks.{entries.GEMMA_SMOKE_LAYER}.hook_resid_post", provenance

    import sae_concept_lab.extracted_runtime.diagnostics as diagnostics_module
    import sae_concept_lab.extracted_runtime.gemma_loader as gemma_loader_module
    import sae_concept_lab.extracted_runtime.hooks as hooks_module

    monkeypatch.setattr(gemma_loader_module, "load_gemma_it_target", fake_load_gemma_it_target)
    monkeypatch.setattr(hooks_module, "_make_clamp_hook", fake_make_clamp_hook)
    fake_wrap, wrap_calls = make_fake_wrap_hook_with_diagnostics()
    monkeypatch.setattr(diagnostics_module, "wrap_hook_with_diagnostics", fake_wrap)
    return wrap_calls


def _base_cli_args(**overrides):
    defaults = dict(
        qwen_model_path="/fake/qwen-model", qwen_sae_path="/fake/layer0.sae.pt", qwen_device="cpu",
        qwen_dtype="bfloat16", qwen_expected_model_revision=None, qwen_expected_sae_revision=None,
        gemma_model_path="/fake/gemma-model", gemma_sae_path="/fake/gemma-sae-root", gemma_device="cpu",
        gemma_dtype="bfloat16", gemma_expected_model_revision=None, gemma_expected_sae_revision=None,
        max_new_tokens=4, server_name="127.0.0.1", server_port=7861, evidence_registry_root=None,
        output="unused.json",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Smoke entries never enter fixture discovery or pass the release gate
# ---------------------------------------------------------------------------


def test_smoke_concept_ids_absent_from_shipped_fixture_files():
    for model_key in ("qwen", "gemma"):
        ids = {e.concept_id for e in load_entries(model_key)}
        assert ids.isdisjoint(entries.ALL_SMOKE_CONCEPT_IDS)


def test_scenario_smoke_entries_hidden_passes():
    result = tamia_smoke.scenario_smoke_entries_hidden()
    assert result.passed, result.as_dict()


# ---------------------------------------------------------------------------
# CPU-safe defensive assertions -- never reach a torch import
# ---------------------------------------------------------------------------


def test_scenario_identity_cannot_be_swapped_passes():
    result = tamia_smoke.scenario_identity_cannot_be_swapped()
    assert result.passed, result.as_dict()


def test_scenario_qwen_multi_sae_prohibited_passes_without_torch():
    result = tamia_smoke.scenario_qwen_multi_sae_prohibited("/placeholder/model", "/placeholder/sae")
    assert result.passed, result.as_dict()
    assert result.detail["classification"] == "PROHIBITED"


def test_scenario_gemma_cross_layer_capability_limit_passes_without_torch():
    result = tamia_smoke.scenario_gemma_cross_layer_capability_limit("/placeholder/model", "/placeholder/sae")
    assert result.passed, result.as_dict()
    assert result.detail["classification"] == "CAPABILITY_LIMIT"


def test_scenario_qwen_backend_layer_mismatch_refused_passes_without_torch():
    result = tamia_smoke.scenario_qwen_backend_layer_mismatch_refused("/placeholder/model", "/placeholder/sae")
    assert result.passed, result.as_dict()


# ---------------------------------------------------------------------------
# _run_guarded / SmokePacket aggregation
# ---------------------------------------------------------------------------


def test_run_guarded_converts_exception_to_failed_result_without_raising():
    def boom():
        raise RuntimeError("boom")

    result = tamia_smoke._run_guarded("x", "qwen", boom)
    assert result.passed is False
    assert "boom" in result.error


def test_smoke_packet_passed_is_false_if_any_scenario_failed_even_when_later_ones_pass():
    ok = tamia_smoke._ok("a", "qwen", "fine")
    bad = tamia_smoke._fail("b", "qwen", "broken")
    later_ok = tamia_smoke._ok("c", "qwen", "also fine")
    packet = tamia_smoke.SmokePacket(
        product_commit="deadbeef", runtime_extraction_source_commits={"qwen": "x", "gemma": "y"},
        acceptance_evidence_commits={"qwen": "z", "gemma": "z"}, scenarios=(ok, bad, later_ok),
    )
    assert packet.passed is False
    assert [s.scenario_id for s in packet.scenarios] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Missing acceptance evidence prevents model loading
# ---------------------------------------------------------------------------


class _RefusingBackend:
    """A spy whose .generate() raises AssertionError if ever called --
    proves the acceptance gate refuses BEFORE any backend interaction, not
    merely before a successful one."""

    pairing = "qwen"

    def generate(self, request):
        raise AssertionError("backend.generate() must never be reached when acceptance evidence is missing")


def test_require_mechanical_acceptance_passes_through_when_accepted():
    tamia_smoke._require_mechanical_acceptance("qwen")
    tamia_smoke._require_mechanical_acceptance("gemma")


def test_missing_acceptance_evidence_refuses_before_touching_a_backend(monkeypatch):
    monkeypatch.setitem(ACCEPTANCE_REGISTRY, "qwen", None)
    with pytest.raises(RuntimeAcceptanceError):
        tamia_smoke.run_qwen_position_scenario(
            _RefusingBackend(), PositionMode.ALL, max_new_tokens=4, product_commit="x", extraction_source_commit="y",
        )


def test_missing_acceptance_evidence_becomes_a_failed_scenario_via_run_guarded(monkeypatch):
    monkeypatch.setitem(ACCEPTANCE_REGISTRY, "qwen", None)
    result = tamia_smoke._run_guarded(
        "qwen_all", "qwen",
        lambda: tamia_smoke.run_qwen_position_scenario(
            _RefusingBackend(), PositionMode.ALL, max_new_tokens=4, product_commit="x", extraction_source_commit="y",
        ),
    )
    assert result.passed is False
    assert "RuntimeAcceptanceError" in result.error


# ---------------------------------------------------------------------------
# REGRESSION (e3c83fb mutation-tested): entries.QWEN_SMOKE_LAYER (0) and
# entries.GEMMA_SMOKE_LAYER (31) happen to equal the layers
# runtime_acceptance.py's records are scoped to, so reverting either
# run_*_position_scenario's `_require_mechanical_acceptance(pairing,
# entries.*_SMOKE_LAYER)` call back to the layer-blind
# `_require_mechanical_acceptance(pairing)` passes the full suite
# unchanged -- the scoped and unscoped questions agree by coincidence. Each
# test below moves the record's OWN accepted_layer away from the smoke
# constant (never touching the smoke constant itself, which genuinely IS
# the layer that scenario exercises) to break that coincidence: the scoped
# call must now refuse, naming the smoke layer; a layer-blind call would
# still find a record and return True regardless of accepted_layer, and
# would NOT refuse.
# ---------------------------------------------------------------------------


def test_qwen_smoke_scenario_refuses_when_its_own_layer_is_not_covered(monkeypatch):
    accepted = ACCEPTANCE_REGISTRY["qwen"]
    monkeypatch.setitem(ACCEPTANCE_REGISTRY, "qwen", dataclasses.replace(accepted, accepted_layer=99))
    with pytest.raises(RuntimeAcceptanceError, match=f"scenario at layer {entries.QWEN_SMOKE_LAYER}:"):
        tamia_smoke.run_qwen_position_scenario(
            _RefusingBackend(), PositionMode.ALL, max_new_tokens=4, product_commit="x", extraction_source_commit="y",
        )


def test_gemma_smoke_scenario_refuses_when_its_own_layer_is_not_covered(monkeypatch):
    accepted = ACCEPTANCE_REGISTRY["gemma"]
    monkeypatch.setitem(ACCEPTANCE_REGISTRY, "gemma", dataclasses.replace(accepted, accepted_layer=99))
    with pytest.raises(RuntimeAcceptanceError, match=f"scenario at layer {entries.GEMMA_SMOKE_LAYER}:"):
        tamia_smoke.run_gemma_position_scenario(
            _RefusingBackend(), PositionMode.ALL, max_new_tokens=4, product_commit="x", extraction_source_commit="y",
        )


# ---------------------------------------------------------------------------
# _finish_position_scenario: the ALL/GENERATED_ONLY prefill assertion logic,
# tested directly against hand-built diagnostics -- deterministic, no
# backend or fakes needed.
# ---------------------------------------------------------------------------


def _fake_generation_result(*, prefill_delta: float) -> GenerationResult:
    diagnostics = {
        "provenance": {"model": {}, "sae": {}, "layer": {}},
        "requested": {}, "resolved_absolute_target": 20.0, "backend_received_value": 20.0,
        "trace": [
            {"call_index": 0, "residual_delta_norm": prefill_delta},
            {"call_index": 1, "residual_delta_norm": 20.0},
        ],
        "verdict": {"hook_invocation_count": 2, "prefill_call_count": 1, "decode_call_count": 1},
    }
    return GenerationResult(text="generated-tokens", is_synthetic=False, resolved_config=None, diagnostics=diagnostics)


def test_finish_position_scenario_all_passes_when_prefill_delta_nonzero():
    outcome = tamia_smoke._finish_position_scenario(
        scenario_id="qwen_all", pairing="qwen", positions=PositionMode.ALL,
        result=_fake_generation_result(prefill_delta=20.0), product_commit="deadbeef",
        extraction_source_commit="e63b08e",
    )
    assert outcome.passed, outcome.as_dict()


def test_finish_position_scenario_all_fails_when_prefill_delta_zero():
    outcome = tamia_smoke._finish_position_scenario(
        scenario_id="qwen_all", pairing="qwen", positions=PositionMode.ALL,
        result=_fake_generation_result(prefill_delta=0.0), product_commit="deadbeef",
        extraction_source_commit="e63b08e",
    )
    assert outcome.passed is False
    assert "ALL must modify prefill" in outcome.summary


def test_finish_position_scenario_generated_only_passes_when_prefill_delta_zero():
    outcome = tamia_smoke._finish_position_scenario(
        scenario_id="qwen_generated_only", pairing="qwen", positions=PositionMode.GENERATED_ONLY,
        result=_fake_generation_result(prefill_delta=0.0), product_commit="deadbeef",
        extraction_source_commit="e63b08e",
    )
    assert outcome.passed, outcome.as_dict()


def test_finish_position_scenario_generated_only_fails_when_prefill_delta_nonzero():
    outcome = tamia_smoke._finish_position_scenario(
        scenario_id="qwen_generated_only", pairing="qwen", positions=PositionMode.GENERATED_ONLY,
        result=_fake_generation_result(prefill_delta=20.0), product_commit="deadbeef",
        extraction_source_commit="e63b08e",
    )
    assert outcome.passed is False
    assert "masking failed" in outcome.summary


def test_finish_position_scenario_fails_when_backend_returns_no_diagnostics():
    result = GenerationResult(text="x", is_synthetic=False, resolved_config=None, diagnostics=None)
    outcome = tamia_smoke._finish_position_scenario(
        scenario_id="qwen_all", pairing="qwen", positions=PositionMode.ALL, result=result,
        product_commit="deadbeef", extraction_source_commit="e63b08e",
    )
    assert outcome.passed is False


# ---------------------------------------------------------------------------
# End-to-end real-scenario functions against fakes (PositionMode.ALL only --
# see module docstring for why GENERATED_ONLY's masking numerics are not
# exercised against the shared, deliberately non-masking-aware fake here)
# ---------------------------------------------------------------------------


def test_run_qwen_all_scenario_end_to_end_with_fakes_passes(monkeypatch):
    _install_qwen_fakes(monkeypatch)
    backend = tamia_smoke.QwenRuntimeBackend(
        model_path="/fake/model", sae_path="/fake/layer0.sae.pt", qwen_layer=entries.QWEN_SMOKE_LAYER
    )
    result = tamia_smoke.run_qwen_position_scenario(
        backend, PositionMode.ALL, max_new_tokens=4, product_commit="deadbeef", extraction_source_commit="e63b08e",
    )
    assert result.passed, result.as_dict()
    assert result.detail["requested"]["requested_layer"] == entries.QWEN_SMOKE_LAYER
    assert result.detail["requested"]["feature_idx"] == entries.QWEN_SMOKE_FEATURE_IDX
    assert result.detail["resolved_absolute_target"] == entries.QWEN_SMOKE_RAW_TARGET
    # The sealed record states the scope of its own claim: an engineering-only
    # smoke identity is never a science-attributed one.
    assert result.detail["claim_scope"] == "ENGINEERING_DEMONSTRATION_ONLY"
    assert result.detail["requested_vs_loaded_identity"]["loaded"]["layer"] == entries.QWEN_SMOKE_LAYER
    assert result.detail["acceptance_evidence_commit"] == ACCEPTANCE_REGISTRY["qwen"].evidence_commit


def test_run_gemma_all_scenario_end_to_end_with_fakes_passes(monkeypatch):
    _install_gemma_fakes(monkeypatch)
    backend = tamia_smoke.GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae_root")
    result = tamia_smoke.run_gemma_position_scenario(
        backend, PositionMode.ALL, max_new_tokens=4, product_commit="deadbeef", extraction_source_commit="de3b499",
    )
    assert result.passed, result.as_dict()
    assert result.detail["requested"]["requested_layer"] == entries.GEMMA_SMOKE_LAYER
    assert result.detail["requested"]["feature_idx"] == entries.GEMMA_SMOKE_FEATURE_IDX
    assert result.detail["resolved_absolute_target"] == entries.GEMMA_SMOKE_RAW_CLAMP
    assert result.detail["claim_scope"] == "ENGINEERING_DEMONSTRATION_ONLY"
    assert result.detail["requested_vs_loaded_identity"]["loaded"]["layer"] == entries.GEMMA_SMOKE_LAYER


# ---------------------------------------------------------------------------
# Application smoke, in isolation: real Gradio boot/HTTP/shutdown against
# fake-loaded real backends (so diagnostics are populated, unlike
# StubConceptLabBackend, which never sets them).
# ---------------------------------------------------------------------------


def test_run_application_smoke_boots_probes_and_shuts_down_cleanly(monkeypatch):
    _install_qwen_fakes(monkeypatch)
    _install_gemma_fakes(monkeypatch)
    qwen_backend = tamia_smoke.QwenRuntimeBackend(
        model_path="/fake/model", sae_path="/fake/layer0.sae.pt", qwen_layer=entries.QWEN_SMOKE_LAYER
    )
    gemma_backend = tamia_smoke.GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae_root")
    results = tamia_smoke.run_application_smoke(
        gemma_backend=gemma_backend, qwen_backend=qwen_backend, server_name="127.0.0.1", server_port=7862,
        evidence_registry_root=None, max_new_tokens=4,
    )
    by_id = {r.scenario_id: r for r in results}
    assert by_id["app_smoke_entries_unreachable_via_ui"].passed, by_id["app_smoke_entries_unreachable_via_ui"].as_dict()
    assert by_id["app_smoke_http_200"].passed, by_id["app_smoke_http_200"].as_dict()
    assert by_id["app_smoke_bounded_request"].passed, by_id["app_smoke_bounded_request"].as_dict()
    assert by_id["app_smoke_release_still_refuses"].passed, by_id["app_smoke_release_still_refuses"].as_dict()


# ---------------------------------------------------------------------------
# build_smoke_packet: full orchestration and sequencing
# ---------------------------------------------------------------------------


def test_build_smoke_packet_end_to_end_aggregates_correctly(monkeypatch, tmp_path):
    _install_qwen_fakes(monkeypatch)
    _install_gemma_fakes(monkeypatch)
    monkeypatch.setattr(
        tamia_smoke, "run_application_smoke",
        lambda **kwargs: [tamia_smoke._ok("app_smoke_stubbed", "both", "stubbed for this orchestration test")],
    )

    args = _base_cli_args(output=str(tmp_path / "packet.json"))
    packet = tamia_smoke.build_smoke_packet(args)
    scenario_by_id = {s.scenario_id: s for s in packet.scenarios}

    # tests._fake_runtime's shared hook fake is deliberately NOT masking-aware -- the two
    # *_generated_only scenarios, which assert real prefill masking, are EXPECTED to fail
    # against it (see this module's docstring); real masking numerics are covered elsewhere.
    assert scenario_by_id["qwen_generated_only"].passed is False
    assert scenario_by_id["gemma_generated_only"].passed is False

    # Every OTHER scenario -- including ones that ran AFTER the two expected failures --
    # must still have run and passed: failure aggregation never short-circuits the run, and
    # a later pass (app_smoke_stubbed, gemma_all) never masks an earlier failure.
    for scenario_id in (
        "smoke_entries_hidden", "identity_cannot_be_swapped", "qwen_multi_sae_prohibited",
        "gemma_cross_layer_capability_limit", "qwen_backend_layer_mismatch_refused",
        "qwen_all", "gemma_all", "app_smoke_stubbed",
    ):
        assert scenario_by_id[scenario_id].passed is True, scenario_by_id[scenario_id].as_dict()

    assert packet.passed is False

    scenario_ids = [s.scenario_id for s in packet.scenarios]
    assert scenario_ids.index("qwen_all") < scenario_ids.index("gemma_all"), "Qwen phase must fully precede Gemma phase"
    assert scenario_ids.index("gemma_generated_only") < scenario_ids.index("app_smoke_stubbed")

    assert packet.runtime_extraction_source_commits["qwen"] == "e63b08eb94667be5b9a10425814e111fb9f6cefb"
    assert packet.runtime_extraction_source_commits["gemma"] == "de3b4994b1b7449fd4d967abfe63bf196772ef11"
    assert packet.acceptance_evidence_commits["qwen"] == "b6d598b5dca8c47861aa77aeefee1f75b2832133"
    assert packet.acceptance_evidence_commits["gemma"] == "b6d598b5dca8c47861aa77aeefee1f75b2832133"
    assert len(packet.product_commit) == 40  # a real git rev-parse HEAD of this repository


def test_build_smoke_packet_passes_when_positions_scenarios_are_stubbed_out(monkeypatch, tmp_path):
    """With every real-weight position scenario replaced by a passing stub
    (isolating orchestration from the shared fake's masking limitation),
    the packet as a whole passes -- proving the aggregation logic itself
    has no hidden always-fail bias."""
    monkeypatch.setattr(
        tamia_smoke, "run_qwen_position_scenario",
        lambda backend, positions, **kwargs: tamia_smoke._ok(f"qwen_{positions.value}", "qwen", "stubbed"),
    )
    monkeypatch.setattr(
        tamia_smoke, "run_gemma_position_scenario",
        lambda backend, positions, **kwargs: tamia_smoke._ok(f"gemma_{positions.value}", "gemma", "stubbed"),
    )
    monkeypatch.setattr(
        tamia_smoke, "run_application_smoke",
        lambda **kwargs: [tamia_smoke._ok("app_smoke_stubbed", "both", "stubbed")],
    )
    # QwenRuntimeBackend/GemmaRuntimeBackend construction itself never touches torch (lazy by
    # construction -- see their own module docstrings), so no fakes are needed here at all.
    args = _base_cli_args(output=str(tmp_path / "packet.json"))
    packet = tamia_smoke.build_smoke_packet(args)
    assert packet.passed, [s.as_dict() for s in packet.scenarios if not s.passed]


# ---------------------------------------------------------------------------
# CLI: parse_args validation, main()'s output-writing and exit code
# ---------------------------------------------------------------------------


def test_max_new_tokens_over_four_is_rejected():
    with pytest.raises(SystemExit):
        tamia_smoke.parse_args(
            [
                "--qwen-model-path", "x", "--qwen-sae-path", "y", "--gemma-model-path", "z", "--gemma-sae-path", "w",
                "--max-new-tokens", "5",
            ]
        )


def test_max_new_tokens_defaults_to_four():
    args = tamia_smoke.parse_args(
        ["--qwen-model-path", "x", "--qwen-sae-path", "y", "--gemma-model-path", "z", "--gemma-sae-path", "w"]
    )
    assert args.max_new_tokens == 4


def test_main_writes_output_json_and_returns_zero_when_passed(monkeypatch, tmp_path):
    packet = tamia_smoke.SmokePacket(
        product_commit="deadbeef", runtime_extraction_source_commits={"qwen": "a", "gemma": "b"},
        acceptance_evidence_commits={"qwen": "c", "gemma": "c"}, scenarios=(tamia_smoke._ok("s1", "qwen", "fine"),),
    )
    monkeypatch.setattr(tamia_smoke, "build_smoke_packet", lambda args: packet)
    output_path = tmp_path / "packet.json"
    exit_code = tamia_smoke.main(
        [
            "--qwen-model-path", "x", "--qwen-sae-path", "y", "--gemma-model-path", "z", "--gemma-sae-path", "w",
            "--output", str(output_path),
        ]
    )
    assert exit_code == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["passed"] is True
    assert written["scenarios"][0]["scenario_id"] == "s1"


def test_main_returns_nonzero_when_any_scenario_failed(monkeypatch, tmp_path):
    packet = tamia_smoke.SmokePacket(
        product_commit="deadbeef", runtime_extraction_source_commits={"qwen": "a", "gemma": "b"},
        acceptance_evidence_commits={"qwen": "c", "gemma": "c"},
        scenarios=(tamia_smoke._ok("s1", "qwen", "fine"), tamia_smoke._fail("s2", "gemma", "broken", error="boom")),
    )
    monkeypatch.setattr(tamia_smoke, "build_smoke_packet", lambda args: packet)
    output_path = tmp_path / "packet.json"
    exit_code = tamia_smoke.main(
        [
            "--qwen-model-path", "x", "--qwen-sae-path", "y", "--gemma-model-path", "z", "--gemma-sae-path", "w",
            "--output", str(output_path),
        ]
    )
    assert exit_code == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["passed"] is False


# ---------------------------------------------------------------------------
# Pins the Tamia no-install launch form: PYTHONPATH=<repo root> is
# sufficient, no `pip install -e .` and no environment mutation required
# (docs/tamia_smoke.md's "Exact Tamia submission command").
# ---------------------------------------------------------------------------


def test_smoke_entry_point_runs_via_pythonpath_alone_without_any_pip_install():
    """Runs `python -S -m sae_concept_lab.smoke.tamia_smoke --help` with
    `-S` (skip site-packages/site-initialization entirely -- no installed
    package, editable or otherwise, is even reachable) and ONLY
    `PYTHONPATH` pointing at this repository's own root added on top of
    the inherited environment. A shared Tamia venv (e.g.
    /home/y/yazid/sprint-venv) is never installed into or mutated by this
    procedure -- this test is the mechanical proof that pointing
    PYTHONPATH at an extracted archive's root is sufficient by itself.
    `--help` also never reaches build_smoke_packet (argparse exits first),
    so this doubles as direct proof that importing this whole module tree
    touches no third-party package at all -- the lazy-import discipline
    extracted_runtime/__init__.py documents, holding all the way up
    through this packet's own entry point."""
    repo_root = tamia_smoke._repo_root()
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, "-S", "-m", "sae_concept_lab.smoke.tamia_smoke", "--help"],
        capture_output=True, text=True, env=env, timeout=30, cwd=str(repo_root),
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "--qwen-model-path" in result.stdout
    assert "--gemma-model-path" in result.stdout
