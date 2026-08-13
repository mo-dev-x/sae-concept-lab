"""Tamia final Tamia product-integration smoke packet: exercises the real,
mechanically-accepted QwenRuntimeBackend/GemmaRuntimeBackend through the
SAME canonical resolution -> execution-guard -> backend-translation path
the application itself uses (core.logic.send_message, execution_guard's
defensive re-enforcement inside each backend, ui.app_ui.build_demo).

This module never imports torch/transformers/transformer_lens/sae_lens at
module scope, and never calls anything in `sae_concept_lab.extracted_runtime`
directly -- every real generation goes through QwenRuntimeBackend/
GemmaRuntimeBackend exactly as the application would call them. The only
things imported at module scope from `extracted_runtime` are the pure,
torch-free identity validators in `targets.py` (used by
`scenario_identity_cannot_be_swapped` below), which is the one place this
packet reaches past the backend layer -- to prove the loader's OWN
fail-closed identity guard, not to bypass it.

See docs/tamia_smoke.md for the exact Tamia submission command, expected
output artifact, and failure classification table.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sae_concept_lab.canonical.concept_bundle import (
    MultipleExecutionGroupsError,
    MultipleSaeIdentitiesAtLayerError,
    PositionMode,
    Provenance,
    resolve_control,
)
from sae_concept_lab.core.gemma_backend import GemmaRuntimeBackend
from sae_concept_lab.core.logic import send_message
from sae_concept_lab.core.protocol import GenerationRequest
from sae_concept_lab.core.qwen_backend import QwenRuntimeBackend
from sae_concept_lab.core.runtime_acceptance import (
    ACCEPTANCE_REGISTRY,
    RuntimeAcceptanceError,
    is_mechanically_accepted,
)
from sae_concept_lab.fixtures.loader import load_entries
from sae_concept_lab.smoke import entries

# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    pairing: str
    passed: bool
    summary: str
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "pairing": self.pairing,
            "passed": self.passed,
            "summary": self.summary,
            "detail": self.detail,
            "error": self.error,
        }


@dataclasses.dataclass(frozen=True)
class SmokePacket:
    product_commit: str
    runtime_extraction_source_commits: dict[str, str]
    acceptance_evidence_commits: dict[str, str | None]
    scenarios: tuple[ScenarioResult, ...]

    @property
    def passed(self) -> bool:
        """Aggregated once, at the end, over the COMPLETE, immutable list
        of every scenario that actually ran -- never a running flag
        threaded through the run, so a later successful scenario can never
        mask an earlier failure."""
        return all(s.passed for s in self.scenarios)

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_commit": self.product_commit,
            "runtime_extraction_source_commits": self.runtime_extraction_source_commits,
            "acceptance_evidence_commits": self.acceptance_evidence_commits,
            "passed": self.passed,
            "scenarios": [s.as_dict() for s in self.scenarios],
        }


def _ok(scenario_id: str, pairing: str, summary: str, detail: dict[str, Any] | None = None) -> ScenarioResult:
    return ScenarioResult(scenario_id=scenario_id, pairing=pairing, passed=True, summary=summary, detail=detail or {})


def _fail(
    scenario_id: str, pairing: str, summary: str, *, error: str | None = None, detail: dict[str, Any] | None = None
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id, pairing=pairing, passed=False, summary=summary, detail=detail or {}, error=error
    )


def _run_guarded(scenario_id: str, pairing: str, fn) -> ScenarioResult:
    """Runs fn() (a zero-arg callable returning a ScenarioResult) and turns
    ANY exception into a failed ScenarioResult instead of letting it abort
    the rest of the packet. This is what makes "a later successful
    scenario cannot mask an earlier failure" true: every scenario is
    attempted and recorded, regardless of what happened before it."""
    try:
        return fn()
    except Exception as exc:  # any scenario failure must become a recorded, non-fatal result
        return _fail(scenario_id, pairing, f"{scenario_id} raised an unexpected exception", error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Identity / provenance helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_head_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-outside-git-checkout"


def _runtime_extraction_source_commits(repo_root: Path) -> dict[str, str]:
    manifest = json.loads((repo_root / "provenance" / "source_import.json").read_text(encoding="utf-8"))
    by_id = {e["extraction_id"]: e for e in manifest["extractions"]}
    return {
        "qwen": by_id["qwen_runtime_mirror"]["source_repository"]["checkout_commit"],
        "gemma": by_id["gemma_runtime_mirror"]["source_repository"]["checkout_commit"],
    }


def _acceptance_evidence_commits() -> dict[str, str | None]:
    return {pairing: (record.evidence_commit if record is not None else None) for pairing, record in ACCEPTANCE_REGISTRY.items()}


def _require_mechanical_acceptance(pairing: str) -> None:
    """The runner's own precondition, checked first in every real-weight
    scenario: missing (or previously reset) acceptance evidence must
    prevent this runner from ever constructing/loading a real backend for
    that pairing -- never a warning, never a degraded dev-mode tag."""
    if not is_mechanically_accepted(pairing):
        raise RuntimeAcceptanceError(
            f"refusing to run a real {pairing!r} scenario: is_mechanically_accepted({pairing!r}) is False -- "
            "core/runtime_acceptance.py has no attached, verified RuntimeAcceptanceRecord for this pairing. "
            "This smoke runner never proceeds to construct or load a real backend in that state."
        )


def _all_smoke_entries_by_concept_id() -> dict[str, Any]:
    all_entries = [
        entries.qwen_smoke_entry(PositionMode.ALL),
        entries.qwen_smoke_entry(PositionMode.GENERATED_ONLY),
        entries.gemma_smoke_entry(PositionMode.ALL),
        entries.gemma_smoke_entry(PositionMode.GENERATED_ONLY),
        entries.qwen_multi_sae_same_layer_entry(),
        entries.gemma_cross_layer_entry(),
    ]
    return {e.concept_id: e for e in all_entries}


def _release_gpu_memory() -> None:
    """Best-effort GPU memory release between pairing phases: gc.collect()
    plus torch.cuda.empty_cache() if torch happens to be importable. A
    no-op (never a failure) when it is not -- e.g. this repository's own
    CPU/no-torch dev venv. Callers must drop every reference to the
    backend whose memory should be released BEFORE calling this."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


# ---------------------------------------------------------------------------
# CPU-safe defensive assertions -- no torch import is ever reached by any
# of these: execution_guard.require_group_from_resolved (called inside
# each backend's generate()) raises before _ensure_loaded()/torch import,
# and extracted_runtime.targets' validators are pure string/regex checks.
# ---------------------------------------------------------------------------


def scenario_smoke_entries_hidden() -> ScenarioResult:
    qwen_ids = {e.concept_id for e in load_entries("qwen")}
    gemma_ids = {e.concept_id for e in load_entries("gemma")}
    leaked = [cid for cid in entries.ALL_SMOKE_CONCEPT_IDS if cid in qwen_ids or cid in gemma_ids]
    non_fake = [cid for cid, entry in _all_smoke_entries_by_concept_id().items() if entry.provenance is not Provenance.FAKE]
    if leaked or non_fake:
        return _fail(
            "smoke_entries_hidden", "both",
            "smoke entries are reachable via fixture discovery or are not provenance=FAKE",
            error=f"leaked={leaked} non_fake={non_fake}",
        )
    return _ok(
        "smoke_entries_hidden", "both",
        "no smoke concept_id appears in fixtures.loader.load_entries() for either model_key, and every "
        "smoke entry is provenance=FAKE, so evaluate_publishability's ATTESTED requirement alone would "
        "already block it even if it were somehow reachable",
        {"checked_concept_ids": list(entries.ALL_SMOKE_CONCEPT_IDS)},
    )


def scenario_identity_cannot_be_swapped() -> ScenarioResult:
    from sae_concept_lab.extracted_runtime.targets import (
        GEMMA_3_12B_IT_TARGET,
        QWEN_3_5_27B_TARGET,
        TargetIdentityMismatch,
        validate_local_snapshot_identity,
    )

    gemma_shaped_path = "models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
    qwen_shaped_path = "models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    problems = []
    try:
        validate_local_snapshot_identity(gemma_shaped_path, QWEN_3_5_27B_TARGET, which="model")
        problems.append("a Gemma-repo-shaped path was accepted as a Qwen model identity")
    except TargetIdentityMismatch:
        pass
    try:
        validate_local_snapshot_identity(qwen_shaped_path, GEMMA_3_12B_IT_TARGET, which="model")
        problems.append("a Qwen-repo-shaped path was accepted as a Gemma model identity")
    except TargetIdentityMismatch:
        pass
    if problems:
        return _fail("identity_cannot_be_swapped", "both", "pairing identities can be swapped", error="; ".join(problems))
    return _ok(
        "identity_cannot_be_swapped", "both",
        "extracted_runtime.targets.validate_local_snapshot_identity refuses a Gemma path presented as a "
        "Qwen model identity, and a Qwen path presented as a Gemma model identity",
        {"gemma_shaped_path": gemma_shaped_path, "qwen_shaped_path": qwen_shaped_path},
    )


def scenario_qwen_multi_sae_prohibited(model_path: str, sae_path: str) -> ScenarioResult:
    backend = QwenRuntimeBackend(model_path=model_path, sae_path=sae_path, qwen_layer=entries.QWEN_SMOKE_LAYER)
    entry = entries.qwen_multi_sae_same_layer_entry()
    resolved = resolve_control(entry, direction=entries.SMOKE_DIRECTION, strength=entries.SMOKE_STRENGTH)
    request = GenerationRequest(
        history=(), prompt=entries.SMOKE_PROMPT, model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    try:
        backend.generate(request)
    except MultipleSaeIdentitiesAtLayerError as exc:
        return _ok(
            "qwen_multi_sae_prohibited", "qwen",
            "two SAE identities at one layer are correctly refused as PROHIBITED",
            {"error": str(exc), "classification": "PROHIBITED"},
        )
    return _fail(
        "qwen_multi_sae_prohibited", "qwen", "a multi-SAE-at-one-layer request was NOT refused",
        error="MultipleSaeIdentitiesAtLayerError was not raised",
    )


def scenario_gemma_cross_layer_capability_limit(model_path: str, sae_path: str) -> ScenarioResult:
    backend = GemmaRuntimeBackend(model_path=model_path, sae_path=sae_path)
    entry = entries.gemma_cross_layer_entry()
    resolved = resolve_control(entry, direction=entries.SMOKE_DIRECTION, strength=entries.SMOKE_STRENGTH)
    request = GenerationRequest(
        history=(), prompt=entries.SMOKE_PROMPT, model_key="gemma", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    try:
        backend.generate(request)
    except MultipleExecutionGroupsError as exc:
        return _ok(
            "gemma_cross_layer_capability_limit", "gemma",
            "two distinct (sae_id, layer) execution groups are correctly refused as CAPABILITY_LIMIT",
            {"error": str(exc), "classification": "CAPABILITY_LIMIT"},
        )
    return _fail(
        "gemma_cross_layer_capability_limit", "gemma", "a cross-layer request was NOT refused",
        error="MultipleExecutionGroupsError was not raised",
    )


def scenario_qwen_backend_layer_mismatch_refused(model_path: str, sae_path: str) -> ScenarioResult:
    """Same-layer enforcement from the other direction: a resolved target
    at the accepted engineering layer, presented to a backend configured
    for a DIFFERENT layer, must be refused rather than silently executed
    against the wrong hook."""
    backend = QwenRuntimeBackend(model_path=model_path, sae_path=sae_path, qwen_layer=entries.QWEN_SMOKE_LAYER + 1)
    entry = entries.qwen_smoke_entry(PositionMode.ALL)
    resolved = resolve_control(entry, direction=entries.SMOKE_DIRECTION, strength=entries.SMOKE_STRENGTH)
    request = GenerationRequest(
        history=(), prompt=entries.SMOKE_PROMPT, model_key="qwen", decoding={}, seed=0,
        apply_intervention=True, resolved_config=resolved,
    )
    try:
        backend.generate(request)
    except ValueError as exc:
        if "does not match this backend's configured qwen_layer" in str(exc):
            return _ok(
                "qwen_backend_layer_mismatch_refused", "qwen",
                "a resolved target layer that disagrees with the backend's configured qwen_layer is refused",
                {"error": str(exc)},
            )
        raise
    return _fail(
        "qwen_backend_layer_mismatch_refused", "qwen", "a backend/resolved-target layer mismatch was NOT refused",
        error="ValueError was not raised",
    )


# ---------------------------------------------------------------------------
# Real-weight scenarios: Qwen ALL/GENERATED_ONLY, Gemma ALL/GENERATED_ONLY.
# Each drives resolve_control -> core.logic.send_message -> backend.generate()
# -- the exact chain ui/tab.py's own _on_send calls.
# ---------------------------------------------------------------------------


def _finish_position_scenario(
    *, scenario_id: str, pairing: str, positions: PositionMode, result, product_commit: str,
    extraction_source_commit: str,
) -> ScenarioResult:
    diagnostics = result.diagnostics
    if diagnostics is None:
        return _fail(scenario_id, pairing, "backend returned no diagnostics for an intervention request")
    resolved = result.resolved_config
    trace = diagnostics["trace"]
    prefill = next((t for t in trace if t["call_index"] == 0), None)
    if prefill is None:
        return _fail(scenario_id, pairing, "no prefill (call_index 0) entry in the returned trace", detail={"trace": trace})

    if positions is PositionMode.ALL:
        positions_ok = prefill["residual_delta_norm"] > 0.0
        positions_note = (
            "ALL: prefill (call_index 0) shows a nonzero residual delta, as required"
            if positions_ok else
            "ALL: prefill (call_index 0) shows ZERO residual delta -- ALL must modify prefill"
        )
    else:
        positions_ok = prefill["residual_delta_norm"] == 0.0
        positions_note = (
            "GENERATED_ONLY: prefill (call_index 0) is correctly masked off (zero residual delta)"
            if positions_ok else
            "GENERATED_ONLY: prefill (call_index 0) shows a NONZERO residual delta -- masking failed"
        )

    record = ACCEPTANCE_REGISTRY.get(pairing)
    detail = {
        "product_commit": product_commit,
        "runtime_extraction_source_commit": extraction_source_commit,
        "acceptance_evidence_commit": record.evidence_commit if record is not None else None,
        "acceptance_claim": record.claim if record is not None else None,
        "model_and_sae_identity": diagnostics.get("provenance"),
        "resolved_canonical_execution_dict": resolved.execution_dict() if resolved is not None else None,
        "execution_fingerprint": resolved.execution_fingerprint() if resolved is not None else None,
        "state_fingerprint": resolved.state_fingerprint() if resolved is not None else None,
        "entry_audit_fingerprint": resolved.entry_audit_fingerprint if resolved is not None else None,
        "requested": diagnostics.get("requested"),
        "resolved_absolute_target": diagnostics.get("resolved_absolute_target"),
        "backend_received_value": diagnostics.get("backend_received_value"),
        "trace": trace,
        "verdict": diagnostics.get("verdict"),
        "generated_text": result.text,
        "positions_assertion": positions_note,
    }
    if not positions_ok:
        return _fail(scenario_id, pairing, positions_note, error="positions semantics assertion failed", detail=detail)
    return _ok(scenario_id, pairing, f"{scenario_id} passed ({positions_note})", detail)


def run_qwen_position_scenario(
    backend: QwenRuntimeBackend, positions: PositionMode, *, max_new_tokens: int, product_commit: str,
    extraction_source_commit: str,
) -> ScenarioResult:
    scenario_id = f"qwen_{positions.value}"
    _require_mechanical_acceptance("qwen")
    entry = entries.qwen_smoke_entry(positions)
    resolved = resolve_control(entry, direction=entries.SMOKE_DIRECTION, strength=entries.SMOKE_STRENGTH)
    _new_history, result = send_message(
        backend=backend, history=[], prompt=entries.SMOKE_PROMPT, model_key="qwen",
        decoding={"max_new_tokens": max_new_tokens}, seed=0, resolved_config=resolved,
    )
    return _finish_position_scenario(
        scenario_id=scenario_id, pairing="qwen", positions=positions, result=result,
        product_commit=product_commit, extraction_source_commit=extraction_source_commit,
    )


def run_gemma_position_scenario(
    backend: GemmaRuntimeBackend, positions: PositionMode, *, max_new_tokens: int, product_commit: str,
    extraction_source_commit: str,
) -> ScenarioResult:
    scenario_id = f"gemma_{positions.value}"
    _require_mechanical_acceptance("gemma")
    entry = entries.gemma_smoke_entry(positions)
    resolved = resolve_control(entry, direction=entries.SMOKE_DIRECTION, strength=entries.SMOKE_STRENGTH)
    _new_history, result = send_message(
        backend=backend, history=[], prompt=entries.SMOKE_PROMPT, model_key="gemma",
        decoding={"max_new_tokens": max_new_tokens}, seed=0, resolved_config=resolved,
    )
    return _finish_position_scenario(
        scenario_id=scenario_id, pairing="gemma", positions=positions, result=result,
        product_commit=product_commit, extraction_source_commit=extraction_source_commit,
    )


# ---------------------------------------------------------------------------
# Application smoke: boot the real Gradio app with real backends, HTTP
# probe, one bounded request via the same adapter ui/tab.py's _on_send
# calls, clean shutdown, confirm release mode still refuses.
# ---------------------------------------------------------------------------


def _probe_http_200(server_name: str, server_port: int) -> ScenarioResult:
    import urllib.request

    host = "127.0.0.1" if server_name == "0.0.0.0" else server_name
    url = f"http://{host}:{server_port}/"
    with urllib.request.urlopen(url, timeout=30) as response:  # localhost-only probe of our own just-launched server
        status = response.status
    if status != 200:
        return _fail("app_smoke_http_200", "both", f"HTTP probe returned {status}, not 200", error=f"status={status}")
    return _ok("app_smoke_http_200", "both", f"application responded HTTP {status} at {url}", {"url": url, "status": status})


def _one_bounded_request_via_app_adapter(qwen_backend: QwenRuntimeBackend, max_new_tokens: int) -> ScenarioResult:
    entry = entries.qwen_smoke_entry(PositionMode.ALL)
    resolved = resolve_control(entry, direction=entries.SMOKE_DIRECTION, strength=entries.SMOKE_STRENGTH)
    _new_history, result = send_message(
        backend=qwen_backend, history=[], prompt=entries.SMOKE_PROMPT, model_key="qwen",
        decoding={"max_new_tokens": max_new_tokens}, seed=0, resolved_config=resolved,
    )
    if result.diagnostics is None:
        return _fail("app_smoke_bounded_request", "qwen", "the running app's own backend returned no diagnostics")
    return _ok(
        "app_smoke_bounded_request", "qwen",
        "one bounded request through core.logic.send_message (the same adapter ui/tab.py's _on_send calls) "
        "against the running application's own real backend instance succeeded",
        {"generated_text": result.text, "verdict": result.diagnostics.get("verdict")},
    )


def _confirm_release_still_refuses(
    gemma_backend: GemmaRuntimeBackend, qwen_backend: QwenRuntimeBackend, evidence_registry_root: str | None
) -> ScenarioResult:
    from sae_concept_lab.fixtures.loader import ReleaseGateError, enforce_release_gate

    refusal_messages: dict[str, str] = {}
    for model_key, backend in (("gemma", gemma_backend), ("qwen", qwen_backend)):
        try:
            enforce_release_gate(mode="release", backend=backend, model_key=model_key, evidence_registry_root=evidence_registry_root)
        except ReleaseGateError as exc:
            refusal_messages[model_key] = str(exc)
            continue
        return _fail(
            "app_smoke_release_still_refuses", "both", f"release mode did NOT refuse for model_key={model_key!r}",
            error="ReleaseGateError was not raised",
        )
    return _ok(
        "app_smoke_release_still_refuses", "both",
        "release mode still refuses for both pairings (no ATTESTED concepts are shipped, regardless of "
        "backend or mechanical-acceptance status)",
        {"refusal_messages": refusal_messages},
    )


def _entries_unreachable_via_ui() -> ScenarioResult:
    qwen_entries = load_entries("qwen")
    gemma_entries = load_entries("gemma")
    reachable = [
        cid for cid in entries.ALL_SMOKE_CONCEPT_IDS
        if any(e.concept_id == cid for e in qwen_entries) or any(e.concept_id == cid for e in gemma_entries)
    ]
    if reachable:
        return _fail(
            "app_smoke_entries_unreachable_via_ui", "both", "a smoke concept_id is reachable via the UI's own entries",
            error=f"reachable={reachable}",
        )
    return _ok(
        "app_smoke_entries_unreachable_via_ui", "both",
        "no hidden smoke concept_id is present in either tab's entries closure -- ui/tab.py's _on_send/"
        "_on_compare could never look one up even if a caller tried",
        {},
    )


def run_application_smoke(
    *, gemma_backend: GemmaRuntimeBackend, qwen_backend: QwenRuntimeBackend, server_name: str, server_port: int,
    evidence_registry_root: str | None, max_new_tokens: int,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = [_run_guarded("app_smoke_entries_unreachable_via_ui", "both", _entries_unreachable_via_ui)]

    demo = None
    try:
        from sae_concept_lab.ui.app_ui import build_demo

        gemma_entries = load_entries("gemma")
        qwen_entries = load_entries("qwen")
        demo = build_demo(
            gemma_entries=gemma_entries, qwen_entries=qwen_entries, gemma_backend=gemma_backend, qwen_backend=qwen_backend
        )
        demo.launch(server_name=server_name, server_port=server_port, share=False, prevent_thread_lock=True, quiet=True)
        results.append(_run_guarded("app_smoke_http_200", "both", lambda: _probe_http_200(server_name, server_port)))
        results.append(
            _run_guarded(
                "app_smoke_bounded_request", "qwen",
                lambda: _one_bounded_request_via_app_adapter(qwen_backend, max_new_tokens),
            )
        )
        results.append(
            _run_guarded(
                "app_smoke_release_still_refuses", "both",
                lambda: _confirm_release_still_refuses(gemma_backend, qwen_backend, evidence_registry_root),
            )
        )
    except Exception as exc:  # a boot failure must become a recorded scenario, not an uncaught crash
        results.append(_fail("app_smoke_boot", "both", "the application failed to boot for the smoke probe", error=f"{type(exc).__name__}: {exc}"))
    finally:
        if demo is not None:
            demo.close()
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_smoke_packet(args: argparse.Namespace) -> SmokePacket:
    repo_root = _repo_root()
    product_commit = _git_head_commit(repo_root)
    extraction_commits = _runtime_extraction_source_commits(repo_root)

    results: list[ScenarioResult] = []
    results.append(_run_guarded("smoke_entries_hidden", "both", scenario_smoke_entries_hidden))
    results.append(_run_guarded("identity_cannot_be_swapped", "both", scenario_identity_cannot_be_swapped))
    results.append(
        _run_guarded(
            "qwen_multi_sae_prohibited", "qwen",
            lambda: scenario_qwen_multi_sae_prohibited(args.qwen_model_path, args.qwen_sae_path),
        )
    )
    results.append(
        _run_guarded(
            "gemma_cross_layer_capability_limit", "gemma",
            lambda: scenario_gemma_cross_layer_capability_limit(args.gemma_model_path, args.gemma_sae_path),
        )
    )
    results.append(
        _run_guarded(
            "qwen_backend_layer_mismatch_refused", "qwen",
            lambda: scenario_qwen_backend_layer_mismatch_refused(args.qwen_model_path, args.qwen_sae_path),
        )
    )

    # --- Qwen phase: construct, run both position scenarios, then release GPU memory before Gemma ---
    qwen_backend = QwenRuntimeBackend(
        model_path=args.qwen_model_path, sae_path=args.qwen_sae_path, qwen_layer=entries.QWEN_SMOKE_LAYER,
        device=args.qwen_device, dtype=args.qwen_dtype,
        expected_model_revision=args.qwen_expected_model_revision, expected_sae_revision=args.qwen_expected_sae_revision,
    )
    for positions in (PositionMode.ALL, PositionMode.GENERATED_ONLY):
        results.append(
            _run_guarded(
                f"qwen_{positions.value}", "qwen",
                lambda positions=positions: run_qwen_position_scenario(
                    qwen_backend, positions, max_new_tokens=args.max_new_tokens, product_commit=product_commit,
                    extraction_source_commit=extraction_commits["qwen"],
                ),
            )
        )
    qwen_backend = None  # drop the only reference so _release_gpu_memory's gc.collect() can reclaim it
    _release_gpu_memory()

    # --- Gemma phase: construct, run both position scenarios, then release GPU memory ---
    gemma_backend = GemmaRuntimeBackend(
        model_path=args.gemma_model_path, sae_path=args.gemma_sae_path, device=args.gemma_device, dtype=args.gemma_dtype,
        expected_model_revision=args.gemma_expected_model_revision, expected_sae_revision=args.gemma_expected_sae_revision,
    )
    for positions in (PositionMode.ALL, PositionMode.GENERATED_ONLY):
        results.append(
            _run_guarded(
                f"gemma_{positions.value}", "gemma",
                lambda positions=positions: run_gemma_position_scenario(
                    gemma_backend, positions, max_new_tokens=args.max_new_tokens, product_commit=product_commit,
                    extraction_source_commit=extraction_commits["gemma"],
                ),
            )
        )
    gemma_backend = None  # drop the only reference so _release_gpu_memory's gc.collect() can reclaim it
    _release_gpu_memory()

    # --- Application smoke: a dual-pairing configuration, exactly as docs/tamia_launch.md documents
    # for launching both real backends together -- fresh backend instances, released after this phase.
    app_qwen_backend = QwenRuntimeBackend(
        model_path=args.qwen_model_path, sae_path=args.qwen_sae_path, qwen_layer=entries.QWEN_SMOKE_LAYER,
        device=args.qwen_device, dtype=args.qwen_dtype,
        expected_model_revision=args.qwen_expected_model_revision, expected_sae_revision=args.qwen_expected_sae_revision,
    )
    app_gemma_backend = GemmaRuntimeBackend(
        model_path=args.gemma_model_path, sae_path=args.gemma_sae_path, device=args.gemma_device, dtype=args.gemma_dtype,
        expected_model_revision=args.gemma_expected_model_revision, expected_sae_revision=args.gemma_expected_sae_revision,
    )
    results.extend(
        run_application_smoke(
            gemma_backend=app_gemma_backend, qwen_backend=app_qwen_backend, server_name=args.server_name,
            server_port=args.server_port, evidence_registry_root=args.evidence_registry_root,
            max_new_tokens=args.max_new_tokens,
        )
    )
    del app_qwen_backend, app_gemma_backend
    _release_gpu_memory()

    return SmokePacket(
        product_commit=product_commit,
        runtime_extraction_source_commits=extraction_commits,
        acceptance_evidence_commits=_acceptance_evidence_commits(),
        scenarios=tuple(results),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _max_new_tokens_type(value: str) -> int:
    n = int(value)
    if not (1 <= n <= 4):
        raise argparse.ArgumentTypeError(
            f"--max-new-tokens must be between 1 and 4 (dispatch requirement: no more than 4 new tokens "
            f"per scenario), got {n}"
        )
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-model-path", required=True, help="Local Qwen3.5-27B snapshot directory.")
    p.add_argument("--qwen-sae-path", required=True, help="Local Qwen-Scope layer0.sae.pt file.")
    p.add_argument("--qwen-device", default="cuda")
    p.add_argument("--qwen-dtype", default="bfloat16")
    p.add_argument(
        "--qwen-expected-model-revision", default="fc05daec18b0a78c049392ed2e771dde82bdf654",
        help="Defaults to the accepted evidence run's own revision (docs/tamia_launch.md); override if your "
        "inventory re-staged the snapshot under a different revision.",
    )
    p.add_argument("--qwen-expected-sae-revision", default="13d4221569f7ca5d3c1e605e3e3dc95117e4807c")
    p.add_argument("--gemma-model-path", required=True, help="Local gemma-3-12b-it snapshot directory.")
    p.add_argument("--gemma-sae-path", required=True, help="Local gemma-scope-2-12b-it-res SAE snapshot ROOT directory.")
    p.add_argument("--gemma-device", default="cuda")
    p.add_argument("--gemma-dtype", default="bfloat16")
    p.add_argument("--gemma-expected-model-revision", default="96b6f1eccf38110c56df3a15bffe176da04bfd80")
    p.add_argument("--gemma-expected-sae-revision", default="4c419f1ba0be8b7754d4151d4f26c23b92a9029e")
    p.add_argument("--max-new-tokens", type=_max_new_tokens_type, default=4)
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7861)
    p.add_argument(
        "--evidence-registry-root", default=None,
        help="Optional; passed through to the release-still-refuses check. Omitting it still proves a "
        "refusal (an absent registry root is itself refused fail-closed).",
    )
    p.add_argument("--output", default="tamia_smoke_packet.json", help="Path to write the aggregate JSON packet to.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    packet = build_smoke_packet(args)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(packet.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_path}", file=sys.stderr)

    for scenario in packet.scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        print(f"[{status}] {scenario.scenario_id} ({scenario.pairing}): {scenario.summary}", file=sys.stderr)

    if packet.passed:
        print("TAMIA SMOKE PACKET: ALL SCENARIOS PASSED", file=sys.stderr)
        return 0
    print("TAMIA SMOKE PACKET: AT LEAST ONE SCENARIO FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
