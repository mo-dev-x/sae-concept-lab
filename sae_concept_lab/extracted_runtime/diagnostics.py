"""Mechanically extracted from qwen-sae-interp scripts/legacy/final_pairing_harness.py
(checkout de3b499; confirmed byte-identical back to e63b08e -- see
provenance/source_import.json's runtime_mirror extractions). SHARED by both
pairings: builds the per-hook-call diagnostic trace and the mechanical
verdict, and enforces HF_HUB_OFFLINE=1. Never re-implemented -- every
function body below is copied verbatim from the source commit."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass
class InterventionTrace:
    call_index: int
    call_classification: Literal["prefill", "decode"]
    requested_mode: str
    requested_dose_or_raw: str
    calibration_input: float | None
    resolved_absolute_target: float
    backend_received_value: float
    hook_name: str
    hooked_tensor_shape: tuple[int, ...]
    feature_activation_before: float
    assigned_feature_value: float
    feature_activation_after: float
    residual_delta_norm: float
    residual_norm: float


def wrap_hook_with_diagnostics(
    inner_hook_fn,
    *,
    sae,
    feature_index: int,
    mode: str,
    dose_or_raw_label: str,
    calibration_input: float | None,
    resolved_absolute_target: float,
    hook_name: str,
    trace_out: list[InterventionTrace],
):
    """Wraps an already-built hook_fn from _make_clamp_hook (imported
    unmodified) with pure observation. Never alters what inner_hook_fn
    computes or returns -- every field below is captured by an INDEPENDENT
    encode() call on the tensor going in and the tensor coming out, not by
    reading _make_clamp_hook's internals.

    feature_activation_after is a diagnostic RE-ENCODE of the modified
    residual, not a guaranteed exact readback of assigned_feature_value:
    encode(decode(x)) is not necessarily an exact identity for a lossy
    dictionary, so a large-but-inexact match to the target is the expected,
    healthy signal; an activation_after indistinguishable from
    activation_before is the actual "intervention disappeared" signal this
    trace exists to catch.

    call_classification: call_index == 0 is "prefill" (the full-prompt
    call), every later call is "decode" (a single new token under the KV
    cache) -- this matches HookedTransformer.generate()'s own per-step call
    pattern (docs/positions_semantics.md) and the standard HF GenerationMixin
    cached-decode pattern the raw-HF Qwen path also relies on."""
    call_counter = {"value": 0}

    def hook_fn(resid, hook):
        import torch

        call_index = call_counter["value"]
        call_counter["value"] += 1
        classification: Literal["prefill", "decode"] = "prefill" if call_index == 0 else "decode"

        with torch.no_grad():
            feats_before = sae.encode(resid.to(torch.float32))
            activation_before = float(feats_before[0, -1, feature_index].item())

        output = inner_hook_fn(resid, hook)

        with torch.no_grad():
            feats_after = sae.encode(output.to(torch.float32))
            activation_after = float(feats_after[0, -1, feature_index].item())
            delta = (output - resid).to(torch.float32)
            residual_delta_norm = float(delta.norm().item())
            residual_norm = float(resid.to(torch.float32).norm().item())

        trace_out.append(
            InterventionTrace(
                call_index=call_index,
                call_classification=classification,
                requested_mode=mode,
                requested_dose_or_raw=dose_or_raw_label,
                calibration_input=calibration_input,
                resolved_absolute_target=resolved_absolute_target,
                backend_received_value=resolved_absolute_target,
                hook_name=hook_name,
                hooked_tensor_shape=tuple(resid.shape),
                feature_activation_before=activation_before,
                assigned_feature_value=resolved_absolute_target,
                feature_activation_after=activation_after,
                residual_delta_norm=residual_delta_norm,
                residual_norm=residual_norm,
            )
        )
        return output

    return hook_fn


def find_first_disappearance_boundary(
    traces: list[InterventionTrace], *, positions: str
) -> InterventionTrace | None:
    """The first call whose residual_delta_norm is 0.0 where that is NOT
    the accepted generated_only first-call no-op (docs/positions_semantics.md,
    preserved here rather than re-litigated). None means no boundary found
    -- every applicable call showed a nonzero delta."""
    for t in traces:
        if positions == "generated_only" and t.call_index == 0:
            continue
        if t.residual_delta_norm == 0.0:
            return t
    return None


def mechanical_verdict(traces: list[InterventionTrace], *, positions: str) -> dict[str, Any]:
    boundary = find_first_disappearance_boundary(traces, positions=positions)
    applicable = [t for t in traces if not (positions == "generated_only" and t.call_index == 0)]
    return {
        "hook_invocation_count": len(traces),
        "prefill_call_count": sum(1 for t in traces if t.call_classification == "prefill"),
        "decode_call_count": sum(1 for t in traces if t.call_classification == "decode"),
        "nonzero_steer_confirmed": bool(applicable) and all(t.residual_delta_norm > 0.0 for t in applicable),
        "first_disappearance_boundary": asdict(boundary) if boundary is not None else None,
    }


def _require_offline() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError(
            "HF_HUB_OFFLINE=1 is not set. Every Tamia compute-node job in this project "
            "requires it -- refusing to proceed rather than risk a silent network fetch."
        )
