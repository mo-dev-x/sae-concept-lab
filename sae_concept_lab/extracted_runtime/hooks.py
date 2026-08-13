"""Mechanically extracted from qwen-sae-interp interplab/interventions/hooks.py
at checkout de3b499 (confirmed byte-identical at e63b08e). SHARED clamp/ablate
hook mechanism, extracted at FUNCTION granularity, not whole-file: the source
file's other public entry point, attach(), and everything only it needs
(_check_positions_contract, _check_direction_seed_contract, AttachHandle,
_fp32_copy, _direction_vector, _make_add_direction_hook), import
interplab.interventions.spec.InterventionSpec -- an internal qwen-sae-interp
package this product must never depend on, structurally, in any form. Neither
final_pairing_harness.py (this product's own qwen_loader.py/gemma_loader.py)
nor this product's own backends call attach() at all -- both import
_make_clamp_hook directly, exactly as the source script does (see its own
docstring: '_make_clamp_hook is imported unmodified, exactly as
scripts/legacy/gemma3_tool.py already does'). ABLATE is CLAMP with
value 0.0 (attach()'s own rule: "clamp_value = 0.0 if spec.kind == 'ablate'
else ...") -- this product's backends apply that same rule themselves before
calling _make_clamp_hook, so _make_add_direction_hook (a different
intervention kind, "add_direction", not used by either final-pairing target)
is correctly excluded, not merely omitted."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Annotation-only: _make_clamp_hook's signature below is copied verbatim
    # from the source, which type-hints its sae_fp32 parameter as SAE. Real
    # sae_lens is never imported at runtime (`from __future__ import
    # annotations` already makes every annotation a lazy string) -- this
    # guard exists only so a static checker can resolve the name, without
    # adding sae_lens as a real dependency of this module.
    from sae_lens import SAE


@dataclasses.dataclass(frozen=True)
class CallStats:
    """One record per hook invocation during an `attach()` context (§5 SS7
    measurement clause: "per-run logging of injected-delta norms relative
    to residual norms")."""

    delta_norm: float
    residual_norm: float


class _PositionCounter:
    """Tracks the absolute sequence position reached so far across hook
    calls within one `attach()` context (prefill call, then one call per
    KV-cached decode step)."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0


def _resolve_prompt_lengths_tensor(prompt_lengths, batch_size: int, device) -> torch.Tensor:
    if isinstance(prompt_lengths, int):
        return torch.full((batch_size,), prompt_lengths, dtype=torch.long, device=device)
    lengths = torch.as_tensor(list(prompt_lengths), dtype=torch.long, device=device)
    if lengths.shape[0] != batch_size:
        raise ValueError(f"prompt_lengths has {lengths.shape[0]} entries but batch size is {batch_size}")
    return lengths


def _positions_mask(counter: _PositionCounter, seq_len: int, batch_size: int, prompt_lengths, device) -> torch.Tensor:
    """[batch, seq_len] bool mask, True where this call's absolute position
    is >= that row's prompt length. Advances `counter` by seq_len."""
    start = counter.value
    counter.value += seq_len
    abs_positions = torch.arange(start, start + seq_len, device=device)
    lengths = _resolve_prompt_lengths_tensor(prompt_lengths, batch_size, device)
    return abs_positions.unsqueeze(0) >= lengths.unsqueeze(1)


def _make_clamp_hook(
    sae_fp32: SAE, feature_index: int, clamp_value: float, positions: str, prompt_lengths, stats: list[CallStats]
):
    counter = _PositionCounter()

    def hook_fn(resid, hook):
        batch, seq_len, _ = resid.shape
        mask = None
        if positions == "generated_only":
            mask = _positions_mask(counter, seq_len, batch, prompt_lengths, resid.device)
            if not bool(mask.any()):
                # Bullet 3: masked positions are never touched -- no encode/decode
                # round trip at all when nothing in this call needs steering.
                stats.append(CallStats(delta_norm=0.0, residual_norm=resid.norm().item()))
                return resid

        x = resid
        x32 = x.to(torch.float32)
        feats = sae_fp32.encode(x32)
        clean_recon = sae_fp32.decode(feats)
        feats_clamped = feats.clone()
        feats_clamped[..., feature_index] = clamp_value
        clamped_recon = sae_fp32.decode(feats_clamped)
        delta32 = clamped_recon - clean_recon
        delta = delta32.to(x.dtype)
        steered = x + delta

        # Structural selection, not additive zeroing: masked positions take `x`
        # directly, regardless of what `steered` computed to there (a
        # multiply-by-zero mask would still propagate NaN/Inf via 0*NaN=NaN;
        # `where` never lets a masked position's output depend on `steered`).
        result = torch.where(mask.unsqueeze(-1), steered, x) if mask is not None else steered

        effective_delta = result - x
        stats.append(CallStats(delta_norm=effective_delta.norm().item(), residual_norm=x.norm().item()))
        return result

    return hook_fn
