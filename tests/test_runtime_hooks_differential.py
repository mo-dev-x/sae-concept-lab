"""Differential tests against the REAL, unmodified, extracted hook/loader
functions (_make_clamp_hook, register_qwen_raw_hook, wrap_hook_with_diagnostics)
using real torch tensors and a minimal duck-typed identity SAE.

Skipped entirely when torch is not installed (`pytest.importorskip` at
module scope) -- this product's base install deliberately has no torch
(see extracted_runtime/__init__.py), so these tests are dormant here and
exist for whenever torch happens to be available (a future CI lane, or a
Tamia venv). tests/test_qwen_runtime_backend.py / test_gemma_runtime_backend.py
cover the BACKEND's own translation logic without needing torch at all --
this file is the complementary piece: proving the extracted numerics
themselves behave exactly as qwen-sae-interp's own documentation claims,
using the real, unedited function bodies, not a fake standing in for them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sae_concept_lab.extracted_runtime.diagnostics import (  # noqa: E402
    mechanical_verdict,
    wrap_hook_with_diagnostics,
)
from sae_concept_lab.extracted_runtime.hooks import _make_clamp_hook  # noqa: E402


class _IdentitySAE:
    """encode(x) == x, decode(feats) == feats -- d_in == d_sae, so the
    only thing _make_clamp_hook's real math does is overwrite one
    coordinate and hand the delta straight back, making the expected
    output trivially predictable without needing a real trained SAE."""

    def encode(self, x):
        return x.clone()

    def decode(self, feats):
        return feats.clone()


def test_clamp_all_positions_overwrites_exactly_the_clamped_feature():
    sae = _IdentitySAE()
    resid = torch.zeros(1, 3, 4)  # batch=1, seq_len=3, d_model=4
    stats = []
    hook_fn = _make_clamp_hook(sae, feature_index=2, clamp_value=5.0, positions="all",
                                prompt_lengths=None, stats=stats)
    out = hook_fn(resid, hook=None)
    assert torch.equal(out[..., 2], torch.full((1, 3), 5.0))
    assert torch.equal(out[..., 0], torch.zeros(1, 3))
    assert len(stats) == 1
    assert stats[0].delta_norm == pytest.approx((5.0**2 * 3) ** 0.5)  # 3 positions each clamped by 5.0


def test_ablate_is_clamp_to_zero():
    sae = _IdentitySAE()
    resid = torch.full((1, 2, 4), 3.0)
    stats = []
    hook_fn = _make_clamp_hook(sae, feature_index=1, clamp_value=0.0, positions="all",
                                prompt_lengths=None, stats=stats)
    out = hook_fn(resid, hook=None)
    assert torch.equal(out[..., 1], torch.zeros(1, 2))
    assert torch.equal(out[..., 0], torch.full((1, 2), 3.0))


def test_generated_only_masks_the_entire_prefill_call_structurally():
    """docs/positions_semantics.md's documented behavior (preserved
    verbatim, not re-implemented): when every position in a call is below
    prompt_lengths, the call is a structural no-op -- output IS input
    (identity, not merely numerically close), and no encode/decode round
    trip happens at all."""
    sae = _IdentitySAE()
    resid = torch.zeros(1, 5, 4)
    stats = []
    hook_fn = _make_clamp_hook(sae, feature_index=0, clamp_value=99.0, positions="generated_only",
                                prompt_lengths=5, stats=stats)
    out = hook_fn(resid, hook=None)
    assert out is resid  # identity, not a copy -- the masked-off early return
    assert stats[0].delta_norm == 0.0


def test_generated_only_steers_positions_at_or_past_prompt_length():
    sae = _IdentitySAE()
    resid = torch.zeros(1, 1, 4)  # single decode-step call, one position
    stats = []
    hook_fn = _make_clamp_hook(sae, feature_index=0, clamp_value=7.0, positions="generated_only",
                                prompt_lengths=0, stats=stats)  # prompt_lengths=0: this position is already "generated"
    out = hook_fn(resid, hook=None)
    assert out[0, 0, 0].item() == 7.0
    assert stats[0].delta_norm > 0.0


def test_wrap_hook_with_diagnostics_does_not_alter_the_inner_computation():
    sae = _IdentitySAE()
    resid = torch.zeros(1, 2, 4)
    stats = []
    inner = _make_clamp_hook(sae, feature_index=1, clamp_value=4.0, positions="all",
                              prompt_lengths=None, stats=stats)
    direct_output = inner(resid, hook=None)

    trace = []
    wrapped = wrap_hook_with_diagnostics(
        _make_clamp_hook(sae, feature_index=1, clamp_value=4.0, positions="all", prompt_lengths=None, stats=[]),
        sae=sae, feature_index=1, mode="steer", dose_or_raw_label="raw engineering value",
        calibration_input=None, resolved_absolute_target=4.0, hook_name="test:hook", trace_out=trace,
    )
    wrapped_output = wrapped(resid, hook=None)
    assert torch.equal(direct_output, wrapped_output)
    assert len(trace) == 1
    assert trace[0].call_classification == "prefill"
    assert trace[0].residual_delta_norm > 0.0


def test_mechanical_verdict_reports_prefill_and_decode_counts_from_real_traces():
    sae = _IdentitySAE()
    trace = []
    hook_fn = wrap_hook_with_diagnostics(
        _make_clamp_hook(sae, feature_index=0, clamp_value=2.0, positions="all", prompt_lengths=None, stats=[]),
        sae=sae, feature_index=0, mode="steer", dose_or_raw_label="raw engineering value",
        calibration_input=None, resolved_absolute_target=2.0, hook_name="test:hook", trace_out=trace,
    )
    hook_fn(torch.zeros(1, 3, 4), hook=None)  # prefill
    for _ in range(3):
        hook_fn(torch.zeros(1, 1, 4), hook=None)  # decode steps

    verdict = mechanical_verdict(trace, positions="all")
    assert verdict["hook_invocation_count"] == 4
    assert verdict["prefill_call_count"] == 1
    assert verdict["decode_call_count"] == 3
    assert verdict["nonzero_steer_confirmed"] is True
    assert verdict["first_disappearance_boundary"] is None
