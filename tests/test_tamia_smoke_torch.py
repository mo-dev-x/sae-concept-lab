"""Torch-enabled test path for the Tamia smoke packet.

tests/test_tamia_smoke.py proves this packet's orchestration and
aggregation logic; it deliberately cannot prove real ALL/GENERATED_ONLY
prefill-masking numerics, because tests._fake_runtime's shared hook fake
is explicitly NOT masking-aware (see that module's docstring). This file
is where that real proof lives, in two tiers:

1. Below `pytest.importorskip("torch")`: runs whenever torch happens to be
   installed, with NO real model/SAE weights required. Only
   load_qwen_target/load_gemma_it_target are stubbed (the "read weights
   off disk" step) -- every other extracted function this packet's own
   run_qwen_position_scenario/run_gemma_position_scenario call
   (get_qwen_decoder_layer, register_qwen_raw_hook, _make_clamp_hook,
   wrap_hook_with_diagnostics, mechanical_verdict) runs FOR REAL, against
   a tiny synthetic torch.nn.Module decoder layer and an identity SAE
   (same fixture idea as tests/test_runtime_hooks_differential.py). This
   is the definitive, non-faked proof that ALL modifies prefill and
   GENERATED_ONLY leaves it unchanged, for THIS packet's own scenario
   functions -- not merely for the extracted hook in isolation.

2. `test_build_smoke_packet_against_real_staged_tamia_snapshots`: gated on
   four SAE_CONCEPT_LAB_TAMIA_*_PATH environment variables pointing at
   genuinely staged snapshots. Skipped everywhere this repository is
   developed and tested away from Tamia -- there is no synthetic stand-in
   for a real 27B/12B checkpoint. See docs/tamia_smoke.md for the exact
   Tamia submission command that sets these and runs this for real.

`pytest.importorskip("torch")` makes tier 1 skip cleanly wherever torch is
not installed (this product's base install has no hard dependency on it
-- see extracted_runtime/__init__.py); the rest of this suite passes
either way.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from sae_concept_lab.canonical.concept_bundle import PositionMode  # noqa: E402
from sae_concept_lab.core.gemma_backend import GemmaRuntimeBackend  # noqa: E402
from sae_concept_lab.core.qwen_backend import QwenRuntimeBackend  # noqa: E402
from sae_concept_lab.smoke import entries, tamia_smoke  # noqa: E402


class _IdentitySAE:
    """encode(x) == x, decode(feats) == feats -- see
    tests/test_runtime_hooks_differential.py's identical fixture: this
    makes the real _make_clamp_hook math trivially predictable without a
    real trained SAE."""

    def encode(self, x):
        return x.clone()

    def decode(self, feats):
        return feats.clone()


class _RealQwenDecoderLayer(torch.nn.Module):
    """A genuine torch.nn.Module standing in for one Qwen3_5DecoderLayer:
    register_qwen_raw_hook (real, unfaked) requires an actual
    register_forward_hook-capable module whose forward() returns a plain
    tensor, matching modeling_qwen3_5.py's contract (qwen_loader.py's own
    docstring)."""

    def forward(self, resid):
        return resid


class _RealQwenTextDecoder:
    def __init__(self, num_layers: int = 1):
        self.layers = [_RealQwenDecoderLayer() for _ in range(num_layers)]


class _RealFakeInputs(dict):
    def to(self, device):
        return self


class _RealTensorTokenizer:
    def __call__(self, prompt, return_tensors="pt"):
        ids = [ord(c) % 50 for c in prompt] or [0]
        return _RealFakeInputs(input_ids=torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=True):
        return f"generated:{list(ids.tolist())}"


class _RealQwenHfModel:
    """.generate() drives the REAL registered forward hook exactly as an
    actual prefill-then-per-token decode loop would: one call with the
    full prompt length, then max_new_tokens-1 single-position calls,
    through the SAME hook_fn closure each time -- so _PositionCounter's
    real, stateful position tracking (hooks.py) advances exactly as it
    would in production."""

    def __init__(self, text_decoder: _RealQwenTextDecoder, *, d_model: int):
        self._text_decoder = text_decoder
        self._d_model = d_model

    def generate(self, *, input_ids, max_new_tokens=8, do_sample=False):
        prompt_len = input_ids.shape[1]
        layer = self._text_decoder.layers[0]
        layer(torch.zeros(1, prompt_len, self._d_model))
        for _ in range(max(max_new_tokens - 1, 0)):
            layer(torch.zeros(1, 1, self._d_model))
        new_ids = torch.tensor([list(range(900, 900 + max_new_tokens))])
        return torch.cat([input_ids, new_ids], dim=1)


class _RealGemmaModel:
    def __init__(self, *, d_model: int):
        self.tokenizer = _RealTensorTokenizer()
        self._active_hook = None
        self._d_model = d_model

    def to_tokens(self, prompt):
        return self.tokenizer(prompt)["input_ids"]

    def hooks(self, fwd_hooks):
        model = self

        class _Ctx:
            def __enter__(_self):
                model._active_hook = fwd_hooks[0][1]
                return model

            def __exit__(_self, *exc_info):
                model._active_hook = None
                return False

        return _Ctx()

    def generate(self, tokens, max_new_tokens=8, do_sample=False, verbose=False):
        prompt_len = tokens.shape[1]
        if self._active_hook is not None:
            self._active_hook(torch.zeros(1, prompt_len, self._d_model), None)
            for _ in range(max(max_new_tokens - 1, 0)):
                self._active_hook(torch.zeros(1, 1, self._d_model), None)
        new_ids = torch.tensor([list(range(900, 900 + max_new_tokens))])
        return torch.cat([tokens, new_ids], dim=1)


#: Large enough that the accepted engineering feature index safely
#: indexes the last dimension of a tiny synthetic (batch, seq, d_model)
#: zero tensor -- no real hidden-dim value is claimed or needed here.
_QWEN_D_MODEL = entries.QWEN_SMOKE_FEATURE_IDX + 8
_GEMMA_D_MODEL = entries.GEMMA_SMOKE_FEATURE_IDX + 8


def _install_fake_transformers_with_real_tensors(monkeypatch):
    """qwen_backend.py itself does `from transformers import
    AutoTokenizer` -- faked here (transformers is not installed in this
    dev venv even when torch is) so AutoTokenizer.from_pretrained returns
    a REAL-tensor-backed tokenizer stand-in, keeping every downstream
    tensor operation genuine torch, not FakeTensor."""
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: _RealTensorTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _install_real_qwen_loader(monkeypatch):
    text_decoder = _RealQwenTextDecoder(num_layers=1)
    hf_model = _RealQwenHfModel(text_decoder, d_model=_QWEN_D_MODEL)
    provenance = {
        "target": "qwen-3.5-27b",
        "model": {"repository": "Qwen/Qwen3.5-27B", "actual_class": "Qwen3_5ForCausalLM"},
        "sae": {
            "repository": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50", "d_in": _QWEN_D_MODEL, "d_sae": _QWEN_D_MODEL, "k": 50,
        },
        "layer": {
            "engineering_layer": entries.QWEN_SMOKE_LAYER, "engineering_only": True,
            "hook_name": f"resid_post:layer_{entries.QWEN_SMOKE_LAYER}",
        },
    }

    def fake_load_qwen_target(
        model_path, sae_layer_file_path, *, layer, k=None, device="cuda", dtype="bfloat16",
        expected_model_revision=None, expected_sae_revision=None,
    ):
        return hf_model, text_decoder, _IdentitySAE(), f"resid_post:layer_{layer}", provenance

    import sae_concept_lab.extracted_runtime.qwen_loader as qwen_loader_module

    monkeypatch.setattr(qwen_loader_module, "load_qwen_target", fake_load_qwen_target)


def _install_real_gemma_loader(monkeypatch):
    model = _RealGemmaModel(d_model=_GEMMA_D_MODEL)
    provenance = {
        "target": "gemma-3-12b-it",
        "model": {"repository": "google/gemma-3-12b-it", "actual_class": "HookedTransformer"},
        "sae": {"repository": "google/gemma-scope-2-12b-it-res", "d_in": _GEMMA_D_MODEL, "d_sae": _GEMMA_D_MODEL},
        "layer": {
            "engineering_layer": entries.GEMMA_SMOKE_LAYER, "hook_name": f"blocks.{entries.GEMMA_SMOKE_LAYER}.hook_resid_post",
        },
    }

    def fake_load_gemma_it_target(
        model_path, sae_path, *, device="cuda", dtype="bfloat16", expected_model_revision=None,
        expected_sae_revision=None,
    ):
        return model, _IdentitySAE(), f"blocks.{entries.GEMMA_SMOKE_LAYER}.hook_resid_post", provenance

    import sae_concept_lab.extracted_runtime.gemma_loader as gemma_loader_module

    monkeypatch.setattr(gemma_loader_module, "load_gemma_it_target", fake_load_gemma_it_target)


# ---------------------------------------------------------------------------
# Tier 1: real torch numerics, no real weights
# ---------------------------------------------------------------------------


def test_qwen_all_scenario_real_masking_shows_nonzero_prefill_delta(monkeypatch):
    _install_fake_transformers_with_real_tensors(monkeypatch)
    _install_real_qwen_loader(monkeypatch)
    backend = QwenRuntimeBackend(model_path="/fake/model", sae_path="/fake/layer0.sae.pt", qwen_layer=entries.QWEN_SMOKE_LAYER)
    result = tamia_smoke.run_qwen_position_scenario(
        backend, PositionMode.ALL, max_new_tokens=4, product_commit="deadbeef", extraction_source_commit="e63b08e",
    )
    assert result.passed, result.as_dict()
    assert result.detail["trace"][0]["residual_delta_norm"] > 0.0


def test_qwen_generated_only_scenario_real_masking_zeroes_prefill_delta(monkeypatch):
    _install_fake_transformers_with_real_tensors(monkeypatch)
    _install_real_qwen_loader(monkeypatch)
    backend = QwenRuntimeBackend(model_path="/fake/model", sae_path="/fake/layer0.sae.pt", qwen_layer=entries.QWEN_SMOKE_LAYER)
    result = tamia_smoke.run_qwen_position_scenario(
        backend, PositionMode.GENERATED_ONLY, max_new_tokens=4, product_commit="deadbeef", extraction_source_commit="e63b08e",
    )
    assert result.passed, result.as_dict()
    assert result.detail["trace"][0]["residual_delta_norm"] == 0.0
    assert result.detail["trace"][-1]["residual_delta_norm"] > 0.0  # a later decode step IS steered


def test_gemma_all_scenario_real_masking_shows_nonzero_prefill_delta(monkeypatch):
    _install_real_gemma_loader(monkeypatch)
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae_root")
    result = tamia_smoke.run_gemma_position_scenario(
        backend, PositionMode.ALL, max_new_tokens=4, product_commit="deadbeef", extraction_source_commit="de3b499",
    )
    assert result.passed, result.as_dict()
    assert result.detail["trace"][0]["residual_delta_norm"] > 0.0


def test_gemma_generated_only_scenario_real_masking_zeroes_prefill_delta(monkeypatch):
    _install_real_gemma_loader(monkeypatch)
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae_root")
    result = tamia_smoke.run_gemma_position_scenario(
        backend, PositionMode.GENERATED_ONLY, max_new_tokens=4, product_commit="deadbeef", extraction_source_commit="de3b499",
    )
    assert result.passed, result.as_dict()
    assert result.detail["trace"][0]["residual_delta_norm"] == 0.0
    assert result.detail["trace"][-1]["residual_delta_norm"] > 0.0


# ---------------------------------------------------------------------------
# Tier 2: real staged Tamia snapshots (skipped everywhere else)
# ---------------------------------------------------------------------------

_TAMIA_ENV_VARS = (
    "SAE_CONCEPT_LAB_TAMIA_QWEN_MODEL_PATH",
    "SAE_CONCEPT_LAB_TAMIA_QWEN_SAE_PATH",
    "SAE_CONCEPT_LAB_TAMIA_GEMMA_MODEL_PATH",
    "SAE_CONCEPT_LAB_TAMIA_GEMMA_SAE_PATH",
)


@pytest.mark.skipif(
    not all(os.environ.get(var) for var in _TAMIA_ENV_VARS),
    reason=(
        "requires real staged Tamia snapshots -- set SAE_CONCEPT_LAB_TAMIA_{QWEN,GEMMA}_{MODEL,SAE}_PATH "
        "to run this against genuine weights (see docs/tamia_smoke.md)"
    ),
)
def test_build_smoke_packet_against_real_staged_tamia_snapshots(tmp_path):
    args = tamia_smoke.parse_args(
        [
            "--qwen-model-path", os.environ["SAE_CONCEPT_LAB_TAMIA_QWEN_MODEL_PATH"],
            "--qwen-sae-path", os.environ["SAE_CONCEPT_LAB_TAMIA_QWEN_SAE_PATH"],
            "--gemma-model-path", os.environ["SAE_CONCEPT_LAB_TAMIA_GEMMA_MODEL_PATH"],
            "--gemma-sae-path", os.environ["SAE_CONCEPT_LAB_TAMIA_GEMMA_SAE_PATH"],
            "--output", str(tmp_path / "tamia_smoke_packet.json"),
        ]
    )
    packet = tamia_smoke.build_smoke_packet(args)
    assert packet.passed, [s.as_dict() for s in packet.scenarios if not s.passed]
