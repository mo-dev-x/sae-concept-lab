"""Mechanically extracted from qwen-sae-interp scripts/legacy/final_pairing_harness.py
at checkout e63b08e (the commit named in the pending Qwen mechanical-acceptance
evidence -- see provenance/source_import.json and
sae_concept_lab/core/runtime_acceptance.py; this extraction is CODE ONLY and
carries no acceptance claim of its own). QWEN-ONLY: the raw-HF loading path
for Qwen3.5-27B + Qwen-Scope (no HookedTransformer -- transformer_lens has no
registry entry for this model). Every function/class body below is copied
verbatim from the source commit; only the import wiring at the top of this
file has been mechanically adapted (see mechanical_adaptations in the
provenance record) -- the source script used a sys.path-manipulated bare
`import final_pairing_targets as targets`, which this package replaces with
`from . import targets`, a plain, standard package-relative import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import targets
from .diagnostics import _require_offline


def _topk_relu(x, k: int):
    import torch

    relu_x = torch.relu(x)
    values, indices = torch.topk(relu_x, k, dim=-1)
    out = torch.zeros_like(relu_x)
    out.scatter_(-1, indices, values)
    return out


class QwenScopeSAE:
    """W_enc/b_enc/W_dec/b_dec are all REQUIRED, per the official
    Qwen-Scope release's own checkpoint contract (orchestrator review,
    2026-08-11: "the checkpoint contract lists b_dec as present"). A layer
    file missing any of the four keys fails closed at construction rather
    than silently substituting a zero bias -- the earlier "unconfirmed,
    defaults to zero" behavior was based only on the release's own app.py
    steering shortcut never reading b_dec, not on the checkpoint's actual
    contents, and has been superseded now that the real contract is known."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec, *, k: int):
        self.W_enc = W_enc  # [d_model, d_sae]
        self.b_enc = b_enc  # [d_sae]
        self.W_dec = W_dec  # [d_model, d_sae]
        self.b_dec = b_dec  # [d_model]
        self.k = k
        self.d_in = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]

    def encode(self, x):
        pre = x.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        return _topk_relu(pre, self.k)

    def decode(self, feats):
        return feats.to(self.W_dec.dtype) @ self.W_dec.T + self.b_dec

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], *, k: int, device: str, target: targets.TargetPairing
    ) -> QwenScopeSAE:
        import torch

        required = ("W_enc", "b_enc", "W_dec", "b_dec")
        missing = [key for key in required if key not in state_dict]
        if missing:
            raise targets.TargetIdentityMismatch(
                f"Qwen-Scope layer file is missing expected key(s) {missing} -- the loaded "
                f"file does not match the release's own checkpoint contract "
                f"(W_enc/b_enc/W_dec/b_dec); refusing to guess a substitute."
            )
        targets.validate_qwen_k(k, target)
        targets.validate_qwen_sae_shapes(
            w_enc_shape=tuple(state_dict["W_enc"].shape),
            b_enc_shape=tuple(state_dict["b_enc"].shape),
            w_dec_shape=tuple(state_dict["W_dec"].shape),
            b_dec_shape=tuple(state_dict["b_dec"].shape),
            target=target,
        )
        w_enc_raw = state_dict["W_enc"].to(dtype=torch.float32, device=device)  # [d_sae, d_model]
        b_enc = state_dict["b_enc"].to(dtype=torch.float32, device=device)
        w_dec = state_dict["W_dec"].to(dtype=torch.float32, device=device)  # [d_model, d_sae]
        b_dec = state_dict["b_dec"].to(dtype=torch.float32, device=device)
        return cls(W_enc=w_enc_raw.T.contiguous(), b_enc=b_enc, W_dec=w_dec, b_dec=b_dec, k=k)

    @classmethod
    def from_layer_file(
        cls, path: str | Path, *, k: int, device: str, target: targets.TargetPairing
    ) -> QwenScopeSAE:
        import torch

        state_dict = torch.load(str(path), map_location=device, weights_only=True)
        return cls.from_state_dict(state_dict, k=k, device=device, target=target)


def resolve_qwen_text_decoder(hf_model):
    """Qwen3_5ForCausalLM's decoder stack: verified directly against
    modeling_qwen3_5.py source, both the public source for transformers
    v5.14.1 (Tamia's actual installed version) and independently against
    this machine's own installed transformers==5.12.1 (not inferred by
    analogy from either alone) -- Qwen3_5ForCausalLM.__init__ sets
    self.model = Qwen3_5TextModel(config); Qwen3_5TextModel.__init__ sets
    self.layers = nn.ModuleList(...) directly, with no vision tower or
    .language_model indirection (that nesting is specific to the
    multimodal Qwen3_5ForConditionalGeneration class this harness no
    longer loads). Mirrors the official Qwen-Scope release's own
    application, which hooks model.model.layers[layer] the same way."""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model
    raise targets.TargetIdentityMismatch(
        f"could not locate a .model.layers decoder stack on the loaded model "
        f"(type={type(hf_model).__name__}); the modeling_qwen3_5.py structure this was "
        f"verified against may not match what actually loaded on this machine -- stop "
        f"rather than guess a different attribute path."
    )


def get_qwen_decoder_layer(text_decoder, layer: int):
    return text_decoder.layers[layer]


def register_qwen_raw_hook(decoder_layer_module, hook_fn):
    """Qwen3_5DecoderLayer.forward() returns hidden_states as a plain
    tensor, not a tuple (verified against modeling_qwen3_5.py's own
    `return hidden_states`, both the public v5.14.1 source and this
    machine's installed transformers==5.12.1) -- register_forward_hook's
    `output` argument is therefore directly the resid-post tensor, no
    unwrapping needed, and returning a replacement tensor from the hook
    replaces the layer's output exactly as _make_clamp_hook expects.

    Orchestrator review, 2026-08-11: validates this assumption at runtime
    rather than trusting it silently -- if Tamia's live implementation ever
    returns something other than a plain tensor (e.g. a tuple, as some
    other decoder-layer conventions do), this fails clearly instead of
    passing a tuple into _make_clamp_hook and failing with a confusing
    downstream tensor-op error."""
    import torch

    def native_hook(module, args, output):
        if not isinstance(output, torch.Tensor):
            raise targets.TargetIdentityMismatch(
                f"expected the hooked Qwen decoder layer to return a plain tensor (verified "
                f"against modeling_qwen3_5.py source), but got {type(output).__name__} instead "
                f"-- Tamia's live implementation differs from what this harness was verified "
                f"against; stop rather than guess how to unwrap it."
            )
        return hook_fn(output, hook=None)

    return decoder_layer_module.register_forward_hook(native_hook)


def load_qwen_target(
    model_path: str | Path, sae_layer_file_path: str | Path, *, layer: int, k: int | None = None,
    device: str = "cuda", dtype: str = "bfloat16",
    expected_model_revision: str | None = None, expected_sae_revision: str | None = None,
):
    """UNTESTED against real weights. Raw-HF path (no HookedTransformer --
    transformer_lens==3.2.1 does not know Qwen3.5, verified in
    final_pairing_targets.py). Loads via AutoModelForCausalLM, matching
    Tamia's actual transformers==5.14.1 dispatch and the official
    Qwen-Scope release's own application (see final_pairing_targets.py's
    module docstring for the full verification trail, including
    independent re-confirmation against this machine's installed
    transformers==5.12.1). k defaults to the ratified target's expected_k
    (50); an explicit override that disagrees with it raises (see
    final_pairing_targets.validate_qwen_k). Returns (hf_model, text_decoder,
    sae, hook_identifier, provenance)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    target = targets.QWEN_3_5_27B_TARGET
    k = target.expected_k if k is None else k
    targets.validate_qwen_layer_choice(layer, target)
    _require_offline()

    model_path = Path(model_path)
    sae_layer_file_path = Path(sae_layer_file_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model snapshot directory not found: {model_path}")
    if not sae_layer_file_path.exists():
        raise FileNotFoundError(f"Qwen-Scope layer file not found: {sae_layer_file_path}")
    targets.validate_qwen_layer_filename(sae_layer_file_path, layer)
    model_identity = targets.validate_local_snapshot_identity(
        model_path, target, which="model", expected_revision=expected_model_revision
    )
    sae_identity = targets.validate_local_snapshot_identity(
        sae_layer_file_path.parent, target, which="sae", expected_revision=expected_sae_revision
    )

    torch_dtype = getattr(torch, dtype)
    # AutoModelForCausalLM dispatches "qwen3_5" to Qwen3_5ForCausalLM on
    # Tamia's actual transformers==5.14.1 (MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
    # -- the same Auto class and model.model.layers[layer] hook path the
    # official Qwen-Scope release's own application uses. AutoModel dispatches
    # to Qwen3_5Model (no .generate()) and AutoModelForImageTextToText
    # dispatches to the multimodal Qwen3_5ForConditionalGeneration -- neither
    # is used here; validate_runtime_class below fails closed if a
    # differently-pinned transformers version ever dispatches this
    # differently than verified.
    hf_model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch_dtype)
    targets.validate_runtime_class(type(hf_model).__name__, target)
    targets.validate_has_callable_generate(hf_model, label="loaded Qwen model")
    hf_model.eval()
    hf_model.to(device)

    text_decoder = resolve_qwen_text_decoder(hf_model)
    hidden_size = text_decoder.config.hidden_size

    sae = QwenScopeSAE.from_layer_file(sae_layer_file_path, k=k, device=device, target=target)
    targets.validate_hidden_dims(hidden_size, sae.d_in, target)
    targets.validate_qwen_sae_shapes(
        w_enc_shape=(sae.d_sae, sae.d_in), b_enc_shape=(sae.d_sae,),
        w_dec_shape=(sae.d_in, sae.d_sae), b_dec_shape=(sae.d_in,), target=target,
    )

    hook_identifier = f"{target.expected_hook_name}:layer_{layer}"
    targets.validate_hook_identity(hook_identifier, target)
    decoder_layer_module = get_qwen_decoder_layer(text_decoder, layer)

    provenance = {
        "target": target.name,
        "model": {
            "repository": target.model_repo_id,
            "local_path": str(model_path),
            "revision": model_identity["revision"],
            "revision_verification": model_identity["verification"],
            "actual_class": type(hf_model).__name__,
            "transformers_version": transformers.__version__,
            "selected_auto_class": "AutoModelForCausalLM",
            "decoder_attribute_path": "model.layers",
        },
        "sae": {
            "repository": target.sae_repo_id,
            "release": None,
            "sae_id": None,
            "local_path": str(sae_layer_file_path),
            "revision": sae_identity["revision"],
            "revision_verification": sae_identity["verification"],
            "resolved_files": [str(sae_layer_file_path)],
            "actual_class": type(sae).__name__,
            "format": target.sae_format,
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
            "k": sae.k,
        },
        "layer": {
            "engineering_layer": layer,
            "engineering_only": True,
            "hook_name": hook_identifier,
            "hooked_module_class": type(decoder_layer_module).__name__,
        },
    }
    return hf_model, text_decoder, sae, hook_identifier, provenance
