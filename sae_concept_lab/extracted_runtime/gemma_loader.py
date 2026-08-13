"""Mechanically extracted from qwen-sae-interp scripts/legacy/final_pairing_harness.py
at checkout de3b499 (current tip of final-pairing-harness as of this
extraction -- see provenance/source_import.json; CODE ONLY, no acceptance
claim -- see sae_concept_lab/core/runtime_acceptance.py). GEMMA-ONLY: the
transformer_lens/sae_lens loading path for gemma-3-12b-it +
gemma-scope-2-12b-it-res, including the local-snapshot-only SAE resolver
(never touches the network/Hub once a local SAE snapshot path is supplied)
and the corrected safetensors shape-lookup shim. Every function body below
is copied verbatim from the source commit; only the import wiring at the
top of this file has been mechanically adapted, identically to
qwen_loader.py (see mechanical_adaptations in the provenance record)."""

from __future__ import annotations

from pathlib import Path

from . import targets
from .diagnostics import _require_offline


def _capture_sae_download_paths(
    capture: list[str],
    *,
    sae_path: str | Path,
    target: targets.TargetPairing,
    requested_files_out: list[dict[str, str]] | None = None,
):
    """Replaces sae_lens.loading.pretrained_sae_loaders's OWN
    hf_hub_download reference (that module does `from huggingface_hub
    import hf_hub_download` at ITS OWN import time -- patching
    huggingface_hub.hf_hub_download itself would not reach calls made from
    inside pretrained_sae_loaders.py, since Python's `from X import Y`
    binds a local name, not a live reference to the module attribute) with
    targets.resolve_local_gemma_sae_path -- see that function's docstring
    for the full local-snapshot-only resolution algorithm.

    Orchestrator review, 2026-08-14 (live job 406259): a prior version of
    this function called THROUGH to the real hf_hub_download and only
    recorded whatever local path it returned -- proving what the
    subdirectory/symlink guards below check, but never preventing the
    real Hub-ref lookup that call performs from being attempted (and
    failing offline against this project's commit-pinned cache) in the
    first place. This version never calls the real hf_hub_download at
    all: every request is resolved directly against sae_path.

    requested_files_out, if given, is appended with one
    {"requested_filename", "resolved_local_path"} dict per request -- the
    provenance record of exactly what was asked for and what was served,
    independent of resolved_files/capture (which existing snapshot/
    subdirectory/symlink validators consume as a plain list[str]).

    Caller MUST restore the returned original via _restore_sae_download_paths
    in a finally block, whether or not the load succeeded."""
    import sae_lens.loading.pretrained_sae_loaders as psl

    original = psl.hf_hub_download

    def local_only(*args, **kwargs):
        call = dict(zip(("repo_id", "filename"), args, strict=False))
        call.update(kwargs)
        local_path, relative = targets.resolve_local_gemma_sae_path(
            repo_id=call["repo_id"],
            filename=call["filename"],
            subfolder=call.get("subfolder"),
            revision=call.get("revision"),
            sae_snapshot_root=sae_path,
            target=target,
        )
        capture.append(local_path)
        if requested_files_out is not None:
            requested_files_out.append({"requested_filename": relative, "resolved_local_path": local_path})
        return local_path

    psl.hf_hub_download = local_only
    return original


def _restore_sae_download_paths(original) -> None:
    import sae_lens.loading.pretrained_sae_loaders as psl

    psl.hf_hub_download = original


def _patch_gemma3_safetensors_shape_lookup() -> None:
    """Based on gemma3_sweep.py's own patch (same duplicate-rather-than-
    cross-import convention this project already uses for out-of-chain
    adapters, e.g. qwen_tool_adapter.pick_control_feature_idx) -- NOT an
    edit to gemma3_sweep.py. Installed sae_lens's Gemma-3 loader issues a
    raw requests.get() HTTP range read for tensor shapes that bypasses
    huggingface_hub AND HF_HUB_OFFLINE entirely; this routes the same
    shape lookup through hf_hub_download instead. Applies to ANY
    conversion_func="gemma_3" release, verified via the locally-installed
    sae_lens==6.44.2 registry to include gemma-scope-2-12b-it-res, not just
    the -pt release this patch was first written against.

    Orchestrator review, 2026-08-14 (live job 406259): unlike gemma3_sweep.
    py's version, this one calls psl.hf_hub_download -- module-qualified,
    so it is routed through WHATEVER psl.hf_hub_download currently is --
    rather than a separately, freshly-imported hf_hub_download. During
    load_gemma_it_target's _capture_sae_download_paths try/finally window,
    that reference is this module's own local-snapshot-only resolver, so
    the shape lookup resolves from the pinned snapshot exactly like every
    other SAE file request; a fresh, independent import would instead hit
    the real hf_hub_download and the exact same offline-refs/main failure
    this review exists to close, one function away from where it looks
    like it was already fixed.

    Orchestrator review, 2026-08-15 (live job 406826): the resolver above
    reached the locally resolved params file correctly, then failed
    deterministically inside THIS function -- `for k in f` (the version
    written 2026-08-14) treats the real safe_open object as directly
    iterable. It is not: `safe_open` is a Rust extension type
    (`builtins.safe_open`, verified against the installed safetensors==
    0.4.5) with no `__iter__` at all, so `for k in f` raises `TypeError:
    'builtins.safe_open' object is not iterable` on every real call, every
    time -- a defect no mocked test caught, because every existing test of
    this shim mocked safe_open itself rather than exercising the real
    installed API. Both of this project's OWN pre-existing, already-
    accepted copies of this exact patch (gemma3_sweep.py, gemma3_
    necessity.py -- Engineer 2 owned, neither touched here) already use
    `for k in f.keys()`, the correct call; this copy alone had drifted
    from that pattern when it was written. Fixed by using f.keys() (not
    bare iteration) exactly like those two existing copies, and wrapping
    every failure mode with the resolved local_path so a failure message
    always names which file was involved -- some real exceptions here
    (safetensors_rust.SafetensorError for a malformed file) do not include
    the path on their own."""
    import sae_lens.loading.pretrained_sae_loaders as psl
    from safetensors import safe_open

    def _local_get_safetensors_tensor_shapes(repo_id: str, filename: str) -> dict:
        local_path = psl.hf_hub_download(repo_id=repo_id, filename=filename)
        try:
            with safe_open(local_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if not keys:
                    raise targets.TargetIdentityMismatch(
                        f"safetensors file at {local_path!r} has zero tensor keys -- cannot "
                        f"derive any SAE dimensions from an empty file."
                    )
                shapes = {key: list(f.get_slice(key).get_shape()) for key in keys}
        except FileNotFoundError as e:
            raise targets.TargetIdentityMismatch(
                f"resolved safetensors file does not exist at {local_path!r}: {e}"
            ) from e
        except targets.TargetIdentityMismatch:
            raise
        except Exception as e:
            raise targets.TargetIdentityMismatch(
                f"could not open or read tensor shapes from the safetensors file at "
                f"{local_path!r}: {e}"
            ) from e
        if not shapes:
            raise targets.TargetIdentityMismatch(
                f"safetensors file at {local_path!r} produced an empty tensor-shape mapping."
            )
        return shapes

    psl.get_safetensors_tensor_shapes = _local_get_safetensors_tensor_shapes


def load_gemma_it_target(
    model_path: str | Path, sae_path: str | Path, *, device: str = "cuda", dtype: str = "bfloat16",
    expected_model_revision: str | None = None, expected_sae_revision: str | None = None,
):
    """UNTESTED against real weights (no GPU allocation was available for
    this investigation) -- see the report for exactly what is and isn't
    verified. Fails closed via final_pairing_targets' validators at every
    step that can be checked mechanically. Returns (model, sae, hook_name,
    provenance) -- provenance is the full machine-readable identity record
    for the JSON trace."""
    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    from transformers import AutoModel, AutoTokenizer

    target = targets.GEMMA_3_12B_IT_TARGET
    _require_offline()
    model_path = Path(model_path)
    sae_path = Path(sae_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model snapshot directory not found: {model_path}")
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE snapshot directory not found: {sae_path}")
    model_identity = targets.validate_local_snapshot_identity(
        model_path, target, which="model", expected_revision=expected_model_revision
    )
    sae_identity = targets.validate_local_snapshot_identity(
        sae_path, target, which="sae", expected_revision=expected_sae_revision
    )

    # Orchestrator review, 2026-08-13 (live job 406092): verify the flat
    # sae_lens loader id is actually registered BEFORE loading the ~24GB
    # model at all -- a pure registry lookup, no weights/network needed --
    # rather than discovering it's wrong only after that load succeeds.
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    available_loader_ids = list(get_pretrained_saes_directory()[target.sae_release].saes_map.keys())
    targets.validate_sae_loader_id_registered(target.sae_loader_id, available_loader_ids, target)

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    hf_model = AutoModel.from_pretrained(str(model_path), dtype=torch_dtype)
    model = HookedTransformer.from_pretrained(
        target.model_repo_id,
        hf_model=hf_model,
        tokenizer=tokenizer,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
        dtype=torch_dtype,
    )
    model.eval()

    _patch_gemma3_safetensors_shape_lookup()
    resolved_sae_files: list[str] = []
    requested_sae_files: list[dict[str, str]] = []
    original_hf_hub_download = _capture_sae_download_paths(
        resolved_sae_files, sae_path=sae_path, target=target, requested_files_out=requested_sae_files
    )
    try:
        sae = SAE.from_pretrained(release=target.sae_release, sae_id=target.sae_loader_id, device=device)
    finally:
        _restore_sae_download_paths(original_hf_hub_download)
    targets.validate_sae_files_match_snapshot(resolved_sae_files, sae_path, target)
    subdirectory_identity = targets.validate_sae_files_match_expected_subdirectory(
        resolved_sae_files, sae_path, target
    )
    targets.validate_sae_symlink_targets_stay_in_repository_cache(resolved_sae_files, sae_path, target)

    sae = sae.to(dtype=torch.float32)
    sae.eval()

    hook_name = sae.cfg.metadata.hook_name
    targets.validate_hook_identity(hook_name, target)
    targets.validate_hidden_dims(model.cfg.d_model, sae.cfg.d_in, target)

    provenance = {
        "target": target.name,
        "model": {
            "repository": target.model_repo_id,
            "local_path": str(model_path),
            "revision": model_identity["revision"],
            "revision_verification": model_identity["verification"],
            "actual_class": type(model).__name__,
        },
        "sae": {
            "repository": target.sae_repo_id,
            "release": target.sae_release,
            "sae_id": target.sae_id,
            "loader_sae_id": target.sae_loader_id,
            "local_path": str(sae_path),
            "revision": sae_identity["revision"],
            "revision_verification": sae_identity["verification"],
            "resolved_files": resolved_sae_files,
            "requested_sae_files": requested_sae_files,
            "local_snapshot_only": True,
            "network_resolution_attempted": False,
            "actual_class": type(sae).__name__,
            "format": target.sae_format,
            "d_in": sae.cfg.d_in,
            "d_sae": sae.cfg.d_sae,
            "k": None,
            "used_zero_b_dec_default": None,
            "expected_sae_subdirectory": subdirectory_identity["expected_sae_subdirectory"],
            "sae_subdirectory_membership_verified": subdirectory_identity["sae_subdirectory_membership_verified"],
        },
        "layer": {
            "engineering_layer": target.expected_layer,
            "hook_name": hook_name,
            "hooked_module_class": "transformer_lens HookPoint (attached by hook_name via model.hooks())",
        },
    }
    return model, sae, hook_name, provenance
