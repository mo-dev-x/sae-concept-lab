"""Ratified final-pairing target identities, and the pure (no torch, no
network) validation functions that check a loaded model/SAE actually match
them. No feature meanings, bundles, weights, calibration, or behavioral
thresholds live here -- engineering identity only.

WHY A NEW MODULE RATHER THAN REUSING gemma3_sweep.load_model_and_sae:
gemma3_sweep.py's load_model_and_sae (frozen, Engineer 2 owned, not edited
here) hardcodes its own module-level MODEL_ID / SAE_RELEASE / SAE_ID
constants inside the function body -- HookedTransformer.from_pretrained(
MODEL_ID, hf_model=hf_model, ...) and SAE.from_pretrained(release=
SAE_RELEASE, sae_id=SAE_ID, ...) both ignore whatever model_path/sae_path
strings are passed in for identity-routing purposes (model_path/sae_path
are only used to load bytes from disk and to existence-check; the identity
that determines WHICH conversion recipe and WHICH registry entry gets used
is the hardcoded constant). Pointing that function's --model-path/--sae-path
at a gemma-3-12b-**it** / gemma-scope-2-12b-**it** snapshot would silently
still resolve identity against the **pt** constants -- the exact "identical
output" failure class this investigation exists to close, one level up.
This module's loaders (final_pairing_harness.py) take repo_id/release/sae_id
as explicit parameters instead, and this module's validators check what
was ACTUALLY loaded against what was ratified.

VERIFIED, not assumed (commands re-runnable against this repo's own pinned
sae_lens==6.44.2 / transformer_lens==3.2.1, both installed locally -- same
pins as the Tamia sprint venv, see project_pi_directive_2026_08.md):

    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    get_pretrained_saes_directory()["gemma-scope-2-12b-it-res"]
        -> repo_id="google/gemma-scope-2-12b-it", model="google/gemma-3-12b-it"
        -> saes_map["layer_31_width_16k_l0_medium"] == "resid_post/layer_31_width_16k_l0_medium"
           the KEY ("layer_31_width_16k_l0_medium", flat -- this is
           sae_loader_id below, the only thing SAE.from_pretrained(sae_id=)
           may be given) maps to the VALUE ("resid_post/layer_31_width_16k_
           l0_medium", the real artifact subdirectory -- this is sae_id
           below, used ONLY for the logical/physical subdirectory guards).
           Orchestrator review, 2026-08-13 (live job 406092): this exact
           fact was already documented here, correctly, before that job --
           and still got misapplied, passing the VALUE where the loader
           wanted the KEY. Read this block literally before touching
           either field again; do not re-derive the relationship from
           memory.

    import transformer_lens.loading_from_pretrained as lfp
    "google/gemma-3-12b-it" in lfp.OFFICIAL_MODEL_NAMES  -> True
    lfp.convert_hf_model_config("google/gemma-3-12b-it")
        -> d_model=3840, n_layers=48  (identical to the already-verified -pt
           figures in docs/pi_directive_plan_2026_08.md -- same architecture
           family, "google/gemma-3-12b" is a startswith() branch covering
           both variants, loading_from_pretrained.py:1199)
    any("qwen3.5" in n.lower() for n in lfp.OFFICIAL_MODEL_NAMES)  -> False
        (newest Qwen family transformer_lens==3.2.1 knows is "Qwen3",
         0.6B/1.7B/4B/8B/14B -- no 27B, and nothing named Qwen3.5 at all)

NOT verifiable from this machine without downloading weights, so recorded
as read-only public-metadata findings (HF's public /api/models endpoint,
config.json, and the release's own app.py source -- no weights fetched):

    Qwen/Qwen3.5-27B: architectures=["Qwen3_5ForConditionalGeneration"],
    model_type="qwen3_5", hidden_size=5120, num_hidden_layers=64,
    pipeline_tag="image-text-to-text" (multimodal, like Gemma3ForConditional
    Generation). Not in transformer_lens's registry (see above) -- the
    HookedTransformer path is not available for this model without either
    an unauthorized version-pin change or a raw-HF-forward-hooks fallback
    (see final_pairing_harness.py's QwenRawHookAdapter and the "unresolved
    ambiguities" list in docs/final_pairing_tamia_packet.md).

    Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50: tag "qwen-scope", model_type=
    "topk_sae", base_model=Qwen/Qwen3.5-27B, d_model=5120, d_sae=81920,
    k=50, hook_point="resid_post", one file per layer (layer0.sae.pt ..
    layer63.sae.pt, num_layers=64) -- NOT a sae_lens registry entry and NOT
    the tracked-directory layout sae_lens.SAE.load_from_pretrained expects;
    a bespoke torch.load()'d dict needing a thin duck-typed wrapper, not
    sae_lens.SAE itself.

Per the ratified target list, the Qwen SIDE's layer is explicitly
engineering-only and NOT pre-registered here: "Qwen layer selected from the
official release or available Tamia snapshots" means whichever of the 64
per-layer files is actually staged locally, supplied by the caller -- this
module does not default or guess one.

Orchestrator review, 2026-08-11 ("Align Qwen harness with official release
and Tamia runtime"): Tamia's actual installed transformers is 5.14.1, not
the 5.12.1 available on this machine, and reports that it dispatches
model_type="qwen3_5" through AutoModelForCausalLM to Qwen3_5ForCausalLM --
the same route the official Qwen-Scope release's own application uses
(hooking model.model.layers[layer] directly, layer 0 in its own example),
not the AutoModelForImageTextToText/Qwen3_5ForConditionalGeneration
multimodal route this harness used previously. VERIFIED, not merely taken
on the orchestrator's word, two ways:

    1. Read from the public transformers GitHub source at tag v5.14.1
       (the exact version Tamia reports) -- src/transformers/models/auto/
       modeling_auto.py's MODEL_FOR_CAUSAL_LM_MAPPING_NAMES contains
       ("qwen3_5", "Qwen3_5ForCausalLM"); Qwen3_5ForCausalLM.model is a
       Qwen3_5TextModel, whose .layers is its own nn.ModuleList directly
       (no .language_model indirection -- that nesting is specific to the
       multimodal Qwen3_5ForConditionalGeneration class this harness no
       longer loads); Qwen3_5DecoderLayer.forward() still returns a plain
       tensor, not a tuple.
    2. Independently re-confirmed against the ACTUAL locally-installed
       transformers==5.12.1 (not just the public source for 5.14.1): this
       older pinned version ALREADY has MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
       ["qwen3_5"] == "Qwen3_5ForCausalLM", and Qwen3_5ForCausalLM.model is
       already a Qwen3_5TextModel with .layers as its own nn.ModuleList
       (module inspection: `inspect.getsource(Qwen3_5ForCausalLM.__init__)`
       / `Qwen3_5TextModel.__init__`), and Qwen3_5DecoderLayer.forward()
       still ends in `return hidden_states` (a plain tensor). This route
       was available in the version already on this machine; the previous
       AutoModelForImageTextToText choice was not required by any version
       constraint, it was simply the wrong Auto class for this task.

Orchestrator review, 2026-08-14 ("Make Gemma SAE loading use the exact
pinned local snapshot", live job 406259): the Gemma model loaded, and the
flat sae_loader_id validated as registered, but SAE.from_pretrained then
failed before any weights loaded. VERIFIED directly against the installed
huggingface_hub==1.24.0 source (src/huggingface_hub/file_download.py):

    hf_hub_download(repo_id, filename, subfolder=None, revision=None, ...)
    -> if subfolder: filename = f"{subfolder}/{filename}"          (join)
    -> if revision is None: revision = constants.DEFAULT_REVISION  ("main")
    -> resolution then needs refs/main -> commit hash, from the LOCAL
       cache when offline

None of sae_lens's own gemma_3-loader hf_hub_download call sites (config.json
fetch, params.safetensors fetch, the raw-HTTP-bypassing shape-lookup patch
this module's harness already installs) pass revision= at all, so every one
of them defaults to "main" -- and this project's cache was populated by
pinning snapshot_download directly to the immutable commit sha
(4c419f1ba0be8b7754d4151d4f26c23b92a9029e for the validated Gemma Scope IT
snapshot), never to the branch name "main", so no local refs/main file
exists to resolve that default against. Offline resolution therefore fails
before ANY of the subdirectory/symlink guards above even run, even though
the exact files those guards would check are sitting on disk the whole
time. resolve_local_gemma_sae_path below (installed by
final_pairing_harness._capture_sae_download_paths as a full replacement for
sae_lens.loading.pretrained_sae_loaders's own hf_hub_download reference, not
a pass-through wrapper around it) resolves every request directly against
the validated snapshot instead -- no Hub ref lookup, no network, no cache
mutation, by construction.

Taken from the work order's stated findings, NOT independently re-verified
this round (a guessed URL for the official Qwen-Scope application's own
source 404'd, and no further URL was guessed per this project's policy
against fabricating URLs): b_dec's presence in the real checkpoint ("the
checkpoint contract lists b_dec as present"), the ReLU-then-TopK(50)
activation order, and layer 0 as the release's own documented example.
QwenScopeSAE now requires and fails closed on a missing b_dec rather than
defaulting to zero -- see that class's docstring in final_pairing_harness.py.

Orchestrator review, 2026-08-16 ("Correct and comprehensively audit Gemma
path-containment guards", live job 406957): a full audit of every
Path.resolve()/os.path.realpath/startswith/is_relative_to occurrence in
final_pairing_targets.py, final_pairing_harness.py, final_pairing_gpu_job.py,
and interplab/interventions/hooks.py found exactly ONE defect:
validate_sae_files_match_snapshot (the OLDEST of the three SAE-file
validators, 2026-08-10) still called Path.resolve() (follows symlinks --
wrong for a LOGICAL identity check, see validate_sae_files_match_expected_
subdirectory's docstring for why) and used str.startswith() (an unsafe
sibling-prefix-permissive containment comparison, e.g. "snapshots/<rev>-
evil".startswith("snapshots/<rev>") is True) instead of Path.is_relative_to.
Fixed to match its two newer siblings exactly. The resulting, now-uniform
invariant across all three SAE-file/path checks in this module:

    LOGICAL identity/containment (validate_sae_files_match_snapshot,
    validate_sae_files_match_expected_subdirectory, and resolve_local_
    gemma_sae_path's own subdirectory check): os.path.abspath (or, for
    resolve_local_gemma_sae_path, no filesystem access at all) plus
    Path.is_relative_to / PurePosixPath.is_relative_to. NEVER Path.resolve()
    or os.path.realpath -- a real HF snapshot entry is normally a symlink
    into a SIBLING blobs/ store, and resolving it defeats the entire
    purpose of a logical check.

    PHYSICAL symlink-target containment (validate_sae_symlink_targets_
    stay_in_repository_cache, the ONLY place in this module that
    dereferences anything): os.path.realpath plus Path.is_relative_to.
    This is the one and only function permitted to call os.path.realpath
    on a resolved SAE file.

Every other Path.resolve() occurrence found in the audited files
(final_pairing_harness.py, final_pairing_gpu_job.py, and this module's own
docstrings/comments) resolves a SCRIPT'S OWN __file__ location to build
REPO_ROOT/SCRIPT_DIR or a sibling-module path, or is display-only prose in
a frozen, Engineer-2-owned file (gemma3_sweep.py/gemma3_necessity.py) with
no containment comparison paired with it -- none of these compare a
resolved path against a validated boundary, so none of them are in scope
for this audit; see docs/final_pairing_tamia_packet.md for the full,
per-occurrence audit table.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal


@dataclass(frozen=True)
class TargetPairing:
    name: str
    model_repo_id: str
    # model_architecture is the checkpoint's OWN config.json-declared
    # "architectures" value (a fact about the checkpoint's metadata) --
    # distinct from expected_runtime_class below, which is the class this
    # harness's CHOSEN AutoModelFor... loader must actually produce.
    # AutoModelFor* dispatch is driven by model_type + that Auto class's
    # own registered mapping, not by config.json's architectures hint, so
    # the two can legitimately name different classes.
    model_architecture: str
    model_supported_by_transformer_lens: bool
    sae_repo_id: str
    sae_format: Literal["sae_lens_registry", "qwen_scope_raw_pt"]
    expected_hidden_dim: int
    # sae_lens_registry fields (Gemma-it) -- None for qwen_scope_raw_pt.
    # sae_id is the SCIENTIFIC/ARTIFACT identity (the real subdirectory
    # this SAE's files actually live under, e.g. "resid_post/layer_31_
    # width_16k_l0_medium") -- used by the logical/physical subdirectory
    # guards below, and NEVER passed to SAE.from_pretrained directly.
    # sae_loader_id is the DISTINCT, flat key the sae_lens registry's own
    # saes_map actually uses for THIS release's SAE.from_pretrained(sae_id=)
    # call (e.g. "layer_31_width_16k_l0_medium", no "resid_post/" prefix).
    # Orchestrator review, 2026-08-13 (live job 406092): passing the
    # artifact identity as the loader id fails before model loading --
    # verified directly against the installed sae_lens's own registry,
    # get_pretrained_saes_directory()["gemma-scope-2-12b-it-res"].saes_map
    # is keyed by the FLAT id, with the artifact path as each entry's
    # VALUE, not its key. The two are related but must never be conflated
    # or silently rewritten into each other.
    sae_release: str | None = None
    sae_id: str | None = None
    sae_loader_id: str | None = None
    expected_layer: int | None = None
    expected_hook_name: str | None = None
    # qwen_scope_raw_pt fields (Qwen 3.5) -- None for sae_lens_registry
    expected_d_sae: int | None = None
    expected_k: int | None = None
    expected_num_layers: int | None = None
    expected_runtime_class: str | None = None
    notes: str = ""


GEMMA_3_12B_IT_TARGET = TargetPairing(
    name="gemma-3-12b-it",
    model_repo_id="google/gemma-3-12b-it",
    model_architecture="Gemma3ForConditionalGeneration",
    model_supported_by_transformer_lens=True,
    sae_repo_id="google/gemma-scope-2-12b-it",
    sae_format="sae_lens_registry",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="resid_post_all/layer_29_width_16k_l0_big",
    sae_loader_id="layer_29_width_16k_l0_big",
    expected_layer=29,
    expected_hook_name="blocks.29.hook_resid_post",
    expected_hidden_dim=3840,
    notes=(
        "Layer 29 is the RATIFIED primary for this pairing. It replaces layer 31, which this "
        "field previously carried and which its own note described as an engineering carry-over "
        "from the -pt pairing, explicitly 'not re-justified here'. Layer 29 lives in the "
        "resid_post_all tree (all 48 layers); the bare resid_post tree publishes only 12/24/31/41, "
        "so the tree prefix changes with the layer and is not interchangeable. "
        "MECHANICAL ACCEPTANCE DOES NOT FOLLOW THE MOVE: the Gemma record is scoped to layer 31 "
        "(runtime_acceptance.accepted_layer), so is_mechanically_accepted('gemma', layer=29) is "
        "False and the backend prefixes its loud unaccepted notice until a layer-29 run is "
        "imported. That is the intended, visible consequence of repointing. "
        "sae_release/sae_id verified present in the locally-installed sae_lens==6.44.2 "
        "registry; not yet verified that the corresponding HF snapshot is staged on Tamia."
    ),
)

QWEN_3_5_27B_TARGET = TargetPairing(
    name="qwen-3.5-27b",
    model_repo_id="Qwen/Qwen3.5-27B",
    model_architecture="Qwen3_5ForConditionalGeneration",
    model_supported_by_transformer_lens=False,
    sae_repo_id="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100",
    sae_format="qwen_scope_raw_pt",
    expected_hidden_dim=5120,
    expected_d_sae=81920,
    expected_k=100,
    expected_num_layers=64,
    expected_layer=None,  # engineering-only, supplied by the caller -- see module docstring
    expected_hook_name="resid_post",  # generic hook_point name from the release's own config.json, not a TL hook string
    expected_runtime_class="Qwen3_5ForCausalLM",
    notes=(
        "The SAE release is the RATIFIED L0_100 (k=100), replacing L0_50: the science this "
        "product presents was measured at layer 38 in the L0_100 dictionary, and k must equal "
        "the L0_<N> suffix of the repository id. Mechanical acceptance is scoped to layer 0 "
        "(runtime_acceptance.accepted_layer), so a layer-38 run is NOT accepted and says so. "
        "model_supported_by_transformer_lens=False is a VERIFIED negative (not an assumption): "
        "'Qwen/Qwen3.5-27B' and every 'qwen3.5'-named entry are absent from transformer_lens==3.2.1's "
        "OFFICIAL_MODEL_NAMES. Loading requires the raw-HF-forward-hooks path (see "
        "final_pairing_harness.load_qwen_target), not HookedTransformer.from_pretrained. "
        "Loaded via AutoModelForCausalLM, which dispatches model_type=qwen3_5 to "
        "Qwen3_5ForCausalLM -- a text-only causal-LM class (self.model.layers reachable directly, "
        "no vision tower or .language_model indirection), matching both Tamia's actual "
        "transformers==5.14.1 runtime and the official Qwen-Scope release's own application "
        "(same Auto class, same model.model.layers[layer] hook path). Superseded the earlier "
        "AutoModelForImageTextToText/Qwen3_5ForConditionalGeneration multimodal route, which was "
        "never actually required -- see module docstring's 2026-08-11 orchestrator review."
    ),
)

ALL_TARGETS = {t.name: t for t in (GEMMA_3_12B_IT_TARGET, QWEN_3_5_27B_TARGET)}


# ---------------------------------------------------------------------------
# Pure validation -- no torch, no network. Fully unit-testable with plain
# strings/ints/fake objects.
# ---------------------------------------------------------------------------

_HF_CACHE_SNAPSHOT_RE = re.compile(
    r"models--(?P<org>[^\\/]+)--(?P<repo>[^\\/]+)[\\/]snapshots[\\/](?P<revision>[^\\/]+)"
)


def parse_hf_cache_snapshot_path(path: str | Path) -> dict[str, str] | None:
    """Extract {org, repo, revision} from a standard huggingface_hub cache
    layout path (.../models--<org>--<repo>/snapshots/<revision>/...). The
    on-disk cache layout embeds repo identity and revision in the path
    itself -- this is a mechanical fact about how huggingface_hub lays out
    its cache, not a guess about any specific path. Returns None (not an
    error) for a path that doesn't follow this layout at all -- e.g. an
    arbitrary local directory a caller staged by hand; callers that require
    the identity check should treat None as "cannot verify", not "verified
    absent"."""
    match = _HF_CACHE_SNAPSHOT_RE.search(str(path).replace("\\", "/"))
    if match is None:
        return None
    org_repo_underscore_fixed = match.group("org"), match.group("repo").replace("--", "/")
    return {
        "org": org_repo_underscore_fixed[0],
        "repo": org_repo_underscore_fixed[1],
        "revision": match.group("revision"),
    }


class TargetIdentityMismatch(ValueError):
    """Raised by every validate_* function below -- fail closed, never warn-and-continue."""


class IdentityUnverified(TargetIdentityMismatch):
    """Raised when a path's identity can be neither mechanically verified
    (standard HF cache layout) nor treated as trusted via an independently
    supplied expected revision. Orchestrator review, 2026-08-10: a prior
    version of validate_local_snapshot_identity treated "cannot verify from
    the path alone" as license to silently continue -- that is claiming
    fail-closed verification while not actually performing it. The fix:
    a non-cache-layout path with no expected_revision now REFUSES rather
    than passing through; supplying expected_revision (Lab Assistant 1's
    inventory attesting to it) is the only way to accept such a path, and
    that acceptance is recorded as explicitly declared, not path-derived."""


def validate_local_snapshot_identity(
    path: str | Path, target: TargetPairing, *, which: Literal["model", "sae"], expected_revision: str | None = None
) -> dict[str, str]:
    """Returns a provenance dict {"verification", "repo_id", "revision"} for
    the JSON trace. Fails closed on a POSITIVE mismatch (path names a
    different repo/revision than ratified). For a path that does not follow
    the standard HF cache layout at all (e.g. a hand-staged directory):
    requires expected_revision to be supplied as trusted inventory
    provenance, recording verification="explicit_revision_declared_not_path_derived";
    if neither the cache layout NOR an expected_revision is available,
    raises IdentityUnverified rather than silently proceeding -- see that
    exception's docstring for why this is not the same as the old
    behavior."""
    expected_repo_id = target.model_repo_id if which == "model" else target.sae_repo_id
    parsed = parse_hf_cache_snapshot_path(path)
    if parsed is not None:
        actual_repo_id = f"{parsed['org']}/{parsed['repo']}"
        if actual_repo_id.lower() != expected_repo_id.lower():
            raise TargetIdentityMismatch(
                f"{which} path {str(path)!r} encodes repo identity {actual_repo_id!r}, but the "
                f"ratified target is {expected_repo_id!r}. This is the exact silent-wrong-checkpoint "
                f"failure mode this validator exists to catch -- refusing to proceed."
            )
        if expected_revision is not None and parsed["revision"] != expected_revision:
            raise TargetIdentityMismatch(
                f"{which} path {str(path)!r} encodes revision {parsed['revision']!r}, but the "
                f"recorded/expected revision is {expected_revision!r}."
            )
        return {"verification": "hf_cache_layout", "repo_id": actual_repo_id, "revision": parsed["revision"]}

    if expected_revision is not None:
        return {
            "verification": "explicit_revision_declared_not_path_derived",
            "repo_id": expected_repo_id,
            "revision": expected_revision,
        }

    raise IdentityUnverified(
        f"{which} path {str(path)!r} does not follow the standard huggingface_hub cache layout "
        f"(models--<org>--<repo>/snapshots/<revision>), and no expected revision was supplied to "
        f"treat as trusted inventory provenance. Refusing to proceed with unverified identity -- "
        f"supply --expected-{which}-revision from Lab Assistant 1's inventory, or re-stage this "
        f"path under the standard cache layout."
    )


def validate_finite_positive(value: float, *, label: str) -> None:
    """Rejects zero, negative, NaN, and infinite values -- the exact
    non-finite/non-positive STEER clamp values orchestrator review flagged
    as unguarded. Applies to --raw-clamp-value directly and to the
    dose_multiple x calibration_value product after
    gemma3_tool.dose_to_absolute_clamp resolves it (that function's own
    guard rejects non-positive max_act_approx, but `<= 0.0` is False for
    NaN, so a NaN calibration_value would slip through it silently)."""
    if math.isnan(value) or math.isinf(value):
        raise TargetIdentityMismatch(f"{label}={value!r} is not finite; refusing to steer with it.")
    if value <= 0:
        raise TargetIdentityMismatch(f"{label}={value!r} is non-positive; refusing to steer with it.")


def validate_feature_index(feature_idx: int, d_sae: int, *, label: str = "feature_idx") -> None:
    if not (0 <= feature_idx < d_sae):
        raise TargetIdentityMismatch(f"{label}={feature_idx} is outside the loaded SAE's range [0, {d_sae}).")


_QWEN_LAYER_FILENAME_RE = re.compile(r"^layer(?P<layer>\d+)\.sae\.pt$")


def validate_qwen_layer_filename(path: str | Path, layer: int) -> None:
    """The Qwen-Scope release ships one file per layer (layerN.sae.pt,
    verified against the release's own file listing) with no layer field
    inside the checkpoint itself (per its own app.py, the state_dict is
    just W_enc/b_enc/W_dec) -- the filename is the only place the layer
    identity is recorded at all, so it's the fail-closed check available."""
    name = Path(path).name
    match = _QWEN_LAYER_FILENAME_RE.match(name)
    if match is None:
        raise TargetIdentityMismatch(
            f"SAE file name {name!r} does not match the release's own layerN.sae.pt convention -- "
            f"cannot confirm which layer this file is for."
        )
    file_layer = int(match.group("layer"))
    if file_layer != layer:
        raise TargetIdentityMismatch(
            f"SAE file {name!r} is for layer {file_layer}, but --qwen-layer={layer} was requested."
        )


def validate_qwen_sae_shapes(
    *, w_enc_shape: tuple[int, ...], b_enc_shape: tuple[int, ...], w_dec_shape: tuple[int, ...],
    b_dec_shape: tuple[int, ...] | None, target: TargetPairing,
) -> None:
    """Full shape validation (not just the d_model cross-check
    validate_hidden_dims already does) -- W_enc/b_enc/W_dec/b_dec must all
    agree with each other AND with the ratified d_model/d_sae, per the
    release's own schema (W_enc: [d_sae, d_model] before transpose,
    b_enc: [d_sae], W_dec: [d_model, d_sae]).

    Orchestrator review, 2026-08-11: the official Qwen-Scope release's own
    checkpoint contract lists b_dec as PRESENT (no longer an unconfirmed
    optional key defaulted to zero -- see QwenScopeSAE's docstring), so
    b_dec_shape=None is now itself a contract violation, not a tolerated
    "unconfirmed" case -- fails closed rather than silently skipping the
    check."""
    d_model = target.expected_hidden_dim
    d_sae = target.expected_d_sae
    if w_enc_shape != (d_sae, d_model):
        raise TargetIdentityMismatch(f"W_enc shape {w_enc_shape} != expected ({d_sae}, {d_model}).")
    if b_enc_shape != (d_sae,):
        raise TargetIdentityMismatch(f"b_enc shape {b_enc_shape} != expected ({d_sae},).")
    if w_dec_shape != (d_model, d_sae):
        raise TargetIdentityMismatch(f"W_dec shape {w_dec_shape} != expected ({d_model}, {d_sae}).")
    if b_dec_shape is None:
        raise TargetIdentityMismatch(
            "b_dec_shape is None, but the release's own checkpoint contract lists b_dec as "
            "present -- refusing to treat a missing decoder bias as an acceptable, unconfirmed case."
        )
    if b_dec_shape != (d_model,):
        raise TargetIdentityMismatch(f"b_dec shape {b_dec_shape} != expected ({d_model},).")


def validate_runtime_class(actual_class_name: str, target: TargetPairing) -> None:
    """Fails closed if the class actually returned by the chosen Auto*
    loader does not match the target's ratified expected_runtime_class --
    e.g. a differently-pinned transformers version dispatching model_type
    "qwen3_5" to a different class than the one this harness was verified
    against (transformers==5.14.1 -> AutoModelForCausalLM -> "
    "Qwen3_5ForCausalLM, per this module's docstring)."""
    if target.expected_runtime_class is None:
        return
    if actual_class_name != target.expected_runtime_class:
        raise TargetIdentityMismatch(
            f"loaded runtime class {actual_class_name!r} != target {target.name!r}'s ratified "
            f"expected_runtime_class {target.expected_runtime_class!r}. This usually means the "
            f"installed transformers version dispatches this model_type differently than the "
            f"version this harness was verified against -- stop rather than proceed with an "
            f"unverified class."
        )


def validate_has_callable_generate(obj: Any, *, label: str) -> None:
    """Confirms the loaded object actually supports generation before any
    intervention is attempted -- a wrong Auto class or an unexpected
    dispatch can otherwise return a model that loads fine but has no
    usable .generate() (e.g. Qwen3_5Model, the base class AutoModel
    dispatches "qwen3_5" to)."""
    if not callable(getattr(obj, "generate", None)):
        raise TargetIdentityMismatch(
            f"{label} has no callable .generate() -- the loaded class does not support "
            f"generation; refusing to proceed with a model that cannot run the intervention."
        )


def validate_sae_loader_id_registered(loader_id: str, available_loader_ids: list[str], target: TargetPairing) -> None:
    """Confirms the flat sae_lens loader id (target.sae_loader_id) is
    actually a KEY the selected release's own registry recognizes, BEFORE
    SAE.from_pretrained is called -- catches exactly the live failure job
    406092 hit: the artifact identity (target.sae_id, e.g.
    "resid_post/layer_31_width_16k_l0_medium") is NOT a key in
    gemma-scope-2-12b-it-res's saes_map at all, only its VALUE for the
    real flat key "layer_31_width_16k_l0_medium" -- passing the artifact
    identity as sae_id fails before any model loading. Pure: the caller
    fetches available_loader_ids from sae_lens's own registry (a local
    package data structure, no network) and passes it in as a plain list,
    so this function itself needs no sae_lens import."""
    if loader_id not in available_loader_ids:
        raise TargetIdentityMismatch(
            f"loader id {loader_id!r} is not a registered SAE id for release "
            f"{target.sae_release!r} -- SAE.from_pretrained would fail before any weights load. "
            f"{len(available_loader_ids)} id(s) are registered for this release; a few: "
            f"{sorted(available_loader_ids)[:20]}"
        )


def validate_qwen_k(k: int, target: TargetPairing) -> None:
    if target.expected_k is not None and k != target.expected_k:
        raise TargetIdentityMismatch(
            f"k={k} != target {target.name!r}'s ratified expected_k={target.expected_k}. k is a "
            f"structural property of how this TopK SAE was trained (encoded in its own release "
            f"name, 'L0_50'), not a free engineering knob -- using the wrong k does not match "
            f"what the dictionary was optimized for."
        )


def validate_sae_files_match_snapshot(resolved_files: list[str], sae_path: str | Path, target: TargetPairing) -> None:
    """Proves the sae_lens REGISTRY loader actually read from the validated
    sae_path snapshot, rather than independently resolving a different
    cached revision through its own release/sae_id lookup. SAE.from_pretrained
    exposes no cache_dir/revision parameter (verified: its signature is
    (release, sae_id, device, dtype, force_download, converter) only), so
    it cannot be pinned to sae_path directly -- this instead captures every
    local file sae_lens's own hf_hub_download call resolved (see
    final_pairing_harness._capture_sae_download_paths) and checks each one
    falls under sae_path. Orchestrator review, 2026-08-10: a prior version
    validated sae_path's identity and then let SAE.from_pretrained resolve
    a completely independent path through the registry with no check that
    the two ever agreed.

    This is the LOGICAL snapshot-identity check -- the broader, family-
    agnostic sibling of validate_sae_files_match_expected_subdirectory
    below (which additionally requires the ratified SAE family
    specifically). Physical symlink-target dereferencing is EXCLUSIVELY
    the job of validate_sae_symlink_targets_stay_in_repository_cache
    further below; this function must never call Path.resolve() or
    os.path.realpath.

    Orchestrator review, 2026-08-16 ("Correct and comprehensively audit
    Gemma path-containment guards", live job 406957): this check used to
    call Path.resolve() on both sides, which FOLLOWS symlinks -- a real
    huggingface_hub snapshot entry is normally a SYMLINK from
    snapshots/<revision>/<repo-relative-path> into a SIBLING blobs/<hash>
    store under the same models--<org>--<repo> cache root (a sibling of
    snapshots/, not nested under it), so resolving a legitimate resolved
    file lands it OUTSIDE sae_path entirely and this check incorrectly
    rejected every real symlinked file as "outside the snapshot" -- the
    exact same class of bug validate_sae_files_match_expected_subdirectory
    was written to avoid (see that function's docstring), just not yet
    applied here. It also used a manual str.startswith() comparison, a
    SEPARATE, opposite-direction defect: a sibling directory sharing the
    same string prefix (e.g. snapshots/<revision>-evil) would incorrectly
    PASS, since the string "<revision>-evil" starts with the string
    "<revision>" even though it names a different directory entirely.
    Fixed identically to that sibling function: os.path.abspath (normalize
    + make absolute, never follows symlinks) on both sides, and
    Path.is_relative_to (full-path-segment containment) instead of a
    string-prefix comparison."""
    if not resolved_files:
        raise TargetIdentityMismatch(
            "the SAE registry loader resolved zero local files while loading -- cannot prove it "
            f"read from the validated snapshot at {str(sae_path)!r}."
        )
    sae_path_logical = Path(os.path.abspath(str(sae_path)))
    mismatched = [
        f
        for f in resolved_files
        if Path(os.path.abspath(f)) == sae_path_logical
        or not Path(os.path.abspath(f)).is_relative_to(sae_path_logical)
    ]
    if mismatched:
        raise TargetIdentityMismatch(
            f"the SAE registry loader resolved file(s) OUTSIDE the validated snapshot "
            f"{str(sae_path)!r}: {mismatched}. This is the exact silent-wrong-revision failure "
            f"this check exists to catch, even though the file(s) fetched successfully."
        )


def _hf_repository_cache_root(path: str | Path) -> Path | None:
    """Returns the huggingface_hub repository cache root
    (.../models--<org>--<repo>) that both snapshots/ and blobs/ live
    under as siblings, for a path following the standard cache layout.
    Returns None for a path that doesn't follow it (e.g. a hand-staged
    directory) -- there is no repo-cache root to compute, and no blobs/
    convention to bound a symlink target against."""
    normalized = str(path).replace("\\", "/")
    match = _HF_CACHE_SNAPSHOT_RE.search(normalized)
    if match is None:
        return None
    repo_root_end = match.start() + len(f"models--{match.group('org')}--{match.group('repo')}")
    return Path(normalized[:repo_root_end])


def validate_sae_files_match_expected_subdirectory(
    resolved_files: list[str], sae_path: str | Path, target: TargetPairing
) -> dict[str, Any]:
    """LOGICAL snapshot identity: proves every resolved SAE file's OWN
    (snapshot-relative) path lives under the RATIFIED SAE family's own
    subdirectory (target.sae_id, e.g. "resid_post/layer_31_width_16k_
    l0_medium"), not merely somewhere under the correctly-validated
    snapshot as a whole. See validate_sae_symlink_targets_stay_in_
    repository_cache below for the complementary PHYSICAL check -- that
    function checks where a symlink's dereferenced bytes actually live;
    this function checks the symlink's OWN logical location, which is
    where the SAE family identity actually lives (see that function's
    docstring for why the physical blob target has no comparable
    structure at all, and must not be required to).

    Orchestrator review, 2026-08-12: the final Gemma Scope IT snapshot
    ships FIVE different SAE families sharing the identical
    "layer_31_width_16k_l0_medium" suffix -- attn_out, mlp_out,
    resid_post, transcoder, and transcoder affine -- so
    validate_sae_files_match_snapshot above (which only proves "somewhere
    under this snapshot") cannot, by itself, prove which of the five was
    actually loaded. Post-hoc inspection of the loaded weights is not
    acceptance; this is the fail-closed pre-generation gate.

    Deliberately does NOT call Path.resolve() / os.path.realpath on
    either sae_path or the resolved files -- only os.path.abspath
    (normalize + make absolute, WITHOUT following symlinks). The standard
    huggingface_hub cache stores each snapshot entry as a SYMLINK from
    snapshots/<revision>/<repo-relative-path> to a flat, subdirectory-free
    blobs/<hash> file that is a SIBLING of (not nested under) the
    snapshots directory -- dereferencing the symlink here would land on
    that flat blob path and incorrectly reject every legitimate file,
    regardless of which SAE family it actually belongs to. Uses
    Path.is_relative_to rather than a manual string-prefix comparison, to
    close the sibling-prefix false-match case (e.g. "resid_post_v2" or
    "layer_31_width_16k_l0_medium_v2" matching a naive str.startswith())
    without introducing any symlink dereferencing in the process.

    Returns a provenance dict {"expected_sae_subdirectory",
    "sae_subdirectory_membership_verified"} for the JSON trace -- returned
    only on success; every failure path raises instead."""
    if not resolved_files:
        raise TargetIdentityMismatch(
            "zero resolved SAE files -- cannot prove any of them belong to the ratified SAE "
            f"subdirectory {target.sae_id!r}."
        )
    if not target.sae_id:
        raise TargetIdentityMismatch(
            f"target {target.name!r} has no ratified sae_id to derive an expected SAE "
            f"subdirectory from -- cannot perform this check."
        )
    expected_dir = Path(os.path.abspath(str(sae_path))) / target.sae_id
    mismatched = []
    for f in resolved_files:
        f_logical = Path(os.path.abspath(f))
        if f_logical == expected_dir or not f_logical.is_relative_to(expected_dir):
            mismatched.append(f)
    if mismatched:
        raise TargetIdentityMismatch(
            f"resolved SAE file(s) are OUTSIDE the ratified SAE subdirectory {target.sae_id!r} "
            f"within the validated snapshot -- sibling SAE families (attn_out, mlp_out, "
            f"transcoder, transcoder affine) share the identical layer/width/l0 suffix and must "
            f"not be silently accepted just because they live under the correct snapshot: "
            f"{mismatched}"
        )
    return {"expected_sae_subdirectory": target.sae_id, "sae_subdirectory_membership_verified": True}


def validate_sae_symlink_targets_stay_in_repository_cache(
    resolved_files: list[str], sae_path: str | Path, target: TargetPairing
) -> None:
    """PHYSICAL cache safety: the complementary check to
    validate_sae_files_match_expected_subdirectory above. Hugging Face
    snapshot entries are normally SYMLINKS from
    snapshots/<revision>/<repo-relative-path> into the repository's own
    blobs/<hash> store -- a sibling of snapshots/, under the same
    models--<org>--<repo> cache root. A real symlink's dereferenced target
    therefore legitimately LEAVES the snapshot subdirectory tree entirely;
    that alone is not a problem. What this DOES check: that the
    dereferenced target still belongs to the SAME repository's cache
    root, not some other repository's cache or an arbitrary filesystem
    location. It deliberately does NOT require the physical blob path to
    retain the sae_id directory structure -- blobs/ is a flat, hash-named
    store with no such structure at all; that identity belongs to the
    revision snapshot entry checked above, not to the blob store.

    Only applies when a resolved file actually IS a symlink on disk
    (Path.is_symlink() returns False, not an error, for a path that does
    not exist or is a plain file/copy -- e.g. under
    HF_HUB_DISABLE_SYMLINKS_DOWNLOAD, or in this project's own synthetic
    unit tests that build path strings without creating real files) --
    there is nothing to dereference otherwise, and no physical-safety
    claim to make. Silently returns (no-op) for a sae_path that doesn't
    follow the standard huggingface_hub cache layout at all (a hand-staged
    directory): there is no blobs/ convention to bound a symlink target
    against in that case either.

    NOTE: this check exercises real Path.is_symlink()/os.path.realpath
    filesystem calls, but has only been proven against a synthetic
    on-disk symlink in this project's own tests where the local machine
    has permission to create one (creating a symlink requires elevated
    privileges on some Windows machines, including the one this was
    written on) -- it remains UNVERIFIED against a real sae_lens/
    huggingface_hub download's actual symlink layout until the first live
    Tamia run."""
    repo_cache_root = _hf_repository_cache_root(sae_path)
    if repo_cache_root is None:
        return
    escaped = []
    for f in resolved_files:
        f_path = Path(f)
        if not f_path.is_symlink():
            continue
        dereferenced = Path(os.path.realpath(f))
        if not dereferenced.is_relative_to(repo_cache_root):
            escaped.append((f, str(dereferenced)))
    if escaped:
        raise TargetIdentityMismatch(
            f"resolved SAE symlink(s) dereference to a target OUTSIDE this repository's own "
            f"cache root {str(repo_cache_root)!r} -- the physical blob does not belong to "
            f"{target.sae_repo_id!r}'s cache at all: {escaped}"
        )


_HF_DEFAULT_REVISION_ALIASES = (None, "main")


def resolve_local_gemma_sae_path(
    *,
    repo_id: str,
    filename: str,
    subfolder: str | None,
    revision: str | None,
    sae_snapshot_root: str | Path,
    target: TargetPairing,
) -> tuple[str, str]:
    """Maps ONE sae_lens-internal hf_hub_download(repo_id=, filename=,
    subfolder=, revision=) request directly onto a file inside the
    already-validated local SAE snapshot at sae_snapshot_root -- no Hub
    resolution, no network, no cache mutation of any kind. The caller
    (final_pairing_harness._capture_sae_download_paths) installs this as a
    full REPLACEMENT for sae_lens.loading.pretrained_sae_loaders's own
    hf_hub_download reference, not a pass-through wrapper around it -- see
    this module's docstring (orchestrator review, 2026-08-14, live job
    406259) for exactly why a pass-through wrapper is not sufficient: the
    real hf_hub_download's OWN revision-defaulting and refs/main lookup
    happen before any wrapper's return value is even inspected, and that
    lookup fails offline against this project's commit-pinned cache
    regardless of what the wrapper would have done with a successful
    result.

    Mirrors huggingface_hub.file_download.hf_hub_download's OWN
    subfolder+filename join convention exactly (verified against the
    installed huggingface_hub==1.24.0 source: `if subfolder: filename =
    f"{subfolder}/{filename}"`) so this stays correct even if sae_lens's
    internal call shape changes slightly between its config.json/
    params.safetensors/shape-lookup call sites.

    Returns (resolved_local_path, requested_relative_filename) -- the
    caller records both for the provenance trace. Fails closed
    (TargetIdentityMismatch) on:
      - repo_id other than the ratified target.sae_repo_id -- this
        resolver is installed exclusively for the one repository it was
        built for;
      - an explicit revision other than the default (None, or
        huggingface_hub's own default branch name "main") -- real
        sae_lens call sites never pass one at all, so this only guards
        against a future/different explicit request silently being served
        THIS pinned snapshot's files instead of failing;
      - an absolute filename, or a ".." path-traversal segment;
      - a filename resolving outside the ratified target.sae_id artifact
        subdirectory -- the same sibling-family case
        validate_sae_files_match_expected_subdirectory above guards
        post-hoc, closed here one request earlier, before any file is even
        read.
    Raises huggingface_hub.utils.EntryNotFoundError (deliberately NOT
    TargetIdentityMismatch) for a file that is genuinely absent from the
    validated snapshot -- the exact exception type sae_lens's own
    get_gemma_3_config_from_hf already catches for its one documented
    optional case (a missing config.json falls back to
    _infer_gemma_3_raw_cfg_dict, a pure computation from repo_id/
    folder_name strings needing no network) -- raising a different type
    here would silently break that already-shipped fallback."""
    from huggingface_hub.utils import EntryNotFoundError

    if repo_id.lower() != target.sae_repo_id.lower():
        raise TargetIdentityMismatch(
            f"local-snapshot-only SAE resolution received a request for repository {repo_id!r}, "
            f"but was installed exclusively for the ratified target {target.sae_repo_id!r} -- "
            f"refusing to resolve it from any local path, let alone fall back to the network."
        )
    if revision not in _HF_DEFAULT_REVISION_ALIASES:
        raise TargetIdentityMismatch(
            f"local-snapshot-only SAE resolution received an explicit revision {revision!r} for "
            f"{filename!r} -- only the default (no revision requested, or huggingface_hub's own "
            f"default branch name 'main') resolves to the validated pinned snapshot; a request "
            f"for a specific different revision must not be silently served that snapshot's "
            f"files instead."
        )
    relative = f"{subfolder}/{filename}" if subfolder else filename
    normalized = relative.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise TargetIdentityMismatch(
            f"local-snapshot-only SAE resolution received an absolute filename {relative!r} -- refusing."
        )
    if ".." in normalized.split("/"):
        raise TargetIdentityMismatch(
            f"local-snapshot-only SAE resolution received a path-traversal filename {relative!r} -- refusing."
        )
    if not target.sae_id or not PurePosixPath(normalized).is_relative_to(target.sae_id):
        raise TargetIdentityMismatch(
            f"local-snapshot-only SAE resolution received a request for {relative!r}, which is "
            f"OUTSIDE the ratified SAE subdirectory {target.sae_id!r} -- sibling SAE families "
            f"(attn_out, mlp_out, transcoder, transcoder affine) share this release's identical "
            f"layer/width/l0 suffix and must not be silently served even though they live in the "
            f"same snapshot."
        )
    local_path = Path(sae_snapshot_root) / normalized
    if not local_path.is_file():
        raise EntryNotFoundError(
            f"{relative!r} does not exist in the validated local SAE snapshot at "
            f"{str(sae_snapshot_root)!r} -- local-snapshot-only resolution never falls back to "
            f"the network to look for it elsewhere."
        )
    return str(local_path), normalized


def validate_hidden_dims(model_d_model: int, sae_d_in: int, target: TargetPairing) -> None:
    if model_d_model != sae_d_in:
        raise TargetIdentityMismatch(
            f"model hidden dim {model_d_model} != SAE d_in {sae_d_in} -- refusing to attach a "
            f"hook between mismatched dimensions."
        )
    if model_d_model != target.expected_hidden_dim:
        raise TargetIdentityMismatch(
            f"loaded hidden dim {model_d_model} != target {target.name!r}'s ratified "
            f"expected_hidden_dim {target.expected_hidden_dim}. Either the wrong model snapshot "
            f"was loaded, or the ratified expectation is stale -- both are stop conditions, not "
            f"something to silently proceed past."
        )


def validate_hook_identity(actual_hook_name: str, target: TargetPairing) -> None:
    """For sae_lens_registry targets (Gemma-it), expected_hook_name is an
    exact TL hook string ('blocks.31.hook_resid_post') and must match
    exactly. For qwen_scope_raw_pt targets, expected_hook_name is the
    release's own generic hook_point name ('resid_post') and this only
    checks that string appears in whatever hook identifier the caller
    constructed -- there is no TL hook-name convention for a raw-HF path,
    and the actual layer number is engineering-only (see module docstring),
    so an exact-match check would be checking a fact this module does not
    own."""
    if target.expected_hook_name is None:
        return
    if target.sae_format == "sae_lens_registry":
        if actual_hook_name != target.expected_hook_name:
            raise TargetIdentityMismatch(
                f"hook name {actual_hook_name!r} != target {target.name!r}'s ratified "
                f"expected_hook_name {target.expected_hook_name!r}."
            )
    else:
        if target.expected_hook_name not in actual_hook_name:
            raise TargetIdentityMismatch(
                f"hook identifier {actual_hook_name!r} does not contain the release's own "
                f"hook_point {target.expected_hook_name!r}."
            )


def validate_qwen_layer_choice(layer: int, target: TargetPairing) -> None:
    """Engineering-only gate: the layer must be a real layer the release
    covers (0..expected_num_layers-1). This is NOT a scientific check --
    it never asserts a specific layer is meaningful, only that the caller's
    choice is one the release actually shipped a file for."""
    if target.expected_num_layers is None:
        return
    if not (0 <= layer < target.expected_num_layers):
        raise TargetIdentityMismatch(
            f"layer {layer} is outside {target.name!r}'s covered range "
            f"[0, {target.expected_num_layers})."
        )
