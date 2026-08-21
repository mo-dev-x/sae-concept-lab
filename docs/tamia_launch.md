# Tamia launch: real backends (Qwen3.5-27B, gemma-3-12b-it)

This document is the product-side launch companion to qwen-sae-interp's
own `docs/final_pairing_tamia_packet.md`. It never re-derives an
identity/revision fact -- every pinned path and revision below is copied
directly from the sealed, hash-verified evidence at
`results/final_pairing/job_406092/` and `results/final_pairing/job_407008/`
in the scientific repository (qwen-sae-interp, commit `b6d598b`), the same
evidence `sae_concept_lab.core.runtime_acceptance` imported after
independently re-verifying it via `git show` + SHA-256. See
[`BOUNDARY.md`](../BOUNDARY.md) for the full account of why that evidence
is trusted and two earlier claims were not.

## What launching a real backend does, and does not, do

`--qwen-backend runtime` / `--gemma-backend runtime` construct a
`QwenRuntimeBackend`/`GemmaRuntimeBackend` (`sae_concept_lab/core/
qwen_backend.py` / `gemma_backend.py`) instead of `StubConceptLabBackend`.
Both are **lazy**: constructing them, and importing this whole product,
touches no GPU and reads no model file -- the extracted loader
(`sae_concept_lab/extracted_runtime/`) only runs the first time a request
actually reaches `generate()`.

Both pairings are **mechanically accepted** as of this product commit (see
`sae_concept_lab/core/runtime_acceptance.py`): an intervention reaching the
hook and measurably moving the residual has been proven, for Qwen under
job 406092's two Qwen scenarios (ALL, GENERATED_ONLY -- that job's two
Gemma scenarios failed and are NOT covered by this acceptance) and for
Gemma under job 407008 (ALL, GENERATED_ONLY, both scenarios). This is
**mechanical evidence only**. It is not a scientific concept claim, not a
calibration, not a behavioral-quality claim, and Qwen layer 0 / feature
4096 and Gemma feature 250 / raw clamp 5000 remain ENGINEERING-ONLY,
exactly as the sealed READMEs require. Launching a real backend never
bypasses `--mode release`'s separate, unrelated requirement for ATTESTED
concepts and content-verified evidence -- this repository ships none, so
`--mode release` refuses regardless of which backend flag is passed (see
below).

## Prerequisites

Both commands require, from a Lab Assistant's inventory (never guessed):

1. A local, offline-only (`HF_HUB_OFFLINE=1`) snapshot for the model.
2. A local snapshot (Gemma: SAE snapshot ROOT directory) or a specific
   `layerN.sae.pt` file (Qwen) for the SAE.
3. The revision recorded for each in that inventory -- pass it via
   `--qwen-expected-model-revision`/`--qwen-expected-sae-revision` (or the
   `--gemma-*` equivalents) whenever your local path does not already
   follow the standard `huggingface_hub` cache layout
   (`models--<org>--<repo>/snapshots/<revision>/...`), or the extracted
   loader raises `IdentityUnverified` and refuses to proceed -- there is no
   silent "cannot verify, continue anyway" path. This is
   `targets.validate_local_snapshot_identity`, extracted unmodified; see
   `sae_concept_lab/extracted_runtime/targets.py`.

The exact revisions the accepted evidence itself used (from
`results/final_pairing/job_406092/qwen_3_5_27b_mechanical_all.json` and
`results/final_pairing/job_407008/gemma_3_12b_it_all.json`'s own
`provenance` blocks, quoted verbatim for reference -- your own inventory's
revisions may legitimately differ if the snapshots were re-staged):

| | Qwen3.5-27B | Qwen-Scope layer 0 | gemma-3-12b-it | gemma-scope-2-12b-it-res |
|---|---|---|---|---|
| repository | `Qwen/Qwen3.5-27B` | `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` | `google/gemma-3-12b-it` | `google/gemma-scope-2-12b-it` |
| revision (evidence run) | `fc05daec18b0a78c049392ed2e771dde82bdf654` | `13d4221569f7ca5d3c1e605e3e3dc95117e4807c` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | `4c419f1ba0be8b7754d4151d4f26c23b92a9029e` |

## Reaching the UI: forward the port, never bind the node

**Bind `127.0.0.1`, never `0.0.0.0`.** A Tamia compute node is shared. Binding
`0.0.0.0` publishes this UI -- and every generation it produces -- to every
other user on that node, on an unauthenticated port. `0.0.0.0` is also
unnecessary: SSH port-forwarding reaches a loopback bind perfectly well.

From your laptop, jump through the login node to the allocated compute node and
forward the port to your own loopback:

```bash
# 1. Allocate interactively, and note the node name it lands on.
ssh yazid@tamia.alliancecan.ca
salloc --account=aip-chgag196 --nodes=1 --gres=gpu:h100:4 --mem=0        --cpus-per-task=32 --time=02:00:00
hostname          # e.g. tg10701

# 2. From a SECOND terminal on your laptop, forward through the login node.
ssh -N -L 7860:127.0.0.1:7860 -J yazid@tamia.alliancecan.ca yazid@tg10701
```

Then open `http://127.0.0.1:7860` locally. The `-J` jump reaches the compute
node, and `-L 7860:127.0.0.1:7860` forwards to *that node's* loopback, which is
exactly where the app is listening.

## Launch -- Qwen real backend, dev mode

```bash
HF_HUB_OFFLINE=1 python -m sae_concept_lab.app \
  --qwen-backend runtime \
  --qwen-model-path /path/from/inventory/qwen3.5-27b \
  --qwen-sae-path /path/from/inventory/qwen-scope/layer0.sae.pt \
  --qwen-layer 0 \
  --qwen-expected-model-revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --qwen-expected-sae-revision 13d4221569f7ca5d3c1e605e3e3dc95117e4807c \
  --server-name 127.0.0.1 --server-port 7860
```

`--qwen-layer 0` is engineering-only (no code default -- see
`targets.py`'s own module docstring: "no ratified layer"); it must match
the layer the accepted evidence used unless you are deliberately
exploring past it (in which case, treat mechanical acceptance as
unestablished for that layer -- `runtime_acceptance.py`'s record is scoped
to layer 0 specifically, not to "any Qwen layer").

## Launch -- Gemma real backend, dev mode

```bash
HF_HUB_OFFLINE=1 python -m sae_concept_lab.app \
  --gemma-backend runtime \
  --gemma-model-path /path/from/inventory/gemma-3-12b-it \
  --gemma-sae-path /path/from/inventory/gemma-scope-2-12b-it \
  --gemma-expected-model-revision 96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  --gemma-expected-sae-revision 4c419f1ba0be8b7754d4151d4f26c23b92a9029e \
  --server-name 127.0.0.1 --server-port 7860
```

`--gemma-sae-path` is the SAE snapshot **root** directory (not a single
file) -- the extracted loader resolves every file `sae_lens` needs
directly from within it, never over the network (`resolve_local_gemma_sae_path`,
`sae_concept_lab/extracted_runtime/gemma_loader.py`).

## Launch -- both real backends together (dual-pairing configuration)

```bash
HF_HUB_OFFLINE=1 python -m sae_concept_lab.app \
  --qwen-backend runtime --qwen-model-path ... --qwen-sae-path ... --qwen-layer 0 \
  --qwen-expected-model-revision ... --qwen-expected-sae-revision ... \
  --gemma-backend runtime --gemma-model-path ... --gemma-sae-path ... \
  --gemma-expected-model-revision ... --gemma-expected-sae-revision ...
```

Both tabs run against real weights simultaneously; each pairing's own
identity/subdirectory/symlink guards apply independently -- one pairing's
loader failing does not affect the other's.

## `--mode release`: still refuses, on this build, regardless of backend

```bash
HF_HUB_OFFLINE=1 python -m sae_concept_lab.app --mode release \
  --qwen-backend runtime --qwen-model-path ... --qwen-sae-path ... --qwen-layer 0 \
  --gemma-backend runtime --gemma-model-path ... --gemma-sae-path ... \
  --evidence-registry-root /path/to/a/real/registry
```

This still exits non-zero: every fixture this repository ships is
`provenance: "fake"` (never `attested`), so `evaluate_publishability`
blocks every one of them regardless of which backend is selected or
whether that backend's pairing is mechanically accepted. Mechanical
acceptance of the intervention MECHANISM
(`sae_concept_lab.core.runtime_acceptance`) and public release of a
SCIENTIFIC CONCEPT (`sae_concept_lab.canonical.concept_bundle.release`)
are, and must remain, two separate gates -- see `BOUNDARY.md`.

## Failure classification

Every exception the extracted loaders raise is the exact, unmodified
qwen-sae-interp class (`sae_concept_lab.extracted_runtime.targets.
TargetIdentityMismatch` / `IdentityUnverified`) -- read
qwen-sae-interp's own `docs/final_pairing_tamia_packet.md` "Failure
classification" table for what each one means; this product re-implements
none of that logic and therefore inherits none of the drift risk of
re-describing it here. `ValueError: --qwen-layer is required` /
`ReleaseGateError` messages are this product's own, and are self-explanatory
in the printed text.
