# Tamia final product-integration smoke packet

This document is the companion to [`docs/tamia_launch.md`](tamia_launch.md)
for `sae_concept_lab/smoke/tamia_smoke.py` -- a reproducible, fail-closed
smoke runner that proves the extracted, mechanically-accepted runtime
backends (`QwenRuntimeBackend`, `GemmaRuntimeBackend`) actually run,
**through the real product adapters**, against real weights on Tamia.

This is **not** a second scientific acceptance harness. The mechanical
acceptance claim it exercises was already sealed and imported by
[`sae_concept_lab/core/runtime_acceptance.py`](../sae_concept_lab/core/runtime_acceptance.py)
from qwen-sae-interp evidence commit `b6d598b` (see that module and
[`BOUNDARY.md`](../BOUNDARY.md)). This packet never re-derives, widens, or
re-litigates that claim -- it only proves this product's own adapters
(`core/qwen_backend.py`, `core/gemma_backend.py`, `core/execution_guard.py`,
`core/logic.py`, `ui/app_ui.build_demo`) correctly carry it through to a
real generation, end to end.

## What it does, and does not, bypass

Every real generation in this packet goes through
`core.logic.send_message` -> `backend.generate()` -> `execution_guard`'s
defensive re-enforcement inside the backend -> `extracted_runtime` -- the
exact chain `ui/tab.py`'s own `_on_send` calls. The runner never imports
`sae_concept_lab.extracted_runtime.qwen_loader`/`gemma_loader`/`hooks`
directly to drive a generation; the one place it reaches past the backend
layer is `scenario_identity_cannot_be_swapped`, which calls
`extracted_runtime.targets.validate_local_snapshot_identity` directly --
to prove the loader's own fail-closed identity guard, not to bypass it.

Every concept-bundle entry the packet resolves against
(`sae_concept_lab/smoke/entries.py`) is built directly in Python and
carries a concept_id prefixed `ENGINEERING-ONLY-SMOKE-`, `provenance:
"fake"`, and is never added to `fixtures/loader._ENTRY_FILENAMES` -- so it
can never enter fixture discovery, never render in the Gradio UI, and
never reach the release gate's publishability check.
`tests/test_tamia_smoke.py::test_scenario_smoke_entries_hidden_passes`
proves this mechanically on every test run.

## Scenarios

| scenario_id | pairing | what it proves |
|---|---|---|
| `smoke_entries_hidden` | both | no smoke concept_id is reachable via `fixtures.loader.load_entries()`, and every smoke entry is `provenance=FAKE` |
| `identity_cannot_be_swapped` | both | a Gemma-repo-shaped path is refused as a Qwen model identity, and vice versa (`extracted_runtime.targets.validate_local_snapshot_identity`) |
| `qwen_multi_sae_prohibited` | qwen | two SAE identities at one layer are refused as **PROHIBITED** (`MultipleSaeIdentitiesAtLayerError`) |
| `gemma_cross_layer_capability_limit` | gemma | two distinct (sae_id, layer) execution groups are refused as **CAPABILITY_LIMIT** (`MultipleExecutionGroupsError`) |
| `qwen_backend_layer_mismatch_refused` | qwen | a resolved target layer that disagrees with the backend's configured `qwen_layer` is refused |
| `qwen_all` | qwen | real generation, layer 0 / feature 4096 / raw target 20, `positions=all` -- prefill (call_index 0) must show a nonzero residual delta |
| `qwen_generated_only` | qwen | same inputs, `positions=generated_only` -- prefill must be masked off (zero residual delta) |
| `gemma_all` | gemma | real generation, resid_post layer 31 / feature 250 / raw clamp 5000, `positions=all` |
| `gemma_generated_only` | gemma | same inputs, `positions=generated_only` |
| `app_smoke_entries_unreachable_via_ui` | both | no hidden smoke concept_id is present in either tab's entries closure |
| `app_smoke_http_200` | both | the real Gradio app, launched with both real backends, responds HTTP 200 |
| `app_smoke_bounded_request` | qwen | one bounded request through `core.logic.send_message` (the same adapter `_on_send` calls) against the running app's own backend instance |
| `app_smoke_release_still_refuses` | both | `--mode release` still refuses for both pairings (no ATTESTED concepts are shipped) |

Every scenario is run and recorded regardless of what happened before it
(`_run_guarded` converts any exception into a failed, non-fatal
`ScenarioResult`) -- a later successful scenario can never mask an
earlier failure. `SmokePacket.passed` is `all(s.passed for s in
scenarios)`, computed once at the end from the complete list.

The Qwen phase (both position scenarios) runs to completion and its
backend is dropped and GPU-released (`gc.collect()` +
`torch.cuda.empty_cache()`) before the Gemma phase constructs its own
backend. The application-smoke phase is a separate, final dual-pairing
phase -- fresh `QwenRuntimeBackend`/`GemmaRuntimeBackend` instances, both
loaded together (as `docs/tamia_launch.md`'s "both real backends
together" configuration documents), released at the end.

## Exact Tamia submission command

**No package installation and no environment mutation occurs anywhere in
this procedure.** `pip install -e .` is neither required nor performed on
Tamia: this repository's package tree has no build step, no compiled
extension, and no `entry_points`/console-script a launch depends on, so
pointing `PYTHONPATH` at the extracted archive's repository root resolves
every import exactly as an editable install would -- proven directly by
`tests/test_tamia_smoke.py::test_smoke_entry_point_runs_via_pythonpath_alone_without_any_pip_install`,
which runs this exact module with a completely bare interpreter (`-S`,
no site-packages at all) and only `PYTHONPATH` set. The existing
`/home/y/yazid/sprint-venv` already carries every heavy dependency (torch/
transformers/sae_lens/transformer_lens) this repository's real backends
need -- nothing on that shared venv is modified, added to, or reinstalled.

```bash
cd /path/to/sae-concept-lab-8332b01   # the isolated extraction (see "Archive" above)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

/home/y/yazid/sprint-venv/bin/python -m sae_concept_lab.smoke.tamia_smoke \
  --qwen-model-path /scratch/y/yazid/hf_cache/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --qwen-sae-path /scratch/y/yazid/hf_cache/hub/models--Qwen--SAE-Res-Qwen3.5-27B-W80K-L0_50/snapshots/13d4221569f7ca5d3c1e605e3e3dc95117e4807c/layer0.sae.pt \
  --qwen-expected-model-revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --qwen-expected-sae-revision 13d4221569f7ca5d3c1e605e3e3dc95117e4807c \
  --gemma-model-path /scratch/y/yazid/hf_cache/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  --gemma-sae-path /scratch/y/yazid/hf_cache/hub/models--google--gemma-scope-2-12b-it/snapshots/4c419f1ba0be8b7754d4151d4f26c23b92a9029e \
  --gemma-expected-model-revision 96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  --gemma-expected-sae-revision 4c419f1ba0be8b7754d4151d4f26c23b92a9029e \
  --output tamia_smoke_packet.json
```

The four snapshot paths above already follow the standard
`huggingface_hub` cache layout (`models--<org>--<repo>/snapshots/<revision>/...`),
so `targets.validate_local_snapshot_identity` verifies each one's identity
directly from the path itself; the `--*-expected-*-revision` flags are
shown explicitly above for clarity and match what the path alone already
proves -- omitting them would resolve identically for these exact paths,
but never omit them for a path staged outside this layout (see
`docs/tamia_launch.md`). `--qwen-device`/`--qwen-dtype`/`--gemma-device`/
`--gemma-dtype` default to `cuda`/`bfloat16` and rarely need overriding.
`--max-new-tokens` defaults to (and is capped at) 4. `--server-port`
defaults to 7861 (distinct from the dev-mode default 7860, so both can
run on the same node without a port clash).

**Never install anything, or write any cache/temp artifact, to `C:`** --
this applies to any Windows-side staging step; on Tamia itself (Linux),
route `--output` and any HF cache directories to project scratch/D-drive
paths per this project's existing devcache convention, never a
system/home-quota-limited path.

## Expected artifacts

- **stdout/stderr**: one `[PASS]`/`[FAIL] <scenario_id> (<pairing>): <summary>`
  line per scenario, then a final `TAMIA SMOKE PACKET: ALL SCENARIOS
  PASSED` or `... AT LEAST ONE SCENARIO FAILED` line.
- **`--output` (default `tamia_smoke_packet.json`)**: the one aggregate,
  machine-readable result -- `product_commit`,
  `runtime_extraction_source_commits` (`{"qwen": "e63b08e...", "gemma":
  "de3b499..."}`), `acceptance_evidence_commits` (`{"qwen": "b6d598b...",
  "gemma": "b6d598b..."}`), `passed`, and the full `scenarios` array. Each
  scenario's `detail` carries, for the four real-generation scenarios:
  the resolved canonical execution dict and fingerprints, the
  backend-translated request (requested/resolved/backend-received
  value), model/SAE/hook identity (`provenance`), the full per-call
  `InterventionTrace` list (activation before/after, residual delta,
  prefill/decode classification), `mechanical_verdict`'s hook-count
  summary, and the generated text.
- **Exit code**: `0` iff every scenario passed; `1` otherwise. A
  pre-flight `ValueError` (e.g. a required path missing) exits `2`
  (argparse's own convention).

## Failure classification

| symptom | meaning |
|---|---|
| `qwen_multi_sae_prohibited`/`gemma_cross_layer_capability_limit` fail (i.e. did NOT raise) | same-layer/cross-layer enforcement regressed -- stop and investigate before trusting any other result in the packet |
| `identity_cannot_be_swapped` fails | `extracted_runtime.targets`'s identity guard regressed -- do not proceed to a real launch until fixed |
| `qwen_all`/`gemma_all`/`*_generated_only` fail with `"positions semantics assertion failed"` | the real ALL/GENERATED_ONLY masking contract did not hold against real weights -- treat mechanical acceptance as suspect for this run, do not re-affirm `core/runtime_acceptance.py`'s claim from this alone |
| any real scenario raises `TargetIdentityMismatch`/`IdentityUnverified` | the extracted loader's own fail-closed guards refused the supplied path/revision -- see `docs/tamia_launch.md`'s failure-classification table (this packet re-implements none of that logic and inherits it unmodified) |
| any real scenario raises `RuntimeAcceptanceError` | `core/runtime_acceptance.py`'s `ACCEPTANCE_REGISTRY` no longer carries an attached record for that pairing -- this smoke runner never proceeds to load a real backend in that state; treat as a hard stop, not a scenario to retry |
| `app_smoke_boot` fails | the Gradio application itself failed to launch with both real backends wired in -- check the recorded exception before treating any other application-smoke result as meaningful |
| `app_smoke_release_still_refuses` fails (i.e. release mode did NOT refuse) | a real regression in the release gate -- stop immediately; this must never pass in this build (see `sae_concept_lab/README.md`'s release-mode section) |

## Tests

- [`tests/test_tamia_smoke.py`](../tests/test_tamia_smoke.py): CPU/fake-loader
  tests for orchestration, aggregation, every CPU-safe defensive
  assertion, and the acceptance-gate precondition -- runs in this
  repository's own dev venv, with real weights faked out via
  `tests/_fake_runtime.py`.
- [`tests/test_tamia_smoke_torch.py`](../tests/test_tamia_smoke_torch.py):
  torch-enabled test path. Tier 1 (`pytest.importorskip("torch")`-gated)
  proves the REAL, unfaked `_make_clamp_hook`/`wrap_hook_with_diagnostics`/
  `mechanical_verdict` numerics satisfy this packet's own ALL/
  GENERATED_ONLY prefill assertions, using a tiny synthetic
  `torch.nn.Module` decoder layer and an identity SAE -- no real model/SAE
  weights needed, only torch itself. Tier 2
  (`test_build_smoke_packet_against_real_staged_tamia_snapshots`) is
  gated on four environment variables and is the one test in this
  repository meant to run against genuine Tamia snapshots:

  ```bash
  export SAE_CONCEPT_LAB_TAMIA_QWEN_MODEL_PATH=/path/from/inventory/qwen3.5-27b
  export SAE_CONCEPT_LAB_TAMIA_QWEN_SAE_PATH=/path/from/inventory/qwen-scope/layer0.sae.pt
  export SAE_CONCEPT_LAB_TAMIA_GEMMA_MODEL_PATH=/path/from/inventory/gemma-3-12b-it
  export SAE_CONCEPT_LAB_TAMIA_GEMMA_SAE_PATH=/path/from/inventory/gemma-scope-2-12b-it
  HF_HUB_OFFLINE=1 pytest tests/test_tamia_smoke_torch.py -k real_staged_tamia -v
  ```
