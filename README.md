# SAE Concept Lab (standalone product repository)

Standalone Gradio UI/stub product build, extracted from the
`qwen-sae-interp` scientific repository under recorded provenance. This
repository owns the product UI and its deployment adapter only; it does
not own, and must never be treated as, a scientific source of truth. See
[`BOUNDARY.md`](BOUNDARY.md) for the full repository-boundary statement.

Everything this app currently serves is synthetic: both model tabs are
backed by `StubConceptLabBackend` (deterministic, GPU-free, every response
tagged `[FAKE STUB -- UI TEST ONLY]`) and two `is_synthetic: true,
release_blocked: true` fixture bundles. `--mode release` is a fail-closed
gate that refuses to launch on this build no matter how the fixture JSON
is edited, because the gate also checks the backend's concrete type. See
[`sae_concept_lab/README.md`](sae_concept_lab/README.md) for the full
detail on the UI, the release gate, and known Gradio limitations.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest
```

## Launch (development mode -- fake UI, always safe)

```bash
python -m sae_concept_lab.app
```

Then open the printed local URL (default `http://127.0.0.1:7860`).

## Launch (release mode -- always refuses on this build)

```bash
python -m sae_concept_lab.app --mode release
```

This exits non-zero and never opens a server, because every backend this
build constructs is `StubConceptLabBackend`. See
[`sae_concept_lab/README.md`](sae_concept_lab/README.md) for why that is a
deliberate, structural refusal rather than a flag someone forgot to flip.

## Canonical concept-bundle contract (extracted, not yet wired)

`sae_concept_lab/canonical/concept_bundle/` carries the eight-module
concept-bundle contract, mechanically extracted from qwen-sae-interp and
certified against Engineer 3's frozen 50-vector conformance pack. It is
standard-library-only, has zero import-time dependency on
qwen-sae-interp, and is not yet wired into the UI (see
[`BOUNDARY.md`](BOUNDARY.md) for what that means and why). To re-run the
conformance check against this repository's own copy:

```bash
python - <<'PY'
import importlib.util, json
from pathlib import Path
runner_path = Path("provenance/runtime_extractions/concept_bundle/concept_bundle_conformance.py")
spec = importlib.util.spec_from_file_location("concept_bundle_conformance", runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
pack = json.loads(Path("provenance/runtime_extractions/concept_bundle/vectors.json").read_text())
failures = runner.verify_pack(pack, package="sae_concept_lab.canonical.concept_bundle")
print(f"{len(pack['vectors'])} vectors, {len(failures)} failures")
PY
```

Or, from a qwen-sae-interp checkout at commit `cdae9c7` (this repository
must be installed and importable from that Python environment):

```bash
python scripts/concept_bundle_conformance.py --check --package sae_concept_lab.canonical.concept_bundle
```

## Provenance

Every extraction into this repository -- the UI import from commit
`9a9f3b7` and the concept-bundle contract extraction from commit
`cdae9c7` -- is recorded in
[`provenance/source_import.json`](provenance/source_import.json): source
repository identity, source commit(s), the exact source-to-destination
path mapping, a SHA-256 per imported file, the import timestamp, and an
explicit statement of what was and was not imported.

To verify that recorded provenance against a live `qwen-sae-interp`
checkout (read-only -- this never modifies that checkout):

```bash
python -m provenance.verify_provenance --qwen-sae-interp-checkout /path/to/qwen-sae-interp
```

This fails loudly if any imported file is missing, has been modified
since import, or exists under an imported path without being recorded in
the manifest -- checked across every extraction in the manifest, not just
one. See [`provenance/verify_provenance.py`](provenance/verify_provenance.py)
for exactly what it checks.

## Dependencies

- `gradio>=6.22,<7` (runtime, for the UI)
- `pytest>=7` (test, optional extra `test`)

The extracted concept-bundle contract (`sae_concept_lab/canonical/`) adds
no dependency of its own -- standard library only. No GPU, no real model
weights, and no scientific dependency (`interplab`, `sae_lens`,
`transformer_lens`, etc.) anywhere in this repository's import graph.
