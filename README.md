# SAE Concept Lab (standalone product repository)

Standalone Gradio UI/stub product build, extracted from the
`qwen-sae-interp` scientific repository under recorded provenance. This
repository owns the product UI and its deployment adapter only; it does
not own, and must never be treated as, a scientific source of truth. See
[`BOUNDARY.md`](BOUNDARY.md) for the full repository-boundary statement.

By default, both model tabs are backed by `StubConceptLabBackend`
(deterministic, GPU-free, every response tagged
`[FAKE STUB -- UI TEST ONLY]`) and eight canonical `provenance: "fake"`
concept-bundle documents (four per pairing), loaded and resolved entirely
through
[`sae_concept_lab/canonical/concept_bundle/`](sae_concept_lab/canonical/concept_bundle/)
-- see below. `--qwen-backend runtime`/`--gemma-backend runtime` select a
REAL backend instead (`sae_concept_lab/core/qwen_backend.py` /
`gemma_backend.py`, wired to mechanically-extracted, mechanically-accepted
intervention code -- see "Runtime backends" below and
[`docs/tamia_launch.md`](docs/tamia_launch.md)). `--mode release` is a
fail-closed gate that refuses to launch on this build no matter which
backend is selected: every fixture this repository ships is
`provenance: "fake"`, never `attested`, so canonical publishability blocks
all of them regardless. See [`sae_concept_lab/README.md`](sae_concept_lab/README.md)
for the full detail on the UI, the release gate, and known Gradio
limitations.

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
python -m sae_concept_lab.app --mode release --evidence-registry-root /path/to/registry
```

This exits non-zero and never opens a server: with the default stub
backend, the backend-type check refuses first; even past that (or with a
real, mechanically-accepted backend selected -- see below),
`--evidence-registry-root` is validated fail-closed (refused if absent/
missing/unreadable/empty), and canonical publishability still blocks
every shipped fixture (all `provenance: "fake"`). See
[`sae_concept_lab/README.md`](sae_concept_lab/README.md) for why that is a
deliberate, structural refusal rather than a flag someone forgot to flip.

## Runtime backends (extracted intervention code, mechanically accepted)

`sae_concept_lab/extracted_runtime/` mirrors the minimum runtime surface
needed to run a real intervention against Qwen3.5-27B + Qwen-Scope or
gemma-3-12b-it + gemma-scope-2-12b-it-res, extracted from qwen-sae-interp
under a third `extraction_class`, `RUNTIME_CODE_MIRROR` (code only -- see
[`BOUNDARY.md`](BOUNDARY.md)). `sae_concept_lab/core/qwen_backend.py` /
`gemma_backend.py` translate a canonical `ResolvedControlState` into calls
on that code, lazily -- no torch/transformers/transformer_lens/sae_lens
import happens until a real backend's `generate()` actually runs.

Whether either pairing's intervention MECHANISM has been mechanically
proven against real weights is `sae_concept_lab/core/runtime_acceptance.py`'s
entirely separate concern from code extraction, checked independently by
the release gate. Both pairings are currently mechanically accepted (see
that module for the exact bounded claim and the qwen-sae-interp evidence
commit it was imported from) -- this is never a scientific or public-release
claim; see `BOUNDARY.md`'s "Runtime backends" section for the full account,
including two earlier acceptance claims this repository rejected before
accepting a third. See [`docs/tamia_launch.md`](docs/tamia_launch.md) for
exact launch commands and pinned paths/revisions.

## Canonical concept-bundle contract (extracted, certified, and wired into the UI)

`sae_concept_lab/canonical/concept_bundle/` carries the eight-module
concept-bundle contract, mechanically extracted from qwen-sae-interp and
certified against the frozen 75-vector conformance pack current as of
this extraction (this mirror has been deliberately re-extracted once
already -- see `provenance/source_import.json`'s `concept_bundle_contract`
entry). It is standard-library-only, has zero import-time dependency on
qwen-sae-interp, and every control this UI renders -- concept selection,
direction availability, resolved dose, execution payload, fingerprints --
is computed by this package directly (see
[`BOUNDARY.md`](BOUNDARY.md) for the full wiring statement and for
`extraction_class`, the code-provenance axis this repository uses to
track it). To re-run the conformance check against this repository's own
copy:

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

Or, from a qwen-sae-interp checkout at the commit named in
`provenance/source_import.json`'s `concept_bundle_contract.source_repository.checkout_commit`
(currently `3a9c153`; this repository must be installed and importable
from that Python environment):

```bash
python scripts/concept_bundle_conformance.py --check --package sae_concept_lab.canonical.concept_bundle
```

## Provenance

Every extraction into this repository -- the UI import from commit
`9a9f3b7` and the concept-bundle contract extraction currently mirroring
checkout `3a9c153` (deliberately re-extracted once already, superseding
an earlier mirror at `cdae9c7`) -- is recorded in
[`provenance/source_import.json`](provenance/source_import.json): source
repository identity, source commit(s), the exact source-to-destination
path mapping, a SHA-256 per imported file, the import timestamp, and an
explicit statement of what was and was not imported.

Each extraction also carries an `extraction_class` -- `HISTORICAL_SEED`
(a past import permitted to evolve, verified against this repository's
own frozen import commit), `CANONICAL_MIRROR` (a byte-for-byte mirror
that may never evolve, verified against current bytes AND every frozen
conformance vector), or `RUNTIME_CODE_MIRROR` (byte-for-byte immutable
extracted runtime code with no conformance pack of its own, verified by
hash alone at whole-file or per-function granularity -- see
`sae_concept_lab/extracted_runtime/`). This is a code-provenance axis,
entirely separate from the scientific `Provenance` field
(`ATTESTED`/`CANDIDATE`/`DRAFT`/`FAKE`/`UNKNOWN`) a `BundleEntry` carries
-- see [`BOUNDARY.md`](BOUNDARY.md) for the full statement.

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

The extracted concept-bundle contract (`sae_concept_lab/canonical/`) and
the extracted runtime code (`sae_concept_lab/extracted_runtime/`) add no
hard dependency of their own -- both import cleanly with neither torch
nor any other heavy package installed (proven directly: this repository's
own test suite runs, and passes in full, with no torch/transformers/
transformer_lens/sae_lens present). `torch`/`transformers` (Qwen) and
`torch`/`transformer_lens`/`sae_lens` (Gemma) are only ever imported
inside a real backend's `generate()` method, at the moment it actually
runs -- install them only if you intend to launch `--qwen-backend runtime`
or `--gemma-backend runtime` (see `docs/tamia_launch.md`). No scientific
dependency (`interplab` or any qwen-sae-interp-internal package) appears
anywhere in this repository's import graph, in any configuration.
