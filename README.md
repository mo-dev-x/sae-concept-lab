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

## Provenance

Every file under `sae_concept_lab/` and every `tests/test_sae_concept_lab_*.py`
test in this repository was imported verbatim from a single commit of
`qwen-sae-interp` (`9a9f3b7`, branch `sae-concept-lab`). The exact mapping,
a SHA-256 per imported file, the import timestamp, and an explicit
no-scientific-runtime-was-imported statement are recorded in
[`provenance/source_import.json`](provenance/source_import.json).

To verify that recorded provenance against a live `qwen-sae-interp`
checkout (read-only -- this never modifies that checkout):

```bash
python -m provenance.verify_provenance --qwen-sae-interp-checkout /path/to/qwen-sae-interp
```

This fails loudly if any imported file is missing, has been modified
since import, or exists under an imported path without being recorded in
the manifest. See [`provenance/verify_provenance.py`](provenance/verify_provenance.py)
for exactly what it checks.

## Dependencies

- `gradio>=6.22,<7` (runtime)
- `pytest>=7` (test, optional extra `test`)

No GPU, no real model weights, and no scientific dependency (`interplab`,
`sae_lens`, `transformer_lens`, etc.) anywhere in this repository's import
graph.
