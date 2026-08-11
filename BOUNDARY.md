# Repository boundary: sae-concept-lab vs. qwen-sae-interp

This repository is a standalone product build. It was created because the
researcher ruled SAE Concept Lab out of the scientific repository
(`qwen-sae-interp`) immediately, rather than letting it accrete inside it.
The two repositories have different owners, different audiences, and
different definitions of "correct" -- keeping them separate is the point,
not an inconvenience to route around.

## qwen-sae-interp owns

- Scientific definitions: what a feature/concept *is*, how it was
  discovered, and what evidence supports it.
- Experiments, calibration, and adjudication protocols (e.g. the
  composition/reserved-index barrier under `reports/`, `scripts/legacy/`).
- Target loaders and the canonical runtime path for real models and real
  SAEs (`interplab/**`, `scripts/legacy/gemma3_sweep.py`,
  `scripts/legacy/gemma3_tool.py`, and everything they depend on).
- Canonical runtime *behavior* -- if a real intervention needs to run
  against a real model, qwen-sae-interp is the only place that is allowed
  to define what "correct" means for that run.

qwen-sae-interp remains the sole source of truth for all of the above.
Nothing in this repository redefines, reimplements, or second-guesses it.

## sae-concept-lab owns

- The product UI: the Gradio Blocks app under `sae_concept_lab/ui/` and
  `sae_concept_lab/app.py`, its i18n strings, its layout, its release gate.
- The deployment adapter: whatever packaging, launch command, and (later)
  runtime wiring is needed to run this UI somewhere a reviewer or user can
  reach it.
- Only runtime that has been **explicitly extracted** here, one file at a
  time, with a recorded source commit and a verifiable hash -- never a
  blanket copy, and never anything reached by convenience.

## Extracted runtime is derivative, never authoritative

Anything under this repository -- including anything that later lands in
`provenance/runtime_extractions/` -- is a **copy at a point in time**,
imported under `provenance/source_import.json`'s recorded mapping. It is
derivative by construction:

- If qwen-sae-interp's scientific definition of a feature, a calibration
  bundle, or a loader changes, this repository does **not** automatically
  follow. Nothing here re-imports itself.
- If this repository's copy of an extracted file and qwen-sae-interp's
  current version of that file ever disagree, qwen-sae-interp is correct
  and this repository's copy is stale, not the other way around.
- This repository must never be treated as a fork that can drift into its
  own scientific claims. A UI string, a fixture label, or a demo value
  living here is never evidence of anything; only qwen-sae-interp's
  results are.

`provenance/verify_provenance.py` exists to make "is this copy still what
it claims to be" a mechanically checkable question against the recorded
source commit, rather than something anyone has to take on faith.

## The interim backend protocol is a non-canonical integration seam

`sae_concept_lab/core/protocol.py` defines `ConceptLabBackend`, the
`Protocol` this UI's core/ui code depends on instead of any concrete
backend class. At the time this UI was built (and at the time it was
extracted here), no bundle/resolution contract existed anywhere in
qwen-sae-interp's history -- a branch reserved for one
(`eng3/concept-bundle`) carried zero commits beyond `main`.

**This protocol is explicitly not canonical.** It is a placeholder seam
that let UI development proceed without waiting on qwen-sae-interp's real
contract. When that contract lands:

- Whoever wires in a real backend should implement `ConceptLabBackend`
  directly, or replace `core/protocol.py` with a thin adapter to the real
  contract.
- Nothing in `sae_concept_lab/ui/` or `sae_concept_lab/core/logic.py`
  should need to change either way -- that is the entire purpose of
  routing everything through the Protocol boundary instead of a concrete
  class.
- Until that day, every backend instantiated in this repository is
  `StubConceptLabBackend`: deterministic, GPU-free, and tagged
  `[FAKE STUB -- UI TEST ONLY]` on every response it produces. See
  `sae_concept_lab/README.md` for the fail-closed `--mode release` gate
  that refuses to launch against it no matter what a fixture bundle's
  JSON claims.

## Reserved space for future runtime extraction

`provenance/runtime_extractions/` exists as a location and is
intentionally empty at this repository's initial commit. Populating it
with real runtime code is a separate, future decision -- not something
this initial import performs implicitly. See
`provenance/runtime_extractions/README.md`.
