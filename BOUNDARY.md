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

## The interim backend protocol is a non-canonical integration seam -- UNCHANGED by the concept-bundle extraction

`sae_concept_lab/core/protocol.py` defines `ConceptLabBackend`, the
`Protocol` this UI's core/ui code depends on instead of any concrete
backend class. At the time this UI was built (and at the time it was
first extracted here), no bundle/resolution contract existed anywhere in
qwen-sae-interp's history -- a branch reserved for one
(`eng3/concept-bundle`) carried zero commits beyond `main`.

Engineer 3 has since landed that contract on `eng3/concept-bundle` and
frozen it behind a conformance pack (see the next section). **This task
mechanically extracted and certified the contract; it did not wire it
in.** `core/protocol.py` is still what `sae_concept_lab/ui/` and
`sae_concept_lab/core/logic.py` depend on, unchanged, and every backend
this repository instantiates today is still `StubConceptLabBackend`:
deterministic, GPU-free, and tagged `[FAKE STUB -- UI TEST ONLY]` on
every response it produces. See `sae_concept_lab/README.md` for the
fail-closed `--mode release` gate that refuses to launch against it no
matter what a fixture bundle's JSON claims.

Wiring the extracted contract into the UI -- implementing
`ConceptLabBackend` against `sae_concept_lab.canonical.concept_bundle`,
or replacing `core/protocol.py` with a thin adapter to it -- is a
**subsequent, bounded task**, deliberately not performed here. Nothing in
`ui/` or `core/logic.py` should need to change either way when it
happens; that is the entire purpose of routing everything through the
Protocol boundary instead of a concrete class.

## The concept-bundle contract: extracted and certified, not yet wired

`sae_concept_lab/canonical/concept_bundle/` is a second, independent
extraction: the eight-module minimum runtime surface of the
concept-bundle contract (schema, codec, typed refusals, execution
grouping, resolution arithmetic, evidence-reference resolution, and the
fail-closed publication gate), mechanically copied byte-for-byte from
qwen-sae-interp's `interplab/concept_bundle/` at commit `cdae9c7` and
certified against Engineer 3's frozen 50-vector conformance pack. See
`provenance/source_import.json`'s `concept_bundle_contract` entry for the
full source-to-destination mapping and hash table, and
`provenance/runtime_extractions/concept_bundle/` for the copied vectors,
export inventory, and check-mode runner.

This is the CONTRACT only -- what a bundle entry is, what runtime v1 can
execute, how resolution and publication work -- never any scientific
concept, feature membership, calibration value, or evidence artifact.
`interplab/concept_bundle/fixtures.py` (invented data) was deliberately
not extracted, matching the canonical export inventory's own exclusion
list.

It is standard-library-only, has no import-time dependency on
qwen-sae-interp/interplab or any third-party package, and is entirely
disconnected from the UI's own bundle discovery and release gate
(`sae_concept_lab/fixtures/loader.py`) -- those two gates are independent
mechanisms guarding independent schemas, and neither can be satisfied by
the other's data (see `tests/test_concept_bundle_release_isolation.py`).
Nothing in `sae_concept_lab/ui/`, `sae_concept_lab/core/`, or
`sae_concept_lab/app.py` imports from `canonical/` yet. That wiring is
the subsequent bounded task named above.

## Reserved space for future runtime extraction

`provenance/runtime_extractions/` now holds one populated subdirectory,
`concept_bundle/` (the copied conformance pack described above), added by
the concept-bundle extraction. It otherwise remains a reserved location
for future extractions, added incrementally and never implicitly. See
`provenance/runtime_extractions/README.md`.
