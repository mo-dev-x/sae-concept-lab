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

## The concept-bundle contract is extracted, certified, AND wired into the UI

`sae_concept_lab/canonical/concept_bundle/` is the eight-module minimum
runtime surface of the concept-bundle contract (schema, codec, typed
refusals, execution grouping, resolution arithmetic, evidence-reference
resolution, and the fail-closed publication gate), mechanically copied
byte-for-byte from qwen-sae-interp's `interplab/concept_bundle/` at
commit `cdae9c7` and certified against Engineer 3's frozen 50-vector
conformance pack. See `provenance/source_import.json`'s
`concept_bundle_contract` entry for the full source-to-destination
mapping and hash table, and `provenance/runtime_extractions/concept_bundle/`
for the copied vectors, export inventory, and check-mode runner.

This is the CONTRACT only -- what a bundle entry is, what runtime v1 can
execute, how resolution and publication work -- never any scientific
concept, feature membership, calibration value, or evidence artifact.
`interplab/concept_bundle/fixtures.py` (invented data) was deliberately
not extracted, matching the canonical export inventory's own exclusion
list. It is standard-library-only and has no import-time dependency on
qwen-sae-interp/interplab or any third-party package.

**Wired**, as of the Engineer 4 P0 dispatch that also introduced
`extraction_class` (below): `sae_concept_lab/fixtures/loader.py` loads
this product's FAKE concept documents through `codec.load_entry_files`
and enforces publishability through `release.select_layout_entries` /
`evaluate_publishability`, never through a product-owned schema or
validation of its own. `sae_concept_lab/core/logic.py` and
`sae_concept_lab/ui/tab.py` read a canonical `ResolvedControlState`
(`resolver.resolve_control`) directly -- `core/protocol.py`'s
`GenerationRequest`/`GenerationResult` now type their `resolved_config`
field as `ResolvedControlState`, and the product-only `ResolvedConfig`
dataclass that used to stand in for it is retired. The ONE thing this
product repository still adds on top is presentation: `fixtures/labels.py`
maps a `concept_id`/`pairing_id` to a display label and description, because
the canonical contract deliberately holds no display name for either (see
`schema.py`'s module note and `resolver.py`'s `public_view()` docstring).
Translating an id into a label is presentation, not scientific validation.

The interim `ConceptLabBackend` Protocol (`core/protocol.py`) is
unchanged in shape and purpose: it is still the seam between this UI and
whatever generates chat text (currently always `StubConceptLabBackend`).
Wiring a REAL chat backend -- one that actually intervenes on a model
using a resolved control state -- remains a separate, not-yet-performed
task; only the CONTROL/CONFIGURATION side (what a concept/direction/
strength resolves to) is wired to canonical now.

## extraction_class: a code-provenance axis, never the scientific-content axis

`provenance/source_import.json` tags every extraction with an
`extraction_class` of `HISTORICAL_SEED` or `CANONICAL_MIRROR`. This is a
CODE axis -- how a file entered this repository and how strictly its
bytes must stay fixed -- and it is deliberately kept in a different
field, with a different vocabulary, from the SCIENTIFIC `provenance` field
(`Provenance.ATTESTED` / `CANDIDATE` / `DRAFT` / `FAKE` / `UNKNOWN`,
defined in `sae_concept_lab/canonical/concept_bundle/schema.py`). The two
must never be confused: a `BundleEntry`'s `provenance` says how well
established a CONCEPT's origin is; an extraction's `extraction_class`
says how strictly a FILE's bytes must be verified. Nothing in this
repository's provenance tooling reads or writes the word `provenance` as
a code-extraction field or verdict.

- **HISTORICAL_SEED** (`sae_concept_lab_ui`): a past import whose current
  bytes are PERMITTED TO EVOLVE. `app.py`, `core/*`, `ui/*`,
  `fixtures/loader.py`, and `tests/test_sae_concept_lab_*.py` were
  imported at this repository's own commit `d1f5e3f` and have since
  evolved (this dispatch is exactly that evolution). Verified by reading
  git objects at `d1f5e3f` in THIS repository's own history -- never
  qwen-sae-interp, never current bytes -- and printing:
  `HISTORICAL_SEED d1f5e3f import faithful at import commit; current bytes not checked`.
- **CANONICAL_MIRROR** (`concept_bundle_contract`): a byte-for-byte
  mirror that may NEVER evolve. Verified by hash-comparing CURRENT bytes
  against both the manifest and a live qwen-sae-interp checkout, AND by
  re-running all 50 frozen conformance vectors, printing:
  `CANONICAL_MIRROR fabf702 current bytes match canonical source; conformance vectors pass`.

`provenance/verify_provenance.py` rejects reclassification: a
HISTORICAL_SEED extraction naming a source_path that the frozen pack's
own `export_inventory.json` lists under `minimum_export_surface` is a
fatal configuration error, not a per-file finding -- it would let a
canonical-mirror-owned path escape strict verification by being
relabelled. New, genuinely product-native files (e.g.
`fixtures/labels.py`, the eight canonical fixture documents under
`fixtures/gemma/` and `fixtures/qwen/`) need no invented extraction class
at all -- they are simply not extractions.

## Reserved space for future runtime extraction

`provenance/runtime_extractions/` now holds one populated subdirectory,
`concept_bundle/` (the copied conformance pack described above), added by
the concept-bundle extraction. It otherwise remains a reserved location
for future extractions, added incrementally and never implicitly. See
`provenance/runtime_extractions/README.md`.
