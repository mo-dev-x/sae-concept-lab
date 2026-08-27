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
commit `3a9c153` and certified against the frozen 75-vector conformance
pack current as of this extraction (frozen pack content `2a95a49`). This
mirror has been deliberately re-extracted once already, superseding an
earlier mirror at checkout `cdae9c7` / frozen pack `fabf702` -- see
`provenance/source_import.json`'s `concept_bundle_contract` entry (in
particular its `supersedes_previous_extraction` field) for the full
source-to-destination mapping, hash table, and the reason for the
replacement, and `provenance/runtime_extractions/concept_bundle/` for the
copied vectors, export inventory, and check-mode runner. The superseded
extraction's own record remains readable in this repository's git history
at commit `d7e4577` and earlier -- replacing the manifest's *current*
record is not the same as erasing the *prior* one.

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

The `ConceptLabBackend` Protocol (`core/protocol.py`) is unchanged in
shape and purpose: it is still the seam between this UI and whatever
generates chat text. As of the dual-runtime integration below, it now has
THREE implementations: `StubConceptLabBackend` (always available,
deterministic, GPU-free) and two real backends,
`sae_concept_lab.core.qwen_backend.QwenRuntimeBackend` and
`sae_concept_lab.core.gemma_backend.GemmaRuntimeBackend`, each wired to
its own extracted intervention loader/hook mechanism -- see "Runtime
backends" below.

## Runtime backends: extracted intervention code, mechanical acceptance as a SEPARATE gate

`sae_concept_lab/extracted_runtime/` mirrors the MINIMUM runtime surface
needed to run a real intervention: `targets.py` (pure-stdlib identity/
validation, whole-file), `hooks.py` (the shared clamp/ablate hook
mechanism, five named functions), `diagnostics.py` (shared trace/verdict
functions, five named functions), `qwen_loader.py` (Qwen3.5-27B raw-HF
loader), `gemma_loader.py` (gemma-3-12b-it transformer_lens/sae_lens
loader) -- extracted from qwen-sae-interp's `scripts/legacy/
final_pairing_harness.py`, `scripts/legacy/final_pairing_targets.py`, and
`interplab/interventions/hooks.py`. This is CODE ONLY, under a new,
third `extraction_class`, `RUNTIME_CODE_MIRROR` (see below) -- it carries
**no claim whatsoever** that either pairing's intervention mechanism has
been proven against real weights. `sae_concept_lab/core/qwen_backend.py`
and `gemma_backend.py` are the product-owned backends that translate a
canonical `ResolvedControlState` into calls on this extracted code,
lazily (no torch/transformers/transformer_lens/sae_lens import happens
until a real backend's `generate()` actually runs), with their own
defensive same-layer/multi-SAE/cross-layer re-check
(`core/execution_guard.py`, reusing canonical's own
`MultipleSaeIdentitiesAtLayerError`/`MultipleExecutionGroupsError`
classes, never a re-implementation).

**Mechanical acceptance against real weights** -- whether the extracted
CODE has actually been proven to move a real residual stream -- is
`sae_concept_lab/core/runtime_acceptance.py`'s entirely separate concern,
checked independently by the release gate and by each backend's own
honesty tag. `import_acceptance_from_evidence_commit()` is the ONLY way a
pairing becomes accepted: it performs the full bounded adjudication --
`git show` every named artifact at a caller-supplied evidence commit in a
real qwen-sae-interp checkout, independently recompute its SHA-256, and
raise (never silently continue) on any mismatch. Both pairings are
currently ACCEPTED, imported from qwen-sae-interp evidence commit
`b6d598b` (`results/final_pairing/job_406092/` for Qwen -- ALL and
GENERATED_ONLY scenarios only, that job's Gemma scenarios failed and are
explicitly excluded from the claim; `results/final_pairing/job_407008/`
for Gemma, both scenarios). Mechanical acceptance of the intervention
MECHANISM and public release of a SCIENTIFIC CONCEPT
(`sae_concept_lab.canonical.concept_bundle.release`) are, and must
remain, two separate gates: this repository ships no ATTESTED concepts,
so `--mode release` refuses regardless of which backend is selected or
whether its pairing is mechanically accepted.

### Three evidence claims, two rejected

Across this integration's dispatch history, three claims were made that a
Qwen and/or Gemma job had mechanically passed. The first two were
REJECTED before anything was written into this repository's provenance,
on grounds this repository's own discipline (mechanically verify against
the tracked source, never trust a pasted assertion) already enforced
elsewhere:

1. A claim that qwen-sae-interp job 406092 was a Qwen mechanical pass.
   qwen-sae-interp's own tracked `docs/final_pairing_tamia_packet.md`, at
   the time, stated the opposite repeatedly and explicitly ("the entire
   Qwen raw-HF path has never run against real Qwen3.5-27B weights") --
   job 406092 was documented there as a Gemma-only finding. Rejected.
2. A claim that Gemma job "407008" had passed, naming qwen-sae-interp
   commit `de3b499` as the source. That commit's actual content is an
   unrelated pytest-removal refactor ("Replace the pytest-based Tamia
   symlink preflight with a standalone, standard-library-only script") --
   no job 407008 existed anywhere in the tracked repository at that
   point. Rejected.
3. A claim citing qwen-sae-interp commit `b6d598b` ("Import and adjudicate
   sealed final-pairing evidence: job 407008 (Gemma pass) and job 406092
   (mixed: Gemma failed, Qwen mechanical pass)"), with
   `results/final_pairing/job_406092/` and `.../job_407008/` as tracked
   evidence trees, each with a `chain_of_custody.json` manifest,
   `inventory.json` hash table, and a bounded README. ACCEPTED, after: (a)
   `git show` confirmed the commit exists and its message matches; (b)
   every named artifact was independently re-hashed against the committed
   tree via `import_acceptance_from_evidence_commit`, which raises rather
   than continues on any mismatch; (c) qwen-sae-interp's own
   `tests/test_final_pairing_evidence_record.py` (29 tests, including
   cross-contamination guards proving the Qwen-pass and Gemma-pass claims
   cannot leak into each other's job) was read and run for real against
   that checkout, and passed.

The lesson this history is kept for: a job number, a commit hash, and a
plausible-looking log are not evidence on their own. Evidence is a
git-show-verifiable commit whose content survives independent re-hashing
and re-running -- attempts 1 and 2 had neither; attempt 3 had both.

## Positions: public default is ALL, ATTESTED ratification is authoritative

`BundleEntry.positions` (canonical, `schema.py`) has never been a
user-selectable control in either mode -- it is authored per entry and
Advanced renders it read-only (`core/logic.py:advanced_positions_text`).
Until the 2026-08-13 researcher ruling on public positions, this
repository deliberately left no product-chosen default for it at all
(see the prior "keep public default configuration-driven until
researcher ratifies ALL" note this replaces). That ruling is now in
force:

- **Public positions default: ALL.** This repository's own eight shipped
  FAKE fixtures (`fixtures/{gemma,qwen}/*.json`) were re-authored from
  `generated_only` to `all` accordingly -- none of them carries
  ATTESTED-level ratification for `generated_only`, so none of them had
  any standing to use it.
- **An explicit, researcher-ratified ATTESTED bundle position remains
  authoritative.** Nothing in `resolve_control`, `advanced_positions_text`,
  or either backend overrides `entry.positions` -- an ATTESTED entry's own
  value is read and used exactly as authored, regardless of what the
  public default is for everything else.
- **GENERATED_ONLY remains fully available**, in Advanced, on both
  pairings, and always surfaces the fixed disclosure "GENERATED_ONLY
  masks prefill and leaves the first generated token unaffected." next to
  the resolved mode (`core/logic.py:GENERATED_ONLY_POSITIONS_DISCLOSURE`)
  -- distinct from, and not a replacement for, `core/qwen_backend.py`'s/
  `core/gemma_backend.py`'s own longer `GENERATED_ONLY_FIRST_TOKEN_DISCLOSURE`
  (a backend diagnostics-log statement quoted verbatim from
  qwen-sae-interp's own docs; both backends now carry it identically --
  previously only the Qwen backend did, a parity gap closed alongside
  this ruling).
- **No hidden model-specific default exists.** `build_model_tab` is the
  one shared component tree instantiated per pairing; `advanced_positions_text`
  is the one function both tabs call to render positions. There is no
  second, per-pairing copy of this logic to drift.

## Bounded Mode-A import slot and the PI-demo preflight (2026-08-13)

`sae_concept_lab/fixtures/attested/{gemma,qwen}/` is the ONE location a
genuinely ATTESTED bundle can be dropped into and be picked up by
`fixtures.loader.load_entries(model_key)` with **no edit to any `.py`
file** -- see `fixtures/attested/README.md`, `fixtures/loader.py`'s
`load_attested_entries`/`ATTESTED_DIR`, and
`tests/test_pi_demo_mode_a.py`. A directory scan here is safe in a way it
deliberately is not for `fixtures/{gemma,qwen}/` (whose explicit
`_ENTRY_FILENAMES` list is unchanged): whether a scanned file's entry ever
publishes remains entirely `evaluate_publishability`'s decision, never
this slot's, and a file that fails to even decode is excluded and
reported (`AttestedImportOutcome.rejected`) rather than raised --
Mode B's guarantee (the shipped FAKE fixtures always load) does not
depend on whatever the slot currently holds or how broken it is.

`sae_concept_lab/app.py`'s `--mode release` path now additionally filters
the entries it passes to `build_demo` through canonical's own
`select_layout_entries(..., exposure=Exposure.RELEASE)` before rendering
-- closing a real gap where a release build would otherwise have shown
every shipped FAKE fixture indistinguishably alongside a genuinely
publishable entry, once one existed. `ui/app_ui.build_demo` takes a new
`mode` parameter that ONLY controls whether the permanent "PLACEHOLDER /
NOT SCIENTIFIC EVIDENCE" banner renders (omitted under `mode="release"`,
since it would misdescribe a build now filtered to publishable-only
content); it never changes any other rendering or behavior, and dev
mode's own long-standing behavior (unfiltered, banner always present) is
unchanged (`tests/test_pi_demo_mode_a.py`).

`sae_concept_lab/smoke/pi_demo_preflight.py` is a separate, local,
GPU-free, D:-only preflight (unrelated to the Tamia GPU smoke packet
below): required files, no C:-based path anywhere, current
release-eligibility status (Mode A vs Mode B, informational), release
still refuses locally with stub backends (hard assertion -- a real
backend is a separate, GPU-side precondition this preflight cannot and
does not satisfy), boot/HTTP-200/visible-status/clean-shutdown of the
real dev-mode app on loopback, and one aggregate machine-readable JSON
result. See `_attic/docs/demo_runbook.md` (the operational script, Mode A/B
branches, five-minute walkthrough, recovery) and
`_attic/docs/pi_demo_scientific_status.md` (exact permitted/prohibited claims,
current ATTESTED count, the working-instrument-vs-validated-science
distinction) -- both archived as internal PI-demo material, kept for the
record rather than deleted.

## Tamia product-integration smoke packet

`sae_concept_lab/smoke/tamia_smoke.py` is a reproducible, fail-closed
smoke runner -- NOT a second scientific acceptance harness -- that proves
the extracted, mechanically-accepted runtime backends actually run,
through the real product adapters (`core/logic.py:send_message`,
`execution_guard`'s defensive re-check inside each backend,
`ui/app_ui.build_demo`), on real weights on Tamia. It never calls
`extracted_runtime.qwen_loader`/`gemma_loader`/`hooks` directly to drive a
generation (the one exception, proving the loader's own identity guard
rather than bypassing it, is documented in `_attic/docs/tamia_smoke.md`,
archived internal material). Every
concept-bundle entry it resolves against (`sae_concept_lab/smoke/entries.py`)
is built directly in Python, prefixed `ENGINEERING-ONLY-SMOKE-`, and
`provenance=FAKE` -- never added to `fixtures.loader._ENTRY_FILENAMES`, so
it can never enter fixture discovery, the Gradio UI, or the release gate.
See `_attic/docs/tamia_smoke.md` (archived) for the full scenario list, exact
Tamia submission command, expected artifacts, and failure classification.

## extraction_class: a code-provenance axis, never the scientific-content axis

`provenance/source_import.json` tags every extraction with an
`extraction_class` of `HISTORICAL_SEED`, `CANONICAL_MIRROR`, or
`RUNTIME_CODE_MIRROR`. This is a CODE axis -- how a file entered this
repository and how strictly its bytes must stay fixed -- and it is
deliberately kept in a different field, with a different vocabulary, from
the SCIENTIFIC `provenance` field (`Provenance.ATTESTED` / `CANDIDATE` /
`DRAFT` / `FAKE` / `UNKNOWN`, defined in
`sae_concept_lab/canonical/concept_bundle/schema.py`). The two must never
be confused: a `BundleEntry`'s `provenance` says how well established a
CONCEPT's origin is; an extraction's `extraction_class` says how strictly
a FILE's bytes must be verified. Nothing in this repository's provenance
tooling reads or writes the word `provenance` as a code-extraction field
or verdict.

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
  re-running every frozen conformance vector (75, as of the currently
  mirrored pack), printing:
  `CANONICAL_MIRROR 2a95a49 current bytes match canonical source; conformance vectors pass`.
  The short commit named in this verdict is always the CURRENTLY mirrored
  frozen pack's commit -- it changes on a deliberate re-extraction (this
  repository has had one: `fabf702` superseded by `2a95a49`), which is a
  property of the pack being replaced whole, never a partial drift.
- **RUNTIME_CODE_MIRROR** (`shared_runtime_mirror`, `qwen_runtime_mirror`,
  `gemma_runtime_mirror`): byte-for-byte immutable like CANONICAL_MIRROR,
  but for extracted RUNTIME code with no frozen conformance pack of its
  own -- verified by hash alone, at whole-file granularity
  (`imported_modules`, e.g. `targets.py`) or per-function granularity
  (`imported_functions`, AST-extracted and independently re-hashed against
  both the source commit and the destination file -- needed because the
  source scripts mix Qwen/Gemma/CLI code in ways that don't separate
  cleanly at the file level), printing e.g.:
  `RUNTIME_CODE_MIRROR e63b08e current bytes match source; no conformance pack applies to this extraction`.
  This class makes **no claim** that the mirrored code has been
  mechanically verified against real weights -- that is
  `sae_concept_lab.core.runtime_acceptance`'s entirely separate concern
  (see "Runtime backends" above).

`provenance/verify_provenance.py` rejects reclassification: a
HISTORICAL_SEED extraction naming a source_path that any
CANONICAL_MIRROR/RUNTIME_CODE_MIRROR extraction in this manifest records,
or that the frozen pack's own `export_inventory.json` lists under
`minimum_export_surface`, is a fatal configuration error, not a per-file
finding -- it would let an immutable-mirror-owned path escape strict
verification by being relabelled. New, genuinely product-native files
(e.g. `fixtures/labels.py`, the eight canonical fixture documents under
`fixtures/gemma/` and `fixtures/qwen/`) need no invented extraction class
at all -- they are simply not extractions.

**A real, caught-in-the-act example of why function-level extraction
matters**: the first attempt at `hooks.py` copied the whole source file
byte-for-byte, including `attach()` and everything only it needs -- which
imports `interplab.interventions.spec.InterventionSpec`, an internal
qwen-sae-interp package. A whole-file mirror of that source is therefore
**structurally unable to ever import standalone in this repository**, not
merely hard to test -- discovered by a test failure
(`ModuleNotFoundError: No module named 'interplab'`), corrected before
being relied on anywhere, by re-extracting only the five functions
`_make_clamp_hook` actually needs. See `provenance/source_import.json`'s
`shared_runtime_mirror.excluded_from_extraction` for the full account.

## Reserved space for future runtime extraction

`provenance/runtime_extractions/` now holds one populated subdirectory,
`concept_bundle/` (the copied conformance pack described above), added by
the concept-bundle extraction. It otherwise remains a reserved location
for future extractions, added incrementally and never implicitly. See
`provenance/runtime_extractions/README.md`.
