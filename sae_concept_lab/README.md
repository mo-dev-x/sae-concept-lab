# SAE Concept Lab (working title) -- UI build, wired to the canonical contract

Gradio 6.22 public-workflow demo. Runs entirely on CPU: both tabs are
backed by `StubConceptLabBackend` (deterministic, GPU-free) and this
product's FAKE-provenance canonical concept documents in `fixtures/`.
Nothing here talks to a real model or SAE. This is a UI/UX build for
review, not a scientific instrument.

Every control -- which concepts exist, which directions are available,
what a resolved dose is, what fingerprints an execution carries -- is
computed by `sae_concept_lab.canonical.concept_bundle`
(`sae_concept_lab/canonical/concept_bundle/`, mechanically extracted from
qwen-sae-interp; see `../BOUNDARY.md`). This package never re-implements
any of that: it loads named files, calls the canonical resolver, and
translates a `concept_id`/`pairing_id` into a display label -- the one
thing the canonical contract deliberately holds none of.

## Launch

```bash
pip install -e ".[test]"   # from the repository root
python -m sae_concept_lab.app
```

Then open the printed local URL (default `http://127.0.0.1:7860`).

`--mode release` is a fail-closed gate that checks, in order: (1)
whether the active backend is `StubConceptLabBackend` by type -- always
true in this build, so release always refuses here regardless of
anything else; (2) `--evidence-registry-root` (required in release mode;
refused if absent, missing, unreadable, or empty); (3) canonical
publishability (`release.select_layout_entries`) against a real
`RepositoryEvidenceRegistry` rooted there. This product repository wires
(1) and (2) -- the canonical package has no notion of "this product's
stub backend" or "where this deployment's registry lives" -- and never
duplicates (3):

```bash
python -m sae_concept_lab.app --mode release
# REFUSING TO LAUNCH: refusing --mode release: backend for model_key='gemma' is
# StubConceptLabBackend (deterministic fake data), regardless of any entry's
# provenance. ...
```

## Layout

```
sae_concept_lab/
  app.py                    CLI entry point, --mode dev|release gate,
                             --evidence-registry-root
  i18n.py                   FR|EN dictionary + t() lookup
  core/
    protocol.py             ConceptLabBackend Protocol, GenerationRequest/
                             Result -- resolved_config is a canonical
                             ResolvedControlState, never a product type
    logic.py                 pure, Gradio-free: reset rule, chat turns,
                              Compare invariant, Public/Advanced rendering
                              (delegates to ResolvedControlState.public_view()/
                              advanced_view()/execution_dict())
    stub_backend.py          StubConceptLabBackend -- deterministic, FAKE-tagged
  fixtures/
    loader.py                 load_entries()/build_registry(): named canonical
                               entry files -> ConceptRegistry (codec.load_entry_files,
                               never a directory scan); enforce_release_gate()
                               (backend type, evidence_registry_root, canonical
                               publishability)
    labels.py                  concept_id/pairing_id -> display label/description
                               (product-owned; the canonical contract holds none)
    gemma/*.json, qwen/*.json  8 canonical BundleEntry documents, provenance=fake
  ui/
    tab.py                    build_model_tab() -- the ONE shared component
                               tree, instantiated once per pairing; direction
                               choices come from calibrated_directions,
                               PROHIBITED/CAPABILITY_LIMIT surfaced via
                               check_direction_executable()
    app_ui.py                 build_demo(): banner, language switch,
                               explainer, both tabs
```

## What this build does NOT do (by design, not oversight)

- No real model or SAE weights anywhere on the import or runtime path.
- No real feature IDs, bundles, weights, calibration, or persona
  definitions -- every concept_id in `fixtures/gemma/*.json` and
  `fixtures/qwen/*.json` is invented and prefixed `FAKE-`, every entry's
  `provenance` is exactly `"fake"`, every numeric feature index/dose is a
  placeholder.
- No token streaming, no cross-layer runtime, no second intervention
  system in Advanced (Advanced reads the exact same canonical
  `ResolvedControlState` Public used -- see
  `core/logic.py:advanced_output_details`, which is `resolved.advanced_view()`
  plus `resolved.execution_dict()`, never a recomputation).
- No auth, persistence, or multi-user infrastructure.
- Public mode never renders a feature index, sae id, unit, or authored
  value -- see `core/logic.py:public_output_summary`, sourced entirely
  from `ResolvedControlState.public_view()`'s own allow-list (model,
  concept, direction, strength, available/unavailable directions only).
- Positions is never a live, user-selectable control in either mode:
  `positions` is read from the canonical entry (`entry.positions`) and
  Advanced displays it read-only (`core/logic.py:advanced_positions_text`).
  An ATTESTED entry's own ratified position is always authoritative and is
  never overridden. Per the 2026-08-13 researcher ruling on public
  positions, this repository's own non-ATTESTED (FAKE) fixtures all
  default to `all` -- a fixture-authoring choice recorded in each entry's
  own JSON, not a default any code path applies at resolution time (there
  is no per-model or otherwise hidden default anywhere in this code).
  GENERATED_ONLY remains fully available (Advanced can render an entry
  ratified that way) and always surfaces the fixed disclosure "GENERATED_ONLY
  masks prefill and leaves the first generated token unaffected." next to
  the resolved mode -- see `tests/test_sae_concept_lab_public_vs_advanced.py`.
- Two of the eight shipped fixtures are deliberately not fully
  executable, to give the PROHIBITED/CAPABILITY_LIMIT surfacing a real
  case rather than only a synthetic vector: `gemma/enthusiasm.json`'s
  amplify direction spans two layers (CAPABILITY_LIMIT, runtime v1 caps
  execution at one `(sae_id, layer)` group per pass), and
  `qwen/directness.json`'s amplify direction names two SAEs at one layer
  (PROHIBITED, composing two reconstructions of one residual stream is
  undefined). Both are still schema-valid and still calibrated; sending a
  message on the affected direction is refused with the canonical
  `CapabilityReport`'s own reason, verbatim, not silently degraded to a
  subset of targets. `gemma/caution.json` and `qwen/skepticism.json` each
  calibrate only one direction, for the disabled-control case.
- No "continue anyway" override anywhere. An earlier draft of this build
  had an Advanced-only checkbox that let a settings change keep stale
  chat history instead of resetting it -- removed as part of the P0
  release-safety correction, because it was a second, divergent
  intervention behavior available only in Advanced. Every
  concept/direction/strength change now unconditionally clears the
  conversation and shows the localized reset notice, with no exceptions
  in either mode.

## Known Gradio limitations hit during this build

- `gr.Chatbot` in the installed 6.22 no longer takes a `type=` argument;
  message-format dicts (`{"role": ..., "content": ...}`) are the only
  supported `value` shape now, so there is nothing to configure there.
- Retranslating a component on language change is done by returning a
  fresh `gr.<Component>(...)` instance with only the changed kwargs set
  (e.g. `gr.Radio(choices=new_choices, label=new_label)`, `value`
  omitted) -- omitted kwargs are confirmed (by direct inspection of the
  constructed update object, not assumed) to leave the component's
  current live value alone. This is the documented Gradio 6 update
  idiom, not a workaround, but it is easy to get wrong by
  over-specifying `value=` and accidentally resetting a user's selection
  on every language switch -- worth flagging for review.
- `gr.Dataset` "cards" support relabeling via a fresh `samples=` list on
  language change, but the component has no persistent `value` to
  preserve across relabeling the way `gr.Radio` does (selection is
  transient, delivered only via the click event) -- there is nothing to
  regress here, just noting the two components behave differently.
- `demo.get_config_file()` always reports `gr.State`'s initial `value` as
  `None` in the exported JSON, even when the component was constructed
  with a real default (confirmed directly: `gr.State(Selection(...))`
  still serializes to `{"value": null}`). The Python-side `.value`
  attribute on the live component object IS correct (readable via
  `demo.blocks[id].value` or `blockfn.inputs[i].value`) -- only the
  static config export masks it. Don't use `get_config_file()` to assert
  anything about a State's default; assert against the component object
  directly, or against `blockfn.inputs`.

## Two real bugs this build caught only by actually clicking through the running app, not from the test suite alone

Both are fixed and now covered by a regression test, but are worth
naming explicitly since they are exactly the class of bug a
component-level or pure-logic test suite structurally cannot see:

1. **The reset rule silently didn't fire on a user's first-ever settings
   change.** `selection_state` was seeded `gr.State(None)`, and
   `apply_selection_change` correctly treats `previous_selection=None` as
   "nothing recorded yet, don't reset an empty conversation on page
   load." But that same branch also swallowed the very first REAL change
   a user made if they touched Direction or Strength before ever
   clicking a concept card -- there was no prior recorded selection to
   compare against, so a genuine change looked identical to page load.
   Every unit test passed throughout, because every unit test supplied
   an explicit `previous_selection`; only driving the actual page (send
   a message, then click "High" with no prior concept click) surfaced
   it. Fixed by seeding `selection_state` with the page's own default
   selection instead of `None`
   (`test_first_ever_direction_or_strength_change_still_resets_using_the_apps_own_live_defaults`
   now regression-tests this using Gradio's own live-wired default
   values, not hand-constructed ones).
2. **Two Compare-panel labels and the Advanced "Seed"/"Positions" labels
   never retranslated on language switch.** Caught by literally reading a
   French screenshot and noticing "Original (no concept applied)" was
   still in English. Root cause: those four components were built with a
   localized initial string but never appended to the `relang` registry
   `ui/app_ui.py`'s language-switch handler iterates -- an omission a
   pure i18n-dictionary test (every key has an en/fr pair) cannot detect,
   because the dictionary itself was complete; only the wiring that
   consumes it was incomplete. A companion gap in the same area (the
   concept-detail panel and the output-summary panel not refreshing
   because their text depends on more than just `lang`) was fixed with
   two dedicated listeners rather than forced into the generic registry.

## P0 release-safety correction (successor commit)

The initial build satisfied the UI brief but had two product/release
invariants that needed correcting before this could be called anything
more than a development preview:

1. **The release gate only checked bundle flags.** `enforce_release_gate()`
   now also requires a `backend` argument and refuses outright if it is
   `StubConceptLabBackend`, independent of what a bundle's JSON says.
   Regression: `test_release_mode_raises_on_stub_backend_even_with_an_entirely_clean_bundle`
   and its end-to-end `app.main()` counterpart build an `is_synthetic:
   false, release_blocked: false` bundle from scratch and confirm release
   mode still refuses.
2. **The Advanced "continue anyway" override let a settings change skip
   the reset**, which made Advanced a second, divergent intervention
   system in exactly the way the original brief prohibited. Removed
   entirely -- the checkbox, the `continue_anyway` parameter on
   `apply_selection_change()`, and the `continue_anyway_*` i18n strings.
   Concept/direction/strength changes now unconditionally reset, in both
   modes, with no escape hatch.

Two smaller correctness fixes rode along with this pass:

3. `resolve_config()`'s `diagnostics["synthetic"]` was a hardcoded `True`
   regardless of the bundle's actual `is_synthetic` flag -- a future real
   bundle/adapter setting `is_synthetic=False` would have kept reporting
   `synthetic=True` in diagnostics forever. Both fields are now derived
   from the same single value.
4. `load_bundle()` gained basic fail-closed structural validation beyond
   "the required keys are present": `is_synthetic`/`release_blocked` must
   actually be booleans (not truthy strings), `model_key` must be a known
   value, `positions_default` must be a supported `PositionsMode`, and
   `seed_default` must be a non-negative int.

## Unresolved integration questions (for Engineer A / next reviewer)

1. **Resolved**, as of the Engineer 4 P0 dispatch: the concept-bundle
   contract landed on `eng3/concept-bundle`, was frozen behind a
   conformance pack, mechanically extracted into
   `sae_concept_lab/canonical/concept_bundle/`, and is now wired into
   `fixtures/loader.py`/`core/logic.py`/`ui/tab.py` directly.
   **Also resolved**, as of the dual-runtime integration: `core/qwen_backend.py`
   and `core/gemma_backend.py` are REAL `ConceptLabBackend` implementations
   -- wired to mechanically-extracted (`sae_concept_lab/extracted_runtime/`)
   and mechanically-accepted (`core/runtime_acceptance.py`) intervention
   code for both pairings. Selected via `--qwen-backend runtime`/
   `--gemma-backend runtime` (`app.py`); `--mode release` still refuses
   regardless, since no shipped concept is ATTESTED. **Also resolved**, as
   of the Tamia product-integration smoke packet:
   `sae_concept_lab/smoke/tamia_smoke.py` exercises both real backends
   through the exact canonical resolution -> execution-guard -> backend
   path the application uses, on real Tamia weights -- see
   `../docs/tamia_smoke.md`. **Also resolved**, as of the PI-demo
   dispatch (2026-08-13): a bounded Mode-A import slot
   (`fixtures/attested/{gemma,qwen}/`, `fixtures/loader.py`'s
   `load_attested_entries`) lets a genuinely ATTESTED bundle reach
   `--mode release` with no `.py` edit, `--mode release` now filters
   rendered entries to the publishable subset
   (`canonical.select_layout_entries`), and a local, GPU-free preflight
   (`sae_concept_lab/smoke/pi_demo_preflight.py`) checks the whole stack
   end to end -- see `../docs/demo_runbook.md` and
   `../docs/pi_demo_scientific_status.md`.
2. Compare is currently a non-committing side-by-side probe: sending the
   same message via "Compare" does NOT append either response to the
   running chat history (only "Send" does). Unclear whether a future
   version should let the user "keep" one of the two Compare arms into
   the main conversation -- not implemented, flagged rather than guessed.
3. Advanced's seed control is live and feeds the next generation
   (StubConceptLabBackend's digest input only -- not a canonical field),
   but there is no per-turn record of what seed produced a PAST message
   once the conversation has moved on -- the resolved `ResolvedControlState`
   is only kept for the MOST RECENT turn (`resolved_config_state`), not
   per-message history. Whether every past turn needs its own recoverable
   resolved state is a product question, not resolved here. Positions is
   no longer a live control at all -- it is read from the canonical entry
   and displayed, never chosen.
4. The permanent FAKE banner and the "How does this work?" explainer are
   global (above both tabs), not per-tab -- if Gemma and Qwen ever need
   materially different disclaimers (e.g. different real-model caveats
   once wired to something real), the banner will need to become
   per-tab.
5. `evidence_registry_root`'s fail-closed pre-flight (absent/missing/
   unreadable/empty) is this product's own addition; canonical
   `RepositoryEvidenceRegistry.resolve()` reads a registry record's bytes
   and recomputes its content digest (sha256 over canonical JSON,
   `self_hash` excluded), comparing that recomputation to the reference --
   see `../BOUNDARY.md` for why this was, until the final evidence
   contract (qwen-sae-interp checkout `3a9c153`, frozen pack `2a95a49`), a
   documented canonical-source defect (the earlier pack verified only a
   record's own self-declared `self_hash` field, never independent
   content). `fixtures/loader.py::enforce_release_gate` prints the
   canonical package's own mandatory release wording
   (`release.RELEASE_EVIDENCE_STATEMENT`) and per-reference note
   (`ReleaseDecision.render_release_evidence_note()`) verbatim, never a
   product-composed paraphrase of what was checked.
