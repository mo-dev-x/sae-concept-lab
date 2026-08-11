# SAE Concept Lab (working title) -- UI-only build

Gradio 6.22 public-workflow demo. Runs entirely on CPU: both tabs are
backed by `StubConceptLabBackend` (deterministic, GPU-free) and two
FAKE-marked, `release_blocked: true` fixture bundles in `fixtures/`.
Nothing here talks to a real model or SAE. This is a UI/UX build for
review, not a scientific instrument.

## Launch

```bash
pip install gradio  # if not already present in your environment
python -m sae_concept_lab.app
```

Then open the printed local URL (default `http://127.0.0.1:7860`).

`--mode release` is a fail-closed gate that checks TWO independent
things, and refuses if either looks fake: (1) the bundle's own
`is_synthetic`/`release_blocked` flags, and (2) whether the active
backend is `StubConceptLabBackend` by type, regardless of what the
bundle's JSON claims. **Editing a bundle's JSON flags alone is not
sufficient to reach release mode** -- this build always constructs
`StubConceptLabBackend` for both tabs, so `--mode release` always refuses
here no matter how the fixture files are edited:

```bash
python -m sae_concept_lab.app --mode release
# REFUSING TO LAUNCH: refusing --mode release: backend for model_key='gemma' is
# StubConceptLabBackend (deterministic fake data), regardless of the bundle's
# is_synthetic/release_blocked flags. ...
```

This was a deliberate correction (P0, successor commit to the initial
build): the first version's gate only checked bundle flags, which meant
someone could have flipped `is_synthetic`/`release_blocked` to `false` in
a fixture JSON file and passed the gate while still serving
`[FAKE STUB -- UI TEST ONLY]` text. `enforce_release_gate()` now requires
a `backend` argument (no longer optional) and checks its type first.

## Layout

```
sae_concept_lab/
  app.py                    CLI entry point, --mode dev|release gate
  i18n.py                   FR|EN dictionary + t() lookup
  core/
    protocol.py             ConceptLabBackend Protocol, ResolvedConfig,
                             GenerationRequest/Result -- the seam a real
                             backend implements later
    config.py                resolve_config(): bundle + selections -> ResolvedConfig
    logic.py                 pure, Gradio-free: reset rule, chat turns,
                              Compare invariant, Public/Advanced rendering
    stub_backend.py          StubConceptLabBackend -- deterministic, FAKE-tagged
  fixtures/
    loader.py                 load_bundle() (structural + basic fail-closed
                               validation: boolean flags, known model_key,
                               supported positions_default, valid
                               seed_default), enforce_release_gate()
                               (bundle flags AND backend type)
    gemma_stub_bundle.json    4 invented concepts, is_synthetic/release_blocked
    qwen_stub_bundle.json     4 different invented concepts (same reasons)
  ui/
    tab.py                    build_model_tab() -- the ONE shared component
                               tree, instantiated once per model
    app_ui.py                 build_demo(): banner, language switch,
                               explainer, both tabs
```

## What this build does NOT do (by design, not oversight)

- No real model or SAE weights anywhere on the import or runtime path.
- No real feature IDs, bundles, weights, calibration, or persona
  definitions -- every id in `fixtures/*.json` is invented and prefixed
  `FAKE-`, every numeric layer/coefficient is a placeholder.
- No token streaming, no cross-layer runtime, no second intervention
  system in Advanced (Advanced reads the exact same `ResolvedConfig`
  Public used -- see `core/logic.py:advanced_output_details`).
- No auth, persistence, or multi-user infrastructure.
- Public mode never renders seed, feature id, sae id, positions, hook
  point, or coefficients -- see `core/logic.py:public_output_summary`
  for the literal allow-list (model, concept, direction, strength only).
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

1. No "bundle/resolution contract" existed anywhere in this repo's
   history at build time (confirmed by direct search) -- a worktree/branch
   reserved for it (`eng3/concept-bundle`) exists but carries zero commits
   beyond `main`. `core/protocol.py` is this build's own seam in the
   meantime; whoever lands the real contract should either implement
   `ConceptLabBackend` directly, or this file gets replaced by a thin
   adapter to it. Nothing in `ui/` or `core/logic.py` should need to change
   either way -- that is the point of the Protocol boundary.
2. Compare is currently a non-committing side-by-side probe: sending the
   same message via "Compare" does NOT append either response to the
   running chat history (only "Send" does). Unclear whether a future
   version should let the user "keep" one of the two Compare arms into
   the main conversation -- not implemented, flagged rather than guessed.
3. Advanced's seed/positions controls are live and feed the next
   generation, but there is no per-turn record of what seed/positions
   produced a PAST message once the conversation has moved on --
   `ResolvedConfig` is only kept for the MOST RECENT turn
   (`resolved_config_state`), not per-message history. Whether every
   past turn needs its own recoverable resolved state is a product
   question, not resolved here.
4. The permanent FAKE banner and the "How does this work?" explainer are
   global (above both tabs), not per-tab -- if Gemma and Qwen ever need
   materially different disclaimers (e.g. different real-model caveats
   once wired to something real), the banner will need to become
   per-tab.
