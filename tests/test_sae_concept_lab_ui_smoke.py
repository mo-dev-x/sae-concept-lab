"""End-to-end, CPU-only smoke checks against the real build_demo() config
-- no GPU, no real model weights, no browser. Complements the pure-logic
tests: this is the "does it actually assemble and render" layer, the
same role gemma3_tool.py's test_build_ui_header_renders_the_sample_max_
proxy_caveat test already plays in this repo."""

from __future__ import annotations

import json
from pathlib import Path

from sae_concept_lab.canonical.concept_bundle.codec import load_entry_file
from sae_concept_lab.core.gemma_backend import MECHANICALLY_UNVERIFIED_TAG
from sae_concept_lab.core.protocol import GenerationResult
from sae_concept_lab.core.scientific_identity import ENGINEERING_DEMONSTRATION_TAG
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.labels import concept_label
from sae_concept_lab.fixtures.loader import load_entries
from sae_concept_lab.i18n import t
from sae_concept_lab.ui.app_ui import build_demo

#: A test-owned entry of a KNOWN SHAPE (one calibrated direction) -- the
#: shipped set is not obliged to contain one, exactly
#: test_sae_concept_lab_config.py's own SHAPE_FIXTURES convention.
_TF = Path(__file__).resolve().parent / "fixtures"


def _build():
    gemma_entries = load_entries("gemma")
    qwen_entries = load_entries("qwen")
    demo = build_demo(
        gemma_entries=gemma_entries,
        qwen_entries=qwen_entries,
        gemma_backend=StubConceptLabBackend(),
        qwen_backend=StubConceptLabBackend(),
    )
    return demo, gemma_entries, qwen_entries


def test_app_builds_with_no_gpu_and_no_real_weights():
    demo, _gemma, _qwen = _build()
    cfg = demo.get_config_file()
    assert len(cfg["components"]) > 20


def test_both_model_tabs_are_present():
    demo, _gemma, _qwen = _build()
    cfg = demo.get_config_file()
    tab_labels = {c["props"].get("label") for c in cfg["components"] if c["type"] == "tabitem"}
    assert t("tab_gemma", "en") in tab_labels
    assert t("tab_qwen", "en") in tab_labels


def test_permanent_fake_banner_is_present_on_initial_render():
    demo, _gemma, _qwen = _build()
    rendered = json.dumps(demo.get_config_file(), default=str)
    assert "PLACEHOLDER" in rendered
    assert "NOT SCIENTIFIC EVIDENCE" in rendered


def test_explainer_text_is_present_on_initial_render():
    demo, _gemma, _qwen = _build()
    rendered = json.dumps(demo.get_config_file(), default=str)
    assert t("explainer_title", "en") in rendered


def test_advanced_accordion_present_and_starts_closed():
    demo, _gemma, _qwen = _build()
    cfg = demo.get_config_file()
    accordions = [c for c in cfg["components"] if c["type"] == "accordion"]
    advanced = [a for a in accordions if a["props"].get("label") == t("advanced_accordion_title", "en")]
    assert len(advanced) == 2  # one per tab
    for a in advanced:
        assert a["props"].get("open") is False


def test_concept_cards_are_present_for_both_models_and_render_the_shipped_concept():
    """The shipped set is now ONE real, measured concept
    ('pro-american-exceptionalism'), independently per pairing -- so both
    tabs render the SAME concept label, one card each, rather than the
    disjoint FAKE sets this test used to pin. What still matters is that
    each pairing's own tab actually renders its own shipped card."""
    demo, gemma_entries, qwen_entries = _build()
    cfg = demo.get_config_file()
    datasets = [c for c in cfg["components"] if c["type"] == "dataset"]
    assert len(datasets) == 2
    assert [len(ds["props"]["samples"]) for ds in datasets] == [1, 1]
    all_samples = [row for ds in datasets for row in ds["props"]["samples"]]
    gemma_labels = {concept_label(e.concept_id, "en") for e in gemma_entries}
    qwen_labels = {concept_label(e.concept_id, "en") for e in qwen_entries}
    expected_label = concept_label("pro-american-exceptionalism", "en")
    assert gemma_labels == qwen_labels == {expected_label}
    rendered_labels = {row[0] for row in all_samples}
    assert gemma_labels <= rendered_labels
    assert qwen_labels <= rendered_labels


def test_first_ever_direction_or_strength_change_still_resets_using_the_apps_own_live_defaults():
    """Regression test for a real bug caught only by clicking through the
    running app: selection_state used to be seeded gr.State(None), so the
    reset rule's "no previous selection recorded yet -> don't reset"
    branch (correct for page load) ALSO silently swallowed a user's very
    first Direction/Strength change if they never clicked a concept card
    first. Fixed by seeding selection_state with the page's own default
    selection instead of None. This test calls the REAL registered Gradio
    callback with the LIVE default values Gradio itself wired in
    (blockfn.inputs[i].value), not hand-constructed stand-ins -- it would
    have failed against the old gr.State(None) seed."""
    demo, _gemma, _qwen = _build()
    checked_any = False
    for blockfn in demo.fns.values():
        if blockfn.fn.__name__ != "_on_direction_or_strength_change":
            continue
        checked_any = True
        concept_id, direction, _strength, history, previous_selection, lang = [c.value for c in blockfn.inputs]
        assert previous_selection is not None, "selection_state must not be seeded None"
        new_history, chatbot_value, notice, _new_selection = blockfn.fn(
            concept_id, direction, "high", history, previous_selection, lang
        )
        assert new_history == [], "first-ever strength change from the page's own default must still reset"
        assert chatbot_value == []
        assert notice
    assert checked_any, "did not find any _on_direction_or_strength_change handler to test"


def test_advanced_accordion_has_no_continue_anyway_control():
    """P0 release-safety correction: the Advanced 'continue anyway'
    override was removed entirely -- concept/direction/strength changes
    must unconditionally clear history, with no escape hatch anywhere in
    the rendered component tree."""
    demo, _gemma, _qwen = _build()
    rendered = json.dumps(demo.get_config_file(), default=str)
    assert "continue_anyway" not in rendered.lower().replace(" ", "_")
    for c in demo.get_config_file()["components"]:
        if c["type"] == "checkbox":
            raise AssertionError(f"unexpected checkbox component in tree: {c['props']}")


def test_no_raw_technical_value_appears_outside_the_advanced_accordion_subtree():
    """Structural version of the public-vs-advanced test: walk the real
    rendered component tree and confirm every occurrence of a raw
    technical value (a feature index, an sae id) lives under an Advanced
    accordion, not in a top-level/public component."""
    demo, gemma_entries, _qwen = _build()
    cfg = demo.get_config_file()

    # Robust across Gradio config-shape versions: check the known raw
    # values only appear inside components whose type is one of the
    # Advanced-only leaf types (json, number-labelled "Seed") -- never
    # inside a markdown/button/chatbot/textbox component (the public
    # surface).
    disallowed_types = {"markdown", "button", "chatbot", "textbox"}
    first_entry = gemma_entries[0]
    first_direction = first_entry.calibrated_directions[0]
    first_target = first_entry.directions[first_direction].targets[0]
    for c in cfg["components"]:
        if c["type"] not in disallowed_types:
            continue
        rendered = json.dumps(c.get("props", {}))
        assert str(first_target.feature_idx) not in rendered, (
            f"raw feature_idx leaked into public component type={c['type']!r}"
        )
        assert first_target.sae_id not in rendered, f"raw sae_id leaked into public component type={c['type']!r}"


def test_one_direction_concept_removes_the_unavailable_choice_and_shows_the_exact_notice():
    """Acceptance case: a one-direction concept's unavailable control is
    not offered as a choice at all, and the exact canonical refusal
    message is rendered verbatim for it. The shipped set has no
    one-direction concept any more, so this is built from a test-owned
    entry of that KNOWN SHAPE (tests/fixtures/gemma/caution.json) rather
    than reading load_entries() -- and never by re-adding a one-direction
    concept to the product to satisfy this test."""
    caution = load_entry_file(_TF / "gemma" / "caution.json")
    assert caution.calibrated_directions == (caution.calibrated_directions[0],)
    demo = build_demo(
        gemma_entries=(caution,),
        qwen_entries=load_entries("qwen"),
        gemma_backend=StubConceptLabBackend(),
        qwen_backend=StubConceptLabBackend(),
    )

    click_fns = [bf for bf in demo.fns.values() if bf.fn.__name__ == "_on_concept_click"]
    fn = click_fns[0].fn  # gemma tab is registered first
    result = fn(0, "low", [], None, "en")
    _concept_id, _detail, direction_update, unavailable_notice, *_rest = result
    assert direction_update.constructor_args["choices"] == [("Amplify", "amplify")]
    assert "this direction is not calibrated for this concept on this model" in unavailable_notice


# ---------------------------------------------------------------------------
# Provenance labels moved OUT of the chat bubble and into their own persistent
# line. The backend still emits them -- that contract is asserted in
# tests/test_gemma_runtime_backend.py, test_qwen_runtime_backend.py and
# test_scientific_identity_gate.py, and none of those changed. What is checked
# here is the presentation: the reply is readable AND the label is still on
# screen. Deleting the label instead of relocating it would pass a "no tag in
# the bubble" assertion just as well, which is why the notice is asserted too.
# ---------------------------------------------------------------------------


class _TaggingBackend:
    """Returns a reply prefixed with both real provenance tags, exactly as a
    real runtime backend does when the mechanism is unaccepted at the loaded
    layer and the loaded SAE is not the certified primary."""

    def generate(self, request):
        return GenerationResult(
            text=f"{MECHANICALLY_UNVERIFIED_TAG} {ENGINEERING_DEMONSTRATION_TAG} Ottawa is the capital.",
            is_synthetic=False,
            resolved_config=request.resolved_config,
            diagnostics={"pairing": "gemma"},
        )


def _send_fn(demo):
    fns = [bf for bf in demo.fns.values() if bf.fn.__name__ == "_on_send"]
    assert fns, "no _on_send handler registered"
    return fns[0]


def test_provenance_tags_are_lifted_out_of_the_reply_and_shown_beside_it():
    gemma_entries = load_entries("gemma")
    demo = build_demo(
        gemma_entries=gemma_entries,
        qwen_entries=load_entries("qwen"),
        gemma_backend=_TaggingBackend(),
        qwen_backend=StubConceptLabBackend(),
    )
    block_fn = _send_fn(demo)
    entry = gemma_entries[0]
    direction = entry.calibrated_directions[0]
    result = block_fn.fn("What is the capital of Canada?", [], entry.concept_id, direction, "low", 0, "en")

    # The wiring itself: a handler returning a different number of values than
    # its declared outputs is the classic way an added output breaks at run
    # time while every pure-logic test still passes.
    assert len(result) == len(block_fn.outputs)

    new_history = result[0]
    reply = new_history[-1]
    reply_text = reply["content"] if isinstance(reply, dict) else reply[1]
    assert MECHANICALLY_UNVERIFIED_TAG not in reply_text
    assert ENGINEERING_DEMONSTRATION_TAG not in reply_text
    assert "Ottawa is the capital." in reply_text

    notice = result[-1]
    assert MECHANICALLY_UNVERIFIED_TAG in notice
    assert ENGINEERING_DEMONSTRATION_TAG in notice


def test_a_tag_string_inside_the_models_own_answer_is_not_stripped():
    """Only a LEADING run of tags is peeled. A reply that happens to quote one
    keeps it, because rewriting a model's own output is a different act from
    relocating a label the product itself attached."""
    from sae_concept_lab.ui.tab import _split_provenance_tags

    quoted = f"The tool prints {MECHANICALLY_UNVERIFIED_TAG} before each reply."
    body, found = _split_provenance_tags(quoted)
    assert body == quoted
    assert found == []
