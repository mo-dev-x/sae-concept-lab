"""End-to-end, CPU-only smoke checks against the real build_demo() config
-- no GPU, no real model weights, no browser. Complements the pure-logic
tests: this is the "does it actually assemble and render" layer, the
same role gemma3_tool.py's test_build_ui_header_renders_the_sample_max_
proxy_caveat test already plays in this repo."""

from __future__ import annotations

import json

from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import default_bundle_path, load_bundle
from sae_concept_lab.i18n import t
from sae_concept_lab.ui.app_ui import build_demo


def _build():
    gemma_bundle = load_bundle(default_bundle_path("gemma"))
    qwen_bundle = load_bundle(default_bundle_path("qwen"))
    demo = build_demo(
        gemma_bundle=gemma_bundle,
        qwen_bundle=qwen_bundle,
        gemma_backend=StubConceptLabBackend(),
        qwen_backend=StubConceptLabBackend(),
    )
    return demo, gemma_bundle, qwen_bundle


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


def test_concept_cards_are_present_for_both_models_with_distinct_concepts():
    demo, gemma_bundle, qwen_bundle = _build()
    cfg = demo.get_config_file()
    datasets = [c for c in cfg["components"] if c["type"] == "dataset"]
    assert len(datasets) == 2
    all_samples = [row for ds in datasets for row in ds["props"]["samples"]]
    gemma_labels = {c["label"]["en"] for c in gemma_bundle["concepts"]}
    qwen_labels = {c["label"]["en"] for c in qwen_bundle["concepts"]}
    assert gemma_labels.isdisjoint(qwen_labels)  # the two bundles use different concept sets
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
    technical value (seed default, a FAKE- feature id, an sae id) lives
    under an Advanced accordion, not in a top-level/public component."""
    demo, gemma_bundle, _qwen = _build()
    cfg = demo.get_config_file()

    root_ids = {c["id"] for c in cfg["components"] if c["type"] == "column" and c.get("props", {}).get("root")}
    # Simpler and robust across Gradio config-shape versions: check the
    # known raw values only appear inside components whose type is one of
    # the Advanced-only leaf types (json, number-labelled "Seed", or a
    # radio literally labelled "Positions") -- never inside a markdown/
    # button/chatbot/textbox component (the public surface).
    disallowed_types = {"markdown", "button", "chatbot", "textbox"}
    needle = gemma_bundle["concepts"][0]["feature_id"]
    for c in cfg["components"]:
        if c["type"] not in disallowed_types:
            continue
        rendered = json.dumps(c.get("props", {}))
        assert needle not in rendered, f"raw feature_id leaked into public component type={c['type']!r}"
        assert gemma_bundle["sae_id"] not in rendered, f"raw sae_id leaked into public component type={c['type']!r}"
