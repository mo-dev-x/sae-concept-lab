"""build_model_tab() -- the ONE shared component tree, instantiated once
per pairing (Gemma, Qwen) by ui/app_ui.py inside its own gr.Tab. There is
no second copy of this layout/logic anywhere: the two tabs differ only in
which canonical entries and backend instance they close over.

Every behavioural decision (the reset rule, chat turns, Compare, what
Public vs Advanced renders, direction availability, executability) is
delegated to sae_concept_lab.core.logic and
sae_concept_lab.canonical.concept_bundle -- this module is Gradio wiring
only: build components, attach handlers, return a language-retranslation
registry to the caller. It never resolves a control, never decides
publishability, and never re-derives which direction is available; it
only asks the canonical package and renders the answer.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import gradio as gr

from sae_concept_lab.canonical.concept_bundle import (
    BundleEntry,
    Direction,
    DirectionNotCalibratedError,
    check_direction_executable,
    resolve_control,
)
from sae_concept_lab.core.gemma_backend import MECHANICALLY_UNVERIFIED_TAG
from sae_concept_lab.core.logic import (
    Selection,
    advanced_output_details,
    advanced_positions_text,
    apply_selection_change,
    assert_compare_invariant,
    public_output_summary,
    run_compare,
    send_message,
)
from sae_concept_lab.core.protocol import ConceptLabBackend
from sae_concept_lab.core.scientific_identity import ENGINEERING_DEMONSTRATION_TAG
from sae_concept_lab.fixtures.labels import concept_description, concept_label
from sae_concept_lab.i18n import DEFAULT_LANG, t

# Deliberately non-zero and deliberately NOT inside StubConceptLabBackend
# (which must stay a pure, instantaneous function of its inputs for the
# determinism tests): this is a UI-layer pause purely so the Gradio
# pending/progress state is visible for a manual demo or a screenshot,
# not a simulation of real generation latency.
DEMO_THINK_TIME_SECONDS = 0.4

#: The provenance labels a real backend prepends to every reply. They are
#: peeled off the CHAT BUBBLE and rendered once, persistently, beside it --
#: not discarded. The backend still emits them (asserted by
#: tests/test_gemma_runtime_backend.py, test_qwen_runtime_backend.py and
#: test_scientific_identity_gate.py) and result.diagnostics still carries the
#: structured verdict; this is presentation only. Repeating a two-line legal
#: notice in front of all eight words of every answer made the answers
#: unreadable, which is its own way of not being read.
_PROVENANCE_TAGS = (MECHANICALLY_UNVERIFIED_TAG, ENGINEERING_DEMONSTRATION_TAG)


def _split_provenance_tags(text: str) -> tuple[str, list[str]]:
    """Return (body, tags_found). Only strips tags that appear as a LEADING
    run, which is the only place a backend puts them -- a tag string occurring
    inside a model's own answer is left exactly where it is."""
    body = text
    found: list[str] = []
    changed = True
    while changed:
        changed = False
        for tag in _PROVENANCE_TAGS:
            if body.startswith(tag):
                found.append(tag)
                body = body[len(tag):].lstrip()
                changed = True
    return body, found


def _provenance_notice(tags: list[str]) -> str:
    if not tags:
        return ""
    return "\n\n".join(f"> {tag}" for tag in tags)


def _strip_tags_from_last_reply(history: list) -> tuple[list, list[str]]:
    """Peel the tags off the most recent assistant turn only. Returns the
    history with that turn's text replaced, plus the tags removed."""
    if not history:
        return history, []
    cleaned = list(history)
    last = cleaned[-1]
    if isinstance(last, dict) and isinstance(last.get("content"), str):
        body, found = _split_provenance_tags(last["content"])
        if found:
            cleaned[-1] = {**last, "content": body}
        return cleaned, found
    if isinstance(last, (list, tuple)) and len(last) == 2 and isinstance(last[1], str):
        body, found = _split_provenance_tags(last[1])
        if found:
            cleaned[-1] = [last[0], body]
        return cleaned, found
    return cleaned, []

#: Chat-determinism control only (StubConceptLabBackend's digest input).
#: Not a canonical field -- the canonical contract has no notion of a
#: generation seed, only of what a control resolves to.
DEFAULT_SEED = 0

#: A single shared default, named at module level rather than called
#: inline in each handler's signature (ruff B008): Gradio's own
#: introspection swaps this sentinel for a live tracker at call time, so
#: the same instance is safe to reuse across every handler that needs one.
_PROGRESS_DEFAULT = gr.Progress()

RelangTarget = tuple[Any, Callable[[str], Any]]


def _direction_choices_for_entry(entry: BundleEntry, lang: str) -> list[tuple[str, str]]:
    """Only the directions this entry has calibrated -- never both
    unconditionally. An unavailable direction is not offered as a choice
    at all, which is this UI's form of "disabled" for a single radio
    option (gr.Radio has no per-choice interactive flag)."""
    return [(t(f"direction_{d.value}", lang), d.value) for d in entry.calibrated_directions]


def _direction_unavailable_notice(entry: BundleEntry, lang: str) -> str:
    """Exactly DirectionNotCalibratedError.MESSAGE for every direction this
    entry did NOT calibrate, each labelled with its own name so the
    reader knows which control is missing and why -- never a rephrasing
    of the canonical message, which is fixed by the contract."""
    lines = [
        f"{t(f'direction_{d.value}', lang)} ({t('direction_unavailable_label', lang)}): "
        f"{DirectionNotCalibratedError.MESSAGE}"
        for d in Direction
        if d not in entry.calibrated_directions
    ]
    return "\n".join(lines)


def _capability_notice(report, lang: str) -> str:
    """Renders a CapabilityReport's own reasons verbatim -- PROHIBITED and
    CAPABILITY_LIMIT text is canonical, English-only, descriptive prose;
    translating it would mean rewriting what it says, which is exactly
    the "reimplement scientific validation" this module must not do. Only
    the title line is localized chrome."""
    if report.executable:
        return ""
    lines = [t("capability_notice_title", lang)]
    for classification, reason in zip(report.classifications, report.reasons, strict=False):
        lines.append(f"[{classification.value}] {reason}")
    if not report.calibrated:
        lines.extend(report.reasons)
    return "\n".join(lines)


def _strength_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (t("strength_low", lang), "low"),
        (t("strength_medium", lang), "medium"),
        (t("strength_high", lang), "high"),
    ]


def _concept_samples(entries: tuple[BundleEntry, ...], lang: str) -> list[list[str]]:
    return [[concept_label(e.concept_id, lang), concept_description(e.concept_id, lang)] for e in entries]


def _concept_detail_text(entry: BundleEntry, lang: str) -> str:
    return f"### {concept_label(entry.concept_id, lang)}\n\n{concept_description(entry.concept_id, lang)}"


def build_model_tab(
    *,
    model_key: str,
    entries: tuple[BundleEntry, ...],
    backend: ConceptLabBackend,
    lang_state: gr.State,
    lang_radio: gr.Radio,
) -> list[RelangTarget]:
    """Builds the full component tree for one pairing's tab. Must be
    called inside a `with gr.Tab(...):` block. Returns the list of
    (component, lang -> update) pairs the caller merges into the single
    global language-switch handler in ui/app_ui.py.

    `lang_radio` is used to attach extra listeners here, for components
    whose localized text depends on more than just `lang`: concept_detail_md
    and direction_radio/direction_unavailable_md need the currently selected
    concept, output_summary_md needs the last ResolvedControlState. These
    run alongside (not instead of) the single global relang handler
    app_ui.py builds from the returned list."""
    lang0 = DEFAULT_LANG
    relang: list[RelangTarget] = []
    first_entry = entries[0]
    first_direction = first_entry.calibrated_directions[0].value

    # ---- 1. concept cards + direction + strength -------------------------
    section1_md = gr.Markdown(t("concept_section_title", lang0))
    relang.append((section1_md, lambda lang: gr.Markdown(t("concept_section_title", lang))))

    concept_dataset = gr.Dataset(
        components=[gr.Textbox(visible=False), gr.Textbox(visible=False)],
        samples=_concept_samples(entries, lang0),
        type="index",
        label=None,
    )
    relang.append((concept_dataset, lambda lang: gr.Dataset(samples=_concept_samples(entries, lang))))

    concept_detail_md = gr.Markdown(_concept_detail_text(first_entry, lang0))
    concept_state = gr.State(first_entry.concept_id)

    direction_radio = gr.Radio(
        choices=_direction_choices_for_entry(first_entry, lang0),
        value=first_direction,
        label=t("direction_label", lang0),
    )
    direction_unavailable_md = gr.Markdown(_direction_unavailable_notice(first_entry, lang0))

    strength_radio = gr.Radio(choices=_strength_choices(lang0), value="low", label=t("strength_label", lang0))
    relang.append(
        (strength_radio, lambda lang: gr.Radio(choices=_strength_choices(lang), label=t("strength_label", lang)))
    )

    # ---- 2. chat -----------------------------------------------------------
    section2_md = gr.Markdown(t("chat_section_title", lang0))
    relang.append((section2_md, lambda lang: gr.Markdown(t("chat_section_title", lang))))

    chatbot = gr.Chatbot(value=[])
    chat_input = gr.Textbox(label=t("chat_input_label", lang0))
    relang.append((chat_input, lambda lang: gr.Textbox(label=t("chat_input_label", lang))))
    chat_send_btn = gr.Button(t("chat_send", lang0))
    relang.append((chat_send_btn, lambda lang: gr.Button(t("chat_send", lang))))

    reset_notice_md = gr.Markdown("")
    # Where the provenance labels live now: once, beside the conversation,
    # instead of prepended to every reply. Empty until a reply carries one.
    provenance_notice_md = gr.Markdown("")
    capability_notice_md = gr.Markdown("")
    output_summary_title_md = gr.Markdown(f"**{t('output_summary_title', lang0)}**")
    relang.append((output_summary_title_md, lambda lang: gr.Markdown(f"**{t('output_summary_title', lang)}**")))
    output_summary_md = gr.Markdown("")

    history_state = gr.State([])
    # Seeded with the page's own defaults (first concept, its first
    # available direction, low), NOT None: apply_selection_change()
    # treats previous_selection=None as "no selection recorded yet" and
    # deliberately skips the reset (so page load itself never resets an
    # empty conversation) -- but that same "first change is free" rule
    # silently swallowed the FIRST real settings change a user made if
    # they touched Direction/Strength before ever clicking a concept
    # card, since selection_state had never been written and was still
    # None. Caught by actually clicking through the running app, not by
    # the unit tests (which always pass an explicit previous_selection).
    selection_state = gr.State(Selection(first_entry.concept_id, first_direction, "low"))
    resolved_config_state = gr.State(None)

    # ---- 3. compare ----------------------------------------------------------
    section3_md = gr.Markdown(t("compare_section_title", lang0))
    relang.append((section3_md, lambda lang: gr.Markdown(t("compare_section_title", lang))))
    compare_btn = gr.Button(t("compare_button", lang0))
    relang.append((compare_btn, lambda lang: gr.Button(t("compare_button", lang))))
    with gr.Row():
        compare_original_md = gr.Markdown(f"**{t('compare_original_label', lang0)}**")
        relang.append(
            (compare_original_md, lambda lang: gr.Markdown(f"**{t('compare_original_label', lang)}**"))
        )
        compare_modified_md = gr.Markdown(f"**{t('compare_modified_label', lang0)}**")
        relang.append(
            (compare_modified_md, lambda lang: gr.Markdown(f"**{t('compare_modified_label', lang)}**"))
        )

    # ---- 4. advanced -----------------------------------------------------------
    with gr.Accordion(t("advanced_accordion_title", lang0), open=False) as advanced_accordion:
        relang.append(
            (advanced_accordion, lambda lang: gr.Accordion(label=t("advanced_accordion_title", lang)))
        )
        seed_number = gr.Number(value=DEFAULT_SEED, label=t("advanced_seed_label", lang0), precision=0)
        relang.append(
            (seed_number, lambda lang: gr.Number(label=t("advanced_seed_label", lang)))
        )
        # Read-only: positions comes from the bundle entry, never a public
        # control. An ATTESTED entry's own ratified position is
        # authoritative; this product's own non-ATTESTED (FAKE) fixtures
        # default to ALL per the 2026-08-13 researcher ruling (BOUNDARY.md)
        # -- a fixture-authoring choice, not a default this rendering or
        # the resolver applies.
        positions_display_md = gr.Markdown(advanced_positions_text(first_entry, lang0))
        resolved_state_title_md = gr.Markdown(f"**{t('advanced_resolved_state_title', lang0)}**")
        relang.append(
            (resolved_state_title_md, lambda lang: gr.Markdown(f"**{t('advanced_resolved_state_title', lang)}**"))
        )
        resolved_state_json = gr.JSON(value={})

    # ---- initial resolved state, so Advanced/Public aren't blank on load ----
    initial_resolved = resolve_control(first_entry, direction=first_direction, strength="low")
    resolved_config_state.value = initial_resolved
    resolved_state_json.value = advanced_output_details(initial_resolved)

    # ---- language-switch refreshers for components whose text depends on
    # more than just `lang` (current concept selection / last resolved
    # state) -- these run alongside the simple relang list above, not
    # instead of it. Without these, a language switch would leave the
    # concept-detail panel, the direction choices/notice, the positions
    # display, and the last output summary stuck in whichever language
    # was active when they were last written.
    def _refresh_concept_detail_on_lang_change(lang: str, concept_id: str):
        entry = next(e for e in entries if e.concept_id == concept_id)
        return _concept_detail_text(entry, lang)

    lang_radio.change(
        _refresh_concept_detail_on_lang_change,
        inputs=[lang_radio, concept_state],
        outputs=[concept_detail_md],
    )

    def _refresh_direction_controls_on_lang_change(lang: str, concept_id: str, current_direction: str):
        entry = next(e for e in entries if e.concept_id == concept_id)
        choices = _direction_choices_for_entry(entry, lang)
        value = current_direction if entry.has_direction(current_direction) else entry.calibrated_directions[0].value
        return gr.Radio(choices=choices, value=value), _direction_unavailable_notice(entry, lang)

    lang_radio.change(
        _refresh_direction_controls_on_lang_change,
        inputs=[lang_radio, concept_state, direction_radio],
        outputs=[direction_radio, direction_unavailable_md],
    )

    def _refresh_positions_display_on_lang_change(lang: str, concept_id: str):
        entry = next(e for e in entries if e.concept_id == concept_id)
        return advanced_positions_text(entry, lang)

    lang_radio.change(
        _refresh_positions_display_on_lang_change,
        inputs=[lang_radio, concept_state],
        outputs=[positions_display_md],
    )

    def _refresh_output_summary_on_lang_change(lang: str, resolved_config):
        if resolved_config is None:
            return ""
        return public_output_summary(resolved_config, lang)

    lang_radio.change(
        _refresh_output_summary_on_lang_change,
        inputs=[lang_radio, resolved_config_state],
        outputs=[output_summary_md],
    )

    # -------------------------------------------------------------------------
    # handlers
    # -------------------------------------------------------------------------

    def _check_and_apply(concept_id, direction, strength, history, previous_selection, lang):
        new_selection = Selection(concept_id=concept_id, direction=direction, strength_level=strength)
        result = apply_selection_change(
            previous_selection=previous_selection,
            new_selection=new_selection,
            history=history,
        )
        notice = t(result.notice_key, lang) if result.notice_key else ""
        return result.new_history, result.new_history, notice, new_selection

    def _on_concept_click(idx, strength, history, previous_selection, lang):
        entry = entries[idx]
        concept_id = entry.concept_id
        detail_text = _concept_detail_text(entry, lang)
        new_direction = entry.calibrated_directions[0].value
        direction_update = gr.Radio(choices=_direction_choices_for_entry(entry, lang), value=new_direction)
        unavailable_notice = _direction_unavailable_notice(entry, lang)
        positions_text = advanced_positions_text(entry, lang)
        new_history, chatbot_value, notice, new_selection = _check_and_apply(
            concept_id, new_direction, strength, history, previous_selection, lang
        )
        return (
            concept_id,
            detail_text,
            direction_update,
            unavailable_notice,
            positions_text,
            new_history,
            chatbot_value,
            notice,
            new_selection,
        )

    concept_dataset.click(
        _on_concept_click,
        inputs=[concept_dataset, strength_radio, history_state, selection_state, lang_state],
        outputs=[
            concept_state,
            concept_detail_md,
            direction_radio,
            direction_unavailable_md,
            positions_display_md,
            history_state,
            chatbot,
            reset_notice_md,
            selection_state,
        ],
    )

    def _on_direction_or_strength_change(concept_id, direction, strength, history, previous_selection, lang):
        return _check_and_apply(concept_id, direction, strength, history, previous_selection, lang)

    for trigger in (direction_radio, strength_radio):
        trigger.change(
            _on_direction_or_strength_change,
            inputs=[
                concept_state,
                direction_radio,
                strength_radio,
                history_state,
                selection_state,
                lang_state,
            ],
            outputs=[history_state, chatbot, reset_notice_md, selection_state],
        )

    def _on_send(message, history, concept_id, direction, strength, seed, lang, progress=_PROGRESS_DEFAULT):
        entry = next(e for e in entries if e.concept_id == concept_id)
        report = check_direction_executable(entry, direction)
        if not report.executable:
            return (history, history, message, "", None, {}, _capability_notice(report, lang), "")

        progress(0, desc=t("loading_label", lang))
        time.sleep(DEMO_THINK_TIME_SECONDS)
        resolved = resolve_control(entry, direction=direction, strength=strength)
        new_history, result = send_message(
            backend=backend,
            history=history,
            prompt=message,
            model_key=model_key,
            decoding={},
            seed=int(seed),
            resolved_config=resolved,
        )
        summary = public_output_summary(resolved, lang)
        details = advanced_output_details(resolved)
        if result.diagnostics is not None:
            details = {**details, "backend_diagnostics": result.diagnostics}
        # The tags come off the bubble and go into their own persistent line.
        # history_state keeps the CLEANED text so the label is not re-shown on
        # every later turn, while result.text and result.diagnostics -- the
        # machine-readable record -- are untouched.
        new_history, tags = _strip_tags_from_last_reply(new_history)
        progress(1)
        return (
            new_history,
            new_history,
            "",
            summary,
            resolved,
            details,
            "",
            _provenance_notice(tags),
        )

    # Enter-to-send. Bound to the SAME handler, inputs and outputs as the
    # button, so pressing Enter and clicking Send cannot diverge -- one
    # wiring described twice is how the two drift apart.
    _send_wiring = dict(
        fn=_on_send,
        inputs=[chat_input, history_state, concept_state, direction_radio, strength_radio, seed_number, lang_state],
        outputs=[
            history_state,
            chatbot,
            chat_input,
            output_summary_md,
            resolved_config_state,
            resolved_state_json,
            capability_notice_md,
            provenance_notice_md,
        ],
    )
    chat_send_btn.click(**_send_wiring)
    chat_input.submit(**_send_wiring)

    def _on_compare(message, history, concept_id, direction, strength, seed, lang, progress=_PROGRESS_DEFAULT):
        entry = next(e for e in entries if e.concept_id == concept_id)
        report = check_direction_executable(entry, direction)
        if not report.executable:
            return ("", "", None, {}, _capability_notice(report, lang), "")

        progress(0, desc=t("loading_label", lang))
        time.sleep(DEMO_THINK_TIME_SECONDS)
        resolved = resolve_control(entry, direction=direction, strength=strength)
        compare = run_compare(
            backend=backend,
            history=history,
            prompt=message,
            model_key=model_key,
            decoding={},
            seed=int(seed),
            resolved_config=resolved,
        )
        assert_compare_invariant(compare)
        original_body, original_tags = _split_provenance_tags(compare.original_text)
        modified_body, modified_tags = _split_provenance_tags(compare.modified_text)
        original_md = f"**{t('compare_original_label', lang)}**\n\n{original_body}"
        modified_md = f"**{t('compare_modified_label', lang)}**\n\n{modified_body}"
        details = advanced_output_details(resolved)
        if compare.modified_result is not None and compare.modified_result.diagnostics is not None:
            details = {**details, "backend_diagnostics": compare.modified_result.diagnostics}
        # Same tags on both arms; show the union once rather than twice.
        seen = list(dict.fromkeys(original_tags + modified_tags))
        progress(1)
        return (
            original_md,
            modified_md,
            resolved,
            details,
            "",
            _provenance_notice(seen),
        )

    compare_btn.click(
        _on_compare,
        inputs=[chat_input, history_state, concept_state, direction_radio, strength_radio, seed_number, lang_state],
        outputs=[
            compare_original_md,
            compare_modified_md,
            resolved_config_state,
            resolved_state_json,
            capability_notice_md,
            provenance_notice_md,
        ],
    )

    return relang
