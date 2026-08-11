"""build_model_tab() -- the ONE shared component tree, instantiated once
per model (Gemma, Qwen) by ui/app_ui.py inside its own gr.Tab. There is no
second copy of this layout/logic anywhere: the two tabs differ only in
which bundle and backend instance they close over.

Every behavioural decision (the reset rule, chat turns, Compare, what
Public vs Advanced renders) is delegated to sae_concept_lab.core.logic --
this module is Gradio wiring only: build components, attach handlers,
return a language-retranslation registry to the caller. No decision made
here is untestable-in-principle; the corresponding pure function in
core/logic.py is what the test suite actually exercises.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable

import gradio as gr

from sae_concept_lab.core.config import resolve_config
from sae_concept_lab.core.logic import (
    Selection,
    advanced_output_details,
    apply_selection_change,
    assert_compare_invariant,
    public_output_summary,
    run_compare,
    send_message,
)
from sae_concept_lab.core.protocol import ConceptLabBackend
from sae_concept_lab.i18n import DEFAULT_LANG, t

# Deliberately non-zero and deliberately NOT inside StubConceptLabBackend
# (which must stay a pure, instantaneous function of its inputs for the
# determinism tests): this is a UI-layer pause purely so the Gradio
# pending/progress state is visible for a manual demo or a screenshot,
# not a simulation of real generation latency.
DEMO_THINK_TIME_SECONDS = 0.4

RelangTarget = tuple[Any, Callable[[str], Any]]


def _direction_choices(lang: str) -> list[tuple[str, str]]:
    return [(t("direction_amplify", lang), "amplify"), (t("direction_suppress", lang), "suppress")]


def _strength_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (t("strength_low", lang), "low"),
        (t("strength_medium", lang), "medium"),
        (t("strength_high", lang), "high"),
    ]


def _localized(field: dict[str, str], lang: str) -> str:
    return field.get(lang, field.get(DEFAULT_LANG, ""))


def _concept_samples(concepts: list[dict[str, Any]], lang: str) -> list[list[str]]:
    return [[_localized(c["label"], lang), _localized(c["description"], lang)] for c in concepts]


def _concept_detail_text(concept: dict[str, Any], lang: str) -> str:
    return f"### {_localized(concept['label'], lang)}\n\n{_localized(concept['description'], lang)}"


def build_model_tab(
    *,
    model_key: str,
    bundle: dict[str, Any],
    backend: ConceptLabBackend,
    lang_state: gr.State,
    lang_radio: gr.Radio,
) -> list[RelangTarget]:
    """Builds the full component tree for one model tab. Must be called
    inside a `with gr.Tab(...):` block. Returns the list of
    (component, lang -> update) pairs the caller merges into the single
    global language-switch handler in ui/app_ui.py.

    `lang_radio` (the actual language selector, not just its `lang_state`
    value) is used to attach two EXTRA listeners here, for the two
    components whose localized text depends on more than just `lang` --
    concept_detail_md needs to know which concept is currently selected,
    output_summary_md needs the last ResolvedConfig. These run alongside
    (not instead of) the single global relang handler app_ui.py builds
    from the returned list."""
    concepts = bundle["concepts"]
    lang0 = DEFAULT_LANG
    relang: list[RelangTarget] = []

    # ---- 1. concept cards + direction + strength -------------------------
    section1_md = gr.Markdown(t("concept_section_title", lang0))
    relang.append((section1_md, lambda lang: gr.Markdown(t("concept_section_title", lang))))

    concept_dataset = gr.Dataset(
        components=[gr.Textbox(visible=False), gr.Textbox(visible=False)],
        samples=_concept_samples(concepts, lang0),
        type="index",
        label=None,
    )
    relang.append((concept_dataset, lambda lang: gr.Dataset(samples=_concept_samples(concepts, lang))))

    concept_detail_md = gr.Markdown(_concept_detail_text(concepts[0], lang0))
    concept_state = gr.State(concepts[0]["concept_id"])

    direction_radio = gr.Radio(choices=_direction_choices(lang0), value="amplify", label=t("direction_label", lang0))
    relang.append(
        (direction_radio, lambda lang: gr.Radio(choices=_direction_choices(lang), label=t("direction_label", lang)))
    )

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
    output_summary_title_md = gr.Markdown(f"**{t('output_summary_title', lang0)}**")
    relang.append((output_summary_title_md, lambda lang: gr.Markdown(f"**{t('output_summary_title', lang)}**")))
    output_summary_md = gr.Markdown("")

    history_state = gr.State([])
    # Seeded with the page's own defaults (first concept, amplify, low), NOT
    # None: apply_selection_change() treats previous_selection=None as "no
    # selection recorded yet" and deliberately skips the reset (so page load
    # itself never resets an empty conversation) -- but that same "first
    # change is free" rule silently swallowed the FIRST real settings change
    # a user made if they touched Direction/Strength before ever clicking a
    # concept card, since selection_state had never been written and was
    # still None. Caught by actually clicking through the running app, not
    # by the unit tests (which always pass an explicit previous_selection).
    selection_state = gr.State(Selection(concepts[0]["concept_id"], "amplify", "low"))
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
    def _positions_choices(lang: str) -> list[tuple[str, str]]:
        return [
            (t("advanced_positions_generated_only", lang), "generated_only"),
            (t("advanced_positions_all", lang), "all"),
        ]

    with gr.Accordion(t("advanced_accordion_title", lang0), open=False) as advanced_accordion:
        relang.append(
            (advanced_accordion, lambda lang: gr.Accordion(label=t("advanced_accordion_title", lang)))
        )
        seed_number = gr.Number(value=bundle["seed_default"], label=t("advanced_seed_label", lang0), precision=0)
        relang.append(
            (seed_number, lambda lang: gr.Number(label=t("advanced_seed_label", lang)))
        )
        positions_radio = gr.Radio(
            choices=_positions_choices(lang0),
            value=bundle["positions_default"],
            label=t("advanced_positions_label", lang0),
        )
        relang.append(
            (
                positions_radio,
                lambda lang: gr.Radio(choices=_positions_choices(lang), label=t("advanced_positions_label", lang)),
            )
        )
        resolved_state_title_md = gr.Markdown(f"**{t('advanced_resolved_state_title', lang0)}**")
        relang.append(
            (resolved_state_title_md, lambda lang: gr.Markdown(f"**{t('advanced_resolved_state_title', lang)}**"))
        )
        resolved_state_json = gr.JSON(value={})
        diagnostics_title_md = gr.Markdown(f"**{t('advanced_diagnostics_title', lang0)}**")
        relang.append(
            (diagnostics_title_md, lambda lang: gr.Markdown(f"**{t('advanced_diagnostics_title', lang)}**"))
        )
        diagnostics_json = gr.JSON(value={})

    # ---- initial resolved state, so Advanced/Public aren't blank on load ----
    initial_resolved = resolve_config(
        bundle=bundle,
        concept_id=concepts[0]["concept_id"],
        direction="amplify",
        strength_level="low",
    )
    resolved_config_state.value = initial_resolved
    resolved_state_json.value = advanced_output_details(initial_resolved)
    diagnostics_json.value = initial_resolved.diagnostics

    # ---- language-switch refreshers for the two components whose text
    # depends on more than just `lang` (current concept selection / last
    # resolved config) -- these run alongside the simple relang list above,
    # not instead of it. Without these, a language switch would leave the
    # concept-detail panel and the last output summary stuck in whichever
    # language was active when they were last written -- a real gap, not a
    # hypothetical one (caught by inspecting an actual rendered screenshot).
    def _refresh_concept_detail_on_lang_change(lang: str, concept_id: str) -> str:
        concept = next(c for c in concepts if c["concept_id"] == concept_id)
        return _concept_detail_text(concept, lang)

    lang_radio.change(
        _refresh_concept_detail_on_lang_change,
        inputs=[lang_radio, concept_state],
        outputs=[concept_detail_md],
    )

    def _refresh_output_summary_on_lang_change(lang: str, resolved_config):
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

    def _on_concept_click(idx, direction, strength, history, previous_selection, lang):
        concept = concepts[idx]
        concept_id = concept["concept_id"]
        detail_text = _concept_detail_text(concept, lang)
        new_history, chatbot_value, notice, new_selection = _check_and_apply(
            concept_id, direction, strength, history, previous_selection, lang
        )
        return concept_id, detail_text, new_history, chatbot_value, notice, new_selection

    concept_dataset.click(
        _on_concept_click,
        inputs=[
            concept_dataset,
            direction_radio,
            strength_radio,
            history_state,
            selection_state,
            lang_state,
        ],
        outputs=[concept_state, concept_detail_md, history_state, chatbot, reset_notice_md, selection_state],
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

    def _on_send(message, history, concept_id, direction, strength, seed, positions, lang, progress=gr.Progress()):
        progress(0, desc=t("loading_label", lang))
        time.sleep(DEMO_THINK_TIME_SECONDS)
        resolved = resolve_config(
            bundle=bundle,
            concept_id=concept_id,
            direction=direction,
            strength_level=strength,
            seed=int(seed),
            positions=positions,
        )
        new_history, _result = send_message(
            backend=backend,
            history=history,
            prompt=message,
            model_key=model_key,
            decoding=resolved.decoding,
            seed=resolved.seed,
            resolved_config=resolved,
        )
        summary = public_output_summary(resolved, lang)
        progress(1)
        return (
            new_history,
            new_history,
            "",
            summary,
            resolved,
            advanced_output_details(resolved),
            resolved.diagnostics,
        )

    chat_send_btn.click(
        _on_send,
        inputs=[
            chat_input,
            history_state,
            concept_state,
            direction_radio,
            strength_radio,
            seed_number,
            positions_radio,
            lang_state,
        ],
        outputs=[
            history_state,
            chatbot,
            chat_input,
            output_summary_md,
            resolved_config_state,
            resolved_state_json,
            diagnostics_json,
        ],
    )

    def _on_compare(message, history, concept_id, direction, strength, seed, positions, lang, progress=gr.Progress()):
        progress(0, desc=t("loading_label", lang))
        time.sleep(DEMO_THINK_TIME_SECONDS)
        resolved = resolve_config(
            bundle=bundle,
            concept_id=concept_id,
            direction=direction,
            strength_level=strength,
            seed=int(seed),
            positions=positions,
        )
        compare = run_compare(
            backend=backend,
            history=history,
            prompt=message,
            model_key=model_key,
            decoding=resolved.decoding,
            seed=resolved.seed,
            resolved_config=resolved,
        )
        assert_compare_invariant(compare)
        original_md = f"**{t('compare_original_label', lang)}**\n\n{compare.original_text}"
        modified_md = f"**{t('compare_modified_label', lang)}**\n\n{compare.modified_text}"
        progress(1)
        return (
            original_md,
            modified_md,
            resolved,
            advanced_output_details(resolved),
            resolved.diagnostics,
        )

    compare_btn.click(
        _on_compare,
        inputs=[
            chat_input,
            history_state,
            concept_state,
            direction_radio,
            strength_radio,
            seed_number,
            positions_radio,
            lang_state,
        ],
        outputs=[
            compare_original_md,
            compare_modified_md,
            resolved_config_state,
            resolved_state_json,
            diagnostics_json,
        ],
    )

    return relang
