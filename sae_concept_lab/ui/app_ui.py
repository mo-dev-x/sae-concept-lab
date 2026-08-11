"""Top-level Gradio Blocks: the permanent FAKE banner, the language
switcher (global -- one switch retranslates both tabs), the "How does
this work?" explainer, and the two model tabs built from
ui.tab.build_model_tab (the one shared component tree)."""

from __future__ import annotations

from typing import Any

import gradio as gr

from sae_concept_lab.core.protocol import ConceptLabBackend
from sae_concept_lab.i18n import DEFAULT_LANG, LANGS, t
from sae_concept_lab.ui.tab import RelangTarget, build_model_tab


def build_demo(
    *,
    gemma_bundle: dict[str, Any],
    qwen_bundle: dict[str, Any],
    gemma_backend: ConceptLabBackend,
    qwen_backend: ConceptLabBackend,
) -> gr.Blocks:
    lang0 = DEFAULT_LANG
    relang: list[RelangTarget] = []

    with gr.Blocks(title=t("app_title", lang0)) as demo:
        lang_state = gr.State(lang0)

        banner_md = gr.Markdown(f"### ⚠️ {t('fake_banner', lang0)}")
        relang.append((banner_md, lambda lang: gr.Markdown(f"### ⚠️ {t('fake_banner', lang)}")))

        title_md = gr.Markdown(f"# {t('app_title', lang0)}")
        relang.append((title_md, lambda lang: gr.Markdown(f"# {t('app_title', lang)}")))

        lang_radio = gr.Radio(choices=list(LANGS), value=lang0, label=t("lang_label", lang0))
        relang.append((lang_radio, lambda lang: gr.Radio(label=t("lang_label", lang))))

        with gr.Accordion(t("explainer_title", lang0), open=False) as explainer_accordion:
            relang.append(
                (explainer_accordion, lambda lang: gr.Accordion(label=t("explainer_title", lang)))
            )
            explainer_body_md = gr.Markdown(t("explainer_body", lang0))
            relang.append((explainer_body_md, lambda lang: gr.Markdown(t("explainer_body", lang))))

        with gr.Tabs():
            with gr.Tab(t("tab_gemma", lang0)) as gemma_tab:
                relang.append((gemma_tab, lambda lang: gr.Tab(label=t("tab_gemma", lang))))
                relang.extend(
                    build_model_tab(
                        model_key="gemma",
                        bundle=gemma_bundle,
                        backend=gemma_backend,
                        lang_state=lang_state,
                        lang_radio=lang_radio,
                    )
                )
            with gr.Tab(t("tab_qwen", lang0)) as qwen_tab:
                relang.append((qwen_tab, lambda lang: gr.Tab(label=t("tab_qwen", lang))))
                relang.extend(
                    build_model_tab(
                        model_key="qwen",
                        bundle=qwen_bundle,
                        backend=qwen_backend,
                        lang_state=lang_state,
                        lang_radio=lang_radio,
                    )
                )

        # One global handler retranslates the banner, title, language
        # label, explainer, and every component either tab registered --
        # a single switch, one shared codebase, per the brief.
        relang_components = [component for component, _update_fn in relang]

        def _apply_language(lang: str):
            return [lang, *(update_fn(lang) for _component, update_fn in relang)]

        lang_radio.change(_apply_language, inputs=[lang_radio], outputs=[lang_state, *relang_components])

    return demo
