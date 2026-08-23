"""FR|EN dictionary for every custom UI string in this app.

Deliberately separate from model-response language: STRINGS below covers
only chrome this app itself renders (labels, buttons, notices, the
explainer). Model output text (currently always the stub backend's
FAKE_TAG'd templates) is never looked up here and is never translated --
switching language must never change what a response says, only how the
surrounding UI is labelled.
"""

from __future__ import annotations

LANGS: tuple[str, ...] = ("en", "fr")
DEFAULT_LANG = "en"

STRINGS: dict[str, dict[str, str]] = {
    "app_title": {
        "en": "SAE Concept Lab",
        "fr": "Laboratoire de concepts SAE",
    },
    "fake_banner": {
        "en": (
            "PLACEHOLDER / NOT SCIENTIFIC EVIDENCE -- every reply and technical value below is "
            "synthetic stub data for interface testing only."
        ),
        "fr": (
            "SUBSTITUT / AUCUNE VALEUR SCIENTIFIQUE -- chaque réponse et valeur technique "
            "ci-dessous est une donnée synthétique de test d'interface uniquement."
        ),
    },
    "lang_label": {"en": "Language", "fr": "Langue"},
    "tab_gemma": {"en": "Gemma", "fr": "Gemma"},
    "tab_qwen": {"en": "Qwen", "fr": "Qwen"},
    "explainer_title": {"en": "How does this work?", "fr": "Comment ça marche ?"},
    "explainer_body": {
        "en": (
            "Pick a concept, choose whether to turn it up (Amplify) or down (Suppress), and how "
            "strongly (Low/Medium/High). Chat normally -- replies are generated with that "
            "concept adjusted. Compare shows what the same reply would have looked like with no "
            "adjustment at all, side by side. Everything in this preview is placeholder data; no "
            "real model or measurement is behind it yet."
        ),
        "fr": (
            "Choisissez un concept, indiquez si vous voulez l'amplifier ou l'atténuer, et avec "
            "quelle intensité (Faible/Moyenne/Élevée). Discutez normalement -- les réponses "
            "sont générées avec ce concept ajusté. Comparer montre à quoi aurait ressemblé la "
            "même réponse sans aucun ajustement, côte à côte. Tout ce qui figure dans cet aperçu "
            "est une donnée fictive ; aucun modèle ni mesure réel n'est encore derrière."
        ),
    },
    "concept_section_title": {"en": "1. Pick a concept", "fr": "1. Choisissez un concept"},
    "direction_label": {"en": "Direction", "fr": "Direction"},
    "direction_amplify": {"en": "Amplify", "fr": "Amplifier"},
    "direction_suppress": {"en": "Suppress", "fr": "Atténuer"},
    "strength_label": {"en": "Strength", "fr": "Intensité"},
    "strength_low": {"en": "Low", "fr": "Faible"},
    "strength_medium": {"en": "Medium", "fr": "Moyenne"},
    "strength_high": {"en": "High", "fr": "Élevée"},
    "chat_section_title": {"en": "2. Chat", "fr": "2. Discuter"},
    "chat_input_label": {"en": "Your message", "fr": "Votre message"},
    "chat_send": {"en": "Send", "fr": "Envoyer"},
    "loading_label": {"en": "Generating…", "fr": "Génération en cours…"},
    "empty_prompt_notice": {
        "en": "Type a message first. Send and Compare both read the chat box above, and it is "
              "cleared after each send -- so clicking Compare straight after sending would "
              "otherwise ask the model about nothing.",
        "fr": "Saisissez d'abord un message. Envoyer et Comparer utilisent tous deux la zone de "
              "chat ci-dessus, qui est vidée après chaque envoi -- cliquer sur Comparer juste "
              "après un envoi interrogerait donc le modèle sur rien.",
    },
    "compare_section_title": {"en": "3. Compare", "fr": "3. Comparer"},
    "compare_button": {"en": "Compare Original vs Modified", "fr": "Comparer Original et Modifié"},
    "compare_original_label": {
        "en": "Original (no concept applied)",
        "fr": "Original (aucun concept appliqué)",
    },
    "compare_modified_label": {
        "en": "Modified (concept applied)",
        "fr": "Modifié (concept appliqué)",
    },
    "reset_notice": {
        "en": "Conversation reset because you changed the concept, direction, or strength.",
        "fr": "Conversation réinitialisée : le concept, la direction ou l'intensité a changé.",
    },
    "output_summary_title": {"en": "What produced this reply", "fr": "Ce qui a produit cette réponse"},
    "output_summary_model": {"en": "Model", "fr": "Modèle"},
    "output_summary_concept": {"en": "Concept", "fr": "Concept"},
    "output_summary_direction": {"en": "Direction", "fr": "Direction"},
    "output_summary_strength": {"en": "Strength", "fr": "Intensité"},
    "advanced_accordion_title": {"en": "Advanced (technical detail)", "fr": "Avancé (détails techniques)"},
    "advanced_resolved_state_title": {
        "en": "Resolved state (canonical advanced view + execution payload)",
        "fr": "État résolu (vue avancée canonique + charge utile d'exécution)",
    },
    "advanced_seed_label": {"en": "Seed", "fr": "Graine"},
    "advanced_positions_readonly_label": {
        "en": "Positions (from bundle, read-only)",
        "fr": "Positions (du bundle, lecture seule)",
    },
    "direction_unavailable_label": {"en": "Unavailable", "fr": "Indisponible"},
    "capability_notice_title": {
        "en": "This direction cannot run on this build",
        "fr": "Cette direction ne peut pas s'exécuter dans cette build",
    },
}


def t(key: str, lang: str) -> str:
    entry = STRINGS.get(key)
    if entry is None:
        raise KeyError(f"unknown i18n key: {key!r}")
    if lang not in entry:
        raise KeyError(f"i18n key {key!r} has no {lang!r} translation")
    return entry[lang]
