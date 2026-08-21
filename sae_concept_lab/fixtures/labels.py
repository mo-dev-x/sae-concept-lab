"""Product-owned presentation metadata for this repository's FAKE concept
fixtures: display labels, descriptions, and pairing labels.

The canonical contract holds no display name for a concept or a pairing
by design -- see sae_concept_lab/canonical/concept_bundle/schema.py's
module note and resolver.py's public_view() docstring ("There is no
display name here because the contract holds none"). Mapping a
concept_id/pairing_id to human-facing text is presentation, not
scientific validation, so it lives here, entirely outside the canonical
package, and is never consulted by anything that decides what a control
does.
"""

from __future__ import annotations

from typing import TypedDict


class Localized(TypedDict):
    en: str
    fr: str


PAIRING_LABELS: dict[str, Localized] = {
    "fake-gemma-demo-pairing": {"en": "[FAKE] Gemma demo model", "fr": "[FAKE] Modèle de démo Gemma"},
    "fake-qwen-demo-pairing": {"en": "[FAKE] Qwen demo model", "fr": "[FAKE] Modèle de démo Qwen"},
    # Real ratified pairings. Present so that a CANDIDATE bundle dropped into the
    # Mode-A slot can be rendered at all -- see CONCEPT_LABELS below.
    "gemma-3-12b-it+gemma-scope-2-12b-it": {
        "en": "Gemma 3 12B-it · Gemma Scope 2 (layer 29)",
        "fr": "Gemma 3 12B-it · Gemma Scope 2 (couche 29)",
    },
    "qwen-3.5-27b+SAE-Res-Qwen3.5-27B-W80K-L0_100": {
        "en": "Qwen3.5 27B · SAE-Res W80K L0-100 (layer 38)",
        "fr": "Qwen3.5 27B · SAE-Res W80K L0-100 (couche 38)",
    },
}

CONCEPT_LABELS: dict[str, Localized] = {
    "FAKE-gemma-warmth": {"en": "Warmth", "fr": "Chaleur"},
    "FAKE-gemma-formality": {"en": "Formality", "fr": "Formalité"},
    "FAKE-gemma-enthusiasm": {"en": "Enthusiasm", "fr": "Enthousiasme"},
    "FAKE-gemma-caution": {"en": "Caution", "fr": "Prudence"},
    "FAKE-qwen-curiosity": {"en": "Curiosity", "fr": "Curiosité"},
    "FAKE-qwen-directness": {"en": "Directness", "fr": "Franchise"},
    "FAKE-qwen-playfulness": {"en": "Playfulness", "fr": "Espièglerie"},
    "FAKE-qwen-skepticism": {"en": "Skepticism", "fr": "Scepticisme"},
    # A REAL, measured concept -- not a fixture. Its features are measured; its
    # causal claim is not, which the release gate enforces (provenance is
    # "candidate", so it can never publish) rather than the display string.
    "pro-american-exceptionalism": {
        "en": "Pro-American exceptionalism",
        "fr": "Exceptionnalisme pro-américain",
    },
}

CONCEPT_DESCRIPTIONS: dict[str, Localized] = {
    "pro-american-exceptionalism": {
        "en": ("Features that passed every "
               "discovery gate in all six evaluation cells of a full-space scan on "
               "real weights. No causal test has been run and no dose has been "
               "calibrated, so only ablation is offered and this concept can never "
               "be published in release mode."),
        "fr": ("Caractéristiques ayant franchi "
               "toutes les portes de découverte dans les six cellules d'un balayage "
               "complet sur poids réels. Aucun test causal n'a été effectué et aucune "
               "dose n'est calibrée : seule l'ablation est proposée, et ce concept ne "
               "peut jamais être publié en mode release."),
    },
    "FAKE-gemma-warmth": {
        "en": "[FAKE] Placeholder concept: friendly, personable tone.",
        "fr": "[FAKE] Concept fictif : ton chaleureux et personnel.",
    },
    "FAKE-gemma-formality": {
        "en": "[FAKE] Placeholder concept: formal, businesslike register.",
        "fr": "[FAKE] Concept fictif : registre formel et professionnel.",
    },
    "FAKE-gemma-enthusiasm": {
        "en": "[FAKE] Placeholder concept: energetic, upbeat tone. Amplify spans two "
              "layers -- a deliberate CAPABILITY_LIMIT demonstration.",
        "fr": "[FAKE] Concept fictif : ton énergique et positif. Amplifier s'étend sur "
              "deux couches -- une démonstration délibérée de CAPABILITY_LIMIT.",
    },
    "FAKE-gemma-caution": {
        "en": "[FAKE] Placeholder concept: hedged, careful phrasing. Suppress was "
              "never calibrated on this pairing.",
        "fr": "[FAKE] Concept fictif : formulation prudente et nuancée. Atténuer n'a "
              "jamais été calibré sur ce couplage.",
    },
    "FAKE-qwen-curiosity": {
        "en": "[FAKE] Placeholder concept: inquisitive, question-asking tone.",
        "fr": "[FAKE] Concept fictif : ton curieux, porté sur les questions.",
    },
    "FAKE-qwen-directness": {
        "en": "[FAKE] Placeholder concept: blunt, to-the-point phrasing. Amplify "
              "names two SAEs at one layer -- a deliberate PROHIBITED demonstration.",
        "fr": "[FAKE] Concept fictif : formulation directe et sans détour. Amplifier "
              "nomme deux SAE à une même couche -- une démonstration délibérée de "
              "PROHIBITED.",
    },
    "FAKE-qwen-playfulness": {
        "en": "[FAKE] Placeholder concept: light, humorous tone.",
        "fr": "[FAKE] Concept fictif : ton léger et humoristique.",
    },
    "FAKE-qwen-skepticism": {
        "en": "[FAKE] Placeholder concept: questioning, evidence-seeking tone. "
              "Amplify was never calibrated on this pairing.",
        "fr": "[FAKE] Concept fictif : ton dubitatif, en quête de preuves. Amplifier "
              "n'a jamais été calibré sur ce couplage.",
    },
}

DEFAULT_LANG = "en"


def pairing_label(pairing_id: str, lang: str) -> str:
    entry = PAIRING_LABELS.get(pairing_id)
    if entry is None:
        raise KeyError(f"no product label registered for pairing_id {pairing_id!r}")
    return entry.get(lang, entry[DEFAULT_LANG])


def concept_label(concept_id: str, lang: str) -> str:
    entry = CONCEPT_LABELS.get(concept_id)
    if entry is None:
        raise KeyError(f"no product label registered for concept_id {concept_id!r}")
    return entry.get(lang, entry[DEFAULT_LANG])


def concept_description(concept_id: str, lang: str) -> str:
    entry = CONCEPT_DESCRIPTIONS.get(concept_id)
    if entry is None:
        raise KeyError(f"no product description registered for concept_id {concept_id!r}")
    return entry.get(lang, entry[DEFAULT_LANG])
