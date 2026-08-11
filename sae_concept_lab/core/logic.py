"""Pure Gradio-free logic: the reset rule, chat turns, and the Compare
invariant. Kept separate from ui/ so every behavioural rule in this
module is directly unit-testable without constructing a single Gradio
component -- the same split scripts/legacy/gemma3_tool.py already uses
(dose_to_absolute_clamp, feature_by_idx, etc. are plain functions; only
build_ui() touches gradio).

Public/Advanced rendering below reads a canonical
`ResolvedControlState` (sae_concept_lab.canonical.concept_bundle.resolver)
through its OWN `public_view()`/`advanced_view()`/`execution_dict()`
methods -- this module never recomputes what those methods already
compute, and never re-derives availability, fingerprints, or the
execution payload itself. The only thing this module adds is translating
a concept_id/pairing_id into product-owned display text (labels.py),
which the canonical contract deliberately holds none of.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from sae_concept_lab.canonical.concept_bundle import ResolvedControlState
from sae_concept_lab.core.protocol import ConceptLabBackend, GenerationRequest, GenerationResult
from sae_concept_lab.fixtures.labels import concept_label, pairing_label
from sae_concept_lab.i18n import t

# ---------------------------------------------------------------------------
# Reset rule: any change to concept/direction/strength unconditionally
# clears the conversation and surfaces a localized one-line notice. There
# is no override -- a prior Advanced-only "continue anyway" escape hatch
# was removed: Public and Advanced share exactly one intervention system,
# and letting Advanced keep stale history across a settings change would
# have made it a second, divergent system in practice.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Selection:
    concept_id: str
    direction: str
    strength_level: str


@dataclasses.dataclass(frozen=True)
class SelectionChangeResult:
    new_history: list[dict[str, str]]
    reset_happened: bool
    notice_key: str | None


def apply_selection_change(
    *,
    previous_selection: Selection | None,
    new_selection: Selection,
    history: list[dict[str, str]],
) -> SelectionChangeResult:
    if previous_selection is None or previous_selection == new_selection:
        return SelectionChangeResult(new_history=history, reset_happened=False, notice_key=None)
    return SelectionChangeResult(new_history=[], reset_happened=True, notice_key="reset_notice")


# ---------------------------------------------------------------------------
# Chat turns
# ---------------------------------------------------------------------------


def build_generation_request(
    *,
    history: list[dict[str, str]],
    prompt: str,
    model_key: str,
    decoding: dict[str, Any],
    seed: int,
    apply_intervention: bool,
    resolved_config: ResolvedControlState | None,
) -> GenerationRequest:
    return GenerationRequest(
        history=tuple((m["role"], m["content"]) for m in history),
        prompt=prompt,
        model_key=model_key,
        decoding=dict(decoding),
        seed=seed,
        apply_intervention=apply_intervention,
        resolved_config=resolved_config,
    )


def send_message(
    *,
    backend: ConceptLabBackend,
    history: list[dict[str, str]],
    prompt: str,
    model_key: str,
    decoding: dict[str, Any],
    seed: int,
    resolved_config: ResolvedControlState,
) -> tuple[list[dict[str, str]], GenerationResult]:
    request = build_generation_request(
        history=history,
        prompt=prompt,
        model_key=model_key,
        decoding=decoding,
        seed=seed,
        apply_intervention=True,
        resolved_config=resolved_config,
    )
    result = backend.generate(request)
    new_history = [*history, {"role": "user", "content": prompt}, {"role": "assistant", "content": result.text}]
    return new_history, result


# ---------------------------------------------------------------------------
# Compare: Original and Modified must share history/prompt/model_key/
# decoding/seed exactly, differing ONLY in apply_intervention and
# resolved_config. Both arms are bound to the SAME resolved_config the
# caller passes in -- run_compare never resolves a second time.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CompareResult:
    original_text: str
    modified_text: str
    original_request: GenerationRequest
    modified_request: GenerationRequest


def run_compare(
    *,
    backend: ConceptLabBackend,
    history: list[dict[str, str]],
    prompt: str,
    model_key: str,
    decoding: dict[str, Any],
    seed: int,
    resolved_config: ResolvedControlState,
) -> CompareResult:
    original_request = build_generation_request(
        history=history,
        prompt=prompt,
        model_key=model_key,
        decoding=decoding,
        seed=seed,
        apply_intervention=False,
        resolved_config=None,
    )
    modified_request = build_generation_request(
        history=history,
        prompt=prompt,
        model_key=model_key,
        decoding=decoding,
        seed=seed,
        apply_intervention=True,
        resolved_config=resolved_config,
    )
    original_result = backend.generate(original_request)
    modified_result = backend.generate(modified_request)
    return CompareResult(
        original_text=original_result.text,
        modified_text=modified_result.text,
        original_request=original_request,
        modified_request=modified_request,
    )


def assert_compare_invariant(compare: CompareResult) -> None:
    """Raises AssertionError unless the two requests differ ONLY in
    apply_intervention/resolved_config. Used by both the app (as a
    belt-and-braces runtime check before rendering Compare) and the test
    suite (as the acceptance check itself)."""
    reconstructed_original = dataclasses.replace(
        compare.modified_request, apply_intervention=False, resolved_config=None
    )
    if reconstructed_original != compare.original_request:
        raise AssertionError(
            "Compare invariant violated: Original and Modified requests differ in a field "
            "other than apply_intervention/resolved_config."
        )


# ---------------------------------------------------------------------------
# Public vs Advanced rendering of a ResolvedControlState -- both built from
# the SAME object's own public_view()/advanced_view()/execution_dict(),
# never a second, product-derived computation of what those already
# compute. The only product-owned addition is the concept/pairing display
# label, which the canonical contract deliberately holds none of.
# ---------------------------------------------------------------------------

_STRENGTH_KEYS = {"low": "strength_low", "medium": "strength_medium", "high": "strength_high"}


def public_output_summary(resolved_config: ResolvedControlState, lang: str) -> str:
    """Model + concept + direction + strength only. Never seed, feature
    idx, sae id, layer, positions, or the authored value -- those are
    Advanced-only, per the project rule that Public mode carries no SAE
    jargon or raw values. Sourced entirely from public_view(), never a
    second read of fields public_view() itself excludes."""
    view = resolved_config.public_view()
    direction_key = "direction_amplify" if view["direction"] == "amplify" else "direction_suppress"
    strength_key = _STRENGTH_KEYS[view["strength"]]
    return (
        f"{t('output_summary_model', lang)}: {pairing_label(view['pairing_id'], lang)}\n"
        f"{t('output_summary_concept', lang)}: {concept_label(view['concept_id'], lang)}\n"
        f"{t('output_summary_direction', lang)}: {t(direction_key, lang)}\n"
        f"{t('output_summary_strength', lang)}: {t(strength_key, lang)}"
    )


def advanced_output_details(resolved_config: ResolvedControlState) -> dict[str, Any]:
    """advanced_view() (full per-target detail, provenance, evidence
    identity and fingerprints) plus the canonical execution payload
    (execution_dict(): only what the executor applies) under its own key
    -- the dispatch calls for both, and they are two distinct, deliberately
    narrower/wider canonical views, not one that subsumes the other."""
    return {
        **resolved_config.advanced_view(),
        "execution_payload": resolved_config.execution_dict(),
    }
