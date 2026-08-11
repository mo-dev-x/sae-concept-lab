"""Turns a bundle + the user's public selections into exactly one
ResolvedConfig -- the shared-state object Public and Advanced both read.
Pure function, no I/O, no Gradio import: resolve_config() is called once
per generation, and its result is threaded through everywhere else
unchanged (never recomputed a second time for Advanced's display)."""

from __future__ import annotations

from typing import Any

from sae_concept_lab.core.protocol import Direction, PositionsMode, ResolvedConfig, StrengthLevel


def find_concept(bundle: dict[str, Any], concept_id: str) -> dict[str, Any]:
    for concept in bundle["concepts"]:
        if concept["concept_id"] == concept_id:
            return concept
    raise KeyError(f"concept_id {concept_id!r} not found in bundle for model_key={bundle.get('model_key')!r}")


def resolve_config(
    *,
    bundle: dict[str, Any],
    concept_id: str,
    direction: Direction,
    strength_level: StrengthLevel,
    seed: int | None = None,
    positions: PositionsMode | None = None,
) -> ResolvedConfig:
    """positions defaults to the bundle's own positions_default -- never a
    literal hardcoded in this function or in the UI layer, per the
    instruction to source the public value from configuration rather than
    hardcoding a scientific choice. Passing an explicit `positions`
    (Advanced-only) overrides it."""
    concept = find_concept(bundle, concept_id)
    coefficient = concept["strength_coefficients"][strength_level]
    resolved_seed = bundle["seed_default"] if seed is None else seed
    resolved_positions = bundle["positions_default"] if positions is None else positions
    is_synthetic = bool(bundle.get("is_synthetic", False))

    # "synthetic" here is deliberately the SAME value as ResolvedConfig.is_synthetic
    # below, derived once from the bundle -- never a separate literal. A future
    # real bundle/adapter that sets is_synthetic=False must not have to remember
    # a second hardcoded flag to flip; there is only ever one source of truth.
    diagnostics = {
        "synthetic": is_synthetic,
        "delta_norm_placeholder": round(coefficient * 0.7, 4),
        "residual_norm_placeholder": 1.0,
        "note": {
            "en": (
                "Placeholder for future real per-run diagnostics (e.g. delta/residual norms "
                "from interplab.interventions.hooks.CallStats). Not computed from any real "
                "forward pass."
            ),
            "fr": (
                "Espace réservé pour de futurs diagnostics réels par exécution (p. ex. normes "
                "delta/résiduelles). N'est calculé à partir d'aucune passe avant réelle."
            ),
        },
    }

    return ResolvedConfig(
        model_key=bundle["model_key"],
        model_label=bundle["model_label"],
        concept_id=concept["concept_id"],
        concept_label_i18n=dict(concept["label"]),
        concept_description_i18n=dict(concept["description"]),
        direction=direction,
        strength_level=strength_level,
        strength_coefficient=coefficient,
        seed=resolved_seed,
        positions=resolved_positions,
        hook_point=bundle["hook_point"],
        sae_id=bundle["sae_id"],
        layer=bundle["layer"],
        feature_id=concept["feature_id"],
        feature_weight=concept["feature_weight"],
        random_feature_control_id=bundle["random_feature_control_id"],
        decoding=dict(bundle["decoding_default"]),
        is_synthetic=is_synthetic,
        diagnostics=diagnostics,
    )
