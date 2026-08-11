"""The narrow seam between this UI and any real backend.

Nothing outside this module should need to know whether generation came
from `StubConceptLabBackend` (fixtures/, fully synthetic, CPU-only) or a
future real backend wired to actual model+SAE weights. A real backend is
just another class implementing `ConceptLabBackend.generate()` -- the UI
layer (ui/) and the pure turn-taking logic (core/logic.py) only ever hold
a `ConceptLabBackend`, never a concrete class.

This also doubles as the answer to "keep integration through a narrow
adapter": at the time this was built, no bundle/resolution contract
existed anywhere in this repo (a reserved worktree/branch for it existed
but carried zero commits beyond main) -- so there was nothing to adapt to
yet. `ResolvedConfig` below is deliberately small and flat so that
whichever real contract lands later can be mapped onto it (or this file
replaced by an adapter to it) without touching ui/ or core/logic.py.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Protocol

Direction = Literal["amplify", "suppress"]
StrengthLevel = Literal["low", "medium", "high"]
PositionsMode = Literal["generated_only", "all"]


@dataclasses.dataclass(frozen=True)
class ResolvedConfig:
    """The one object Public mode and the Advanced accordion both read.

    Public renders a small, friendly subset of this (model + concept +
    direction + strength -- see core.logic.public_output_summary).
    Advanced renders this object in full (core.logic.advanced_output_details).
    Neither view ever recomputes its own copy -- there is exactly one
    ResolvedConfig per generation, constructed once by core.config.resolve_config.

    is_synthetic is always True for every fixture-backed bundle this
    build ships with; a real backend's resolver would set it False.
    """

    model_key: str
    model_label: str
    concept_id: str
    concept_label_i18n: dict[str, str]
    concept_description_i18n: dict[str, str]
    direction: Direction
    strength_level: StrengthLevel
    strength_coefficient: float
    seed: int
    positions: PositionsMode
    hook_point: str
    sae_id: str
    layer: int
    feature_id: str
    feature_weight: float
    random_feature_control_id: str
    decoding: dict[str, Any]
    is_synthetic: bool
    diagnostics: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class GenerationRequest:
    """Everything a backend needs for one turn. `history` is serialized
    as a tuple of (role, content) pairs (hashable/comparable, unlike a
    list of dicts) specifically so the Compare invariant test can assert
    structural equality between the Original and Modified requests.

    `apply_intervention` and `resolved_config` are the ONLY two fields
    that are allowed to differ between a Compare pair's two requests --
    resolved_config is None exactly when apply_intervention is False
    (the Original/baseline arm has no concept applied, so there is
    nothing to resolve)."""

    history: tuple[tuple[str, str], ...]
    prompt: str
    model_key: str
    decoding: dict[str, Any]
    seed: int
    apply_intervention: bool
    resolved_config: ResolvedConfig | None


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    text: str
    is_synthetic: bool
    resolved_config: ResolvedConfig | None


class ConceptLabBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
