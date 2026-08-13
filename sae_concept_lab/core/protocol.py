"""The narrow seam between this UI and any real backend.

Nothing outside this module should need to know whether generation came
from `StubConceptLabBackend` (fixtures/, fully synthetic, CPU-only) or a
future real backend wired to actual model+SAE weights. A real backend is
just another class implementing `ConceptLabBackend.generate()` -- the UI
layer (ui/) and the pure turn-taking logic (core/logic.py) only ever hold
a `ConceptLabBackend`, never a concrete class.

`resolved_config` on both dataclasses below is the canonical
`ResolvedControlState` (sae_concept_lab.canonical.concept_bundle.resolver):
the product-only `ResolvedConfig` dataclass this module used to define
here is retired now that sae_concept_lab.fixtures.loader resolves
controls through the canonical package directly (the bundle/resolution
contract this module's docstring once said did not exist yet). This
module does not construct a ResolvedControlState; it only names the type
its own dataclasses carry, so ui/ and core/logic.py keep depending on a
Protocol rather than a concrete backend.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

from sae_concept_lab.canonical.concept_bundle import ResolvedControlState


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
    resolved_config: ResolvedControlState | None


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    text: str
    is_synthetic: bool
    resolved_config: ResolvedControlState | None
    #: Backend-produced diagnostics: requested/resolved/backend-received
    #: target, model/SAE/hook identity, activation before/after, residual
    #: delta, prefill/decode hook counts, execution/audit fingerprints.
    #: None for StubConceptLabBackend and for the Compare baseline arm
    #: (apply_intervention=False -- there is no intervention to diagnose).
    #: A real backend populates this; nothing here is computed by this
    #: module -- see core/qwen_backend.py / core/gemma_backend.py.
    diagnostics: dict[str, Any] | None = None


class ConceptLabBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
