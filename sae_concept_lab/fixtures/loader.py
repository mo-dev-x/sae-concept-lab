"""Loads the JSON public-preset bundles in this directory and enforces the
fail-closed release gate. Pure I/O + validation -- no gradio import -- but
DOES import the concrete StubConceptLabBackend, because the release gate's
whole job is to know about that one specific fake implementation and
refuse it by identity, not just by trusting whatever a bundle's JSON claims
about itself. Safe to call from a CPU-only test or the login-node side of
any future deploy script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

from sae_concept_lab.core.protocol import ConceptLabBackend, PositionsMode
from sae_concept_lab.core.stub_backend import StubConceptLabBackend

FIXTURES_DIR = Path(__file__).resolve().parent

KNOWN_MODEL_KEYS = ("gemma", "qwen")
SUPPORTED_POSITIONS_VALUES = get_args(PositionsMode)

REQUIRED_BUNDLE_FIELDS = (
    "is_synthetic",
    "release_blocked",
    "model_key",
    "model_label",
    "sae_id",
    "layer",
    "hook_point",
    "positions_default",
    "decoding_default",
    "seed_default",
    "random_feature_control_id",
    "concepts",
)

REQUIRED_CONCEPT_FIELDS = (
    "concept_id",
    "label",
    "description",
    "feature_id",
    "feature_weight",
    "strength_coefficients",
)


class ReleaseGateError(RuntimeError):
    """Raised when a release/public launch is requested against a bundle
    marked synthetic and/or release_blocked, OR against the stub backend
    regardless of what the bundle claims. There is deliberately no
    override parameter on enforce_release_gate() -- fixing this means
    swapping in both a real, non-synthetic bundle AND a real backend, not
    passing a flag or editing a JSON file."""


def load_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"public-preset bundle not found at {path}")
    bundle = json.loads(path.read_text(encoding="utf-8"))

    missing = [f for f in REQUIRED_BUNDLE_FIELDS if f not in bundle]
    if missing:
        raise ValueError(f"bundle {path} is missing required field(s): {missing}")
    if not bundle["concepts"]:
        raise ValueError(f"bundle {path} has zero concepts")
    for concept in bundle["concepts"]:
        concept_missing = [f for f in REQUIRED_CONCEPT_FIELDS if f not in concept]
        if concept_missing:
            raise ValueError(
                f"bundle {path} concept {concept.get('concept_id')!r} is missing "
                f"field(s): {concept_missing}"
            )

    for flag_name in ("is_synthetic", "release_blocked"):
        if not isinstance(bundle[flag_name], bool):
            raise ValueError(
                f"bundle {path} field {flag_name!r} must be a boolean, got {bundle[flag_name]!r}"
            )
    if bundle["model_key"] not in KNOWN_MODEL_KEYS:
        raise ValueError(
            f"bundle {path} has model_key={bundle['model_key']!r}; expected one of {KNOWN_MODEL_KEYS}"
        )
    if bundle["positions_default"] not in SUPPORTED_POSITIONS_VALUES:
        raise ValueError(
            f"bundle {path} has positions_default={bundle['positions_default']!r}; "
            f"expected one of {SUPPORTED_POSITIONS_VALUES}"
        )
    seed_default = bundle["seed_default"]
    if not isinstance(seed_default, int) or isinstance(seed_default, bool) or seed_default < 0:
        raise ValueError(f"bundle {path} has seed_default={seed_default!r}; expected a non-negative int")

    return bundle


def enforce_release_gate(bundle: dict[str, Any], *, mode: str, backend: ConceptLabBackend) -> None:
    """Fail closed: refuses outright (raises ReleaseGateError) if `mode`
    is "release" and EITHER the backend is the known stub implementation
    OR the bundle is synthetic and/or release_blocked. `backend` is a
    required argument -- there is no bundle-only call path left, precisely
    because a bundle's JSON flags alone were never sufficient evidence
    that real data is actually in play. `mode` == "dev" never raises here
    -- dev mode is exactly where fake data is expected to run."""
    if mode != "release":
        return
    if isinstance(backend, StubConceptLabBackend):
        raise ReleaseGateError(
            f"refusing --mode release: backend for model_key={bundle.get('model_key')!r} is "
            "StubConceptLabBackend (deterministic fake data), regardless of the bundle's "
            "is_synthetic/release_blocked flags. Marking a bundle's JSON is_synthetic=false and "
            "release_blocked=false is NOT sufficient on its own to reach release mode -- a real, "
            "non-stub backend implementing ConceptLabBackend.generate() is also required."
        )
    if bundle.get("is_synthetic") or bundle.get("release_blocked"):
        raise ReleaseGateError(
            f"refusing --mode release: bundle for model_key={bundle.get('model_key')!r} has "
            f"is_synthetic={bundle.get('is_synthetic')!r}, "
            f"release_blocked={bundle.get('release_blocked')!r}. Fake data must never reach a "
            "release/public launch -- load a real, non-synthetic bundle instead."
        )


def default_bundle_path(model_key: str) -> Path:
    if model_key not in ("gemma", "qwen"):
        raise ValueError(f"unknown model_key {model_key!r}; expected 'gemma' or 'qwen'")
    return FIXTURES_DIR / f"{model_key}_stub_bundle.json"
