"""Loads this repository's FAKE-marked canonical concept-bundle documents
and enforces the fail-closed release gate.

Every validation decision -- is this document well-formed, is this entry
publishable, does this evidence reference resolve -- is made by
sae_concept_lab.canonical.concept_bundle (codec.py, release.py,
evidence.py). This module never re-implements any of it: it only names
which explicit files belong to which pairing (codec.py's load_entry_files
takes a list, never a directory scan) and wires the canonical release
gate to this product's own StubConceptLabBackend identity check and
evidence_registry_root configuration -- the two things the canonical
package cannot know about, because it has no concept of "the product's
stub backend" or "where this deployment's registry lives".
"""

from __future__ import annotations

import sys
from pathlib import Path

from sae_concept_lab.canonical.concept_bundle import (
    BundleEntry,
    ConceptRegistry,
    Exposure,
    RepositoryEvidenceRegistry,
    assert_release_text_clean,
    evaluate_publishability,
    load_entry_files,
    select_layout_entries,
)
from sae_concept_lab.core.protocol import ConceptLabBackend
from sae_concept_lab.core.stub_backend import StubConceptLabBackend

FIXTURES_DIR = Path(__file__).resolve().parent

GEMMA_PAIRING_ID = "fake-gemma-demo-pairing"
QWEN_PAIRING_ID = "fake-qwen-demo-pairing"

#: Explicit, named files only -- never a directory scan (codec.py's own
#: rule: "a build must name every entry it loads"). Adding a ninth file to
#: fixtures/gemma/ without adding it here does not make it appear anywhere.
_ENTRY_FILENAMES: dict[str, tuple[str, ...]] = {
    "gemma": ("warmth.json", "formality.json", "enthusiasm.json", "caution.json"),
    "qwen": ("curiosity.json", "directness.json", "playfulness.json", "skepticism.json"),
}

PAIRING_ID_FOR_MODEL_KEY: dict[str, str] = {"gemma": GEMMA_PAIRING_ID, "qwen": QWEN_PAIRING_ID}


class ReleaseGateError(RuntimeError):
    """Raised when a release/public launch is requested against the stub
    backend, or against an entry set with no publishable concept, or
    against a missing/invalid evidence_registry_root. There is
    deliberately no override parameter -- fixing this means wiring in a
    real backend, a real registry, and a genuinely ATTESTED entry, not
    passing a flag."""


def _entry_paths(model_key: str) -> tuple[Path, ...]:
    if model_key not in _ENTRY_FILENAMES:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {sorted(_ENTRY_FILENAMES)}")
    return tuple(FIXTURES_DIR / model_key / name for name in _ENTRY_FILENAMES[model_key])


def load_entries(model_key: str) -> tuple[BundleEntry, ...]:
    """Strictly decodes this model's explicit entry files via the canonical
    codec. Raises whatever canonical.concept_bundle.errors exception the
    codec raises on a malformed document -- never a product-defined
    ValueError standing in for it."""
    return load_entry_files(_entry_paths(model_key))


def build_registry(model_key: str) -> ConceptRegistry:
    return ConceptRegistry(load_entries(model_key))


def _validate_evidence_registry_root(evidence_registry_root: Path | str | None) -> Path:
    """Fail-closed pre-flight, independent of (and prior to) any specific
    evidence reference: an absent, missing, unreadable, or empty root is
    refused before canonical evidence resolution is even attempted, so the
    operator sees which of these it was rather than a generic
    'not publishable'."""
    if evidence_registry_root is None:
        raise ReleaseGateError(
            "refusing --mode release: no evidence_registry_root was supplied. Release mode "
            "requires an explicit, existing, readable, non-empty registry directory to resolve "
            "evidence references against. Dev mode may omit it -- dev mode never evaluates "
            "publishability."
        )
    root = Path(evidence_registry_root)
    if not root.exists():
        raise ReleaseGateError(f"refusing --mode release: evidence_registry_root {root} does not exist")
    if not root.is_dir():
        raise ReleaseGateError(f"refusing --mode release: evidence_registry_root {root} is not a directory")
    try:
        contents = list(root.iterdir())
    except OSError as exc:
        raise ReleaseGateError(
            f"refusing --mode release: evidence_registry_root {root} is unreadable: {exc}"
        ) from exc
    if not contents:
        raise ReleaseGateError(f"refusing --mode release: evidence_registry_root {root} is empty")
    return root


def enforce_release_gate(
    *,
    mode: str,
    backend: ConceptLabBackend,
    model_key: str,
    evidence_registry_root: Path | str | None = None,
) -> None:
    """Fail closed: refuses outright (raises ReleaseGateError) if `mode` is
    "release" and EITHER the backend is the known stub implementation, OR
    the evidence_registry_root is absent/missing/unreadable/empty, OR no
    entry for this model_key is publishable against it (which subsumes
    "unresolved": an unresolved evidence reference is one of the reasons
    evaluate_publishability collects). `mode` == "dev" never raises here.
    """
    if mode != "release":
        return
    if isinstance(backend, StubConceptLabBackend):
        raise ReleaseGateError(
            f"refusing --mode release: backend for model_key={model_key!r} is "
            "StubConceptLabBackend (deterministic fake data), regardless of any entry's "
            "provenance. A real, non-stub backend implementing ConceptLabBackend.generate() is "
            "required before evidence/publishability is even worth checking."
        )

    root = _validate_evidence_registry_root(evidence_registry_root)
    registry = RepositoryEvidenceRegistry(root=root)
    entries = load_entries(model_key)

    selection = select_layout_entries(entries, exposure=Exposure.RELEASE, evidence_registry=registry)
    if not selection:
        per_entry_reasons = []
        for entry in entries:
            decision = evaluate_publishability(entry, evidence_registry=registry)
            per_entry_reasons.append(f"{entry.concept_id}: {'; '.join(decision.reasons)}")
        raise ReleaseGateError(
            f"refusing --mode release: no publishable concept entries for model_key={model_key!r} "
            f"against evidence_registry_root={root}. " + " | ".join(per_entry_reasons)
        )

    header = (
        f"RELEASE DIAGNOSTICS model_key={model_key!r}: {len(selection)} publishable "
        f"entr{'y' if len(selection) == 1 else 'ies'} against evidence_registry_root={root}. "
        "The wording and every label below are sae_concept_lab.canonical.concept_bundle's own "
        "(release.ReleaseDecision.render_release_evidence_note()) -- this adapter prints them "
        "verbatim and never composes its own claim about what was verified."
    )
    # Belt-and-suspenders on every line this build shows: assert_release_text_clean is
    # canonical's own over-claim detector, run here on both the header this adapter
    # writes and the note canonical writes, rather than either being trusted by eye.
    assert_release_text_clean(header)
    print(header, file=sys.stderr)
    for layout_entry in selection:
        decision = evaluate_publishability(layout_entry.entry, evidence_registry=registry)
        note = decision.render_release_evidence_note()
        assert_release_text_clean(note)
        print(note, file=sys.stderr)
