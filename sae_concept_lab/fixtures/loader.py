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

import dataclasses
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
from sae_concept_lab.core.runtime_acceptance import accepted_layer_for, is_mechanically_accepted
from sae_concept_lab.core.stub_backend import StubConceptLabBackend

FIXTURES_DIR = Path(__file__).resolve().parent

GEMMA_PAIRING_ID = "gemma-3-12b-it+gemma-scope-2-12b-it"
QWEN_PAIRING_ID = "qwen-3.5-27b+SAE-Res-Qwen3.5-27B-W80K-L0_100"

#: Explicit, named files only -- never a directory scan (codec.py's own
#: rule: "a build must name every entry it loads"). Adding a ninth file to
#: fixtures/gemma/ without adding it here does not make it appear anywhere.
_ENTRY_FILENAMES: dict[str, tuple[str, ...]] = {
    # The eight FAKE placeholders were removed from the build. What ships now is
    # measured: features that passed every discovery gate in all six evaluation
    # cells of a full-space scan on real weights.
    "gemma": ("pro_american_exceptionalism.json",),
    "qwen": ("pro_american_exceptionalism.json",),
}

PAIRING_ID_FOR_MODEL_KEY: dict[str, str] = {"gemma": GEMMA_PAIRING_ID, "qwen": QWEN_PAIRING_ID}

#: The bounded Mode-A import slot (2026-08-13 PI-demo dispatch): the ONE
#: location a genuinely ATTESTED bundle can be dropped into and be picked
#: up by load_entries() with NO edit to this or any other .py file. A
#: directory scan here is safe in a way it deliberately is not for
#: fixtures/{gemma,qwen}/ above: whether a scanned file's entry ever
#: PUBLISHES remains entirely evaluate_publishability's decision (ATTESTED
#: provenance, resolvable full-digest evidence, no placeholder markers),
#: never this module's, and a file that fails to even decode is excluded
#: and reported rather than crashing anything -- see load_attested_entries.
ATTESTED_DIR = FIXTURES_DIR / "attested"


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


def _attested_entry_paths(model_key: str) -> tuple[Path, ...]:
    directory = ATTESTED_DIR / model_key
    if not directory.is_dir():
        return ()
    return tuple(sorted(directory.glob("*.json")))


@dataclasses.dataclass(frozen=True)
class AttestedImportOutcome:
    """What the attested slot currently holds for one model_key.

    `entries` decoded successfully through the canonical codec (their
    provenance/evidence have NOT been checked here -- that is still
    evaluate_publishability's job, downstream). `rejected` names every
    file that failed to even decode (malformed JSON, a schema violation,
    an unsafe artifact_type -- whatever BundleDecodeError or its siblings
    report), each paired with the exact exception text, so a Lab
    Assistant staging a bundle sees precisely why a drop-in was not
    picked up instead of it silently vanishing."""

    entries: tuple[BundleEntry, ...]
    rejected: tuple[tuple[Path, str], ...] = ()


def load_attested_entries(model_key: str) -> AttestedImportOutcome:
    """Decodes every *.json file currently in the Mode-A slot
    (ATTESTED_DIR / model_key), in sorted filename order, through the SAME
    canonical codec every shipped FAKE fixture goes through. A file that
    fails to decode is excluded and reported in `.rejected` rather than
    raised -- this function is called unconditionally by load_entries()
    below, and Mode B's guarantee (the shipped FAKE fixtures always load)
    must not depend on whatever a Mode-A drop-in happens to contain."""
    entries: list[BundleEntry] = []
    rejected: list[tuple[Path, str]] = []
    for path in _attested_entry_paths(model_key):
        try:
            (entry,) = load_entry_files((path,))
            entries.append(entry)
        except Exception as exc:  # reported in .rejected, never allowed to propagate -- see docstring
            rejected.append((path, f"{type(exc).__name__}: {exc}"))
    return AttestedImportOutcome(tuple(entries), tuple(rejected))


def load_entries(model_key: str) -> tuple[BundleEntry, ...]:
    """The shipped FAKE fixtures (unconditional: this call can never fail
    because of anything in the attested slot, which is Mode B's guarantee)
    plus whatever the bounded Mode-A slot (ATTESTED_DIR / model_key)
    currently holds and could decode. Raises whatever
    canonical.concept_bundle.errors exception the codec raises on a
    malformed FAKE fixture file -- never a product-defined ValueError
    standing in for it. A malformed or tampered file in the ATTESTED slot
    never raises here; see load_attested_entries(model_key).rejected for
    diagnostics, printed by sae_concept_lab.app and the PI-demo preflight."""
    fake_entries = load_entry_files(_entry_paths(model_key))
    attested = load_attested_entries(model_key)
    for path, reason in attested.rejected:
        print(f"WARNING: attested slot entry {path} was not loaded: {reason}", file=sys.stderr)
    return (*fake_entries, *attested.entries)


def build_registry(model_key: str) -> ConceptRegistry:
    return ConceptRegistry(load_entries(model_key))


def _target_layers_for_model_key(entries: tuple[BundleEntry, ...]) -> tuple[int, ...]:
    """Every layer named by a target in any calibrated direction of
    `entries` -- the layer(s) a real backend would actually need its
    mechanism verified at to run any of them, mirroring how
    core/gemma_backend.py and core/qwen_backend.py derive `layer` from
    `require_group_from_resolved(resolved)` for one resolved request, just
    ahead of resolving any one of them. Empty only if none of this
    model_key's entries has a calibrated direction at all, which is not
    true of anything shipped today."""
    layers: set[int] = set()
    for entry in entries:
        for direction in entry.calibrated_directions:
            layers.update(entry.direction(direction).layers)
    return tuple(sorted(layers))


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

    entries = load_entries(model_key)

    # A real backend's own `pairing` attribute (QwenRuntimeBackend.pairing,
    # GemmaRuntimeBackend.pairing) is checked here by duck typing, not by
    # importing either concrete class -- this module stays decoupled from
    # specific backend implementations, matching ConceptLabBackend's own
    # Protocol design. Mechanical CODE extraction (RUNTIME_CODE_MIRROR) is
    # entirely independent of mechanical ACCEPTANCE against real weights;
    # a backend whose code imports and runs perfectly is still refused
    # here until its pairing has an attached, verified
    # RuntimeAcceptanceRecord (core/runtime_acceptance.py), SCOPED to the
    # layer this model_key's own shipped concept(s) actually target --
    # exactly the same scoping core/gemma_backend.py and
    # core/qwen_backend.py apply per generated request, worked out here
    # from the entries themselves since no single resolved request exists
    # yet at gate time.
    pairing = getattr(backend, "pairing", None)
    if pairing is not None:
        target_layers = _target_layers_for_model_key(entries)
        if target_layers:
            unaccepted_layers = tuple(
                layer for layer in target_layers if not is_mechanically_accepted(pairing, layer)
            )
            if unaccepted_layers:
                raise ReleaseGateError(
                    f"refusing --mode release: backend for model_key={model_key!r} is a real "
                    f"backend for pairing={pairing!r}, but its mechanical-acceptance record "
                    f"(core/runtime_acceptance.py) is scoped to layer "
                    f"{accepted_layer_for(pairing)!r}, not layer(s) {list(unaccepted_layers)} -- "
                    f"the layer(s) this model_key's own shipped concept(s) actually target. "
                    "Mechanical acceptance against real weights has not been imported for this "
                    "layer from a tracked qwen-sae-interp evidence commit yet."
                )
        elif not is_mechanically_accepted(pairing):
            raise ReleaseGateError(
                f"refusing --mode release: backend for model_key={model_key!r} is a real backend for "
                f"pairing={pairing!r}, but that pairing has no attached, verified "
                "RuntimeAcceptanceRecord (core/runtime_acceptance.py) -- mechanical acceptance against "
                "real weights has not been imported from a tracked qwen-sae-interp evidence commit yet."
            )

    root = _validate_evidence_registry_root(evidence_registry_root)
    registry = RepositoryEvidenceRegistry(root=root)

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
