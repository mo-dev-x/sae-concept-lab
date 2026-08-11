"""Deterministic control resolver.

ONE resolver produces the state used by BOTH modes. Public mode exposes
(concept, pairing, direction, strength) and nothing else; Advanced mode exposes
the same object with more of its fields rendered. They are not two code paths,
so they cannot disagree about what a control does.

Determinism: given the same entry, the same (direction, strength) and the same
denominators, this returns byte-identical state, including target order and
fingerprints. Nothing here reads a clock, a random source, or the filesystem.

RESOLUTION ARITHMETIC, exactly:

  ABLATE   no arithmetic at all. There is no dose to scale, so no value, no
           unit, no denominator and no product -- an ablation request carries
           only which features to zero.

  CLAMP    absolute = spec.value x target.weight x denominator(unit,
           unit_source, target)

           ABSOLUTE_ACTIVATION has no denominator source: the value is already
           an activation, so the denominator is 1.0 and the value is used
           directly before target weighting.

The denominator for a multiple unit is NOT in the contract and never will be.
A sample maximum and a corpus maximum are measurements, they differ per feature,
per SAE and per pairing, and holding stale copies of them next to a dose is how
a number silently stops meaning what it says. The caller supplies them, and a
caller that supplies none gets a refusal rather than a default.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol, runtime_checkable

from .errors import (
    InvalidDenominatorError,
    MissingDenominatorError,
    UnknownConceptError,
    UnknownPairingError,
)
from .schema import (
    BundleEntry,
    CalibrationProvenance,
    Direction,
    Operation,
    PositionMode,
    Provenance,
    Spec,
    Strength,
    Target,
    Unit,
    canonical_target_sort_key,
)

logger = logging.getLogger(__name__)

UNIT_CAVEAT = (
    "A multiple-unit value is a multiple of THAT target's own measured maximum, "
    "on THAT pairing, from the source named in unit_source. The same number does "
    "NOT mean the same thing across pairings, across SAEs, or between a sample "
    "maximum and a corpus maximum: each is interpretable only against the "
    "specific maximum it was divided by, and those maxima differ in how they were "
    "estimated. Do not compare values across entries."
)

# SUPPRESSION IS NOT A NEGATED AMPLIFICATION.
# There is deliberately no direction->sign table here. Each direction owns its
# own targets and its own three specs, and the resolver reads the record for the
# direction it was asked for rather than deriving one direction from the other.
# Collapsing the two into a signed scalar would assert that the response is
# symmetric about baseline -- an empirical claim nothing in this contract has
# established, and not one a schema should smuggle in as a convenience.


@runtime_checkable
class DenominatorSource(Protocol):
    """Supplies the measured maximum a multiple-unit value is taken against.

    Called only for `sample_max_multiple` and `corpus_max_multiple`. Must return
    a positive finite activation. Raising `LookupError` for an unknown target is
    supported and is reported as a missing denominator.
    """

    def __call__(self, *, unit: Unit, unit_source: str, target: Target) -> float: ...


class MappingDenominatorSource:
    """A denominator source backed by an explicit dict.

    Keyed by (unit, unit_source, sae_id, layer, feature_idx), so a maximum
    measured one way cannot be silently reused for a value that names a
    different source.
    """

    def __init__(self, values: Mapping[tuple[Unit, str, str, int, int], float]) -> None:
        self._values = dict(values)

    def __call__(self, *, unit: Unit, unit_source: str, target: Target) -> float:
        key = (unit, unit_source, target.sae_id, target.layer, target.feature_idx)
        if key not in self._values:
            raise LookupError(key)
        return self._values[key]


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """One target with its strength already applied.

    `absolute_value` is UNSIGNED and, for a clamp, is in raw activation units:
    the arithmetic above has already been done, so the executor does not repeat
    it and cannot repeat it differently. It is None for an ablation, which has
    no dose. The direction lives on the parent state and is never encoded in
    this number's sign.

    `denominator` is recorded next to the value it produced so an audit can
    reconstruct the arithmetic without holding the maxima the run used.
    """

    sae_id: str
    layer: int
    feature_idx: int
    weight: float
    operation: Operation
    unit: Unit | None
    unit_source: str | None
    denominator: float | None
    absolute_value: float | None

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.sae_id, self.layer, self.feature_idx)

    @property
    def group_key(self) -> tuple[str, int]:
        return (self.sae_id, self.layer)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sae_id": self.sae_id,
            "layer": self.layer,
            "feature_idx": self.feature_idx,
            "weight": self.weight,
            "operation": self.operation.value,
            "unit": None if self.unit is None else self.unit.value,
            "unit_source": self.unit_source,
            "denominator": self.denominator,
            "absolute_value": self.absolute_value,
        }

    def execution_dict(self) -> dict[str, Any]:
        """Only what the executor applies. Derivation is excluded."""
        return {
            "sae_id": self.sae_id,
            "layer": self.layer,
            "feature_idx": self.feature_idx,
            "absolute_value": self.absolute_value,
        }


@dataclass(frozen=True, slots=True)
class ResolvedControlState:
    """Immutable, fully-specified description of one control setting.

    Everything the executor needs, everything the UI needs to render, and
    everything an audit needs to reconstruct what was run -- in one object.
    Publishability is NOT computed here: that is the release gate's
    responsibility, and duplicating the decision would create two answers.
    """

    concept_id: str
    pairing_id: str
    direction: Direction
    strength: Strength
    #: Every direction this entry HAS calibrated, not just the resolved one.
    #: Carried here so the UI can disable the other control from the same object
    #: it renders this one from, rather than re-deriving availability and owning
    #: a second copy of the policy. A resolved state for one direction of a
    #: one-direction concept is normal, not degraded.
    available_directions: tuple[Direction, ...]
    operation: Operation
    unit: Unit | None
    unit_source: str | None
    value: float | None
    positions: PositionMode
    provenance: Provenance
    calibration_provenance: CalibrationProvenance | None
    targets: tuple[ResolvedTarget, ...]
    entry_audit_fingerprint: str
    schema_version: str

    # -- derived ---------------------------------------------------------
    @property
    def is_ablation(self) -> bool:
        return self.operation is Operation.ABLATE

    @property
    def unavailable_directions(self) -> tuple[Direction, ...]:
        """Directions this entry did not calibrate. The UI renders these
        disabled; selecting one is refused by the contract, never resolved into
        an empty request."""
        return tuple(d for d in Direction if d not in self.available_directions)

    @property
    def is_single_direction_concept(self) -> bool:
        return len(self.available_directions) == 1

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({t.layer for t in self.targets}))

    @property
    def sae_ids(self) -> tuple[str, ...]:
        return tuple(sorted({t.sae_id for t in self.targets}))

    @property
    def group_keys(self) -> tuple[tuple[str, int], ...]:
        """The (sae_id, layer) groups this state resolves to. More than one is a
        runtime question, asked by `runtime.require_single_execution_group`."""
        return tuple(sorted({t.group_key for t in self.targets},
                            key=lambda g: (g[1], g[0])))

    @property
    def evidence_identity(self) -> tuple[tuple[str, str], ...]:
        """(artifact_type, artifact_hash) for every cited artifact."""
        if self.calibration_provenance is None:
            return ()
        return tuple((e.artifact_type, e.artifact_hash)
                     for e in self.calibration_provenance.evidence)

    # -- views for the two modes ----------------------------------------
    def public_view(self) -> dict[str, Any]:
        """What Public mode may show.

        Includes `pairing_id` so an output can state which model and SAE
        produced a generation -- one concept may be authored against several
        pairings, and an unlabelled result is not attributable.

        Deliberately excludes feature indices, SAE ids, layers, weights and
        doses -- not because they are secret, but because Public mode's contract
        is a named concept on a named pairing at a strength, and leaking
        internals into it would make the two modes promise different things.

        There is no display name here because the contract holds none; see the
        module note in `schema.py` on where human-facing text must come from.
        """
        return {
            "concept_id": self.concept_id,
            "pairing_id": self.pairing_id,
            "direction": self.direction.value,
            "strength": self.strength.value,
            # so the other control can be greyed out without a second policy
            "available_directions": [d.value for d in self.available_directions],
            "unavailable_directions": [d.value for d in self.unavailable_directions],
        }

    def advanced_view(self) -> dict[str, Any]:
        """Public view plus full per-target detail, provenance and evidence
        identity. The same underlying state object -- not a second code path."""
        cp = self.calibration_provenance
        return {
            **self.public_view(),
            "operation": self.operation.value,
            "unit": None if self.unit is None else self.unit.value,
            "unit_source": self.unit_source,
            "unit_caveat": UNIT_CAVEAT,
            "authored_value": self.value,
            "is_ablation": self.is_ablation,
            "positions": self.positions.value,
            "provenance": self.provenance.value,
            "calibration_provenance": None if cp is None else cp.as_dict(),
            "evidence_identity": [list(x) for x in self.evidence_identity],
            "n_targets": self.n_targets,
            "layers": list(self.layers),
            "sae_ids": list(self.sae_ids),
            "groups": [list(g) for g in self.group_keys],
            "targets": [t.as_dict() for t in self.targets],
            "entry_audit_fingerprint": self.entry_audit_fingerprint,
            "state_fingerprint": self.state_fingerprint(),
            "execution_fingerprint": self.execution_fingerprint(),
            "schema_version": self.schema_version,
        }

    def as_dict(self) -> dict[str, Any]:
        """FULL AUDIT STATE, including provenance and evidence identity."""
        cp = self.calibration_provenance
        return {
            "concept_id": self.concept_id,
            "pairing_id": self.pairing_id,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "available_directions": [d.value for d in self.available_directions],
            "operation": self.operation.value,
            "unit": None if self.unit is None else self.unit.value,
            "unit_source": self.unit_source,
            "value": self.value,
            "positions": self.positions.value,
            "provenance": self.provenance.value,
            "calibration_provenance": None if cp is None else cp.as_dict(),
            "targets": [t.as_dict() for t in self.targets],
            "entry_audit_fingerprint": self.entry_audit_fingerprint,
            "schema_version": self.schema_version,
        }

    def execution_dict(self) -> dict[str, Any]:
        """ONLY what determines what the model computes.

        Per target: identity and the already-absolute value. The authored value,
        the weight, the unit, the unit source and the denominator are the
        DERIVATION of that number, not the number, and two runs that arrived at
        the same absolute clamp by different routes drive the model identically.

        Excluded for the same reason: concept id, provenance, evidence,
        strength, the schema version and the audit fingerprint. None of them
        changes an activation. Including any would make two identical
        computations look different, which defeats the only purpose an execution
        identity has.

        Note that every strength of an ABLATE direction produces the same
        execution dict, because ablation has no dose. That is the correct
        answer, not a collision.
        """
        return {
            "pairing_id": self.pairing_id,
            "direction": self.direction.value,
            "operation": self.operation.value,
            "positions": self.positions.value,
            "targets": [t.execution_dict() for t in self.targets],
        }

    def state_fingerprint(self) -> str:
        """FULL AUDIT IDENTITY of this resolved state, including provenance.

        Answers "is this the same resolved artefact, evidence included?".
        Correcting a calibrator's name changes this. Use
        `execution_fingerprint()` if you mean "would this compute the same
        thing".
        """
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def execution_fingerprint(self) -> str:
        """EXECUTION IDENTITY: digest over computation-affecting fields only.

        Two resolutions sharing this fingerprint drive the model identically,
        whatever their provenance says. `execution_dict` is deliberately narrow
        so the claim is true rather than approximately true.
        """
        payload = json.dumps(self.execution_dict(), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _denominator_for(spec: Spec, target: Target,
                     denominators: DenominatorSource | None) -> float:
    """The measured maximum this target's value is a multiple of.

    1.0 for `absolute_activation`, which has no denominator source at all --
    that is not a neutral default standing in for a missing measurement, it is
    the arithmetic identity for a value that is already an activation.
    """
    if not spec.needs_denominator:
        return 1.0
    # Past this point the schema guarantees unit and unit_source are set: only a
    # CLAMP spec with a multiple unit reaches here, and both are required there.
    if denominators is None:
        raise MissingDenominatorError(
            f"spec is expressed in {spec.unit.value!r} against source "
            f"{spec.unit_source!r}, so resolving it needs the measured maximum "
            f"for sae {target.sae_id!r} layer {target.layer} feature "
            f"{target.feature_idx}. No denominator source was supplied. The "
            f"contract holds no activation statistics and will not default to "
            f"1.0, which would reinterpret a multiple as a raw activation.")
    try:
        raw = denominators(unit=spec.unit, unit_source=spec.unit_source, target=target)
    except LookupError as exc:
        raise MissingDenominatorError(
            f"denominator source has no entry for unit {spec.unit.value!r}, "
            f"source {spec.unit_source!r}, sae {target.sae_id!r}, layer "
            f"{target.layer}, feature {target.feature_idx}: {exc}") from exc
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
            or not isfinite(float(raw)) or float(raw) <= 0.0:
        raise InvalidDenominatorError(
            f"denominator for unit {spec.unit.value!r}, source "
            f"{spec.unit_source!r}, sae {target.sae_id!r}, layer {target.layer}, "
            f"feature {target.feature_idx} must be a positive finite number, got "
            f"{raw!r}")
    return float(raw)


def resolve_control(
    entry: BundleEntry,
    *,
    direction: Direction | str,
    strength: Strength | str,
    denominators: DenominatorSource | None = None,
) -> ResolvedControlState:
    """Resolves (entry, direction, strength) into one immutable state.

    Raises `DirectionNotCalibratedError` if the direction is null on this entry.

    Does NOT enforce runtime capability or publishability. Those are separate
    gates, called explicitly, so Advanced mode can inspect a multi-group or fake
    entry without the resolver deciding on its behalf what may be looked at.
    """
    d = Direction(direction)
    s = Strength(strength)

    record = entry.direction(d)          # raises the contract's refusal if null
    spec = record.spec(s)

    resolved: list[ResolvedTarget] = []
    for target in sorted(record.targets, key=canonical_target_sort_key):
        if spec.is_ablation:
            # No arithmetic. An ablation request names features, nothing else.
            resolved.append(ResolvedTarget(
                sae_id=target.sae_id, layer=target.layer,
                feature_idx=target.feature_idx, weight=float(target.weight),
                operation=spec.operation, unit=None, unit_source=None,
                denominator=None, absolute_value=None))
            continue

        # spec.value is non-None here: a non-ablating spec is a CLAMP, and the
        # schema requires a positive value on every CLAMP.
        denominator = _denominator_for(spec, target, denominators)
        absolute = float(spec.value) * float(target.weight) * denominator
        resolved.append(ResolvedTarget(
            sae_id=target.sae_id, layer=target.layer,
            feature_idx=target.feature_idx, weight=float(target.weight),
            operation=spec.operation, unit=spec.unit, unit_source=spec.unit_source,
            denominator=denominator, absolute_value=absolute))
        logger.info(
            "resolved clamp concept=%s pairing=%s direction=%s strength=%s "
            "sae=%s layer=%d feature=%d value=%s weight=%s denominator=%s "
            "unit=%s unit_source=%s absolute_value=%s",
            entry.concept_id, entry.pairing_id, d.value, s.value,
            target.sae_id, target.layer, target.feature_idx, spec.value,
            target.weight, denominator, spec.unit.value if spec.unit else None,
            spec.unit_source, absolute)

    return ResolvedControlState(
        concept_id=entry.concept_id,
        pairing_id=entry.pairing_id,
        direction=d,
        strength=s,
        available_directions=entry.calibrated_directions,
        operation=spec.operation,
        unit=spec.unit,
        unit_source=spec.unit_source,
        value=None if spec.value is None else float(spec.value),
        positions=entry.positions,
        provenance=entry.provenance,
        calibration_provenance=entry.calibration_provenance,
        targets=tuple(resolved),
        entry_audit_fingerprint=entry.audit_fingerprint(),
        schema_version=entry.schema_version,
    )


class ConceptRegistry:
    """An in-memory, read-only collection of entries keyed by (concept, pairing).

    Pure data: it loads nothing from disk. Whoever owns persistence hands it
    entries. The key is the pair because one concept on two pairings is two
    independent entries with nothing shared between them.
    """

    def __init__(self, entries: tuple[BundleEntry, ...] = ()) -> None:
        seen: dict[tuple[str, str], BundleEntry] = {}
        for e in entries:
            if e.key in seen:
                raise ValueError(
                    f"duplicate entry for concept {e.concept_id!r} on pairing "
                    f"{e.pairing_id!r}")
            seen[e.key] = e
        self._entries = seen

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._entries))

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(sorted({cid for cid, _ in self._entries}))

    def pairings_for(self, concept_id: str) -> tuple[str, ...]:
        pairings = tuple(sorted(pid for cid, pid in self._entries if cid == concept_id))
        if not pairings:
            raise UnknownConceptError(
                f"no bundle entry registered for concept {concept_id!r}; "
                f"available: {list(self.concept_ids)}")
        return pairings

    def get(self, concept_id: str, pairing_id: str) -> BundleEntry:
        try:
            return self._entries[(concept_id, pairing_id)]
        except KeyError:
            raise UnknownPairingError(
                f"no bundle entry for concept {concept_id!r} on pairing "
                f"{pairing_id!r}; available: {list(self.keys)}"
            ) from None

    def resolve(self, *, concept_id: str, pairing_id: str,
                direction: Direction | str, strength: Strength | str,
                denominators: DenominatorSource | None = None) -> ResolvedControlState:
        return resolve_control(self.get(concept_id, pairing_id), direction=direction,
                               strength=strength, denominators=denominators)
