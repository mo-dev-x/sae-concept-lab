"""Concept-bundle schema -- pure Python, no torch, no I/O, no model loading.

A *bundle entry* is the researcher-authored description of one steerable
concept ON ONE MODEL/SAE PAIRING: which features constitute it in each
direction, what dose to apply at Low/Medium/High, where that calibration came
from, and which evidence attests it.

ONE ENTRY PER (concept_id, pairing_id).
There is deliberately no cross-model container. A concept on two pairings is two
entries, because nothing about the two is shared: not the features, not the
doses, not the units those doses are expressed in. The previous cross-model
mapping invited exactly the inference this contract refuses -- that a number
authored against one pairing carries over to another.

DIRECTIONS OWN THEIR TARGETS.
Amplifying a concept and suppressing it need not touch the same features, so
membership lives under each direction rather than above both. A direction may
be null, which is a positive statement that the concept was not calibrated that
way on this pairing -- not an omission to be filled in with the other
direction's data.

CROSS-LAYER IS A FIRST-CLASS SCHEMA CAPABILITY.
Every target carries its own `sae_id` and `layer`, so an entry may span several
layers and several SAEs. Runtime v1 cannot execute all such entries (see
`runtime.py`), but the schema represents them without strain, so lifting that
limitation later requires no migration. No layer number is privileged anywhere
in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .errors import DirectionNotCalibratedError, SchemaValidationError

# Substrings that mark a value as obviously not real research data. Matched
# case-insensitively against ids by the release gate, so a fixture stays
# self-quarantining even if every flag around it is set wrongly.
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "synthetic", "fake", "placeholder", "dummy", "lorem", "tbd", "todo",
    "do_not_ship", "do-not-ship", "example", "sample", "xxx", "test_only",
    "test-only",
)


class Direction(StrEnum):
    """Which way the concept is driven. Exactly two, always both present as
    keys on an entry, either of which may be null."""

    AMPLIFY = "amplify"
    SUPPRESS = "suppress"


class Strength(StrEnum):
    """The three public-facing control positions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Operation(StrEnum):
    """The intervention the executor performs. A closed set: an operation the
    executor does not implement cannot be authored in the first place."""

    CLAMP = "clamp"
    ABLATE = "ablate"


class Unit(StrEnum):
    """What a CLAMP value is measured in.

    The two multiple units are NOT interchangeable and neither is comparable
    across pairings. A sample maximum is the largest activation observed over
    whatever probe set was used; a corpus maximum is over the reference corpus.
    They differ by an unknown factor that depends on the probe set, which is why
    `unit_source` is mandatory for both: the number is uninterpretable without
    naming the maximum it was divided by.

    ABSOLUTE_ACTIVATION is a raw activation value with no denominator at all.
    """

    SAMPLE_MAX_MULTIPLE = "sample_max_multiple"
    CORPUS_MAX_MULTIPLE = "corpus_max_multiple"
    ABSOLUTE_ACTIVATION = "absolute_activation"


#: The units whose value is a MULTIPLE of a measured maximum, and therefore
#: require both a `unit_source` naming that maximum and, at resolution time, a
#: denominator supplied by the caller.
MULTIPLE_UNITS: frozenset[Unit] = frozenset({
    Unit.SAMPLE_MAX_MULTIPLE,
    Unit.CORPUS_MAX_MULTIPLE,
})


class PositionMode(StrEnum):
    """Which token positions an intervention applies to."""

    ALL = "all"
    GENERATED_ONLY = "generated_only"


class Provenance(StrEnum):
    """How well established the entry's origin is.

    Only ATTESTED may be published. UNKNOWN is the default so that an entry
    which nobody classified cannot pass by omission, and FAKE exists so that
    fixture data can say what it is in the data rather than in a comment.
    """

    ATTESTED = "attested"
    CANDIDATE = "candidate"
    DRAFT = "draft"
    FAKE = "fake"
    UNKNOWN = "unknown"


_CONCEPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_PAIRING_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]*$")
#: Fixed by the contract. A 12-hex prefix is the registry's on-disk file name;
#: the full 64-hex form is the content hash itself.
ARTIFACT_HASH_RE = re.compile(r"^(sha256:)?[0-9a-f]{12,64}$")

#: An `artifact_type` names a registry DIRECTORY. Constrained here, at authoring
#: time, rather than only where a path is built from it.
#:
#: Lowercase is part of the rule, not tidiness. Windows and macOS filesystems are
#: case-insensitive, so `Census_Report` and `census_report` are the same
#: directory there and two different directories on Linux: an entry authored on
#: one platform would resolve against a different artifact on another, or fail to
#: resolve at all. Refusing the collision at construction is the only place it
#: can be refused once for every consumer.
#:
#: Not an enum. The registry's artifact types are the canonical repository's to
#: extend, and a closed set here would make this contract the thing that has to
#: be edited before a new evidence type can be cited.
ARTIFACT_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

#: The ONLY form an evidence reference may take in a PUBLIC RELEASE: the
#: algorithm prefix and all 64 lowercase hex characters.
#:
#: `ARTIFACT_HASH_RE` above stays wider on purpose -- a 12-hex prefix is what the
#: registry names files by, and development and inspection must keep working
#: against it. Publication is the one place where a prefix is not a content
#: address: twelve hex characters is 48 bits, and a second artifact colliding
#: with it is findable. The algorithm prefix is mandatory because a bare digest
#: does not say what produced it, and "the algorithm was obvious at the time" is
#: not a property a published artefact can rely on.
PUBLICATION_ARTIFACT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaValidationError(message)


def _require_finite(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} must be a finite number, got {value!r}",
    )
    return float(value)


def _require_non_negative_int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             f"{name} must be a non-negative int, got {value!r}")
    return int(value)


def _require_non_empty_str(value: Any, name: str) -> None:
    _require(isinstance(value, str) and value.strip() != "",
             f"{name} must be a non-empty string, got {value!r}")


def canonical_target_sort_key(target: Target) -> tuple[int, str, int]:
    """The one canonical target ordering, shared by serialization and the
    resolver.

    Authoring order must never be observable: two entries that differ only in
    the order someone typed their targets are the same entry, and must produce
    the same `canonical_json`, the same fingerprint, and the same resolved
    state. Defining this in one place is what keeps those three consistent.
    """
    return (target.layer, target.sae_id, target.feature_idx)


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A pointer into the repository registry.

    Two fields and no free text: an evidence reference either resolves to a
    registry artifact or it does not, and prose in the middle would create a
    third state where it looks supported and is not.
    """

    artifact_type: str
    artifact_hash: str

    def __post_init__(self) -> None:
        _require_non_empty_str(self.artifact_type, "EvidenceRef.artifact_type")
        _require(bool(ARTIFACT_TYPE_RE.match(self.artifact_type)),
                 f"EvidenceRef.artifact_type must match "
                 f"{ARTIFACT_TYPE_RE.pattern} -- lowercase letters, digits and "
                 f"underscores only, because it names a registry directory. "
                 f"Uppercase is refused so that a type cannot mean one directory "
                 f"on a case-insensitive filesystem and another on Linux; "
                 f"separators and dots are refused so that it cannot be a path. "
                 f"Got {self.artifact_type!r}")
        _require(isinstance(self.artifact_hash, str)
                 and bool(ARTIFACT_HASH_RE.match(self.artifact_hash)),
                 f"EvidenceRef.artifact_hash must match "
                 f"{ARTIFACT_HASH_RE.pattern} (a 12-to-64 character lowercase hex "
                 f"digest, optionally 'sha256:'-prefixed), got "
                 f"{self.artifact_hash!r}")

    @property
    def bare_hash(self) -> str:
        """The hex digest without the optional `sha256:` prefix."""
        return self.artifact_hash.removeprefix("sha256:")

    @property
    def hash12(self) -> str:
        """The 12-character prefix the registry names files by."""
        return self.bare_hash[:12]

    @property
    def is_publication_digest(self) -> bool:
        """True iff this reference is written in the form a public release
        requires: `sha256:` followed by all 64 hex characters.

        A reference that is not is still resolvable, still verifiable by content,
        and still perfectly usable in development. It is only unpublishable.
        """
        return bool(PUBLICATION_ARTIFACT_HASH_RE.match(self.artifact_hash))

    def as_dict(self) -> dict[str, Any]:
        return {"artifact_type": self.artifact_type,
                "artifact_hash": self.artifact_hash}


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Who calibrated this entry, when, and against what evidence.

    At least one `EvidenceRef` is structurally required, so an entry cannot
    claim provenance while pointing at nothing. Whether those references
    actually resolve is a separate question, asked by the release gate against
    the repository registry -- the schema can only require that they were
    written down.
    """

    calibrated_by: str
    calibrated_at: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        _require_non_empty_str(self.calibrated_by,
                               "CalibrationProvenance.calibrated_by")
        _require(isinstance(self.calibrated_at, str),
                 "CalibrationProvenance.calibrated_at must be a string")
        try:
            parsed = datetime.fromisoformat(self.calibrated_at)
        except ValueError:
            raise SchemaValidationError(
                f"CalibrationProvenance.calibrated_at must be an ISO-8601 "
                f"timestamp, got {self.calibrated_at!r}") from None
        _require(parsed.tzinfo is not None,
                 f"CalibrationProvenance.calibrated_at must be offset-qualified "
                 f"or end in 'Z': a local wall-clock time is not a point in time "
                 f"and cannot be ordered against another lab's record. Got "
                 f"{self.calibrated_at!r}")
        _require(isinstance(self.evidence, tuple) and len(self.evidence) > 0,
                 "CalibrationProvenance.evidence must contain at least one "
                 "EvidenceRef; provenance that cites nothing is not provenance")
        for ref in self.evidence:
            _require(isinstance(ref, EvidenceRef),
                     f"CalibrationProvenance.evidence entry {ref!r} is not an "
                     f"EvidenceRef")

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibrated_by": self.calibrated_by,
            "calibrated_at": self.calibrated_at,
            "evidence": [e.as_dict() for e in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class Target:
    """One SAE feature participating in one direction of a concept.

    Four fields, all executable. There is deliberately nowhere here to put a
    label, a rationale, an author or a confidence note: this object is read by
    the executor, and free text next to a feature index invites a consumer to
    parse prose a human wrote.

    `sae_id` and `layer` are per-target because a direction may name features
    from several layers. `weight` is a positive multiplier on the authored dose
    and is NEVER normalized -- rescaling weights so they sum to one would change
    every dose in the entry to preserve a property nobody authored.
    """

    sae_id: str
    layer: int
    feature_idx: int
    weight: float

    def __post_init__(self) -> None:
        _require_non_empty_str(self.sae_id, "Target.sae_id")
        _require_non_negative_int(self.layer, "Target.layer")
        _require_non_negative_int(self.feature_idx, "Target.feature_idx")
        weight = _require_finite(self.weight, "Target.weight")
        _require(weight > 0.0,
                 f"Target.weight must be > 0, got {weight!r}. A zero weight would "
                 f"silently contribute nothing; a negative weight would invert the "
                 f"direction of one target against the direction the entry names.")

    @property
    def identity(self) -> tuple[str, int, int]:
        """What makes two targets the same target."""
        return (self.sae_id, self.layer, self.feature_idx)

    @property
    def group_key(self) -> tuple[str, int]:
        """The (sae_id, layer) group this target executes in."""
        return (self.sae_id, self.layer)

    def as_dict(self) -> dict[str, Any]:
        return {"sae_id": self.sae_id, "layer": self.layer,
                "feature_idx": self.feature_idx, "weight": float(self.weight)}


@dataclass(frozen=True, slots=True)
class Spec:
    """The intervention for one strength of one direction.

    FOUR PERMITTED SHAPES, exhaustively:

      CLAMP + SAMPLE_MAX_MULTIPLE  value > 0, unit, unit_source required
      CLAMP + CORPUS_MAX_MULTIPLE  value > 0, unit, unit_source required
      CLAMP + ABSOLUTE_ACTIVATION  value > 0, unit; unit_source PROHIBITED
      ABLATE                       value, unit, unit_source ALL PROHIBITED

    Ablation has no dose. Carrying a value on an ABLATE spec would let a reader
    conclude the feature was zeroed "by 2.0 of something", and letting the field
    sit unused would leave a number in the audit record that nothing consumes.

    `unit_source` names the specific measured maximum a multiple is taken
    against. It is required rather than optional because "0.5x max" is not a
    dose until someone says which max, measured over what.
    """

    operation: Operation
    value: float | None = None
    unit: Unit | None = None
    unit_source: str | None = None

    def __post_init__(self) -> None:
        _require(isinstance(self.operation, Operation),
                 f"Spec.operation must be an Operation, got {self.operation!r}")

        if self.operation is Operation.ABLATE:
            for name in ("value", "unit", "unit_source"):
                _require(getattr(self, name) is None,
                         f"Spec.{name} is prohibited when operation is 'ablate': "
                         f"ablation has no dose, so a {name} here would describe a "
                         f"quantity nothing applies. Got {getattr(self, name)!r}")
            return

        # CLAMP
        _require(self.value is not None,
                 "Spec.value is required when operation is 'clamp'")
        value = _require_finite(self.value, "Spec.value")
        _require(value > 0.0,
                 f"Spec.value must be > 0, got {value!r}. It is the SIZE of the "
                 f"intervention for this direction, not a signed offset: "
                 f"suppression is authored with its own positive value rather than "
                 f"a negated amplification.")
        _require(isinstance(self.unit, Unit),
                 f"Spec.unit is required when operation is 'clamp' and must be a "
                 f"Unit, got {self.unit!r}")

        if self.unit is Unit.ABSOLUTE_ACTIVATION:
            _require(self.unit_source is None,
                     f"Spec.unit_source is prohibited for unit "
                     f"'absolute_activation': the value is already an activation, "
                     f"so there is no denominator for a source to name. Got "
                     f"{self.unit_source!r}")
        else:
            _require_non_empty_str(
                self.unit_source,
                f"Spec.unit_source (required for unit {self.unit.value!r}, which is "
                f"a multiple of a measured maximum)")

    @property
    def is_ablation(self) -> bool:
        return self.operation is Operation.ABLATE

    @property
    def needs_denominator(self) -> bool:
        """True iff resolving this spec requires an externally supplied maximum."""
        return self.unit in MULTIPLE_UNITS

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "value": None if self.value is None else float(self.value),
            "unit": None if self.unit is None else self.unit.value,
            "unit_source": self.unit_source,
        }


@dataclass(frozen=True, slots=True)
class DirectionRecord:
    """One direction of one entry: its features and its three doses.

    The ABLATE weight rule lives here because it is a cross-field constraint.
    Ablation zeroes a feature; there is no partial ablation, so a weight other
    than 1.0 would be a multiplier on nothing -- readable as "ablate this one
    less", which the operation cannot express. Requiring exactly 1.0 across the
    whole direction, rather than only on the ablating strengths, keeps one
    membership list meaning one thing at every position of the control.
    """

    targets: tuple[Target, ...]
    specs: Mapping[Strength, Spec]

    def __post_init__(self) -> None:
        _require(isinstance(self.targets, tuple) and len(self.targets) > 0,
                 "DirectionRecord.targets must be a non-empty tuple")
        for t in self.targets:
            _require(isinstance(t, Target),
                     f"DirectionRecord.targets entry {t!r} is not a Target")
        identities = [t.identity for t in self.targets]
        dupes = sorted({i for i in identities if identities.count(i) > 1})
        _require(not dupes,
                 f"DirectionRecord.targets contains duplicate targets {dupes}; a "
                 f"repeated feature would be driven twice")

        _require(isinstance(self.specs, Mapping),
                 "DirectionRecord.specs must be a mapping")
        for strength in Strength:
            _require(strength in self.specs,
                     f"DirectionRecord.specs is missing {strength.value!r}; all "
                     f"three control positions must be authored")
        unknown = sorted(str(k) for k in self.specs if k not in set(Strength))
        _require(not unknown,
                 f"DirectionRecord.specs has unknown strength key(s) {unknown}")
        for strength, spec in self.specs.items():
            _require(isinstance(spec, Spec),
                     f"DirectionRecord.specs[{strength!r}] is not a Spec")

        if self.uses_ablation:
            offenders = sorted({t.weight for t in self.targets if t.weight != 1.0})
            _require(not offenders,
                     f"this direction ablates at "
                     f"{[s.value for s in self.ablating_strengths]}, so every target "
                     f"weight must be exactly 1.0; found {offenders}. A weight other "
                     f"than 1.0 would scale an operation that has no dose.")

        object.__setattr__(self, "specs",
                           MappingProxyType({s: self.specs[s] for s in Strength}))

    @property
    def uses_ablation(self) -> bool:
        return any(spec.is_ablation for spec in self.specs.values())

    @property
    def ablating_strengths(self) -> tuple[Strength, ...]:
        return tuple(s for s in Strength if self.specs[s].is_ablation)

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({t.layer for t in self.targets}))

    @property
    def sae_ids(self) -> tuple[str, ...]:
        return tuple(sorted({t.sae_id for t in self.targets}))

    @property
    def group_keys(self) -> tuple[tuple[str, int], ...]:
        """Distinct (sae_id, layer) execution groups, in canonical order."""
        return tuple(sorted({t.group_key for t in self.targets},
                            key=lambda g: (g[1], g[0])))

    def targets_by_group(self) -> dict[tuple[str, int], tuple[Target, ...]]:
        """Targets keyed by (sae_id, layer), each group in canonical order.

        Same-layer targets sharing one SAE land in one group and execute as one
        batch; nothing here splits them.
        """
        out: dict[tuple[str, int], list[Target]] = {}
        for t in sorted(self.targets, key=canonical_target_sort_key):
            out.setdefault(t.group_key, []).append(t)
        return {g: tuple(out[g]) for g in self.group_keys}

    def sae_ids_at_layer(self, layer: int) -> tuple[str, ...]:
        return tuple(sorted({t.sae_id for t in self.targets if t.layer == layer}))

    def spec(self, strength: Strength) -> Spec:
        return self.specs[strength]

    def as_dict(self) -> dict[str, Any]:
        return {
            # canonical order, NOT authoring order: see canonical_target_sort_key
            "targets": [t.as_dict() for t in
                        sorted(self.targets, key=canonical_target_sort_key)],
            "specs": {s.value: self.specs[s].as_dict() for s in Strength},
        }


@dataclass(frozen=True, slots=True)
class BundleEntry:
    """One concept on one model/SAE pairing.

    `pairing_id` is the identity of the model-and-SAE combination this entry was
    authored against. Everything numeric in the entry is meaningful only against
    that pairing, and nothing in this contract relates a value here to a value
    in an entry with a different `pairing_id`.
    """

    concept_id: str
    pairing_id: str
    positions: PositionMode
    provenance: Provenance
    directions: Mapping[Direction, DirectionRecord | None]
    calibration_provenance: CalibrationProvenance | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        _require_non_empty_str(self.concept_id, "BundleEntry.concept_id")
        _require(bool(_CONCEPT_ID_RE.match(self.concept_id)),
                 f"BundleEntry.concept_id {self.concept_id!r} must match "
                 f"{_CONCEPT_ID_RE.pattern} -- it is a stable key that appears in "
                 f"URLs, logs and saved states")
        _require_non_empty_str(self.pairing_id, "BundleEntry.pairing_id")
        _require(bool(_PAIRING_ID_RE.match(self.pairing_id)),
                 f"BundleEntry.pairing_id {self.pairing_id!r} must match "
                 f"{_PAIRING_ID_RE.pattern}")
        _require(isinstance(self.positions, PositionMode),
                 f"BundleEntry.positions must be a PositionMode, got "
                 f"{self.positions!r}")
        _require(isinstance(self.provenance, Provenance),
                 f"BundleEntry.provenance must be a Provenance, got "
                 f"{self.provenance!r}")

        _require(isinstance(self.directions, Mapping),
                 "BundleEntry.directions must be a mapping")
        for direction in Direction:
            _require(direction in self.directions,
                     f"BundleEntry.directions must contain the key "
                     f"{direction.value!r}, explicitly null if the concept is not "
                     f"calibrated that way on this pairing")
        unknown = sorted(str(k) for k in self.directions if k not in set(Direction))
        _require(not unknown,
                 f"BundleEntry.directions has unknown key(s) {unknown}; exactly "
                 f"'amplify' and 'suppress' are permitted")
        for direction in Direction:
            record = self.directions[direction]
            _require(record is None or isinstance(record, DirectionRecord),
                     f"BundleEntry.directions[{direction.value!r}] must be a "
                     f"DirectionRecord or null, got {record!r}")

        _require(self.calibration_provenance is None
                 or isinstance(self.calibration_provenance, CalibrationProvenance),
                 "BundleEntry.calibration_provenance must be a "
                 "CalibrationProvenance or null")
        # Permitted on any entry; REQUIRED on an attested one. An entry cannot
        # claim to be attested and then decline to say by whom, when, or against
        # what.
        if self.provenance is Provenance.ATTESTED:
            _require(self.calibration_provenance is not None,
                     f"BundleEntry {self.concept_id!r} on pairing "
                     f"{self.pairing_id!r} declares provenance 'attested' but "
                     f"carries no calibration_provenance. Attestation without a "
                     f"calibrator, a timestamp and evidence is an assertion, not "
                     f"an attestation.")
        _require_non_empty_str(self.schema_version, "BundleEntry.schema_version")

        object.__setattr__(self, "directions",
                           MappingProxyType({d: self.directions[d] for d in Direction}))

    # -- accessors -------------------------------------------------------
    @property
    def key(self) -> tuple[str, str]:
        return (self.concept_id, self.pairing_id)

    @property
    def calibrated_directions(self) -> tuple[Direction, ...]:
        return tuple(d for d in Direction if self.directions[d] is not None)

    def has_direction(self, direction: Direction | str) -> bool:
        return self.directions[Direction(direction)] is not None

    def direction(self, direction: Direction | str) -> DirectionRecord:
        """The record for `direction`, or the contract's refusal if it is null."""
        d = Direction(direction)
        record = self.directions[d]
        if record is None:
            raise DirectionNotCalibratedError(
                concept_id=self.concept_id, pairing_id=self.pairing_id,
                direction=d.value)
        return record

    @property
    def sae_ids(self) -> tuple[str, ...]:
        """Every SAE named anywhere in the entry, across both directions."""
        return tuple(sorted({t.sae_id
                             for d in self.calibrated_directions
                             for t in self.directions[d].targets}))

    @property
    def evidence(self) -> tuple[EvidenceRef, ...]:
        if self.calibration_provenance is None:
            return ()
        return self.calibration_provenance.evidence

    # -- serialization ---------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        cp = self.calibration_provenance
        return {
            "schema_version": self.schema_version,
            "concept_id": self.concept_id,
            "pairing_id": self.pairing_id,
            "positions": self.positions.value,
            "provenance": self.provenance.value,
            "calibration_provenance": None if cp is None else cp.as_dict(),
            "directions": {
                d.value: (None if self.directions[d] is None
                          else self.directions[d].as_dict())
                for d in Direction
            },
        }

    def canonical_json(self) -> str:
        """Stable serialization: sorted keys, canonical target order, no
        whitespace variance. Two semantically identical entries produce
        identical bytes, which is what makes the fingerprint below meaningful."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)

    def audit_fingerprint(self) -> str:
        """FULL AUDIT IDENTITY of the entry, deliberately INCLUDING provenance.

        Answers "is this the same authored artefact?". Two entries with
        identical doses but a different calibrator or a different evidence
        reference have DIFFERENT audit fingerprints.

        It is therefore NOT an execution identity and must not be used as one --
        correcting a timestamp would change it while changing nothing the model
        does. For that, see `ResolvedControlState.execution_fingerprint()`.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
