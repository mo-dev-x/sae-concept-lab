"""Typed errors for the concept-bundle contract.

Every failure mode a caller might reasonably want to distinguish gets its own
type, because the UI has to say different things for "this is a limitation of
the current runtime" and "this data must never reach a user".

TWO RUNTIME REFUSALS, DELIBERATELY NOT ONE.
`MultipleSaeIdentitiesAtLayerError` and `MultipleExecutionGroupsError` are kept
apart because they mean opposite things about the future. Composing
interventions from two different SAEs at the SAME layer is PROHIBITED: each SAE
reconstructs the residual stream with its own error, and adding two such
reconstructions is not a defined operation -- no amount of engineering makes it
well-posed. Executing several (sae_id, layer) groups in one pass is a
CAPABILITY_LIMIT: runtime v1 attaches one group, and that ceiling lifts when the
executor grows. Collapsing both into one "too many SAE identities" error would
tell an author to wait for a fix that is coming in one case and never in the
other.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class Classification(StrEnum):
    """Why a refusal happened, in terms of whether it can ever be lifted."""

    #: Undefined operation. Not a roadmap item; no executor will support it.
    PROHIBITED = "PROHIBITED"
    #: Implemented ceiling of runtime v1. Liftable without a data migration.
    CAPABILITY_LIMIT = "CAPABILITY_LIMIT"


class ConceptBundleError(Exception):
    """Base class for every error raised by this package."""


class SchemaValidationError(ConceptBundleError):
    """The entry is structurally invalid: a missing strength, an empty target
    list, a non-finite or negative number, a spec whose operation and unit
    fields do not form one of the permitted shapes."""


class UnknownPairingError(ConceptBundleError):
    """No entry is registered for the requested (concept, pairing)."""


class UnknownConceptError(ConceptBundleError):
    """No entry with the requested concept id is registered."""


class SchemaVersionError(ConceptBundleError):
    """The encoded entry declares a schema version this build cannot decode.

    Fail-closed: an unknown version is refused rather than decoded on the guess
    that it is probably compatible.
    """


class BundleDecodeError(ConceptBundleError):
    """The encoded entry is malformed, or carries unknown/missing required
    fields. Refused rather than partially decoded."""


class ReleaseBuildError(ConceptBundleError):
    """A release build was asked to expose entries that are not publishable --
    typically development stubs left switched on."""


class DenominatorError(ConceptBundleError):
    """A CLAMP value expressed as a multiple could not be turned into an
    absolute activation."""


class MissingDenominatorError(DenominatorError):
    """A CLAMP spec is expressed as a multiple of an activation maximum, and no
    denominator was supplied for the target.

    The contract holds no activation statistics, by design: a sample maximum
    over a probe set is not a corpus maximum, and inventing either here would
    make an authored dose mean something nobody measured. The caller supplies
    the lookup, and a missing one fails closed rather than defaulting to 1.0 --
    which would silently reinterpret "0.5x the corpus max" as "0.5 activation
    units" and drive the model at a dose nobody chose.
    """


class InvalidDenominatorError(DenominatorError):
    """A denominator was supplied but is not a positive finite number.

    Zero or negative would make the resolved dose zero or sign-flipped, and a
    NaN would propagate silently into the clamp value. Refused at the boundary
    rather than multiplied through.
    """


@dataclass(eq=False)
class DirectionNotCalibratedError(ConceptBundleError):
    """The requested direction exists in the schema but is null on this entry.

    A null direction is a positive statement -- "this concept was not calibrated
    this way here" -- and is distinct from a missing entry. The message is fixed
    by the contract; the identifying detail is carried as attributes.
    """

    concept_id: str
    pairing_id: str
    direction: str

    MESSAGE: ClassVar[str] = (
        "this direction is not calibrated for this concept on this model"
    )

    def __post_init__(self) -> None:
        super().__init__(str(self))

    def __str__(self) -> str:
        return self.MESSAGE


@dataclass(eq=False)
class MultipleSaeIdentitiesAtLayerError(ConceptBundleError):
    """PROHIBITED. Two SAEs at one layer cannot be composed.

    Each SAE is a lossy reconstruction of the same residual stream. Two of them
    at the same site do not decompose one activation into disjoint parts; they
    give two overlapping approximations with two error terms, and there is no
    defined way to add interventions across them. This is not a runtime ceiling
    and will not be lifted by a better executor.
    """

    concept_id: str
    pairing_id: str
    direction: str
    layer: int
    sae_ids: tuple[str, ...]

    CLASSIFICATION: ClassVar[Classification] = Classification.PROHIBITED

    def __post_init__(self) -> None:
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"multiple SAE identities at layer {self.layer}: "
            f"reconstruction-error composition is undefined"
        )


@dataclass(eq=False)
class MultipleExecutionGroupsError(ConceptBundleError):
    """CAPABILITY_LIMIT. Runtime v1 executes one (sae_id, layer) group.

    Distinct from the prohibition above: these groups are individually
    well-defined, and running them together is a coherent thing to want. The
    executor simply attaches one group per pass today. The schema accepts
    cross-layer entries deliberately so that lifting this needs no migration,
    and no layer number is privileged.
    """

    concept_id: str
    pairing_id: str
    direction: str
    groups: tuple[tuple[str, int], ...]

    CLASSIFICATION: ClassVar[Classification] = Classification.CAPABILITY_LIMIT

    def __post_init__(self) -> None:
        super().__init__(str(self))

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def __str__(self) -> str:
        return (
            f"runtime v1 executes one (sae_id, layer) group; this bundle "
            f"resolves to {self.n_groups}"
        )


@dataclass(eq=False)
class ReleaseBlockedError(ConceptBundleError):
    """The entry may not be published or used for any public-facing operation.

    Fail-closed: raised whenever publishability cannot be positively
    established -- provenance short of ATTESTED, absent calibration provenance,
    or any evidence reference that does not resolve against the repository
    registry.
    """

    concept_id: str
    pairing_id: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__init__(str(self))

    def __str__(self) -> str:
        bullets = "; ".join(self.reasons)
        return (
            f"concept {self.concept_id!r} on pairing {self.pairing_id!r} is NOT "
            f"publishable: {bullets}. Publishability is fail-closed -- an entry "
            f"passes only by positively demonstrating every condition, never by "
            f"failing to trip a check."
        )
