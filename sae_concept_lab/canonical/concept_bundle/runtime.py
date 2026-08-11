"""Runtime v1 capability validation -- separate from schema validation.

The split is the whole point of this module. `schema.py` answers "is this a
well-formed bundle entry?" and accepts cross-layer, multi-SAE entries.  This
module answers the narrower question "can runtime v1 actually execute it?" and
does not.

EXECUTION IS GROUPED BY (sae_id, layer).
That pair, not the layer alone, is the unit of execution: one SAE attached at
one site. Targets sharing a group execute as ONE BATCH -- there is no per-target
pass and nothing here splits a group up.

TWO REFUSALS THAT MEAN DIFFERENT THINGS.
  * Several SAEs at the SAME layer is PROHIBITED. Two SAEs reconstruct the same
    residual stream with two different error terms; composing interventions
    across them is not a defined operation, so no future executor fixes it.
  * Several groups across DISTINCT layers is a CAPABILITY_LIMIT. Each group is
    well defined and running them together is coherent; runtime v1 just attaches
    one group per pass.
They are separate exception types so a caller can tell an author "rebuild this
entry" in the first case and "this is coming" in the second. There is
deliberately no single "too many SAE identities" error covering both.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import (
    Classification,
    DirectionNotCalibratedError,
    MultipleExecutionGroupsError,
    MultipleSaeIdentitiesAtLayerError,
)
from .schema import BundleEntry, Direction, DirectionRecord, Target

RUNTIME_VERSION = "v1"

# --- capability surface of runtime v1 --------------------------------------
# A claim about what the EXECUTOR implements, not about what is scientifically
# sensible. Widening it is a one-line change once multi-site execution lands;
# nothing else in the package needs to move.
MAX_EXECUTION_GROUPS_PER_PASS = 1


@dataclass(frozen=True, slots=True)
class ExecutionGroup:
    """One SAE attached at one layer, with every target it drives.

    The whole group is one batch. `targets` is in canonical order, so two
    authorings of the same group produce the same plan.
    """

    sae_id: str
    layer: int
    targets: tuple[Target, ...]

    @property
    def key(self) -> tuple[str, int]:
        return (self.sae_id, self.layer)

    @property
    def feature_indices(self) -> tuple[int, ...]:
        return tuple(t.feature_idx for t in self.targets)

    def as_dict(self) -> dict[str, object]:
        return {"sae_id": self.sae_id, "layer": self.layer,
                "targets": [t.as_dict() for t in self.targets]}


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Whether one direction of an entry is executable, and every reason it is
    not.

    Non-raising, so the UI can grey a control out before a user clicks it rather
    than catching an exception to find out. All reasons are collected, and each
    carries its classification, so an author learns in one query which problems
    are permanent and which are a current ceiling.
    """

    executable: bool
    runtime_version: str
    concept_id: str
    pairing_id: str
    direction: str
    calibrated: bool
    groups: tuple[tuple[str, int], ...] = ()
    reasons: tuple[str, ...] = ()
    classifications: tuple[Classification, ...] = ()

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def is_prohibited(self) -> bool:
        return Classification.PROHIBITED in self.classifications

    @property
    def is_capability_limited(self) -> bool:
        return Classification.CAPABILITY_LIMIT in self.classifications


def _layers_with_multiple_saes(record: DirectionRecord) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Every layer carrying more than one SAE, lowest layer first."""
    return tuple(
        (layer, record.sae_ids_at_layer(layer))
        for layer in record.layers
        if len(record.sae_ids_at_layer(layer)) > 1
    )


def execution_groups(record: DirectionRecord) -> tuple[ExecutionGroup, ...]:
    """Groups a direction's targets by (sae_id, layer), canonical order.

    Structural only: this reports the grouping without judging how many groups
    runtime v1 can run. `require_single_execution_group` applies the ceiling.
    """
    return tuple(
        ExecutionGroup(sae_id=sae_id, layer=layer, targets=targets)
        for (sae_id, layer), targets in record.targets_by_group().items()
    )


def check_direction_executable(entry: BundleEntry,
                               direction: Direction | str) -> CapabilityReport:
    """Non-raising capability query for one direction of one entry."""
    d = Direction(direction)
    record = entry.directions[d]

    if record is None:
        return CapabilityReport(
            executable=False, runtime_version=RUNTIME_VERSION,
            concept_id=entry.concept_id, pairing_id=entry.pairing_id,
            direction=d.value, calibrated=False,
            reasons=("this direction is not calibrated for this concept on this "
                     "model",),
            classifications=(),
        )

    reasons: list[str] = []
    classifications: list[Classification] = []

    offending = _layers_with_multiple_saes(record)
    for layer, sae_ids in offending:
        reasons.append(
            f"layer {layer} carries {len(sae_ids)} SAE identities "
            f"{list(sae_ids)}: composing interventions across two reconstructions "
            f"of the same residual stream is undefined, and this is PROHIBITED "
            f"rather than pending")
        classifications.append(Classification.PROHIBITED)

    groups = record.group_keys
    if not offending and len(groups) > MAX_EXECUTION_GROUPS_PER_PASS:
        reasons.append(
            f"targets resolve to {len(groups)} (sae_id, layer) groups "
            f"{[f'{s}@L{layer}' for s, layer in groups]}; runtime "
            f"{RUNTIME_VERSION} executes {MAX_EXECUTION_GROUPS_PER_PASS} group per "
            f"pass. This is a CURRENT IMPLEMENTATION LIMIT, not a scientific "
            f"constraint, and no layer number is privileged")
        classifications.append(Classification.CAPABILITY_LIMIT)

    return CapabilityReport(
        executable=not reasons, runtime_version=RUNTIME_VERSION,
        concept_id=entry.concept_id, pairing_id=entry.pairing_id,
        direction=d.value, calibrated=True, groups=groups,
        reasons=tuple(reasons), classifications=tuple(classifications),
    )


def require_single_execution_group(entry: BundleEntry,
                                   direction: Direction | str) -> ExecutionGroup:
    """Raising form. Returns the one group runtime v1 will execute.

    Checked in a fixed order -- prohibition before capability limit -- so the
    message a caller sees does not depend on dict iteration, and so an entry
    that is both malformed and oversized is reported as malformed.
    """
    d = Direction(direction)
    record = entry.directions[d]
    if record is None:
        raise DirectionNotCalibratedError(
            concept_id=entry.concept_id, pairing_id=entry.pairing_id,
            direction=d.value)

    offending = _layers_with_multiple_saes(record)
    if offending:
        layer, sae_ids = offending[0]
        raise MultipleSaeIdentitiesAtLayerError(
            concept_id=entry.concept_id, pairing_id=entry.pairing_id,
            direction=d.value, layer=layer, sae_ids=sae_ids)

    groups = execution_groups(record)
    if len(groups) > MAX_EXECUTION_GROUPS_PER_PASS:
        raise MultipleExecutionGroupsError(
            concept_id=entry.concept_id, pairing_id=entry.pairing_id,
            direction=d.value, groups=tuple(g.key for g in groups))
    return groups[0]


def validate_execution_request(entry: BundleEntry, *,
                               direction: Direction | str) -> ExecutionGroup:
    """The executor's pre-flight call. Alias kept deliberately narrow.

    Strength is not an argument: no strength of a calibrated direction is
    executable while another is not. Operations and positions are closed enums
    whose every member the executor implements, so there is nothing left for a
    per-strength capability check to reject -- and offering one would suggest
    otherwise.
    """
    return require_single_execution_group(entry, direction)


def executable_directions(entry: BundleEntry) -> tuple[Direction, ...]:
    """Which directions runtime v1 can run. One unexecutable direction does not
    disqualify the other."""
    return tuple(d for d in Direction
                 if check_direction_executable(entry, d).executable)
