"""Defensive same-layer/multi-SAE/cross-layer enforcement for a real
backend, independent of whatever pre-flight check a caller already ran.

`ui/tab.py` already calls `check_direction_executable()` before ever
resolving a control or invoking a backend, so in the UI's own call path
this module's check can never actually trip. It exists for callers that
reach a backend WITHOUT going through that pre-flight -- a test, or any
future non-UI caller -- so an unsupported request still fails as
PROHIBITED/CAPABILITY_LIMIT here, rather than being silently executed (or
silently degraded) by a backend that trusted its caller.

This module never reimplements runtime.py's grouping rules with different
wording: `require_group_from_resolved` raises the exact same canonical
error classes (`MultipleSaeIdentitiesAtLayerError`,
`MultipleExecutionGroupsError`) that `runtime.require_single_execution_group`
raises, just computed from a `ResolvedControlState.targets` tuple directly
-- the resolved state carries no reference back to the `BundleEntry` that
`require_single_execution_group` needs, so the grouping arithmetic is
repeated here (over the same three-line (sae_id, layer) grouping rule
runtime.py's own `execution_groups`/`_layers_with_multiple_saes` apply),
never the CONCLUSION of what to do about a violation, which is always
delegated to the canonical error classes themselves.
"""

from __future__ import annotations

from sae_concept_lab.canonical.concept_bundle import (
    MultipleExecutionGroupsError,
    MultipleSaeIdentitiesAtLayerError,
    ResolvedControlState,
    Target,
)


class UnsupportedTargetCountError(ValueError):
    """A single (sae_id, layer) execution group carries more than one
    target. Distinct from PROHIBITED/CAPABILITY_LIMIT: the group itself is
    perfectly legal runtime-v1 execution, but the extracted hook mechanism
    (_make_clamp_hook) clamps exactly one feature per call -- multi-target
    execution within one group is not implemented, and is refused here
    rather than silently averaged, summed, or truncated to the first
    target."""


def require_group_from_resolved(resolved: ResolvedControlState) -> tuple[str, int, Target]:
    """Groups resolved.targets by (sae_id, layer); raises
    MultipleSaeIdentitiesAtLayerError if more than one SAE shares a layer,
    MultipleExecutionGroupsError if more than one (sae_id, layer) group
    exists, and UnsupportedTargetCountError if the single group carries
    more than one target. Returns (sae_id, layer, the_one_target)."""
    by_layer: dict[int, set[str]] = {}
    by_group: dict[tuple[str, int], list[Target]] = {}
    for target in resolved.targets:
        by_layer.setdefault(target.layer, set()).add(target.sae_id)
        by_group.setdefault((target.sae_id, target.layer), []).append(target)

    offending = sorted(
        (layer, tuple(sorted(sae_ids))) for layer, sae_ids in by_layer.items() if len(sae_ids) > 1
    )
    if offending:
        layer, sae_ids = offending[0]
        raise MultipleSaeIdentitiesAtLayerError(
            concept_id=resolved.concept_id,
            pairing_id=resolved.pairing_id,
            direction=resolved.direction.value,
            layer=layer,
            sae_ids=sae_ids,
        )

    if len(by_group) > 1:
        raise MultipleExecutionGroupsError(
            concept_id=resolved.concept_id,
            pairing_id=resolved.pairing_id,
            direction=resolved.direction.value,
            groups=tuple(sorted(by_group.keys())),
        )

    (sae_id, layer), targets = next(iter(by_group.items()))
    if len(targets) != 1:
        raise UnsupportedTargetCountError(
            f"execution group {sae_id}@L{layer} carries {len(targets)} targets "
            f"{[t.feature_idx for t in targets]}; this backend's extracted hook mechanism "
            f"(_make_clamp_hook) clamps exactly one feature per call -- refusing rather than "
            f"silently averaging, summing, or truncating to the first target."
        )
    return sae_id, layer, targets[0]
