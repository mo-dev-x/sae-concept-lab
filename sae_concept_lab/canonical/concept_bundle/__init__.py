"""Concept-bundle contract for the steer/ablate tool.

Pure Python. No torch, no model loading, no hooks, no generation. The only
module that touches the filesystem is `evidence.py`, and only when a caller asks
it to resolve a reference. This package is the contract that the execution layer
and the UI both read; it implements neither.

Six pieces, deliberately separate:

  schema.py    what a bundle entry IS. One concept on one pairing, directions
               owning their own targets and doses. Accepts multi-group entries.
  runtime.py   what runtime v1 can EXECUTE. Groups by (sae_id, layer), and
               refuses in two different ways -- PROHIBITED for two SAEs at one
               layer, CAPABILITY_LIMIT for more than one group.
  resolver.py  (entry, direction, strength) -> one immutable
               ResolvedControlState shared by Public and Advanced modes.
  evidence.py  resolves evidence references against the repository registry.
  release.py   fail-closed publication gate: ATTESTED provenance and evidence
               that actually resolves, or it does not ship.
  codec.py     strict versioned JSON for explicitly named files. No scanning.

The schema/runtime split is the load-bearing design decision: the one-group
ceiling is a property of today's executor, not a scientific claim about any
layer, so it lives in a capability check that can be deleted without touching
stored data.
"""

from __future__ import annotations

from .codec import (
    BUNDLES_DIR,
    SUPPORTED_SCHEMA_VERSIONS,
    decode_entry,
    encode_entry,
    load_entry_file,
    load_entry_files,
)
from .errors import (
    BundleDecodeError,
    Classification,
    ConceptBundleError,
    DenominatorError,
    DirectionNotCalibratedError,
    InvalidDenominatorError,
    MissingDenominatorError,
    MultipleExecutionGroupsError,
    MultipleSaeIdentitiesAtLayerError,
    ReleaseBlockedError,
    ReleaseBuildError,
    SchemaValidationError,
    SchemaVersionError,
    UnknownConceptError,
    UnknownPairingError,
)
from .evidence import (
    NO_EVIDENCE_REGISTRY,
    REGISTRY_ROOT,
    EvidenceRegistry,
    EvidenceResolution,
    EvidenceStatus,
    InMemoryEvidenceRegistry,
    NullEvidenceRegistry,
    RepositoryEvidenceRegistry,
    resolve_all,
)
from .release import (
    MIN_PUBLISHED_DIRECTIONS,
    PUBLISHABLE_PROVENANCE,
    Exposure,
    LayoutEntry,
    ReleaseDecision,
    assert_publishable,
    assert_release_build_clean,
    evaluate_publishability,
    filter_publishable,
    select_layout_entries,
)
from .resolver import (
    UNIT_CAVEAT,
    ConceptRegistry,
    DenominatorSource,
    MappingDenominatorSource,
    ResolvedControlState,
    ResolvedTarget,
    resolve_control,
)
from .runtime import (
    MAX_EXECUTION_GROUPS_PER_PASS,
    RUNTIME_VERSION,
    CapabilityReport,
    ExecutionGroup,
    check_direction_executable,
    executable_directions,
    execution_groups,
    require_single_execution_group,
    validate_execution_request,
)
from .schema import (
    ARTIFACT_HASH_RE,
    MULTIPLE_UNITS,
    PLACEHOLDER_MARKERS,
    BundleEntry,
    CalibrationProvenance,
    Direction,
    DirectionRecord,
    EvidenceRef,
    Operation,
    PositionMode,
    Provenance,
    Spec,
    Strength,
    Target,
    Unit,
    canonical_target_sort_key,
)

__all__ = [
    "ARTIFACT_HASH_RE",
    "BUNDLES_DIR",
    "MAX_EXECUTION_GROUPS_PER_PASS",
    "MIN_PUBLISHED_DIRECTIONS",
    "MULTIPLE_UNITS",
    "NO_EVIDENCE_REGISTRY",
    "PLACEHOLDER_MARKERS",
    "PUBLISHABLE_PROVENANCE",
    "REGISTRY_ROOT",
    "RUNTIME_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "UNIT_CAVEAT",
    "BundleDecodeError",
    "BundleEntry",
    "CalibrationProvenance",
    "CapabilityReport",
    "Classification",
    "ConceptBundleError",
    "ConceptRegistry",
    "DenominatorError",
    "DenominatorSource",
    "Direction",
    "DirectionNotCalibratedError",
    "DirectionRecord",
    "EvidenceRef",
    "EvidenceRegistry",
    "EvidenceResolution",
    "EvidenceStatus",
    "ExecutionGroup",
    "Exposure",
    "InMemoryEvidenceRegistry",
    "InvalidDenominatorError",
    "LayoutEntry",
    "MappingDenominatorSource",
    "MissingDenominatorError",
    "MultipleExecutionGroupsError",
    "MultipleSaeIdentitiesAtLayerError",
    "NullEvidenceRegistry",
    "Operation",
    "PositionMode",
    "Provenance",
    "ReleaseBlockedError",
    "ReleaseBuildError",
    "ReleaseDecision",
    "RepositoryEvidenceRegistry",
    "ResolvedControlState",
    "ResolvedTarget",
    "SchemaValidationError",
    "SchemaVersionError",
    "Spec",
    "Strength",
    "Target",
    "Unit",
    "UnknownConceptError",
    "UnknownPairingError",
    "assert_publishable",
    "assert_release_build_clean",
    "canonical_target_sort_key",
    "check_direction_executable",
    "decode_entry",
    "encode_entry",
    "evaluate_publishability",
    "executable_directions",
    "execution_groups",
    "filter_publishable",
    "load_entry_file",
    "load_entry_files",
    "require_single_execution_group",
    "resolve_all",
    "resolve_control",
    "select_layout_entries",
    "validate_execution_request",
]
