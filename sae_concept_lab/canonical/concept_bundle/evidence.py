"""Evidence-registry resolution.

An `EvidenceRef` is a claim that a registry artifact exists and says something.
This module is the only place that checks whether the claim holds, and it is the
only module in the package that touches the filesystem.

WHY IT IS A SEPARATE MODULE AND AN INJECTED OBJECT.
The schema, the runtime and the resolver are pure: same input, same output, no
disk. Publishing is not pure -- it depends on what is actually in `registry/` at
build time -- so the impurity is confined here and handed to the release gate as
an argument. A caller must name the registry it is resolving against, which
means no code path can publish by accidentally resolving against nothing.

WHAT COUNTS AS RESOLVED.
The repository registry stores artifacts as `registry/<artifact_type>/<12-hex
prefix of the content hash>.json`, each carrying its own `artifact_type` and
`self_hash`. A reference resolves only if the file exists, parses, is an object,
declares the artifact type the reference claims, and carries a `self_hash` the
reference's digest is a prefix of. Anything else is a distinct, named failure --
UNRESOLVABLE, MALFORMED or MISMATCHED -- because "the file is missing" and "the
file is for a different artifact" call for different corrections.

This module deliberately does NOT import `interplab.registry`. It reads the same
on-disk layout with the standard library, so the concept-bundle contract stays
importable without pulling in the rest of the codebase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from .schema import EvidenceRef

#: Repository root, then the registry tree inside it. Resolved from this file's
#: location so the default works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = REPO_ROOT / "registry"


class EvidenceStatus(StrEnum):
    """Outcome of resolving one reference. Only RESOLVED permits publishing."""

    RESOLVED = "resolved"
    #: No artifact at the path the reference names.
    UNRESOLVABLE = "unresolvable"
    #: The artifact exists but cannot be read as a registry record.
    MALFORMED = "malformed"
    #: The artifact exists and is well formed, but is not the one referenced.
    MISMATCHED = "mismatched"


@dataclass(frozen=True, slots=True)
class EvidenceResolution:
    """What happened when one reference was looked up."""

    ref: EvidenceRef
    status: EvidenceStatus
    location: str = ""
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.status is EvidenceStatus.RESOLVED

    def as_reason(self) -> str:
        """One line for the release gate's blocking reasons."""
        return (f"evidence {self.ref.artifact_type}:{self.ref.artifact_hash} is "
                f"{self.status.value}: {self.detail}")


@runtime_checkable
class EvidenceRegistry(Protocol):
    """Anything that can answer whether a reference resolves."""

    def resolve(self, ref: EvidenceRef) -> EvidenceResolution: ...


@dataclass(frozen=True, slots=True)
class NullEvidenceRegistry:
    """Resolves nothing. The honest answer when no registry is available.

    Not a convenience default -- it exists so a caller with no registry gets a
    refusal that names the reason, rather than a gate that quietly skips the
    evidence check because it had nothing to check against.
    """

    def resolve(self, ref: EvidenceRef) -> EvidenceResolution:
        return EvidenceResolution(
            ref=ref, status=EvidenceStatus.UNRESOLVABLE,
            detail="no evidence registry was supplied; evidence cannot be "
                   "confirmed and publishing therefore fails closed")


#: The fail-closed default used wherever a registry argument is omitted. A
#: module singleton rather than a call in a default argument, so it is one
#: greppable name and every caller that forgot to pass a registry shares it.
NO_EVIDENCE_REGISTRY: EvidenceRegistry = NullEvidenceRegistry()


class InMemoryEvidenceRegistry:
    """A registry backed by a dict of (artifact_type, hash12) -> self_hash.

    For tests and for callers that have already loaded their index. Applies the
    same prefix rule as the repository registry, so a test cannot pass by being
    laxer than the real thing.
    """

    def __init__(self, records: dict[tuple[str, str], str] | None = None) -> None:
        self._records = dict(records or {})

    def add(self, artifact_type: str, self_hash: str) -> None:
        bare = self_hash.removeprefix("sha256:")
        self._records[(artifact_type, bare[:12])] = bare

    def resolve(self, ref: EvidenceRef) -> EvidenceResolution:
        key = (ref.artifact_type, ref.hash12)
        location = f"memory:{ref.artifact_type}/{ref.hash12}"
        if key not in self._records:
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.UNRESOLVABLE, location=location,
                detail=f"no artifact of type {ref.artifact_type!r} with hash12 "
                       f"{ref.hash12!r} in this registry")
        stored = self._records[key]
        if not stored.startswith(ref.bare_hash):
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MISMATCHED, location=location,
                detail=f"stored self_hash {stored!r} does not extend the "
                       f"referenced digest {ref.bare_hash!r}")
        return EvidenceResolution(ref=ref, status=EvidenceStatus.RESOLVED,
                                  location=location, detail="ok")


class RepositoryEvidenceRegistry:
    """Resolves references against the on-disk `registry/` tree.

    Read-only: this class opens files and never writes one. `interplab.registry`
    remains the only writer.
    """

    def __init__(self, root: Path | str = REGISTRY_ROOT) -> None:
        self.root = Path(root)

    def path_for(self, ref: EvidenceRef) -> Path:
        return self.root / ref.artifact_type / f"{ref.hash12}.json"

    def resolve(self, ref: EvidenceRef) -> EvidenceResolution:
        path = self.path_for(ref)
        location = str(path)

        if not path.is_file():
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.UNRESOLVABLE, location=location,
                detail="no registry artifact at this path")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MALFORMED, location=location,
                detail=f"registry artifact could not be read as JSON: {exc}")
        if not isinstance(raw, dict):
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MALFORMED, location=location,
                detail=f"registry artifact is a {type(raw).__name__}, not an object")

        missing = [k for k in ("artifact_type", "self_hash") if k not in raw]
        if missing:
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MALFORMED, location=location,
                detail=f"registry artifact is missing {missing}")
        if raw["artifact_type"] != ref.artifact_type:
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MISMATCHED, location=location,
                detail=f"registry artifact declares artifact_type "
                       f"{raw['artifact_type']!r}, reference claims "
                       f"{ref.artifact_type!r}")
        self_hash = raw["self_hash"]
        if not isinstance(self_hash, str):
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MALFORMED, location=location,
                detail=f"registry artifact self_hash is a "
                       f"{type(self_hash).__name__}, not a string")
        stored = self_hash.removeprefix("sha256:")
        if not stored.startswith(ref.bare_hash):
            # The 12-hex file name matched but the full digest did not: either a
            # prefix collision or a reference to a superseded artifact. Either
            # way this is not the artifact that was cited.
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MISMATCHED, location=location,
                detail=f"registry artifact self_hash {self_hash!r} does not "
                       f"extend the referenced digest {ref.bare_hash!r}")
        return EvidenceResolution(ref=ref, status=EvidenceStatus.RESOLVED,
                                  location=location, detail="ok")


def resolve_all(refs: tuple[EvidenceRef, ...],
                registry: EvidenceRegistry) -> tuple[EvidenceResolution, ...]:
    """Resolves every reference, in order. Never short-circuits: an author
    fixing an entry should see all of its broken references at once."""
    return tuple(registry.resolve(ref) for ref in refs)
