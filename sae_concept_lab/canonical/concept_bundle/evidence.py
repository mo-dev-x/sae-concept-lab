"""Evidence resolution: read the artifact, recompute its digest, compare.

An `EvidenceRef` is a claim that a registry artifact with a given content hash
exists and says something. This module is the only place that checks whether the
claim holds, and the only module in the package that touches the filesystem.

THE DEFECT THIS MODULE WAS CORRECTED FOR. An earlier version compared the
reference's `artifact_hash` against the `self_hash` the artifact file DECLARED
ABOUT ITSELF. That is not verification, it is transcription: a file whose payload
had been edited while its `self_hash` field was left alone passed, and so did a
file whose `self_hash` was simply typed to match the reference. Nothing was
hashed. The registry root is an ACCESS PATH, not a trust anchor, and neither the
existence of a file nor its own claim about its digest can attest to anything.

WHAT VERIFICATION NOW MEANS. The artifact's bytes are read, parsed, and its
content hash is recomputed here, independently, in the artifact's documented
digest domain:

    self_hash = "sha256:" + sha256(canonical_json(artifact without self_hash))

where canonical JSON is object members sorted by key, no insignificant
whitespace, UTF-8 with non-ASCII unescaped, and non-finite numbers refused
(blueprint 2.1 / 2.2). The recomputed digest is compared to the REFERENCE. The
artifact's own `self_hash` field is then compared to the recomputation as well,
but only to detect and report a record that lies about itself -- it is never the
thing that attests.

WHY THE DIGEST IS REIMPLEMENTED HERE. `interplab.core.hashing.hash_self` is the
canonical implementation and remains the only writer of registry digests. This
package must stay importable with the standard library alone, so it cannot import
it. The safeguard against the two drifting is a differential test that recomputes
every tracked registry artifact both ways and requires agreement, plus edge cases
covering non-ASCII, floats, nesting and empty containers. Divergence is a test
failure rather than a silent difference of opinion about what an artifact is.

CANONICAL DOMAIN, NOT BYTE DOMAIN. The digest covers the artifact's CONTENT, so
reformatting the file's whitespace does not change it -- that is what a canonical
form is for, and the registry addresses artifacts by exactly this digest. Any
change to a key or a value does change it. The digest of the literal file bytes
is recorded alongside as `raw_sha256` for callers that want to pin byte identity
too; it is not authoritative, because it is not the address.

CONTENT INTEGRITY IS NOT RECORD VALIDITY. A file can hash to exactly the digest
that was cited and still not be a registry record: no `created_at`, a
`schema_version` that is a string, a `created_by` missing the run that produced
it. The digest only says the bytes are the bytes that were cited -- it says
nothing about whether they are an artifact. Every canonical field the release
path reads is therefore checked for presence, type and emptiness separately, and
a record that fails is INVALID_RECORD even though its content verified.

WHAT IS NOT VERIFIED, SAID INLINE. A registry record's `subject` entries and its
payload carry hashes of OTHER things -- corpora, checkpoints, directories. None
of them is read or rehashed here, and neither the omission nor the surrounding
prose is allowed to communicate that: every such hash is emitted carrying the
words "recorded, not revalidated", and the digest of the literal file bytes is
emitted carrying "non-authoritative". A caller that renders these values
therefore cannot render them as verified without deleting a label.

FAIL CLOSED, WITH A NAMED REASON. Missing, outside the root, unreadable, empty,
a different artifact type, a digest that does not match the reference, a record
whose own declaration disagrees with its content, a record that is not a valid
registry record, and one content address filed under two types are nine distinct
outcomes, because they call for nine different corrections. Only RESOLVED, and
only with `content_verified`, permits publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .schema import EvidenceRef

#: Repository root, then the registry tree inside it. Resolved from this file's
#: location so the default works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = REPO_ROOT / "registry"

SHA256_PREFIX = "sha256:"

#: The field an artifact declares its own digest in. Excluded from the digest
#: domain, for the obvious reason.
SELF_HASH_FIELD = "self_hash"

#: Stated in every resolution so a caller never has to guess what was hashed.
CONTENT_DIGEST_DOMAIN = (
    "sha256(canonical_json(artifact without the self_hash field)); canonical "
    "JSON = keys sorted, separators (',',':'), non-ASCII unescaped, non-finite "
    "numbers refused"
)

#: An `artifact_type` becomes a path component under the registry root, so it
#: must be exactly one safe component. The schema now constrains the field at
#: construction and decoding time as well; this check is kept as the independent
#: barrier at the point a path is actually built, because a reference can reach
#: here without having been through either -- reconstructed from a cache,
#: unpickled, or built by a caller that bypassed validation.
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: Refused separately from the pattern above, with its own reason: on Windows and
#: macOS `Census_Report` and `census_report` name the SAME directory, so a
#: reference that resolves to one artifact there would resolve to a different one
#: (or to nothing) on Linux.
_CASE_CANONICAL_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_]*$")

#: The digest of the literal file bytes is reported for callers that want to pin
#: byte identity, and is NOT what the registry addresses by. Emitted with this
#: label attached rather than left to be inferred from its field name.
RAW_SHA256_LABEL = "non-authoritative"

#: Hashes a record carries of things that are not the record -- corpora,
#: checkpoints, directories, other artifacts. Resolution does not read or rehash
#: any of them (that is deferred work, not done work), so every one is emitted
#: carrying this label inline.
PAYLOAD_HASH_LABEL = "recorded, not revalidated"

#: The digest that IS authoritative: the one recomputed here, in the canonical
#: domain, and compared to the reference.
CONTENT_DIGEST_LABEL = "authoritative"


class ContentDigestError(ValueError):
    """The artifact cannot be canonicalized, so it has no content digest."""


def _reject_non_canonical(obj: Any) -> None:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ContentDigestError(
                f"canonical JSON forbids non-finite float: {obj!r}")
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ContentDigestError(
                    f"canonical JSON object keys must be str, got {type(key)!r}")
            _reject_non_canonical(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _reject_non_canonical(value)


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON as UTF-8 bytes, ready for hashing.

    Byte-identical to `interplab.core.canonical_json.canonicalize`, which is
    asserted by a differential test rather than assumed.
    """
    _reject_non_canonical(obj)
    try:
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise ContentDigestError(f"artifact is not canonicalizable: {exc}") from exc
    return text.encode("utf-8")


def content_digest(record: Mapping[str, Any]) -> str:
    """The artifact's content hash, recomputed from the artifact.

    The `self_hash` field is stripped before hashing: an artifact cannot be part
    of its own digest. Whatever that field claims is irrelevant here.
    """
    if not isinstance(record, Mapping):
        raise ContentDigestError(
            f"artifact must be an object, got {type(record).__name__}")
    stripped = {k: v for k, v in record.items() if k != SELF_HASH_FIELD}
    return SHA256_PREFIX + hashlib.sha256(canonical_json_bytes(stripped)).hexdigest()


def _bare(digest: str) -> str:
    return digest.removeprefix(SHA256_PREFIX)


# ---------------------------------------------------------------------------
# RECORD VALIDITY -- separate from, and not substitutable by, content integrity
# ---------------------------------------------------------------------------
# The set below is DERIVED FROM THE PUBLICATION CODE: every field named here is
# read by resolution or emitted into the release record, and `consumed_by` names
# where. Nothing is required because it is conventional -- a field the release
# path does not read is not the release path's business, and requiring it here
# would make this module the second, drifting definition of what an artifact is.
#
# The types are the canonical §2.1 envelope's (see `interplab.core.envelope`).
# This module cannot import that definition and stay standard-library-only, so a
# test validates every tracked registry artifact against the set below instead.

@dataclass(frozen=True, slots=True)
class RecordField:
    """One canonical evidence-record field the release path consumes."""

    name: str
    json_type: str
    non_empty: bool
    consumed_by: str

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.name, "json_type": self.json_type,
                "non_empty_required": self.non_empty,
                "consumed_by": self.consumed_by}


#: Nested requirements, kept explicit rather than expressed as a schema, because
#: the point is that they are checkable without a schema library.
_CREATED_BY_FIELDS: tuple[str, ...] = ("run_id", "code_commit", "entrypoint", "host")
_SUBJECT_FIELDS: tuple[str, ...] = ("content_hash", "location", "role")

PUBLICATION_RECORD_FIELDS: tuple[RecordField, ...] = (
    RecordField("artifact_type", "string", True,
                "RepositoryEvidenceRegistry.resolve compares it to the "
                "reference and it names the directory the artifact was read "
                "from; the release record emits it"),
    RecordField("self_hash", "string", True,
                "_compare_digests reads it as the record's declaration about "
                "itself, to detect a record that contradicts its own content; "
                "emitted as declared_digest"),
    RecordField("schema_version", "integer", True,
                "EvidenceRecordIdentity: the release record states which "
                "envelope version the evidence was written under"),
    RecordField("created_at", "string", True,
                "EvidenceRecordIdentity: the release record states when the "
                "evidence was created"),
    RecordField("created_by", "object", True,
                "EvidenceRecordIdentity: the release record states which run, "
                "commit, entrypoint and host produced the evidence"),
    RecordField("subject", "array", False,
                "payload_hash_claims: each subject's content_hash is emitted "
                "labelled 'recorded, not revalidated'. An empty subject list is "
                "valid -- a corpus manifest is a root and has no subject"),
    RecordField("payload", "object", True,
                "content_digest hashes it as the substance being attested, and "
                "payload_hash_claims scans it for hashes of other things"),
)


def _type_problem(name: str, value: Any, json_type: str, non_empty: bool) -> str | None:
    if json_type == "string":
        if not isinstance(value, str):
            return f"{name} must be a string, got {type(value).__name__}"
        if non_empty and not value.strip():
            return f"{name} is empty"
    elif json_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{name} must be an integer, got {type(value).__name__}"
        if non_empty and value < 1:
            return f"{name} is {value}, which names no schema version"
    elif json_type == "object":
        if not isinstance(value, Mapping):
            return f"{name} must be an object, got {type(value).__name__}"
        if non_empty and not value:
            return f"{name} is empty"
    elif json_type == "array":
        if not isinstance(value, list):
            return f"{name} must be an array, got {type(value).__name__}"
        if non_empty and not value:
            return f"{name} is empty"
    return None


def record_validity_problems(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Every way this record fails to be a usable registry record, at once.

    Returns an empty tuple for a valid one. Never raises: a caller asking whether
    a record is valid should not have to catch anything to find out.

    Unknown extra fields are NOT refused. They are covered by the content digest,
    so they cannot be introduced unnoticed, and refusing them here would make
    this consumer the thing that has to be edited before the canonical writer can
    add a field.
    """
    if not isinstance(record, Mapping):
        return (f"record must be an object, got {type(record).__name__}",)

    problems: list[str] = []
    for field in PUBLICATION_RECORD_FIELDS:
        if field.name not in record:
            problems.append(f"{field.name} is missing")
            continue
        problem = _type_problem(field.name, record[field.name], field.json_type,
                                field.non_empty)
        if problem is not None:
            problems.append(problem)

    created_by = record.get("created_by")
    if isinstance(created_by, Mapping):
        for name in _CREATED_BY_FIELDS:
            if name not in created_by:
                problems.append(f"created_by.{name} is missing")
            else:
                problem = _type_problem(f"created_by.{name}", created_by[name],
                                        "string", True)
                if problem is not None:
                    problems.append(problem)

    subject = record.get("subject")
    if isinstance(subject, list):
        for index, entry in enumerate(subject):
            if not isinstance(entry, Mapping):
                problems.append(
                    f"subject[{index}] must be an object, got "
                    f"{type(entry).__name__}")
                continue
            for name in _SUBJECT_FIELDS:
                if name not in entry:
                    problems.append(f"subject[{index}].{name} is missing")
                else:
                    problem = _type_problem(f"subject[{index}].{name}",
                                            entry[name], "string", True)
                    if problem is not None:
                        problems.append(problem)
    return tuple(problems)


@dataclass(frozen=True, slots=True)
class EvidenceRecordIdentity:
    """What the release record says about the evidence artifact itself.

    Constructed only from a record that passed `record_validity_problems`, which
    is what makes reading these fields safe here and what makes them *consumed*
    rather than merely present.
    """

    artifact_type: str
    schema_version: int
    created_at: str
    run_id: str
    code_commit: str
    entrypoint: str
    host: str
    subject_count: int

    @classmethod
    def of(cls, record: Mapping[str, Any]) -> EvidenceRecordIdentity:
        created_by = record["created_by"]
        return cls(artifact_type=record["artifact_type"],
                   schema_version=record["schema_version"],
                   created_at=record["created_at"],
                   run_id=created_by["run_id"],
                   code_commit=created_by["code_commit"],
                   entrypoint=created_by["entrypoint"],
                   host=created_by["host"],
                   subject_count=len(record["subject"]))

    def as_dict(self) -> dict[str, Any]:
        return {"artifact_type": self.artifact_type,
                "schema_version": self.schema_version,
                "created_at": self.created_at,
                "run_id": self.run_id,
                "code_commit": self.code_commit,
                "entrypoint": self.entrypoint,
                "host": self.host,
                "subject_count": self.subject_count}


@dataclass(frozen=True, slots=True)
class PayloadHashClaim:
    """A hash a record carries of something that is NOT the record.

    The label is a field of the object, not a note next to it, so a claim cannot
    be rendered without it: `as_dict()` always carries the words "recorded, not
    revalidated", and there is no constructor that omits them.
    """

    path: str
    value: str
    label: str = PAYLOAD_HASH_LABEL

    def __post_init__(self) -> None:
        if self.label != PAYLOAD_HASH_LABEL:
            raise ValueError(
                f"a payload hash claim is always labelled "
                f"{PAYLOAD_HASH_LABEL!r}: it names something this module did not "
                f"read, and relabelling it would assert otherwise")

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "value": self.value, "label": self.label}


_HASH_KEY_MARKERS: tuple[str, ...] = ("hash", "digest", "sha256", "checksum")


def _looks_like_a_hash_field(key: str) -> bool:
    low = key.lower()
    return any(marker in low for marker in _HASH_KEY_MARKERS)


def _scan_for_hashes(node: Any, path: str, out: list[PayloadHashClaim]) -> None:
    if isinstance(node, Mapping):
        for key in sorted(str(k) for k in node):
            value = node[key]
            child = f"{path}.{key}"
            if isinstance(value, str) and _looks_like_a_hash_field(key):
                out.append(PayloadHashClaim(path=child, value=value))
            else:
                _scan_for_hashes(value, child, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_for_hashes(value, f"{path}[{index}]", out)


def payload_hash_claims(record: Mapping[str, Any]) -> tuple[PayloadHashClaim, ...]:
    """Every hash this record carries of something other than itself.

    Two sources: the canonical `subject[].content_hash` links, and any hash-named
    string anywhere in the opaque payload. The payload scan is by key name, which
    is a heuristic -- but the direction it errs in is labelling one value too
    many, never claiming verification for one it missed.

    NONE of these is resolved or rehashed. That is deferred work, and the labels
    say so at every point the values are emitted.
    """
    claims: list[PayloadHashClaim] = []
    subject = record.get("subject")
    if isinstance(subject, list):
        for index, entry in enumerate(subject):
            if isinstance(entry, Mapping) and isinstance(entry.get("content_hash"), str):
                claims.append(PayloadHashClaim(
                    path=f"subject[{index}].content_hash",
                    value=entry["content_hash"]))
    payload = record.get("payload")
    if isinstance(payload, Mapping):
        _scan_for_hashes(payload, "payload", claims)
    return tuple(claims)


class EvidenceStatus(StrEnum):
    """Outcome of resolving one reference. Only RESOLVED permits publishing."""

    #: Read, digest recomputed from content, and it matches the reference.
    RESOLVED = "resolved"
    #: No artifact at the address the reference names.
    UNRESOLVABLE = "unresolvable"
    #: The reference would resolve outside the registry root.
    OUT_OF_ROOT = "out_of_root"
    #: Present but not readable as a registry artifact: unreadable, empty, not
    #: JSON, not an object, or not canonicalizable.
    MALFORMED = "malformed"
    #: Readable and well formed, but a different artifact than the one cited.
    MISMATCHED = "mismatched"
    #: Content recomputes to a digest the reference does not name.
    DIGEST_MISMATCH = "digest_mismatch"
    #: The artifact's own `self_hash` disagrees with its own content.
    TAMPERED = "tampered"
    #: The content hashes to exactly what was cited, and the record is still not
    #: a valid registry record: a field the release path reads is missing, of the
    #: wrong type, or empty. Integrity does not substitute for validity.
    INVALID_RECORD = "invalid_record"
    #: One content address is filed under more than one artifact type.
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class EvidenceResolution:
    """What happened when one reference was looked up, and what was verified.

    `location` is RELATIVE to the registry root so that a resolution is
    reproducible across machines and comparable in a conformance vector; the root
    itself is recorded separately in `registry_root`.
    """

    ref: EvidenceRef
    status: EvidenceStatus
    #: Set only on a resolution that is USABLE AS EVIDENCE: the digest was
    #: recomputed from content, it matched the reference, and the record is a
    #: valid registry record. A record whose content verified but whose fields
    #: did not is INVALID_RECORD and does not carry this flag -- its
    #: `recomputed_digest` and `detail` still report that the digest matched, so
    #: nothing is hidden. Publishing requires this flag.
    content_verified: bool = False
    location: str = ""
    registry_root: str = ""
    detail: str = ""
    #: The digest this module computed from the artifact's content. Authoritative:
    #: it is the address the registry files content under.
    recomputed_digest: str = ""
    #: What the artifact claimed about itself. Reported, never trusted.
    declared_digest: str = ""
    #: sha256 of the literal file bytes. NOT authoritative -- it is not the
    #: address, and it changes when the file is reformatted. Emitted through
    #: `as_record()` carrying that label rather than as a bare number.
    raw_sha256: str = ""
    digest_domain: str = ""
    #: "full" when the reference carried all 64 hex characters, "prefix" when it
    #: carried fewer and could therefore only be matched as a prefix.
    digest_comparison: str = ""
    #: Every way the record failed to be a valid registry record. Empty on a
    #: resolution that never got far enough to look.
    record_validity_problems: tuple[str, ...] = ()
    #: The evidence artifact's own identity, on a valid record.
    record_identity: EvidenceRecordIdentity | None = None
    #: Hashes the record carries of OTHER things. Never resolved or rehashed.
    payload_hash_claims: tuple[PayloadHashClaim, ...] = ()

    def __post_init__(self) -> None:
        if self.content_verified and self.status is not EvidenceStatus.RESOLVED:
            raise ValueError(
                "content_verified is only meaningful on a RESOLVED resolution")

    @property
    def resolved(self) -> bool:
        """Resolved AND content-verified. Existence alone never counts."""
        return self.status is EvidenceStatus.RESOLVED and self.content_verified

    @property
    def is_full_digest_match(self) -> bool:
        return self.resolved and self.digest_comparison == "full"

    @property
    def is_publication_digest(self) -> bool:
        """Verified by content AND cited in the form a public release requires.

        Distinct from `is_full_digest_match`, which reports how many characters
        were compared. This reports whether the reference is written the way a
        published artefact must write it, algorithm prefix included.
        """
        return self.resolved and self.ref.is_publication_digest

    def as_reason(self) -> str:
        """One line for the publication gate's blocking reasons."""
        detail = self.detail
        if self.record_validity_problems:
            detail = f"{detail} [{'; '.join(self.record_validity_problems)}]"
        return (f"evidence {self.ref.artifact_type}:{self.ref.artifact_hash} is "
                f"{self.status.value}: {detail}")

    def as_record(self) -> dict[str, Any]:
        """The auditable form, with every digest carrying what it is worth.

        The two labels are attached to the values themselves. A caller rendering
        this cannot present the literal-byte digest as authoritative, or a
        recorded payload hash as revalidated, without deleting a label that is
        sitting next to the number.
        """
        return {
            "artifact_type": self.ref.artifact_type,
            "artifact_hash": self.ref.artifact_hash,
            "status": self.status.value,
            "content_verified": self.content_verified,
            "location": self.location,
            "digest_comparison": self.digest_comparison,
            "publication_digest_form": self.ref.is_publication_digest,
            "content_digest": {"value": self.recomputed_digest,
                               "label": CONTENT_DIGEST_LABEL,
                               "domain": self.digest_domain},
            "declared_self_hash": self.declared_digest,
            "raw_sha256": {"value": self.raw_sha256, "label": RAW_SHA256_LABEL},
            "record_validity_problems": list(self.record_validity_problems),
            "record_identity": (None if self.record_identity is None
                                else self.record_identity.as_dict()),
            "payload_hashes": [c.as_dict() for c in self.payload_hash_claims],
            "detail": self.detail,
        }


@runtime_checkable
class EvidenceRegistry(Protocol):
    """Anything that can resolve a reference by reading and hashing content."""

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
            digest_domain=CONTENT_DIGEST_DOMAIN,
            detail="no evidence registry was supplied; no content could be read "
                   "or hashed, and publishing therefore fails closed")


def _compare_digests(ref: EvidenceRef, record: Mapping[str, Any], *,
                     location: str, registry_root: str,
                     raw_sha256: str) -> EvidenceResolution:
    """The one comparison, shared by every registry implementation.

    Sharing it is deliberate: a test-only registry that were laxer than the
    repository one would let a conformance vector pass for the wrong reason.
    """
    common = {"ref": ref, "location": location, "registry_root": registry_root,
              "raw_sha256": raw_sha256, "digest_domain": CONTENT_DIGEST_DOMAIN}

    declared = record.get(SELF_HASH_FIELD)
    if not isinstance(declared, str) or not declared:
        return EvidenceResolution(
            status=EvidenceStatus.MALFORMED, **common,
            detail=f"artifact has no usable {SELF_HASH_FIELD} field, so it does "
                   f"not declare an identity to check its content against")

    try:
        recomputed = content_digest(record)
    except ContentDigestError as exc:
        return EvidenceResolution(status=EvidenceStatus.MALFORMED, **common,
                                  declared_digest=declared, detail=str(exc))

    common = {**common, "recomputed_digest": recomputed,
              "declared_digest": declared}
    reference_bare, recomputed_bare = ref.bare_hash, _bare(recomputed)
    declared_bare = _bare(declared)
    full = len(reference_bare) == 64

    if not recomputed_bare.startswith(reference_bare):
        lies = declared_bare != recomputed_bare
        return EvidenceResolution(
            status=EvidenceStatus.DIGEST_MISMATCH, **common,
            detail=(
                f"content recomputes to {recomputed} which the reference "
                f"{ref.artifact_hash} does not name"
                + (f"; the artifact's own {SELF_HASH_FIELD} {declared} also "
                   f"disagrees with its content, so it was edited after it was "
                   f"written" if lies else
                   f"; the artifact's own {SELF_HASH_FIELD} agrees with its "
                   f"content, so the reference cites a different artifact")))

    if declared_bare != recomputed_bare:
        return EvidenceResolution(
            status=EvidenceStatus.TAMPERED, **common,
            detail=f"content recomputes to {recomputed} but the artifact "
                   f"declares {declared}; the reference happens to match the "
                   f"content, and the record still lies about itself")

    # Integrity is settled. It is NOT a substitute for the record being a record:
    # these bytes are the bytes that were cited, which says nothing about whether
    # they carry the fields the release path reads. Checked after the digest, so
    # that an edited artifact is reported as edited rather than as incomplete --
    # the two call for different corrections, and tampering is the one nobody may
    # be allowed to mistake for sloppiness.
    problems = record_validity_problems(record)
    if problems:
        return EvidenceResolution(
            status=EvidenceStatus.INVALID_RECORD, **common,
            record_validity_problems=problems,
            detail=(f"content read and digest recomputed, and it MATCHES the "
                    f"reference -- the record is still not a valid registry "
                    f"record: {'; '.join(problems)}. A correct digest attests "
                    f"that these are the bytes that were cited, not that they "
                    f"are an artifact."))

    return EvidenceResolution(
        status=EvidenceStatus.RESOLVED, content_verified=True, **common,
        digest_comparison="full" if full else "prefix",
        record_identity=EvidenceRecordIdentity.of(record),
        payload_hash_claims=payload_hash_claims(record),
        detail=("content read and digest recomputed; matches the reference in "
                "full" if full else
                "content read and digest recomputed; the reference carries only "
                f"{len(reference_bare)} of 64 hex characters, so it was matched "
                "as a prefix"))


class InMemoryEvidenceRegistry:
    """A registry holding artifact CONTENT in memory.

    It stores records, not digests, and recomputes on every resolution through
    the same comparison the repository registry uses. An earlier version stored
    bare digests, which made it structurally incapable of verifying anything and
    therefore useless as a test double for a content-verifying gate.
    """

    def __init__(self, records: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
                 ) -> None:
        self._records: dict[tuple[str, str], dict[str, Any]] = {
            key: dict(value) for key, value in (records or {}).items()}

    def add(self, artifact_type: str, record: Mapping[str, Any]) -> str:
        """Stores an artifact and returns the digest a reference must cite.

        The record is stored as given, including whatever `self_hash` it
        carries -- a tampered record must be storable, or the tampering case
        could not be tested.
        """
        digest = content_digest(record)
        self._records[(artifact_type, _bare(digest)[:12])] = dict(record)
        return digest

    def add_at(self, artifact_type: str, hash12: str,
               record: Mapping[str, Any]) -> None:
        """Stores a record at an EXPLICIT address, including one its content does
        not hash to.

        Needed to reproduce the on-disk tampering case in memory: a file whose
        contents were edited keeps the path it was written at, so the address and
        the content disagree. Without this, the in-memory double could only ever
        hold self-consistent records and would be unable to express the failure
        the gate exists to catch.
        """
        self._records[(artifact_type, hash12)] = dict(record)

    def resolve(self, ref: EvidenceRef) -> EvidenceResolution:
        location = f"{ref.artifact_type}/{ref.hash12}.json"
        key = (ref.artifact_type, ref.hash12)
        if key not in self._records:
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.UNRESOLVABLE, location=location,
                registry_root="memory:", digest_domain=CONTENT_DIGEST_DOMAIN,
                detail=f"no artifact of type {ref.artifact_type!r} with hash12 "
                       f"{ref.hash12!r} in this registry")
        record = self._records[key]
        declared_type = record.get("artifact_type")
        if declared_type != ref.artifact_type:
            return EvidenceResolution(
                ref=ref, status=EvidenceStatus.MISMATCHED, location=location,
                registry_root="memory:", digest_domain=CONTENT_DIGEST_DOMAIN,
                detail=f"artifact declares artifact_type {declared_type!r}, "
                       f"reference claims {ref.artifact_type!r}")
        raw = canonical_json_bytes(record)
        return _compare_digests(ref, record, location=location,
                                registry_root="memory:",
                                raw_sha256=hashlib.sha256(raw).hexdigest())


class RepositoryEvidenceRegistry:
    """Resolves references against an on-disk `registry/` tree.

    The root is an access path and nothing more. Being under it does not make a
    file evidence; being read, hashed, and matching the reference does.

    Read-only: this class opens files and never writes one. `interplab.registry`
    remains the only writer.
    """

    def __init__(self, root: Path | str = REGISTRY_ROOT) -> None:
        self.root = Path(root)

    # -- addressing ------------------------------------------------------
    def path_for(self, ref: EvidenceRef) -> Path:
        return self.root / ref.artifact_type / f"{ref.hash12}.json"

    def _unsafe_component(self, ref: EvidenceRef) -> str | None:
        """Why this reference may not be turned into a path, or None.

        `artifact_type` is used here as a directory name, which makes it the
        traversal vector. Refused before any path is joined, and the containment
        of the joined path is then checked independently -- two barriers, because
        the first is a pattern and patterns get relaxed.

        The schema refuses the same shapes at construction and the codec refuses
        them at decoding. This is not redundancy: a reference can arrive here
        without passing either, and the barrier that matters is the one standing
        where the path is actually built.
        """
        if not _SAFE_PATH_COMPONENT.match(ref.artifact_type):
            return (f"artifact_type {ref.artifact_type!r} is not a single safe "
                    f"path component")
        if ref.artifact_type in {".", ".."}:
            return f"artifact_type {ref.artifact_type!r} is a path traversal"
        if not _CASE_CANONICAL_COMPONENT.match(ref.artifact_type):
            return (f"artifact_type {ref.artifact_type!r} is not a canonical "
                    f"lowercase registry directory name "
                    f"({_CASE_CANONICAL_COMPONENT.pattern}); an uppercase form "
                    f"names the same directory as its lowercase twin on a "
                    f"case-insensitive filesystem and a different one on Linux, "
                    f"so the same reference would resolve to different artifacts "
                    f"on different machines")
        if not _SAFE_PATH_COMPONENT.match(ref.hash12):
            return f"hash12 {ref.hash12!r} is not a single safe path component"
        return None

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.name

    # -- resolution ------------------------------------------------------
    def resolve(self, ref: EvidenceRef) -> EvidenceResolution:
        root_string = str(self.root)
        common = {"ref": ref, "registry_root": root_string,
                  "digest_domain": CONTENT_DIGEST_DOMAIN}

        unsafe = self._unsafe_component(ref)
        if unsafe is not None:
            return EvidenceResolution(status=EvidenceStatus.OUT_OF_ROOT, **common,
                                      detail=f"{unsafe}; refused before any path "
                                             f"was joined")

        path = self.path_for(ref)
        resolved_root = self.root.resolve()
        if not path.resolve().is_relative_to(resolved_root):
            return EvidenceResolution(
                status=EvidenceStatus.OUT_OF_ROOT, **common,
                detail="the reference resolves outside the registry root")

        location = self._relative(path)
        common = {**common, "location": location}

        if not path.is_file():
            return EvidenceResolution(status=EvidenceStatus.UNRESOLVABLE, **common,
                                      detail="no registry artifact at this path")
        try:
            data = path.read_bytes()
        except OSError as exc:
            return EvidenceResolution(status=EvidenceStatus.MALFORMED, **common,
                                      detail=f"artifact could not be read: {exc}")
        raw_sha256 = hashlib.sha256(data).hexdigest()
        common = {**common, "raw_sha256": raw_sha256}

        if not data.strip():
            return EvidenceResolution(
                status=EvidenceStatus.MALFORMED, **common,
                detail="artifact is empty; an empty file has no content to "
                       "attest with")
        try:
            record = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return EvidenceResolution(
                status=EvidenceStatus.MALFORMED, **common,
                detail=f"artifact could not be read as JSON: {exc}")
        if not isinstance(record, dict):
            return EvidenceResolution(
                status=EvidenceStatus.MALFORMED, **common,
                detail=f"artifact is a {type(record).__name__}, not an object")

        declared_type = record.get("artifact_type")
        if declared_type != ref.artifact_type:
            return EvidenceResolution(
                status=EvidenceStatus.MISMATCHED, **common,
                detail=f"artifact declares artifact_type {declared_type!r}, "
                       f"reference claims {ref.artifact_type!r}")

        outcome = _compare_digests(ref, record, location=location,
                                   registry_root=root_string,
                                   raw_sha256=raw_sha256)
        if not outcome.resolved:
            return outcome

        duplicates = self._other_types_claiming(ref, outcome.recomputed_digest)
        if duplicates:
            return EvidenceResolution(
                status=EvidenceStatus.AMBIGUOUS, **common,
                recomputed_digest=outcome.recomputed_digest,
                declared_digest=outcome.declared_digest,
                detail=f"this content digest is also filed under "
                       f"{sorted(duplicates)}; one content address naming "
                       f"several artifact types means the registry disagrees "
                       f"with itself about what the content is")
        return outcome

    def _other_types_claiming(self, ref: EvidenceRef, digest: str) -> set[str]:
        """Artifact types other than the cited one holding this same content.

        A bounded, one-level look at the root's immediate subdirectories for a
        file of the same name. This is not discovery -- nothing enters a build by
        being found here; it only detects a registry that contradicts itself, the
        same condition `interplab.registry.get` refuses outright.
        """
        found: set[str] = set()
        if not self.root.is_dir():
            return found
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name == ref.artifact_type:
                continue
            if not _SAFE_PATH_COMPONENT.match(child.name):
                continue
            candidate = child / f"{ref.hash12}.json"
            if not candidate.is_file():
                continue
            try:
                record = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(record, dict) and content_digest(record) == digest:
                    found.add(child.name)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                    ContentDigestError):
                continue
        return found


#: The fail-closed default used wherever a registry argument is omitted. A
#: module singleton rather than a call in a default argument, so it is one
#: greppable name and every caller that forgot to pass a registry shares it.
NO_EVIDENCE_REGISTRY: EvidenceRegistry = NullEvidenceRegistry()


def resolve_all(refs: tuple[EvidenceRef, ...],
                registry: EvidenceRegistry) -> tuple[EvidenceResolution, ...]:
    """Resolves every reference, in order. Never short-circuits: an author
    fixing an entry should see all of its broken references at once."""
    return tuple(registry.resolve(ref) for ref in refs)
