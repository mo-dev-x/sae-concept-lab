"""Fail-closed publication gate.

An entry is publishable only by positively satisfying every condition below. It
is never publishable merely because no check happened to trip -- which is why
the schema default for `provenance` is UNKNOWN and why the default evidence
registry resolves nothing. An entry authored by someone who never thought about
publication is blocked by construction.

FOUR CONDITIONS, ALL POSITIVE:

  1. `provenance` is exactly ATTESTED. CANDIDATE, DRAFT, FAKE and UNKNOWN are
     all inspectable in development and none of them publishes. There is no
     "attested enough".

  2. Every evidence reference is RESOLVED AND VERIFIED BY CONTENT against the
     registry: the artifact is read, its digest is recomputed here, and the
     recomputation matches the reference. Existence does not attest, and neither
     does the artifact's own claim about its digest. Missing, out of root,
     malformed, mismatched, digest-mismatched, self-contradictory, invalid and
     ambiguous are distinct failures and every one of them blocks. An attestation
     whose evidence cannot be found -- or can be found and does not hash to what
     was cited -- is a claim about a document nobody can read.

  3. Every cited artifact is a VALID REGISTRY RECORD, field by field. A correct
     digest says these are the bytes that were cited; it does not say they are an
     artifact. See `evidence.PUBLICATION_RECORD_FIELDS`, which is derived from
     what this module and the resolver actually read.

  4. Every reference is written in the PUBLISHABLE FORM: `sha256:` and all 64
     hex characters. Applied unconditionally, with no override anywhere.

The gate also sniffs ids for placeholder markers, because the likeliest
publication accident is not a mis-set enum -- it is real-looking metadata
attached to fixture data somebody forgot to replace. That check is what keeps
fixture quarantine a property of the data rather than of a flag.

WHAT THE RELEASE SAYS IT CHECKED. One registry record was read and rehashed.
The corpora, checkpoints and directories that record POINTS AT were not, and
`RELEASE_EVIDENCE_STATEMENT` says so in the same breath as the positive claim,
because a positive claim shipped on its own is read as covering everything
downstream of it. `prohibited_release_claims` refuses the phrasings that would
make that larger claim, and is applied to the rendered output rather than left
as a style note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .errors import ReleaseBlockedError, ReleaseBuildError
from .evidence import (
    CONTENT_DIGEST_DOMAIN,
    CONTENT_DIGEST_LABEL,
    NO_EVIDENCE_REGISTRY,
    PAYLOAD_HASH_LABEL,
    PUBLICATION_RECORD_FIELDS,
    RAW_SHA256_LABEL,
    EvidenceRegistry,
    EvidenceResolution,
    resolve_all,
)
from .schema import (
    PLACEHOLDER_MARKERS,
    PUBLICATION_ARTIFACT_HASH_RE,
    BundleEntry,
    Direction,
    Provenance,
)

#: The only provenance that may be published. Named as a constant so the rule is
#: greppable and so no caller can widen it by writing `in (...)` at a call site.
PUBLISHABLE_PROVENANCE = Provenance.ATTESTED

#: A published concept needs ONE calibrated direction, not both.
#:
#: Product ruling: a concept whose suppression was never calibrated is still a
#: usable control, and withholding it until both directions exist would hide
#: finished work behind unfinished work. The uncalibrated direction is disabled
#: in the UI rather than absent from the catalog, which is why availability
#: travels in the catalog and resolved state (see `LayoutEntry` below and
#: `ResolvedControlState.available_directions`) instead of being re-derived by
#: whoever renders the toggle.
#:
#: This is not a relaxation of the calibration rules. The one direction that is
#: present must satisfy every structural requirement in full, and the entry must
#: still be ATTESTED with evidence that resolves.
MIN_PUBLISHED_DIRECTIONS = 1

#: A PUBLIC-RELEASE INVARIANT, ratified: an evidence reference may be published
#: only as `sha256:` followed by all 64 lowercase hex characters.
#:
#: This constant is a DECLARATION, not a switch. `evaluate_publishability` does
#: not branch on it -- it applies `PUBLICATION_ARTIFACT_HASH_RE` unconditionally
#: -- so setting it to False, patching it at import time, or shadowing it from
#: another module changes nothing about what publishes. There is deliberately no
#: configuration file, environment variable, CLI flag or call-site keyword that
#: weakens the gate, because a gate with an override is a gate that will be
#: found switched off in the build that mattered. A test asserts the constant is
#: True, and a second test asserts that forcing it False still blocks a prefix
#: reference.
#:
#: Prefix references remain fully supported everywhere else: the registry names
#: files by a 12-hex prefix, and resolution reads the artifact and recomputes its
#: digest for a prefix reference exactly as for a full one, reporting
#: `digest_comparison == "prefix"`. Development and inspection are unaffected.
#: Publication is the one place a prefix is not a content address -- twelve hex
#: characters is 48 bits, and a colliding artifact is findable.
REQUIRE_FULL_DIGEST_FOR_PUBLICATION = True

#: Stated in the release record so an auditor does not have to read a regex.
PUBLICATION_DIGEST_FORM = "sha256:<64 lowercase hexadecimal characters>"


# ---------------------------------------------------------------------------
# MANDATORY RELEASE WORDING
# ---------------------------------------------------------------------------
# Ruled by the PM and shipped verbatim. The two sentences are ONE string, joined
# by a single space, because the second is what stops the first from being read
# as a claim about the corpora, datasets and checkpoints the evidence record
# points at -- none of which is read or rehashed by anything in this package.
# Keeping them as separate constants that a caller assembles would make the
# negative sentence droppable by a caller who found it inconvenient; keeping
# them as one string means dropping it is an edit to this file.

EVIDENCE_VERIFICATION_SENTENCE = (
    "Evidence registry record resolved and content-verified "
    "(SHA-256 over canonical JSON, self_hash excluded).")

PAYLOAD_LIMIT_SENTENCE = (
    "Payload targets referenced by this record were not resolved or verified.")

#: The exact text every public and release rendering must carry, in this order,
#: adjacent, with nothing between them.
RELEASE_EVIDENCE_STATEMENT = (
    f"{EVIDENCE_VERIFICATION_SENTENCE} {PAYLOAD_LIMIT_SENTENCE}")

#: Phrasings that are forbidden in public and release output because each one
#: asserts more than was checked. The first four are the ruled list; the fifth is
#: the general rule they are instances of.
PROHIBITED_RELEASE_CLAIMS: tuple[str, ...] = (
    "evidence verified",
    "artifacts verified",
    "verified against the corpus",
    "verified against the dataset",
    "fully verified",
)

#: The trip-wire. If a public claim ever asserts verification against one of
#: these, transitive verification stops being deferred work and becomes required
#: work -- so the claim is refused instead.
_TRANSITIVE_CLAIM_RE = re.compile(
    r"verif\w*\s+(?:against|of)\s+(?:the\s+|a\s+|an\s+|this\s+|its\s+|these\s+)?"
    r"(corpus|corpora|dataset|datasets|checkpoint|checkpoints)", re.IGNORECASE)

#: What an affirmative verification claim must name, so that it says WHAT was
#: verified rather than that something was.
_VERIFICATION_ANCHOR = "registry record"
_NEGATIONS = ("not ", "never ", "no ", "cannot ", "without ", "neither ")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def prohibited_release_claims(text: str) -> tuple[str, ...]:
    """Every over-claim in `text`, or an empty tuple.

    Three rules, all applied:

      1. the ruled literal phrases, case-insensitively;
      2. any claim of verification against a corpus, dataset or checkpoint,
         which is the trip-wire: this package verifies none of them;
      3. any AFFIRMATIVE sentence claiming verification that does not name the
         registry record. "Verified" on its own is the whole problem -- a reader
         supplies the object, and the object they supply is the science.

    A negated sentence is not a claim, which is what lets the mandatory wording's
    second sentence -- "were not resolved or verified" -- pass a checker whose
    whole purpose is to catch the word it contains.
    """
    lowered = text.lower()
    found: list[str] = []
    found += [f"prohibited phrase: {phrase!r}"
              for phrase in PROHIBITED_RELEASE_CLAIMS if phrase in lowered]
    found += [f"claims verification against {match.group(1)!r}, which this "
              f"package does not read: {match.group(0)!r}"
              for match in _TRANSITIVE_CLAIM_RE.finditer(text)]
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        low = sentence.lower()
        if "verif" not in low or _VERIFICATION_ANCHOR in low:
            continue
        if any(negation in low for negation in _NEGATIONS):
            continue
        found.append(
            f"verification claimed without naming the {_VERIFICATION_ANCHOR}: "
            f"{sentence.strip()!r}")
    return tuple(dict.fromkeys(found))


def assert_release_text_clean(text: str) -> None:
    """Fail-closed check for anything a build shows the public."""
    claims = prohibited_release_claims(text)
    if claims:
        raise ReleaseBuildError(
            f"release text makes {len(claims)} claim(s) larger than what was "
            f"checked: {list(claims)}. Only the registry record is read and "
            f"rehashed; the corpora, datasets and checkpoints it names are not.")


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    """Outcome of the gate. `reasons` is empty iff `publishable` is True."""

    publishable: bool
    concept_id: str
    pairing_id: str
    reasons: tuple[str, ...]
    evidence: tuple[EvidenceResolution, ...] = ()

    @property
    def unresolved_evidence(self) -> tuple[EvidenceResolution, ...]:
        return tuple(r for r in self.evidence if not r.resolved)

    @property
    def evidence_content_verified(self) -> bool:
        """Every cited artifact was read and its digest recomputed from content.

        False when there is no evidence at all: "nothing was cited" must not
        read as "everything checked out".
        """
        return bool(self.evidence) and all(r.resolved for r in self.evidence)

    @property
    def prefix_verified_evidence(self) -> tuple[EvidenceResolution, ...]:
        """Verified by content, but not cited in the publishable form."""
        return tuple(r for r in self.evidence
                     if r.resolved and not r.is_publication_digest)

    @property
    def payload_hash_claims(self) -> tuple[tuple[str, str, str], ...]:
        """(reference, path, value) for every hash the records merely CARRY.

        Flattened here so a release build can render them all without walking
        into the resolutions and losing the labels on the way out.
        """
        return tuple(
            (f"{r.ref.artifact_type}:{r.ref.artifact_hash}", claim.path, claim.value)
            for r in self.evidence for claim in r.payload_hash_claims)

    def content_verification_record(self) -> dict[str, object]:
        """What was verified, for the release record.

        Emitted whether or not the entry passes, because "we checked and it
        failed" and "we never checked" are different facts about a build.

        Carries the mandatory statement, and every digest carries what it is
        worth: the recomputed content digest is labelled authoritative, the
        literal-byte digest non-authoritative, and every hash a record merely
        points with is labelled "recorded, not revalidated".
        """
        return {
            "statement": RELEASE_EVIDENCE_STATEMENT,
            "concept_id": self.concept_id,
            "pairing_id": self.pairing_id,
            "publishable": self.publishable,
            "evidence_refs": len(self.evidence),
            "all_registry_records_content_verified": self.evidence_content_verified,
            "digest_domain": CONTENT_DIGEST_DOMAIN,
            "publication_digest_form": PUBLICATION_DIGEST_FORM,
            "full_digest_required": REQUIRE_FULL_DIGEST_FOR_PUBLICATION,
            "record_validity_fields": [f.as_dict() for f in PUBLICATION_RECORD_FIELDS],
            "labels": {"content_digest": CONTENT_DIGEST_LABEL,
                       "raw_sha256": RAW_SHA256_LABEL,
                       "payload_hashes": PAYLOAD_HASH_LABEL},
            "evidence": [r.as_record() for r in self.evidence],
        }

    def render_release_evidence_note(self) -> str:
        """The human-readable release note.

        Opens with the mandatory statement -- both sentences, adjacent, nothing
        between them -- and then attaches a label to every number it prints. A
        renderer downstream may reformat this; it cannot reformat away a label
        that is inside the line the value is on.
        """
        lines = [RELEASE_EVIDENCE_STATEMENT, ""]
        lines.append(f"concept {self.concept_id} on pairing {self.pairing_id}: "
                     f"{'publishable' if self.publishable else 'blocked'}")
        lines.append(f"digest domain: {CONTENT_DIGEST_DOMAIN}")
        lines.append(f"publication reference form: {PUBLICATION_DIGEST_FORM}")
        if not self.evidence:
            lines.append("registry records cited: none")
        for resolution in self.evidence:
            lines.append(
                f"registry record {resolution.ref.artifact_type}:"
                f"{resolution.ref.artifact_hash} [{resolution.status.value}]")
            lines.append(f"  content digest ({CONTENT_DIGEST_LABEL}): "
                         f"{resolution.recomputed_digest or 'not computed'}")
            lines.append(f"  literal-byte sha256 ({RAW_SHA256_LABEL}): "
                         f"{resolution.raw_sha256 or 'not computed'}")
            for problem in resolution.record_validity_problems:
                lines.append(f"  record field problem: {problem}")
            if not resolution.payload_hash_claims:
                lines.append("  payload targets carried by this record: none")
            for claim in resolution.payload_hash_claims:
                lines.append(f"  payload target {claim.path}: {claim.value} "
                             f"({claim.label})")
        for reason in self.reasons:
            lines.append(f"blocked: {reason}")
        return "\n".join(lines)

    def raise_if_blocked(self) -> None:
        if not self.publishable:
            raise ReleaseBlockedError(concept_id=self.concept_id,
                                      pairing_id=self.pairing_id,
                                      reasons=self.reasons)


def _placeholder_hits(text: str) -> tuple[str, ...]:
    low = text.lower()
    return tuple(m for m in PLACEHOLDER_MARKERS if m in low)


def evaluate_publishability(
    entry: BundleEntry,
    *,
    evidence_registry: EvidenceRegistry = NO_EVIDENCE_REGISTRY,
) -> ReleaseDecision:
    """Non-raising evaluation.

    Collects EVERY reason, not just the first, so an author fixes one entry in
    one pass rather than discovering blockers one build at a time.

    The default registry resolves nothing, so an omitted argument blocks rather
    than skipping the evidence check.
    """
    reasons: list[str] = []

    if entry.provenance is not PUBLISHABLE_PROVENANCE:
        reasons.append(
            f"provenance is {entry.provenance.value!r}, must be exactly "
            f"{PUBLISHABLE_PROVENANCE.value!r}")

    if len(entry.calibrated_directions) < MIN_PUBLISHED_DIRECTIONS:
        reasons.append(
            f"no direction is calibrated: both 'amplify' and 'suppress' are "
            f"null, so there is no control to operate. Publishing needs at "
            f"least {MIN_PUBLISHED_DIRECTIONS} calibrated direction -- not both, "
            f"and an entry with one is published with the other disabled.")

    cp = entry.calibration_provenance
    if cp is None:
        # Unreachable for an ATTESTED entry (the schema requires it), reachable
        # for every other provenance, and re-checked here so the gate does not
        # depend on that invariant holding elsewhere.
        reasons.append("no calibration_provenance: nothing records who "
                       "calibrated this entry, when, or against what evidence")
        resolutions: tuple[EvidenceResolution, ...] = ()
    else:
        resolutions = resolve_all(cp.evidence, evidence_registry)
        for resolution in resolutions:
            if not resolution.resolved:
                reasons.append(resolution.as_reason())
            # NOT `if REQUIRE_FULL_DIGEST_FOR_PUBLICATION`. The pattern is applied
            # unconditionally, so there is no state anywhere -- module constant,
            # environment, config, argument -- that can turn this branch off.
            elif not PUBLICATION_ARTIFACT_HASH_RE.match(resolution.ref.artifact_hash):
                reasons.append(
                    f"evidence registry record {resolution.ref.artifact_type}:"
                    f"{resolution.ref.artifact_hash} was resolved and "
                    f"content-verified, and its reference is not in the form a "
                    f"public release requires. Publication needs "
                    f"{PUBLICATION_DIGEST_FORM}: the algorithm prefix is "
                    f"mandatory, and a 12-to-63 character prefix is 48-to-252 "
                    f"bits of address rather than a full one. Cite "
                    f"{resolution.recomputed_digest or 'the full digest'}")
        if resolutions and not all(r.resolved for r in resolutions):
            # Redundant with the per-reference reasons above, and deliberately
            # so: this is the invariant publication actually depends on, stated
            # once where it cannot be lost by editing the loop.
            reasons.append(
                "not every evidence registry record was resolved AND verified by "
                "content; existence and a self-declared digest do not attest")
        hits = _placeholder_hits(cp.calibrated_by)
        if hits:
            reasons.append(
                f"calibration_provenance.calibrated_by contains placeholder "
                f"marker(s) {list(hits)}")

    # textual placeholder sniffing -- ids, sae ids, and every unit source
    for label, text in (("concept_id", entry.concept_id),
                        ("pairing_id", entry.pairing_id)):
        hits = _placeholder_hits(text)
        if hits:
            reasons.append(f"{label} contains placeholder marker(s) {list(hits)}")
    for sae_id in entry.sae_ids:
        hits = _placeholder_hits(sae_id)
        if hits:
            reasons.append(
                f"target sae_id {sae_id!r} contains placeholder marker(s) "
                f"{list(hits)}")
    for direction in Direction:
        record = entry.directions[direction]
        if record is None:
            continue
        for strength, spec in record.specs.items():
            if spec.unit_source is None:
                continue
            hits = _placeholder_hits(spec.unit_source)
            if hits:
                reasons.append(
                    f"{direction.value}/{strength.value} unit_source "
                    f"{spec.unit_source!r} contains placeholder marker(s) "
                    f"{list(hits)}")

    return ReleaseDecision(not reasons, entry.concept_id, entry.pairing_id,
                           tuple(reasons), resolutions)


def assert_publishable(
    entry: BundleEntry,
    *,
    evidence_registry: EvidenceRegistry = NO_EVIDENCE_REGISTRY,
) -> None:
    """Raising form of the gate. Call before any release build."""
    evaluate_publishability(entry, evidence_registry=evidence_registry).raise_if_blocked()


def filter_publishable(
    entries: tuple[BundleEntry, ...],
    *,
    evidence_registry: EvidenceRegistry = NO_EVIDENCE_REGISTRY,
) -> tuple[BundleEntry, ...]:
    """The subset that may ship. Empty is a correct answer."""
    return tuple(e for e in entries
                 if evaluate_publishability(e, evidence_registry=evidence_registry)
                 .publishable)


# ---------------------------------------------------------------------------
# DEVELOPMENT STUB EXPOSURE -- a different question from publishability
# ---------------------------------------------------------------------------
# The UI has to be built and demoed before any attested entry exists, so loudly
# marked fake entries must be able to populate it. That is NOT a relaxation of
# the gate: `evaluate_publishability` is unchanged and still blocks every one of
# them. The two questions are kept apart so neither can answer the other:
#
#   evaluate_publishability(entry)            -> may this ship to the public?
#   select_layout_entries(..., exposure=...)  -> what does this build render?
#
# A release build renders only publishable entries. A development build may
# render blocked stubs, but ONLY when explicitly configured to, and every one it
# renders is flagged so the UI can mark it on screen.

class Exposure(StrEnum):
    """Which entries a given build is permitted to render."""

    RELEASE = "release"
    DEVELOPMENT_STUBS = "development_stubs"


@dataclass(frozen=True, slots=True)
class LayoutEntry:
    """An entry selected for rendering, with its exposure basis recorded.

    `is_development_stub` exists so the UI cannot render a fake concept without
    being told it is fake. A caller that ignores the flag has to ignore it
    deliberately.
    """

    entry: BundleEntry
    is_development_stub: bool
    block_reasons: tuple[str, ...] = ()

    @property
    def requires_fake_data_banner(self) -> bool:
        return self.is_development_stub

    # -- direction availability, for disabling controls ------------------
    # Derived from the entry every time, never stored. A copy of this list on
    # the catalog object could drift from the data it describes, and then two
    # answers to "may the user press Suppress?" would exist -- which is the one
    # thing a shared state object is for preventing. A one-direction concept
    # stays IN the catalog; it is the control that is disabled, not the concept
    # that is withheld.

    @property
    def available_directions(self) -> tuple[Direction, ...]:
        """Directions the UI may offer for this entry."""
        return self.entry.calibrated_directions

    @property
    def unavailable_directions(self) -> tuple[Direction, ...]:
        """Directions the UI must render disabled. Reported explicitly rather
        than left as "whatever is not in the other list", so a renderer has
        something to attach a reason to."""
        return tuple(d for d in Direction if not self.entry.has_direction(d))

    def is_available(self, direction: Direction | str) -> bool:
        return self.entry.has_direction(direction)


def select_layout_entries(
    entries: tuple[BundleEntry, ...],
    *,
    exposure: Exposure = Exposure.RELEASE,
    evidence_registry: EvidenceRegistry = NO_EVIDENCE_REGISTRY,
) -> tuple[LayoutEntry, ...]:
    """Chooses what a build renders. Defaults to RELEASE, i.e. fail-closed.

    Under RELEASE, only publishable entries are returned, and a development
    build that forgets to opt in gets an empty list rather than silently
    shipping stubs.

    Under DEVELOPMENT_STUBS, blocked entries are returned too, each flagged
    `is_development_stub=True` and carrying the reasons it is blocked, so the UI
    can display them behind a fake-data banner.
    """
    if exposure is Exposure.RELEASE:
        return tuple(
            LayoutEntry(e, False) for e in entries
            if evaluate_publishability(e, evidence_registry=evidence_registry)
            .publishable)
    out: list[LayoutEntry] = []
    for e in entries:
        decision = evaluate_publishability(e, evidence_registry=evidence_registry)
        out.append(LayoutEntry(e, not decision.publishable, decision.reasons))
    return tuple(out)


def assert_release_build_clean(selection: tuple[LayoutEntry, ...]) -> None:
    """Fail-closed check for a release build.

    Raises if any selected entry is a development stub -- the failure mode being
    guarded is a development exposure flag left switched on in a release
    artefact.
    """
    stubs = [le for le in selection if le.is_development_stub]
    if stubs:
        raise ReleaseBuildError(
            f"release build would expose {len(stubs)} development stub(s): "
            f"{[(le.entry.concept_id, le.entry.pairing_id) for le in stubs]}. "
            f"Development exposure must not be enabled in a release build."
        )
