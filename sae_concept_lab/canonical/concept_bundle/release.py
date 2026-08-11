"""Fail-closed publication gate.

An entry is publishable only by positively satisfying every condition below. It
is never publishable merely because no check happened to trip -- which is why
the schema default for `provenance` is UNKNOWN and why the default evidence
registry resolves nothing. An entry authored by someone who never thought about
publication is blocked by construction.

TWO CONDITIONS, BOTH POSITIVE:

  1. `provenance` is exactly ATTESTED. CANDIDATE, DRAFT, FAKE and UNKNOWN are
     all inspectable in development and none of them publishes. There is no
     "attested enough".

  2. Every evidence reference RESOLVES against the repository registry.
     Missing, malformed, mismatched and unresolvable are four distinct failures
     and all four block. An attestation whose evidence cannot be found is a
     claim about a document nobody can read.

The gate also sniffs ids for placeholder markers, because the likeliest
publication accident is not a mis-set enum -- it is real-looking metadata
attached to fixture data somebody forgot to replace. That check is what keeps
fixture quarantine a property of the data rather than of a flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import ReleaseBlockedError, ReleaseBuildError
from .evidence import (
    NO_EVIDENCE_REGISTRY,
    EvidenceRegistry,
    EvidenceResolution,
    resolve_all,
)
from .schema import PLACEHOLDER_MARKERS, BundleEntry, Direction, Provenance

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
