"""Whether a real backend's underlying intervention mechanism has been
mechanically verified against real model/SAE weights -- a fact entirely
separate from, and checked independently of, whether its CODE was extracted
cleanly (sae_concept_lab.extracted_runtime, provenance/source_import.json's
RUNTIME_CODE_MIRROR extractions).

BOTH pairings are now ACTIVATED, as of qwen-sae-interp evidence commit
`b6d598b` ("Import and adjudicate sealed final-pairing evidence: job 407008
(Gemma pass) and job 406092 (mixed: Gemma failed, Qwen mechanical pass)").
This is the third evidence claim made across this product's dispatch
history, and the first two were REJECTED on exactly the grounds this
module's own `import_acceptance_from_evidence_commit()` enforces -- see
BOUNDARY.md for the full account of why a loose artifact file (job 406092,
attempt 1) and an unrelated commit misattributed as "job 407008" (attempt
2) were both refused. This third claim was accepted only after: (1) git
show confirmed commit b6d598b exists and its message matches the claim;
(2) both jobs' `chain_of_custody.json` manifests and every named artifact
were independently re-hashed against the committed tree via
`import_acceptance_from_evidence_commit`, which raises rather than
continues on any mismatch; (3) qwen-sae-interp's OWN
`tests/test_final_pairing_evidence_record.py` (29 tests) was read and run
for real against that checkout and passed, including cross-contamination
guards proving the Qwen-pass claim and the Gemma-pass claim cannot leak
into each other's job. See `BOUNDARY.md`'s "Runtime acceptance: three
evidence claims, two rejected" section for the full account.

Qwen's acceptance is SCOPED to job 406092's two Qwen scenarios only --
that job also ran two Gemma scenarios, and Gemma FAILED both (a
loader-identity defect fixed in a later commit, 8005679); job 406092 was
explicitly NOT a global acceptance pass, and this module's Qwen record
says so in its own `claim` text. Gemma's acceptance comes from a wholly
separate job, 407008, sealed and adjudicated independently. Neither record
implies anything about the other pairing, about ATTESTED scientific
content, or about calibration -- see MECHANICAL_ACCEPTANCE_IS_NOT.

`enforce_release_gate` (fixtures/loader.py) and app.py's backend
construction both consult `is_mechanically_accepted()` before a real
backend (QwenRuntimeBackend/GemmaRuntimeBackend) may run in release mode --
independent of, and in addition to, the existing StubConceptLabBackend
type check and the canonical ATTESTED-provenance/content-verified-evidence
gate. Both pairings now pass this specific check; release mode still
refuses because this repository ships no ATTESTED concept entries and no
evidence registry content -- mechanical acceptance of the INTERVENTION
MECHANISM and public release of a SCIENTIFIC CONCEPT remain, and must
remain, two separate gates. Dev mode may now run either real backend
without the unverified-mechanism tag (core/qwen_backend.py /
core/gemma_backend.py's `_tag()`).

TO RE-VERIFY OR RE-IMPORT (e.g. if either evidence commit is ever amended
or superseded): call `import_acceptance_from_evidence_commit()` again with
the new commit and artifact hashes -- it performs the full bounded
adjudication independently each time, raising rather than silently
continuing on any mismatch -- and assign the result into
`ACCEPTANCE_REGISTRY` below.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Every scientific/behavioral claim a RuntimeAcceptanceRecord explicitly
#: does NOT make, regardless of pairing. Attached to every record so a
#: caller rendering it cannot present it as more than what job 406092 (or
#: any future evidence commit) actually established.
MECHANICAL_ACCEPTANCE_IS_NOT = (
    "scientific concept validity",
    "feature calibration",
    "behavioral quality or safety",
    "acceptance of the complete job the scenarios were part of",
    "acceptance of any OTHER pairing's scenarios",
)


class RuntimeAcceptanceError(RuntimeError):
    """Raised when a real backend's pairing has no attached, verified
    RuntimeAcceptanceRecord -- by enforce_release_gate/app.py in release
    mode, and (as a loud, non-raising on-screen notice, not an exception)
    surfaced by the backend itself in dev mode."""


@dataclass(frozen=True, slots=True)
class RuntimeAcceptanceRecord:
    """The bounded, mechanically-verified claim: these specific scenarios,
    from this specific job, at this specific tracked evidence commit,
    mechanically passed. Nothing wider is asserted -- see
    MECHANICAL_ACCEPTANCE_IS_NOT."""

    pairing: str
    job_id: str
    evidence_commit: str
    #: The layer the acceptance run actually used. Acceptance is scoped to it:
    #: the mechanism was exercised in ONE layer's dictionary, and a feature
    #: index means nothing outside the dictionary it was found in. A record
    #: that did not carry this let `is_mechanically_accepted` keep returning
    #: True after the pinned target moved to a different layer -- a true
    #: statement about layer 31 silently re-read as a claim about layer 29.
    scenarios_passed: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    claim: str
    imported_at_utc: str
    #: The layer the acceptance run used; None means unscoped. Scoped records
    #: make is_mechanically_accepted(pairing, layer) False on a different layer.
    accepted_layer: int | None = None

    def __post_init__(self) -> None:
        if not self.scenarios_passed:
            raise ValueError("a RuntimeAcceptanceRecord must name at least one passed scenario")
        if not self.artifact_hashes:
            raise ValueError("a RuntimeAcceptanceRecord must cite at least one verified artifact")

    def render_notice(self) -> str:
        lines = [
            f"MECHANICAL ACCEPTANCE ({self.pairing}): {self.claim}",
            f"  job: {self.job_id}",
            f"  evidence commit: {self.evidence_commit}",
            f"  scenarios passed: {', '.join(self.scenarios_passed)}",
        ]
        for path, digest in self.artifact_hashes:
            lines.append(f"  verified artifact: {path} sha256:{digest}")
        lines.append("  this is NOT: " + "; ".join(MECHANICAL_ACCEPTANCE_IS_NOT))
        return "\n".join(lines)


#: qwen-sae-interp evidence commit b6d598b, verified via
#: import_acceptance_from_evidence_commit() against a live checkout (every
#: artifact hash below independently re-read and re-hashed from that
#: commit, not copied from any dispatch's claim) before being assigned
#: here. A module-level dict, not a function default, so every caller
#: (release gate, backend constructors, tests) shares the exact same state.
_EVIDENCE_COMMIT = "b6d598b5dca8c47861aa77aeefee1f75b2832133"

ACCEPTANCE_REGISTRY: dict[str, RuntimeAcceptanceRecord | None] = {
    "qwen": RuntimeAcceptanceRecord(
        pairing="qwen",
        job_id="406092",
        accepted_layer=0,
        evidence_commit=_EVIDENCE_COMMIT,
        scenarios_passed=("all", "generated_only"),
        artifact_hashes=(
            (
                "results/final_pairing/job_406092/qwen_3_5_27b_mechanical_all.json",
                "bfb0d5bcc7fd7a8ff334121ba4f8be78e9a099ae7ef22177569f76f6e6c7dce4",
            ),
            (
                "results/final_pairing/job_406092/qwen_3_5_27b_mechanical_generated_only.json",
                "d23a5f60ca4a6103e8bdebe37a3dd65bd0a44953e91ca0cb86f3a7aa0b0a609f",
            ),
            (
                "results/final_pairing/job_406092/chain_of_custody.json",
                "fbbfbaf0f8ee48a789f7217c87461f1752bb46657e5087a81f74108a90309f16",
            ),
            (
                "results/final_pairing/job_406092/inventory.json",
                "4e2dd274fee36bd60283207b771ed18fcf082c66ff4233839b49fe8bedcc688c",
            ),
        ),
        claim=(
            "Qwen3.5-27B with Qwen-Scope engineering layer 0 passed mechanical steering under ALL "
            "and GENERATED_ONLY in mixed job 406092. Job 406092 was NOT a global acceptance pass -- "
            "Gemma failed in this job (loader-identity defect, fixed in 8005679); only the Qwen "
            "statement is supported. Layer 0 and feature 4096 are ENGINEERING-ONLY."
        ),
        imported_at_utc="2026-08-13T09:00:00Z",
    ),
    "gemma": RuntimeAcceptanceRecord(
        pairing="gemma",
        job_id="407008",
        accepted_layer=31,
        evidence_commit=_EVIDENCE_COMMIT,
        scenarios_passed=("all", "generated_only"),
        artifact_hashes=(
            (
                "results/final_pairing/job_407008/gemma_3_12b_it_all.json",
                "c566dd33c38040df4eef332c3dab0e98bd3f266232f043229aaff1c6ccd9d1fe",
            ),
            (
                "results/final_pairing/job_407008/gemma_3_12b_it_generated_only.json",
                "18421cd1cc8b22a7e83eb3fe383bfcd087a9fbf103efaf7c563c3c5e9fa89b69",
            ),
            (
                "results/final_pairing/job_407008/job_result.json",
                "792c73067c0598972e4529923cfe62182b82bc55714768a24a44b71c673c06a3",
            ),
            (
                "results/final_pairing/job_407008/symlink_preflight_result.json",
                "eba4cd1125f38ebadd5518d3d03ff06d5614ca2f01cd1cb99674ea202d983cc3",
            ),
            (
                "results/final_pairing/job_407008/chain_of_custody.json",
                "10cbbb6e92b5fc5b7ec4a48974e3940c9c8495d71b1e5e5a0cf38ceb6b88984c",
            ),
        ),
        claim=(
            "Gemma-3-12B-IT with Gemma Scope 2 resid_post layer 31 passed mechanical steering "
            "acceptance under ALL and GENERATED_ONLY in job 407008. Feature 250 and raw clamp 5000 "
            "are engineering acceptance inputs only, not a public concept."
        ),
        imported_at_utc="2026-08-13T09:00:00Z",
    ),
}


def is_mechanically_accepted(pairing: str, layer: int | None = None) -> bool:
    """True only if a record exists AND it covers `layer`.

    `layer=None` preserves the original "does a record exist" question for
    callers that genuinely have no layer in hand. Callers that DO know which
    layer they loaded must pass it: acceptance was established in one layer's
    dictionary, and reporting it for another is the confident-wrong-answer
    failure this repository refuses everywhere else."""
    record = ACCEPTANCE_REGISTRY.get(pairing)
    if record is None:
        return False
    if layer is None or record.accepted_layer is None:
        return True
    return int(layer) == int(record.accepted_layer)


def accepted_layer_for(pairing: str) -> int | None:
    record = ACCEPTANCE_REGISTRY.get(pairing)
    return None if record is None else record.accepted_layer


def _run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True)


def _read_blob_at_commit(repo: Path, commit: str, path: str) -> bytes | None:
    exists = _run_git(repo, ["cat-file", "-e", f"{commit}:{path}"])
    if exists.returncode != 0:
        return None
    shown = _run_git(repo, ["show", f"{commit}:{path}"])
    if shown.returncode != 0:
        return None
    return shown.stdout


def import_acceptance_from_evidence_commit(
    *,
    pairing: str,
    qwen_sae_interp_checkout: Path | str,
    evidence_commit: str,
    job_id: str,
    scenarios_passed: tuple[str, ...],
    artifact_relative_paths_and_expected_hashes: tuple[tuple[str, str], ...],
    claim: str,
    imported_at_utc: str,
) -> RuntimeAcceptanceRecord:
    """The bounded adjudication step. Reads each named artifact via
    `git show <evidence_commit>:<path>` from a REAL qwen-sae-interp
    checkout (read-only -- never touches that checkout's working tree,
    index, or HEAD), independently recomputes its SHA-256, and raises
    RuntimeAcceptanceError (never silently continues) if the commit is
    unreachable, an artifact is missing, or any hash disagrees with what
    the caller claims it to be. Returns a RuntimeAcceptanceRecord only when
    every artifact verifies -- the caller is then responsible for the one
    remaining step, assigning it into ACCEPTANCE_REGISTRY.

    This function does not, and must never, read anything from a scratch
    or cache directory outside qwen_sae_interp_checkout's own git object
    database -- that is precisely the gap this project's provenance
    discipline exists to close (see BOUNDARY.md's account of the two
    dispatches this rejected for citing loose, untracked files instead)."""
    repo = Path(qwen_sae_interp_checkout)
    if pairing not in ACCEPTANCE_REGISTRY:
        raise RuntimeAcceptanceError(f"unknown pairing {pairing!r}; expected one of {sorted(ACCEPTANCE_REGISTRY)}")

    commit_check = _run_git(repo, ["cat-file", "-e", f"{evidence_commit}^{{commit}}"])
    if commit_check.returncode != 0:
        raise RuntimeAcceptanceError(
            f"evidence commit {evidence_commit!r} was not found in {repo} -- refusing to import an "
            "acceptance record against a commit this checkout does not have"
        )

    verified: list[tuple[str, str]] = []
    for rel_path, expected_hash in artifact_relative_paths_and_expected_hashes:
        blob = _read_blob_at_commit(repo, evidence_commit, rel_path)
        if blob is None:
            raise RuntimeAcceptanceError(
                f"artifact {rel_path!r} does not exist at {evidence_commit} in {repo} -- refusing to "
                "import an acceptance record citing an artifact that is not actually tracked there"
            )
        actual_hash = hashlib.sha256(blob).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeAcceptanceError(
                f"artifact {rel_path!r} at {evidence_commit} hashes to {actual_hash}, not the claimed "
                f"{expected_hash} -- refusing to import a mismatched acceptance record"
            )
        verified.append((rel_path, actual_hash))

    return RuntimeAcceptanceRecord(
        pairing=pairing,
        job_id=job_id,
        evidence_commit=evidence_commit,
        scenarios_passed=scenarios_passed,
        artifact_hashes=tuple(verified),
        claim=claim,
        imported_at_utc=imported_at_utc,
    )
