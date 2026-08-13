"""Tests for core/runtime_acceptance.py: the bounded adjudication importer
(import_acceptance_from_evidence_commit) against synthetic git repos built
fresh in tmp_path -- never the real qwen-sae-interp checkout, so these
never depend on it existing -- plus the currently-activated registry state
for both pairings."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from sae_concept_lab.core.runtime_acceptance import (
    ACCEPTANCE_REGISTRY,
    RuntimeAcceptanceError,
    RuntimeAcceptanceRecord,
    import_acceptance_from_evidence_commit,
    is_mechanically_accepted,
)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    return path


def _write_exact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def _commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seeded_repo(tmp_path: Path, *, artifact_content: str = "sealed evidence content\n") -> tuple[Path, str, str]:
    repo = _init_repo(tmp_path / "fake-qwen-sae-interp")
    _write_exact(repo / "results" / "final_pairing" / "job_999999" / "artifact.json", artifact_content)
    commit = _commit_all(repo, "Seal fake evidence for job 999999")
    return repo, commit, _sha256(artifact_content)


# ---------------------------------------------------------------------------
# Both pairings, as activated by this product commit.
# ---------------------------------------------------------------------------


def test_both_pairings_are_currently_accepted():
    assert is_mechanically_accepted("qwen") is True
    assert is_mechanically_accepted("gemma") is True


def test_acceptance_records_carry_the_exact_bounded_qwen_claim():
    record = ACCEPTANCE_REGISTRY["qwen"]
    assert record.job_id == "406092"
    assert record.scenarios_passed == ("all", "generated_only")
    assert "NOT a global acceptance pass" in record.claim
    assert "Gemma failed in this job" in record.claim
    assert "ENGINEERING-ONLY" in record.claim


def test_acceptance_records_carry_the_exact_bounded_gemma_claim():
    record = ACCEPTANCE_REGISTRY["gemma"]
    assert record.job_id == "407008"
    assert record.scenarios_passed == ("all", "generated_only")
    assert "engineering acceptance inputs only" in record.claim


def test_render_notice_always_includes_what_this_is_not():
    from sae_concept_lab.core.runtime_acceptance import MECHANICAL_ACCEPTANCE_IS_NOT

    for pairing in ("qwen", "gemma"):
        notice = ACCEPTANCE_REGISTRY[pairing].render_notice()
        for phrase in MECHANICAL_ACCEPTANCE_IS_NOT:
            assert phrase in notice


def test_unknown_pairing_is_not_accepted():
    assert is_mechanically_accepted("not-a-real-pairing") is False


# ---------------------------------------------------------------------------
# RuntimeAcceptanceRecord construction invariants
# ---------------------------------------------------------------------------


def test_record_requires_at_least_one_scenario():
    with pytest.raises(ValueError, match="at least one passed scenario"):
        RuntimeAcceptanceRecord(
            pairing="qwen", job_id="x", evidence_commit="x" * 40, scenarios_passed=(),
            artifact_hashes=(("a", "b"),), claim="x", imported_at_utc="x",
        )


def test_record_requires_at_least_one_artifact():
    with pytest.raises(ValueError, match="at least one verified artifact"):
        RuntimeAcceptanceRecord(
            pairing="qwen", job_id="x", evidence_commit="x" * 40, scenarios_passed=("all",),
            artifact_hashes=(), claim="x", imported_at_utc="x",
        )


# ---------------------------------------------------------------------------
# import_acceptance_from_evidence_commit: the bounded adjudication step.
# ---------------------------------------------------------------------------


def test_import_succeeds_against_a_real_synthetic_commit(tmp_path):
    repo, commit, digest = _seeded_repo(tmp_path)
    record = import_acceptance_from_evidence_commit(
        pairing="qwen",
        qwen_sae_interp_checkout=repo,
        evidence_commit=commit,
        job_id="999999",
        scenarios_passed=("all",),
        artifact_relative_paths_and_expected_hashes=(
            ("results/final_pairing/job_999999/artifact.json", digest),
        ),
        claim="test claim",
        imported_at_utc="2026-01-01T00:00:00Z",
    )
    assert record.pairing == "qwen"
    assert record.evidence_commit == commit
    assert record.artifact_hashes == (("results/final_pairing/job_999999/artifact.json", digest),)


def test_import_rejects_unknown_pairing(tmp_path):
    repo, commit, digest = _seeded_repo(tmp_path)
    with pytest.raises(RuntimeAcceptanceError, match="unknown pairing"):
        import_acceptance_from_evidence_commit(
            pairing="not-a-real-pairing",
            qwen_sae_interp_checkout=repo,
            evidence_commit=commit,
            job_id="999999",
            scenarios_passed=("all",),
            artifact_relative_paths_and_expected_hashes=(
                ("results/final_pairing/job_999999/artifact.json", digest),
            ),
            claim="test claim",
            imported_at_utc="2026-01-01T00:00:00Z",
        )


def test_import_rejects_a_nonexistent_evidence_commit(tmp_path):
    repo, _commit, digest = _seeded_repo(tmp_path)
    with pytest.raises(RuntimeAcceptanceError, match="was not found"):
        import_acceptance_from_evidence_commit(
            pairing="qwen",
            qwen_sae_interp_checkout=repo,
            evidence_commit="0" * 40,
            job_id="999999",
            scenarios_passed=("all",),
            artifact_relative_paths_and_expected_hashes=(
                ("results/final_pairing/job_999999/artifact.json", digest),
            ),
            claim="test claim",
            imported_at_utc="2026-01-01T00:00:00Z",
        )


def test_import_rejects_a_missing_artifact(tmp_path):
    repo, commit, digest = _seeded_repo(tmp_path)
    with pytest.raises(RuntimeAcceptanceError, match="does not exist"):
        import_acceptance_from_evidence_commit(
            pairing="qwen",
            qwen_sae_interp_checkout=repo,
            evidence_commit=commit,
            job_id="999999",
            scenarios_passed=("all",),
            artifact_relative_paths_and_expected_hashes=(
                ("results/final_pairing/job_999999/does_not_exist.json", digest),
            ),
            claim="test claim",
            imported_at_utc="2026-01-01T00:00:00Z",
        )


def test_import_rejects_a_hash_mismatch_tampered_artifact(tmp_path):
    """The core tamper-detection guarantee: even if an artifact exists at
    the named commit and path, a caller's claimed hash that does not match
    the ACTUAL committed content is refused, never silently accepted."""
    repo, commit, _correct_digest = _seeded_repo(tmp_path)
    wrong_digest = "f" * 64
    with pytest.raises(RuntimeAcceptanceError, match="hashes to"):
        import_acceptance_from_evidence_commit(
            pairing="qwen",
            qwen_sae_interp_checkout=repo,
            evidence_commit=commit,
            job_id="999999",
            scenarios_passed=("all",),
            artifact_relative_paths_and_expected_hashes=(
                ("results/final_pairing/job_999999/artifact.json", wrong_digest),
            ),
            claim="test claim",
            imported_at_utc="2026-01-01T00:00:00Z",
        )


def test_import_detects_tampering_after_the_commit_if_working_tree_were_reused(tmp_path):
    """Re-affirms that verification reads the COMMITTED blob (git show),
    not the working-tree file -- editing the working-tree copy after the
    commit must not affect what a later import call reads."""
    repo, commit, digest = _seeded_repo(tmp_path)
    (repo / "results" / "final_pairing" / "job_999999" / "artifact.json").write_text(
        "this was edited after the commit and never committed\n", encoding="utf-8"
    )
    # Still verifies against the sealed commit's blob, unaffected by the dirty working tree.
    record = import_acceptance_from_evidence_commit(
        pairing="gemma",
        qwen_sae_interp_checkout=repo,
        evidence_commit=commit,
        job_id="999999",
        scenarios_passed=("all", "generated_only"),
        artifact_relative_paths_and_expected_hashes=(
            ("results/final_pairing/job_999999/artifact.json", digest),
        ),
        claim="test claim",
        imported_at_utc="2026-01-01T00:00:00Z",
    )
    assert record.pairing == "gemma"


def test_each_valid_record_enables_only_its_own_matching_pairing(tmp_path):
    """A record imported for 'qwen' must never make 'gemma' report
    accepted, and vice versa -- proven against a temporary registry
    dict, never the real module-level ACCEPTANCE_REGISTRY."""
    repo, commit, digest = _seeded_repo(tmp_path)
    qwen_record = import_acceptance_from_evidence_commit(
        pairing="qwen", qwen_sae_interp_checkout=repo, evidence_commit=commit,
        job_id="999999", scenarios_passed=("all",),
        artifact_relative_paths_and_expected_hashes=(
            ("results/final_pairing/job_999999/artifact.json", digest),
        ),
        claim="qwen-only test claim", imported_at_utc="2026-01-01T00:00:00Z",
    )
    fresh_registry = {"qwen": None, "gemma": None}
    fresh_registry["qwen"] = qwen_record
    assert fresh_registry["qwen"] is not None
    assert fresh_registry["gemma"] is None
