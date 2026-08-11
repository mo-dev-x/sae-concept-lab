"""Proves provenance.verify_provenance's extraction_class policy: the two
classes (HISTORICAL_SEED, CANONICAL_MIRROR) are verified by genuinely
different mechanisms, print the two exact ratified verdict strings, and
reject reclassification -- for any extraction, since every manifest here
is synthetic rather than the repository's real source_import.json.

Fully standalone: every git checkout used is built fresh in tmp_path via
`git init`, never the real qwen-sae-interp repository or the real
provenance manifest, so these tests never require either to exist. The
real source_import.json is exercised end to end only when an operator
explicitly runs `python -m provenance.verify_provenance
--qwen-sae-interp-checkout ...` by hand.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from provenance.verify_provenance import (
    ProvenanceError,
    assert_is_git_checkout,
    load_manifest,
    verify,
)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    return path


def _write_exact(path: Path, content: str) -> None:
    """write_text() with no explicit newline= translates every '\\n' to
    os.linesep on write (CRLF on Windows) -- silently changing the bytes
    a sha256 computed from the literal Python string would expect. Every
    write in this file that must later hash-match uses this instead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def _commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# HISTORICAL_SEED
# ---------------------------------------------------------------------------


def _historical_seed_manifest(*, commit, entries, roots) -> dict:
    return {
        "extractions": [
            {
                "extraction_id": "synthetic_seed",
                "extraction_class": "HISTORICAL_SEED",
                "historical_seed_commit": commit,
                "historical_seed_commit_short": commit[:7],
                "source_repository": {"identity": "synthetic"},
                "import_path_roots": roots,
                "imported_files": entries,
            }
        ]
    }


def test_historical_seed_clean_verdict_is_the_exact_ratified_string(tmp_path):
    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "pkg" / "module.py", "ORIGINAL CONTENT\n")
    commit = _commit_all(repo, "seed import")
    manifest = _historical_seed_manifest(
        commit=commit,
        entries=[{"source_path": "pkg/module.py", "dest_path": "pkg/module.py", "sha256": _sha256("ORIGINAL CONTENT\n")}],
        roots=["pkg"],
    )
    report = verify(repo_root=repo, checkout=repo, manifest=manifest)
    assert report["verdicts"] == [
        f"HISTORICAL_SEED {commit[:7]} import faithful at import commit; current bytes not checked"
    ]


def test_historical_seed_permits_current_bytes_to_evolve(tmp_path):
    """The defining property of this class: editing the file AFTER the
    seed commit must not affect the verdict at all -- current bytes are
    never read."""
    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "pkg" / "module.py", "ORIGINAL CONTENT\n")
    commit = _commit_all(repo, "seed import")
    manifest = _historical_seed_manifest(
        commit=commit,
        entries=[{"source_path": "pkg/module.py", "dest_path": "pkg/module.py", "sha256": _sha256("ORIGINAL CONTENT\n")}],
        roots=["pkg"],
    )
    _write_exact(repo / "pkg" / "module.py", "COMPLETELY DIFFERENT, EVOLVED CONTENT\n")
    report = verify(repo_root=repo, checkout=repo, manifest=manifest)
    assert report["verdicts"] == [
        f"HISTORICAL_SEED {commit[:7]} import faithful at import commit; current bytes not checked"
    ]
    assert report["modified_dest"] == []
    assert report["missing_dest"] == []


def test_historical_seed_new_product_native_file_is_not_flagged_unrecorded(tmp_path):
    """'New product-native files need no invented extraction class':
    adding a brand-new file under a HISTORICAL_SEED root must not be
    reported as unrecorded."""
    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "pkg" / "module.py", "ORIGINAL CONTENT\n")
    commit = _commit_all(repo, "seed import")
    manifest = _historical_seed_manifest(
        commit=commit,
        entries=[{"source_path": "pkg/module.py", "dest_path": "pkg/module.py", "sha256": _sha256("ORIGINAL CONTENT\n")}],
        roots=["pkg"],
    )
    _write_exact(repo / "pkg" / "brand_new_product_file.py", "# new work, never extracted from anywhere\n")
    report = verify(repo_root=repo, checkout=repo, manifest=manifest)
    assert report["unrecorded"] == []


def test_historical_seed_missing_at_import_commit_is_not_faithful(tmp_path):
    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "pkg" / "module.py", "ORIGINAL CONTENT\n")
    commit = _commit_all(repo, "seed import")
    manifest = _historical_seed_manifest(
        commit=commit,
        entries=[{"source_path": "pkg/module.py", "dest_path": "pkg/other_name.py", "sha256": _sha256("x")}],
        roots=["pkg"],
    )
    report = verify(repo_root=repo, checkout=repo, manifest=manifest)
    assert report["verdicts"] == [f"HISTORICAL_SEED {commit[:7]} import NOT faithful at import commit: 1 missing at import commit"]
    assert report["missing_dest"] == ["synthetic_seed:pkg/other_name.py"]


def test_historical_seed_wrong_recorded_hash_is_not_faithful(tmp_path):
    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "pkg" / "module.py", "ORIGINAL CONTENT\n")
    commit = _commit_all(repo, "seed import")
    manifest = _historical_seed_manifest(
        commit=commit,
        entries=[{"source_path": "pkg/module.py", "dest_path": "pkg/module.py", "sha256": _sha256("WRONG\n")}],
        roots=["pkg"],
    )
    report = verify(repo_root=repo, checkout=repo, manifest=manifest)
    assert "NOT faithful at import commit: 1 modified since import commit" in report["verdicts"][0]
    assert report["modified_dest"] == ["synthetic_seed:pkg/module.py"]


# ---------------------------------------------------------------------------
# CANONICAL_MIRROR
# ---------------------------------------------------------------------------

_FAKE_RUNNER_SOURCE = textwrap.dedent(
    """
    def verify_pack(pack, package="unused"):
        return list(pack.get("forced_failures", []))
    """
)


def _write_fake_conformance_pack(repo_root: Path, *, forced_failures=()) -> None:
    pack_dir = repo_root / "provenance" / "runtime_extractions" / "concept_bundle"
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "concept_bundle_conformance.py").write_text(_FAKE_RUNNER_SOURCE, encoding="utf-8")
    (pack_dir / "vectors.json").write_text(
        json.dumps({"vectors": [{"id": "v1"}, {"id": "v2"}], "forced_failures": list(forced_failures)}),
        encoding="utf-8",
    )


def _canonical_mirror_manifest(*, commit, frozen_pack_commit_short, entries, roots) -> dict:
    return {
        "extractions": [
            {
                "extraction_id": "synthetic_mirror",
                "extraction_class": "CANONICAL_MIRROR",
                "source_repository": {
                    "identity": "synthetic",
                    "checkout_commit": commit,
                    "frozen_pack_commit": "0000000",
                    "frozen_pack_commit_short": frozen_pack_commit_short,
                },
                "import_path_roots": roots,
                "imported_modules": entries,
            }
        ]
    }


def test_canonical_mirror_clean_verdict_is_the_exact_ratified_string(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")
    _write_exact(checkout / "src" / "module.py", "CANONICAL CONTENT\n")
    commit = _commit_all(checkout, "canonical commit")

    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "mirror" / "module.py", "CANONICAL CONTENT\n")
    _write_fake_conformance_pack(repo)

    manifest = _canonical_mirror_manifest(
        commit=commit,
        frozen_pack_commit_short="abc1234",
        entries=[
            {"source_path": "src/module.py", "dest_path": "mirror/module.py", "sha256": _sha256("CANONICAL CONTENT\n")}
        ],
        roots=["mirror"],
    )
    report = verify(repo_root=repo, checkout=checkout, manifest=manifest)
    assert report["verdicts"] == [
        "CANONICAL_MIRROR abc1234 current bytes match canonical source; conformance vectors pass"
    ]


def test_canonical_mirror_byte_mismatch_is_not_clean(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")
    _write_exact(checkout / "src" / "module.py", "CANONICAL CONTENT\n")
    commit = _commit_all(checkout, "canonical commit")

    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "mirror" / "module.py", "TAMPERED CONTENT\n")
    _write_fake_conformance_pack(repo)

    manifest = _canonical_mirror_manifest(
        commit=commit,
        frozen_pack_commit_short="abc1234",
        entries=[
            {"source_path": "src/module.py", "dest_path": "mirror/module.py", "sha256": _sha256("CANONICAL CONTENT\n")}
        ],
        roots=["mirror"],
    )
    report = verify(repo_root=repo, checkout=checkout, manifest=manifest)
    assert "NOT clean: 1 byte mismatch(es)" in report["verdicts"][0]
    assert report["modified_dest"] == ["synthetic_mirror:mirror/module.py"]


def test_canonical_mirror_new_file_under_its_root_is_flagged_unrecorded(tmp_path):
    """The opposite of HISTORICAL_SEED's evolution allowance: a
    CANONICAL_MIRROR root must never gain an unrecorded file."""
    checkout = _init_repo(tmp_path / "checkout")
    _write_exact(checkout / "src" / "module.py", "CANONICAL CONTENT\n")
    commit = _commit_all(checkout, "canonical commit")

    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "mirror" / "module.py", "CANONICAL CONTENT\n")
    _write_fake_conformance_pack(repo)
    _write_exact(repo / "mirror" / "stray.py", "# never recorded\n")

    manifest = _canonical_mirror_manifest(
        commit=commit,
        frozen_pack_commit_short="abc1234",
        entries=[
            {"source_path": "src/module.py", "dest_path": "mirror/module.py", "sha256": _sha256("CANONICAL CONTENT\n")}
        ],
        roots=["mirror"],
    )
    report = verify(repo_root=repo, checkout=checkout, manifest=manifest)
    assert report["unrecorded"] == ["mirror/stray.py"]


def test_canonical_mirror_conformance_vector_failure_is_not_clean(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")
    _write_exact(checkout / "src" / "module.py", "CANONICAL CONTENT\n")
    commit = _commit_all(checkout, "canonical commit")

    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "mirror" / "module.py", "CANONICAL CONTENT\n")
    _write_fake_conformance_pack(repo, forced_failures=["v1: deliberately broken for this test"])

    manifest = _canonical_mirror_manifest(
        commit=commit,
        frozen_pack_commit_short="abc1234",
        entries=[
            {"source_path": "src/module.py", "dest_path": "mirror/module.py", "sha256": _sha256("CANONICAL CONTENT\n")}
        ],
        roots=["mirror"],
    )
    report = verify(repo_root=repo, checkout=checkout, manifest=manifest)
    assert "NOT clean: 1 conformance vector failure(s)" in report["verdicts"][0]


def test_canonical_mirror_missing_conformance_pack_is_not_clean(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")
    _write_exact(checkout / "src" / "module.py", "CANONICAL CONTENT\n")
    commit = _commit_all(checkout, "canonical commit")

    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "mirror" / "module.py", "CANONICAL CONTENT\n")
    # deliberately no _write_fake_conformance_pack() call

    manifest = _canonical_mirror_manifest(
        commit=commit,
        frozen_pack_commit_short="abc1234",
        entries=[
            {"source_path": "src/module.py", "dest_path": "mirror/module.py", "sha256": _sha256("CANONICAL CONTENT\n")}
        ],
        roots=["mirror"],
    )
    report = verify(repo_root=repo, checkout=checkout, manifest=manifest)
    assert "conformance vectors could not be run" in report["verdicts"][0]


# ---------------------------------------------------------------------------
# Reclassification rejection
# ---------------------------------------------------------------------------


def test_reclassification_of_a_canonical_path_as_historical_seed_is_rejected(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")
    _write_exact(checkout / "README.md", "unrelated file, establishes a commit\n")
    commit = _commit_all(checkout, "unused for this test")

    repo = _init_repo(tmp_path / "product")
    inventory_dir = repo / "provenance" / "runtime_extractions" / "concept_bundle"
    inventory_dir.mkdir(parents=True)
    (inventory_dir / "export_inventory.json").write_text(
        json.dumps({
            "minimum_export_surface": [
                {"path": "interplab/concept_bundle/schema.py", "sha256": "x", "bytes": 1},
            ]
        }),
        encoding="utf-8",
    )

    manifest = _historical_seed_manifest(
        commit=commit,
        entries=[
            {
                "source_path": "interplab/concept_bundle/schema.py",  # a CANONICAL_MIRROR path
                "dest_path": "sae_concept_lab/schema.py",
                "sha256": "y",
            }
        ],
        roots=["sae_concept_lab"],
    )
    with pytest.raises(ProvenanceError, match="reclassification rejected"):
        verify(repo_root=repo, checkout=checkout, manifest=manifest)


def test_reclassification_guard_is_silent_when_no_canonical_inventory_is_present(tmp_path):
    """No canonical pack copied yet (e.g. a repo with only a
    HISTORICAL_SEED extraction) -- nothing to cross-check against, so
    this must not raise."""
    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "pkg" / "module.py", "X\n")
    commit = _commit_all(repo, "seed")

    manifest = _historical_seed_manifest(
        commit=commit,
        entries=[{"source_path": "pkg/module.py", "dest_path": "pkg/module.py", "sha256": _sha256("X\n")}],
        roots=["pkg"],
    )
    report = verify(repo_root=repo, checkout=repo, manifest=manifest)
    assert report["verdicts"][0].startswith("HISTORICAL_SEED")


# ---------------------------------------------------------------------------
# Unknown extraction_class / setup errors
# ---------------------------------------------------------------------------


def test_unknown_extraction_class_raises(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")
    repo = _init_repo(tmp_path / "product")
    manifest = {
        "extractions": [
            {
                "extraction_id": "bad",
                "extraction_class": "SOMETHING_ELSE",
                "imported_files": [],
                "import_path_roots": [],
            }
        ]
    }
    with pytest.raises(ProvenanceError, match="no recognized extraction_class"):
        verify(repo_root=repo, checkout=checkout, manifest=manifest)


def test_assert_is_git_checkout_rejects_a_non_git_directory(tmp_path):
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    with pytest.raises(ProvenanceError):
        assert_is_git_checkout(not_a_repo)


def test_load_manifest_missing_file_raises():
    with pytest.raises(ProvenanceError):
        load_manifest(Path("does/not/exist.json"))


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


def test_main_cli_reports_failure_on_a_missing_destination_file(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")  # unused by HISTORICAL_SEED verification, only by --qwen-sae-interp-checkout validation

    repo = _init_repo(tmp_path / "product")
    _write_exact(repo / "README.md", "unrelated file, establishes a commit\n")
    commit = _commit_all(repo, "seed")
    # deliberately never write pkg/module.py in `repo` -- missing at the seed commit

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _historical_seed_manifest(
                commit=commit,
                entries=[
                    {"source_path": "pkg/module.py", "dest_path": "pkg/module.py", "sha256": _sha256("ORIGINAL CONTENT\n")}
                ],
                roots=["pkg"],
            )
        ),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "provenance" / "verify_provenance.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--qwen-sae-interp-checkout",
            str(checkout),
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "MISSING: synthetic_seed:pkg/module.py" in result.stderr
    assert f"verdict: HISTORICAL_SEED {commit[:7]} import NOT faithful at import commit" in result.stdout
