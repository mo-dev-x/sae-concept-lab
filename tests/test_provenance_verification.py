"""Proves provenance.verify_provenance detects modification, deletion, and
unrecorded extraction files -- for the concept-bundle extraction and, by
construction, for any extraction, since the verifier's `verify()` takes a
synthetic manifest here rather than the repository's real
source_import.json.

Fully standalone: every git checkout used is built fresh in tmp_path via
`git init`, never the real qwen-sae-interp repository, so these tests
never require that checkout to exist or be reachable. The real
source_import.json is exercised end to end (against the real
qwen-sae-interp checkout) only when an operator explicitly runs
`python -m provenance.verify_provenance --qwen-sae-interp-checkout ...`
by hand -- that is an integration check, not something the standalone
suite can assume a path for.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
    path.write_text(content, encoding="utf-8", newline="")


def _write_and_commit(repo: Path, relative_path: str, content: str) -> str:
    """Writes, commits, and returns the resulting HEAD commit hash."""
    target = repo / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_exact(target, content)
    subprocess.run(["git", "add", relative_path], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"add {relative_path}"], cwd=repo, check=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _one_extraction_manifest(*, extraction_id, commit, entries, roots, scaffolding=None) -> dict:
    return {
        "extractions": [
            {
                "extraction_id": extraction_id,
                "source_repository": {"checkout_commit": commit},
                "import_path_roots": roots,
                "imported_files": entries,
                "scaffolding_added": scaffolding or [],
            }
        ]
    }


@pytest.fixture
def synthetic_checkout(tmp_path):
    checkout = _init_repo(tmp_path / "checkout")
    commit = _write_and_commit(checkout, "pkg/module.py", "ORIGINAL CONTENT\n")
    return checkout, commit


@pytest.fixture
def product_repo_with_module(tmp_path, synthetic_checkout):
    _checkout, commit = synthetic_checkout
    repo_root = tmp_path / "product"
    (repo_root / "pkg").mkdir(parents=True)
    (repo_root / "pkg" / "module.py").write_text("ORIGINAL CONTENT\n", encoding="utf-8", newline="")
    manifest = _one_extraction_manifest(
        extraction_id="synthetic_extraction",
        commit=commit,
        entries=[
            {
                "source_path": "pkg/module.py",
                "dest_path": "pkg/module.py",
                "sha256": _sha256("ORIGINAL CONTENT\n"),
            }
        ],
        roots=["pkg"],
    )
    return repo_root, manifest


def test_clean_extraction_verifies_with_zero_findings(product_repo_with_module, synthetic_checkout):
    repo_root, manifest = product_repo_with_module
    checkout, _commit = synthetic_checkout
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["verified_count"] == 1
    assert report["missing_source"] == []
    assert report["modified_source"] == []
    assert report["missing_dest"] == []
    assert report["modified_dest"] == []
    assert report["unrecorded"] == []


def test_modified_destination_file_is_detected(product_repo_with_module, synthetic_checkout):
    repo_root, manifest = product_repo_with_module
    checkout, _commit = synthetic_checkout
    (repo_root / "pkg" / "module.py").write_text("TAMPERED CONTENT\n", encoding="utf-8", newline="")
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["modified_dest"] == ["synthetic_extraction:pkg/module.py"]
    assert report["verified_count"] == 0


def test_deleted_destination_file_is_detected(product_repo_with_module, synthetic_checkout):
    repo_root, manifest = product_repo_with_module
    checkout, _commit = synthetic_checkout
    (repo_root / "pkg" / "module.py").unlink()
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["missing_dest"] == ["synthetic_extraction:pkg/module.py"]
    assert report["verified_count"] == 0


def test_unrecorded_file_under_an_import_root_is_detected(product_repo_with_module, synthetic_checkout):
    repo_root, manifest = product_repo_with_module
    checkout, _commit = synthetic_checkout
    (repo_root / "pkg" / "stray.py").write_text("# never recorded\n", encoding="utf-8", newline="")
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["unrecorded"] == ["pkg/stray.py"]
    # the recorded, unmodified file is still reported clean independently
    assert report["verified_count"] == 1


def test_manifest_recording_a_wrong_hash_is_detected_as_modified_source(
    product_repo_with_module, synthetic_checkout
):
    """Simulates a stale/incorrect provenance record: the commit is real
    and the destination file matches it, but the manifest's own sha256
    disagrees with what the source commit actually contains."""
    repo_root, manifest = product_repo_with_module
    checkout, _commit = synthetic_checkout
    manifest["extractions"][0]["imported_files"][0]["sha256"] = _sha256("SOMETHING ELSE\n")
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["modified_source"] == ["synthetic_extraction:pkg/module.py"]


def test_source_path_absent_at_the_recorded_commit_is_detected_as_missing_source(
    product_repo_with_module, synthetic_checkout
):
    repo_root, manifest = product_repo_with_module
    checkout, _commit = synthetic_checkout
    manifest["extractions"][0]["imported_files"][0]["source_path"] = "pkg/does_not_exist.py"
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["missing_source"] == ["synthetic_extraction:pkg/does_not_exist.py"]


def test_missing_declared_scaffolding_is_detected(tmp_path, synthetic_checkout):
    checkout, commit = synthetic_checkout
    repo_root = tmp_path / "product"
    (repo_root / "pkg").mkdir(parents=True)
    manifest = _one_extraction_manifest(
        extraction_id="synthetic_extraction",
        commit=commit,
        entries=[],
        roots=["pkg"],
        scaffolding=[{"path": "pkg/__init__.py"}],
    )
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["missing_scaffolding"] == ["synthetic_extraction:pkg/__init__.py"]


def test_declared_scaffolding_present_is_not_flagged_unrecorded(tmp_path, synthetic_checkout):
    checkout, commit = synthetic_checkout
    repo_root = tmp_path / "product"
    (repo_root / "pkg").mkdir(parents=True)
    (repo_root / "pkg" / "__init__.py").write_text("", encoding="utf-8", newline="")
    manifest = _one_extraction_manifest(
        extraction_id="synthetic_extraction",
        commit=commit,
        entries=[],
        roots=["pkg"],
        scaffolding=[{"path": "pkg/__init__.py"}],
    )
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["missing_scaffolding"] == []
    assert report["unrecorded"] == []


def test_nested_extraction_roots_do_not_false_positive_each_other(tmp_path):
    """The exact scenario this repository actually has: an outer
    extraction's root (sae_concept_lab-equivalent) contains an inner
    extraction's root (canonical/concept_bundle-equivalent). Scanning
    globally (as verify() does) must not report the inner extraction's
    own recorded files as unrecorded from the outer extraction's point of
    view."""
    checkout = _init_repo(tmp_path / "checkout")
    outer_commit = _write_and_commit(checkout, "outer/a.py", "OUTER\n")
    inner_commit = _write_and_commit(checkout, "outer/inner/b.py", "INNER\n")

    repo_root = tmp_path / "product"
    (repo_root / "outer" / "inner").mkdir(parents=True)
    (repo_root / "outer" / "a.py").write_text("OUTER\n", encoding="utf-8", newline="")
    (repo_root / "outer" / "inner" / "b.py").write_text("INNER\n", encoding="utf-8", newline="")

    manifest = {
        "extractions": [
            {
                "extraction_id": "outer",
                "source_repository": {"checkout_commit": outer_commit},
                "import_path_roots": ["outer"],
                "imported_files": [
                    {"source_path": "outer/a.py", "dest_path": "outer/a.py", "sha256": _sha256("OUTER\n")}
                ],
            },
            {
                "extraction_id": "inner",
                "source_repository": {"checkout_commit": inner_commit},
                "import_path_roots": ["outer/inner"],
                "imported_files": [
                    {
                        "source_path": "outer/inner/b.py",
                        "dest_path": "outer/inner/b.py",
                        "sha256": _sha256("INNER\n"),
                    }
                ],
            },
        ]
    }
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    assert report["unrecorded"] == []
    assert report["verified_count"] == 2


def test_glob_style_import_path_root_is_supported(tmp_path, synthetic_checkout):
    """import_path_roots entries containing '*' are matched as a glob
    relative to repo_root, mirroring the real manifest's
    'tests/test_sae_concept_lab_*.py' root."""
    checkout, commit = synthetic_checkout
    repo_root = tmp_path / "product"
    (repo_root / "tests").mkdir(parents=True)
    (repo_root / "tests" / "test_foo_bar.py").write_text("# recorded\n", encoding="utf-8", newline="")
    (repo_root / "tests" / "test_unrelated.py").write_text("# not under this extraction's glob\n", encoding="utf-8", newline="")
    manifest = _one_extraction_manifest(
        extraction_id="glob_extraction",
        commit=commit,
        entries=[
            {
                "source_path": "pkg/module.py",
                "dest_path": "tests/test_foo_bar.py",
                "sha256": _sha256("ORIGINAL CONTENT\n"),
            }
        ],
        roots=["tests/test_foo_bar.py"],
    )
    (repo_root / "tests" / "test_foo_bar.py").write_text("ORIGINAL CONTENT\n", encoding="utf-8", newline="")
    report = verify(repo_root=repo_root, checkout=checkout, manifest=manifest)
    # test_unrelated.py is outside this extraction's exact-name glob root,
    # so it correctly does not appear as unrecorded.
    assert report["unrecorded"] == []
    assert report["verified_count"] == 1


def test_commit_not_present_in_checkout_raises_provenance_error(tmp_path, synthetic_checkout):
    checkout, _commit = synthetic_checkout
    repo_root = tmp_path / "product"
    repo_root.mkdir()
    manifest = _one_extraction_manifest(
        extraction_id="synthetic_extraction",
        commit="0" * 40,
        entries=[],
        roots=[],
    )
    with pytest.raises(ProvenanceError):
        verify(repo_root=repo_root, checkout=checkout, manifest=manifest)


def test_assert_is_git_checkout_rejects_a_non_git_directory(tmp_path):
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    with pytest.raises(ProvenanceError):
        assert_is_git_checkout(not_a_repo)


def test_load_manifest_missing_file_raises():
    with pytest.raises(ProvenanceError):
        load_manifest(Path("does/not/exist.json"))


# ---------------------------------------------------------------------------
# End-to-end CLI, against the real manifest but a synthetic checkout that
# mirrors it -- exercises main()/argparse without requiring a real
# qwen-sae-interp checkout on disk.
# ---------------------------------------------------------------------------


def test_main_cli_reports_failure_on_a_missing_destination_file(tmp_path):
    checkout, commit = _init_repo(tmp_path / "checkout"), None
    commit = _write_and_commit(checkout, "pkg/module.py", "ORIGINAL CONTENT\n")

    repo_root = tmp_path / "product"
    (repo_root / "pkg").mkdir(parents=True)
    # deliberately never write pkg/module.py -- missing_dest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _one_extraction_manifest(
                extraction_id="cli_extraction",
                commit=commit,
                entries=[
                    {
                        "source_path": "pkg/module.py",
                        "dest_path": "pkg/module.py",
                        "sha256": _sha256("ORIGINAL CONTENT\n"),
                    }
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
            str(repo_root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "MISSING IN THIS CHECKOUT: cli_extraction:pkg/module.py" in result.stderr
