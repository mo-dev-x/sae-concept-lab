"""Read-only provenance verification for SAE Concept Lab's initial import.

Checks provenance/source_import.json's recorded mapping (source commit,
source path, destination path, SHA-256) against two independent things:

  1. The destination file in THIS repository -- has it been modified or
     deleted since import?
  2. The source blob at the recorded commit in a qwen-sae-interp checkout
     the caller points this at -- does the source commit still contain
     exactly what was imported?

It also walks this repository's imported paths (sae_concept_lab/** and
tests/test_sae_concept_lab_*.py) for files that exist but are NOT listed
in the manifest -- an unrecorded import is exactly the kind of drift a
hash-only check of the recorded entries would miss.

READ-ONLY, by construction: every git operation against the caller-
supplied qwen-sae-interp checkout is `git -C <checkout> cat-file -e/-p
<commit>:<path>` (or `git show <commit>:<path>`) -- reading a blob
directly out of the object database. Neither of these touches that
checkout's working tree, index, or HEAD; there is no `checkout`,
`reset`, `fetch`, `pull`, or `switch` call anywhere in this module. This
script never writes to the qwen-sae-interp checkout it is pointed at.

Usage:
    python -m provenance.verify_provenance --qwen-sae-interp-checkout /path/to/qwen-sae-interp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "provenance" / "source_import.json"

# The two import path patterns this repository's initial import was scoped
# to (per the task that created it) -- used only to find UNRECORDED files,
# never to decide what to trust; trust always comes from the manifest.
IMPORTED_PATH_ROOTS = ("sae_concept_lab",)
IMPORTED_TEST_GLOB = "test_sae_concept_lab_*.py"


class ProvenanceError(RuntimeError):
    """Raised for a fatal setup problem (bad checkout, missing manifest,
    unknown commit) -- distinct from a per-file finding, which is
    accumulated and reported instead of raised."""


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise ProvenanceError(f"provenance manifest not found at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _run_git(checkout: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True,
    )


def assert_is_git_checkout(checkout: Path) -> None:
    if not checkout.exists():
        raise ProvenanceError(f"qwen-sae-interp checkout path does not exist: {checkout}")
    result = _run_git(checkout, ["rev-parse", "--git-dir"])
    if result.returncode != 0:
        raise ProvenanceError(f"{checkout} is not a git checkout (git rev-parse --git-dir failed)")


def assert_commit_exists(checkout: Path, commit: str) -> None:
    result = _run_git(checkout, ["cat-file", "-e", f"{commit}^{{commit}}"])
    if result.returncode != 0:
        raise ProvenanceError(
            f"commit {commit!r} was not found in the checkout at {checkout} -- "
            "cannot verify provenance against a commit this checkout does not have"
        )


def read_source_blob(checkout: Path, commit: str, source_path: str) -> bytes | None:
    """Returns the raw blob bytes at commit:source_path, or None if that
    path does not resolve to a blob at that commit (missing)."""
    exists = _run_git(checkout, ["cat-file", "-e", f"{commit}:{source_path}"])
    if exists.returncode != 0:
        return None
    shown = _run_git(checkout, ["show", f"{commit}:{source_path}"])
    if shown.returncode != 0:
        return None
    return shown.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_actual_imported_files(repo_root: Path) -> set[str]:
    found: set[str] = set()
    for root_name in IMPORTED_PATH_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                found.add(path.relative_to(repo_root).as_posix())
    tests_dir = repo_root / "tests"
    if tests_dir.exists():
        for path in tests_dir.glob(IMPORTED_TEST_GLOB):
            if path.is_file():
                found.add(path.relative_to(repo_root).as_posix())
    return found


def verify(
    *,
    repo_root: Path,
    checkout: Path,
    manifest: dict,
) -> dict:
    """Returns a report dict: {missing_source, modified_source,
    missing_dest, modified_dest, unrecorded, verified_count}. Every value
    except verified_count is a list of relative path strings -- never a
    diff, never file content, nothing beyond "which path and which
    category of finding"."""
    commit = manifest["source_repository"]["commit"]
    assert_commit_exists(checkout, commit)

    missing_source: list[str] = []
    modified_source: list[str] = []
    missing_dest: list[str] = []
    modified_dest: list[str] = []
    verified_count = 0

    recorded_dest_paths: set[str] = set()

    for entry in manifest["imported_files"]:
        source_path = entry["source_path"]
        dest_path = entry["dest_path"]
        expected_sha256 = entry["sha256"]
        recorded_dest_paths.add(dest_path)

        source_blob = read_source_blob(checkout, commit, source_path)
        if source_blob is None:
            missing_source.append(source_path)
        elif sha256_bytes(source_blob) != expected_sha256:
            modified_source.append(source_path)

        dest_file = repo_root / dest_path
        if not dest_file.exists():
            missing_dest.append(dest_path)
        else:
            dest_bytes = dest_file.read_bytes()
            if sha256_bytes(dest_bytes) != expected_sha256:
                modified_dest.append(dest_path)

        if (
            source_path not in missing_source
            and source_path not in modified_source
            and dest_path not in missing_dest
            and dest_path not in modified_dest
        ):
            verified_count += 1

    actual_files = find_actual_imported_files(repo_root)
    unrecorded = sorted(actual_files - recorded_dest_paths)

    return {
        "missing_source": missing_source,
        "modified_source": modified_source,
        "missing_dest": missing_dest,
        "modified_dest": modified_dest,
        "unrecorded": unrecorded,
        "verified_count": verified_count,
        "manifest_entry_count": len(manifest["imported_files"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--qwen-sae-interp-checkout",
        required=True,
        type=Path,
        help="Path to a local qwen-sae-interp git checkout. Read-only: never modified.",
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        assert_is_git_checkout(args.qwen_sae_interp_checkout)
        report = verify(repo_root=args.repo_root, checkout=args.qwen_sae_interp_checkout, manifest=manifest)
    except ProvenanceError as exc:
        print(f"PROVENANCE VERIFICATION FAILED TO RUN: {exc}", file=sys.stderr)
        return 2

    problems = (
        report["missing_source"]
        + report["modified_source"]
        + report["missing_dest"]
        + report["modified_dest"]
        + report["unrecorded"]
    )

    print(f"manifest entries: {report['manifest_entry_count']}")
    print(f"verified clean: {report['verified_count']}")
    print(f"missing at source commit: {len(report['missing_source'])}")
    print(f"modified at source commit: {len(report['modified_source'])}")
    print(f"missing in this checkout: {len(report['missing_dest'])}")
    print(f"modified in this checkout: {len(report['modified_dest'])}")
    print(f"unrecorded imported-path files: {len(report['unrecorded'])}")

    if not problems:
        print("PROVENANCE OK: every imported file matches its recorded source commit and hash.")
        return 0

    print("PROVENANCE VERIFICATION FAILED:", file=sys.stderr)
    for path in report["missing_source"]:
        print(f"  MISSING AT SOURCE COMMIT: {path}", file=sys.stderr)
    for path in report["modified_source"]:
        print(f"  MODIFIED AT SOURCE COMMIT (should be immutable -- check the commit hash): {path}", file=sys.stderr)
    for path in report["missing_dest"]:
        print(f"  MISSING IN THIS CHECKOUT: {path}", file=sys.stderr)
    for path in report["modified_dest"]:
        print(f"  MODIFIED IN THIS CHECKOUT SINCE IMPORT: {path}", file=sys.stderr)
    for path in report["unrecorded"]:
        print(f"  UNRECORDED (present under an imported path, not in the manifest): {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
