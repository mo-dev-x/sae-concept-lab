"""Read-only provenance verification for this repository's extractions.

Checks provenance/source_import.json's recorded mapping (source commit,
source path, destination path, SHA-256) against two independent things,
for every extraction listed in the manifest's `extractions` array:

  1. The destination file in THIS repository -- has it been modified or
     deleted since import?
  2. The source blob at that extraction's own recorded commit in a
     qwen-sae-interp checkout the caller points this at -- does the
     source commit still contain exactly what was imported?

Each extraction is independent and may name its own source commit(s) --
the original UI import (`sae_concept_lab_ui`) records a single `commit`;
the concept-bundle contract extraction (`concept_bundle_contract`)
records `checkout_commit` (what was actually read), plus
`contract_base_commit` and `frozen_pack_commit` for the record (not
separately re-fetched, since checkout_commit is asserted unchanged from
both -- see that extraction's `contract_base_commit_note`).

It also walks the UNION of every extraction's declared
`import_path_roots` for files that exist but are NOT recorded by ANY
extraction -- an unrecorded import is exactly the kind of drift a
hash-only check of the recorded entries would miss. This is computed
globally, not per extraction, because one extraction's root can nest
inside another's (concept_bundle_contract's
sae_concept_lab/canonical/concept_bundle sits inside
sae_concept_lab_ui's sae_concept_lab root) -- scanning in isolation would
flag every file the inner extraction owns as unrecorded from the outer
one's point of view.

Scaffolding (parent-package markers added only to make an extraction
importable, carrying no extracted content -- e.g.
sae_concept_lab/canonical/__init__.py) is declared per extraction under
`scaffolding_added` and is checked for existence only, never hashed
against a source commit, and never flagged unrecorded.

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

# Manifest keys that carry {source_path, dest_path, sha256} entries subject
# to hash verification. An extraction may use any subset of these -- the
# original UI import uses only `imported_files`; the concept-bundle
# extraction splits runtime modules from copied conformance artifacts so
# each can be described with its own role, but both are checked identically.
ENTRY_LIST_KEYS = ("imported_files", "imported_modules", "extracted_artifacts")


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


def extraction_commit(extraction: dict) -> str:
    """The commit an extraction's entries should be read against. The
    original import names it `commit`; the concept-bundle extraction names
    the commit it actually read from `checkout_commit` (contract_base_commit
    and frozen_pack_commit are recorded for the audit trail but are
    ancestors already asserted unchanged, not independently re-fetched)."""
    source_repo = extraction["source_repository"]
    commit = source_repo.get("commit") or source_repo.get("checkout_commit")
    if not commit:
        raise ProvenanceError(
            f"extraction {extraction.get('extraction_id')!r} has no 'commit' or "
            "'checkout_commit' in source_repository"
        )
    return commit


def extraction_entries(extraction: dict) -> list[dict]:
    entries: list[dict] = []
    for key in ENTRY_LIST_KEYS:
        entries.extend(extraction.get(key, []))
    return entries


def extraction_scaffolding_paths(extraction: dict) -> list[str]:
    return [item["path"] for item in extraction.get("scaffolding_added", [])]


def _expand_root(repo_root: Path, root: str) -> set[str]:
    """A root is either a literal directory (walked recursively) or a glob
    pattern (containing '*', matched directly) -- both relative to
    repo_root. Returns relative-path strings of every file found."""
    found: set[str] = set()
    if "*" in root:
        for path in repo_root.glob(root):
            if path.is_file():
                found.add(path.relative_to(repo_root).as_posix())
        return found
    base = repo_root / root
    if not base.exists():
        return found
    for path in base.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            found.add(path.relative_to(repo_root).as_posix())
    return found


def find_actual_files_under_all_roots(repo_root: Path, extractions: list[dict]) -> set[str]:
    """The union of every extraction's declared import_path_roots, walked
    once. Must be computed globally (not per extraction) because one
    extraction's root can nest inside another's."""
    found: set[str] = set()
    for extraction in extractions:
        for root in extraction.get("import_path_roots", []):
            found |= _expand_root(repo_root, root)
    return found


def verify(
    *,
    repo_root: Path,
    checkout: Path,
    manifest: dict,
) -> dict:
    """Returns a report dict. `missing_source`, `modified_source`,
    `missing_dest`, `modified_dest`, and `missing_scaffolding` are lists of
    "extraction_id:path" strings; `unrecorded` is a list of plain relative
    paths (computed globally, not per extraction). Every value is a path
    string and a category label -- never a diff, never file content."""
    extractions = manifest["extractions"]

    missing_source: list[str] = []
    modified_source: list[str] = []
    missing_dest: list[str] = []
    modified_dest: list[str] = []
    missing_scaffolding: list[str] = []
    verified_count = 0
    total_entries = 0

    recorded_dest_paths: set[str] = set()

    for extraction in extractions:
        extraction_id = extraction["extraction_id"]
        commit = extraction_commit(extraction)
        assert_commit_exists(checkout, commit)

        for entry in extraction_entries(extraction):
            total_entries += 1
            source_path = entry["source_path"]
            dest_path = entry["dest_path"]
            expected_sha256 = entry["sha256"]
            recorded_dest_paths.add(dest_path)

            entry_ok = True

            source_blob = read_source_blob(checkout, commit, source_path)
            if source_blob is None:
                missing_source.append(f"{extraction_id}:{source_path}")
                entry_ok = False
            elif sha256_bytes(source_blob) != expected_sha256:
                modified_source.append(f"{extraction_id}:{source_path}")
                entry_ok = False

            dest_file = repo_root / dest_path
            if not dest_file.exists():
                missing_dest.append(f"{extraction_id}:{dest_path}")
                entry_ok = False
            else:
                dest_bytes = dest_file.read_bytes()
                if sha256_bytes(dest_bytes) != expected_sha256:
                    modified_dest.append(f"{extraction_id}:{dest_path}")
                    entry_ok = False

            if entry_ok:
                verified_count += 1

        for scaffold_path in extraction_scaffolding_paths(extraction):
            recorded_dest_paths.add(scaffold_path)
            if not (repo_root / scaffold_path).exists():
                missing_scaffolding.append(f"{extraction_id}:{scaffold_path}")

    actual_files = find_actual_files_under_all_roots(repo_root, extractions)
    unrecorded = sorted(actual_files - recorded_dest_paths)

    return {
        "missing_source": missing_source,
        "modified_source": modified_source,
        "missing_dest": missing_dest,
        "modified_dest": modified_dest,
        "missing_scaffolding": missing_scaffolding,
        "unrecorded": unrecorded,
        "verified_count": verified_count,
        "manifest_entry_count": total_entries,
        "extraction_count": len(extractions),
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
        + report["missing_scaffolding"]
        + report["unrecorded"]
    )

    print(f"extractions: {report['extraction_count']}")
    print(f"manifest entries: {report['manifest_entry_count']}")
    print(f"verified clean: {report['verified_count']}")
    print(f"missing at source commit: {len(report['missing_source'])}")
    print(f"modified at source commit: {len(report['modified_source'])}")
    print(f"missing in this checkout: {len(report['missing_dest'])}")
    print(f"modified in this checkout: {len(report['modified_dest'])}")
    print(f"missing declared scaffolding: {len(report['missing_scaffolding'])}")
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
    for path in report["missing_scaffolding"]:
        print(f"  MISSING DECLARED SCAFFOLDING: {path}", file=sys.stderr)
    for path in report["unrecorded"]:
        print(f"  UNRECORDED (present under an imported path, not in the manifest): {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
