"""Read-only provenance verification for this repository's extractions.

Three extraction classes exist, and they are verified completely
differently -- this is the ratified policy, not a convenience split:

  HISTORICAL_SEED     A past import whose current bytes are permitted to
                      evolve (e.g. sae_concept_lab_ui: app.py, core/*, ui/*,
                      fixtures/loader.py -- all now wired to call the
                      canonical package directly, which is expected
                      evolution, not drift). Verified by reading git objects
                      at this repository's OWN `historical_seed_commit`
                      (never qwen-sae-interp, never current working-tree
                      bytes) and hash-comparing them against the manifest.
                      The verdict this class prints declares faithfulness
                      AT THE IMPORT COMMIT and explicitly disclaims checking
                      current bytes.

  CANONICAL_MIRROR    A byte-for-byte mirror that may NEVER evolve (the
                      eight concept_bundle contract modules). Verified by
                      hash-comparing CURRENT bytes against both the manifest
                      and a live qwen-sae-interp checkout, AND by re-running
                      every frozen conformance vector against the extracted
                      package (the count is read from the copied vectors.json
                      at verification time, never hardcoded here, since the
                      frozen pack is replaced whole on a deliberate
                      re-extraction -- see provenance/source_import.json's
                      concept_bundle_contract.supersedes_previous_extraction
                      for the currently mirrored pack's commit). Membership
                      is derived from that pack's own export_inventory.json
                      minimum_export_surface list, not merely asserted --
                      see assert_no_reclassification().

  RUNTIME_CODE_MIRROR A byte-for-byte mirror of extracted RUNTIME code (the
                      Qwen/Gemma intervention loaders) that may NEVER evolve,
                      like CANONICAL_MIRROR, but with NO frozen conformance
                      pack of its own -- there is nothing analogous to
                      concept_bundle's 50/75-vector pack for this code, so
                      this class never runs one. Verified by hash-comparing
                      CURRENT bytes against both the manifest and a live
                      qwen-sae-interp checkout, at either whole-file
                      granularity (`imported_modules`/`imported_files`) or
                      per-function granularity (`imported_functions`, for a
                      destination file mechanically assembled from a NAMED
                      SUBSET of a source file's functions -- each function's
                      body is independently AST-extracted and hashed, both
                      at the source commit and in the destination file).
                      CRITICALLY: this class carries NO claim that the
                      mirrored code has been mechanically verified against
                      real model/SAE weights -- that is an entirely separate
                      fact, tracked by
                      sae_concept_lab.core.runtime_acceptance and checked by
                      the release gate independently of this verifier.

The three classes never share a field, a serialized key, or a verdict
vocabulary with the SCIENTIFIC `provenance` field (Provenance.ATTESTED /
CANDIDATE / DRAFT / FAKE / UNKNOWN, defined in
sae_concept_lab.canonical.concept_bundle.schema) -- extraction_class is a
code-provenance axis, entirely orthogonal to that scientific-content
axis, and this module never reads or writes the word "provenance" as a
code-extraction field.

Every verdict is scope-qualified: main() never prints a bare "PASS" --
each extraction's line names its class and the commit its faithfulness
is judged against, and reads exactly one of the three forms below (with
the actual short commit hash and vector count substituted in):

  HISTORICAL_SEED <short-commit> import faithful at import commit; current bytes not checked
  CANONICAL_MIRROR <short-commit> current bytes match canonical source; conformance vectors pass
  RUNTIME_CODE_MIRROR <short-commit> current bytes match source; no conformance pack applies to this extraction

Rejects reclassification: a HISTORICAL_SEED extraction naming a
source_path that appears in the canonical pack's own minimum_export_surface,
OR in any CANONICAL_MIRROR/RUNTIME_CODE_MIRROR extraction's own recorded
source_paths, is a fatal configuration error (ProvenanceError), not a
per-file finding -- it would let an immutable-mirror-owned path escape
strict verification by being relabelled.

READ-ONLY, by construction. Every git operation -- against the caller-
supplied qwen-sae-interp checkout (CANONICAL_MIRROR only) or against this
repository's own history (HISTORICAL_SEED) -- is `git -C <path>
cat-file -e/-p <commit>:<file>` or `git show <commit>:<file>`: reading a
blob directly out of the object database. Neither touches a working
tree, an index, or HEAD; there is no `checkout`, `reset`, `fetch`,
`pull`, or `switch` call anywhere in this module.

Usage:
    python -m provenance.verify_provenance --qwen-sae-interp-checkout /path/to/qwen-sae-interp
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "provenance" / "source_import.json"

CANONICAL_PACK_DIR = Path("provenance") / "runtime_extractions" / "concept_bundle"
CANONICAL_INVENTORY_RELATIVE_PATH = CANONICAL_PACK_DIR / "export_inventory.json"
CANONICAL_RUNNER_RELATIVE_PATH = CANONICAL_PACK_DIR / "concept_bundle_conformance.py"
CANONICAL_VECTORS_RELATIVE_PATH = CANONICAL_PACK_DIR / "vectors.json"
CANONICAL_PACKAGE = "sae_concept_lab.canonical.concept_bundle"

# Manifest keys that carry {source_path, dest_path, sha256} entries subject
# to hash verification. An extraction may use any subset of these -- the
# original UI import uses only `imported_files`; the concept-bundle
# extraction splits runtime modules from copied conformance artifacts so
# each can be described with its own role, but both are checked identically.
ENTRY_LIST_KEYS = ("imported_files", "imported_modules", "extracted_artifacts")


class ProvenanceError(RuntimeError):
    """Raised for a fatal setup problem (bad checkout, missing manifest,
    unknown commit, reclassification) -- distinct from a per-file
    finding, which is accumulated and reported instead of raised."""


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise ProvenanceError(f"provenance manifest not found at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True)


def assert_is_git_checkout(repo: Path) -> None:
    if not repo.exists():
        raise ProvenanceError(f"path does not exist: {repo}")
    result = _run_git(repo, ["rev-parse", "--git-dir"])
    if result.returncode != 0:
        raise ProvenanceError(f"{repo} is not a git checkout (git rev-parse --git-dir failed)")


def assert_commit_exists(repo: Path, commit: str) -> None:
    result = _run_git(repo, ["cat-file", "-e", f"{commit}^{{commit}}"])
    if result.returncode != 0:
        raise ProvenanceError(
            f"commit {commit!r} was not found in {repo} -- cannot verify provenance against a "
            "commit this checkout does not have"
        )


def read_blob_at_commit(repo: Path, commit: str, path: str) -> bytes | None:
    """Returns the raw blob bytes at commit:path in `repo`'s own object
    database, or None if that path does not resolve to a blob at that
    commit (missing). `repo` may be this repository itself (HISTORICAL_SEED)
    or an external qwen-sae-interp checkout (CANONICAL_MIRROR) -- both are
    ordinary git repositories from this function's point of view."""
    exists = _run_git(repo, ["cat-file", "-e", f"{commit}:{path}"])
    if exists.returncode != 0:
        return None
    shown = _run_git(repo, ["show", f"{commit}:{path}"])
    if shown.returncode != 0:
        return None
    return shown.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    """The union of every CANONICAL_MIRROR/RUNTIME_CODE_MIRROR extraction's
    declared import_path_roots, walked once. Deliberately excludes
    HISTORICAL_SEED roots: those extractions permit current evolution, so a
    new file appearing under sae_concept_lab/ (e.g. fixtures/labels.py, or a
    new canonical fixture document) is expected product-native work, not an
    unrecorded import -- "new product-native files need no invented
    extraction class" is the ratified policy this encodes. Computed as a
    union (not per extraction) because an immutable-mirror root can nest
    inside a HISTORICAL_SEED root (canonical/concept_bundle or
    extracted_runtime inside sae_concept_lab), and only the inner, stricter
    root's contents must ever be flagged."""
    found: set[str] = set()
    for extraction in extractions:
        if extraction.get("extraction_class") not in ("CANONICAL_MIRROR", "RUNTIME_CODE_MIRROR"):
            continue
        for root in extraction.get("import_path_roots", []):
            found |= _expand_root(repo_root, root)
    return found


def _immutable_mirror_source_paths(manifest: dict) -> set[str]:
    """Every source_path recorded by a CANONICAL_MIRROR or
    RUNTIME_CODE_MIRROR extraction, across imported_modules/imported_files
    (whole-file entries) and imported_functions (per-function entries) --
    the full set of paths a HISTORICAL_SEED extraction must never be able
    to claim."""
    paths: set[str] = set()
    for extraction in manifest["extractions"]:
        if extraction.get("extraction_class") not in ("CANONICAL_MIRROR", "RUNTIME_CODE_MIRROR"):
            continue
        for entry in extraction_entries(extraction):
            paths.add(entry["source_path"])
        for entry in extraction.get("imported_functions", []):
            paths.add(entry["source_path"])
    return paths


def assert_no_reclassification(repo_root: Path, manifest: dict) -> None:
    """A HISTORICAL_SEED extraction naming a source_path that the frozen
    canonical pack's own export_inventory.json lists under
    minimum_export_surface, OR that any CANONICAL_MIRROR/RUNTIME_CODE_MIRROR
    extraction in this manifest itself records, is a fatal reclassification
    attempt: it would let a path that must be verified byte-for-byte escape
    into the more lenient "faithful at import commit, current bytes not
    checked" class by being relabelled."""
    canonical_source_paths = _immutable_mirror_source_paths(manifest)
    inventory_path = repo_root / CANONICAL_INVENTORY_RELATIVE_PATH
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        canonical_source_paths |= {m["path"] for m in inventory["minimum_export_surface"]}

    for extraction in manifest["extractions"]:
        if extraction.get("extraction_class") != "HISTORICAL_SEED":
            continue
        for entry in extraction_entries(extraction):
            if entry["source_path"] in canonical_source_paths:
                raise ProvenanceError(
                    f"reclassification rejected: extraction {extraction['extraction_id']!r} "
                    f"(HISTORICAL_SEED) names source_path {entry['source_path']!r}, which is an "
                    "immutable-mirror path (CANONICAL_MIRROR or RUNTIME_CODE_MIRROR) per this "
                    "manifest or the frozen pack's own export inventory. An immutable-mirror-owned "
                    "path may never be verified under the more lenient historical-seed rules."
                )


def run_conformance_vectors(repo_root: Path) -> dict:
    """Loads the copied runner and vectors from THIS repository (never
    qwen-sae-interp) and calls verify_pack() directly, in process, against
    the extracted package. Returns {"ran": bool, "vectors_checked": int,
    "failures": list[str]}."""
    runner_path = repo_root / CANONICAL_RUNNER_RELATIVE_PATH
    vectors_path = repo_root / CANONICAL_VECTORS_RELATIVE_PATH
    if not runner_path.exists() or not vectors_path.exists():
        return {"ran": False, "vectors_checked": 0, "failures": ["runner or vectors file is missing"]}

    spec = importlib.util.spec_from_file_location("concept_bundle_conformance_check", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack = json.loads(vectors_path.read_text(encoding="utf-8"))
    failures = module.verify_pack(pack, package=CANONICAL_PACKAGE)
    return {"ran": True, "vectors_checked": len(pack["vectors"]), "failures": failures}


def _ast_top_level_source(text: str, name: str) -> str | None:
    """The exact source segment of a top-level def/class named `name`,
    decorator excluded (ast.get_source_segment's own behavior since Python
    3.8: a decorated FunctionDef/ClassDef's own `lineno` is the def/class
    keyword's line, not its decorator's -- a caller reconstructing a
    decorated definition must reattach the decorator itself, exactly as
    this repository's extraction script did for InterventionTrace's
    @dataclass). Returns None if `name` is not a top-level definition, or
    if `text` does not parse as Python at all."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if getattr(node, "name", None) == name:
            return ast.get_source_segment(text, node)
    return None


def verify_function_entry(checkout: Path, repo_root: Path, entry: dict, default_commit: str) -> str | None:
    """One `imported_functions` entry: independently AST-extracts the named
    function/class from BOTH the source blob at its commit and the
    destination file, hashes each segment, and confirms both match the
    recorded hash. Returns None if clean, else a human-readable problem
    string naming exactly what did not match."""
    entry_commit = entry.get("source_commit", default_commit)
    if entry_commit != default_commit:
        assert_commit_exists(checkout, entry_commit)

    source_blob = read_blob_at_commit(checkout, entry_commit, entry["source_path"])
    if source_blob is None:
        return f"source {entry['source_path']!r} missing at {entry_commit}"
    try:
        source_text = source_blob.decode("utf-8")
    except UnicodeDecodeError:
        return f"source {entry['source_path']!r} at {entry_commit} is not valid UTF-8"
    source_seg = _ast_top_level_source(source_text, entry["function_name"])
    if source_seg is None:
        return f"{entry['function_name']!r} not found as a top-level def/class in source {entry['source_path']!r} at {entry_commit}"
    source_hash = sha256_bytes(source_seg.encode("utf-8"))

    dest_file = repo_root / entry["dest_path"]
    if not dest_file.exists():
        return f"dest {entry['dest_path']!r} missing"
    try:
        dest_text = dest_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"dest {entry['dest_path']!r} is not valid UTF-8"
    dest_name = entry.get("dest_function_name", entry["function_name"])
    dest_seg = _ast_top_level_source(dest_text, dest_name)
    if dest_seg is None:
        return f"{dest_name!r} not found as a top-level def/class in dest {entry['dest_path']!r}"
    dest_hash = sha256_bytes(dest_seg.encode("utf-8"))

    if source_hash != entry["sha256"]:
        return f"source body of {entry['function_name']!r} no longer matches its recorded hash"
    if dest_hash != entry["sha256"]:
        return f"dest body of {dest_name!r} no longer matches its recorded hash"
    return None


def verify_runtime_code_mirror_extraction(repo_root: Path, checkout: Path, extraction: dict) -> dict:
    """RUNTIME_CODE_MIRROR: byte-for-byte immutable like CANONICAL_MIRROR,
    but with no frozen conformance pack -- verified purely by hash, at
    whole-file granularity (imported_modules/imported_files) and/or
    per-function granularity (imported_functions, via verify_function_entry).
    Carries no claim about mechanical verification against real weights;
    that is sae_concept_lab.core.runtime_acceptance's separate concern."""
    commit = extraction["source_repository"].get("checkout_commit") or extraction["source_repository"].get("commit")
    assert_commit_exists(checkout, commit)

    missing_source: list[str] = []
    modified_source: list[str] = []
    missing_dest: list[str] = []
    modified_dest: list[str] = []
    function_problems: list[str] = []

    for entry in extraction_entries(extraction):
        entry_commit = entry.get("source_commit", commit)
        if entry_commit != commit:
            assert_commit_exists(checkout, entry_commit)
        source_blob = read_blob_at_commit(checkout, entry_commit, entry["source_path"])
        if source_blob is None:
            missing_source.append(entry["source_path"])
        elif sha256_bytes(source_blob) != entry["sha256"]:
            modified_source.append(entry["source_path"])

        dest_file = repo_root / entry["dest_path"]
        if not dest_file.exists():
            missing_dest.append(entry["dest_path"])
        elif sha256_bytes(dest_file.read_bytes()) != entry["sha256"]:
            modified_dest.append(entry["dest_path"])

    for entry in extraction.get("imported_functions", []):
        problem = verify_function_entry(checkout, repo_root, entry, commit)
        if problem is not None:
            function_problems.append(f"{entry['function_name']} -> {entry['dest_path']}: {problem}")

    clean = not (missing_source or modified_source or missing_dest or modified_dest or function_problems)
    return {
        "missing_source": missing_source,
        "modified_source": modified_source,
        "missing_dest": missing_dest,
        "modified_dest": modified_dest,
        "function_problems": function_problems,
        "clean": clean,
    }


def verify_historical_seed_extraction(repo_root: Path, extraction: dict) -> dict:
    """HISTORICAL_SEED: reads git objects at THIS repository's own
    historical_seed_commit and hash-compares them against the manifest.
    Never touches current working-tree bytes, and never touches
    qwen-sae-interp."""
    commit = extraction["historical_seed_commit"]
    assert_commit_exists(repo_root, commit)

    missing: list[str] = []
    modified: list[str] = []
    for entry in extraction_entries(extraction):
        blob = read_blob_at_commit(repo_root, commit, entry["dest_path"])
        if blob is None:
            missing.append(entry["dest_path"])
        elif sha256_bytes(blob) != entry["sha256"]:
            modified.append(entry["dest_path"])
    return {"missing": missing, "modified": modified, "clean": not missing and not modified}


def verify_canonical_mirror_extraction(repo_root: Path, checkout: Path, extraction: dict) -> dict:
    """CANONICAL_MIRROR: hash-compares CURRENT bytes (both in this
    repository and at the source commit in the qwen-sae-interp checkout)
    AND re-runs every frozen conformance vector.

    An entry may declare its own `source_commit`, overriding the
    extraction-level checkout_commit for that one entry. This exists for
    export_inventory.json: the frozen pack stamps frozen_at_commit in an
    immediately-following commit, since a commit cannot name its own
    hash inside a file it contains, so THAT file's byte-for-byte source is
    one commit later than the modules and vectors it describes."""
    commit = extraction["source_repository"].get("checkout_commit") or extraction["source_repository"].get("commit")
    assert_commit_exists(checkout, commit)

    missing_source: list[str] = []
    modified_source: list[str] = []
    missing_dest: list[str] = []
    modified_dest: list[str] = []

    for entry in extraction_entries(extraction):
        entry_commit = entry.get("source_commit", commit)
        if entry_commit != commit:
            assert_commit_exists(checkout, entry_commit)
        source_blob = read_blob_at_commit(checkout, entry_commit, entry["source_path"])
        if source_blob is None:
            missing_source.append(entry["source_path"])
        elif sha256_bytes(source_blob) != entry["sha256"]:
            modified_source.append(entry["source_path"])

        dest_file = repo_root / entry["dest_path"]
        if not dest_file.exists():
            missing_dest.append(entry["dest_path"])
        else:
            if sha256_bytes(dest_file.read_bytes()) != entry["sha256"]:
                modified_dest.append(entry["dest_path"])

    vector_result = run_conformance_vectors(repo_root)
    bytes_clean = not (missing_source or modified_source or missing_dest or modified_dest)
    vectors_clean = vector_result["ran"] and not vector_result["failures"]
    return {
        "missing_source": missing_source,
        "modified_source": modified_source,
        "missing_dest": missing_dest,
        "modified_dest": modified_dest,
        "vector_result": vector_result,
        "clean": bytes_clean and vectors_clean,
    }


def _short_commit(value: str) -> str:
    return value[:7]


def historical_seed_verdict(extraction: dict, result: dict) -> str:
    short = extraction.get("historical_seed_commit_short") or _short_commit(extraction["historical_seed_commit"])
    if result["clean"]:
        return f"HISTORICAL_SEED {short} import faithful at import commit; current bytes not checked"
    problems = []
    if result["missing"]:
        problems.append(f"{len(result['missing'])} missing at import commit")
    if result["modified"]:
        problems.append(f"{len(result['modified'])} modified since import commit")
    return f"HISTORICAL_SEED {short} import NOT faithful at import commit: " + "; ".join(problems)


def canonical_mirror_verdict(extraction: dict, result: dict) -> str:
    source_repo = extraction["source_repository"]
    short = source_repo.get("frozen_pack_commit_short") or _short_commit(source_repo["frozen_pack_commit"])
    if result["clean"]:
        return f"CANONICAL_MIRROR {short} current bytes match canonical source; conformance vectors pass"
    problems = []
    n_byte_problems = (
        len(result["missing_source"]) + len(result["modified_source"])
        + len(result["missing_dest"]) + len(result["modified_dest"])
    )
    if n_byte_problems:
        problems.append(f"{n_byte_problems} byte mismatch(es)")
    vector_result = result["vector_result"]
    if not vector_result["ran"]:
        problems.append("conformance vectors could not be run")
    elif vector_result["failures"]:
        problems.append(f"{len(vector_result['failures'])} conformance vector failure(s)")
    return f"CANONICAL_MIRROR {short} current bytes/vectors NOT clean: " + "; ".join(problems)


def runtime_code_mirror_verdict(extraction: dict, result: dict) -> str:
    source_repo = extraction["source_repository"]
    short = source_repo.get("checkout_commit_short") or _short_commit(
        source_repo.get("checkout_commit") or source_repo["commit"]
    )
    if result["clean"]:
        return f"RUNTIME_CODE_MIRROR {short} current bytes match source; no conformance pack applies to this extraction"
    problems = []
    n_byte_problems = (
        len(result["missing_source"]) + len(result["modified_source"])
        + len(result["missing_dest"]) + len(result["modified_dest"])
    )
    if n_byte_problems:
        problems.append(f"{n_byte_problems} byte mismatch(es)")
    if result["function_problems"]:
        problems.append(f"{len(result['function_problems'])} function-level mismatch(es): " + "; ".join(result["function_problems"]))
    return f"RUNTIME_CODE_MIRROR {short} NOT clean: " + "; ".join(problems)


def verify(
    *,
    repo_root: Path,
    checkout: Path,
    manifest: dict,
) -> dict:
    """Returns a report dict with per-class findings, a `verdicts` list of
    scope-qualified strings (one per extraction, in manifest order --
    never a bare PASS), and the usual global unrecorded/scaffolding
    checks. Raises ProvenanceError (not a finding) on reclassification."""
    extractions = manifest["extractions"]
    assert_no_reclassification(repo_root, manifest)

    verdicts: list[str] = []
    missing_source: list[str] = []
    modified_source: list[str] = []
    missing_dest: list[str] = []
    modified_dest: list[str] = []
    missing_scaffolding: list[str] = []
    verified_count = 0
    total_entries = 0

    for extraction in extractions:
        extraction_id = extraction["extraction_id"]
        extraction_class = extraction.get("extraction_class")
        entries = extraction_entries(extraction)
        total_entries += len(entries) + len(extraction.get("imported_functions", []))

        if extraction_class == "HISTORICAL_SEED":
            result = verify_historical_seed_extraction(repo_root, extraction)
            if result["clean"]:
                verified_count += len(entries)
            missing_dest.extend(f"{extraction_id}:{p}" for p in result["missing"])
            modified_dest.extend(f"{extraction_id}:{p}" for p in result["modified"])
            verdicts.append(historical_seed_verdict(extraction, result))
        elif extraction_class == "CANONICAL_MIRROR":
            result = verify_canonical_mirror_extraction(repo_root, checkout, extraction)
            if result["clean"]:
                verified_count += len(entries)
            missing_source.extend(f"{extraction_id}:{p}" for p in result["missing_source"])
            modified_source.extend(f"{extraction_id}:{p}" for p in result["modified_source"])
            missing_dest.extend(f"{extraction_id}:{p}" for p in result["missing_dest"])
            modified_dest.extend(f"{extraction_id}:{p}" for p in result["modified_dest"])
            verdicts.append(canonical_mirror_verdict(extraction, result))
        elif extraction_class == "RUNTIME_CODE_MIRROR":
            result = verify_runtime_code_mirror_extraction(repo_root, checkout, extraction)
            if result["clean"]:
                verified_count += len(entries) + len(extraction.get("imported_functions", []))
            missing_source.extend(f"{extraction_id}:{p}" for p in result["missing_source"])
            modified_source.extend(f"{extraction_id}:{p}" for p in result["modified_source"])
            missing_dest.extend(f"{extraction_id}:{p}" for p in result["missing_dest"])
            modified_dest.extend(f"{extraction_id}:{p}" for p in result["function_problems"])
            verdicts.append(runtime_code_mirror_verdict(extraction, result))
        else:
            raise ProvenanceError(
                f"extraction {extraction_id!r} has no recognized extraction_class "
                "(expected HISTORICAL_SEED, CANONICAL_MIRROR, or RUNTIME_CODE_MIRROR)"
            )

        for scaffold_path in extraction_scaffolding_paths(extraction):
            if not (repo_root / scaffold_path).exists():
                missing_scaffolding.append(f"{extraction_id}:{scaffold_path}")

    all_scaffolding_and_recorded = (
        {p for extraction in extractions for p in extraction_scaffolding_paths(extraction)}
        | {entry["dest_path"] for extraction in extractions for entry in extraction_entries(extraction)}
        | {
            entry["dest_path"]
            for extraction in extractions
            for entry in extraction.get("imported_functions", [])
        }
    )
    actual_files = find_actual_files_under_all_roots(repo_root, extractions)
    unrecorded = sorted(actual_files - all_scaffolding_and_recorded)

    return {
        "verdicts": verdicts,
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
        help=(
            "Path to a local qwen-sae-interp git checkout. Read-only: never modified. Required for "
            "CANONICAL_MIRROR and RUNTIME_CODE_MIRROR extractions; HISTORICAL_SEED extractions "
            "never use it."
        ),
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        assert_is_git_checkout(args.qwen_sae_interp_checkout)
        assert_is_git_checkout(args.repo_root)
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
    for verdict in report["verdicts"]:
        print(f"verdict: {verdict}")
    print(f"missing at source commit: {len(report['missing_source'])}")
    print(f"modified at source commit: {len(report['modified_source'])}")
    print(f"missing in this checkout: {len(report['missing_dest'])}")
    print(f"modified in this checkout: {len(report['modified_dest'])}")
    print(f"missing declared scaffolding: {len(report['missing_scaffolding'])}")
    print(f"unrecorded imported-path files: {len(report['unrecorded'])}")

    if not problems:
        print("PROVENANCE OK: every extraction's verdict is clean (see verdict lines above).")
        return 0

    print("PROVENANCE VERIFICATION FAILED:", file=sys.stderr)
    for path in report["missing_source"]:
        print(f"  MISSING AT SOURCE COMMIT: {path}", file=sys.stderr)
    for path in report["modified_source"]:
        print(f"  MODIFIED AT SOURCE COMMIT (should be immutable -- check the commit hash): {path}", file=sys.stderr)
    for path in report["missing_dest"]:
        print(f"  MISSING: {path}", file=sys.stderr)
    for path in report["modified_dest"]:
        print(f"  MODIFIED: {path}", file=sys.stderr)
    for path in report["missing_scaffolding"]:
        print(f"  MISSING DECLARED SCAFFOLDING: {path}", file=sys.stderr)
    for path in report["unrecorded"]:
        print(f"  UNRECORDED (present under an imported path, not in the manifest): {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
