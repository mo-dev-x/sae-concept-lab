"""Certifies sae_concept_lab.canonical.concept_bundle against Engineer 3's
frozen 50-vector conformance pack (qwen-sae-interp commit cdae9c7,
contract base 4675def, frozen at fabf702).

Loads the pack and calls verify_pack() from the COPIED runner
(provenance/runtime_extractions/concept_bundle/concept_bundle_conformance.py)
directly, in process -- no qwen-sae-interp checkout required, so this
test is fully standalone. verify_pack() reads its inputs from the pack's
own JSON and never imports the canonical fixtures module, which is what
makes the pack evidence rather than a restatement: an extraction that
dropped a rule and the fixtures exercising it would still fail here.

The runner is dynamically loaded (importlib.util.spec_from_file_location)
rather than imported as a package, matching this project's established
convention for loading a file with no __init__.py of its own -- see
scripts/legacy/gemma3_tool.py's _load_sweep_module() in qwen-sae-interp
for the precedent this follows.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "provenance" / "runtime_extractions" / "concept_bundle"
RUNNER_PATH = PACK_DIR / "concept_bundle_conformance.py"
VECTORS_PATH = PACK_DIR / "vectors.json"
EXPORT_INVENTORY_PATH = PACK_DIR / "export_inventory.json"

EXTRACTED_PACKAGE = "sae_concept_lab.canonical.concept_bundle"


def _load_runner():
    spec = importlib.util.spec_from_file_location("concept_bundle_conformance", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pack() -> dict:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


runner = _load_runner()
pack = _load_pack()


def test_pack_has_exactly_fifty_vectors():
    """Pinned per the dispatch and per fabf702's own commit message --
    a pack that silently grew or shrank would invalidate 'all 50 vectors
    conform' as a claim."""
    assert len(pack["vectors"]) == 50


def test_pack_schema_version_matches_extracted_codec_supported_versions():
    from sae_concept_lab.canonical.concept_bundle import SUPPORTED_SCHEMA_VERSIONS

    assert pack["schema_version"] in SUPPORTED_SCHEMA_VERSIONS


def test_all_fifty_vectors_conform_against_the_extracted_package():
    failures = runner.verify_pack(pack, package=EXTRACTED_PACKAGE)
    assert failures == [], f"{len(failures)} conformance failure(s): {failures}"


@pytest.mark.parametrize("vector", pack["vectors"], ids=lambda v: v["id"])
def test_each_vector_conforms_individually(vector):
    """The aggregate check above proves 'all 50 pass'; this parametrized
    form additionally proves 'no vector was silently skipped' -- each of
    the 50 ids actually reached the checker and reported no failure,
    visible individually in a test report rather than folded into one
    assertion."""
    single_vector_pack = {**pack, "vectors": [vector]}
    failures = runner.verify_pack(single_vector_pack, package=EXTRACTED_PACKAGE)
    assert failures == []


def test_conformance_mutation_self_test_the_checker_can_actually_fail():
    """Adversarial check on the checker itself, mirroring fabf702's own
    'the checker is tested for being able to fail' discipline: corrupt one
    expected value in a real vector and confirm verify_pack names it,
    rather than trusting a checker that always reports conformance."""
    vector = next(v for v in pack["vectors"] if v["kind"] == "codec_accept")
    mutated = json.loads(json.dumps(vector))
    # canonical_json/expected shape: corrupt whatever scalar the expected
    # side carries by flipping a string field so it disagrees with the
    # extracted package's actual (correct) output.
    if isinstance(mutated["expected"], dict) and "canonical_json" in mutated["expected"]:
        mutated["expected"]["canonical_json"] = mutated["expected"]["canonical_json"] + "TAMPERED"
    else:
        pytest.skip("first codec_accept vector's expected shape changed; not exercising this guard")
    single_vector_pack = {**pack, "vectors": [mutated]}
    failures = runner.verify_pack(single_vector_pack, package=EXTRACTED_PACKAGE)
    assert failures, "mutated vector must be reported as a failure, not silently pass"
    assert vector["id"] in failures[0]


def test_export_inventory_declares_this_extraction_standard_library_only():
    inventory = json.loads(EXPORT_INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inventory["standard_library_only"] is True
    assert inventory["third_party_runtime_dependencies"] == []


def test_runner_and_vectors_files_are_byte_identical_to_the_hashes_recorded_in_provenance():
    """Regression guard against silent drift between the copied artifacts
    on disk and what provenance/source_import.json claims about them."""
    import hashlib

    manifest = json.loads((REPO_ROOT / "provenance" / "source_import.json").read_text(encoding="utf-8"))
    extraction = next(
        e for e in manifest["extractions"] if e["extraction_id"] == "concept_bundle_contract"
    )
    recorded = {e["dest_path"]: e["sha256"] for e in extraction["extracted_artifacts"]}
    for rel_path in (
        "provenance/runtime_extractions/concept_bundle/vectors.json",
        "provenance/runtime_extractions/concept_bundle/export_inventory.json",
        "provenance/runtime_extractions/concept_bundle/concept_bundle_conformance.py",
    ):
        actual = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual == recorded[rel_path], f"{rel_path} no longer matches its recorded hash"
