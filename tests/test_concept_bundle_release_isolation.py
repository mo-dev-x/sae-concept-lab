"""Proves conformance-pack fixtures cannot enter UI bundle discovery and
cannot pass either release gate this repository has:

  1. sae_concept_lab.fixtures.loader -- the UI's own gate (--mode
     release), keyed to StubConceptLabBackend's type, an explicit named
     list of this pairing's entry files (never a directory scan), and an
     evidence_registry_root.
  2. sae_concept_lab.canonical.concept_bundle.release -- the extracted
     contract's own fail-closed publication gate, keyed to ATTESTED
     provenance and evidence that resolves against a real registry.

Since the canonical UI integration, both the UI's fixtures AND the
conformance pack's vectors are canonical BundleEntry documents -- the
schema is no longer what keeps them apart. What keeps them apart now is
that codec.py's load_entry_files takes an EXPLICIT, named list of files
(never a directory scan, per its own module docstring: "a build must
name every entry it loads") -- a conformance vector's document has no
path by which it could reach that list without someone editing it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sae_concept_lab.canonical.concept_bundle import (
    RepositoryEvidenceRegistry,
    decode_entry,
    evaluate_publishability,
)
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import (
    FIXTURES_DIR,
    ReleaseGateError,
    enforce_release_gate,
    load_entries,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = REPO_ROOT / "provenance" / "runtime_extractions" / "concept_bundle" / "vectors.json"
RUNTIME_EXTRACTIONS_DIR = REPO_ROOT / "provenance" / "runtime_extractions"


def _vectors() -> list[dict]:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]


#: Kinds whose vectors are deliberately malformed/refused inputs -- their
#: `input.document` (when present) is EXPECTED to raise out of decode_entry,
#: so they must be excluded from any check that decodes-then-evaluates.
_REJECTION_KINDS = {"codec_reject", "resolve_reject", "runtime_reject"}


def _decodable_documents() -> list[tuple[str, str]]:
    """(vector_id, document_json_text) for every vector whose document is
    expected to decode successfully (excludes the *_reject kinds, whose
    documents are deliberately invalid by design)."""
    out = []
    for v in _vectors():
        if v["kind"] in _REJECTION_KINDS:
            continue
        doc = v.get("input", {}).get("document")
        if isinstance(doc, str):
            out.append((v["id"], doc))
    return out


# ---------------------------------------------------------------------------
# 1. UI bundle discovery: an explicit named list, never a directory scan
# ---------------------------------------------------------------------------


def test_ui_bundle_discovery_is_scoped_under_sae_concept_lab_fixtures_only():
    assert FIXTURES_DIR.name == "fixtures"
    assert FIXTURES_DIR.parent.name == "sae_concept_lab"
    for model_key, count in (("gemma", 4), ("qwen", 4)):
        entries = load_entries(model_key)
        assert len(entries) == count


def test_no_conformance_pack_path_is_named_in_the_explicit_entry_list():
    """The list load_entries reads from names only files under
    fixtures/<model_key>/ -- never anything under provenance/, which is
    where the conformance pack's copied vectors live."""
    from sae_concept_lab.fixtures.loader import _ENTRY_FILENAMES

    for model_key, filenames in _ENTRY_FILENAMES.items():
        for name in filenames:
            path = FIXTURES_DIR / model_key / name
            assert RUNTIME_EXTRACTIONS_DIR not in path.parents
            assert path.is_relative_to(FIXTURES_DIR)


def test_load_entries_never_scans_the_fixtures_directory(tmp_path, monkeypatch):
    """Adding an extra, unnamed file to a pairing's fixtures directory
    must not change what load_entries returns -- codec.load_entry_files
    takes an explicit list, never a directory listing."""
    before = load_entries("gemma")
    stray = FIXTURES_DIR / "gemma" / "_stray_unnamed_entry.json"
    stray.write_text("{}", encoding="utf-8")
    try:
        after = load_entries("gemma")
        assert after == before
    finally:
        stray.unlink()


@pytest.mark.parametrize("vector_id,document", _decodable_documents())
def test_no_conformance_vector_document_is_one_of_the_named_ui_entry_files(vector_id, document):
    """Even though a conformance vector's document is now schema-COMPATIBLE
    with a UI entry (both are canonical BundleEntry JSON), it is never
    reachable through UI bundle discovery: its concept_id/pairing_id never
    appear in the explicit list load_entries reads, decoded or not."""
    vector_entry = decode_entry(document, where=vector_id)
    for model_key in ("gemma", "qwen"):
        for shipped_entry in load_entries(model_key):
            assert (vector_entry.concept_id, vector_entry.pairing_id) != (
                shipped_entry.concept_id,
                shipped_entry.pairing_id,
            )


# ---------------------------------------------------------------------------
# 2. The UI's own release gate: unaffected by this extraction
# ---------------------------------------------------------------------------


def test_ui_release_gate_still_refuses_the_stub_backend_after_extraction():
    """Regression: this task's extraction must not have weakened the
    pre-existing UI release gate."""
    for model_key in ("gemma", "qwen"):
        with pytest.raises(ReleaseGateError):
            enforce_release_gate(mode="release", backend=StubConceptLabBackend(), model_key=model_key)


# ---------------------------------------------------------------------------
# 3. The extracted contract's own release gate: refuses every conformance
#    vector against a real (here: empty) repository registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vector_id,document", _decodable_documents())
def test_no_conformance_vector_publishes_against_a_real_repository_registry(vector_id, document, tmp_path):
    """Reproduces the canonical repository's own documented guarantee
    (fabf702: 'neither can publish in a real build... a test asserts
    exactly that, by running every decodable vector document through the
    real RepositoryEvidenceRegistry') against THIS repository's extracted
    copy. tmp_path/registry is empty -- structurally the same situation
    as this product repository, which ships no registry data at all."""
    entry = decode_entry(document, where=vector_id)
    registry = RepositoryEvidenceRegistry(root=tmp_path / "registry")
    decision = evaluate_publishability(entry, evidence_registry=registry)
    assert decision.publishable is False, (
        f"{vector_id} unexpectedly publishable against an empty real registry"
    )


def test_release_gate_registry_default_is_not_accidentally_satisfied_in_this_repository():
    """The mechanical consequence recorded in provenance's
    mechanical_adaptations (evidence.py's REGISTRY_ROOT resolves to
    sae_concept_lab/registry here, not the real registry/ tree): confirm
    that path does not exist, so RepositoryEvidenceRegistry()'s own
    default -- used with no explicit root -- is fail-closed in this repo
    by construction, not by coincidence."""
    from sae_concept_lab.canonical.concept_bundle.evidence import REGISTRY_ROOT

    assert not REGISTRY_ROOT.exists()
