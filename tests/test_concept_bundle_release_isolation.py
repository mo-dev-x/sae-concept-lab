"""Proves conformance-pack fixtures cannot enter UI bundle discovery and
cannot pass either release gate this repository has:

  1. sae_concept_lab.fixtures.loader -- the UI's own gate (--mode
     release), keyed to StubConceptLabBackend's type and a bundle's
     is_synthetic/release_blocked flags, over the UI's own bundle schema
     (is_synthetic/model_key/concepts/...).
  2. sae_concept_lab.canonical.concept_bundle.release -- the extracted
     contract's own fail-closed publication gate, keyed to ATTESTED
     provenance and evidence that resolves against a real registry, over
     the contract's own schema (schema_version/concept_id/pairing_id/
     directions/...).

These are two independent mechanisms over two structurally incompatible
schemas. Neither can be satisfied by the other's data, and this task's
extraction did not connect them (see BOUNDARY.md's note that wiring is a
subsequent, bounded task) -- both facts are demonstrated here rather than
asserted.
"""

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
    default_bundle_path,
    enforce_release_gate,
    load_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = REPO_ROOT / "provenance" / "runtime_extractions" / "concept_bundle" / "vectors.json"


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
# 1. UI bundle discovery: structurally scoped away from the extraction
# ---------------------------------------------------------------------------


def test_ui_bundle_discovery_is_scoped_to_sae_concept_lab_fixtures_only():
    assert FIXTURES_DIR.name == "fixtures"
    assert FIXTURES_DIR.parent.name == "sae_concept_lab"
    for model_key in ("gemma", "qwen"):
        path = default_bundle_path(model_key)
        assert path.parent == FIXTURES_DIR
        assert "canonical" not in path.parts
        assert "provenance" not in path.parts


@pytest.mark.parametrize("vector_id,document", _decodable_documents())
def test_no_conformance_vector_document_loads_as_a_ui_bundle(vector_id, document, tmp_path):
    """The two schemas are incompatible by construction: a canonical
    document has no is_synthetic/release_blocked/model_key/concepts, all
    of which load_bundle requires. Every decodable vector document must
    be rejected if handed to the UI's own loader."""
    path = tmp_path / "not_a_ui_bundle.json"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError):
        load_bundle(path)


def test_real_ui_bundles_are_unaffected_and_still_load(tmp_path):
    """Sanity control for the test above: load_bundle DOES work on an
    actual UI bundle, so the rejections above are about schema
    incompatibility, not a broken loader."""
    for model_key in ("gemma", "qwen"):
        bundle = load_bundle(default_bundle_path(model_key))
        assert bundle["model_key"] == model_key


# ---------------------------------------------------------------------------
# 2. The UI's own release gate: unaffected by this extraction
# ---------------------------------------------------------------------------


def test_ui_release_gate_still_refuses_the_stub_backend_after_extraction():
    """Regression: this task's extraction must not have weakened the
    pre-existing UI release gate. Reproduces the P0 guarantee from
    sae_concept_lab/README.md directly against the real fixtures."""
    from sae_concept_lab.fixtures.loader import ReleaseGateError

    for model_key in ("gemma", "qwen"):
        bundle = load_bundle(default_bundle_path(model_key))
        with pytest.raises(ReleaseGateError):
            enforce_release_gate(bundle, mode="release", backend=StubConceptLabBackend())


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
