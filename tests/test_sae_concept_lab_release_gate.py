"""Fail-closed release gate: fake data must never reach --mode release.

The gate checks, in order: (1) the concrete type of the active backend,
refusing the known StubConceptLabBackend outright; (2) evidence_registry_root
pre-flight (absent/missing/unreadable/empty all refuse before canonical
evidence resolution is even attempted); (3) canonical publishability
(sae_concept_lab.canonical.concept_bundle.release.select_layout_entries)
against a real RepositoryEvidenceRegistry -- this product repository never
duplicates or re-implements any of (3), only wires (1) and (2), which the
canonical package has no way to know about.

Since the final evidence contract (qwen-sae-interp checkout 3a9c153, frozen
pack 2a95a49), RepositoryEvidenceRegistry reads a registry artifact's bytes
and recomputes its content digest (sha256 over canonical JSON, self_hash
excluded) rather than trusting the artifact's self-declared self_hash field,
and publication additionally requires the reference to be written in the
full `sha256:<64 lowercase hex>` form -- a 12-hex prefix still resolves for
development but never publishes. Every registry artifact this file writes
therefore has to be a real, valid registry record (schema_version,
created_at, created_by, subject, payload), not a bare {"self_hash": ...}
stand-in."""

from __future__ import annotations

import json

import pytest

from sae_concept_lab.canonical.concept_bundle import Provenance, content_digest
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import ReleaseGateError, enforce_release_gate, load_entries

GEMMA_ENTRIES = load_entries("gemma")
QWEN_ENTRIES = load_entries("qwen")


def _registry_record(*, artifact_type="release_gate_check_artifact", payload=None):
    """A genuinely valid registry record body (no self_hash yet) -- every
    field release.py's PUBLICATION_RECORD_FIELDS requires, so a record built
    here only fails a test for the one thing that test deliberately breaks."""
    return {
        "artifact_type": artifact_type,
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "created_by": {
            "run_id": "release-gate-check-run",
            "code_commit": "0" * 40,
            "entrypoint": "pytest",
            "host": "test-host",
        },
        "subject": [],
        "payload": payload or {"note": "release-gate-check positive control"},
    }


def _write_registry_record(root, record, *, declared_self_hash=None):
    """Writes `record` (with self_hash set to `declared_self_hash`, or to its
    own correct content digest if omitted) at the path its CORRECT content
    digest addresses. Returns (full_digest, hash12) for the correct digest --
    the address the file is actually written at -- so a caller can build an
    EvidenceRef citing it, a prefix of it, or a wrong digest entirely."""
    correct_digest = content_digest(record)
    hash12 = correct_digest.removeprefix("sha256:")[:12]
    record = {**record, "self_hash": declared_self_hash or correct_digest}
    artifact_type = record["artifact_type"]
    (root / artifact_type).mkdir(parents=True, exist_ok=True)
    (root / artifact_type / f"{hash12}.json").write_text(json.dumps(record), encoding="utf-8")
    return correct_digest, hash12


class _FakeRealBackend:
    """A backend that is merely NOT StubConceptLabBackend -- stands in for
    "some future real backend" without implementing one. Never used to
    generate anything in these tests; only its type identity matters."""

    def generate(self, request):  # pragma: no cover - never called
        raise NotImplementedError


def _populated_registry_root(tmp_path):
    root = tmp_path / "registry"
    (root / "placeholder_type").mkdir(parents=True)
    (root / "placeholder_type" / ("0" * 12 + ".json")).write_text("{}", encoding="utf-8")
    return root


def test_shipped_entries_never_claim_attested_provenance():
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        assert entry.provenance is not Provenance.ATTESTED


def test_dev_mode_never_raises_with_stub_backend_and_no_evidence_registry_root():
    enforce_release_gate(mode="dev", backend=StubConceptLabBackend(), model_key="gemma")  # must not raise


def test_release_mode_raises_on_stub_backend_even_with_a_populated_registry_root(tmp_path):
    root = _populated_registry_root(tmp_path)
    with pytest.raises(ReleaseGateError, match="StubConceptLabBackend"):
        enforce_release_gate(
            mode="release", backend=StubConceptLabBackend(), model_key="gemma",
            evidence_registry_root=root,
        )


def test_release_mode_raises_when_evidence_registry_root_is_absent():
    with pytest.raises(ReleaseGateError, match="no evidence_registry_root was supplied"):
        enforce_release_gate(mode="release", backend=_FakeRealBackend(), model_key="gemma")


def test_release_mode_raises_when_evidence_registry_root_does_not_exist(tmp_path):
    missing = tmp_path / "does" / "not" / "exist"
    with pytest.raises(ReleaseGateError, match="does not exist"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma",
            evidence_registry_root=missing,
        )


def test_release_mode_raises_when_evidence_registry_root_is_a_file_not_a_directory(tmp_path):
    not_a_dir = tmp_path / "registry_file"
    not_a_dir.write_text("{}", encoding="utf-8")
    with pytest.raises(ReleaseGateError, match="not a directory"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma",
            evidence_registry_root=not_a_dir,
        )


def test_release_mode_raises_when_evidence_registry_root_is_empty(tmp_path):
    empty = tmp_path / "empty_registry"
    empty.mkdir()
    with pytest.raises(ReleaseGateError, match="is empty"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma",
            evidence_registry_root=empty,
        )


def test_release_mode_raises_when_no_entry_is_publishable_even_with_a_populated_registry_root(tmp_path):
    """This build's shipped fixtures are always provenance=fake -- a
    populated, readable, non-empty registry root is not by itself
    sufficient to publish anything."""
    root = _populated_registry_root(tmp_path)
    with pytest.raises(ReleaseGateError, match="no publishable concept entries"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma",
            evidence_registry_root=root,
        )


class _FakeRealBackendForPairing:
    """Like _FakeRealBackend, but with a real pairing name -- exercises the
    mechanical-acceptance branch enforce_release_gate skips entirely for a
    backend with no `.pairing` at all (every other test in this file)."""

    def __init__(self, pairing):
        self.pairing = pairing

    def generate(self, request):  # pragma: no cover - never called
        raise NotImplementedError


@pytest.mark.parametrize(("model_key", "shipped_layer"), [("gemma", 29), ("qwen", 38)])
def test_release_mode_refuses_a_real_backend_whose_acceptance_is_scoped_to_a_different_layer(
    tmp_path, model_key, shipped_layer,
):
    """REGRESSION: the layer-blind is_mechanically_accepted(pairing) question
    (no layer) is True for both pairings -- a record exists for each. The
    SCOPED question is what must be asked here: runtime_acceptance.py's
    Gemma record is scoped to layer 31 (job 407008), Qwen's to layer 0 (job
    406092), and this model_key's own shipped concept targets a different
    layer ({shipped_layer}) neither record covers. Fails if
    enforce_release_gate ever reverts to the unscoped call."""
    root = _populated_registry_root(tmp_path)
    with pytest.raises(ReleaseGateError, match="is scoped to layer"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackendForPairing(model_key), model_key=model_key,
            evidence_registry_root=root,
        )


def _attested_entry_citing(artifact_type, artifact_hash):
    """Builds a genuinely ATTESTED, schema-valid entry citing the given
    evidence reference -- never shipped by this repository, constructed
    here only to exercise the release gate's positive and negative paths.
    Built via the canonical codec (decode_entry), like every real entry,
    rather than the dataclasses directly -- decode_entry is what converts
    plain JSON strings/enums, which the dataclasses require as actual enum
    members."""
    from sae_concept_lab.canonical.concept_bundle import decode_entry

    spec = {"operation": "clamp", "value": 1.0, "unit": "absolute_activation", "unit_source": None}
    direction = {
        "targets": [{"sae_id": "release-gate-check-sae", "layer": 0, "feature_idx": 1, "weight": 1.0}],
        "specs": {"low": spec, "medium": spec, "high": spec},
    }
    document = json.dumps({
        "schema_version": "1.0",
        "concept_id": "release-gate-check-concept",
        "pairing_id": "release-gate-check-pairing",
        "positions": "generated_only",
        "provenance": "attested",
        "calibration_provenance": {
            "calibrated_by": "test suite",
            "calibrated_at": "2026-01-01T00:00:00+00:00",
            "evidence": [{"artifact_type": artifact_type, "artifact_hash": artifact_hash}],
        },
        "directions": {"amplify": direction, "suppress": None},
    })
    return decode_entry(document, where="test-only-attested-entry")


def _attested_entry_with_resolvable_evidence(registry_root, *, artifact_type="release_gate_check_artifact"):
    """The positive control's entry: cites a correctly-written registry
    record by its full sha256:<64 hex> content digest -- the one reference
    form that both resolves AND publishes. The record's payload carries a
    hash-named field so the release note's PAYLOAD_HASH_LABEL path is
    genuinely exercised, not merely printed as 'none'."""
    record = _registry_record(
        artifact_type=artifact_type,
        payload={"note": "release-gate-check positive control",
                 "corpus_sha256": "d" * 64},
    )
    digest, _ = _write_registry_record(registry_root, record)
    return _attested_entry_citing(artifact_type, digest)


def test_release_mode_passes_with_a_real_backend_and_a_genuinely_attested_resolvable_entry(tmp_path, monkeypatch):
    """The positive control: release mode is not unconditionally refusing
    -- it passes when every condition is genuinely satisfied. Never
    exercised by this repository's shipped fixtures, which stay FAKE by
    design; this entry is constructed in-test only."""
    import sae_concept_lab.fixtures.loader as loader_module

    root = tmp_path / "registry"
    root.mkdir()
    entry = _attested_entry_with_resolvable_evidence(root)
    monkeypatch.setattr(loader_module, "load_entries", lambda model_key: (entry,))

    enforce_release_gate(
        mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
    )  # must not raise


def test_release_diagnostics_carry_the_exact_mandatory_wording_and_labels(tmp_path, monkeypatch, capsys):
    """The mandatory release wording and its inline digest labels are
    sae_concept_lab.canonical.concept_bundle's own constants, printed
    verbatim by the adapter -- not composed or paraphrased here."""
    import sae_concept_lab.fixtures.loader as loader_module
    from sae_concept_lab.canonical.concept_bundle import (
        CONTENT_DIGEST_LABEL,
        PAYLOAD_HASH_LABEL,
        RAW_SHA256_LABEL,
        RELEASE_EVIDENCE_STATEMENT,
    )

    root = tmp_path / "registry"
    root.mkdir()
    entry = _attested_entry_with_resolvable_evidence(root)
    monkeypatch.setattr(loader_module, "load_entries", lambda model_key: (entry,))

    enforce_release_gate(
        mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
    )

    stderr = capsys.readouterr().err
    assert RELEASE_EVIDENCE_STATEMENT in stderr
    assert f"({CONTENT_DIGEST_LABEL})" in stderr
    assert f"({RAW_SHA256_LABEL})" in stderr
    assert f"({PAYLOAD_HASH_LABEL})" in stderr
    for prohibited in ("evidence verified", "artifacts verified", "fully verified"):
        assert prohibited not in stderr.lower()


def test_release_mode_raises_when_registry_record_self_hash_disagrees_with_its_own_content(tmp_path, monkeypatch):
    """TAMPERED: the artifact's own self_hash field disagrees with its
    recomputed content digest, even though the REFERENCE correctly cites
    the current content digest -- the record lies about itself, and a
    lying record must not publish just because its content happens to
    match what was cited."""
    import sae_concept_lab.fixtures.loader as loader_module

    root = tmp_path / "registry"
    root.mkdir()
    artifact_type = "release_gate_check_artifact"
    correct_digest, _ = _write_registry_record(
        root, _registry_record(artifact_type=artifact_type),
        declared_self_hash="sha256:" + "0" * 64,
    )
    entry = _attested_entry_citing(artifact_type, correct_digest)
    monkeypatch.setattr(loader_module, "load_entries", lambda model_key: (entry,))

    with pytest.raises(ReleaseGateError, match="no publishable concept entries"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
        )


def test_release_mode_raises_when_registry_record_is_missing_a_required_field(tmp_path, monkeypatch):
    """INVALID_RECORD: content integrity is not record validity -- a
    record whose content matches exactly what was cited still blocks if
    it is missing a field the release path reads (here, created_at)."""
    import sae_concept_lab.fixtures.loader as loader_module

    root = tmp_path / "registry"
    root.mkdir()
    artifact_type = "release_gate_check_artifact"
    record = _registry_record(artifact_type=artifact_type)
    del record["created_at"]
    digest, _ = _write_registry_record(root, record)
    entry = _attested_entry_citing(artifact_type, digest)
    monkeypatch.setattr(loader_module, "load_entries", lambda model_key: (entry,))

    with pytest.raises(ReleaseGateError, match="no publishable concept entries"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
        )


def test_decode_entry_rejects_an_unsafe_artifact_type():
    """An artifact_type that is not a case-canonical lowercase registry
    directory name (uppercase, a separator, a dot) is refused at decode
    time, before any registry lookup could turn it into a path component --
    the codec-level barrier this product's own entries pass through."""
    from sae_concept_lab.canonical.concept_bundle import BundleDecodeError

    for unsafe_type in ("Uppercase_Type", "has/separator", "has.dot", "../traversal"):
        with pytest.raises(BundleDecodeError):
            _attested_entry_citing(unsafe_type, "sha256:" + "0" * 64)


def test_prefix_reference_resolves_in_development_but_is_not_publishable(tmp_path, monkeypatch):
    """A 12-hex prefix reference is fully supported for development and
    inspection (RepositoryEvidenceRegistry resolves and content-verifies
    it exactly as a full reference) but is not written in the PUBLISHABLE
    form, so it must not publish."""
    from sae_concept_lab.canonical.concept_bundle import RepositoryEvidenceRegistry, resolve_all

    root = tmp_path / "registry"
    root.mkdir()
    artifact_type = "release_gate_check_artifact"
    _, hash12 = _write_registry_record(root, _registry_record(artifact_type=artifact_type))
    entry = _attested_entry_citing(artifact_type, hash12)

    registry = RepositoryEvidenceRegistry(root=root)
    (resolution,) = resolve_all(entry.calibration_provenance.evidence, registry)
    assert resolution.resolved is True
    assert resolution.is_publication_digest is False

    import sae_concept_lab.fixtures.loader as loader_module

    monkeypatch.setattr(loader_module, "load_entries", lambda model_key: (entry,))
    with pytest.raises(ReleaseGateError, match="no publishable concept entries"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
        )


def test_bare_digest_reference_without_algorithm_prefix_resolves_but_is_not_publishable(tmp_path, monkeypatch):
    """A full 64-hex digest with no `sha256:` prefix resolves as a full
    match (the bare hex is unambiguous once read), but publication
    requires the algorithm prefix explicitly -- a bare digest does not say
    what produced it."""
    import sae_concept_lab.fixtures.loader as loader_module

    root = tmp_path / "registry"
    root.mkdir()
    artifact_type = "release_gate_check_artifact"
    digest, _ = _write_registry_record(root, _registry_record(artifact_type=artifact_type))
    bare_digest = digest.removeprefix("sha256:")
    entry = _attested_entry_citing(artifact_type, bare_digest)
    monkeypatch.setattr(loader_module, "load_entries", lambda model_key: (entry,))

    with pytest.raises(ReleaseGateError, match="no publishable concept entries"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
        )


def test_release_gate_cannot_be_weakened_by_patching_the_full_digest_requirement(tmp_path, monkeypatch):
    """REQUIRE_FULL_DIGEST_FOR_PUBLICATION is a declaration, not a switch:
    evaluate_publishability applies PUBLICATION_ARTIFACT_HASH_RE
    unconditionally and never branches on the constant. Proved from the
    product side: even monkeypatching the canonical module's own constant
    to False does not let a prefix reference publish through this
    adapter, because nothing in either the canonical package or this
    product's code reads that constant to decide anything."""
    import sae_concept_lab.canonical.concept_bundle.release as canonical_release
    import sae_concept_lab.fixtures.loader as loader_module

    monkeypatch.setattr(canonical_release, "REQUIRE_FULL_DIGEST_FOR_PUBLICATION", False)

    root = tmp_path / "registry"
    root.mkdir()
    artifact_type = "release_gate_check_artifact"
    _, hash12 = _write_registry_record(root, _registry_record(artifact_type=artifact_type))
    entry = _attested_entry_citing(artifact_type, hash12)
    monkeypatch.setattr(loader_module, "load_entries", lambda model_key: (entry,))

    with pytest.raises(ReleaseGateError, match="no publishable concept entries"):
        enforce_release_gate(
            mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
        )


def test_app_main_exits_nonzero_in_release_mode_without_opening_a_server(monkeypatch):
    """End-to-end acceptance check: app.main(['--mode', 'release']) must
    refuse and return non-zero -- and must never call demo.launch()."""
    import sae_concept_lab.app as app_module

    launched = {"called": False}

    class _ExplodingDemo:
        def launch(self, *a, **k):
            launched["called"] = True

    monkeypatch.setattr(app_module, "build_demo", lambda **kwargs: _ExplodingDemo())

    exit_code = app_module.main(["--mode", "release"])
    assert exit_code != 0
    assert launched["called"] is False


def test_app_main_exits_nonzero_in_release_mode_even_with_a_populated_evidence_registry_root(monkeypatch, tmp_path):
    """End-to-end: supplying --evidence-registry-root alone is not
    sufficient -- app.py's shipped fixtures are still all FAKE."""
    import sae_concept_lab.app as app_module

    root = _populated_registry_root(tmp_path)

    launched = {"called": False}

    class _ExplodingDemo:
        def launch(self, *a, **k):
            launched["called"] = True

    monkeypatch.setattr(app_module, "build_demo", lambda **kwargs: _ExplodingDemo())

    exit_code = app_module.main(["--mode", "release", "--evidence-registry-root", str(root)])
    assert exit_code != 0
    assert launched["called"] is False
