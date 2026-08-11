"""Fail-closed release gate: fake data must never reach --mode release.

The gate checks, in order: (1) the concrete type of the active backend,
refusing the known StubConceptLabBackend outright; (2) evidence_registry_root
pre-flight (absent/missing/unreadable/empty all refuse before canonical
evidence resolution is even attempted); (3) canonical publishability
(sae_concept_lab.canonical.concept_bundle.release.select_layout_entries)
against a real RepositoryEvidenceRegistry -- this product repository never
duplicates or re-implements any of (3), only wires (1) and (2), which the
canonical package has no way to know about."""

from __future__ import annotations

import json

import pytest

from sae_concept_lab.canonical.concept_bundle import Provenance
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import ReleaseGateError, enforce_release_gate, load_entries

GEMMA_ENTRIES = load_entries("gemma")
QWEN_ENTRIES = load_entries("qwen")


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


def test_shipped_entries_are_all_provenance_fake():
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        assert entry.provenance is Provenance.FAKE


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


def _attested_entry_with_resolvable_evidence(registry_root):
    """Builds a genuinely ATTESTED, schema-valid entry (never shipped by
    this repository -- constructed here only to prove the POSITIVE path
    works) and writes the one registry artifact its evidence reference
    needs, so evaluate_publishability actually resolves it. Built via the
    canonical codec (decode_entry), like every real entry, rather than
    the dataclasses directly -- decode_entry is what converts plain JSON
    strings/enums, which the dataclasses require as actual enum members."""
    from sae_concept_lab.canonical.concept_bundle import decode_entry

    artifact_type = "test_only_artifact_type"
    digest = "ab" * 32  # 64 hex chars
    (registry_root / artifact_type).mkdir(parents=True, exist_ok=True)
    (registry_root / artifact_type / f"{digest[:12]}.json").write_text(
        json.dumps({"artifact_type": artifact_type, "self_hash": f"sha256:{digest}"}), encoding="utf-8"
    )
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
            "evidence": [{"artifact_type": artifact_type, "artifact_hash": digest}],
        },
        "directions": {"amplify": direction, "suppress": None},
    })
    return decode_entry(document, where="test-only-attested-entry")


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
