"""Mode-A bounded import slot and Mode-B guarantee -- the 2026-08-13
PI-demo dispatch. Covers fixtures/loader.py's attested import slot
(load_attested_entries, load_entries), sae_concept_lab.app's release-mode
entry filtering, and ui/app_ui.build_demo's mode-conditioned banner.

Companion to tests/test_sae_concept_lab_release_gate.py, which this file
never duplicates: the canonical release-gate machinery itself (provenance,
evidence resolution, digest form) is unchanged and already covered there.
This file only covers what is NEW: the slot's own mechanics (no-code-edit,
malformed-file robustness), and what this product's own app.py/app_ui.py
now do with a bundle that arrives through it."""

from __future__ import annotations

import json

import pytest

import sae_concept_lab.fixtures.loader as loader_module
from sae_concept_lab.canonical.concept_bundle import content_digest
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import (
    ReleaseGateError,
    enforce_release_gate,
    load_attested_entries,
    load_entries,
)


def _registry_record(*, artifact_type="pi_demo_check_artifact"):
    """A genuinely valid registry record body (no self_hash yet) -- see
    test_sae_concept_lab_release_gate.py's own _registry_record for the
    field-by-field rationale; not imported from there because no test
    file in this repository imports from another (tests/_fake_runtime.py
    is the one shared, non-test-prefixed helper module)."""
    return {
        "artifact_type": artifact_type,
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "created_by": {
            "run_id": "pi-demo-check-run",
            "code_commit": "0" * 40,
            "entrypoint": "pytest",
            "host": "test-host",
        },
        "subject": [],
        "payload": {"note": "pi-demo-check positive control"},
    }


def _write_registry_record(root, record, *, declared_self_hash=None):
    correct_digest = content_digest(record)
    hash12 = correct_digest.removeprefix("sha256:")[:12]
    record = {**record, "self_hash": declared_self_hash or correct_digest}
    artifact_type = record["artifact_type"]
    (root / artifact_type).mkdir(parents=True, exist_ok=True)
    (root / artifact_type / f"{hash12}.json").write_text(json.dumps(record), encoding="utf-8")
    return correct_digest


def _attested_document(*, concept_id, pairing_id, artifact_type, artifact_hash, feature_idx=1):
    spec = {"operation": "clamp", "value": 1.0, "unit": "absolute_activation", "unit_source": None}
    direction = {
        "targets": [{"sae_id": "pi-demo-check-sae", "layer": 0, "feature_idx": feature_idx, "weight": 1.0}],
        "specs": {"low": spec, "medium": spec, "high": spec},
    }
    return json.dumps({
        "schema_version": "1.0",
        "concept_id": concept_id,
        "pairing_id": pairing_id,
        "positions": "all",
        "provenance": "attested",
        "calibration_provenance": {
            "calibrated_by": "pi-demo readiness check",
            "calibrated_at": "2026-01-01T00:00:00+00:00",
            "evidence": [{"artifact_type": artifact_type, "artifact_hash": artifact_hash}],
        },
        "directions": {"amplify": direction, "suppress": None},
    })


class _FakeRealBackend:
    """Merely NOT StubConceptLabBackend -- stands in for a future real
    backend without implementing one, exactly test_sae_concept_lab_release_gate.py's
    own _FakeRealBackend. `pairing = None` so enforce_release_gate skips
    the mechanical-acceptance check and evaluates bundle publishability."""

    pairing = None

    def generate(self, request):  # pragma: no cover - never called
        raise NotImplementedError


@pytest.fixture
def attested_slot(tmp_path, monkeypatch):
    """Points loader_module.ATTESTED_DIR at an isolated tmp_path for the
    duration of one test -- the real, committed
    sae_concept_lab/fixtures/attested/ directory is never written to by
    any test."""
    slot = tmp_path / "attested_slot"
    (slot / "gemma").mkdir(parents=True)
    (slot / "qwen").mkdir(parents=True)
    monkeypatch.setattr(loader_module, "ATTESTED_DIR", slot)
    return slot


# ---------------------------------------------------------------------------
# The slot's own mechanics
# ---------------------------------------------------------------------------


def test_attested_slot_is_empty_in_this_repositorys_own_committed_state():
    """Today's shipped behavior (exactly 4 FAKE entries per pairing,
    nothing more) must not silently change because a staged file was
    left behind uncommitted."""
    assert load_attested_entries("gemma").entries == ()
    assert load_attested_entries("qwen").entries == ()


def test_a_valid_attested_bundle_dropped_into_the_slot_is_picked_up_with_no_code_edit(attested_slot):
    """The literal dispatch requirement: staging a well-formed file here
    -- with ZERO changes to any .py file -- makes load_entries() include
    it, additively alongside the shipped FAKE fixtures."""
    root = attested_slot.parent / "registry"
    root.mkdir()
    digest = _write_registry_record(root, _registry_record())
    document = _attested_document(
        concept_id="pi-demo-mode-a-concept", pairing_id="pi-demo-mode-a-pairing",
        artifact_type="pi_demo_check_artifact", artifact_hash=digest,
    )
    (attested_slot / "gemma" / "arrived.json").write_text(document, encoding="utf-8")

    entries = load_entries("gemma")
    arrived = [e for e in entries if e.concept_id == "pi-demo-mode-a-concept"]
    assert len(arrived) == 1
    assert arrived[0].provenance.value == "attested"
    assert any(e.concept_id == "FAKE-gemma-warmth" for e in entries)


def test_release_gate_passes_for_that_pairing_once_evidence_resolves(attested_slot):
    root = attested_slot.parent / "registry"
    root.mkdir()
    digest = _write_registry_record(root, _registry_record())
    document = _attested_document(
        concept_id="pi-demo-mode-a-concept", pairing_id="pi-demo-mode-a-pairing",
        artifact_type="pi_demo_check_artifact", artifact_hash=digest,
    )
    (attested_slot / "gemma" / "arrived.json").write_text(document, encoding="utf-8")

    enforce_release_gate(
        mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root,
    )  # must not raise


def test_a_malformed_file_in_the_slot_is_excluded_and_reported_not_raised(attested_slot, capsys):
    (attested_slot / "gemma" / "broken.json").write_text("not json{{{", encoding="utf-8")

    outcome = load_attested_entries("gemma")
    assert outcome.entries == ()
    assert len(outcome.rejected) == 1
    path, reason = outcome.rejected[0]
    assert path.name == "broken.json"
    assert reason

    # Mode B's guarantee: load_entries() must not raise, and the shipped
    # FAKE fixtures must still be present, even with a broken file sitting
    # in the attested slot.
    entries = load_entries("gemma")
    assert any(e.concept_id == "FAKE-gemma-warmth" for e in entries)
    stderr = capsys.readouterr().err
    assert "broken.json" in stderr


def test_a_schema_valid_but_tampered_evidence_bundle_decodes_but_remains_unpublishable(attested_slot):
    """Decode success is not publication: an entry can be well-formed
    enough to load (and would render in dev mode, correctly labelled by
    its own provenance) while still being refused by
    evaluate_publishability because its cited evidence artifact's own
    self_hash disagrees with its content -- the exact TAMPERED case
    test_sae_concept_lab_release_gate.py's own test already covers for the
    canonical gate directly; this proves it holds for a file staged
    through the bounded import slot too, i.e. the slot adds no override."""
    root = attested_slot.parent / "registry"
    root.mkdir()
    record = _registry_record()
    correct_digest = _write_registry_record(root, record, declared_self_hash="sha256:" + "0" * 64)
    document = _attested_document(
        concept_id="pi-demo-tampered-concept", pairing_id="pi-demo-mode-a-pairing",
        artifact_type="pi_demo_check_artifact", artifact_hash=correct_digest,
    )
    (attested_slot / "gemma" / "tampered.json").write_text(document, encoding="utf-8")

    entries = load_entries("gemma")
    tampered_entry = next(e for e in entries if e.concept_id == "pi-demo-tampered-concept")
    assert tampered_entry.provenance.value == "attested"  # decoded fine

    with pytest.raises(ReleaseGateError, match="no publishable concept entries"):
        enforce_release_gate(mode="release", backend=_FakeRealBackend(), model_key="gemma", evidence_registry_root=root)


def test_mode_b_release_still_refuses_with_stub_backend_even_with_a_publishable_bundle_staged(attested_slot):
    """Mode B's guarantee at the backend level: --mode release with
    StubConceptLabBackend refuses REGARDLESS of what the attested slot
    holds -- backend identity is checked FIRST, before bundle
    publishability is even evaluated."""
    root = attested_slot.parent / "registry"
    root.mkdir()
    digest = _write_registry_record(root, _registry_record())
    document = _attested_document(
        concept_id="pi-demo-mode-a-concept", pairing_id="pi-demo-mode-a-pairing",
        artifact_type="pi_demo_check_artifact", artifact_hash=digest,
    )
    (attested_slot / "gemma" / "arrived.json").write_text(document, encoding="utf-8")

    with pytest.raises(ReleaseGateError, match="StubConceptLabBackend"):
        enforce_release_gate(
            mode="release", backend=StubConceptLabBackend(), model_key="gemma", evidence_registry_root=root,
        )


# ---------------------------------------------------------------------------
# sae_concept_lab.app: release-mode filtering end to end
# ---------------------------------------------------------------------------


def test_app_main_release_mode_renders_only_the_publishable_entries_once_bundles_are_staged(monkeypatch, attested_slot):
    """Once a publishable bundle exists for BOTH pairings, --mode
    release opens (does not refuse) and passes build_demo ONLY the
    publishable entries -- never the shipped FAKE ones alongside them,
    even though the FAKE ones are still what load_entries() itself
    returns. No .py file was edited to stage the bundle; only
    monkeypatching the backend constructors (there is no CLI flag for a
    fake "real" backend, and this repository has no real one that does
    not require actual GPU weights)."""
    import sae_concept_lab.app as app_module

    root = attested_slot.parent / "registry"
    root.mkdir()

    gemma_digest = _write_registry_record(root, _registry_record(artifact_type="pi_demo_gemma_artifact"))
    (attested_slot / "gemma" / "arrived.json").write_text(
        _attested_document(
            concept_id="pi-demo-gemma-concept", pairing_id="pi-demo-gemma-pairing",
            artifact_type="pi_demo_gemma_artifact", artifact_hash=gemma_digest,
        ),
        encoding="utf-8",
    )
    qwen_digest = _write_registry_record(root, _registry_record(artifact_type="pi_demo_qwen_artifact"))
    (attested_slot / "qwen" / "arrived.json").write_text(
        _attested_document(
            concept_id="pi-demo-qwen-concept", pairing_id="pi-demo-qwen-pairing",
            artifact_type="pi_demo_qwen_artifact", artifact_hash=qwen_digest,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module, "_build_gemma_backend", lambda args: _FakeRealBackend())
    monkeypatch.setattr(app_module, "_build_qwen_backend", lambda args: _FakeRealBackend())

    captured = {}

    class _ExplodingDemo:
        def launch(self, *a, **k):
            pass

    def _fake_build_demo(**kwargs):
        captured.update(kwargs)
        return _ExplodingDemo()

    monkeypatch.setattr(app_module, "build_demo", _fake_build_demo)

    exit_code = app_module.main(["--mode", "release", "--evidence-registry-root", str(root)])

    assert exit_code == 0
    assert captured["mode"] == "release"
    assert {e.concept_id for e in captured["gemma_entries"]} == {"pi-demo-gemma-concept"}
    assert {e.concept_id for e in captured["qwen_entries"]} == {"pi-demo-qwen-concept"}


def test_app_main_dev_mode_renders_shipped_fake_and_staged_attested_entries_unfiltered(monkeypatch, attested_slot):
    """Dev mode's own existing behavior (render whatever load_entries()
    returns, unfiltered) is unchanged by the attested-slot machinery: a
    staged bundle appears ALONGSIDE the shipped FAKE fixtures rather than
    replacing or hiding them -- dev mode is a superset, never a filter."""
    import sae_concept_lab.app as app_module

    root = attested_slot.parent / "registry"
    root.mkdir()
    digest = _write_registry_record(root, _registry_record())
    (attested_slot / "gemma" / "arrived.json").write_text(
        _attested_document(
            concept_id="pi-demo-dev-mode-concept", pairing_id="pi-demo-mode-a-pairing",
            artifact_type="pi_demo_check_artifact", artifact_hash=digest,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "_build_gemma_backend", lambda args: StubConceptLabBackend())
    monkeypatch.setattr(app_module, "_build_qwen_backend", lambda args: StubConceptLabBackend())

    captured = {}

    class _ExplodingDemo:
        def launch(self, *a, **k):
            pass

    def _fake_build_demo(**kwargs):
        captured.update(kwargs)
        return _ExplodingDemo()

    monkeypatch.setattr(app_module, "build_demo", _fake_build_demo)

    exit_code = app_module.main([])  # default: --mode dev

    assert exit_code == 0
    assert captured["mode"] == "dev"
    gemma_ids = {e.concept_id for e in captured["gemma_entries"]}
    assert "pi-demo-dev-mode-concept" in gemma_ids
    assert "FAKE-gemma-warmth" in gemma_ids


# ---------------------------------------------------------------------------
# ui/app_ui.build_demo: mode-conditioned banner
# ---------------------------------------------------------------------------


def test_build_demo_release_mode_omits_the_placeholder_banner():
    from sae_concept_lab.ui.app_ui import build_demo

    demo = build_demo(
        gemma_entries=load_entries("gemma"), qwen_entries=load_entries("qwen"),
        gemma_backend=StubConceptLabBackend(), qwen_backend=StubConceptLabBackend(), mode="release",
    )
    rendered = json.dumps(demo.get_config_file(), default=str)
    assert "PLACEHOLDER" not in rendered
    assert "NOT SCIENTIFIC EVIDENCE" not in rendered


def test_build_demo_dev_mode_default_still_renders_the_placeholder_banner():
    from sae_concept_lab.ui.app_ui import build_demo

    demo = build_demo(
        gemma_entries=load_entries("gemma"), qwen_entries=load_entries("qwen"),
        gemma_backend=StubConceptLabBackend(), qwen_backend=StubConceptLabBackend(),
    )  # mode omitted -> default "dev", unchanged from before this dispatch
    rendered = json.dumps(demo.get_config_file(), default=str)
    assert "PLACEHOLDER" in rendered
    assert "NOT SCIENTIFIC EVIDENCE" in rendered
