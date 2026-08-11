"""Fail-closed release gate: fake data must never reach --mode release.

Since the P0 release-safety correction, the gate checks TWO independent
things -- the bundle's own is_synthetic/release_blocked flags, AND the
concrete type of the active backend -- and refuses if either one looks
fake. Editing a bundle's JSON flags alone is deliberately not enough."""

from __future__ import annotations

import json

import pytest

from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import (
    ReleaseGateError,
    default_bundle_path,
    enforce_release_gate,
    load_bundle,
)

GEMMA_BUNDLE = load_bundle(default_bundle_path("gemma"))
QWEN_BUNDLE = load_bundle(default_bundle_path("qwen"))


class _FakeRealBackend:
    """A backend that is merely NOT StubConceptLabBackend -- stands in for
    "some future real backend" without implementing one. Never used to
    generate anything in these tests; only its type identity matters."""

    def generate(self, request):  # pragma: no cover - never called
        raise NotImplementedError


def test_shipped_bundles_are_marked_synthetic_and_release_blocked():
    for bundle in (GEMMA_BUNDLE, QWEN_BUNDLE):
        assert bundle["is_synthetic"] is True
        assert bundle["release_blocked"] is True


def test_dev_mode_never_raises_on_synthetic_bundle_with_stub_backend():
    enforce_release_gate(GEMMA_BUNDLE, mode="dev", backend=StubConceptLabBackend())  # must not raise


def test_release_mode_raises_on_synthetic_bundle_even_with_a_non_stub_backend():
    with pytest.raises(ReleaseGateError):
        enforce_release_gate(GEMMA_BUNDLE, mode="release", backend=_FakeRealBackend())


def test_release_mode_raises_on_release_blocked_even_if_not_synthetic():
    bundle = dict(GEMMA_BUNDLE, is_synthetic=False, release_blocked=True)
    with pytest.raises(ReleaseGateError):
        enforce_release_gate(bundle, mode="release", backend=_FakeRealBackend())


def test_release_mode_raises_on_synthetic_even_if_not_release_blocked():
    bundle = dict(GEMMA_BUNDLE, is_synthetic=True, release_blocked=False)
    with pytest.raises(ReleaseGateError):
        enforce_release_gate(bundle, mode="release", backend=_FakeRealBackend())


def test_release_mode_passes_a_genuinely_real_bundle_with_a_non_stub_backend():
    real_bundle = dict(GEMMA_BUNDLE, is_synthetic=False, release_blocked=False)
    enforce_release_gate(real_bundle, mode="release", backend=_FakeRealBackend())  # must not raise


def test_release_mode_raises_on_stub_backend_even_with_an_entirely_clean_bundle():
    """The P0 regression this correction exists for: a bundle can claim
    is_synthetic=False and release_blocked=False all it likes -- if the
    backend actually wired in is still StubConceptLabBackend, release must
    still refuse. Replacing the JSON alone is not sufficient."""
    clean_bundle = dict(GEMMA_BUNDLE, is_synthetic=False, release_blocked=False)
    with pytest.raises(ReleaseGateError, match="StubConceptLabBackend"):
        enforce_release_gate(clean_bundle, mode="release", backend=StubConceptLabBackend())


def test_load_bundle_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_bundle("does/not/exist.json")


def test_load_bundle_missing_required_field_raises(tmp_path):
    bad = dict(GEMMA_BUNDLE)
    del bad["release_blocked"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="release_blocked"):
        load_bundle(path)


def test_load_bundle_zero_concepts_raises(tmp_path):
    bad = dict(GEMMA_BUNDLE, concepts=[])
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="zero concepts"):
        load_bundle(path)


def _write(tmp_path, bundle, name="bad.json"):
    path = tmp_path / name
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


@pytest.mark.parametrize("flag_name", ["is_synthetic", "release_blocked"])
def test_load_bundle_non_boolean_release_flag_raises(tmp_path, flag_name):
    bad = dict(GEMMA_BUNDLE, **{flag_name: "true"})
    with pytest.raises(ValueError, match=flag_name):
        load_bundle(_write(tmp_path, bad))


def test_load_bundle_unexpected_model_key_raises(tmp_path):
    bad = dict(GEMMA_BUNDLE, model_key="not-a-real-model")
    with pytest.raises(ValueError, match="model_key"):
        load_bundle(_write(tmp_path, bad))


def test_load_bundle_unsupported_positions_default_raises(tmp_path):
    bad = dict(GEMMA_BUNDLE, positions_default="every_token_ever")
    with pytest.raises(ValueError, match="positions_default"):
        load_bundle(_write(tmp_path, bad))


@pytest.mark.parametrize("bad_seed", [-1, "0", 1.5, True])
def test_load_bundle_invalid_seed_default_raises(tmp_path, bad_seed):
    bad = dict(GEMMA_BUNDLE, seed_default=bad_seed)
    with pytest.raises(ValueError, match="seed_default"):
        load_bundle(_write(tmp_path, bad))


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


def test_app_main_exits_nonzero_in_release_mode_even_against_a_bundle_edited_to_look_real(monkeypatch, tmp_path):
    """End-to-end version of test_release_mode_raises_on_stub_backend_even_with_an_entirely_clean_bundle:
    drives it through app.main() itself, since app.py is where the stub
    backend instances are actually constructed and handed to the gate."""
    import sae_concept_lab.app as app_module

    real_looking_gemma = dict(GEMMA_BUNDLE, is_synthetic=False, release_blocked=False)
    real_looking_qwen = dict(QWEN_BUNDLE, is_synthetic=False, release_blocked=False)
    gemma_path = _write(tmp_path, real_looking_gemma, "gemma.json")
    qwen_path = _write(tmp_path, real_looking_qwen, "qwen.json")

    launched = {"called": False}

    class _ExplodingDemo:
        def launch(self, *a, **k):
            launched["called"] = True

    monkeypatch.setattr(app_module, "build_demo", lambda **kwargs: _ExplodingDemo())

    exit_code = app_module.main(
        [
            "--mode",
            "release",
            "--gemma-bundle-path",
            str(gemma_path),
            "--qwen-bundle-path",
            str(qwen_path),
        ]
    )
    assert exit_code != 0
    assert launched["called"] is False
