"""sae_concept_lab.smoke.pi_demo_preflight: the deterministic local,
GPU-free, D:-only preflight for the PI demo (2026-08-13 dispatch). Covers
every check function individually (files, C:-drive avoidance, release
status, boot/HTTP/shutdown) and the CLI's aggregation/exit-code/JSON
behavior -- mirroring tests/test_tamia_smoke.py's own split between
CPU-safe orchestration tests and the one real (but still CPU/no-model)
boot test."""

from __future__ import annotations

import json

import pytest

from sae_concept_lab.smoke import pi_demo_preflight


def _repo_root():
    return pi_demo_preflight._repo_root()


# ---------------------------------------------------------------------------
# required files
# ---------------------------------------------------------------------------


def test_required_files_present_passes_against_the_real_repository():
    result = pi_demo_preflight.check_required_files_present(_repo_root())
    assert result.passed is True


def test_required_files_present_fails_and_names_the_missing_file(tmp_path):
    result = pi_demo_preflight.check_required_files_present(tmp_path)  # an empty directory
    assert result.passed is False
    assert result.detail["missing"] == list(pi_demo_preflight.REQUIRED_FILES)


# ---------------------------------------------------------------------------
# no C:-based paths
# ---------------------------------------------------------------------------


def test_no_c_drive_paths_passes_when_everything_checked_is_off_c(tmp_path, monkeypatch):
    for name in pi_demo_preflight.ENV_VARS_MUST_NOT_BE_ON_C:
        monkeypatch.delenv(name, raising=False)
    result = pi_demo_preflight.check_no_c_drive_paths(tmp_path, output_path=tmp_path / "out.json")
    assert result.passed is True


def test_no_c_drive_paths_fails_when_repo_root_is_on_c():
    result = pi_demo_preflight.check_no_c_drive_paths(
        pi_demo_preflight.Path("C:/pretend-repo-root"), output_path=pi_demo_preflight.Path("D:/out.json"),
    )
    assert result.passed is False
    assert "repository root" in result.error


def test_no_c_drive_paths_fails_when_output_path_is_on_c(tmp_path):
    result = pi_demo_preflight.check_no_c_drive_paths(
        tmp_path, output_path=pi_demo_preflight.Path("C:/pretend-output/result.json"),
    )
    assert result.passed is False
    assert "--output path" in result.error


def test_no_c_drive_paths_fails_when_a_checked_env_var_points_at_c(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "C:/pretend-cache/hf")
    result = pi_demo_preflight.check_no_c_drive_paths(tmp_path, output_path=tmp_path / "out.json")
    assert result.passed is False
    assert "HF_HOME" in result.error


def test_no_c_drive_paths_ignores_an_unset_env_var(tmp_path, monkeypatch):
    for name in pi_demo_preflight.ENV_VARS_MUST_NOT_BE_ON_C:
        monkeypatch.delenv(name, raising=False)
    result = pi_demo_preflight.check_no_c_drive_paths(tmp_path, output_path=tmp_path / "out.json")
    assert result.passed is True


# ---------------------------------------------------------------------------
# release status (informational + hard local-refusal assertion)
# ---------------------------------------------------------------------------


def test_release_gate_status_reports_mode_b_against_the_shipped_repository():
    result = pi_demo_preflight.check_release_gate_status(None)
    assert result.passed is True
    assert result.detail["mode_implied"] == "B"
    assert result.detail["publishable_counts"] == {"gemma": 0, "qwen": 0}


def test_release_refuses_with_local_stub_backends_passes_against_the_shipped_repository():
    result = pi_demo_preflight.check_release_refuses_with_local_stub_backends(None)
    assert result.passed is True


# ---------------------------------------------------------------------------
# boot / HTTP / visible status / clean shutdown -- one real (CPU-only, no
# GPU, no model weights) Gradio launch, matching tests/test_tamia_smoke.py's
# own isolated run_application_smoke test.
# ---------------------------------------------------------------------------


def test_run_boot_http_status_shutdown_passes_end_to_end():
    results = pi_demo_preflight.run_boot_http_status_shutdown(server_name="127.0.0.1", server_port=7863)
    by_id = {r.check_id: r for r in results}
    assert set(by_id) == {"http_200", "visible_status", "clean_shutdown"}
    for check_id, result in by_id.items():
        assert result.passed is True, f"{check_id} failed: {result.error}"


# ---------------------------------------------------------------------------
# CLI: aggregation, JSON output, exit codes
# ---------------------------------------------------------------------------


def test_main_writes_json_and_returns_zero_when_every_check_passes(tmp_path, monkeypatch):
    for name in pi_demo_preflight.ENV_VARS_MUST_NOT_BE_ON_C:
        monkeypatch.delenv(name, raising=False)
    output_path = tmp_path / "result.json"
    exit_code = pi_demo_preflight.main(["--server-port", "7864", "--output", str(output_path)])
    assert exit_code == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["mode"] == "dev"
    assert {c["check_id"] for c in report["checks"]} >= {
        "required_files_present", "no_c_drive_paths", "release_gate_status",
        "release_refuses_locally", "http_200", "visible_status", "clean_shutdown",
    }


def test_main_returns_nonzero_and_still_writes_json_when_a_check_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "C:/pretend-cache/hf")
    output_path = tmp_path / "result.json"
    exit_code = pi_demo_preflight.main(["--server-port", "7865", "--output", str(output_path)])
    assert exit_code == 1

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    failed = [c for c in report["checks"] if not c["passed"]]
    assert any(c["check_id"] == "no_c_drive_paths" for c in failed)
    # A later, unrelated check succeeding must not mask the earlier failure.
    assert any(c["check_id"] == "http_200" and c["passed"] for c in report["checks"])


def test_a_later_successful_check_cannot_mask_an_earlier_failure(monkeypatch):
    """Same aggregation discipline as tamia_smoke.SmokePacket.passed:
    computed once, at the end, over the complete list."""
    monkeypatch.setattr(
        pi_demo_preflight, "check_required_files_present",
        lambda repo_root: pi_demo_preflight._fail("required_files_present", "forced failure for this test"),
    )
    args = pi_demo_preflight.parse_args(["--server-port", "7866"])
    report = pi_demo_preflight.build_preflight_report(args)
    assert report.passed is False
    assert any(c.check_id == "required_files_present" and not c.passed for c in report.checks)
    assert any(c.passed for c in report.checks if c.check_id != "required_files_present")


@pytest.mark.parametrize("bad_arg", ["not-an-int"])
def test_server_port_must_be_an_int(bad_arg):
    with pytest.raises(SystemExit):
        pi_demo_preflight.parse_args(["--server-port", bad_arg])
