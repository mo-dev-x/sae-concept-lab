"""Deterministic local preflight for the PI demo: no GPU, no real model
weights, D: only. Boots the actual dev-mode Gradio app (StubConceptLabBackend,
exactly tests/test_sae_concept_lab_ui_smoke.py's own build) on loopback,
probes it, and shuts it down cleanly -- while separately, and without ever
needing a real backend, reporting whether a genuine Mode-A release build
is currently eligible to launch (sae_concept_lab.fixtures.loader's bounded
attested import slot).

This is NOT the Tamia GPU smoke packet (sae_concept_lab.smoke.tamia_smoke)
and never constructs QwenRuntimeBackend/GemmaRuntimeBackend -- that
real-weight acceptance evidence is already sealed (see
core/runtime_acceptance.py and the Tamia smoke job this repository already
ran). This module only proves that THIS machine, right now, can boot the
application, that it renders the correct preview/release status for
whichever mode is actually staged, and that it shuts down cleanly --
exactly the set of facts a PI demo needs confirmed the morning of.

See docs/pi_demo_runbook.md for the exact command and how this fits into
the Mode A / Mode B decision, and docs/pi_demo_scientific_status.md for
what "Mode A eligible" is, and is not, permitted to claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Structured result types -- same shape/aggregation discipline as
# sae_concept_lab.smoke.tamia_smoke's ScenarioResult/SmokePacket: every
# check is attempted and recorded regardless of what happened before it,
# and `passed` is computed once, at the end, over the complete list.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool
    summary: str
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "summary": self.summary,
            "detail": self.detail,
            "error": self.error,
        }


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    product_commit: str
    mode: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_commit": self.product_commit,
            "mode": self.mode,
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
        }


def _ok(check_id: str, summary: str, detail: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(check_id, True, summary, detail or {})


def _fail(check_id: str, summary: str, *, error: str | None = None, detail: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(check_id, False, summary, detail or {}, error)


def _run_guarded(check_id: str, fn) -> CheckResult:
    try:
        return fn()
    except Exception as exc:  # any check failure must become a recorded, non-fatal result
        return _fail(check_id, f"{check_id} raised an unexpected exception", error=f"{type(exc).__name__}: {exc}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_head_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-outside-git-checkout"


# ---------------------------------------------------------------------------
# Check: required files present
# ---------------------------------------------------------------------------

#: Named explicitly, never a directory scan -- matching this repository's
#: own rule for what may load unnamed (fixtures/loader.py's
#: _ENTRY_FILENAMES docstring). Every path is repo-root-relative.
REQUIRED_FILES: tuple[str, ...] = (
    "sae_concept_lab/app.py",
    "sae_concept_lab/ui/app_ui.py",
    "sae_concept_lab/ui/tab.py",
    "sae_concept_lab/core/logic.py",
    "sae_concept_lab/core/runtime_acceptance.py",
    "sae_concept_lab/fixtures/loader.py",
    "sae_concept_lab/fixtures/attested/README.md",
    "sae_concept_lab/fixtures/gemma/warmth.json",
    "sae_concept_lab/fixtures/gemma/formality.json",
    "sae_concept_lab/fixtures/gemma/enthusiasm.json",
    "sae_concept_lab/fixtures/gemma/caution.json",
    "sae_concept_lab/fixtures/qwen/curiosity.json",
    "sae_concept_lab/fixtures/qwen/directness.json",
    "sae_concept_lab/fixtures/qwen/playfulness.json",
    "sae_concept_lab/fixtures/qwen/skepticism.json",
    "provenance/source_import.json",
    "BOUNDARY.md",
    "docs/pi_demo_runbook.md",
    "docs/pi_demo_scientific_status.md",
)


def check_required_files_present(repo_root: Path) -> CheckResult:
    missing = [name for name in REQUIRED_FILES if not (repo_root / name).is_file()]
    if missing:
        return _fail(
            "required_files_present", f"{len(missing)} required file(s) missing",
            error=str(missing), detail={"missing": missing, "repo_root": str(repo_root)},
        )
    return _ok(
        "required_files_present", f"all {len(REQUIRED_FILES)} required files are present",
        {"checked": list(REQUIRED_FILES), "repo_root": str(repo_root)},
    )


# ---------------------------------------------------------------------------
# Check: no C:-based path anywhere this run could write to or read from
# ---------------------------------------------------------------------------

#: Checked only if actually set -- an unset variable is not a violation
#: (this machine's default may not even exist), but a SET one pointing at
#: C: is, because a preflight or the app itself could write there.
ENV_VARS_MUST_NOT_BE_ON_C: tuple[str, ...] = (
    "TEMP", "TMP", "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME", "GRADIO_TEMP_DIR",
    "PIP_CACHE_DIR", "TORCH_HOME",
)


def _is_c_drive(path_text: str) -> bool:
    try:
        drive = Path(path_text).resolve().drive
    except (OSError, ValueError):
        return False
    return drive.upper() == "C:"


def check_no_c_drive_paths(repo_root: Path, *, output_path: Path) -> CheckResult:
    problems: list[str] = []
    if _is_c_drive(str(repo_root)):
        problems.append(f"repository root is on C: ({repo_root})")
    if _is_c_drive(str(Path.cwd())):
        problems.append(f"current working directory is on C: ({Path.cwd()})")
    if _is_c_drive(str(output_path)):
        problems.append(f"--output path is on C: ({output_path})")
    for name in ENV_VARS_MUST_NOT_BE_ON_C:
        value = os.environ.get(name)
        if value and _is_c_drive(value):
            problems.append(f"{name}={value!r} is on C:")
    if problems:
        return _fail(
            "no_c_drive_paths", f"{len(problems)} C:-based path(s) found",
            error="; ".join(problems), detail={"problems": problems},
        )
    return _ok(
        "no_c_drive_paths",
        "repository root, current working directory, --output, and every checked cache/temp "
        "environment variable are off C:",
        {"repo_root": str(repo_root), "cwd": str(Path.cwd()), "output_path": str(output_path),
         "checked_env_vars": list(ENV_VARS_MUST_NOT_BE_ON_C)},
    )


# ---------------------------------------------------------------------------
# Check: current release eligibility (Mode A vs Mode B), backend-independent
# ---------------------------------------------------------------------------


def check_release_gate_status(evidence_registry_root: str | None) -> CheckResult:
    """Reports, truthfully, whether the CURRENTLY staged concept bundle
    (fixtures/loader.load_entries -- shipped FAKE fixtures plus whatever
    the bounded attested slot holds) has at least one publishable entry
    per pairing. Deliberately independent of backend type: a real
    backend's mechanical acceptance is a different, GPU-side question
    (core/runtime_acceptance.py, the Tamia smoke) that this local,
    GPU-free preflight cannot and does not evaluate. Always a PASS -- this
    is a status report, not an assertion that one specific answer is
    correct; whichever branch it reports is Mode A or Mode B, and both are
    valid outcomes the runbook has a script for."""
    from sae_concept_lab.canonical.concept_bundle import (
        Exposure,
        RepositoryEvidenceRegistry,
        select_layout_entries,
    )
    from sae_concept_lab.fixtures.loader import load_entries

    kwargs: dict[str, Any] = {}
    if evidence_registry_root is not None:
        kwargs["evidence_registry"] = RepositoryEvidenceRegistry(root=evidence_registry_root)

    publishable_counts: dict[str, int] = {}
    for model_key in ("gemma", "qwen"):
        entries = load_entries(model_key)
        selection = select_layout_entries(entries, exposure=Exposure.RELEASE, **kwargs)
        publishable_counts[model_key] = len(selection)

    total = sum(publishable_counts.values())
    if total == 0:
        return _ok(
            "release_gate_status",
            "Mode B: zero publishable (ATTESTED, evidence-verified) concepts currently staged for "
            "either pairing -- a real --mode release launch refuses",
            {"publishable_counts": publishable_counts, "evidence_registry_root": evidence_registry_root, "mode_implied": "B"},
        )
    return _ok(
        "release_gate_status",
        f"Mode A: {total} publishable concept(s) currently staged ({publishable_counts}) -- a real "
        "--mode release launch (with a mechanically-accepted real backend) would render exactly this "
        "publishable subset",
        {"publishable_counts": publishable_counts, "evidence_registry_root": evidence_registry_root, "mode_implied": "A"},
    )


def check_release_refuses_with_local_stub_backends(evidence_registry_root: str | None) -> CheckResult:
    """Hard assertion, not a status report: on THIS machine, with only
    StubConceptLabBackend available (no GPU, no real weights), --mode
    release must refuse for both pairings regardless of what the attested
    slot currently holds -- a real, mechanically-accepted backend is a
    separate, harder-to-satisfy precondition enforce_release_gate checks
    before it ever looks at bundle publishability. This is what makes
    "Mode B is GUARANTEED" true even if a malformed or half-staged Mode-A
    bundle sits in the attested slot."""
    from sae_concept_lab.core.stub_backend import StubConceptLabBackend
    from sae_concept_lab.fixtures.loader import ReleaseGateError, enforce_release_gate

    for model_key in ("gemma", "qwen"):
        try:
            enforce_release_gate(
                mode="release", backend=StubConceptLabBackend(), model_key=model_key,
                evidence_registry_root=evidence_registry_root,
            )
        except ReleaseGateError:
            continue
        return _fail(
            "release_refuses_locally",
            f"release mode did NOT refuse for model_key={model_key!r} with a local stub backend",
            error="ReleaseGateError was not raised",
        )
    return _ok(
        "release_refuses_locally",
        "release mode still refuses for both pairings using this machine's local stub backends -- a "
        "real, GPU-loaded, mechanically-accepted backend is required before release could ever open here",
        {},
    )


# ---------------------------------------------------------------------------
# Boot / HTTP / visible status / clean shutdown -- always dev mode: this is
# the one mode local Stub backends can actually boot (release mode's
# StubConceptLabBackend refusal is proven separately, above, without
# opening a server at all).
# ---------------------------------------------------------------------------


def _probe_http_200(host: str, port: int) -> CheckResult:
    import urllib.request

    url = f"http://{host}:{port}/"
    with urllib.request.urlopen(url, timeout=30) as response:  # localhost-only probe of our own just-launched server
        status = response.status
    if status != 200:
        return _fail("http_200", f"HTTP probe returned {status}, not 200", error=f"status={status}")
    return _ok("http_200", f"application responded HTTP {status} at {url}", {"url": url, "status": status})


def _check_visible_status(demo) -> CheckResult:
    rendered = json.dumps(demo.get_config_file(), default=str)
    has_banner = "PLACEHOLDER" in rendered and "NOT SCIENTIFIC EVIDENCE" in rendered
    if not has_banner:
        return _fail(
            "visible_status", "dev-mode build is missing the permanent FAKE/placeholder banner",
            error="banner text absent from the rendered component tree",
        )
    return _ok(
        "visible_status", "dev-mode build renders the permanent FAKE/placeholder banner unmistakably",
        {"mode": "dev"},
    )


def _confirm_port_released(host: str, port: int, *, attempts: int = 10, delay_seconds: float = 0.2) -> bool:
    import urllib.error
    import urllib.request

    for _ in range(attempts):
        try:
            urllib.request.urlopen(f"http://{host}:{port}/", timeout=2)
        except (OSError, urllib.error.URLError):
            return True
        time.sleep(delay_seconds)
    return False


def run_boot_http_status_shutdown(*, server_name: str, server_port: int) -> list[CheckResult]:
    from sae_concept_lab.core.stub_backend import StubConceptLabBackend
    from sae_concept_lab.fixtures.loader import load_entries
    from sae_concept_lab.ui.app_ui import build_demo

    results: list[CheckResult] = []
    host = "127.0.0.1" if server_name == "0.0.0.0" else server_name
    demo = None
    try:
        demo = build_demo(
            gemma_entries=load_entries("gemma"), qwen_entries=load_entries("qwen"),
            gemma_backend=StubConceptLabBackend(), qwen_backend=StubConceptLabBackend(), mode="dev",
        )
        demo.launch(server_name=server_name, server_port=server_port, share=False, prevent_thread_lock=True, quiet=True)
        results.append(_run_guarded("http_200", lambda: _probe_http_200(host, server_port)))
        results.append(_run_guarded("visible_status", lambda: _check_visible_status(demo)))
    except Exception as exc:  # a boot failure must become a recorded check, not an uncaught crash
        results.append(_fail("boot", "the application failed to boot for the preflight probe", error=f"{type(exc).__name__}: {exc}"))
    finally:
        if demo is not None:
            demo.close()
        released = _confirm_port_released(host, server_port)
        if released:
            results.append(_ok("clean_shutdown", "demo.close() completed and the loopback port was released"))
        else:
            results.append(_fail("clean_shutdown", "the loopback port still answers after demo.close()", error=f"http://{host}:{server_port}/ still responds"))
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_preflight_report(args: argparse.Namespace) -> PreflightReport:
    repo_root = _repo_root()
    output_path = Path(args.output)

    checks: list[CheckResult] = []
    checks.append(_run_guarded("required_files_present", lambda: check_required_files_present(repo_root)))
    checks.append(_run_guarded("no_c_drive_paths", lambda: check_no_c_drive_paths(repo_root, output_path=output_path)))
    checks.append(_run_guarded("release_gate_status", lambda: check_release_gate_status(args.evidence_registry_root)))
    checks.append(
        _run_guarded(
            "release_refuses_locally", lambda: check_release_refuses_with_local_stub_backends(args.evidence_registry_root)
        )
    )
    checks.extend(run_boot_http_status_shutdown(server_name=args.server_name, server_port=args.server_port))

    return PreflightReport(product_commit=_git_head_commit(repo_root), mode="dev", checks=tuple(checks))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--evidence-registry-root", default=None,
        help="Optional. If a Mode-A bundle has been staged in the attested import slot along with a real "
        "evidence registry, pass its root here so release_gate_status reports Mode A eligibility "
        "accurately. Omitting it still reports Mode B truthfully.",
    )
    p.add_argument("--server-name", default="127.0.0.1", help="Loopback bind address for the boot/HTTP check.")
    p.add_argument("--server-port", type=int, default=7862)
    p.add_argument("--output", default="pi_demo_preflight_result.json", help="Path to write the aggregate JSON report to.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_preflight_report(args)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_path}", file=sys.stderr)

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.check_id}: {check.summary}", file=sys.stderr)

    if report.passed:
        print("PI DEMO PREFLIGHT: ALL CHECKS PASSED", file=sys.stderr)
        return 0
    print("PI DEMO PREFLIGHT: AT LEAST ONE CHECK FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
