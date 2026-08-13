"""CLI entry point for SAE Concept Lab (working title).

Launches entirely on CPU, no GPU weights required, BY DEFAULT: both tabs
are backed by StubConceptLabBackend and this repository's FAKE-marked
canonical concept-bundle fixtures. --qwen-backend/--gemma-backend select a
real backend instead (QwenRuntimeBackend/GemmaRuntimeBackend,
sae_concept_lab.core.*_backend) -- see README.md in this directory and
docs/tamia_launch.md for the launch commands these map to.

--mode release is a fail-closed gate, not a feature flag with an escape
hatch. It refuses to start if ANY of the following is true: the active
backend is the known StubConceptLabBackend; the active backend is a real
backend for a pairing with no attached, verified RuntimeAcceptanceRecord
(core/runtime_acceptance.py -- mechanical acceptance against real weights
is tracked independently of whether the backend's code runs at all);
--evidence-registry-root is absent/missing/unreadable/empty; or no concept
entry for a pairing is publishable against it (this build's fixtures are
always provenance=fake, so 'release' always exits non-zero without
opening a server regardless of which backend is selected). There is no
flag on this path that overrides any of these checks.
"""

from __future__ import annotations

import argparse
import sys

from sae_concept_lab.canonical.concept_bundle import (
    Exposure,
    RepositoryEvidenceRegistry,
    select_layout_entries,
)
from sae_concept_lab.core.gemma_backend import GemmaRuntimeBackend
from sae_concept_lab.core.qwen_backend import QwenRuntimeBackend
from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import ReleaseGateError, enforce_release_gate, load_entries
from sae_concept_lab.ui.app_ui import build_demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("dev", "release"),
        default="dev",
        help=(
            "'dev' (default) runs the fixture-backed UI as-is. 'release' enforces the fail-closed "
            "gate: see the module docstring above. This build's fixtures are always "
            "provenance=fake, so 'release' always exits non-zero without opening a server."
        ),
    )
    p.add_argument(
        "--evidence-registry-root",
        default=None,
        help=(
            "Path to the evidence registry directory release mode resolves evidence_refs "
            "against. Required in --mode release (refused if absent, missing, unreadable, or "
            "empty); ignored in dev mode, which never evaluates publishability."
        ),
    )
    p.add_argument("--qwen-backend", choices=("stub", "runtime"), default="stub")
    p.add_argument("--qwen-model-path", default=None, help="Local Qwen3.5-27B snapshot directory. Required if --qwen-backend runtime.")
    p.add_argument("--qwen-sae-path", default=None, help="Local Qwen-Scope layerN.sae.pt file. Required if --qwen-backend runtime.")
    p.add_argument("--qwen-layer", type=int, default=None, help="Engineering-only layer index. Required if --qwen-backend runtime.")
    p.add_argument("--qwen-device", default="cuda")
    p.add_argument("--qwen-dtype", default="bfloat16")
    p.add_argument("--qwen-expected-model-revision", default=None)
    p.add_argument("--qwen-expected-sae-revision", default=None)
    p.add_argument("--gemma-backend", choices=("stub", "runtime"), default="stub")
    p.add_argument("--gemma-model-path", default=None, help="Local gemma-3-12b-it snapshot directory. Required if --gemma-backend runtime.")
    p.add_argument("--gemma-sae-path", default=None, help="Local gemma-scope-2-12b-it-res SAE snapshot ROOT directory. Required if --gemma-backend runtime.")
    p.add_argument("--gemma-device", default="cuda")
    p.add_argument("--gemma-dtype", default="bfloat16")
    p.add_argument("--gemma-expected-model-revision", default=None)
    p.add_argument("--gemma-expected-sae-revision", default=None)
    p.add_argument("--server-name", default="127.0.0.1", help="Bind address -- localhost only by default.")
    p.add_argument("--server-port", type=int, default=7860)
    return p.parse_args(argv)


def _build_qwen_backend(args: argparse.Namespace):
    if args.qwen_backend == "stub":
        return StubConceptLabBackend()
    missing = [
        name
        for name, value in (
            ("--qwen-model-path", args.qwen_model_path),
            ("--qwen-sae-path", args.qwen_sae_path),
            ("--qwen-layer", args.qwen_layer),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"--qwen-backend runtime requires {missing}")
    return QwenRuntimeBackend(
        model_path=args.qwen_model_path,
        sae_path=args.qwen_sae_path,
        qwen_layer=args.qwen_layer,
        device=args.qwen_device,
        dtype=args.qwen_dtype,
        expected_model_revision=args.qwen_expected_model_revision,
        expected_sae_revision=args.qwen_expected_sae_revision,
    )


def _build_gemma_backend(args: argparse.Namespace):
    if args.gemma_backend == "stub":
        return StubConceptLabBackend()
    missing = [
        name
        for name, value in (
            ("--gemma-model-path", args.gemma_model_path),
            ("--gemma-sae-path", args.gemma_sae_path),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"--gemma-backend runtime requires {missing}")
    return GemmaRuntimeBackend(
        model_path=args.gemma_model_path,
        sae_path=args.gemma_sae_path,
        device=args.gemma_device,
        dtype=args.gemma_dtype,
        expected_model_revision=args.gemma_expected_model_revision,
        expected_sae_revision=args.gemma_expected_sae_revision,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gemma_entries = load_entries("gemma")
    qwen_entries = load_entries("qwen")
    try:
        gemma_backend = _build_gemma_backend(args)
        qwen_backend = _build_qwen_backend(args)
    except ValueError as exc:
        print(f"REFUSING TO LAUNCH: {exc}", file=sys.stderr)
        return 2

    try:
        enforce_release_gate(
            mode=args.mode, backend=gemma_backend, model_key="gemma",
            evidence_registry_root=args.evidence_registry_root,
        )
        enforce_release_gate(
            mode=args.mode, backend=qwen_backend, model_key="qwen",
            evidence_registry_root=args.evidence_registry_root,
        )
    except ReleaseGateError as exc:
        print(f"REFUSING TO LAUNCH: {exc}", file=sys.stderr)
        return 2

    # enforce_release_gate above already proved (per model_key,
    # independently) that at least one entry is publishable against
    # evidence_registry_root -- so this filter can never leave either tab
    # with zero entries. Dev mode is unfiltered by design (it may render
    # FAKE stubs; the permanent banner in ui/app_ui.py says so). Release
    # mode must never render an entry that has not itself individually
    # passed evaluate_publishability, even if OTHER entries for the same
    # model_key are what made the gate above pass -- a shipped FAKE
    # fixture and a genuinely ATTESTED bundle for the same pairing must
    # never be indistinguishable on screen.
    if args.mode == "release":
        registry = RepositoryEvidenceRegistry(root=args.evidence_registry_root)
        gemma_entries = tuple(
            layout.entry
            for layout in select_layout_entries(gemma_entries, exposure=Exposure.RELEASE, evidence_registry=registry)
        )
        qwen_entries = tuple(
            layout.entry
            for layout in select_layout_entries(qwen_entries, exposure=Exposure.RELEASE, evidence_registry=registry)
        )

    demo = build_demo(
        gemma_entries=gemma_entries,
        qwen_entries=qwen_entries,
        gemma_backend=gemma_backend,
        qwen_backend=qwen_backend,
        mode=args.mode,
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
