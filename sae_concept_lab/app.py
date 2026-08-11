"""CLI entry point for SAE Concept Lab (working title).

Launches entirely on CPU, no GPU weights required: both tabs are backed
by StubConceptLabBackend and the two FAKE-marked fixtures/*.json bundles.
See README.md in this directory for the launch command this maps to.

--mode release is a fail-closed gate, not a feature flag with an escape
hatch: it refuses to start if EITHER the active backend is the known
StubConceptLabBackend OR the loaded bundle is synthetic/release_blocked
(both shipped fixtures always are, and both tabs are always backed by the
stub backend in this build). There is no flag on this path that overrides
the check -- the only way past it is to load a bundle that is genuinely
neither synthetic nor release_blocked AND wire in a real, non-stub
backend. Editing a bundle's JSON flags alone is not sufficient.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sae_concept_lab.core.stub_backend import StubConceptLabBackend
from sae_concept_lab.fixtures.loader import ReleaseGateError, default_bundle_path, enforce_release_gate, load_bundle
from sae_concept_lab.ui.app_ui import build_demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("dev", "release"),
        default="dev",
        help=(
            "'dev' (default) runs the fixture-backed UI as-is. 'release' enforces the fail-closed "
            "gate: it refuses to launch if the active backend is the stub backend OR the loaded "
            "bundle is synthetic or release_blocked -- this build always uses the stub backend, "
            "so 'release' always exits non-zero without opening a server, regardless of bundle flags."
        ),
    )
    p.add_argument("--gemma-bundle-path", default=str(default_bundle_path("gemma")))
    p.add_argument("--qwen-bundle-path", default=str(default_bundle_path("qwen")))
    p.add_argument("--server-name", default="127.0.0.1", help="Bind address -- localhost only by default.")
    p.add_argument("--server-port", type=int, default=7860)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gemma_bundle = load_bundle(Path(args.gemma_bundle_path))
    qwen_bundle = load_bundle(Path(args.qwen_bundle_path))
    gemma_backend = StubConceptLabBackend()
    qwen_backend = StubConceptLabBackend()

    try:
        enforce_release_gate(gemma_bundle, mode=args.mode, backend=gemma_backend)
        enforce_release_gate(qwen_bundle, mode=args.mode, backend=qwen_backend)
    except ReleaseGateError as exc:
        print(f"REFUSING TO LAUNCH: {exc}", file=sys.stderr)
        return 2

    demo = build_demo(
        gemma_bundle=gemma_bundle,
        qwen_bundle=qwen_bundle,
        gemma_backend=gemma_backend,
        qwen_backend=qwen_backend,
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
