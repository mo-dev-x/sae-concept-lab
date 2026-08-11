"""CLI entry point for SAE Concept Lab (working title).

Launches entirely on CPU, no GPU weights required: both tabs are backed
by StubConceptLabBackend and this repository's FAKE-marked canonical
concept-bundle fixtures. See README.md in this directory for the launch
command this maps to.

--mode release is a fail-closed gate, not a feature flag with an escape
hatch: it refuses to start if EITHER the active backend is the known
StubConceptLabBackend, OR --evidence-registry-root is absent/missing/
unreadable/empty, OR no concept entry for a pairing is publishable
against it (this build's fixtures are always provenance=fake, so
'release' always exits non-zero without opening a server). There is no
flag on this path that overrides the check -- the only way past it is a
real, non-stub backend AND a real, populated evidence registry AND a
genuinely ATTESTED entry with evidence that resolves.
"""

from __future__ import annotations

import argparse
import sys

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
    p.add_argument("--server-name", default="127.0.0.1", help="Bind address -- localhost only by default.")
    p.add_argument("--server-port", type=int, default=7860)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gemma_entries = load_entries("gemma")
    qwen_entries = load_entries("qwen")
    gemma_backend = StubConceptLabBackend()
    qwen_backend = StubConceptLabBackend()

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

    demo = build_demo(
        gemma_entries=gemma_entries,
        qwen_entries=qwen_entries,
        gemma_backend=gemma_backend,
        qwen_backend=qwen_backend,
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
