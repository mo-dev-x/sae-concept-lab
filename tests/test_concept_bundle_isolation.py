"""Proves sae_concept_lab.canonical.concept_bundle has no import-time
dependency on qwen-sae-interp/interplab and no third-party runtime
dependency -- the two isolation guarantees the dispatch requires, checked
both statically (AST over the extracted source) and dynamically (a fresh
subprocess import, checking what actually landed in sys.modules).

Mirrors the discipline fabf702's own commit message describes for the
canonical repository ("a test greps the runner for all of those and
another confirms in a subprocess that importing the contract pulls in
neither torch nor numpy") -- reproduced here against the extracted copy
rather than trusted by inheritance from the canonical side.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

CANONICAL_DIR = Path(__file__).resolve().parents[1] / "sae_concept_lab" / "canonical" / "concept_bundle"

MODULE_FILES = (
    "__init__.py",
    "errors.py",
    "schema.py",
    "codec.py",
    "runtime.py",
    "resolver.py",
    "evidence.py",
    "release.py",
)

FORBIDDEN_MODULE_ROOTS = ("interplab", "torch", "numpy", "sae_lens", "transformer_lens", "gradio")


def _absolute_import_roots(path: Path) -> set[str]:
    """Every non-relative top-level import name a module makes. Relative
    imports (`from .schema import ...`, level > 0) are excluded -- those
    are internal cross-references within the extracted package, not
    external dependencies."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import inside the package
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _all_module_import_roots() -> set[str]:
    roots: set[str] = set()
    for name in MODULE_FILES:
        roots |= _absolute_import_roots(CANONICAL_DIR / name)
    roots.discard("__future__")
    return roots


def test_no_absolute_import_of_interplab_or_qwen_sae_interp_anywhere_in_the_extraction():
    roots = _all_module_import_roots()
    for forbidden in FORBIDDEN_MODULE_ROOTS:
        assert forbidden not in roots, f"{forbidden!r} imported absolutely by the extracted package"


def test_every_absolute_import_is_standard_library():
    roots = _all_module_import_roots()
    non_stdlib = sorted(r for r in roots if r not in sys.stdlib_module_names)
    assert non_stdlib == [], f"non-standard-library import(s) found: {non_stdlib}"


def test_no_relative_import_escapes_the_package_with_multiple_leading_dots():
    """A relative import with level > 1 (e.g. `from ..foo import bar`)
    would reach OUTSIDE concept_bundle/ into sae_concept_lab/canonical/ or
    sae_concept_lab/ itself -- which would silently reintroduce a coupling
    this extraction must not have. Every internal cross-reference must be
    level == 1 (same-package)."""
    for name in MODULE_FILES:
        tree = ast.parse((CANONICAL_DIR / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, (
                    f"{name}: relative import with level {node.level} escapes the "
                    f"concept_bundle package (from {'.' * node.level}{node.module or ''} import ...)"
                )


def test_subprocess_import_never_loads_interplab_or_any_third_party_package():
    """Dynamic confirmation, not just static: import the extracted package
    in a fresh interpreter and inspect sys.modules afterward. Static AST
    analysis cannot see a dependency pulled in transitively through
    something dynamic (getattr-based imports, importlib.import_module
    calls); this check would catch that class of gap too."""
    code = (
        "import sys, json\n"
        "import sae_concept_lab.canonical.concept_bundle\n"
        "forbidden = ('interplab', 'torch', 'numpy', 'sae_lens', 'transformer_lens')\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in forbidden)\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    import json

    loaded_forbidden = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded_forbidden == [], f"forbidden module(s) loaded transitively: {loaded_forbidden}"


def test_subprocess_import_of_extracted_package_alone_never_loads_gradio():
    """The extracted contract must be importable with no UI dependency
    present at all -- a future consumer that only needs the contract
    (e.g. a non-Gradio backend, or a CLI) must not be forced to install
    gradio merely to import sae_concept_lab.canonical.concept_bundle."""
    code = "import sys; import sae_concept_lab.canonical.concept_bundle; print('gradio' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_extracted_package_requires_no_gpu_no_clock_no_random_no_network():
    """Static discipline check mirroring the canonical repository's own
    ('offline, deterministic, cpu-only... a test greps the runner for all
    of those'): none of the eight files import time, random, socket, or
    any GPU-related module."""
    disallowed = {"time", "random", "socket", "torch", "cuda"}
    roots = _all_module_import_roots()
    hit = disallowed & roots
    assert not hit, f"disallowed non-deterministic/network/GPU import(s): {hit}"
