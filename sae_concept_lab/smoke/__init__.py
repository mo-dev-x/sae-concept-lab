"""Tamia product-integration smoke packet.

This package proves that the extracted, mechanically-accepted runtime
backends (QwenRuntimeBackend, GemmaRuntimeBackend -- core/qwen_backend.py,
core/gemma_backend.py) actually run, through the SAME canonical
resolution -> execution-guard -> backend-translation path the application
itself uses, on Tamia's real GPU hardware and real weights.

This is NOT a second scientific acceptance harness -- the mechanical
acceptance claim it exercises was already sealed and imported by
core/runtime_acceptance.py from qwen-sae-interp evidence commit `b6d598b`
(see that module and BOUNDARY.md). This package never re-derives, widens,
or re-litigates that claim; it only proves the PRODUCT's own adapters
(backends, execution_guard, core.logic, ui/app_ui.build_demo) correctly
carry it through to a real generation.

Every concept-bundle entry this package constructs (`entries.py`) is built
directly in Python and is deliberately NEVER added to
`sae_concept_lab/fixtures/{gemma,qwen}/*.json` or to
`fixtures/loader._ENTRY_FILENAMES` -- so it can never enter fixture
discovery, never render in the Gradio UI, and never reach the release
gate's publishability check (which only ever evaluates
`fixtures.loader.load_entries()`'s explicit file list). See
`entries.py`'s own docstring and `tests/test_tamia_smoke.py`'s
`test_smoke_entries_never_enter_fixture_discovery_or_release_gate` for the
mechanical proof of that isolation.

`pi_demo_preflight.py` is a SEPARATE, unrelated smoke check added for the
2026-08-13 PI-demo dispatch: a local, GPU-free, D:-only preflight that
boots the real dev-mode app with StubConceptLabBackend, HTTP-probes it,
checks the currently staged release-eligibility status, and shuts it down
cleanly. It never touches Tamia, never constructs a real backend, and
never depends on anything in this module. See docs/demo_runbook.md.
"""

from __future__ import annotations
