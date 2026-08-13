"""Mechanically extracted runtime code for the Qwen3.5-27B and
gemma-3-12b-it intervention paths -- see BOUNDARY.md and
provenance/source_import.json's qwen_runtime_mirror/gemma_runtime_mirror
extractions for source commit, path, and hash detail.

This package is CODE ONLY. Nothing here asserts that either pairing has
been mechanically accepted against real weights -- that is a separate,
explicit fact tracked by sae_concept_lab.core.runtime_acceptance, checked
by the release gate independently of whether this code imports cleanly.

Every submodule stays import-safe on a CPU-only machine: torch/transformers/
transformer_lens/sae_lens are imported inside functions, never at module
scope, in every file except hooks.py (a verbatim mirror of
interplab/interventions/hooks.py, which does import torch/numpy/sae_lens at
its own module scope) -- callers must import hooks.py lazily, inside the
function that actually runs a real backend, never at this package's own
import time.
"""

from __future__ import annotations
