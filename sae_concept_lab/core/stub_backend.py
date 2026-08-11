"""Deterministic, GPU-free stand-in for a real backend.

Every response is templated and tagged with FAKE_TAG so it can never be
mistaken for a real generation, even out of context (a screenshot, a bug
report, a copy-pasted transcript). This is a deliberate design choice,
not a placeholder to "improve" later: a stub that produces plausible
prose is a stub that will eventually fool someone, and this project has
already paid for that mistake once (see gemma3_sweep.py's MAX_ACT_APPROX_
CAVEAT history) in a different guise -- a number that looked real enough
to be trusted was worse than one that was visibly a placeholder.

Determinism: `generate()` is a pure function of its `GenerationRequest`
(history, prompt, model_key, decoding, seed, apply_intervention,
resolved_config) -- no `random`, no `time`, no hidden state. Same request
in, byte-identical result out, every time, on any machine.
"""

from __future__ import annotations

import hashlib
import json

from sae_concept_lab.core.protocol import GenerationRequest, GenerationResult

FAKE_TAG = "[FAKE STUB -- UI TEST ONLY]"


class StubConceptLabBackend:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        digest = _request_digest(request)
        turn_index = len(request.history) // 2

        if request.apply_intervention:
            cfg = request.resolved_config
            if cfg is None:
                raise ValueError(
                    "GenerationRequest.apply_intervention=True requires a resolved_config; "
                    "got None. This is a wiring bug in the caller, not a data problem."
                )
            body = (
                f"turn {turn_index} | concept={cfg.concept_id} | direction={cfg.direction} | "
                f"strength={cfg.strength_level} (coef={cfg.strength_coefficient}) | "
                f"digest={digest}"
            )
        else:
            if request.resolved_config is not None:
                raise ValueError(
                    "GenerationRequest.apply_intervention=False must carry resolved_config=None "
                    "(the baseline arm has no concept applied) -- got a non-None resolved_config."
                )
            body = f"turn {turn_index} | baseline, no intervention applied | digest={digest}"

        return GenerationResult(
            text=f"{FAKE_TAG} {body}",
            is_synthetic=True,
            resolved_config=request.resolved_config,
        )


def _request_digest(request: GenerationRequest) -> str:
    """Short, deterministic fingerprint of everything that should make two
    responses differ (history/prompt/seed/decoding) -- lets a manual
    tester see at a glance whether two turns were actually given the same
    inputs, without leaking anything that looks like a real activation
    value."""
    basis = "\x00".join(
        [
            request.prompt,
            "|".join(f"{role}:{text}" for role, text in request.history),
            str(request.seed),
            json.dumps(request.decoding, sort_keys=True),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:10]
