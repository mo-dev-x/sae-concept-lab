"""Production backend for gemma-3-12b-it + gemma-scope-2-12b-it-res, behind
ConceptLabBackend -- the same Protocol StubConceptLabBackend and
QwenRuntimeBackend implement.

Lazy by construction: see qwen_backend.py's module docstring -- the same
discipline applies here (torch/transformer_lens/sae_lens imported inside
generate(), model/SAE loaded on first use via _ensure_loaded()).

MECHANICALLY ACCEPTED as of qwen-sae-interp evidence commit `b6d598b`
(job 407008, both scenarios, ALL and GENERATED_ONLY -- see
core/runtime_acceptance.py for the full, independently-verified record and
its exact bounded claim). Two earlier dispatch claims for a Gemma
acceptance were rejected first -- one cited a commit ("job 407008") that
turned out to be an unrelated pytest-removal refactor; see BOUNDARY.md's
account. Feature 250 and raw clamp 5000 remain engineering acceptance
inputs only, never a public concept. If `is_mechanically_accepted("gemma")`
were ever False, `_tag()` prefixes every response with a loud, unmissable
notice. fixtures/loader.enforce_release_gate refuses this backend in
RELEASE mode regardless of acceptance status -- mechanical acceptance of
the intervention MECHANISM and public release of a SCIENTIFIC CONCEPT are,
and must remain, two separate gates.

Every fail-closed identity/subdirectory/symlink-containment guard the
extracted loader carries is preserved verbatim -- this module calls
sae_concept_lab.extracted_runtime.gemma_loader.load_gemma_it_target
unmodified and never re-implements or loosens any of it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from sae_concept_lab.canonical.concept_bundle import Operation
from sae_concept_lab.core.execution_guard import require_group_from_resolved
from sae_concept_lab.core.protocol import GenerationRequest, GenerationResult
from sae_concept_lab.core.runtime_acceptance import is_mechanically_accepted

#: Identical to core/qwen_backend.py's own constant of the same name --
#: the masking contract this describes (hooks.py's _positions_mask /
#: _PositionCounter) is shared code, not pairing-specific, so the
#: disclosure is the same statement for both backends. Printed verbatim
#: in docs/final_pairing_tamia_packet.md (qwen-sae-interp); quoted here,
#: not re-derived.
GENERATED_ONLY_FIRST_TOKEN_DISCLOSURE = (
    "positions=generated_only: the first generated token is always sampled with "
    "zero influence from the intervention -- the prefill call's masked-off "
    "positions include the position whose logits produce it. The intervention "
    "first affects the second generated token onward (docs/positions_semantics.md, "
    "qwen-sae-interp)."
)

MECHANICALLY_UNVERIFIED_TAG = "[MECHANICALLY UNVERIFIED AGAINST REAL WEIGHTS -- see core/runtime_acceptance.py]"

DEFAULT_MAX_NEW_TOKENS = 8


class GemmaRuntimeBackend:
    """model_path/sae_path are local filesystem paths: model_path is the
    gemma-3-12b-it snapshot directory, sae_path is the
    gemma-scope-2-12b-it-res SAE snapshot ROOT directory (never a single
    file -- see load_gemma_it_target's own local-snapshot-only resolver).
    expected_model_revision/expected_sae_revision feed
    targets.validate_local_snapshot_identity exactly as they do in the
    original harness."""

    pairing = "gemma"

    def __init__(
        self,
        *,
        model_path: Path | str,
        sae_path: Path | str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        expected_model_revision: str | None = None,
        expected_sae_revision: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._sae_path = sae_path
        self._device = device
        self._dtype = dtype
        self._expected_model_revision = expected_model_revision
        self._expected_sae_revision = expected_sae_revision
        self._loaded: tuple[Any, Any, str, dict[str, Any]] | None = None

    def _ensure_loaded(self):
        if self._loaded is None:
            from sae_concept_lab.extracted_runtime.gemma_loader import load_gemma_it_target

            self._loaded = load_gemma_it_target(
                self._model_path,
                self._sae_path,
                device=self._device,
                dtype=self._dtype,
                expected_model_revision=self._expected_model_revision,
                expected_sae_revision=self._expected_sae_revision,
            )
        return self._loaded

    def _tag(self, text: str) -> str:
        if is_mechanically_accepted(self.pairing):
            return text
        return f"{MECHANICALLY_UNVERIFIED_TAG} {text}"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not request.apply_intervention:
            return self._generate_baseline(request)
        return self._generate_with_intervention(request)

    def _generate_baseline(self, request: GenerationRequest) -> GenerationResult:
        if request.resolved_config is not None:
            raise ValueError(
                "GenerationRequest.apply_intervention=False must carry resolved_config=None "
                "(the baseline/Compare-original arm has no concept applied) -- got a non-None "
                "resolved_config."
            )
        import torch

        model, _sae, _hook_name, _provenance = self._ensure_loaded()
        tokens = model.to_tokens(request.prompt)
        torch.manual_seed(request.seed)
        output_ids = model.generate(
            tokens,
            max_new_tokens=request.decoding.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            do_sample=False,
            verbose=False,
        )
        text = model.tokenizer.decode(output_ids[0][tokens.shape[1] :], skip_special_tokens=True)
        return GenerationResult(text=self._tag(text), is_synthetic=False, resolved_config=None, diagnostics=None)

    def _generate_with_intervention(self, request: GenerationRequest) -> GenerationResult:
        resolved = request.resolved_config
        if resolved is None:
            raise ValueError(
                "GenerationRequest.apply_intervention=True requires a resolved_config; got None."
            )
        sae_id, layer, target = require_group_from_resolved(resolved)

        import torch

        from sae_concept_lab.extracted_runtime.diagnostics import (
            mechanical_verdict,
            wrap_hook_with_diagnostics,
        )
        from sae_concept_lab.extracted_runtime.hooks import _make_clamp_hook

        model, sae, hook_name, provenance = self._ensure_loaded()

        clamp_value = 0.0 if resolved.operation is Operation.ABLATE else float(resolved.value)
        positions = resolved.positions.value

        tokens = model.to_tokens(request.prompt)
        prompt_lengths = tokens.shape[1] if positions == "generated_only" else None

        trace: list[Any] = []
        inner_hook = _make_clamp_hook(sae, target.feature_idx, clamp_value, positions, prompt_lengths, [])
        hook_fn = wrap_hook_with_diagnostics(
            inner_hook,
            sae=sae,
            feature_index=target.feature_idx,
            mode=resolved.operation.value,
            dose_or_raw_label=(
                "ablate (always 0.0 regardless of dose)"
                if resolved.operation is Operation.ABLATE
                else f"canonical resolved value ({resolved.unit.value})"
            ),
            calibration_input=None,
            resolved_absolute_target=clamp_value,
            hook_name=hook_name,
            trace_out=trace,
        )
        torch.manual_seed(request.seed)
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            output_ids = model.generate(
                tokens,
                max_new_tokens=request.decoding.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                do_sample=False,
                verbose=False,
            )
        text = model.tokenizer.decode(output_ids[0][tokens.shape[1] :], skip_special_tokens=True)

        verdict = mechanical_verdict(trace, positions=positions)
        diagnostics: dict[str, Any] = {
            "pairing": self.pairing,
            "mechanically_accepted": is_mechanically_accepted(self.pairing),
            "requested": {
                "concept_id": resolved.concept_id,
                "pairing_id": resolved.pairing_id,
                "direction": resolved.direction.value,
                "strength": resolved.strength.value,
                "operation": resolved.operation.value,
                "positions": positions,
                "sae_id": sae_id,
                "layer": layer,
                "feature_idx": target.feature_idx,
            },
            "resolved_absolute_target": clamp_value,
            "backend_received_value": clamp_value,
            "provenance": provenance,
            "trace": [dataclasses.asdict(t) for t in trace],
            "verdict": verdict,
            "entry_audit_fingerprint": resolved.entry_audit_fingerprint,
            "execution_fingerprint": resolved.execution_fingerprint(),
            "state_fingerprint": resolved.state_fingerprint(),
        }
        if positions == "generated_only":
            diagnostics["generated_only_first_token_disclosure"] = GENERATED_ONLY_FIRST_TOKEN_DISCLOSURE
        return GenerationResult(
            text=self._tag(text),
            is_synthetic=False,
            resolved_config=resolved,
            diagnostics=diagnostics,
        )
