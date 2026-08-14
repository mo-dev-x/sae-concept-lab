"""Production backend for Qwen3.5-27B + Qwen-Scope, behind
ConceptLabBackend -- the same Protocol StubConceptLabBackend implements.

Lazy by construction: the constructor stores plain configuration only;
`torch`/`transformers` are imported inside `generate()`, and the model/SAE
are loaded on first use (`_ensure_loaded`), never at import time or
construction time. Importing this module, or constructing this class,
never touches a GPU or reads a model file.

MECHANICALLY ACCEPTED as of qwen-sae-interp evidence commit `b6d598b`
(job 406092's two Qwen scenarios, ALL and GENERATED_ONLY -- see
core/runtime_acceptance.py for the full, independently-verified record and
its exact bounded claim). Layer 0 and feature 4096 remain ENGINEERING-ONLY:
this acceptance is mechanical only (an intervention reached the hook and
moved the residual), never a scientific or concept claim. If
`is_mechanically_accepted("qwen")` were ever False (e.g. the record is
reset in a future revision), `_tag()` prefixes every response with a loud,
unmissable notice -- dev mode may still run the mechanism, but never
silently. fixtures/loader.enforce_release_gate refuses this backend in
RELEASE mode regardless of acceptance status, independent of the existing
StubConceptLabBackend-type and ATTESTED-evidence checks: mechanical
acceptance of the intervention MECHANISM and public release of a
SCIENTIFIC CONCEPT are, and must remain, two separate gates.

Every scientific-identity/fail-closed guard the extracted loader carries
(TargetIdentityMismatch, IdentityUnverified, the SAE shape/k/subdirectory
validators) is preserved verbatim -- this module calls
sae_concept_lab.extracted_runtime.qwen_loader.load_qwen_target unmodified
and never re-implements or loosens any of it.

SCIENTIFIC-IDENTITY GATE (core/scientific_identity.py). This backend already
refused a resolved layer that disagreed with its OWN CONFIGURED `qwen_layer`.
That check is retained and is not sufficient: it compares a request against
another statement of intent, never against what the loader reported it
produced. `_generate_with_intervention` now additionally compares the
resolved layer against the LOADED layer (LoadedLayerIdentityMismatch),
reports the loaded identity beside the requested one under unambiguous keys,
and tags any result whose loaded identity is not the certified primary as an
engineering demonstration. Note the standing consequence recorded in
scientific_identity.py: the extracted Qwen loader records release/sae_id as
literal None, so a Qwen run is never science-attributed by this gate.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from sae_concept_lab.canonical.concept_bundle import Operation
from sae_concept_lab.core.execution_guard import require_group_from_resolved
from sae_concept_lab.core.protocol import GenerationRequest, GenerationResult
from sae_concept_lab.core.runtime_acceptance import is_mechanically_accepted
from sae_concept_lab.core.scientific_identity import (
    ENGINEERING_DEMONSTRATION_TAG,
    ScientificAttributionVerdict,
    evaluate_science_attribution,
    loaded_identity_from_provenance,
    require_loaded_layer_matches_request,
)

#: Printed verbatim in docs/final_pairing_tamia_packet.md (qwen-sae-interp)
#: -- quoted here, not re-derived, because the MECHANISM this describes
#: (find_first_disappearance_boundary/mechanical_verdict's own
#: call_index==0 exemption) is extracted verbatim and already enforces it;
#: this constant only makes the same fact visible in the product's own
#: diagnostics output, in the source repository's own words.
GENERATED_ONLY_FIRST_TOKEN_DISCLOSURE = (
    "positions=generated_only: the first generated token is always sampled with "
    "zero influence from the intervention -- the prefill call's masked-off "
    "positions include the position whose logits produce it. The intervention "
    "first affects the second generated token onward (docs/positions_semantics.md, "
    "qwen-sae-interp)."
)

MECHANICALLY_UNVERIFIED_TAG = "[MECHANICALLY UNVERIFIED AGAINST REAL WEIGHTS -- see core/runtime_acceptance.py]"

DEFAULT_MAX_NEW_TOKENS = 8


class QwenRuntimeBackend:
    """model_path/sae_path are local filesystem paths (never a Hub ref --
    matching the extracted loader's own offline-only contract).
    expected_model_revision/expected_sae_revision feed
    targets.validate_local_snapshot_identity exactly as they do in the
    original harness; omitting them requires model_path/sae_path to follow
    the standard huggingface_hub cache layout, or loading fails closed with
    IdentityUnverified."""

    pairing = "qwen"

    def __init__(
        self,
        *,
        model_path: Path | str,
        sae_path: Path | str,
        qwen_layer: int,
        device: str = "cuda",
        dtype: str = "bfloat16",
        expected_model_revision: str | None = None,
        expected_sae_revision: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._sae_path = sae_path
        self._qwen_layer = qwen_layer
        self._device = device
        self._dtype = dtype
        self._expected_model_revision = expected_model_revision
        self._expected_sae_revision = expected_sae_revision
        self._loaded: tuple[Any, Any, Any, str, dict[str, Any]] | None = None

    def _ensure_loaded(self):
        if self._loaded is None:
            from sae_concept_lab.extracted_runtime.qwen_loader import load_qwen_target

            self._loaded = load_qwen_target(
                self._model_path,
                self._sae_path,
                layer=self._qwen_layer,
                device=self._device,
                dtype=self._dtype,
                expected_model_revision=self._expected_model_revision,
                expected_sae_revision=self._expected_sae_revision,
            )
        return self._loaded

    def _tag(self, text: str, attribution: ScientificAttributionVerdict | None = None) -> str:
        """Two INDEPENDENT tags -- see core/gemma_backend.py's identical
        method for why they are not collapsed into one. Mechanical acceptance
        (the mechanism ran against real weights) and scientific attribution
        (the SAE loaded is the certified primary) are separate claims, and a
        single tag would end up asserting the more flattering of the two."""
        if attribution is not None and not attribution.science_attributed:
            text = f"{ENGINEERING_DEMONSTRATION_TAG} {text}"
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
        from transformers import AutoTokenizer

        hf_model, _text_decoder, _sae, _hook_identifier, _provenance = self._ensure_loaded()
        tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        inputs = tokenizer(request.prompt, return_tensors="pt").to(self._device)
        torch.manual_seed(request.seed)
        with torch.no_grad():
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=request.decoding.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                do_sample=False,
            )
        text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return GenerationResult(text=self._tag(text), is_synthetic=False, resolved_config=None, diagnostics=None)

    def _generate_with_intervention(self, request: GenerationRequest) -> GenerationResult:
        resolved = request.resolved_config
        if resolved is None:
            raise ValueError(
                "GenerationRequest.apply_intervention=True requires a resolved_config; got None."
            )
        sae_id, layer, target = require_group_from_resolved(resolved)
        if layer != self._qwen_layer:
            raise ValueError(
                f"resolved target layer {layer} does not match this backend's configured "
                f"qwen_layer {self._qwen_layer} -- refusing a cross-layer mismatch between the "
                f"resolved control state and this backend's own configuration."
            )

        import torch
        from transformers import AutoTokenizer

        from sae_concept_lab.extracted_runtime.diagnostics import (
            mechanical_verdict,
            wrap_hook_with_diagnostics,
        )
        from sae_concept_lab.extracted_runtime.hooks import _make_clamp_hook
        from sae_concept_lab.extracted_runtime.qwen_loader import (
            get_qwen_decoder_layer,
            register_qwen_raw_hook,
        )

        hf_model, text_decoder, sae, hook_identifier, provenance = self._ensure_loaded()

        # THE GATE. The `layer != self._qwen_layer` check above compares the
        # request against this backend's own CONFIGURATION -- two statements
        # of intent. `provenance` is the first thing here that reports what
        # was actually LOADED, so the comparison that matters happens now,
        # before a hook is built.
        loaded_identity = loaded_identity_from_provenance(provenance)
        require_loaded_layer_matches_request(
            pairing=self.pairing,
            requested_layer=layer,
            requested_sae_id=sae_id,
            feature_idx=target.feature_idx,
            loaded=loaded_identity,
        )
        attribution = evaluate_science_attribution(pairing=self.pairing, loaded=loaded_identity)

        clamp_value = 0.0 if resolved.operation is Operation.ABLATE else float(resolved.value)
        positions = resolved.positions.value

        tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        inputs = tokenizer(request.prompt, return_tensors="pt").to(self._device)
        prompt_lengths = inputs["input_ids"].shape[1] if positions == "generated_only" else None

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
            hook_name=hook_identifier,
            trace_out=trace,
        )
        decoder_layer = get_qwen_decoder_layer(text_decoder, self._qwen_layer)
        handle = register_qwen_raw_hook(decoder_layer, hook_fn)
        try:
            torch.manual_seed(request.seed)
            with torch.no_grad():
                output_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=request.decoding.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                    do_sample=False,
                )
        finally:
            handle.remove()
        text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        verdict = mechanical_verdict(trace, positions=positions)
        diagnostics: dict[str, Any] = {
            "pairing": self.pairing,
            "mechanically_accepted": is_mechanically_accepted(self.pairing),
            "claim_scope": attribution.claim_scope,
            "science_attributed": attribution.science_attributed,
            "scientific_attribution": attribution.as_dict(),
            # See core/gemma_backend.py for why every identity key is now
            # spelled requested_* or reported under "loaded".
            "identity": {
                "requested": {
                    "bundle_sae_id": sae_id,
                    "layer": layer,
                    "feature_idx": target.feature_idx,
                    "read_from": "the resolved control state (the concept bundle), not the loader",
                },
                "loaded": loaded_identity.as_dict(),
                "requested_layer_equals_loaded_layer": True,
                "backend_configured_qwen_layer": self._qwen_layer,
                "note": (
                    "requested_layer_equals_loaded_layer is unconditionally True here because a "
                    "mismatch raises LoadedLayerIdentityMismatch and no result is produced at "
                    "all; it is recorded so a reader of this record does not have to infer that "
                    "the comparison happened. backend_configured_qwen_layer is this backend's "
                    "own configuration -- a statement of intent, not evidence of what loaded."
                ),
            },
            "requested": {
                "concept_id": resolved.concept_id,
                "pairing_id": resolved.pairing_id,
                "direction": resolved.direction.value,
                "strength": resolved.strength.value,
                "operation": resolved.operation.value,
                "positions": positions,
                "requested_bundle_sae_id": sae_id,
                "requested_layer": layer,
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
            text=self._tag(text, attribution),
            is_synthetic=False,
            resolved_config=resolved,
            diagnostics=diagnostics,
        )
