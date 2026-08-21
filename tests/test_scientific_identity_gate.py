"""The scientific-identity gate: a demonstration may not be attributed to an
SAE that was not the one loaded, and a layer mismatch must refuse rather than
clamp a feature index into the wrong dictionary.

VERIFICATION DISCIPLINE THESE TESTS ARE BOUND BY (and where each is met):

* POSITIVE CONTROL asserting the EXACT exception and the EXACT message, never
  merely "an error occurred" --
  `test_positive_control_layer_mismatch_raises_the_exact_named_exception`
  and `test_loader_reporting_no_layer_refuses_rather_than_assuming_agreement`.
  Both compare `str(exc.value)` against a literal spelled out in this file, so
  the assertion does not reduce to "the module equals itself".

* NEGATIVE CONTROL: a matching bundle still runs, and the attribution gate can
  actually PASS --
  `test_negative_control_matching_bundle_still_generates` and
  `test_negative_control_certified_primary_identity_is_science_attributed`.
  A gate that cannot pass is as broken as one that cannot fail.

* THE EXPECTED VALUES ARE ESTABLISHED INDEPENDENTLY, not asserted from
  intuition --
  `test_certified_constants_match_the_certified_candidates_own_protocol_artifacts`
  re-reads `protocols/final_pairing/v1/scientific_config_identity.json` and
  `protocols/final_pairing/v1/qwen_config_identity.json` out of the
  qwen-sae-interp certified candidate 8ed2809 (read-only `git show`) and
  re-derives every constant `core/scientific_identity.py` declares.

* NEAR-MISSES, i.e. cases a plausibly-broken detector accepts and a correct
  one rejects:
  - `test_near_miss_request_agreeing_with_the_targets_py_pin_still_refuses`:
    the request agrees with `extracted_runtime/targets.py`'s OWN declared
    expected_layer while the loader produced a different layer. A detector
    that compared the request against the ratified pin (a statement of
    intent) accepts this; only a detector that compares against what was
    LOADED rejects it.
  - `test_diagnostics_report_the_loaded_identity_not_the_requested_one`:
    built specifically on a case where the loaded and requested identities
    DIFFER. A diagnostics test where the two agree cannot distinguish a
    correct implementation from one that echoes the request, so it could not
    have exposed this defect at all.
  - `test_absent_loaded_field_is_a_disagreement_never_an_agreement`: a loader
    that reports nothing is the case a "compare only what is present"
    detector accepts.

* No control or escape character is counted anywhere in this file, by grep or
  otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from sae_concept_lab.canonical.concept_bundle import decode_entry, resolve_control
from sae_concept_lab.core.gemma_backend import GemmaRuntimeBackend
from sae_concept_lab.core.protocol import GenerationRequest
from sae_concept_lab.core.qwen_backend import QwenRuntimeBackend
from sae_concept_lab.core.scientific_identity import (
    CERTIFIED_PRIMARY,
    CLAIM_SCOPE_ENGINEERING_ONLY,
    CLAIM_SCOPE_SCIENCE_ATTRIBUTED,
    ENGINEERING_DEMONSTRATION_TAG,
    RATIFIED_BACKUP,
    LoadedIdentityUnavailable,
    LoadedLayerIdentityMismatch,
    LoadedSaeIdentity,
    evaluate_science_attribution,
    loaded_identity_from_provenance,
)
from sae_concept_lab.extracted_runtime import targets
from tests._fake_runtime import (
    FakeGemmaModel,
    FakeQwenHfModel,
    FakeQwenTextDecoder,
    fake_make_clamp_hook,
    fake_register_qwen_raw_hook,
    install_fake_qwen_transformers,
    install_fake_torch,
    make_fake_wrap_hook_with_diagnostics,
)

#: The identity extracted_runtime/targets.py actually pins today. Spelled out
#: here rather than read from that module, so these tests state what they
#: expect instead of agreeing with whatever the module happens to say.
PINNED_GEMMA_RELEASE = "gemma-scope-2-12b-it-res"
PINNED_GEMMA_SCIENTIFIC_SAE_ID = "resid_post_all/layer_29_width_16k_l0_big"
PINNED_GEMMA_LAYER = 29

BUNDLE_SAE_ID = "fake-sae-demo-gemma-000"
FEATURE_IDX = 1001


# ---------------------------------------------------------------------------
# Fakes: a loader whose returned provenance is chosen by the test, which is
# the only way to construct a case where loaded and requested DIFFER.
# ---------------------------------------------------------------------------


def _install_gemma_fakes(
    monkeypatch,
    *,
    loaded_layer: int | None,
    release: str | None,
    scientific_sae_id: str | None,
    repository: str = "google/gemma-scope-2-12b-it",
):
    install_fake_torch(monkeypatch)
    model = FakeGemmaModel()
    hook_name = f"blocks.{loaded_layer}.hook_resid_post"
    layer_block: dict = {"hook_name": hook_name}
    if loaded_layer is not None:
        layer_block["engineering_layer"] = loaded_layer
    provenance = {
        "target": "gemma-3-12b-it",
        "model": {"repository": "google/gemma-3-12b-it", "actual_class": "HookedTransformer"},
        "sae": {
            "repository": repository,
            "release": release,
            "sae_id": scientific_sae_id,
            "d_in": 3840,
            "d_sae": 16384,
        },
        "layer": layer_block,
    }

    def fake_load_gemma_it_target(model_path, sae_path, *, device="cuda", dtype="bfloat16",
                                  expected_model_revision=None, expected_sae_revision=None):
        return model, object(), hook_name, provenance

    import sae_concept_lab.extracted_runtime.diagnostics as diagnostics_module
    import sae_concept_lab.extracted_runtime.gemma_loader as gemma_loader_module
    import sae_concept_lab.extracted_runtime.hooks as hooks_module

    monkeypatch.setattr(gemma_loader_module, "load_gemma_it_target", fake_load_gemma_it_target)
    monkeypatch.setattr(hooks_module, "_make_clamp_hook", fake_make_clamp_hook)
    fake_wrap, wrap_calls = make_fake_wrap_hook_with_diagnostics()
    monkeypatch.setattr(diagnostics_module, "wrap_hook_with_diagnostics", fake_wrap)
    return wrap_calls


def _install_qwen_fakes(
    monkeypatch,
    *,
    loaded_layer: int,
    repository: str,
    release: str | None = None,
    scientific_sae_id: str | None = None,
):
    install_fake_torch(monkeypatch)
    install_fake_qwen_transformers(monkeypatch)
    text_decoder = FakeQwenTextDecoder(num_layers=loaded_layer + 1)
    hf_model = FakeQwenHfModel(text_decoder)
    hook_identifier = f"resid_post:layer_{loaded_layer}"
    provenance = {
        "target": "qwen-3.5-27b",
        "model": {"repository": "Qwen/Qwen3.5-27B", "actual_class": "Qwen3_5ForCausalLM"},
        "sae": {
            "repository": repository,
            "release": release,
            "sae_id": scientific_sae_id,
            "d_in": 5120,
            "d_sae": 81920,
            "k": 50,
        },
        "layer": {
            "engineering_layer": loaded_layer,
            "engineering_only": True,
            "hook_name": hook_identifier,
        },
    }

    def fake_load_qwen_target(model_path, sae_layer_file_path, *, layer, k=None, device="cuda",
                              dtype="bfloat16", expected_model_revision=None,
                              expected_sae_revision=None):
        return hf_model, text_decoder, object(), hook_identifier, provenance

    import sae_concept_lab.extracted_runtime.diagnostics as diagnostics_module
    import sae_concept_lab.extracted_runtime.hooks as hooks_module
    import sae_concept_lab.extracted_runtime.qwen_loader as qwen_loader_module

    monkeypatch.setattr(qwen_loader_module, "load_qwen_target", fake_load_qwen_target)
    monkeypatch.setattr(qwen_loader_module, "register_qwen_raw_hook", fake_register_qwen_raw_hook)
    monkeypatch.setattr(hooks_module, "_make_clamp_hook", fake_make_clamp_hook)
    fake_wrap, wrap_calls = make_fake_wrap_hook_with_diagnostics()
    monkeypatch.setattr(diagnostics_module, "wrap_hook_with_diagnostics", fake_wrap)
    return wrap_calls


def _resolved_at_layer(layer: int, *, sae_id: str = BUNDLE_SAE_ID, concept_id: str = "FAKE-gate-check"):
    document = json.dumps({
        "schema_version": "1.0",
        "concept_id": concept_id,
        "pairing_id": "fake-gate-check-pairing",
        "positions": "all",
        "provenance": "fake",
        "calibration_provenance": None,
        "directions": {
            "amplify": {
                "targets": [
                    {"sae_id": sae_id, "layer": layer, "feature_idx": FEATURE_IDX, "weight": 1.0}
                ],
                "specs": {
                    strength: {
                        "operation": "clamp", "value": 5.0,
                        "unit": "absolute_activation", "unit_source": None,
                    }
                    for strength in ("low", "medium", "high")
                },
            },
            "suppress": None,
        },
    })
    entry = decode_entry(document, where=concept_id)
    return resolve_control(entry, direction="amplify", strength="medium")


def _request(resolved, model_key: str = "gemma"):
    return GenerationRequest(
        history=(), prompt="hi", model_key=model_key, decoding={"max_new_tokens": 4},
        seed=0, apply_intervention=True, resolved_config=resolved,
    )


# ---------------------------------------------------------------------------
# POSITIVE CONTROLS -- the gate must fire, with the exact named exception.
# ---------------------------------------------------------------------------


def test_positive_control_layer_mismatch_raises_the_exact_named_exception(monkeypatch):
    """The defect, reproduced with the roles as they now stand: a bundle
    authored at layer 31 -- the engineering layer this runtime used to pin --
    against a runtime now pinned to the certified primary, layer 29. Before the
    gate existed this produced a generation reporting `"layer": 31` while a
    layer-31 feature index was clamped inside the layer-29 dictionary."""
    _install_gemma_fakes(
        monkeypatch,
        loaded_layer=PINNED_GEMMA_LAYER,
        release=PINNED_GEMMA_RELEASE,
        scientific_sae_id=PINNED_GEMMA_SCIENTIFIC_SAE_ID,
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    with pytest.raises(LoadedLayerIdentityMismatch) as excinfo:
        backend.generate(_request(_resolved_at_layer(31)))

    # The EXACT type, not a superclass that a broader except would also catch.
    assert type(excinfo.value) is LoadedLayerIdentityMismatch
    # The EXACT message, spelled out here rather than rebuilt from the module.
    assert str(excinfo.value) == (
        "REFUSING TO GENERATE for pairing 'gemma': the resolved control state targets layer 31 "
        "(bundle sae_id 'fake-sae-demo-gemma-000', feature_idx 1001), but the SAE actually loaded "
        "is at layer 29 (release 'gemma-scope-2-12b-it-res', scientific_sae_id "
        "'resid_post_all/layer_29_width_16k_l0_big'). A feature index is only meaningful inside the "
        "dictionary it was found in, so clamping a layer-31 index inside the layer-29 dictionary "
        "would produce a confident, wrong, well-labelled result. This is refused outright -- never "
        "clamped, never warned past, and never reported as a layer it was not run at."
    )


def test_near_miss_request_agreeing_with_the_targets_py_pin_still_refuses(monkeypatch):
    """NEAR-MISS. The request names layer 29, which is exactly what
    extracted_runtime/targets.py declares as GEMMA_3_12B_IT_TARGET
    .expected_layer -- so a detector that compared the request against the
    RATIFIED PIN would find perfect agreement and let this through. The
    loader, however, produced layer 31. Only a detector that compares against
    what was actually LOADED can reject this."""
    # The near-miss is real, not accidental: the request genuinely agrees with
    # the module-level pin a plausibly-broken detector would have consulted.
    assert targets.GEMMA_3_12B_IT_TARGET.expected_layer == PINNED_GEMMA_LAYER

    _install_gemma_fakes(
        monkeypatch,
        loaded_layer=31,
        release="gemma-scope-2-12b-it-res",
        scientific_sae_id="resid_post/layer_31_width_16k_l0_medium",
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    with pytest.raises(LoadedLayerIdentityMismatch) as excinfo:
        backend.generate(_request(_resolved_at_layer(PINNED_GEMMA_LAYER)))

    assert str(excinfo.value) == (
        "REFUSING TO GENERATE for pairing 'gemma': the resolved control state targets layer 29 "
        "(bundle sae_id 'fake-sae-demo-gemma-000', feature_idx 1001), but the SAE actually loaded "
        "is at layer 31 (release 'gemma-scope-2-12b-it-res', scientific_sae_id "
        "'resid_post/layer_31_width_16k_l0_medium'). A feature index is only meaningful inside "
        "the dictionary it was found in, so clamping a layer-29 index inside the layer-31 "
        "dictionary would produce a confident, wrong, well-labelled result. This is refused "
        "outright -- never clamped, never warned past, and never reported as a layer it was not "
        "run at."
    )


def test_loader_reporting_no_layer_refuses_rather_than_assuming_agreement(monkeypatch):
    """A loader that records no layer at all cannot be compared against, so
    the comparison is neither confirmed nor denied. Refused, not assumed --
    the same ruling targets.IdentityUnverified already records for snapshot
    identity."""
    _install_gemma_fakes(
        monkeypatch,
        loaded_layer=None,
        release=PINNED_GEMMA_RELEASE,
        scientific_sae_id=PINNED_GEMMA_SCIENTIFIC_SAE_ID,
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    with pytest.raises(LoadedIdentityUnavailable) as excinfo:
        backend.generate(_request(_resolved_at_layer(PINNED_GEMMA_LAYER)))

    assert type(excinfo.value) is LoadedIdentityUnavailable
    assert str(excinfo.value) == (
        "REFUSING TO GENERATE for pairing 'gemma': the resolved control state targets layer 31 "
        "(bundle sae_id 'fake-sae-demo-gemma-000', feature_idx 1001), but the loader's provenance "
        "record does not report which layer was actually loaded, so the two cannot be compared. "
        "An unverifiable layer identity is refused, not assumed to agree."
    )


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS -- the gate must be passable.
# ---------------------------------------------------------------------------


def test_negative_control_matching_bundle_still_generates(monkeypatch):
    """A bundle whose layer agrees with the loaded layer runs to completion.
    Asserts concrete produced values, not merely "no exception": the fake
    model appends ids 900..903 for max_new_tokens=4 and the fake tokenizer
    renders them verbatim, so the generated text is fully determined."""
    wrap_calls = _install_gemma_fakes(
        monkeypatch,
        loaded_layer=PINNED_GEMMA_LAYER,
        release=PINNED_GEMMA_RELEASE,
        scientific_sae_id=PINNED_GEMMA_SCIENTIFIC_SAE_ID,
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    result = backend.generate(_request(_resolved_at_layer(PINNED_GEMMA_LAYER)))

    assert result.text.endswith("generated-tokens:[900, 901, 902, 903]")
    assert result.is_synthetic is False
    assert result.diagnostics["verdict"]["hook_invocation_count"] == 4
    assert result.diagnostics["identity"]["loaded"]["layer"] == PINNED_GEMMA_LAYER
    assert result.diagnostics["identity"]["requested"]["layer"] == PINNED_GEMMA_LAYER
    assert len(wrap_calls) == 1
    assert wrap_calls[0]["hook_name"] == "blocks.29.hook_resid_post"


def test_negative_control_certified_primary_identity_is_science_attributed(monkeypatch):
    """The attribution gate can PASS. Loading exactly the certified primary
    and requesting its layer yields a science-attributed result with no
    engineering-demonstration tag. A gate that could only refuse would be as
    broken as one that could only permit."""
    primary = CERTIFIED_PRIMARY["gemma"]
    _install_gemma_fakes(
        monkeypatch,
        loaded_layer=primary.layer,
        release=primary.release,
        scientific_sae_id=primary.scientific_sae_id,
        repository=primary.sae_repository,
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    result = backend.generate(_request(_resolved_at_layer(primary.layer)))

    assert result.diagnostics["science_attributed"] is True
    assert result.diagnostics["claim_scope"] == CLAIM_SCOPE_SCIENCE_ATTRIBUTED
    assert result.diagnostics["scientific_attribution"]["mismatched_fields"] == []
    assert ENGINEERING_DEMONSTRATION_TAG not in result.text
    assert result.text.endswith("generated-tokens:[900, 901, 902, 903]")


# ---------------------------------------------------------------------------
# DIAGNOSTICS -- built on a case where loaded and requested DIFFER.
# ---------------------------------------------------------------------------


def test_diagnostics_report_the_loaded_identity_not_the_requested_one(monkeypatch):
    """The requested bundle sae_id and the loaded scientific sae_id differ,
    which is the only arrangement that can tell a correct implementation apart
    from one that echoes the request back. The layers agree, so the run
    proceeds and there IS a diagnostics dict to inspect."""
    _install_gemma_fakes(
        monkeypatch,
        loaded_layer=PINNED_GEMMA_LAYER,
        release=PINNED_GEMMA_RELEASE,
        scientific_sae_id=PINNED_GEMMA_SCIENTIFIC_SAE_ID,
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    diagnostics = backend.generate(_request(_resolved_at_layer(PINNED_GEMMA_LAYER))).diagnostics

    loaded = diagnostics["identity"]["loaded"]
    requested = diagnostics["identity"]["requested"]

    # The two genuinely differ -- otherwise this test proves nothing.
    assert requested["bundle_sae_id"] != loaded["scientific_sae_id"]
    # The loaded block reports what the LOADER returned.
    assert loaded["release"] == PINNED_GEMMA_RELEASE
    assert loaded["scientific_sae_id"] == PINNED_GEMMA_SCIENTIFIC_SAE_ID
    assert loaded["sae_repository"] == "google/gemma-scope-2-12b-it"
    assert loaded["hook_name"] == "blocks.29.hook_resid_post"
    # The requested block reports the BUNDLE, under keys that say so.
    assert requested["bundle_sae_id"] == BUNDLE_SAE_ID
    assert requested["feature_idx"] == FEATURE_IDX

    # The old ambiguous keys are gone: a bare "sae_id"/"layer" sitting beside
    # the loader's provenance is what let a requested value be read as a
    # loaded one.
    assert "sae_id" not in diagnostics["requested"]
    assert "layer" not in diagnostics["requested"]
    assert diagnostics["requested"]["requested_bundle_sae_id"] == BUNDLE_SAE_ID
    assert diagnostics["requested"]["requested_layer"] == PINNED_GEMMA_LAYER


def test_pinned_engineering_identity_is_refused_scientific_attribution(monkeypatch):
    """The identity extracted_runtime/targets.py pins today is neither the
    certified primary nor the ratified backup, so the result is an engineering
    demonstration -- permitted, and labelled as one in both the text and the
    diagnostics."""
    _install_gemma_fakes(
        monkeypatch,
        loaded_layer=PINNED_GEMMA_LAYER,
        release=PINNED_GEMMA_RELEASE,
        scientific_sae_id=PINNED_GEMMA_SCIENTIFIC_SAE_ID,
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    result = backend.generate(_request(_resolved_at_layer(PINNED_GEMMA_LAYER)))
    attribution = result.diagnostics["scientific_attribution"]

    assert result.diagnostics["science_attributed"] is False
    assert result.diagnostics["claim_scope"] == CLAIM_SCOPE_ENGINEERING_ONLY
    # Repointing to layer 29 aligned scientific_sae_id and layer with the
    # certified primary, so RELEASE is now the only scientific identity field
    # that still diverges. Attribution is STILL refused on that one field
    # alone: a release names which training run produced the dictionary, and
    # two runs at the same layer and width are not the same dictionary.
    assert attribution["mismatched_fields"] == ["release"]
    assert attribution["matches_ratified_backup"] is False
    assert result.text.startswith(ENGINEERING_DEMONSTRATION_TAG)


def test_ratified_backup_is_named_as_the_backup_and_still_refused_attribution(monkeypatch):
    """The dangerous case: the ratified BACKUP is present, ratified and
    internally coherent, so a reader who checks it stops there. The verdict
    must say it is the backup AND still refuse attribution."""
    backup = RATIFIED_BACKUP["gemma"]
    _install_gemma_fakes(
        monkeypatch,
        loaded_layer=backup.layer,
        release=backup.release,
        scientific_sae_id=backup.scientific_sae_id,
        repository=backup.sae_repository,
    )
    backend = GemmaRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae")

    result = backend.generate(_request(_resolved_at_layer(backup.layer)))
    attribution = result.diagnostics["scientific_attribution"]

    assert attribution["science_attributed"] is False
    assert attribution["matches_ratified_backup"] is True
    assert attribution["mismatched_fields"] == ["release", "scientific_sae_id", "layer"]
    assert "RATIFIED BACKUP configuration, not the certified PRIMARY" in attribution["statement"]
    assert result.text.startswith(ENGINEERING_DEMONSTRATION_TAG)


def test_qwen_l0_50_pin_is_reported_as_not_the_certified_primary(monkeypatch):
    """targets.py pins Qwen to the L0_50 repository. L0_50 is the RATIFIED
    BACKUP, which is exactly why it is dangerous: it is present and coherent.
    The certified primary is L0_100 at layer 38."""
    _install_qwen_fakes(
        monkeypatch,
        loaded_layer=32,
        repository="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
    )
    backend = QwenRuntimeBackend(model_path="/fake/model", sae_path="/fake/sae", qwen_layer=32)

    result = backend.generate(_request(_resolved_at_layer(32), model_key="qwen"))
    attribution = result.diagnostics["scientific_attribution"]

    assert attribution["science_attributed"] is False
    assert result.diagnostics["claim_scope"] == CLAIM_SCOPE_ENGINEERING_ONLY
    assert attribution["certified_primary"]["sae_repository"] == (
        "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100"
    )
    assert attribution["certified_primary"]["layer"] == 38
    # `release` agrees (both None: the raw-pt format has no release
    # namespace). `scientific_sae_id` disagrees because the extracted Qwen
    # loader records None -- an absence, never read as agreement.
    assert attribution["mismatched_fields"] == [
        "sae_repository", "scientific_sae_id", "layer",
    ]
    assert result.diagnostics["identity"]["loaded"]["scientific_sae_id"] is None
    assert result.diagnostics["identity"]["backend_configured_qwen_layer"] == 32
    assert result.text.startswith(ENGINEERING_DEMONSTRATION_TAG)


# ---------------------------------------------------------------------------
# Unit-level semantics of the comparison itself.
# ---------------------------------------------------------------------------


def test_absent_loaded_field_is_a_disagreement_never_an_agreement():
    """NEAR-MISS for the comparison function: a loader that reports nothing at
    all. A "compare only the fields that are present" detector finds zero
    disagreements and grants attribution; a correct one grants none."""
    verdict = evaluate_science_attribution(pairing="gemma", loaded=LoadedSaeIdentity())
    assert verdict.science_attributed is False
    assert verdict.mismatched_fields == ("sae_repository", "release", "scientific_sae_id", "layer")


def test_unknown_pairing_is_never_science_attributed():
    verdict = evaluate_science_attribution(
        pairing="not-a-real-pairing",
        loaded=LoadedSaeIdentity(sae_repository="x", release="y", scientific_sae_id="z", layer=1),
    )
    assert verdict.science_attributed is False
    assert verdict.claim_scope == CLAIM_SCOPE_ENGINEERING_ONLY


def test_loaded_identity_is_read_from_provenance_only():
    identity = loaded_identity_from_provenance({
        "sae": {"repository": "r", "release": "rel", "sae_id": "sid", "loader_sae_id": "lid"},
        "layer": {"engineering_layer": 7, "hook_name": "h"},
    })
    assert identity == LoadedSaeIdentity(
        sae_repository="r", release="rel", scientific_sae_id="sid", layer=7,
        hook_name="h", loader_sae_id="lid",
    )
    # A malformed record yields all-None rather than raising or inventing.
    assert loaded_identity_from_provenance(None) == LoadedSaeIdentity()
    assert loaded_identity_from_provenance({"sae": "not-a-dict"}) == LoadedSaeIdentity()


def test_a_boolean_layer_is_not_accepted_as_an_int_layer():
    """bool is a subclass of int in Python, so `isinstance(True, int)` is
    True. A provenance record carrying True where a layer belongs must be
    treated as "no layer reported", not as layer 1."""
    assert loaded_identity_from_provenance({"layer": {"engineering_layer": True}}).layer is None


# ---------------------------------------------------------------------------
# The expected values, re-derived from the certified candidate itself.
# ---------------------------------------------------------------------------

CERTIFIED_CANDIDATE_COMMIT = "8ed280953bdcdd3007ca6196c817fe37ffade0a9"


def _qwen_sae_interp_checkout() -> Path:
    override = os.environ.get("QWEN_SAE_INTERP_CHECKOUT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "qwen-sae-interp"


def _read_blob(repo: Path, path: str) -> dict:
    shown = subprocess.run(
        ["git", "-C", str(repo), "show", f"{CERTIFIED_CANDIDATE_COMMIT}:{path}"],
        capture_output=True,
    )
    if shown.returncode != 0:
        pytest.skip(f"{path} not readable at {CERTIFIED_CANDIDATE_COMMIT} in {repo}")
    return json.loads(shown.stdout.decode("utf-8"))


def test_certified_constants_match_the_certified_candidates_own_protocol_artifacts():
    """Establishes the expected values INDEPENDENTLY of this product: reads
    the two gating identity protocols out of the qwen-sae-interp certified
    candidate (read-only `git show`, never a working-tree read) and re-derives
    every constant core/scientific_identity.py declares. Skipped, never
    silently passed, when that checkout is not reachable."""
    repo = _qwen_sae_interp_checkout()
    if not (repo / ".git").exists():
        pytest.skip(f"qwen-sae-interp checkout not found at {repo}")

    gemma = _read_blob(repo, "protocols/final_pairing/v1/scientific_config_identity.json")
    qwen = _read_blob(repo, "protocols/final_pairing/v1/qwen_config_identity.json")

    gemma_primary = gemma["configurations"]["PRIMARY"]
    assert CERTIFIED_PRIMARY["gemma"].sae_repository == gemma_primary["repository_id"]
    assert CERTIFIED_PRIMARY["gemma"].release == gemma_primary["release"]
    assert CERTIFIED_PRIMARY["gemma"].scientific_sae_id == gemma_primary["scientific_sae_id"]
    assert CERTIFIED_PRIMARY["gemma"].layer == gemma_primary["layer"]
    assert CERTIFIED_PRIMARY["gemma"].revision == gemma_primary["revision"]
    assert CERTIFIED_PRIMARY["gemma"].params_expected_sha256 == (
        gemma_primary["params_expected_sha256"]
    )

    gemma_backup = gemma["configurations"]["BACKUP"]
    assert RATIFIED_BACKUP["gemma"].release == gemma_backup["release"]
    assert RATIFIED_BACKUP["gemma"].scientific_sae_id == gemma_backup["scientific_sae_id"]
    assert RATIFIED_BACKUP["gemma"].layer == gemma_backup["layer"]

    qwen_primary = qwen["configurations"]["PRIMARY"]
    assert CERTIFIED_PRIMARY["qwen"].sae_repository == qwen_primary["sae_repository_id"]
    assert CERTIFIED_PRIMARY["qwen"].scientific_sae_id == qwen_primary["params_file"]
    assert CERTIFIED_PRIMARY["qwen"].layer == qwen_primary["layer"]
    assert CERTIFIED_PRIMARY["qwen"].revision == qwen_primary["sae_revision"]

    qwen_backup = qwen["configurations"]["BACKUP"]
    assert RATIFIED_BACKUP["qwen"].sae_repository == qwen_backup["sae_repository_id"]
    assert RATIFIED_BACKUP["qwen"].scientific_sae_id == qwen_backup["params_file"]
    assert RATIFIED_BACKUP["qwen"].layer == qwen_backup["layer"]

    # extracted_runtime/targets.py now pins the CERTIFIED PRIMARY on the
    # scientific identity fields. This assertion was previously the inverse: it
    # recorded that the pin DIVERGED from the certified science (Gemma layer 31,
    # an admitted engineering carry-over; Qwen's L0_50 release). Repointing to
    # layer 29 / L0_100 closed that divergence, so the same fields are asserted
    # equal here.
    #
    # THE GATE IS NOT MADE UNNECESSARY BY THIS. It compares what was LOADED at
    # run time against what the resolved control state REQUESTS, which is a
    # different question from whether the pin matches the certified constants;
    # a matching pin can still be loaded against a mismatched request, and the
    # refusal tests above cover exactly that.
    assert targets.GEMMA_3_12B_IT_TARGET.sae_id == gemma_primary["scientific_sae_id"]
    assert targets.GEMMA_3_12B_IT_TARGET.expected_layer == gemma_primary["layer"]
    assert targets.QWEN_3_5_27B_TARGET.sae_repo_id == qwen_primary["sae_repository_id"]
