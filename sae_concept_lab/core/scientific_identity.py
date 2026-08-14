"""The scientific-identity gate: what SAE was ACTUALLY loaded, versus what a
result is being attributed to.

WHY THIS MODULE EXISTS (the defect it closes, stated plainly).

`core/gemma_backend.py::_generate_with_intervention` unpacked `(sae_id, layer,
target)` from the RESOLVED CONTROL STATE (the concept bundle) and then loaded
the model/SAE from `_ensure_loaded()` -- i.e. from
`extracted_runtime/targets.py`'s pinned identity, which the unpacked `layer`
had no influence over whatsoever. The unpacked `layer` was never used to load
anything; it reappeared only as a diagnostics field. So a feature index chosen
in one layer's dictionary could be clamped inside a DIFFERENT layer's
dictionary while the diagnostics confidently reported the requested layer.
That is the worst available failure mode: confident, wrong, and well-labelled.

`extracted_runtime/targets.py` is provenance-locked (see
`provenance/source_import.json`) and is NOT edited here. Which SAE is primary
is a ruled, frozen scientific decision and is NOT re-decided here. This module
gates the CLAIM, not the pin: an engineering demonstration on any pin remains
permitted, and remains available, WHEN IT IS LABELLED AS SUCH.

THE THREE RULES THIS MODULE IMPLEMENTS.

1. A LAYER MISMATCH REFUSES. If the resolved control state names a layer other
   than the layer actually loaded, `require_loaded_layer_matches_request`
   raises `LoadedLayerIdentityMismatch`. It does not clamp, does not warn, and
   does not continue. A feature index is meaningful only inside the dictionary
   it was found in.

2. SCIENCE ATTRIBUTION REQUIRES THE CERTIFIED PRIMARY. A result may be
   presented as science-attributed only when the identity that was ACTUALLY
   LOADED equals the certified PRIMARY configuration on all four compared
   fields (sae_repository, release, scientific_sae_id, layer). Anything else
   is an ENGINEERING DEMONSTRATION and is tagged as one.

3. DIAGNOSTICS REPORT THE LOADED IDENTITY. `LoadedSaeIdentity` is built from
   the loader's own returned provenance record -- what `_ensure_loaded()`
   actually produced -- never from the request. Requested values may still be
   reported, but only under keys that say `requested`, beside a `loaded` block,
   so the two can never be read as each other.

WHERE THE CERTIFIED CONSTANTS BELOW COME FROM (read, not assumed).

Transcribed from the qwen-sae-interp certified candidate
`8ed280953bdcdd3007ca6196c817fe37ffade0a9`, read read-only via `git show`:

  protocols/final_pairing/v1/scientific_config_identity.json
    (protocol_version "final-pairing-config-identity/1.3.0")
      configurations.PRIMARY -> repository_id "google/gemma-scope-2-12b-it",
        revision "4c419f1ba0be8b7754d4151d4f26c23b92a9029e",
        release "gemma-scope-2-12b-it-res-all",
        loader_sae_id "layer_29_width_16k_l0_big",
        scientific_sae_id "resid_post_all/layer_29_width_16k_l0_big", layer 29
      configurations.BACKUP -> same repository/revision, release
        "gemma-scope-2-12b-it-res", scientific_sae_id
        "resid_post/layer_24_width_16k_l0_medium", layer 24

  protocols/final_pairing/v1/qwen_config_identity.json
    (protocol_version "final-pairing-qwen-config-identity/1.0.0", gating)
      configurations.PRIMARY -> sae_repository_id
        "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100",
        sae_revision "82852e98c9b33d02194e92dd514b12fafd09ed25",
        params_file "layer38.sae.pt", layer 38, k 100
      configurations.BACKUP -> sae_repository_id
        "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
        sae_revision "13d4221569f7ca5d3c1e605e3e3dc95117e4807c",
        params_file "layer32.sae.pt", layer 32, k 50

`tests/test_scientific_identity_gate.py::
test_certified_constants_match_the_certified_candidates_own_protocol_artifacts`
re-reads both artifacts from that commit and re-derives every constant below,
so these values are not merely asserted here from a work order's summary.

NOTE ON THE RATIFIED BACKUP, which is the dangerous case. The Gemma backup
(`resid_post/layer_24_width_16k_l0_medium`) and the Qwen backup
(`-W80K-L0_50`) are real, ratified, coherent configurations. A reader who
checks one finds it present and defensible and stops there. That is exactly
why `evaluate_science_attribution` reports `matches_ratified_backup`
explicitly and still refuses attribution: being the ratified BACKUP is not
being the certified PRIMARY, and a verdict that only said "mismatch" would
leave a reader to rediscover the difference on their own.

NOTE ON THE QWEN ARM. `extracted_runtime/qwen_loader.py` records
`provenance["sae"]["release"]` and `["sae_id"]` as literal `None` (the
qwen_scope_raw_pt format has no sae_lens release namespace and the loader
writes no artifact id). That module is provenance-locked and is not edited
here. The consequence is deliberate and is not worked around: a Qwen run
cannot report a scientific SAE id, therefore a Qwen run is never
science-attributed by this gate. A missing field is never read as agreement --
see `_mismatched_fields`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Prefixed to generated text whenever the loaded identity is not the certified
#: PRIMARY. Engineering demonstrations remain fully permitted -- this is the
#: label that permits them.
ENGINEERING_DEMONSTRATION_TAG = (
    "[ENGINEERING DEMONSTRATION ONLY -- the loaded SAE is not the certified primary "
    "configuration; nothing here is attributed to it -- see core/scientific_identity.py]"
)

#: The two values `diagnostics["claim_scope"]` may take. A consumer reads this
#: one key rather than inferring a scope from the presence or absence of
#: several others.
CLAIM_SCOPE_SCIENCE_ATTRIBUTED = "SCIENCE_ATTRIBUTED"
CLAIM_SCOPE_ENGINEERING_ONLY = "ENGINEERING_DEMONSTRATION_ONLY"

#: The fields compared, in report order. `sae_repository` is included because
#: the Qwen arm's two configurations are two SEPARATE REPOSITORIES (L0_100 vs
#: L0_50) rather than two releases inside one, so a comparison that skipped it
#: would be blind to the entire Qwen primary/backup distinction.
COMPARED_IDENTITY_FIELDS: tuple[str, ...] = (
    "sae_repository",
    "release",
    "scientific_sae_id",
    "layer",
)


class ScientificIdentityError(RuntimeError):
    """Base for every refusal in this module. Fail closed, never
    warn-and-continue -- the same discipline `extracted_runtime/targets.py`'s
    `TargetIdentityMismatch` already applies one level down."""


class LoadedLayerIdentityMismatch(ScientificIdentityError):
    """THE P0. The resolved control state names one layer and the SAE actually
    loaded is a different layer, so the requested feature index would be
    clamped inside a dictionary it was never found in. Refused outright."""


class LoadedIdentityUnavailable(ScientificIdentityError):
    """The loader's provenance record does not report which layer was loaded
    at all, so the layer agreement above can be neither confirmed nor denied.
    "Cannot verify" is refused rather than passed through -- the same ruling
    `extracted_runtime/targets.IdentityUnverified` already records for
    snapshot identity."""


@dataclass(frozen=True, slots=True)
class SaeScientificIdentity:
    """A ratified configuration's identity. `revision` and
    `params_expected_sha256` are RECORDED for audit only: nothing in this
    module hashes a file, and no field here may be presented as verified by
    this module."""

    sae_repository: str
    release: str | None
    scientific_sae_id: str | None
    layer: int
    revision: str | None = None
    params_expected_sha256: str | None = None
    configuration: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "sae_repository": self.sae_repository,
            "release": self.release,
            "scientific_sae_id": self.scientific_sae_id,
            "layer": self.layer,
            "recorded_revision_not_verified_here": self.revision,
            "recorded_params_sha256_not_verified_here": self.params_expected_sha256,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class LoadedSaeIdentity:
    """What `_ensure_loaded()` ACTUALLY returned, read out of the loader's own
    provenance record. Every field is optional because a loader may genuinely
    not record it (the Qwen arm does not record release/sae_id) -- an absent
    field is reported as absent and compared as a disagreement, never quietly
    filled in from the request or from the ratified target."""

    sae_repository: str | None = None
    release: str | None = None
    scientific_sae_id: str | None = None
    layer: int | None = None
    hook_name: str | None = None
    loader_sae_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sae_repository": self.sae_repository,
            "release": self.release,
            "scientific_sae_id": self.scientific_sae_id,
            "layer": self.layer,
            "hook_name": self.hook_name,
            "loader_sae_id": self.loader_sae_id,
            "read_from": "the loader's own returned provenance record, not from the request",
        }


#: Certified PRIMARY per pairing. See this module's docstring for the exact
#: artifact, commit and field each value was read from.
CERTIFIED_PRIMARY: dict[str, SaeScientificIdentity] = {
    "gemma": SaeScientificIdentity(
        sae_repository="google/gemma-scope-2-12b-it",
        release="gemma-scope-2-12b-it-res-all",
        scientific_sae_id="resid_post_all/layer_29_width_16k_l0_big",
        layer=29,
        revision="4c419f1ba0be8b7754d4151d4f26c23b92a9029e",
        params_expected_sha256="6bb44c8c68797942d097604bfd8df50f4865c86282e2c4667e364382ea26120e",
        configuration="PRIMARY",
        source=(
            "qwen-sae-interp 8ed280953bdcdd3007ca6196c817fe37ffade0a9 "
            "protocols/final_pairing/v1/scientific_config_identity.json configurations.PRIMARY"
        ),
    ),
    "qwen": SaeScientificIdentity(
        sae_repository="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100",
        # The qwen_scope_raw_pt format has no sae_lens release namespace: the
        # repository IS the namespace, so there is no release string to record
        # and None here is a positive statement, not a gap.
        release=None,
        # For a raw-pt release the per-layer params file IS the scientific
        # artifact identity -- there is no subdirectory structure to name.
        scientific_sae_id="layer38.sae.pt",
        layer=38,
        revision="82852e98c9b33d02194e92dd514b12fafd09ed25",
        params_expected_sha256="78b94bf19d4c120e70ba2767734b6d904468d127537e5d16c2a76cbc0963aeb0",
        configuration="PRIMARY",
        source=(
            "qwen-sae-interp 8ed280953bdcdd3007ca6196c817fe37ffade0a9 "
            "protocols/final_pairing/v1/qwen_config_identity.json configurations.PRIMARY"
        ),
    ),
}

#: Ratified BACKUP per pairing. Present so a mismatch verdict can SAY that the
#: loaded identity is the ratified backup instead of leaving a reader to
#: rediscover it -- and so that saying so still does not confer attribution.
RATIFIED_BACKUP: dict[str, SaeScientificIdentity] = {
    "gemma": SaeScientificIdentity(
        sae_repository="google/gemma-scope-2-12b-it",
        release="gemma-scope-2-12b-it-res",
        scientific_sae_id="resid_post/layer_24_width_16k_l0_medium",
        layer=24,
        revision="4c419f1ba0be8b7754d4151d4f26c23b92a9029e",
        params_expected_sha256="2e5f3bc8edc5340ac101fe967f5b59d7a14b40c47315baf5a3446232cb2e799e",
        configuration="BACKUP",
        source=(
            "qwen-sae-interp 8ed280953bdcdd3007ca6196c817fe37ffade0a9 "
            "protocols/final_pairing/v1/scientific_config_identity.json configurations.BACKUP"
        ),
    ),
    "qwen": SaeScientificIdentity(
        sae_repository="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
        release=None,
        scientific_sae_id="layer32.sae.pt",
        layer=32,
        revision="13d4221569f7ca5d3c1e605e3e3dc95117e4807c",
        params_expected_sha256="fbbae7cf93c1e385c68213ae871ede349ac666f3a8c4e6a75ef959db2b6612ab",
        configuration="BACKUP",
        source=(
            "qwen-sae-interp 8ed280953bdcdd3007ca6196c817fe37ffade0a9 "
            "protocols/final_pairing/v1/qwen_config_identity.json configurations.BACKUP"
        ),
    ),
}


def loaded_identity_from_provenance(provenance: Any) -> LoadedSaeIdentity:
    """Reads the loaded identity out of the provenance dict the extracted
    loaders return (`gemma_loader.load_gemma_it_target` /
    `qwen_loader.load_qwen_target`). Reads ONLY provenance -- never the
    request, never `extracted_runtime.targets`' ratified constants. A field
    the record does not carry comes back None."""
    if not isinstance(provenance, dict):
        return LoadedSaeIdentity()
    sae = provenance.get("sae")
    sae = sae if isinstance(sae, dict) else {}
    layer_block = provenance.get("layer")
    layer_block = layer_block if isinstance(layer_block, dict) else {}
    layer = layer_block.get("engineering_layer")
    return LoadedSaeIdentity(
        sae_repository=sae.get("repository"),
        release=sae.get("release"),
        scientific_sae_id=sae.get("sae_id"),
        layer=layer if isinstance(layer, int) and not isinstance(layer, bool) else None,
        hook_name=layer_block.get("hook_name"),
        loader_sae_id=sae.get("loader_sae_id"),
    )


def require_loaded_layer_matches_request(
    *,
    pairing: str,
    requested_layer: int,
    requested_sae_id: str,
    feature_idx: int,
    loaded: LoadedSaeIdentity,
) -> None:
    """THE P0 REFUSAL. Raises `LoadedLayerIdentityMismatch` when the resolved
    control state's layer is not the layer that was actually loaded, and
    `LoadedIdentityUnavailable` when the loader recorded no layer at all.
    Returns None (and only None) when the two agree.

    Deliberately compares against `loaded.layer` -- what the loader reported
    it produced -- and NEVER against `targets.GEMMA_3_12B_IT_TARGET
    .expected_layer` or a backend's own configured layer. Those are statements
    of intent; a detector built on either would accept a request that agreed
    with the intent while the loader had produced something else, which is the
    exact failure this function exists to catch."""
    if loaded.layer is None:
        raise LoadedIdentityUnavailable(
            f"REFUSING TO GENERATE for pairing {pairing!r}: the resolved control state targets "
            f"layer {requested_layer} (bundle sae_id {requested_sae_id!r}, feature_idx "
            f"{feature_idx}), but the loader's provenance record does not report which layer was "
            f"actually loaded, so the two cannot be compared. An unverifiable layer identity is "
            f"refused, not assumed to agree."
        )
    if loaded.layer != requested_layer:
        raise LoadedLayerIdentityMismatch(
            f"REFUSING TO GENERATE for pairing {pairing!r}: the resolved control state targets "
            f"layer {requested_layer} (bundle sae_id {requested_sae_id!r}, feature_idx "
            f"{feature_idx}), but the SAE actually loaded is at layer {loaded.layer} (release "
            f"{loaded.release!r}, scientific_sae_id {loaded.scientific_sae_id!r}). A feature index "
            f"is only meaningful inside the dictionary it was found in, so clamping a layer-"
            f"{requested_layer} index inside the layer-{loaded.layer} dictionary would produce a "
            f"confident, wrong, well-labelled result. This is refused outright -- never clamped, "
            f"never warned past, and never reported as a layer it was not run at."
        )


def _mismatched_fields(
    certified: SaeScientificIdentity, loaded: LoadedSaeIdentity
) -> tuple[str, ...]:
    """Names of `COMPARED_IDENTITY_FIELDS` on which the loaded identity does
    not equal the certified one, in report order.

    A field the loaded identity does not report (None) counts as a
    DISAGREEMENT whenever the certified side names a value: "the loader did
    not say" is never evidence of agreement. Where the certified side is
    itself None (the Qwen arm has no release namespace, by construction), two
    Nones agree -- that is a recorded absence on both sides, not a silence."""
    return tuple(
        field
        for field in COMPARED_IDENTITY_FIELDS
        if getattr(loaded, field) != getattr(certified, field)
    )


@dataclass(frozen=True, slots=True)
class ScientificAttributionVerdict:
    """Whether this result may be presented as attributed to the certified
    primary SAE, and, when not, exactly why not."""

    pairing: str
    science_attributed: bool
    mismatched_fields: tuple[str, ...]
    matches_ratified_backup: bool
    certified_primary: SaeScientificIdentity | None
    loaded: LoadedSaeIdentity
    statement: str

    @property
    def claim_scope(self) -> str:
        return (
            CLAIM_SCOPE_SCIENCE_ATTRIBUTED
            if self.science_attributed
            else CLAIM_SCOPE_ENGINEERING_ONLY
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairing": self.pairing,
            "science_attributed": self.science_attributed,
            "claim_scope": self.claim_scope,
            "mismatched_fields": list(self.mismatched_fields),
            "compared_fields": list(COMPARED_IDENTITY_FIELDS),
            "matches_ratified_backup": self.matches_ratified_backup,
            "certified_primary": (
                None if self.certified_primary is None else self.certified_primary.as_dict()
            ),
            "loaded": self.loaded.as_dict(),
            "statement": self.statement,
        }


def evaluate_science_attribution(
    *, pairing: str, loaded: LoadedSaeIdentity
) -> ScientificAttributionVerdict:
    """Compares the ACTUALLY LOADED identity against the certified PRIMARY on
    every field in `COMPARED_IDENTITY_FIELDS`. Attribution is granted only on
    a clean sweep; anything else is an engineering demonstration.

    Never raises: an engineering demonstration is a permitted outcome, not an
    error. What it is not permitted to be is unlabelled -- the caller tags the
    text with `ENGINEERING_DEMONSTRATION_TAG` and records `claim_scope`."""
    certified = CERTIFIED_PRIMARY.get(pairing)
    if certified is None:
        return ScientificAttributionVerdict(
            pairing=pairing,
            science_attributed=False,
            mismatched_fields=COMPARED_IDENTITY_FIELDS,
            matches_ratified_backup=False,
            certified_primary=None,
            loaded=loaded,
            statement=(
                f"ENGINEERING DEMONSTRATION ONLY: pairing {pairing!r} has no certified PRIMARY "
                f"configuration recorded in core/scientific_identity.CERTIFIED_PRIMARY, so no "
                f"result on it can be attributed to one."
            ),
        )

    mismatched = _mismatched_fields(certified, loaded)
    backup = RATIFIED_BACKUP.get(pairing)
    matches_backup = backup is not None and _mismatched_fields(backup, loaded) == ()

    if not mismatched:
        return ScientificAttributionVerdict(
            pairing=pairing,
            science_attributed=True,
            mismatched_fields=(),
            matches_ratified_backup=matches_backup,
            certified_primary=certified,
            loaded=loaded,
            statement=(
                f"SCIENCE-ATTRIBUTED: the SAE actually loaded for pairing {pairing!r} equals the "
                f"certified PRIMARY configuration on every compared identity field "
                f"({', '.join(COMPARED_IDENTITY_FIELDS)}). This states identity agreement only; "
                f"it asserts nothing about calibration, concept validity, or behavioral quality."
            ),
        )

    if matches_backup:
        statement = (
            f"ENGINEERING DEMONSTRATION ONLY: the SAE actually loaded for pairing {pairing!r} is "
            f"the RATIFIED BACKUP configuration, not the certified PRIMARY. The backup is real, "
            f"ratified and internally coherent -- and it is still not the configuration a "
            f"scientific claim on this pairing is attributed to. Differs from PRIMARY on: "
            f"{', '.join(mismatched)}."
        )
    else:
        statement = (
            f"ENGINEERING DEMONSTRATION ONLY: the SAE actually loaded for pairing {pairing!r} "
            f"differs from the certified PRIMARY configuration on {', '.join(mismatched)}, and is "
            f"neither the certified PRIMARY nor the ratified BACKUP. Nothing produced with it is "
            f"attributed to either."
        )
    return ScientificAttributionVerdict(
        pairing=pairing,
        science_attributed=False,
        mismatched_fields=mismatched,
        matches_ratified_backup=matches_backup,
        certified_primary=certified,
        loaded=loaded,
        statement=statement,
    )
