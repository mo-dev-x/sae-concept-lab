"""Hidden, ENGINEERING-ONLY concept-bundle entries for the Tamia smoke
packet -- built directly in Python, never written to
`sae_concept_lab/fixtures/{gemma,qwen}/*.json`, so they can never be named
in `fixtures.loader._ENTRY_FILENAMES`, never enter fixture discovery
(`load_entries()`), never render in the Gradio UI (`ui/tab.py`'s
`_on_send`/`_on_compare` look concepts up by `concept_id` only within the
entries closure `build_model_tab` was constructed with, which is always
`load_entries()`'s output), and never reach the release gate (which only
ever evaluates that same explicit file list). `concept_id` is prefixed
`ENGINEERING-ONLY-SMOKE-` for the same reason the shipped fixtures are
prefixed `FAKE-` -- so it is self-quarantining even if it ever leaked into
a log line by mistake.

The two pinned engineering acceptance inputs -- Qwen layer 0 / feature
4096 / raw target 20, Gemma resid_post layer 31 / feature 250 / raw clamp
5000 -- are copied from `sae_concept_lab/core/runtime_acceptance.py`'s own
bounded claim text and `docs/tamia_launch.md`, not re-derived here. They
are ENGINEERING-ONLY: see `RuntimeAcceptanceRecord.claim` for both
pairings for the exact bounded statement of what mechanically passed and
what it does not mean.

`Provenance.FAKE` on every entry below is deliberate, for the same reason
the shipped UI fixtures use it: these are not, and must never become, an
ATTESTED scientific claim -- they exist only to give the resolver
something to resolve, so this package can drive a real backend through
exactly the same canonical `ResolvedControlState` shape the UI produces.
"""

from __future__ import annotations

from sae_concept_lab.canonical.concept_bundle import (
    BundleEntry,
    Direction,
    DirectionRecord,
    Operation,
    PositionMode,
    Provenance,
    Spec,
    Strength,
    Target,
    Unit,
)

QWEN_SMOKE_PAIRING_ID = "engineering-only-smoke-qwen-pairing"
QWEN_SMOKE_SAE_ID = "engineering-only-smoke-qwen-scope-layer0"
QWEN_SMOKE_LAYER = 0
QWEN_SMOKE_FEATURE_IDX = 4096
QWEN_SMOKE_RAW_TARGET = 20.0

GEMMA_SMOKE_PAIRING_ID = "engineering-only-smoke-gemma-pairing"
GEMMA_SMOKE_SAE_ID = "engineering-only-smoke-gemma-scope-layer31"
GEMMA_SMOKE_LAYER = 31
GEMMA_SMOKE_FEATURE_IDX = 250
GEMMA_SMOKE_RAW_CLAMP = 5000.0

SMOKE_DIRECTION = Direction.AMPLIFY
SMOKE_STRENGTH = Strength.MEDIUM
SMOKE_PROMPT = "Describe the weather today in one sentence."


def _uniform_clamp_direction(*, sae_id: str, layer: int, feature_idx: int, value: float) -> DirectionRecord:
    """One target, the same absolute-activation CLAMP spec at every
    strength -- this package never exercises Low/Medium/High as distinct
    doses, only whichever one strength (`SMOKE_STRENGTH`) a scenario asks
    to resolve, so there is nothing to differentiate them by."""
    target = Target(sae_id=sae_id, layer=layer, feature_idx=feature_idx, weight=1.0)
    spec = Spec(operation=Operation.CLAMP, value=value, unit=Unit.ABSOLUTE_ACTIVATION, unit_source=None)
    return DirectionRecord(targets=(target,), specs={s: spec for s in Strength})


def _smoke_entry(
    *, concept_id: str, pairing_id: str, sae_id: str, layer: int, feature_idx: int, value: float,
    positions: PositionMode,
) -> BundleEntry:
    return BundleEntry(
        concept_id=concept_id,
        pairing_id=pairing_id,
        positions=positions,
        provenance=Provenance.FAKE,
        directions={
            Direction.AMPLIFY: _uniform_clamp_direction(
                sae_id=sae_id, layer=layer, feature_idx=feature_idx, value=value
            ),
            Direction.SUPPRESS: None,
        },
    )


def qwen_smoke_entry(positions: PositionMode) -> BundleEntry:
    """The Qwen engineering acceptance input (layer 0, feature 4096, raw
    target 20), at the requested position mode. Two distinct entries exist
    (ALL and GENERATED_ONLY) because `BundleEntry.positions` is fixed per
    entry -- the canonical contract has no per-request position override."""
    return _smoke_entry(
        concept_id=f"ENGINEERING-ONLY-SMOKE-qwen-{positions.value}",
        pairing_id=QWEN_SMOKE_PAIRING_ID,
        sae_id=QWEN_SMOKE_SAE_ID,
        layer=QWEN_SMOKE_LAYER,
        feature_idx=QWEN_SMOKE_FEATURE_IDX,
        value=QWEN_SMOKE_RAW_TARGET,
        positions=positions,
    )


def gemma_smoke_entry(positions: PositionMode) -> BundleEntry:
    """The Gemma engineering acceptance input (resid_post layer 31,
    feature 250, raw clamp 5000), at the requested position mode."""
    return _smoke_entry(
        concept_id=f"ENGINEERING-ONLY-SMOKE-gemma-{positions.value}",
        pairing_id=GEMMA_SMOKE_PAIRING_ID,
        sae_id=GEMMA_SMOKE_SAE_ID,
        layer=GEMMA_SMOKE_LAYER,
        feature_idx=GEMMA_SMOKE_FEATURE_IDX,
        value=GEMMA_SMOKE_RAW_CLAMP,
        positions=positions,
    )


# ---------------------------------------------------------------------------
# Deliberately-invalid entries, for the defensive enforcement assertions
# only (execution_guard.require_group_from_resolved) -- never resolved
# against a real backend's generate() with apply_intervention reaching a
# model load; both raise before this package's runner ever imports torch.
# ---------------------------------------------------------------------------


def qwen_multi_sae_same_layer_entry() -> BundleEntry:
    """Two distinct SAE identities at the SAME layer -- PROHIBITED
    (`MultipleSaeIdentitiesAtLayerError`), never a runtime ceiling."""
    t1 = Target(sae_id=QWEN_SMOKE_SAE_ID, layer=QWEN_SMOKE_LAYER, feature_idx=QWEN_SMOKE_FEATURE_IDX, weight=1.0)
    t2 = Target(
        sae_id=QWEN_SMOKE_SAE_ID + "-second", layer=QWEN_SMOKE_LAYER, feature_idx=QWEN_SMOKE_FEATURE_IDX + 1,
        weight=1.0,
    )
    spec = Spec(operation=Operation.CLAMP, value=QWEN_SMOKE_RAW_TARGET, unit=Unit.ABSOLUTE_ACTIVATION, unit_source=None)
    record = DirectionRecord(targets=(t1, t2), specs={s: spec for s in Strength})
    return BundleEntry(
        concept_id="ENGINEERING-ONLY-SMOKE-qwen-multi-sae-prohibited",
        pairing_id=QWEN_SMOKE_PAIRING_ID,
        positions=PositionMode.ALL,
        provenance=Provenance.FAKE,
        directions={Direction.AMPLIFY: record, Direction.SUPPRESS: None},
    )


def gemma_cross_layer_entry() -> BundleEntry:
    """Two distinct (sae_id, layer) execution groups -- CAPABILITY_LIMIT
    (`MultipleExecutionGroupsError`), a current runtime ceiling, not a
    prohibition."""
    t1 = Target(sae_id=GEMMA_SMOKE_SAE_ID, layer=GEMMA_SMOKE_LAYER, feature_idx=GEMMA_SMOKE_FEATURE_IDX, weight=1.0)
    t2 = Target(
        sae_id=GEMMA_SMOKE_SAE_ID, layer=GEMMA_SMOKE_LAYER + 1, feature_idx=GEMMA_SMOKE_FEATURE_IDX, weight=1.0
    )
    spec = Spec(operation=Operation.CLAMP, value=GEMMA_SMOKE_RAW_CLAMP, unit=Unit.ABSOLUTE_ACTIVATION, unit_source=None)
    record = DirectionRecord(targets=(t1, t2), specs={s: spec for s in Strength})
    return BundleEntry(
        concept_id="ENGINEERING-ONLY-SMOKE-gemma-cross-layer-capability-limit",
        pairing_id=GEMMA_SMOKE_PAIRING_ID,
        positions=PositionMode.ALL,
        provenance=Provenance.FAKE,
        directions={Direction.AMPLIFY: record, Direction.SUPPRESS: None},
    )


#: Every concept_id this module can produce -- used by
#: tests/test_tamia_smoke.py to prove none of them appear in
#: fixtures.loader.load_entries()'s output for either model_key.
ALL_SMOKE_CONCEPT_IDS: tuple[str, ...] = (
    f"ENGINEERING-ONLY-SMOKE-qwen-{PositionMode.ALL.value}",
    f"ENGINEERING-ONLY-SMOKE-qwen-{PositionMode.GENERATED_ONLY.value}",
    f"ENGINEERING-ONLY-SMOKE-gemma-{PositionMode.ALL.value}",
    f"ENGINEERING-ONLY-SMOKE-gemma-{PositionMode.GENERATED_ONLY.value}",
    "ENGINEERING-ONLY-SMOKE-qwen-multi-sae-prohibited",
    "ENGINEERING-ONLY-SMOKE-gemma-cross-layer-capability-limit",
)
