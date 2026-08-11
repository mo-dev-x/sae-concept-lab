"""sae_concept_lab.fixtures.loader: the adapter that names this product's
explicit canonical entry files and builds a ConceptRegistry from them.
Resolution arithmetic itself (positions, seed_default-equivalents,
strength coefficients) is entirely canonical.concept_bundle's own
responsibility and is exhaustively covered by
tests/test_concept_bundle_conformance.py against the frozen pack; this
file only tests the loader's OWN wiring: which files it names, that they
decode, and that they carry this product's FAKE quarantine markers.
"""

from __future__ import annotations

import pytest

from sae_concept_lab.canonical.concept_bundle import Provenance, resolve_control
from sae_concept_lab.fixtures.loader import PAIRING_ID_FOR_MODEL_KEY, build_registry, load_entries

GEMMA_ENTRIES = load_entries("gemma")
QWEN_ENTRIES = load_entries("qwen")


def test_load_entries_returns_four_concepts_per_pairing():
    assert len(GEMMA_ENTRIES) == 4
    assert len(QWEN_ENTRIES) == 4


def test_load_entries_unknown_model_key_raises():
    with pytest.raises(ValueError, match="unknown model_key"):
        load_entries("not-a-real-model")


def test_every_entry_shares_its_pairings_own_registered_pairing_id():
    for model_key, entries in (("gemma", GEMMA_ENTRIES), ("qwen", QWEN_ENTRIES)):
        expected = PAIRING_ID_FOR_MODEL_KEY[model_key]
        for entry in entries:
            assert entry.pairing_id == expected


def test_build_registry_resolves_every_concept_by_id_and_pairing():
    registry = build_registry("gemma")
    for entry in GEMMA_ENTRIES:
        assert registry.get(entry.concept_id, entry.pairing_id) == entry


def test_positions_comes_from_the_entry_never_a_product_default():
    """The product adapter must never choose positions -- resolve_control
    always reads it from the entry it was handed."""
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        direction = entry.calibrated_directions[0]
        resolved = resolve_control(entry, direction=direction, strength="low")
        assert resolved.positions == entry.positions


def test_concept_ids_are_all_prefixed_fake_never_a_real_looking_id():
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        assert entry.concept_id.upper().startswith("FAKE-")


def test_pairing_ids_are_all_prefixed_fake():
    for pairing_id in PAIRING_ID_FOR_MODEL_KEY.values():
        assert pairing_id.lower().startswith("fake-")


def test_every_shipped_entry_is_provenance_fake_and_never_attested():
    """Product-repo invariant, independent of the canonical release gate:
    nothing shipped here may even claim ATTESTED provenance, regardless
    of whether it would also fail on evidence."""
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        assert entry.provenance is Provenance.FAKE


def test_strength_levels_resolve_to_distinct_absolute_values():
    entry = GEMMA_ENTRIES[0]
    direction = entry.calibrated_directions[0]
    values = {
        strength: resolve_control(entry, direction=direction, strength=strength).targets[0].absolute_value
        for strength in ("low", "medium", "high")
    }
    assert len(set(values.values())) == 3


def test_at_least_one_concept_per_pairing_has_a_single_calibrated_direction():
    """Demonstrates the disabled-control acceptance case with real
    shipped data, not only via a hand-built vector."""
    for entries in (GEMMA_ENTRIES, QWEN_ENTRIES):
        assert any(len(e.calibrated_directions) == 1 for e in entries)


def test_at_least_one_concept_per_pairing_is_capability_limited_or_prohibited():
    """Demonstrates the PROHIBITED/CAPABILITY_LIMIT surfacing requirement
    with real shipped data: at least one concept per pairing has a
    calibrated direction that runtime v1 cannot execute."""
    from sae_concept_lab.canonical.concept_bundle import check_direction_executable

    for entries in (GEMMA_ENTRIES, QWEN_ENTRIES):
        found = any(
            not check_direction_executable(e, d).executable
            for e in entries
            for d in e.calibrated_directions
        )
        assert found
