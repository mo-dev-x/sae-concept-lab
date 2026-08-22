"""sae_concept_lab.fixtures.loader: the adapter that names this product's
explicit canonical entry files and builds a ConceptRegistry from them.
Resolution arithmetic itself (positions, seed_default-equivalents,
strength coefficients) is entirely canonical.concept_bundle's own
responsibility and is exhaustively covered by
tests/test_concept_bundle_conformance.py against the frozen pack; this
file only tests the loader's OWN wiring: which files it names, that they
decode, and that nothing it ships claims attestation it does not have.

The build ships REAL, measured concepts; the eight FAKE placeholders it used
to ship now live under tests/fixtures/ and are loaded explicitly by the tests
that need an entry of a particular shape. Tests about the PRODUCT read
load_entries(); tests about resolution BEHAVIOUR read tests/fixtures/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sae_concept_lab.canonical.concept_bundle import Provenance, resolve_control
from sae_concept_lab.canonical.concept_bundle.codec import load_entry_file
from sae_concept_lab.fixtures.loader import PAIRING_ID_FOR_MODEL_KEY, build_registry, load_entries

GEMMA_ENTRIES = load_entries("gemma")
QWEN_ENTRIES = load_entries("qwen")

#: Test-owned entries of known shape, for the behaviours the shipped set is not
#: obliged to exhibit (a one-direction concept, a non-executable direction).
_TF = Path(__file__).resolve().parent / "fixtures"
SHAPE_FIXTURES = {
    "gemma": tuple(load_entry_file(_TF / "gemma" / f"{n}.json")
                   for n in ("warmth", "formality", "enthusiasm", "caution")),
    "qwen": tuple(load_entry_file(_TF / "qwen" / f"{n}.json")
                  for n in ("curiosity", "directness", "playfulness", "skepticism")),
}


def test_load_entries_ships_the_real_measured_concept_for_each_pairing():
    for entries in (GEMMA_ENTRIES, QWEN_ENTRIES):
        assert len(entries) == 1
        assert entries[0].concept_id == "pro-american-exceptionalism"


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


def test_no_shipped_concept_id_is_a_leftover_fake_placeholder():
    """The eight FAKE placeholders must not reappear in the build. They are
    test data now (tests/fixtures/), and shipping one again would put an
    invented concept in front of a user beside a measured one."""
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        assert not entry.concept_id.upper().startswith("FAKE-")


def test_pairing_ids_name_the_ratified_model_and_sae():
    assert PAIRING_ID_FOR_MODEL_KEY["gemma"] == "gemma-3-12b-it+gemma-scope-2-12b-it"
    assert PAIRING_ID_FOR_MODEL_KEY["qwen"] == "qwen-3.5-27b+SAE-Res-Qwen3.5-27B-W80K-L0_100"


def test_no_shipped_entry_claims_attested_provenance():
    """Product-repo invariant, independent of the canonical release gate:
    nothing shipped here may even CLAIM attestation, regardless of whether
    it would also fail on evidence. The shipped concepts are measured but
    not causally validated, so CANDIDATE is the most they may assert."""
    for entry in (*GEMMA_ENTRIES, *QWEN_ENTRIES):
        assert entry.provenance is not Provenance.ATTESTED
        assert entry.provenance is Provenance.CANDIDATE


def test_strength_levels_resolve_to_distinct_absolute_values():
    entry = GEMMA_ENTRIES[0]
    direction = entry.calibrated_directions[0]
    values = {
        strength: resolve_control(entry, direction=direction, strength=strength).targets[0].absolute_value
        for strength in ("low", "medium", "high")
    }
    assert len(set(values.values())) == 3


def test_at_least_one_concept_per_pairing_has_a_single_calibrated_direction():
    """Demonstrates the disabled-control acceptance case against a real
    decoded document, not a hand-built vector. Read from tests/fixtures/:
    the shipped set is not obliged to contain a one-direction concept."""
    for entries in SHAPE_FIXTURES.values():
        assert any(len(e.calibrated_directions) == 1 for e in entries)


def test_at_least_one_concept_per_pairing_is_capability_limited_or_prohibited():
    """Demonstrates the PROHIBITED/CAPABILITY_LIMIT surfacing requirement
    against real decoded documents. Read from tests/fixtures/: the shipped
    set is not obliged to contain a non-executable direction."""
    from sae_concept_lab.canonical.concept_bundle import check_direction_executable

    for entries in SHAPE_FIXTURES.values():
        found = any(
            not check_direction_executable(e, d).executable
            for e in entries
            for d in e.calibrated_directions
        )
        assert found
