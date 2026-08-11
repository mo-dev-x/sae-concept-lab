"""Strict, versioned JSON encoding/decoding for bundle entries.

Entries live as explicit files under `configs/concept_lab/bundles/`. This module
reads a NAMED FILE and nothing else.

  NO DIRECTORY SCANNING. There is deliberately no `load_all()`, no glob, and no
  discovery. A build must name every entry it loads, so a file dropped into the
  directory cannot enter a release by being present. This also means an
  unreviewed entry cannot appear in the UI without someone editing a list.

  STRICT DECODING AT EVERY LEVEL. An unknown `schema_version` is refused rather
  than decoded on the assumption that it is probably compatible. Unknown fields
  are refused rather than ignored, because a typo would otherwise silently drop
  the value it was meant to set -- `unit_sorce: "corpus max"` reading as "no
  source" is precisely the accident that must be impossible. Missing required
  fields are refused rather than defaulted.

  NULLABLE IS NOT OPTIONAL. `calibration_provenance`, a null direction, and the
  three conditional spec fields are all written explicitly as `null`. The key is
  always required; only its value may be null. An absent key is a malformed
  file, not an unstated default.

  ROUND-TRIP PRESERVES CANONICAL IDENTITY. decode(encode(e)) has the same
  `audit_fingerprint()` as e.

No real entry content is added by this module, and it creates no files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import BundleDecodeError, SchemaVersionError
from .schema import (
    BundleEntry,
    CalibrationProvenance,
    Direction,
    DirectionRecord,
    EvidenceRef,
    Operation,
    PositionMode,
    Provenance,
    Spec,
    Strength,
    Target,
    Unit,
)

# Documentation only -- nothing in this module scans it.
BUNDLES_DIR = Path("configs") / "concept_lab" / "bundles"

SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

_ENTRY_REQUIRED = frozenset({
    "schema_version", "concept_id", "pairing_id", "positions", "provenance",
    "calibration_provenance", "directions",
})
_CALIBRATION_PROVENANCE_REQUIRED = frozenset({
    "calibrated_by", "calibrated_at", "evidence",
})
_EVIDENCE_REQUIRED = frozenset({"artifact_type", "artifact_hash"})
_DIRECTION_REQUIRED = frozenset({"targets", "specs"})
_TARGET_REQUIRED = frozenset({"sae_id", "layer", "feature_idx", "weight"})
_SPEC_REQUIRED = frozenset({"operation", "value", "unit", "unit_source"})


def _check_keys(obj: Any, required: frozenset[str], where: str) -> dict[str, Any]:
    """Exact key set: nothing missing, nothing extra. There are no optional
    fields at any level of this format."""
    if not isinstance(obj, dict):
        raise BundleDecodeError(f"{where}: expected an object, got {type(obj).__name__}")
    keys = set(obj)
    missing = sorted(required - keys)
    if missing:
        raise BundleDecodeError(f"{where}: missing required field(s) {missing}")
    unknown = sorted(keys - required)
    if unknown:
        raise BundleDecodeError(
            f"{where}: unknown field(s) {unknown}. Unknown fields are refused, not "
            f"ignored: a misspelled field would otherwise silently fall back to a "
            f"default nobody chose.")
    return obj


def _enum(cls, raw: Any, where: str):
    try:
        return cls(raw)
    except ValueError:
        valid = [m.value for m in cls]
        raise BundleDecodeError(
            f"{where}: {raw!r} is not a valid {cls.__name__}; valid: {valid}") from None


def _decode_evidence(raw: Any, where: str) -> EvidenceRef:
    obj = _check_keys(raw, _EVIDENCE_REQUIRED, where)
    return EvidenceRef(artifact_type=obj["artifact_type"],
                       artifact_hash=obj["artifact_hash"])


def _decode_calibration_provenance(raw: Any, where: str) -> CalibrationProvenance:
    obj = _check_keys(raw, _CALIBRATION_PROVENANCE_REQUIRED, where)
    evidence = obj["evidence"]
    if not isinstance(evidence, list):
        raise BundleDecodeError(f"{where}.evidence: expected a list")
    return CalibrationProvenance(
        calibrated_by=obj["calibrated_by"],
        calibrated_at=obj["calibrated_at"],
        evidence=tuple(_decode_evidence(e, f"{where}.evidence[{i}]")
                       for i, e in enumerate(evidence)),
    )


def _decode_target(raw: Any, where: str) -> Target:
    obj = _check_keys(raw, _TARGET_REQUIRED, where)
    return Target(sae_id=obj["sae_id"], layer=obj["layer"],
                  feature_idx=obj["feature_idx"], weight=obj["weight"])


def _decode_spec(raw: Any, where: str) -> Spec:
    obj = _check_keys(raw, _SPEC_REQUIRED, where)
    unit = obj["unit"]
    return Spec(
        operation=_enum(Operation, obj["operation"], f"{where}.operation"),
        value=obj["value"],
        unit=None if unit is None else _enum(Unit, unit, f"{where}.unit"),
        unit_source=obj["unit_source"],
    )


def _decode_direction(raw: Any, where: str) -> DirectionRecord:
    obj = _check_keys(raw, _DIRECTION_REQUIRED, where)
    targets = obj["targets"]
    if not isinstance(targets, list):
        raise BundleDecodeError(f"{where}.targets: expected a list")
    specs_raw = obj["specs"]
    if not isinstance(specs_raw, dict):
        raise BundleDecodeError(f"{where}.specs: expected an object keyed by strength")
    expected = {s.value for s in Strength}
    missing = sorted(expected - set(specs_raw))
    if missing:
        raise BundleDecodeError(f"{where}.specs: missing required field(s) {missing}")
    unknown = sorted(set(specs_raw) - expected)
    if unknown:
        raise BundleDecodeError(f"{where}.specs: unknown field(s) {unknown}")
    return DirectionRecord(
        targets=tuple(_decode_target(t, f"{where}.targets[{i}]")
                      for i, t in enumerate(targets)),
        specs={s: _decode_spec(specs_raw[s.value], f"{where}.specs[{s.value!r}]")
               for s in Strength},
    )


def encode_entry(entry: BundleEntry, *, indent: int | None = 2) -> str:
    """Serializes an entry. `indent=None` gives the canonical compact form."""
    if indent is None:
        return entry.canonical_json()
    return json.dumps(entry.as_dict(), indent=indent, sort_keys=True,
                      ensure_ascii=False)


def decode_entry(text: str, *, where: str = "<string>") -> BundleEntry:
    """Strictly decodes an entry. Refuses unknown versions and unknown fields."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BundleDecodeError(f"{where}: not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundleDecodeError(f"{where}: top level must be an object")

    version = raw.get("schema_version")
    if version is None:
        raise BundleDecodeError(f"{where}: missing required field 'schema_version'")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"{where}: schema_version {version!r} is not supported by this build "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}). Refused rather than "
            f"decoded on the assumption of compatibility.")

    obj = _check_keys(raw, _ENTRY_REQUIRED, where)
    directions_raw = obj["directions"]
    if not isinstance(directions_raw, dict):
        raise BundleDecodeError(
            f"{where}.directions: expected an object keyed by direction")
    expected = {d.value for d in Direction}
    missing = sorted(expected - set(directions_raw))
    if missing:
        raise BundleDecodeError(
            f"{where}.directions: missing required field(s) {missing}; a direction "
            f"that is not calibrated must be written explicitly as null")
    unknown = sorted(set(directions_raw) - expected)
    if unknown:
        raise BundleDecodeError(f"{where}.directions: unknown field(s) {unknown}")

    cp_raw = obj["calibration_provenance"]
    return BundleEntry(
        concept_id=obj["concept_id"],
        pairing_id=obj["pairing_id"],
        positions=_enum(PositionMode, obj["positions"], f"{where}.positions"),
        provenance=_enum(Provenance, obj["provenance"], f"{where}.provenance"),
        directions={
            d: (None if directions_raw[d.value] is None
                else _decode_direction(directions_raw[d.value],
                                       f"{where}.directions[{d.value!r}]"))
            for d in Direction
        },
        calibration_provenance=(
            None if cp_raw is None
            else _decode_calibration_provenance(
                cp_raw, f"{where}.calibration_provenance")),
        schema_version=version,
    )


def load_entry_file(path: Path | str) -> BundleEntry:
    """Loads ONE explicitly named file. Never scans a directory."""
    p = Path(path)
    if not p.is_file():
        raise BundleDecodeError(f"bundle entry file not found: {p}")
    return decode_entry(p.read_text(encoding="utf-8"), where=p.name)


def load_entry_files(paths: tuple[Path | str, ...]) -> tuple[BundleEntry, ...]:
    """Loads an EXPLICIT list of files, in the order given.

    Takes a list, not a directory, so nothing is loaded by virtue of existing on
    disk.
    """
    return tuple(load_entry_file(p) for p in paths)
