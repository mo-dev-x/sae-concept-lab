"""Conformance pack for the concept-bundle contract: build it, or check an
implementation against it.

WHY THIS EXISTS. `qwen-sae-interp` is the canonical scientific source for the
concept-bundle contract. A separate product repository will later carry an
extracted copy of the minimum runtime surface, and "we copied it carefully"
is not evidence. This module freezes a set of synthetic vectors -- inputs plus
the exact outputs and exact refusals the canonical implementation produces --
so an extracted copy can be checked against the original mechanically rather
than by reading the diff.

TWO PATHS, DELIBERATELY SEPARATE.

  build_pack()   runs in THIS repository only. It constructs the vectors from
                 the live implementation and the package fixtures, and records
                 what the implementation actually did.

  verify_pack()  replays the frozen pack against ANY importable package that
                 claims to implement the contract, canonical or extracted. It
                 reads inputs from the pack's own JSON and never touches the
                 fixtures module, so an extractor does not have to copy the
                 fixtures to prove conformance.

The split is what makes the pack evidence: if verification rebuilt its inputs
from the same fixtures that produced its expectations, an extraction that
dropped a rule and the fixtures exercising it would still pass.

COMPARISON MODE. Where a serialization is defined by the contract -- canonical
entry JSON, the executor payload, the two view dicts, every fingerprint -- the
comparison is byte-for-byte on the canonical encoding. Where byte equality is
not meaningful -- execution grouping, direction availability, publishability
reasons -- the comparison is structural on sorted, typed values. Exception
classes are compared BY NAME, and classifications by their string value, so a
correctly extracted package is not failed for having a different module path.

DETERMINISTIC, OFFLINE, CPU-ONLY. No clock, no random source, no network, no
model weights, no GPU. The only filesystem access is reading and writing the
pack itself; evidence resolution in the vectors runs against an in-memory
registry described by the vector, never against the repository registry.

FIXTURE STATUS. Every entry in the pack is scientifically meaningless: invented
ids, round feature indices, invented doses, and no real feature membership. Most
carry `provenance: "fake"` and are refused by the publication gate outright. The
few that must pass the gate -- the vectors covering attested evidence resolution
and one-direction public availability -- cannot be published in any real build
either, because their evidence resolves only against the in-memory registry the
vector carries and is absent from the repository registry. A test asserts
exactly that.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import from THIS checkout, ahead of any installed copy of the same package.
# Verifying an implementation against a pack while silently importing a
# different, installed copy of it is the exact failure this harness exists to
# prevent, and it is invisible in the output when it happens.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PACK_DIR = REPO_ROOT / "conformance" / "concept_bundle"
VECTORS_PATH = PACK_DIR / "vectors.json"
INVENTORY_PATH = PACK_DIR / "export_inventory.json"

CANONICAL_PACKAGE = "interplab.concept_bundle"
CANONICAL_REPOSITORY = "qwen-sae-interp"
CANONICAL_BRANCH = "eng3/concept-bundle"

#: The commit this pack succeeds. It is NOT where the module hashes below come
#: from: those belong to the tree of the commit that adds or regenerates the
#: pack, which cannot be named inside the pack (a file cannot contain the hash of
#: the commit containing it). `frozen_at_commit` carries that, stamped by an
#: immediately following commit.
CONTRACT_BASE_COMMIT = "1f617f30e6272dd1cdb344948120acb844c6f459"

PACK_VERSION = "1.0"
#: The codec/schema version every vector in this pack is written against.
SCHEMA_VERSION = "1.0"

PYTHON_REQUIRES = ">=3.11,<4"

#: The modules an extracted runtime needs, and nothing more. `fixtures.py` and
#: this script are deliberately absent: one is invented data, the other is the
#: verification harness.
MINIMUM_EXPORT_MODULES: tuple[tuple[str, str], ...] = (
    ("interplab/concept_bundle/__init__.py",
     "public surface; re-exports every name the runtime and UI use"),
    ("interplab/concept_bundle/errors.py",
     "typed refusals, including the two runtime classifications"),
    ("interplab/concept_bundle/schema.py",
     "BundleEntry, DirectionRecord, Spec, Target, provenance and evidence types"),
    ("interplab/concept_bundle/codec.py",
     "strict versioned JSON for explicitly named files; no directory scanning"),
    ("interplab/concept_bundle/runtime.py",
     "(sae_id, layer) execution grouping and the v1 capability ceiling"),
    ("interplab/concept_bundle/resolver.py",
     "resolution arithmetic, shared Public/Advanced state, fingerprints"),
    ("interplab/concept_bundle/evidence.py",
     "evidence-reference resolution against a registry tree"),
    ("interplab/concept_bundle/release.py",
     "fail-closed publication gate and development-stub exposure"),
)

EXCLUDED_FROM_EXPORT: tuple[tuple[str, str], ...] = (
    ("interplab/concept_bundle/fixtures.py",
     "invented data. Not runtime. An extracted product must ship no fixtures."),
    ("scripts/concept_bundle_conformance.py",
     "verification harness. Copy it to RUN the check, not as part of the "
     "runtime surface."),
    ("tests/concept_bundle_helpers.py",
     "test-only construction, including the gate-passing object deliberately "
     "kept out of the production package."),
)


# ---------------------------------------------------------------------------
# canonical encoding helpers
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """The one encoding used for every byte-for-byte comparison in this pack."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_text_lf(path: Path, text: str) -> bytes:
    """Writes with LF endings unconditionally.

    A hash-addressed artefact written through Python's text mode on Windows
    would carry CRLF and hash differently from the same content on Linux, which
    would make every inventory hash platform-dependent.
    """
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# the API surface a conforming package must expose
# ---------------------------------------------------------------------------

REQUIRED_API: tuple[str, ...] = (
    "Direction", "Strength", "Exposure",
    "decode_entry", "encode_entry",
    "resolve_control", "require_single_execution_group",
    "check_direction_executable", "executable_directions",
    "evaluate_publishability", "select_layout_entries",
    # Evidence must be resolved by reading content and recomputing its digest.
    # `InMemoryEvidenceRegistry` is deliberately NOT required: it is a test
    # double, and a product is not obliged to ship one.
    "EvidenceRef", "RepositoryEvidenceRegistry", "content_digest",
    # A record that hashes correctly is not thereby a record.
    "record_validity_problems", "PUBLICATION_RECORD_FIELDS",
    # The mandatory release wording and the two labels. An extracted package that
    # renders its own words about what was checked is not this contract.
    "RELEASE_EVIDENCE_STATEMENT", "EVIDENCE_VERIFICATION_SENTENCE",
    "PAYLOAD_LIMIT_SENTENCE", "RAW_SHA256_LABEL", "PAYLOAD_HASH_LABEL",
    "PROHIBITED_RELEASE_CLAIMS", "prohibited_release_claims",
)


# ---------------------------------------------------------------------------
# registry trees, materialized from the vector's own declaration
# ---------------------------------------------------------------------------
# Evidence vectors describe FILES, not digests, because the property under test
# is that the resolver reads bytes and hashes them. A vector that handed the
# resolver a digest could not distinguish verification from transcription -- which
# is the defect this pack was extended to catch.

def _safe_relative(path: str) -> Path:
    """Refuses a vector that would write outside the directory it is given.

    The pack is data, and a harness that writes data-driven paths without
    checking them is the same class of defect as a resolver that joins an
    unchecked artifact_type.
    """
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe vector file path: {path!r}")
    return candidate


@contextlib.contextmanager
def materialized_registry(api, data: dict[str, Any]):
    """Writes the vector's declared files and yields a registry over the root.

    `files` land under the registry root. `outside_files` land in a sibling
    directory, which is how the path-escape vector places a correctly sealed
    artifact somewhere the resolver must refuse to read.
    """
    with tempfile.TemporaryDirectory(prefix="cb-conformance-") as temporary:
        base = Path(temporary)
        for key, subdirectory in (("files", "registry"), ("outside_files", "outside")):
            for entry in data.get(key) or ():
                target = base / subdirectory / _safe_relative(entry["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(entry["text"].encode("utf-8"))
        yield api.RepositoryEvidenceRegistry(base / "registry")


def _unvalidated_evidence_ref(api, ref: dict[str, str]):
    """An `EvidenceRef` assembled WITHOUT the schema's validation.

    The schema refuses a traversing or uppercase `artifact_type` at construction
    and the codec refuses it at decoding, which is where they should be caught.
    The resolver's own barrier still has to be exercised, because a reference can
    reach it without having passed either -- so these vectors reach past
    validation deliberately, and say so in their input.
    """
    instance = object.__new__(api.EvidenceRef)
    for field, value in ref.items():
        object.__setattr__(instance, field, value)
    return instance


def _evidence_ref(api, data: dict[str, Any]):
    if data.get("unvalidated_ref"):
        return _unvalidated_evidence_ref(api, data["ref"])
    return api.EvidenceRef(**data["ref"])


def _resolution_record(resolution) -> dict[str, Any]:
    """Every field of a resolution except the root, which is a temporary path.

    `location` is root-relative by contract, so it IS comparable; the root is
    not, and is excluded rather than normalized away.
    """
    identity = resolution.record_identity
    return {
        "status": str(resolution.status),
        "resolved": resolution.resolved,
        "content_verified": resolution.content_verified,
        "location": resolution.location,
        "detail": resolution.detail,
        "recomputed_digest": resolution.recomputed_digest,
        "declared_digest": resolution.declared_digest,
        "raw_sha256": resolution.raw_sha256,
        "digest_comparison": resolution.digest_comparison,
        "record_validity_problems": list(resolution.record_validity_problems),
        "record_identity": None if identity is None else identity.as_dict(),
        "payload_hashes": [c.as_dict() for c in resolution.payload_hash_claims],
        "as_record": canonical_json(resolution.as_record()),
    }


def load_api(package: str):
    """Imports a package claiming to implement the contract.

    Missing names are reported as a conformance failure rather than an
    ImportError halfway through a run, so an extractor sees the whole gap at
    once.
    """
    module = importlib.import_module(package)
    missing = [name for name in REQUIRED_API if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"package {package!r} is missing required contract names: {missing}")
    return module


def _denominator_source(entries: list[dict]):
    """Builds a denominator lookup from a vector's own declaration.

    Keyed on the unit's STRING value rather than an enum member, so the runner
    does not assume the target package's enum identity -- only that a Unit has a
    `.value`, which any StrEnum implementation satisfies.
    """
    table = {
        (e["unit"], e["unit_source"], e["sae_id"], e["layer"], e["feature_idx"]):
            float(e["denominator"])
        for e in entries
    }

    def source(*, unit, unit_source, target):
        key = (getattr(unit, "value", unit), unit_source, target.sae_id,
               target.layer, target.feature_idx)
        if key not in table:
            raise LookupError(key)
        return table[key]

    return source


def _exception_record(exc: BaseException) -> dict[str, Any]:
    """Class NAME, classification and message -- never the module path.

    An extracted package legitimately has different module paths; it must not
    have different refusals.
    """
    classification = getattr(type(exc), "CLASSIFICATION", None)
    return {
        "exception": type(exc).__name__,
        "classification": None if classification is None else str(classification),
        "message": str(exc),
    }


# ---------------------------------------------------------------------------
# vector construction -- CANONICAL REPOSITORY ONLY
# ---------------------------------------------------------------------------

FAKE_ARTIFACT_TYPE = "conformance_artifact_not_in_the_registry"


def _fake_evidence_record(api, *, artifact_type: str = FAKE_ARTIFACT_TYPE,
                          note: str = "meaningless conformance content") -> dict:
    """A registry-envelope-shaped artifact whose digest is COMPUTED, not typed.

    Invented content under an artifact type that has no directory in the
    repository registry, so a vector that passes the gate here cannot pass it in
    a real build.
    """
    record = {
        "artifact_type": artifact_type,
        "schema_version": 1,
        "created_at": "2000-01-01T00:00:00Z",
        "created_by": {"run_id": "r20000101-0000-fake", "code_commit": "0" * 40,
                       "entrypoint": "scripts.concept_bundle_conformance",
                       "host": "conformance"},
        "subject": [],
        "payload": {"note": note},
    }
    record["self_hash"] = api.content_digest(record)
    return record


def _registry_files(api, record: dict) -> list[dict[str, str]]:
    """One artifact, written where its own content digest says it belongs."""
    hash12 = api.content_digest(record).removeprefix("sha256:")[:12]
    return [{"path": f"{record['artifact_type']}/{hash12}.json",
             "text": canonical_json(record)}]


def _attested_entry_document(api, *, concept_id: str, one_direction: bool) -> str:
    """A gate-passing entry, built here rather than imported from fixtures.

    The production fixtures module deliberately contains nothing that can pass
    the publication gate, and that stays true: this object lives in the
    conformance harness and is serialized into a data file, not into an
    importable template. Every value is meaningless -- the point is only that
    none of them is *labelled* fake, which is what the marker sniffing catches.
    """
    unit_source = "corpus maximum, conformance vector"
    spec = {"operation": "clamp", "value": None, "unit": "corpus_max_multiple",
            "unit_source": unit_source}
    direction = {
        "targets": [{"sae_id": "conformance-sae-not-a-real-sae", "layer": 11,
                     "feature_idx": 4242, "weight": 1.0}],
        "specs": {"low": {**spec, "value": 0.5},
                  "medium": {**spec, "value": 1.0},
                  "high": {**spec, "value": 2.0}},
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "concept_id": concept_id,
        "pairing_id": "conformance-pairing/not-a-real-pairing",
        "positions": "generated_only",
        "provenance": "attested",
        "calibration_provenance": {
            "calibrated_by": "conformance harness",
            "calibrated_at": "2026-01-01T00:00:00+00:00",
            # The FULL 64-hex content digest of the evidence record, recomputed
            # rather than typed. Publication refuses a prefix reference.
            "evidence": [{"artifact_type": FAKE_ARTIFACT_TYPE,
                          "artifact_hash": api.content_digest(
                              _fake_evidence_record(api))}],
        },
        "directions": {"amplify": direction,
                       "suppress": None if one_direction else direction},
    }
    # Round-trips through the codec so the stored document is the canonical
    # encoding rather than whatever this function happened to type.
    return api.encode_entry(api.decode_entry(canonical_json(document)), indent=None)


def _mutate(document: str, mutate) -> str:
    raw = json.loads(document)
    mutate(raw)
    return canonical_json(raw)


def build_pack() -> dict[str, Any]:
    """Constructs the pack from the live canonical implementation."""
    api = load_api(CANONICAL_PACKAGE)
    fixtures = importlib.import_module(f"{CANONICAL_PACKAGE}.fixtures")

    def doc(entry) -> str:
        return api.encode_entry(entry, indent=None)

    base = doc(fixtures.fake_entry())
    vectors: list[dict[str, Any]] = []

    # -- strict codec acceptance ----------------------------------------
    accepted = [
        ("codec.accept.two_direction_clamp", fixtures.fake_entry(),
         "both directions calibrated, clamp in multiple units"),
        ("codec.accept.ablate_direction", fixtures.fake_ablate_entry(),
         "an ablating direction: value, unit and unit_source all null"),
        ("codec.accept.absolute_activation", fixtures.fake_absolute_activation_entry(),
         "clamp in raw activation units: unit_source prohibited"),
        ("codec.accept.null_suppress", fixtures.fake_entry_with_null_suppress(),
         "one calibrated direction; the other is explicitly null"),
        ("codec.accept.both_directions_null",
         fixtures.fake_entry_with_no_calibrated_directions(),
         "an authoring skeleton: valid, inert, unpublishable"),
        ("codec.accept.divergent_memberships",
         fixtures.fake_entry_with_divergent_memberships(),
         "the two directions drive different features"),
        ("codec.accept.multi_group", fixtures.fake_multi_group_entry(),
         "targets on two layers: valid schema, refused by runtime v1"),
        ("codec.accept.two_saes_at_one_layer",
         fixtures.fake_two_saes_at_one_layer_entry(),
         "two SAE identities at one layer: valid schema, prohibited at runtime"),
    ]
    for vector_id, entry, description in accepted:
        document = doc(entry)
        vectors.append({
            "id": vector_id, "kind": "codec_accept", "schema_version": SCHEMA_VERSION,
            "description": description,
            "input": {"document": document},
            "expected": {"canonical_json": entry.canonical_json(),
                         "audit_fingerprint": entry.audit_fingerprint(),
                         "calibrated_directions":
                             [d.value for d in entry.calibrated_directions]},
        })

    # -- strict codec rejection -----------------------------------------
    ablate_doc = doc(fixtures.fake_ablate_entry())
    attested_doc = _attested_entry_document(
        api, concept_id="conformance.vector.attested", one_direction=False)

    def _set(key, value):
        def apply(raw):
            raw[key] = value
        return apply

    def _drop(key):
        def apply(raw):
            del raw[key]
        return apply

    rejected = [
        ("codec.reject.unknown_schema_version", base,
         _set("schema_version", "2.0"),
         "an unknown version is refused, not decoded on the assumption of "
         "compatibility"),
        ("codec.reject.missing_schema_version", base, _drop("schema_version"),
         "the version is required"),
        ("codec.reject.unknown_top_level_field", base, _set("surprise", True),
         "unknown fields are refused, not ignored"),
        ("codec.reject.misspelled_field", base, _set("provenence", "attested"),
         "a typo must not read as 'no opinion' and inherit a default"),
        ("codec.reject.missing_required_field", base, _drop("provenance"),
         "a missing required field is refused, not defaulted"),
        ("codec.reject.unknown_target_field", base,
         lambda raw: raw["directions"]["amplify"]["targets"][0]
         .__setitem__("extra", 1),
         "strictness applies at every object level, not only the top"),
        ("codec.reject.missing_direction_key", base,
         lambda raw: raw["directions"].pop("suppress"),
         "an uncalibrated direction must be written explicitly as null"),
        ("codec.reject.unknown_direction_key", base,
         lambda raw: raw["directions"].__setitem__("sideways", None),
         "exactly amplify and suppress"),
        ("codec.reject.missing_strength", base,
         lambda raw: raw["directions"]["amplify"]["specs"].pop("medium"),
         "all three control positions must be authored"),
        ("codec.reject.missing_conditional_spec_field", base,
         lambda raw: raw["directions"]["amplify"]["specs"]["low"].pop("unit_source"),
         "nullable is not optional: the key is required, the value may be null"),
        ("codec.reject.invalid_operation", base,
         lambda raw: raw["directions"]["amplify"]["specs"]["low"]
         .__setitem__("operation", "teleport"),
         "operations are a closed set"),
        ("codec.reject.invalid_unit", base,
         lambda raw: raw["directions"]["amplify"]["specs"]["low"]
         .__setitem__("unit", "maxact_relative"),
         "units are a closed set"),
        ("codec.reject.invalid_position_mode", base,
         _set("positions", "prompt_only"),
         "position modes are a closed set"),
        ("codec.reject.ablate_with_a_value", ablate_doc,
         lambda raw: raw["directions"]["suppress"]["specs"]["high"]
         .__setitem__("value", 2.0),
         "ablation has no dose"),
        ("codec.reject.negative_weight", base,
         lambda raw: raw["directions"]["amplify"]["targets"][0]
         .__setitem__("weight", -1.0),
         "no numeric field may be negative"),
        ("codec.reject.zero_weight", base,
         lambda raw: raw["directions"]["amplify"]["targets"][0]
         .__setitem__("weight", 0.0),
         "a zero-weight target would contribute nothing silently"),
        ("codec.reject.negative_layer", base,
         lambda raw: raw["directions"]["amplify"]["targets"][0]
         .__setitem__("layer", -3),
         "no numeric field may be negative"),
        ("codec.reject.bad_artifact_hash", attested_doc,
         lambda raw: raw["calibration_provenance"]["evidence"][0]
         .__setitem__("artifact_hash", "NOTAHASH"),
         "evidence digests must match the contract pattern"),
        ("codec.reject.uppercase_artifact_type", attested_doc,
         lambda raw: raw["calibration_provenance"]["evidence"][0]
         .__setitem__("artifact_type", "Census_Report"),
         "an artifact_type names a registry directory: on a case-insensitive "
         "filesystem this is the same directory as its lowercase twin and a "
         "different one on Linux"),
        ("codec.reject.artifact_type_with_a_separator", attested_doc,
         lambda raw: raw["calibration_provenance"]["evidence"][0]
         .__setitem__("artifact_type", "../outside/run_card"),
         "a decoder that passed this through would hand the resolver a path "
         "component out of an untrusted document"),
        ("codec.reject.artifact_type_with_a_hyphen", attested_doc,
         lambda raw: raw["calibration_provenance"]["evidence"][0]
         .__setitem__("artifact_type", "census-report"),
         "one spelling of a registry directory name, not two"),
        ("codec.reject.unqualified_timestamp", attested_doc,
         lambda raw: raw["calibration_provenance"]
         .__setitem__("calibrated_at", "2026-01-01T00:00:00"),
         "a local wall clock is not a point in time"),
    ]
    for vector_id, document, mutate, description in rejected:
        mutated = _mutate(document, mutate)
        try:
            api.decode_entry(mutated)
        except Exception as exc:  # recording whatever it raises IS the observation
            record = _exception_record(exc)
        else:
            raise AssertionError(f"{vector_id} did not raise")
        vectors.append({
            "id": vector_id, "kind": "codec_reject", "schema_version": SCHEMA_VERSION,
            "description": description,
            "input": {"document": mutated},
            "expected": record,
        })

    # malformed input that never reaches the schema
    for vector_id, text, description in [
        ("codec.reject.malformed_json", "{not json", "unparseable input"),
        ("codec.reject.non_object_top_level", "[]", "the top level is an object"),
    ]:
        try:
            api.decode_entry(text)
        except Exception as exc:
            record = _exception_record(exc)
        else:
            raise AssertionError(f"{vector_id} did not raise")
        vectors.append({
            "id": vector_id, "kind": "codec_reject", "schema_version": SCHEMA_VERSION,
            "description": description,
            "input": {"document": text},
            "expected": record,
        })

    # -- resolved executor payloads --------------------------------------
    sample_denominators = [
        {"unit": "sample_max_multiple", "unit_source": fixtures.FAKE_UNIT_SOURCE,
         "sae_id": fixtures.FAKE_SAE_ID, "layer": 7, "feature_idx": 1000,
         "denominator": 4.0},
        {"unit": "sample_max_multiple", "unit_source": fixtures.FAKE_UNIT_SOURCE,
         "sae_id": fixtures.FAKE_SAE_ID, "layer": 7, "feature_idx": 1100,
         "denominator": 2.0},
    ]
    resolutions = [
        ("resolve.clamp.sample_max_multiple", fixtures.fake_entry(), "amplify",
         "high", sample_denominators,
         "clamp resolved as value x weight x the measured maximum"),
        ("resolve.clamp.absolute_activation",
         fixtures.fake_absolute_activation_entry(), "amplify", "medium", [],
         "absolute activation needs no denominator source at all"),
        ("resolve.ablate.no_arithmetic", fixtures.fake_ablate_entry(), "suppress",
         "high", [], "ablation carries no value, unit or denominator"),
        ("resolve.one_direction.amplify_only",
         fixtures.fake_entry_with_null_suppress(), "amplify", "low",
         sample_denominators,
         "the calibrated direction of a one-direction concept resolves normally"),
    ]
    for vector_id, entry, direction, strength, denominators, description in resolutions:
        state = api.resolve_control(
            entry, direction=direction, strength=strength,
            denominators=_denominator_source(denominators) if denominators
            else None)
        vectors.append({
            "id": vector_id, "kind": "resolve", "schema_version": SCHEMA_VERSION,
            "description": description,
            "input": {"document": doc(entry), "direction": direction,
                      "strength": strength, "denominators": denominators},
            "expected": {
                "execution_dict": canonical_json(state.execution_dict()),
                "public_view": canonical_json(state.public_view()),
                "advanced_view": canonical_json(state.advanced_view()),
                "execution_fingerprint": state.execution_fingerprint(),
                "state_fingerprint": state.state_fingerprint(),
                "entry_audit_fingerprint": state.entry_audit_fingerprint,
                "n_targets": state.n_targets,
            },
        })

    # -- refusals: null direction, and both runtime classifications ------
    refusals = [
        ("resolve.reject.null_direction", "resolve",
         fixtures.fake_entry_with_null_suppress(), "suppress",
         "selecting an uncalibrated direction is refused, never resolved into "
         "a request that intervenes on nothing"),
        ("resolve.reject.both_directions_null", "resolve",
         fixtures.fake_entry_with_no_calibrated_directions(), "amplify",
         "an authoring skeleton executes in neither direction"),
        ("runtime.reject.multiple_sae_identities_at_one_layer", "runtime",
         fixtures.fake_two_saes_at_one_layer_entry(), "amplify",
         "PROHIBITED: composing two reconstructions of one residual stream is "
         "undefined"),
        ("runtime.reject.multiple_execution_groups", "runtime",
         fixtures.fake_multi_group_entry(), "amplify",
         "CAPABILITY_LIMIT: runtime v1 attaches one group per pass"),
        ("runtime.reject.null_direction", "runtime",
         fixtures.fake_entry_with_null_suppress(), "suppress",
         "the same refusal on the executor's pre-flight path"),
    ]
    for vector_id, path, entry, direction, description in refusals:
        try:
            if path == "resolve":
                api.resolve_control(entry, direction=direction, strength="low")
            else:
                api.require_single_execution_group(entry, direction)
        except Exception as exc:
            record = _exception_record(exc)
        else:
            raise AssertionError(f"{vector_id} did not raise")
        vectors.append({
            "id": vector_id,
            "kind": "resolve_reject" if path == "resolve" else "runtime_reject",
            "schema_version": SCHEMA_VERSION, "description": description,
            "input": {"document": doc(entry), "direction": direction,
                      "strength": "low"},
            "expected": record,
        })

    # -- accepted execution grouping -------------------------------------
    for vector_id, entry, direction, description in [
        ("runtime.accept.single_group", fixtures.fake_entry(), "amplify",
         "one SAE at one layer: every target executes in one batch"),
        ("runtime.accept.single_group_ablate", fixtures.fake_ablate_entry(),
         "suppress", "ablation groups exactly like a clamp"),
    ]:
        group = api.require_single_execution_group(entry, direction)
        vectors.append({
            "id": vector_id, "kind": "runtime_accept",
            "schema_version": SCHEMA_VERSION, "description": description,
            "input": {"document": doc(entry), "direction": direction},
            "expected": {"sae_id": group.sae_id, "layer": group.layer,
                         "feature_indices": list(group.feature_indices),
                         "executable_directions":
                             [d.value for d in api.executable_directions(entry)]},
        })

    # -- evidence resolution: read the bytes, recompute the digest -------
    good_record = _fake_evidence_record(api)
    good_files = _registry_files(api, good_record)
    good_digest = api.content_digest(good_record)
    good_hash12 = good_digest.removeprefix("sha256:")[:12]
    good_ref = {"artifact_type": FAKE_ARTIFACT_TYPE, "artifact_hash": good_digest}

    tampered_record = {**good_record, "payload": {"note": "edited after writing"}}
    other_type_record = _fake_evidence_record(
        api, artifact_type="conformance_other_type", note="a different artifact")

    # Hashes to exactly what its reference cites, and is still not a record.
    invalid_record = {k: v for k, v in good_record.items()
                      if k not in ("created_at", "self_hash")}
    invalid_digest = api.content_digest(invalid_record)
    invalid_record["self_hash"] = invalid_digest
    invalid_files = _registry_files(api, invalid_record)

    # A record carrying hashes of things nothing here reads.
    pointing_record = {k: v for k, v in good_record.items() if k != "self_hash"}
    pointing_record["subject"] = [
        {"content_hash": "sha256:" + "c" * 64,
         "location": "local:data/raw/not_read_by_this_package",
         "role": "corpus_manifest"}]
    pointing_record["self_hash"] = api.content_digest(pointing_record)
    pointing_files = _registry_files(api, pointing_record)

    evidence_cases = [
        ("evidence.accept.correct_content", good_files, [], good_ref,
         "content read, digest recomputed here, and it matches the reference"),
        ("evidence.accept.prefix_reference", good_files, [],
         {"artifact_type": FAKE_ARTIFACT_TYPE, "artifact_hash": good_hash12},
         "a 12-character reference is still verified against content, and is "
         "marked as a prefix match rather than a full one"),
        ("evidence.accept.payload_targets_are_labelled_not_verified",
         pointing_files, [],
         {"artifact_type": FAKE_ARTIFACT_TYPE,
          "artifact_hash": pointing_record["self_hash"]},
         "the record points at a corpus. That hash is emitted labelled "
         "'recorded, not revalidated' -- nothing here reads or rehashes it"),
        ("evidence.reject.invalid_record_missing_created_at", invalid_files, [],
         {"artifact_type": FAKE_ARTIFACT_TYPE, "artifact_hash": invalid_digest},
         "the digest matches the reference EXACTLY and the record is still "
         "refused: a correct digest says these are the bytes that were cited, "
         "not that they are an artifact"),
        ("evidence.reject.tampered_content_same_self_hash",
         [{"path": good_files[0]["path"], "text": canonical_json(tampered_record)}],
         [], good_ref,
         "the defect: payload edited, self_hash left untouched, file left where "
         "it was. A resolver that read the declaration would publish this"),
        ("evidence.reject.self_hash_typed_to_match",
         [{"path": good_files[0]["path"],
           "text": canonical_json({"artifact_type": FAKE_ARTIFACT_TYPE,
                                   "payload": {"note": "unrelated"},
                                   "self_hash": good_digest})}],
         [], good_ref,
         "unrelated content whose self_hash was simply typed to match"),
        ("evidence.reject.wrong_but_readable_artifact",
         [{"path": good_files[0]["path"],
           "text": canonical_json(other_type_record)}], [], good_ref,
         "a valid, self-consistent artifact of another type sitting exactly "
         "where this reference looks"),
        ("evidence.reject.missing_content", [], [], good_ref,
         "nothing at the address the reference names"),
        ("evidence.reject.empty_content",
         [{"path": good_files[0]["path"], "text": ""}], [], good_ref,
         "an empty file has no content to attest with"),
        ("evidence.reject.no_self_hash",
         [{"path": good_files[0]["path"],
           "text": canonical_json({k: v for k, v in good_record.items()
                                   if k != "self_hash"})}], [], good_ref,
         "declares no identity at all, which is not the same as declaring the "
         "wrong one"),
        ("evidence.reject.path_escape", [], good_files,
         {"artifact_type": f"../outside/{FAKE_ARTIFACT_TYPE}",
          "artifact_hash": good_digest},
         "a correctly sealed artifact outside the root, reachable only by "
         "traversal. The root is an access path, not a trust anchor. The "
         "reference reaches past schema validation deliberately: the resolver's "
         "barrier is the one being tested here", True),
        ("evidence.reject.uppercase_artifact_type", good_files, [],
         {"artifact_type": FAKE_ARTIFACT_TYPE.upper(),
          "artifact_hash": good_digest},
         "the same directory as its lowercase twin on Windows and macOS, a "
         "different one on Linux. Refused independently of the schema", True),
        ("evidence.reject.ambiguous_across_types",
         [*good_files,
          {"path": f"conformance_second_type/{good_hash12}.json",
           "text": canonical_json(good_record)}], [], good_ref,
         "one content address filed under two artifact types: the registry "
         "contradicting itself about what the content is"),
    ]
    for case in evidence_cases:
        vector_id, files, outside, ref, description = case[:5]
        data: dict[str, Any] = {"files": files, "outside_files": outside, "ref": ref}
        if len(case) > 5 and case[5]:
            data["unvalidated_ref"] = True
        with materialized_registry(api, data) as registry:
            resolution = registry.resolve(_evidence_ref(api, data))
        vectors.append({
            "id": vector_id, "kind": "evidence", "schema_version": SCHEMA_VERSION,
            "description": description,
            "input": data,
            "expected": _resolution_record(resolution),
        })

    # The other half of the same rule: the schema and the codec refuse these
    # references at construction, so a document carrying one never reaches the
    # resolver in the first place.
    for vector_id, ref, description in [
        ("schema.reject.traversing_artifact_type",
         {"artifact_type": f"../outside/{FAKE_ARTIFACT_TYPE}",
          "artifact_hash": good_digest},
         "refused before it can be a path"),
        ("schema.reject.uppercase_artifact_type",
         {"artifact_type": "Census_Report", "artifact_hash": good_digest},
         "refused before it can be a case collision"),
    ]:
        try:
            api.EvidenceRef(**ref)
        except Exception as exc:
            record = _exception_record(exc)
        else:
            raise AssertionError(f"{vector_id} did not raise")
        vectors.append({
            "id": vector_id, "kind": "schema_reject",
            "schema_version": SCHEMA_VERSION, "description": description,
            "input": {"evidence_ref": ref},
            "expected": record,
        })

    # -- publication: exact acceptance and rejection consequences --------
    one_direction_doc = _attested_entry_document(
        api, concept_id="conformance.vector.one.direction", one_direction=True)
    prefix_doc = _mutate(
        attested_doc,
        lambda raw: raw["calibration_provenance"]["evidence"][0].__setitem__(
            "artifact_hash", good_hash12))
    unprefixed_doc = _mutate(
        attested_doc,
        lambda raw: raw["calibration_provenance"]["evidence"][0].__setitem__(
            "artifact_hash", good_digest.removeprefix("sha256:")))
    invalid_record_doc = _mutate(
        attested_doc,
        lambda raw: raw["calibration_provenance"]["evidence"][0].__setitem__(
            "artifact_hash", invalid_digest))
    pointing_doc = _mutate(
        attested_doc,
        lambda raw: raw["calibration_provenance"]["evidence"][0].__setitem__(
            "artifact_hash", pointing_record["self_hash"]))
    publications = [
        ("publish.accept.attested_with_verified_evidence", attested_doc,
         good_files,
         "ATTESTED provenance plus evidence read and hashed: the only way to "
         "publish"),
        ("publish.accept.one_calibrated_direction", one_direction_doc, good_files,
         "one calibrated direction is sufficient; the other stays null"),
        ("publish.reject.missing_evidence", attested_doc, [],
         "attested, and still refused: the cited artifact is not there to hash"),
        ("publish.reject.tampered_evidence", attested_doc,
         [{"path": good_files[0]["path"], "text": canonical_json(tampered_record)}],
         "attested, the artifact is present, its self_hash still matches the "
         "reference -- and its content does not hash to it"),
        ("publish.accept.record_pointing_at_unread_targets", pointing_doc,
         pointing_files,
         "the record names a corpus this package never reads. It publishes, and "
         "the release output labels that hash 'recorded, not revalidated' rather "
         "than omitting it"),
        ("publish.reject.prefix_digest_reference", prefix_doc, good_files,
         "verified by content, but cited by a 12-character prefix; publication "
         "requires the full digest"),
        ("publish.reject.digest_without_algorithm_prefix", unprefixed_doc,
         good_files,
         "all 64 characters and no 'sha256:'. A bare digest does not say what "
         "produced it"),
        ("publish.reject.invalid_evidence_record", invalid_record_doc,
         invalid_files,
         "the cited content hashes to exactly what the reference names, and the "
         "record is missing created_at. Integrity does not substitute for "
         "validity"),
        ("publish.reject.fake_provenance", base, good_files,
         "a fixture is refused on provenance and on its own marker text"),
        ("publish.reject.both_directions_null",
         _mutate(attested_doc,
                 lambda raw: raw["directions"].update({"amplify": None,
                                                       "suppress": None})),
         good_files,
         "attested, evidence verified, and nothing to operate"),
    ]
    for vector_id, document, files, description in publications:
        entry = api.decode_entry(document)
        data = {"files": files, "outside_files": [], "document": document}
        with materialized_registry(api, data) as registry:
            decision = api.evaluate_publishability(entry, evidence_registry=registry)
        vectors.append({
            "id": vector_id, "kind": "publish", "schema_version": SCHEMA_VERSION,
            "description": description,
            "input": data,
            "expected": {
                "publishable": decision.publishable,
                "reasons": list(decision.reasons),
                "evidence_content_verified": decision.evidence_content_verified,
                "evidence": [
                    {"artifact_type": r.ref.artifact_type,
                     "artifact_hash": r.ref.artifact_hash,
                     "status": str(r.status),
                     "resolved": r.resolved,
                     "content_verified": r.content_verified,
                     "recomputed_digest": r.recomputed_digest,
                     "digest_comparison": r.digest_comparison,
                     "record_validity_problems": list(r.record_validity_problems)}
                    for r in decision.evidence],
                # Byte-for-byte: the release output is part of the contract, not
                # a presentation detail. The mandatory wording and both labels
                # are inside these two.
                "verification_record":
                    canonical_json(decision.content_verification_record()),
                "release_note": decision.render_release_evidence_note(),
            },
        })

    # -- catalog availability --------------------------------------------
    availability = [
        ("availability.one_direction_public", one_direction_doc, good_files,
         "a one-direction concept stays in the public catalog with the other "
         "control disabled"),
        ("availability.two_direction_public", attested_doc, good_files,
         "nothing disabled"),
        ("availability.tampered_evidence_is_development_only", attested_doc,
         [{"path": good_files[0]["path"], "text": canonical_json(tampered_record)}],
         "unverifiable evidence keeps a concept out of the public catalog"),
        ("availability.both_null_development_only", base, [],
         "a fake entry renders only under development exposure"),
    ]
    for vector_id, document, files, description in availability:
        entry = api.decode_entry(document)
        data = {"files": files, "outside_files": [], "document": document}
        with materialized_registry(api, data) as registry:
            public = api.select_layout_entries(
                (entry,), exposure=api.Exposure.RELEASE, evidence_registry=registry)
            (layout,) = api.select_layout_entries(
                (entry,), exposure=api.Exposure.DEVELOPMENT_STUBS,
                evidence_registry=registry)
        vectors.append({
            "id": vector_id, "kind": "availability",
            "schema_version": SCHEMA_VERSION, "description": description,
            "input": data,
            "expected": {
                "in_public_catalog": len(public) == 1,
                "available_directions":
                    [d.value for d in layout.available_directions],
                "unavailable_directions":
                    [d.value for d in layout.unavailable_directions],
                "is_development_stub": layout.is_development_stub,
                "block_reasons": list(layout.block_reasons),
            },
        })

    # -- fingerprint relations -------------------------------------------
    provenance_edited = _mutate(
        attested_doc,
        lambda raw: raw["calibration_provenance"].update(
            {"calibrated_by": "a corrected name"}))
    attested_denominators = [
        {"unit": "corpus_max_multiple",
         "unit_source": "corpus maximum, conformance vector",
         "sae_id": "conformance-sae-not-a-real-sae", "layer": 11,
         "feature_idx": 4242, "denominator": 3.0},
    ]
    fake_corpus_denominators = [
        {"unit": "corpus_max_multiple", "unit_source": fixtures.FAKE_UNIT_SOURCE,
         "sae_id": fixtures.FAKE_SAE_ID, "layer": 7, "feature_idx": 1000,
         "denominator": 10.0},
        {"unit": "corpus_max_multiple", "unit_source": fixtures.FAKE_UNIT_SOURCE,
         "sae_id": fixtures.FAKE_SAE_ID, "layer": 7, "feature_idx": 1100,
         "denominator": 20.0},
    ]
    relations = [
        ("fingerprints.execution_identity_ignores_provenance", attested_doc,
         provenance_edited, "amplify", "medium", attested_denominators,
         "correcting a calibrator changes the audit identity and nothing the "
         "model computes"),
        ("fingerprints.audit_identity_separates_entries", base,
         doc(fixtures.fake_ablate_entry()), "suppress", "high",
         fake_corpus_denominators,
         "two different entries have three different identities"),
    ]
    for (vector_id, left_doc, right_doc, direction, strength, denominators,
         description) in relations:
        source = _denominator_source(denominators) if denominators else None
        left = api.resolve_control(api.decode_entry(left_doc), direction=direction,
                                   strength=strength, denominators=source)
        right = api.resolve_control(api.decode_entry(right_doc), direction=direction,
                                    strength=strength, denominators=source)
        vectors.append({
            "id": vector_id, "kind": "fingerprint_relation",
            "schema_version": SCHEMA_VERSION, "description": description,
            "input": {"left_document": left_doc, "right_document": right_doc,
                      "direction": direction, "strength": strength,
                      "denominators": denominators},
            "expected": {
                "audit_fingerprints_equal":
                    left.entry_audit_fingerprint == right.entry_audit_fingerprint,
                "state_fingerprints_equal":
                    left.state_fingerprint() == right.state_fingerprint(),
                "execution_fingerprints_equal":
                    left.execution_fingerprint() == right.execution_fingerprint(),
                "left": {"audit": left.entry_audit_fingerprint,
                         "state": left.state_fingerprint(),
                         "execution": left.execution_fingerprint()},
                "right": {"audit": right.entry_audit_fingerprint,
                          "state": right.state_fingerprint(),
                          "execution": right.execution_fingerprint()},
            },
        })

    # -- mandatory release wording ---------------------------------------
    # Frozen as data because it is data: the sentences are ruled, not authored,
    # and an extracted package that renders its own words about what was checked
    # is not this contract however well its resolver behaves.
    vectors.append({
        "id": "wording.mandatory_release_statement", "kind": "release_wording",
        "schema_version": SCHEMA_VERSION,
        "description": (
            "the exact sentences every public and release rendering carries, "
            "adjacent and in this order, plus the two inline labels and the "
            "phrasings that are refused"),
        "input": {},
        "expected": {
            "statement": api.RELEASE_EVIDENCE_STATEMENT,
            "verification_sentence": api.EVIDENCE_VERIFICATION_SENTENCE,
            "payload_limit_sentence": api.PAYLOAD_LIMIT_SENTENCE,
            "separator_between_sentences": " ",
            "raw_sha256_label": api.RAW_SHA256_LABEL,
            "payload_hash_label": api.PAYLOAD_HASH_LABEL,
            "prohibited_claims": list(api.PROHIBITED_RELEASE_CLAIMS),
            # Refused phrasings, and the fact that the mandatory wording itself
            # passes the same checker -- a checker its own required text failed
            # would be deleted within a week.
            "refused_examples": [
                "All evidence verified.",
                "Artifacts verified against the registry.",
                "Content verified against the corpus.",
                "Hashes verified against the dataset.",
                "Evidence fully verified.",
                "The checkpoint was verified.",
                "Everything was verified.",
            ],
            "accepted_examples": [
                api.RELEASE_EVIDENCE_STATEMENT,
                api.PAYLOAD_LIMIT_SENTENCE,
                "The registry record was content-verified.",
                "The corpus was never verified.",
            ],
            "record_validity_fields": [f.as_dict()
                                       for f in api.PUBLICATION_RECORD_FIELDS],
        },
    })

    ids = [v["id"] for v in vectors]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise AssertionError(f"duplicate vector ids: {duplicates}")

    return {
        "pack_version": PACK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "canonical_repository": CANONICAL_REPOSITORY,
        "canonical_package": CANONICAL_PACKAGE,
        "contract_base_commit": CONTRACT_BASE_COMMIT,
        "generator": "scripts/concept_bundle_conformance.py",
        "determinism": "no clock, no random source, no network, no model weights",
        "fixture_status": (
            "Every entry here is scientifically meaningless. Most carry "
            "provenance 'fake' and are refused outright. The attested vectors "
            "cite evidence artifacts whose content each vector declares and "
            "which are written into a temporary registry at verification time; "
            "that content is invented and its artifact types have no directory "
            "in the repository registry, so no vector can publish in a real "
            "build."),
        "evidence_verification": (
            "Evidence vectors declare FILES, not digests. Resolution reads the "
            "bytes and recomputes the content digest in the artifact's own "
            "domain -- sha256 over canonical JSON of the artifact without its "
            "self_hash field -- and compares that to the reference. A "
            "self-declared self_hash never attests."),
        "record_validity": (
            "A correct digest says the bytes are the bytes that were cited. It "
            "does not say they are an artifact. Every canonical field the "
            "release path reads is checked separately for presence, type and "
            "emptiness, and a record that hashes correctly and fails that check "
            "is invalid_record -- not resolved, not publishable."),
        "publication_digest": (
            "A public release requires sha256: followed by all 64 lowercase hex "
            "characters. Prefix references stay fully supported for development "
            "and inspection and are verified by content exactly as full ones "
            "are; they are only unpublishable. The rule is applied "
            "unconditionally: no configuration, environment variable, CLI flag "
            "or call-site argument weakens it."),
        "scope_of_the_claim": (
            "ONE registry record is read and rehashed per reference. The "
            "corpora, datasets, checkpoints and directories that record points "
            "at are NOT read or rehashed, every hash of them is emitted "
            "labelled 'recorded, not revalidated', and the digest of the "
            "literal file bytes is emitted labelled 'non-authoritative'. The "
            "mandatory release wording states both halves together."),
        "comparison": {
            "byte_for_byte": ["canonical_json", "execution_dict", "public_view",
                              "advanced_view", "every fingerprint"],
            "structural": ["execution grouping", "direction availability",
                           "publishability reasons",
                           "evidence resolution, field by field"],
            "exceptions": "compared by class NAME, classification and message",
        },
        "vector_count": len(vectors),
        "vectors": sorted(vectors, key=lambda v: v["id"]),
    }


# ---------------------------------------------------------------------------
# verification -- runs against ANY package claiming to implement the contract
# ---------------------------------------------------------------------------

def _compare(failures: list[str], vector_id: str, field: str, expected, actual) -> None:
    if expected != actual:
        failures.append(
            f"{vector_id}: {field} mismatch\n  expected: {expected!r}\n"
            f"  actual:   {actual!r}")


def _verify_codec_accept(api, vector, failures):
    entry = api.decode_entry(vector["input"]["document"])
    expected = vector["expected"]
    _compare(failures, vector["id"], "canonical_json",
             expected["canonical_json"], entry.canonical_json())
    _compare(failures, vector["id"], "audit_fingerprint",
             expected["audit_fingerprint"], entry.audit_fingerprint())
    _compare(failures, vector["id"], "calibrated_directions",
             expected["calibrated_directions"],
             [d.value for d in entry.calibrated_directions])


def _verify_rejection(vector, failures, call):
    try:
        call()
    except Exception as exc:  # the refusal IS the observation
        _compare(failures, vector["id"], "rejection", vector["expected"],
                 _exception_record(exc))
    else:
        failures.append(f"{vector['id']}: expected a refusal, none was raised")


def _verify_resolve(api, vector, failures):
    entry = api.decode_entry(vector["input"]["document"])
    denominators = vector["input"].get("denominators") or []
    state = api.resolve_control(
        entry, direction=vector["input"]["direction"],
        strength=vector["input"]["strength"],
        denominators=_denominator_source(denominators) if denominators else None)
    expected = vector["expected"]
    _compare(failures, vector["id"], "execution_dict", expected["execution_dict"],
             canonical_json(state.execution_dict()))
    _compare(failures, vector["id"], "public_view", expected["public_view"],
             canonical_json(state.public_view()))
    _compare(failures, vector["id"], "advanced_view", expected["advanced_view"],
             canonical_json(state.advanced_view()))
    for field, actual in (("execution_fingerprint", state.execution_fingerprint()),
                          ("state_fingerprint", state.state_fingerprint()),
                          ("entry_audit_fingerprint", state.entry_audit_fingerprint),
                          ("n_targets", state.n_targets)):
        _compare(failures, vector["id"], field, expected[field], actual)


def _verify_runtime_accept(api, vector, failures):
    entry = api.decode_entry(vector["input"]["document"])
    group = api.require_single_execution_group(entry, vector["input"]["direction"])
    expected = vector["expected"]
    _compare(failures, vector["id"], "sae_id", expected["sae_id"], group.sae_id)
    _compare(failures, vector["id"], "layer", expected["layer"], group.layer)
    _compare(failures, vector["id"], "feature_indices",
             expected["feature_indices"], list(group.feature_indices))
    _compare(failures, vector["id"], "executable_directions",
             expected["executable_directions"],
             [d.value for d in api.executable_directions(entry)])


def _verify_evidence(api, vector, failures):
    ref = _evidence_ref(api, vector["input"])
    with materialized_registry(api, vector["input"]) as registry:
        resolution = registry.resolve(ref)
    actual = _resolution_record(resolution)
    for field in sorted(vector["expected"]):
        _compare(failures, vector["id"], field, vector["expected"][field],
                 actual[field])


def _verify_release_wording(api, vector, failures):
    """The sentences are compared verbatim, their adjacency structurally, and the
    checker is exercised in both directions.

    An implementation that shipped the positive sentence and dropped the negative
    one would pass every behavioural vector in this pack. This is the vector that
    catches it.
    """
    expected = vector["expected"]
    actual = {
        "statement": api.RELEASE_EVIDENCE_STATEMENT,
        "verification_sentence": api.EVIDENCE_VERIFICATION_SENTENCE,
        "payload_limit_sentence": api.PAYLOAD_LIMIT_SENTENCE,
        "raw_sha256_label": api.RAW_SHA256_LABEL,
        "payload_hash_label": api.PAYLOAD_HASH_LABEL,
        "prohibited_claims": list(api.PROHIBITED_RELEASE_CLAIMS),
        "record_validity_fields": [f.as_dict() for f in api.PUBLICATION_RECORD_FIELDS],
    }
    for field in sorted(actual):
        _compare(failures, vector["id"], field, expected[field], actual[field])

    statement = api.RELEASE_EVIDENCE_STATEMENT
    head, tail = api.EVIDENCE_VERIFICATION_SENTENCE, api.PAYLOAD_LIMIT_SENTENCE
    _compare(failures, vector["id"], "sentence_adjacency",
             expected["separator_between_sentences"],
             statement[len(head):len(statement) - len(tail)]
             if statement.startswith(head) and statement.endswith(tail)
             else "<the two sentences are not adjacent in the statement>")

    for text in expected["refused_examples"]:
        if not api.prohibited_release_claims(text):
            failures.append(
                f"{vector['id']}: prohibited claim not caught: {text!r}")
    for text in expected["accepted_examples"]:
        found = api.prohibited_release_claims(text)
        if found:
            failures.append(
                f"{vector['id']}: admissible text refused: {text!r} -> {found!r}")


def _verify_publish(api, vector, failures):
    entry = api.decode_entry(vector["input"]["document"])
    with materialized_registry(api, vector["input"]) as registry:
        decision = api.evaluate_publishability(entry, evidence_registry=registry)
    expected = vector["expected"]
    _compare(failures, vector["id"], "publishable", expected["publishable"],
             decision.publishable)
    _compare(failures, vector["id"], "reasons", expected["reasons"],
             list(decision.reasons))
    _compare(failures, vector["id"], "evidence_content_verified",
             expected["evidence_content_verified"],
             decision.evidence_content_verified)
    _compare(failures, vector["id"], "evidence", expected["evidence"],
             [{"artifact_type": r.ref.artifact_type,
               "artifact_hash": r.ref.artifact_hash,
               "status": str(r.status), "resolved": r.resolved,
               "content_verified": r.content_verified,
               "recomputed_digest": r.recomputed_digest,
               "digest_comparison": r.digest_comparison,
               "record_validity_problems": list(r.record_validity_problems)}
              for r in decision.evidence])
    _compare(failures, vector["id"], "verification_record",
             expected["verification_record"],
             canonical_json(decision.content_verification_record()))
    _compare(failures, vector["id"], "release_note", expected["release_note"],
             decision.render_release_evidence_note())


def _verify_availability(api, vector, failures):
    entry = api.decode_entry(vector["input"]["document"])
    with materialized_registry(api, vector["input"]) as registry:
        public = api.select_layout_entries(
            (entry,), exposure=api.Exposure.RELEASE, evidence_registry=registry)
        (layout,) = api.select_layout_entries(
            (entry,), exposure=api.Exposure.DEVELOPMENT_STUBS,
            evidence_registry=registry)
    expected = vector["expected"]
    _compare(failures, vector["id"], "in_public_catalog",
             expected["in_public_catalog"], len(public) == 1)
    _compare(failures, vector["id"], "available_directions",
             expected["available_directions"],
             [d.value for d in layout.available_directions])
    _compare(failures, vector["id"], "unavailable_directions",
             expected["unavailable_directions"],
             [d.value for d in layout.unavailable_directions])
    _compare(failures, vector["id"], "is_development_stub",
             expected["is_development_stub"], layout.is_development_stub)
    _compare(failures, vector["id"], "block_reasons", expected["block_reasons"],
             list(layout.block_reasons))


def _verify_fingerprint_relation(api, vector, failures):
    data = vector["input"]
    denominators = data.get("denominators") or []
    source = _denominator_source(denominators) if denominators else None
    left = api.resolve_control(api.decode_entry(data["left_document"]),
                               direction=data["direction"],
                               strength=data["strength"], denominators=source)
    right = api.resolve_control(api.decode_entry(data["right_document"]),
                                direction=data["direction"],
                                strength=data["strength"], denominators=source)
    expected = vector["expected"]
    actual = {
        "audit_fingerprints_equal":
            left.entry_audit_fingerprint == right.entry_audit_fingerprint,
        "state_fingerprints_equal":
            left.state_fingerprint() == right.state_fingerprint(),
        "execution_fingerprints_equal":
            left.execution_fingerprint() == right.execution_fingerprint(),
        "left": {"audit": left.entry_audit_fingerprint,
                 "state": left.state_fingerprint(),
                 "execution": left.execution_fingerprint()},
        "right": {"audit": right.entry_audit_fingerprint,
                  "state": right.state_fingerprint(),
                  "execution": right.execution_fingerprint()},
    }
    for field in sorted(expected):
        _compare(failures, vector["id"], field, expected[field], actual[field])


def verify_pack(pack: dict[str, Any], package: str = CANONICAL_PACKAGE) -> list[str]:
    """Replays every vector against `package`. Returns the failures, all of them.

    Never short-circuits: an extractor fixing a copy should see every divergence
    in one run, not one per attempt.
    """
    api = load_api(package)
    failures: list[str] = []
    for vector in pack["vectors"]:
        try:
            _verify_one(api, vector, failures, pack["schema_version"])
        except Exception as exc:
            # A package that RAISES where the canonical one returned has diverged,
            # and the divergence belongs in the failure list with every other one.
            # Letting it propagate would abandon the rest of the run and leave an
            # extractor fixing one thing per attempt.
            failures.append(
                f"{vector['id']}: raised {type(exc).__name__}: {exc}, where the "
                f"canonical implementation returned a result")
    return failures


def _verify_one(api, vector: dict[str, Any], failures: list[str],
                pack_schema_version: str = SCHEMA_VERSION) -> None:
    kind = vector["kind"]
    if vector.get("schema_version") not in (None, pack_schema_version):
        failures.append(f"{vector['id']}: schema_version disagrees with the pack")
    if kind == "codec_accept":
        _verify_codec_accept(api, vector, failures)
    elif kind == "codec_reject":
        _verify_rejection(vector, failures,
                          lambda v=vector: api.decode_entry(v["input"]["document"]))
    elif kind == "schema_reject":
        _verify_rejection(
            vector, failures,
            lambda v=vector: api.EvidenceRef(**v["input"]["evidence_ref"]))
    elif kind == "release_wording":
        _verify_release_wording(api, vector, failures)
    elif kind == "resolve":
        _verify_resolve(api, vector, failures)
    elif kind == "resolve_reject":
        _verify_rejection(
            vector, failures,
            lambda v=vector: api.resolve_control(
                api.decode_entry(v["input"]["document"]),
                direction=v["input"]["direction"],
                strength=v["input"]["strength"]))
    elif kind == "runtime_reject":
        _verify_rejection(
            vector, failures,
            lambda v=vector: api.require_single_execution_group(
                api.decode_entry(v["input"]["document"]),
                v["input"]["direction"]))
    elif kind == "runtime_accept":
        _verify_runtime_accept(api, vector, failures)
    elif kind == "evidence":
        _verify_evidence(api, vector, failures)
    elif kind == "publish":
        _verify_publish(api, vector, failures)
    elif kind == "availability":
        _verify_availability(api, vector, failures)
    elif kind == "fingerprint_relation":
        _verify_fingerprint_relation(api, vector, failures)
    else:
        failures.append(f"{vector['id']}: unknown vector kind {kind!r}")


# ---------------------------------------------------------------------------
# export inventory
# ---------------------------------------------------------------------------

def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # relative import inside the package
                continue
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def runtime_dependencies() -> list[dict[str, Any]]:
    """Every non-relative import the minimum surface makes, flagged as
    standard-library or not.

    Computed by walking the modules rather than transcribed, so the answer
    cannot drift from the code it describes.
    """
    found: set[str] = set()
    for relative, _ in MINIMUM_EXPORT_MODULES:
        found |= _imports_of(REPO_ROOT / relative)
    found.discard("__future__")
    return [{"module": name,
             "standard_library": name in sys.stdlib_module_names,
             "purpose": _DEPENDENCY_PURPOSE.get(name, "")}
            for name in sorted(found)]


_DEPENDENCY_PURPOSE = {
    "collections": "Mapping protocol for the frozen mapping fields",
    "dataclasses": "frozen slotted value objects",
    "datetime": "ISO-8601 validation of calibrated_at",
    "enum": "the closed vocabularies (StrEnum, 3.11+)",
    "hashlib": "SHA-256 fingerprints",
    "json": "canonical serialization",
    "logging": "one line per resolved absolute clamp value",
    "math": "finiteness checks",
    "pathlib": "explicitly named bundle files and the registry tree",
    "re": "id and artifact-hash patterns",
    "types": "MappingProxyType for immutability after validation",
    "typing": "Protocol and annotations",
}


def build_inventory(*, vectors_sha256: str, vectors_bytes: int,
                    frozen_at_commit: str | None = None) -> dict[str, Any]:
    modules = []
    for relative, role in MINIMUM_EXPORT_MODULES:
        data = file_bytes(REPO_ROOT / relative)
        modules.append({"path": relative, "sha256": sha256_bytes(data),
                        "bytes": len(data), "role": role})
    runner = REPO_ROOT / "scripts" / "concept_bundle_conformance.py"
    runner_bytes = file_bytes(runner)
    dependencies = runtime_dependencies()
    return {
        "inventory_version": "1.0",
        "canonical_repository": {
            "name": CANONICAL_REPOSITORY,
            "role": "canonical scientific source of the concept-bundle contract",
            "package": CANONICAL_PACKAGE,
            "branch": CANONICAL_BRANCH,
            "contract_base_commit": CONTRACT_BASE_COMMIT,
            "frozen_at_commit": frozen_at_commit,
            "frozen_at_commit_note": (
                "The module hashes below are of the tree of frozen_at_commit. "
                "The commit that regenerates this pack cannot name itself -- a "
                "file cannot contain the hash of the commit containing it -- so "
                "that field is stamped by an immediately following commit whose "
                "only change is this line. contract_base_commit is the "
                "predecessor this pack succeeds, not where the hashes come "
                "from."),
        },
        "product_repository": {
            "note": (
                "A separate product repository will carry an extracted copy. "
                "This inventory is written FOR that extraction and does not "
                "name, import, write to, or depend on it. No dependency exists "
                "in either direction."),
        },
        "python": {
            "requires": PYTHON_REQUIRES,
            "features_requiring_3_11": [
                "enum.StrEnum",
                "datetime.fromisoformat accepting a trailing Z",
            ],
        },
        "runtime_dependencies": dependencies,
        "third_party_runtime_dependencies": [
            d for d in dependencies if not d["standard_library"]],
        "standard_library_only": all(d["standard_library"] for d in dependencies),
        "minimum_export_surface": modules,
        "excluded_from_minimum_surface": [
            {"path": path, "reason": reason} for path, reason in EXCLUDED_FROM_EXPORT],
        "conformance": {
            "vectors": {
                "path": VECTORS_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": vectors_sha256, "bytes": vectors_bytes},
            "runner": {
                "path": runner.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_bytes(runner_bytes), "bytes": len(runner_bytes)},
            "how_to_check_an_extracted_copy": (
                "python scripts/concept_bundle_conformance.py --check "
                "--package <extracted.package.name>"),
            "properties": ["cpu_only", "deterministic", "offline",
                           "no_model_weights"],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="regenerate the pack and the inventory from the "
                             "canonical implementation")
    parser.add_argument("--check", action="store_true",
                        help="verify a package against the frozen pack")
    parser.add_argument("--package", default=CANONICAL_PACKAGE,
                        help="package to check (default: the canonical one)")
    parser.add_argument("--frozen-at-commit", default=None,
                        help="stamp the inventory with the commit that froze it")
    args = parser.parse_args(argv)

    if args.write:
        pack_bytes = write_text_lf(VECTORS_PATH, _dump(build_pack()))
        inventory = build_inventory(vectors_sha256=sha256_bytes(pack_bytes),
                                    vectors_bytes=len(pack_bytes),
                                    frozen_at_commit=args.frozen_at_commit)
        write_text_lf(INVENTORY_PATH, _dump(inventory))
        print(f"wrote {VECTORS_PATH.relative_to(REPO_ROOT).as_posix()} "
              f"({len(pack_bytes)} bytes, sha256 {sha256_bytes(pack_bytes)})")
        print(f"wrote {INVENTORY_PATH.relative_to(REPO_ROOT).as_posix()}")
        return 0

    if args.check:
        pack = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
        failures = verify_pack(pack, args.package)
        report = {"package": args.package, "vectors": pack["vector_count"],
                  "failures": failures, "conformant": not failures}
        print(_dump(report), end="")
        return 0 if not failures else 1

    parser.error("choose --write or --check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
