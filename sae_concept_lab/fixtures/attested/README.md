# The bounded Mode-A import slot

This directory is the ONE location a genuinely ATTESTED concept-bundle
document can be dropped into so that `sae_concept_lab.fixtures.loader.
load_entries(model_key)` picks it up -- with **no edit to any `.py` file
in this repository**. Drop a schema-valid document into:

- `gemma/<anything>.json` for a `gemma` pairing entry
- `qwen/<anything>.json` for a `qwen` pairing entry

and the next `load_entries("gemma")` / `load_entries("qwen")` call
includes it, in sorted filename order, appended after the shipped FAKE
fixtures.

## What landing here does NOT do

Dropping a file here does not make it publish. `loader.load_attested_entries`
decodes it through the exact same canonical codec every shipped FAKE
fixture goes through (`load_entry_files`) -- nothing more. Whether the
decoded entry is actually publishable (provenance exactly `ATTESTED`,
every evidence reference resolved AND content-verified against a real
`--evidence-registry-root`, written in the full `sha256:<64 hex>` form,
no placeholder markers) is entirely
`sae_concept_lab.canonical.concept_bundle.release.evaluate_publishability`'s
decision, evaluated exactly as it is for every other entry -- this slot
adds no shortcut and no override anywhere.

## What a bad drop-in does NOT do

A file that fails to decode (malformed JSON, a schema violation, an
unsafe `artifact_type`) is excluded and reported -- see
`load_attested_entries(model_key).rejected` -- rather than raised. The
shipped FAKE fixtures (Mode B, the guaranteed engineering preview) always
load regardless of what this directory currently contains or how broken
it is. See the internal `demo_runbook.md` (archived, not shipped publicly) and
the internal `pi_demo_scientific_status.md` (archived, not shipped publicly) (both archived internal
PI-demo material) for the full Mode A / Mode B procedure this slot exists
to support, and
`BOUNDARY.md`'s "Bounded Mode-A import slot" section for the design
rationale.

## Before staging a file here

Verify it first -- this slot does not do that for you:

1. Decode it through the canonical codec (`decode_entry`/`load_entry_file`)
   and confirm no exception is raised.
2. Confirm `provenance` is exactly `"attested"` and `calibration_provenance`
   is present with the evidence references you expect.
3. Run `evaluate_publishability(entry, evidence_registry=RepositoryEvidenceRegistry(root=<your real registry root>))`
   and confirm `.publishable is True` before relying on it for a release
   build.

This directory currently ships empty (no tracked `*.json` files) in every
committed state of this repository.
