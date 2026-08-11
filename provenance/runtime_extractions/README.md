# Runtime extractions

This directory holds copies of conformance/verification artifacts for
runtime that has been mechanically extracted into this repository, kept
alongside (not instead of) the manifest at `../source_import.json`.

## concept_bundle/

The frozen conformance pack for the concept-bundle contract, copied
byte-for-byte from qwen-sae-interp at commit `cdae9c7`:

- `vectors.json` -- the 50 frozen vectors (inputs plus exact expected
  outputs/refusals) Engineer 3's pack froze against the canonical
  implementation.
- `export_inventory.json` -- the canonical accounting of the eight-module
  minimum export surface, its hashes, its exclusions, and its
  standard-library-only runtime dependency list.
- `concept_bundle_conformance.py` -- the check-mode runner. Copied here
  for the record and so `verify_pack()` can be called directly, in
  process, against this repository's own extracted package
  (`sae_concept_lab.canonical.concept_bundle`) with no qwen-sae-interp
  checkout required -- see `tests/test_concept_bundle_conformance.py`.
  This is the verification harness, not runtime surface: nothing under
  `sae_concept_lab/` imports it.

The extracted runtime itself lives at
`sae_concept_lab/canonical/concept_bundle/`, not here -- this directory
is the provenance/conformance record for that extraction, not the
extraction's own code.

## Adding another extraction here

Any future addition must follow the same discipline already established:
an explicit source repository identity, an explicit source commit (or
commits, if a contract base and a freeze commit differ, as
`concept_bundle_contract` records), an explicit
source-path -> destination-path mapping, a SHA-256 per file, an import
timestamp, and an explicit statement of what was and was not imported --
recorded as a new entry in `../source_import.json`'s `extractions` list,
never informally.
