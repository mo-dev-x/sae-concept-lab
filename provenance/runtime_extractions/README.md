# Reserved: future runtime extractions

This directory is a reserved location for future, explicitly-extracted
runtime code from `qwen-sae-interp` (e.g. a real backend implementing
`sae_concept_lab.core.protocol.ConceptLabBackend`, once qwen-sae-interp's
bundle/resolution contract exists).

It is deliberately empty at this repository's initial commit. No runtime
code has been extracted yet -- this initial import is UI/product files
only (see `../source_import.json` and `../../BOUNDARY.md`).

When something is extracted here in the future, it must follow the same
discipline `source_import.json` already establishes: an explicit source
repository identity, an explicit source commit, an explicit
source-path -> destination-path mapping, a SHA-256 per imported file, an
import timestamp, and an explicit statement of what was and was not
imported. Do not add files here informally.
