# Real, measured concept bundles (not shipped active)

These are **real** concept-bundle documents built from measured features, not
fixtures. They live here rather than in `sae_concept_lab/fixtures/attested/`
because that slot is auto-loaded, and the repository deliberately ships it
empty: `load_entries()` returning exactly the four FAKE fixtures per pairing is
a contract the UI smoke tests pin. Dropping these in changes the shipped state
and fails 25 tests that are asserting that contract correctly.

## What they are

`candidate/{gemma,qwen}/pro_american_exceptionalism.json` — the features that
survived the full gate conjunction in **all six cells** of a full-space scan, on
real weights, from `qwen-sae-interp` job **418185** (`la-b-afc-grid`, exit `0:0`):

| pairing | layer | SAE | features |
|---|---|---|---|
| `gemma-3-12b-it` + `gemma-scope-2-12b-it` | 29 | `resid_post_all/layer_29_width_16k_l0_big` | 3048, 15405 |
| `Qwen3.5-27B` + `SAE-Res-Qwen3.5-27B-W80K-L0_100` | 38 | `layer38.sae.pt` | 26943, 41745 |

Both decode cleanly through the canonical codec (`decode_entry`).

## Why `provenance: "candidate"` and not `"attested"`

`ATTESTED` requires `calibration_provenance` with content-verified evidence
references. **No calibration boundary has ever been pinned and no causal test
has ever been run**, so attesting these would be a false claim that the release
gate exists to catch. `candidate` is the honest value: the features are real and
measured, the causal claim is absent.

## Why `amplify` is null

`ABLATE` carries no dose by contract, so the suppress direction invents nothing.
`CLAMP` requires a `value`, that value would be a steering dose, and no
calibrated dose exists. Authoring one would be inventing the number the whole
calibration protocol exists to produce. `amplify` is therefore `null` until a
control-calibrated dose exists.

## Activating them for a demo

Copy into the bounded Mode-A import slot — no `.py` edit required:

```bash
mkdir -p sae_concept_lab/fixtures/attested/gemma sae_concept_lab/fixtures/attested/qwen
cp bundles/candidate/gemma/*.json sae_concept_lab/fixtures/attested/gemma/
cp bundles/candidate/qwen/*.json  sae_concept_lab/fixtures/attested/qwen/
```

`load_entries("gemma")` then returns 5 entries, the fifth being the real one.
The release gate still refuses it — `--mode release` remains fail-closed,
because `candidate` is not `attested`. That refusal is correct and must not be
worked around.

Remove the copies before running the test suite.
