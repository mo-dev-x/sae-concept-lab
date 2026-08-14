# PI demo: exact scientific status (2026-08-13)

This document is the single source of truth for what this build may, and
may not, be described as during the PI demo. It is deliberately narrow:
every claim below is traceable to a specific commit, evidence artifact, or
test, and nothing here restates or widens any of them. See
[`BOUNDARY.md`](../BOUNDARY.md) for the full account and
[`docs/pi_demo_runbook.md`](pi_demo_runbook.md) for how to present this
live.

## 1. Mechanically accepted runtimes: Qwen and Gemma

`sae_concept_lab.core.runtime_acceptance` currently records BOTH pairings
as mechanically accepted, imported from qwen-sae-interp evidence commit
`b6d598b` after independent re-verification (`git show` + recomputed
SHA-256 of every named artifact, plus running qwen-sae-interp's own
29-test evidence record suite against that checkout):

| Pairing | Scope of the accepted claim | Evidence source |
|---|---|---|
| Qwen3.5-27B + Qwen-Scope layer 0 | ALL and GENERATED_ONLY scenarios only (job 406092's Gemma scenarios in the same job failed and are explicitly excluded) | `results/final_pairing/job_406092/` |
| gemma-3-12b-it + gemma-scope-2-12b-it | ALL and GENERATED_ONLY, both scenarios | `results/final_pairing/job_407008/` |

**What "mechanically accepted" means, exactly**: an intervention reaching
the real hook and measurably moving the real residual stream has been
proven, for the specific engineering-only layer/feature/positions
combination each job exercised. It is **mechanical evidence only** --
never a scientific concept claim, never a calibration, never a
behavioral-quality claim.

## 2. Product smoke job 407815

The Tamia product-integration smoke packet (`sae_concept_lab/smoke/
tamia_smoke.py`, `docs/tamia_smoke.md`) ran as Tamia job **407815** and
passed all 13 scenarios: hidden smoke-entry isolation, identity-swap
refusal, multi-SAE-at-one-layer PROHIBITED, cross-layer CAPABILITY_LIMIT,
backend/layer-mismatch refusal, Qwen ALL/GENERATED_ONLY, Gemma
ALL/GENERATED_ONLY, and the four application-smoke checks (hidden entries
unreachable via the UI, HTTP 200, one bounded request through the real
application adapter, release mode still refuses). This proves the
**product's own adapters** (`QwenRuntimeBackend`, `GemmaRuntimeBackend`,
`core.logic.send_message`, `ui.app_ui.build_demo`) correctly carry the
mechanical acceptance above through to a real generation on real Tamia
GPU hardware -- it is not a second, independent scientific finding, and it
does not widen what section 1 already claims.

## 3. Current count of ATTESTED concepts: zero (as of this commit)

Every concept-bundle entry this repository ships is `provenance: "fake"`.
`sae_concept_lab.fixtures.loader`'s bounded Mode-A import slot
(`fixtures/attested/{gemma,qwen}/`) is empty in this commit. **Zero**
concepts anywhere in this build are eligible to publish
(`evaluate_publishability`), and `--mode release` refuses to launch for
both pairings, unconditionally, regardless of which backend is selected --
run `sae_concept_lab.smoke.pi_demo_preflight` (`docs/pi_demo_runbook.md`)
for a live, timestamped confirmation of this exact count immediately
before the demo. If a bundle arrives and is staged before the cutoff,
update this line to state the new count and cite the specific evidence
commit(s) it imports from -- never leave a stale zero next to a live
Mode-A launch.

## 4. A working instrument vs. validated scientific content

These are two separate claims, and the demo must keep them separate at
every step:

- **A working instrument** (proven): the concept-bundle contract, the
  publication gate, the extracted intervention mechanism, and the product
  adapters that wire them together all function correctly against real
  weights, on this exact build, as of section 1 and section 2 above. This
  is an engineering claim about *machinery*.
- **Validated scientific content** (not yet proven, for anything this
  build ships): that any specific feature index represents a
  meaningful, well-calibrated, or discovered "concept" in the sense a
  reader of the demo would assume from the word. Section 3's zero count
  is the honest current answer to "how much of that exists in this
  build."

Every FAKE-provenance fixture demonstrates the INSTRUMENT (concept
selection, direction/strength resolution, Compare, capability refusals,
positions semantics) using placeholder labels and templated text -- never
a real generation, never a real measurement, and the permanent
"PLACEHOLDER / NOT SCIENTIFIC EVIDENCE" banner is present specifically so
no viewer mistakes one for the other. If Mode A is live, its rendered
entries are real generations through mechanically-accepted, evidence-
verified backends -- but "evidence-verified" here means exactly the two
sentences `sae_concept_lab.canonical.concept_bundle.release.
RELEASE_EVIDENCE_STATEMENT` states, no more: the evidence *registry
record* was read and its digest recomputed; the corpora, datasets and
checkpoints it POINTS AT were not independently re-verified by this
build.

## 5. Qwen layer 0 / feature 4096 and Gemma feature 250 are NOT personas

These three numbers (`sae_concept_lab.smoke.entries`'s
`QWEN_SMOKE_FEATURE_IDX = 4096`, `GEMMA_SMOKE_FEATURE_IDX = 250`, and
Qwen's engineering-only layer `0`) are **hidden, ENGINEERING-ONLY smoke
identities**. They exist only to prove the intervention mechanism moves a
real residual stream at a real, addressable hook -- exactly as documented
in `docs/tamia_smoke.md` and the sealed job-406092/407008 READMEs. They
are:

- never added to `fixtures/{gemma,qwen}/*.json` or `fixtures.loader.
  _ENTRY_FILENAMES`,
- never reachable through the Gradio UI (`sae_concept_lab.smoke.entries.
  ALL_SMOKE_CONCEPT_IDS`, mechanically proven unreachable by
  `tests/test_tamia_smoke.py`),
- and, most importantly for this document: **not a claim that clamping
  this feature produces "curiosity," "warmth," or any other named trait**.
  No calibration, no human evaluation, and no discovery protocol has ever
  been run against either index. If asked live whether feature 4096 "is"
  a concept, the correct and complete answer is: "No -- it is the
  engineering index the mechanical-acceptance smoke test used to prove
  the hook moves the residual stream. Nothing about what it represents,
  if anything, has been evaluated."

## 6. The loaded SAE is not the certified primary, and the tool now says so

`sae_concept_lab/extracted_runtime/targets.py` pins Gemma to release
`gemma-scope-2-12b-it-res`, sae_id
`resid_post/layer_31_width_16k_l0_medium`, layer 31, and Qwen to
`Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`. Neither is the certified primary
configuration. Per the qwen-sae-interp certified candidate
`8ed2809`'s own gating identity protocols
(`protocols/final_pairing/v1/scientific_config_identity.json` and
`qwen_config_identity.json`), the certified **PRIMARY** configurations are:

| Arm | release | scientific_sae_id | layer |
|---|---|---|---|
| Gemma | `gemma-scope-2-12b-it-res-all` | `resid_post_all/layer_29_width_16k_l0_big` | 29 |
| Qwen | (raw-pt: repository is the namespace) `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100` | `layer38.sae.pt` | 38 |

The Qwen `L0_50` pin is the **ratified BACKUP**, which is the dangerous
case rather than the harmless one: it is present, ratified and internally
coherent, so a reader who checks it stops there.

Which SAE is primary is a frozen scientific ruling and is **not** changed
by this. What changed is the CLAIM:

- `sae_concept_lab/core/scientific_identity.py` compares the identity the
  loader actually returned against the certified primary on
  `sae_repository`, `release`, `scientific_sae_id` and `layer`. Nothing
  else may be presented as science-attributed.
- A resolved control state naming a layer other than the layer actually
  loaded now raises `LoadedLayerIdentityMismatch` and produces no output
  at all. Previously the unpacked layer was used for nothing but a
  diagnostics field, so a feature index could be clamped inside a
  different layer's dictionary while the diagnostics reported the
  requested layer.
- Diagnostics report `identity.loaded` (read from the loader's own
  provenance) beside `identity.requested` (read from the bundle). Every
  requested identity field is spelled `requested_*`; there is no longer a
  bare `sae_id`/`layer` key that could be read as either.
- Engineering demonstrations on any pin remain fully permitted. They
  carry `claim_scope: ENGINEERING_DEMONSTRATION_ONLY` and their generated
  text is prefixed with `ENGINEERING_DEMONSTRATION_TAG`.

Consequently every demonstration this build currently produces on either
pairing is an **engineering demonstration**, mechanically accepted and
not science-attributed. Section 1's mechanical acceptance is unchanged
and unaffected: it was always a claim about the mechanism, never about
the identity.

## Exact permitted and prohibited claims (quick reference)

**Permitted:**
- "Both the Qwen3.5-27B and gemma-3-12b-it intervention mechanisms are
  mechanically accepted against real weights, evidence commit `b6d598b`."
- "The full product stack -- UI, resolution, execution guard, backend
  translation -- was proven end to end on real Tamia GPU hardware, job
  407815."
- "This build currently ships zero ATTESTED, publicly releasable
  concepts." (or the updated count + citation, if Mode A is live)
- "Everything below this banner is synthetic placeholder data for
  interface testing." (Mode B, verbatim, unchanged)
- (Mode A only) "This entry's evidence registry record was read and its
  digest independently recomputed and matched." (verbatim
  `RELEASE_EVIDENCE_STATEMENT` wording only -- never a paraphrase)

**Prohibited:**
- Any claim that a FAKE-provenance fixture's label ("curiosity",
  "warmth", ...) corresponds to anything measured, discovered, or
  calibrated.
- Any claim that feature 4096 (Qwen) or feature 250 (Gemma) represents a
  concept, persona, or trait.
- "Evidence verified" / "artifacts verified" / "verified against the
  corpus" / "verified against the dataset" / "fully verified" -- these
  exact phrasings are mechanically refused by `assert_release_text_clean`
  for a reason; do not say them live either.
- Any claim that mechanical acceptance (section 1) or the product smoke
  (section 2) is itself a scientific finding about a concept.
- Any claim attributing a demonstration on this build to the certified
  primary SAE. Per section 6, the pinned identity is not it, on all three
  scientific identity fields at once, and the tool refuses the
  attribution mechanically.
