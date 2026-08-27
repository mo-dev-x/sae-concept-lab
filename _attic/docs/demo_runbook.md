# PI demo runbook (2026-08-13)

Read [`docs/pi_demo_scientific_status.md`](pi_demo_scientific_status.md)
first -- it is the authority on exact permitted/prohibited claims. This
document is the operational script: environment, launch command, the
five-minute walkthrough, the Mode A / Mode B branch, and recovery.

Everything below runs on **D: only**. Do not install, cache, or write any
temporary output to C: at any point -- `sae_concept_lab.smoke.
pi_demo_preflight` checks this mechanically (see step 2).

## 0. The night before / first thing in the morning

```bash
cd /d/sae-concept-lab
git status                       # must be clean, on the successor commit
python -m pytest -q              # full suite must be green (see final report)
```

## 1. Decide the mode

```bash
python - <<'PY'
from sae_concept_lab.smoke.pi_demo_preflight import check_release_gate_status
result = check_release_gate_status(None)  # pass a real registry root here if one was staged
print(result.summary)
print(result.detail)
PY
```

- `mode_implied: "B"` (publishable_counts all zero) -> **Mode B**, section
  3 below. This is the guaranteed path; prepare it regardless of whether
  you expect Mode A to land.
- `mode_implied: "A"` (publishable_counts nonzero) -> a bundle is staged
  in `sae_concept_lab/fixtures/attested/{gemma,qwen}/` and passes
  publishability against the registry root you passed. Go to **Mode A**,
  section 2, but keep Mode B ready as the fallback -- a real backend
  requiring GPU weights can still fail to construct even when the bundle
  itself is eligible.

## 2. Mode A -- REAL SCIENCE (only if an ATTESTED bundle arrived before cutoff)

Procedure for whoever stages the arriving bundle (before it is dropped
into the slot):

1. **Import through the canonical codec**: `decode_entry(document, where=...)`
   or `load_entry_file(path)` -- must not raise.
2. **Verify extraction/provenance**: confirm `entry.provenance is
   Provenance.ATTESTED` and `entry.calibration_provenance` names the
   exact evidence artifact(s), by content hash, that were actually
   produced.
3. **Run release eligibility**:
   ```python
   from sae_concept_lab.canonical.concept_bundle import RepositoryEvidenceRegistry, evaluate_publishability
   decision = evaluate_publishability(entry, evidence_registry=RepositoryEvidenceRegistry(root="<real registry root>"))
   assert decision.publishable, decision.reasons
   ```
4. Only once (3) passes: copy the file into
   `sae_concept_lab/fixtures/attested/<gemma|qwen>/<name>.json`. **No
   `.py` file is edited for this step** -- `fixtures.loader.load_entries`
   picks it up immediately; see `tests/test_pi_demo_mode_a.py::
   test_a_valid_attested_bundle_dropped_into_the_slot_is_picked_up_with_no_code_edit`
   for the mechanical proof.
5. **Run focused/full tests**: `python -m pytest -q` (full suite) plus
   whatever focused test covers the specific pairing.
6. **Run dev and release smoke**:
   ```bash
   python -m sae_concept_lab.smoke.pi_demo_preflight \
     --evidence-registry-root <real registry root> \
     --output pi_demo_preflight_mode_a.json
   ```
   Confirm `release_gate_status.detail.mode_implied == "A"` in the
   written JSON, and that `release_refuses_locally` still passes (it
   always will -- see that check's own docstring: a real backend is a
   separate, GPU-side precondition this local preflight cannot satisfy).
   The actual real-backend release launch is rehearsed on Tamia, the same
   way job 407815 already did for the mechanical-acceptance evidence.
7. **Launch instructions** (real backend, real registry, on Tamia or any
   machine with the real weights and `HF_HUB_OFFLINE=1` available):
   ```bash
   HF_HUB_OFFLINE=1 python -m sae_concept_lab.app --mode release \
     --evidence-registry-root <real registry root> \
     --qwen-backend runtime --qwen-model-path <path> --qwen-sae-path <path> --qwen-layer <N> \
     --gemma-backend runtime --gemma-model-path <path> --gemma-sae-path <path> \
     --server-name 127.0.0.1 --server-port 7860
   # 127.0.0.1, NEVER 0.0.0.0: a Tamia compute node is shared, and 0.0.0.0
   # publishes this UI to every other user on it. Forward the port instead --
   # see docs/tamia_launch.md, "Reaching the UI".
   ```
   This renders **only** the publishable entries (canonical's
   `select_layout_entries(..., exposure=Exposure.RELEASE)`, wired in
   `sae_concept_lab/app.py`) -- the shipped FAKE fixtures are excluded
   from the rendered UI even though they are still present in
   `load_entries()`'s return value, and the permanent "PLACEHOLDER / NOT
   SCIENTIFIC EVIDENCE" banner is omitted (`ui/app_ui.build_demo(...,
   mode="release")`) because it would otherwise misdescribe the real
   content now on screen.

If any of steps 1-6 fails, or the real backend cannot construct (missing
weights, identity mismatch, GPU unavailable), **stop and fall back to Mode
B** -- do not attempt to demo a half-verified Mode A build.

## 3. Mode B -- GUARANTEED ENGINEERING PREVIEW

```bash
cd /d/sae-concept-lab
$env:HF_HUB_OFFLINE = "1"
python -m sae_concept_lab.app --server-name 127.0.0.1 --server-port 7860
```

(`--mode` defaults to `dev`; both backends default to `stub`; no GPU, no
model weights, no network access needed.) Open `http://127.0.0.1:7860/`.

### Five-minute click-by-click script

1. **(0:00-0:30) The banner.** Point at the "⚠️ PLACEHOLDER / NOT
   SCIENTIFIC EVIDENCE" banner at the top. Say: "Every reply and
   technical value below this line is synthetic stub data for interface
   testing -- this build ships zero ATTESTED concepts" (see
   `pi_demo_scientific_status.md` section 3 for the live count).
2. **(0:30-1:15) Concept selection.** Click a concept card (e.g.
   "Warmth" on the Gemma tab). Point at the concept detail panel updating
   and the Direction radio narrowing to only calibrated directions for
   concepts that have one-sided calibration (click "Caution" to show the
   unavailable-direction notice rendering the exact canonical refusal
   text).
3. **(1:15-2:15) Chat + Public output.** Pick Amplify / Medium, type a
   short message, click Send. Point at "What produced this reply" (Public
   view: model, concept, direction, strength only -- no feature index, no
   SAE id, no raw value).
4. **(2:15-3:15) Advanced.** Open the Advanced accordion. Point at the
   resolved-state JSON: full per-target detail, `provenance: "fake"`
   printed explicitly on this entry, evidence identity, fingerprints, and
   the positions read-only display. Change strength to High and show the
   conversation reset notice (any concept/direction/strength change
   unconditionally clears history -- no override anywhere).
5. **(3:15-4:15) Compare.** Type a message, click "Compare Original vs
   Modified." Point at Original/Modified side by side and mention the
   Compare invariant (both arms share history/prompt/model/decoding/seed
   exactly; only intervention differs) -- tested directly in
   `tests/test_sae_concept_lab_compare.py`.
6. **(4:15-5:00) Capability refusal.** Switch language (FR/EN) to show
   the whole page retranslate instantly, then close by restating section
   4 of `pi_demo_scientific_status.md`: "this is a working instrument,
   not validated scientific content -- yet."

## 4. Recovery

- **Server won't start / port already in use**: pick a different
  `--server-port` (7860 is only a default). Check `git status` is clean
  and you are on the intended commit; re-run
  `python -m sae_concept_lab.smoke.pi_demo_preflight` (step 5 below) to
  get a specific failure reason before retrying.
- **Browser can't reach it**: confirm you bound `--server-name 127.0.0.1`
  (loopback) and are browsing `http://127.0.0.1:<port>/` on the SAME
  machine -- this build never binds `0.0.0.0` in the demo command above,
  by design, so it is not reachable from another machine or over Wi-Fi.
- **A click produces an unexpected error / capability refusal you did not
  plan to show**: that is expected for some concept/direction
  combinations (e.g. "Caution" on Gemma calibrates only one direction) --
  narrate it as the capability-refusal mechanism working correctly
  (section 6 of the script above), not a bug. If it is genuinely
  unexpected, close the browser tab, `Ctrl+C` the server, and restart from
  step 3's command -- state is entirely in-memory (`gr.State`); a restart
  always returns to the exact same defaults.
- **Mode A was supposed to be live and isn't**: re-run
  `check_release_gate_status` (step 1). If it now reports Mode B, the
  bundle was withdrawn or a dependency (registry root, evidence artifact)
  is no longer reachable -- fall back to Mode B (section 3) without
  announcing why unless asked.

## 5. Preflight -- run this immediately before the PI arrives

```bash
cd /d/sae-concept-lab
python -m sae_concept_lab.smoke.pi_demo_preflight --output pi_demo_preflight_result.json
echo $?   # 0 == every check passed
cat pi_demo_preflight_result.json
```

Checks, in order (all in `sae_concept_lab/smoke/pi_demo_preflight.py`):
required files present, no C:-based path anywhere (repo root, cwd,
`--output`, and every cache/temp environment variable actually set),
current release-eligibility status (Mode A vs Mode B, informational),
release still refuses locally with stub backends (hard assertion), boots
the real dev-mode app on loopback, HTTP 200, the correct banner is
visible for the mode that was booted, and a clean shutdown (the loopback
port is confirmed released, not just that `.close()` did not raise).
Returns nonzero if ANY check fails; the full aggregate result -- including
every check that ran AFTER a failure -- is always in the written JSON, so
a later pass can never hide an earlier failure.

Pass `--evidence-registry-root <path>` to also get an accurate Mode-A
eligibility read in the same run (see section 2, step 6).
