<p align="center">
  <img src="assets/logo-wide.svg" alt="sae-concept-lab" width="360">
</p>

<p align="center">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-1e7b4f.svg">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-1e7b4f.svg">
</p>

An interactive tool for steering sparse-autoencoder (SAE) features in language models: pick a concept, turn it up or down while you chat, and watch the reply change.

## What this is, and is not

This is a **Gradio UI and a release gate**, not a scientific instrument. It exists to make one question answerable by clicking, rather than by reading code: *if I amplify or suppress this specific SAE feature, what does the model say differently?*

What it does:
- Runs two model tabs, `gemma-3-12b-it` + `gemma-scope-2-12b-it` and `Qwen3.5-27B` + `Qwen-Scope`, each wired to a real intervention hook that clamps (amplify) or zeroes (suppress) one SAE feature at a fixed layer.
- Ships with a deterministic, GPU-free stub backend by default, so the interface is fully explorable on a laptop with no model weights anywhere.
- Ships one real, **measured** concept per pairing (`pro-american-exceptionalism`): a feature that survived a full-space discovery scan on real weights. Its `provenance` is recorded as `candidate`, not `attested` — see Limitations.
- Renders every prompt through the model's own chat template, and refuses to generate rather than fall back to a hand-written prompt wrapper if a template is unavailable.
- Refuses to launch in `--mode release` on this build, on purpose: the fail-closed release gate only publishes concepts with fully verified evidence, and none ship yet.

What it does **not** do:
- It does not discover, calibrate, or validate concepts. That happens in a separate scientific repository; this tool only renders and executes what that repository hands it. See [`BOUNDARY.md`](BOUNDARY.md) for the exact line between the two.
- It does not claim a calibrated steering dose. The shipped amplify strengths are engineering defaults derived from an activation measurement, not an experimentally validated dose-response curve.
- It does not do anything with ablation strength: a feature is zeroed or it is not, so Low/Medium/High are identical under Suppress and differ only under Amplify.
- It has no auth, no persistence, and no multi-user support. State lives in memory for one session.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/mo-dev-x/sae-concept-lab.git
cd sae-concept-lab
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

Verify the install:

```bash
pytest -q
```

This runs entirely on CPU with no model weights. On a clean install it passes in full — 360 tests, 2 skipped (the 2 skips require real GPU-staged weights on a cluster and are gated behind environment variables that are unset by default).

## Quickstart

```bash
python -m sae_concept_lab.app --server-name 127.0.0.1 --server-port 7860
```

Open `http://127.0.0.1:7860` in a browser on the same machine. Both tabs are backed by the stub model, so replies are synthetic and clearly tagged `[FAKE STUB -- UI TEST ONLY]` — this is the guaranteed, always-working path, meant for exploring the interface rather than the science.

To drive a real model instead, pass real, locally staged weights:

```bash
HF_HUB_OFFLINE=1 python -m sae_concept_lab.app \
  --gemma-backend runtime --gemma-model-path /path/to/gemma-3-12b-it --gemma-sae-path /path/to/gemma-scope-2-12b-it \
  --qwen-backend runtime  --qwen-model-path /path/to/Qwen3.5-27B     --qwen-sae-path /path/to/qwen-scope/layer38.sae.pt --qwen-layer 38 \
  --server-name 127.0.0.1 --server-port 7860
```

Each `--<model>-backend runtime` flag is independent; passing only one leaves the other tab on the stub. This needs a GPU, the four model/SAE snapshots downloaded ahead of time (compute nodes are typically offline), and `torch` installed alongside the extras above. Full step-by-step instructions, including reaching a headless GPU node over SSH port-forwarding and every `--help` flag, are in [`docs/RUNNING.md`](docs/RUNNING.md).

## The interactive UI

Each model tab is one shared component tree:

- **Concept cards** — click a concept to select it. One is shipped per model today.
- **Direction** — Amplify (clamp the feature up) or Suppress (ablate it to zero). Unavailable directions are hidden with the exact reason (`PROHIBITED` / `CAPABILITY_LIMIT`) rather than silently disabled.
- **Strength** — Low / Medium / High, meaningful for Amplify only (see above).
- **Chat** — type a message, get a reply generated with that intervention applied. Any change to concept, direction, or strength resets the conversation, in both modes, with no override — a reply produced under a different setting is never left on screen next to a changed control.
- **Compare** — runs the same prompt through the model with and without the intervention, side by side, so you can see exactly what the feature changed.
- **Public / Advanced views** — Public shows only model, concept, direction, and strength. Advanced additionally shows the resolved feature index, SAE id, layer, and positions mode, plus (on a real backend) the raw diagnostic verdict — mechanical-acceptance status, provenance, and fingerprints — behind the reply.
- **Language** — the whole interface (not model output) switches between English and French from one control.

A permanent banner states plainly when everything on screen is placeholder data, and disappears only once a real, non-stub backend is actually answering.

## Limitations

Read in full before treating any output as more than an engineering demonstration:

- **No concept here is scientifically validated.** `pro-american-exceptionalism`'s feature indices were measured — they passed a full discovery-gate scan on real weights — but no calibration boundary and no causal test has been run. `provenance: "candidate"` is the honest label; `--mode release` refuses to publish it, correctly.
- **The Gemma amplify doses are unmeasured.** Qwen's amplify strengths were remeasured against the feature's own observed activation range; Gemma's (1000 / 2500 / 5000) are the original placeholder values and are known, in engineering testing, to produce no visible effect at the high setting — that intervention path is suspect on its own terms, not merely uncalibrated.
- **The loaded SAE is not always the certified-primary configuration.** The mechanically-accepted intervention mechanism was proven at specific engineering layers (Gemma layer 31, Qwen layer 0); the shipped concept runs at the certified-primary layers instead (Gemma 29, Qwen 38). Every real generation is tagged accordingly and the tool refuses outright — rather than mislabeling the source — if a resolved target's layer disagrees with the layer actually loaded.
- **A feature index is not a "persona."** No feature has been shown to correspond to a stable trait, only to move a residual stream in a measured, on-concept direction on a handful of prompts.
- **This is a single-session, single-user tool.** No queue, no auth, no rate limiting — do not bind `0.0.0.0` on a shared machine.
- **The scientific record lives elsewhere.** This repository owns the UI and the release gate only; it is a downstream consumer of another repository's discovery and evidence work, never a source of truth for either. See [`BOUNDARY.md`](BOUNDARY.md).

## Citation

If this tool is useful in your own work, please cite it — see [`CITATION.cff`](CITATION.cff).

## Licence

[MIT](LICENSE).

## Author

Mohamed El Yazid El Yaakoubi — IID. [LinkedIn](https://www.linkedin.com/in/mohamed-el-yazid-el-yaakoubi/)
