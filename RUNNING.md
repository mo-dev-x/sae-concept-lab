# Running SAE Concept Lab — step by step

This guide takes you from a fresh clone to a working interface. It covers two
modes:

- **Mode 1 — local, no GPU.** Works on any laptop in about two minutes. The
  interface is fully functional; the model replies are synthetic.
- **Mode 2 — real models on a GPU cluster.** The same interface, driving a real
  Gemma or Qwen with real sparse-autoencoder interventions.

Start with Mode 1. If it works, Mode 2 is the same app with different flags.

---

## Mode 1 — run it locally (no GPU, ~2 minutes)

### 1. Get the code

```bash
git clone https://github.com/mo-dev-x/sae-concept-lab.git
cd sae-concept-lab
```

### 2. Make a virtual environment

Python **3.11 or newer** is required.

```bash
python -m venv .venv
```

Activate it — this differs by platform:

| Platform | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Git Bash) | `source .venv/Scripts/activate` |
| macOS / Linux | `source .venv/bin/activate` |

### 3. Install

```bash
pip install -e ".[test]"
```

### 4. Check the install (optional, ~30 s)

```bash
pytest -q
```

### 5. Launch

```bash
python -m sae_concept_lab.app --server-name 127.0.0.1 --server-port 7860
```

Open **http://127.0.0.1:7860** in a browser on the same machine.

`127.0.0.1` is deliberate: it means only this machine can reach the interface.
Do not change it to `0.0.0.0` on any shared or multi-user machine — that
publishes the interface, and everything you type into it, to every other user.

### What you will see

A concept, a direction (**amplify** or **suppress**), a strength, and a chat box.
Type a message and press **Enter** (or click Send).

Replies in this mode are prefixed `[FAKE STUB -- UI TEST ONLY]` and a banner says
so. That is correct and intended: no model is loaded, so there is nothing real to
report. The banner disappears by itself in Mode 2, because it is wired to whether
a stub is actually answering rather than to a flag.

---

## Mode 2 — real models on a GPU cluster

You need: a machine with an NVIDIA GPU, the model and SAE weights already on
disk, and Python 3.11+.

### 1. Stage the weights

Download the four snapshots on a machine **with** internet, then copy them to the
GPU machine. Compute nodes on most clusters have no internet, which is why this
is a separate step.

| What | Repository |
|---|---|
| Gemma model | `google/gemma-3-12b-it` |
| Gemma SAE | `google/gemma-scope-2-12b-it` |
| Qwen model | `Qwen/Qwen3.5-27B` |
| Qwen SAE | `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` |

### 2. Launch against the real weights

```bash
HF_HUB_OFFLINE=1 python -m sae_concept_lab.app \
  --gemma-backend runtime \
  --gemma-model-path  /path/to/gemma-3-12b-it \
  --gemma-sae-path    /path/to/gemma-scope-2-12b-it \
  --qwen-backend runtime \
  --qwen-model-path   /path/to/Qwen3.5-27B \
  --qwen-sae-path     /path/to/qwen-scope/layer0.sae.pt \
  --qwen-layer 0 \
  --server-name 127.0.0.1 --server-port 7860
```

Each backend is independent. Pass only the `--gemma-*` flags and the Qwen tab
stays on the stub, which is a normal way to run it.

`HF_HUB_OFFLINE=1` guarantees nothing is fetched from the network at run time.
Weights load from the paths you gave and nowhere else.

### 3. Reach the interface from your laptop

The app binds loopback on the GPU machine, so forward the port rather than
exposing it. On a SLURM cluster:

```bash
# on the cluster: allocate a node, and note its name
salloc --nodes=1 --gres=gpu:1 --time=03:00:00
hostname          # e.g. node042

# in a SECOND terminal on your laptop
ssh -N -L 7860:127.0.0.1:7860 -J you@login.cluster you@node042
```

Then open **http://127.0.0.1:7860** locally.

`ssh -N` prints nothing and does not return a prompt. **That is success**, not a
hang — leave the terminal open. Close it and the interface goes away.

---

## Troubleshooting

**`Port 7860 is in use`** — something is already listening. Use
`--server-port 7861`, or stop the other process.

**The page loads but replies say `[FAKE STUB]`** — that tab is on the stub
backend. Check you passed `--gemma-backend runtime` (and/or `--qwen-backend
runtime`) and that the paths exist.

**`ssh -N` seems frozen** — it is meant to. It has no shell to give you.

**`REFUSING TO GENERATE ... targets layer N ... loaded is at layer M`** — the
concept was measured in one layer's dictionary and the loaded SAE is a different
layer. A feature index only means something inside the dictionary it was found
in, so this is refused rather than producing a confident wrong answer. Load the
matching SAE, or use a concept measured at the layer you loaded.

**`carries N targets ... clamps exactly one feature per call`** — the concept
names several features and this backend drives one at a time. It refuses rather
than silently steering only the first. Use a single-feature concept.

**`--mode release` refuses to start** — intended, and not a bug. Release mode
publishes only concepts whose evidence has been verified end to end. No shipped
concept meets that bar yet, so release mode is closed. Development mode is the
working mode.

---

## What this tool does and does not claim

It shows what happens to a language model's output when specific
sparse-autoencoder features are amplified or ablated. The concepts it ships carry
**measured** feature indices, but no calibrated dose and no causal validation:
the amplification strengths are engineering defaults, not experimentally derived
values. `provenance` is recorded on every concept, and the release gate refuses
to publish anything not fully attested.

Ablation has no strength. A feature is zeroed or it is not, so low, medium and
high are identical for **suppress** and differ only for **amplify**. That is a
property of the operation, not a missing feature.

The scientific definitions, the discovery process and the evidence live in a
separate repository. See [`BOUNDARY.md`](BOUNDARY.md) for what belongs where.
