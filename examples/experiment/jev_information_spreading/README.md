# JEV — Information Spreading in X (OASIS paper F.2.2 reproduction)

This directory reproduces the OASIS paper's **"Information Spreading in X"**
baseline (Appendix F.2.2 / Table 16) using the **JEV engine** instead of stock
per-agent CAMEL inference. Everything else — platform, clock, dataset, agent
generation, recsys — is identical to the upstream Twitter harness
(`examples/experiment/twitter_gpt_example/twitter_simulation.py`).

## Baseline spec (from the paper)

| Parameter | Paper value |
|---|---|
| Action space | `like_post`, `repost`, `follow`, `do_nothing` |
| Time steps | 50 (each = 3 min sandbox time) |
| Comparison window | first 150 simulated minutes vs real propagation |
| Agents | ~300 (dataset-dependent) |
| Hardware | 1× NVIDIA A100-SXM4-80GB |
| Agent model | `meta-llama/Meta-Llama-3-8B-Instruct` (App. C.4; ablation: Qwen1.5-7B-Chat, Internlm2-chat-20b) |
| Repetitions | 10× per topic (one noisy, one clean) |

### Action mapping (paper → JEV classifier token)

| Paper action | JEV token | Target |
|---|---|---|
| `like_post` | `L` | the post |
| `repost` | `R` | the post |
| `follow` | `F` | **the post's author** (`post.user_id`) |
| `do_nothing` | `S` (Skip) | — |

`follow` targets the author of the post the agent reacted to, consistent with
the paper (§2.1: *"the user's relations network is updated when they follow a
new user"*). It fits JEV's per-(agent, post) 1-token model cleanly.

## Files

- `jev_information_spreading.py` — the driver. Mirrors the upstream Twitter
  harness but drives the sim with `OasisEnv.step_jev()` (sole step driver, so
  the clock advances strictly by +1 — avoids bug T-08).
- `jev_information_spreading.yaml` — config encoding the F.2.2 baseline.
- `jev_oasis.def` — Apptainer image (Python 3.11 + torch/CUDA 12.1 + vLLM +
  OASIS deps).
- `run_jev_hpc.sbatch` — SLURM job: starts vLLM in-container, waits for health,
  runs the driver, tears down.

## Run on the HPC grid

```bash
# From the repository root, on the feat/jev-engine-core branch:

# 1) Build the image (on a build node with internet + fakeroot):
apptainer build \
    examples/experiment/jev_information_spreading/jev_oasis.sif \
    examples/experiment/jev_information_spreading/jev_oasis.def

# 2) Submit the job (UPB-grid defaults baked in: account=phd, partition=dgxa100):
sbatch examples/experiment/jev_information_spreading/run_jev_hpc.sbatch

# Override model / image / config / steps at submit time:
MODEL_PATH=/models/Qwen2.5-14B-Instruct \
CONTAINER_IMAGE=$HOME/oasis.sif \
NUM_TIMESTEPS=50 \
sbatch examples/experiment/jev_information_spreading/run_jev_hpc.sbatch
```

### Pre-downloaded models (`~/models`)

The runner follows the **same `~/models` convention as the P0 batch runners**
(`launch_fep_batch.sh`, `scratch/run_fep_baseline.sh`): it binds
`$HOME/models -> /models` into the container and resolves the weights from
there.

**Paper model:** the OASIS real-world information-propagation reproduction uses
**`meta-llama/Meta-Llama-3-8B-Instruct`** as the agent backend (paper App. C.4,
line 479; the ablation also tried `Qwen/Qwen1.5-7B-Chat` and
`internlm/internlm2-chat-20b`, Figure 13). The runner targets Llama-3-8B-Instruct
by default.

**First-run download + reuse:** if `~/models/Meta-Llama-3-8B-Instruct` is not
present, the runner downloads it there once via `huggingface-cli download
--local-dir ~/models/Meta-Llama-3-8B-Instruct` and reuses it on every later run.
Because the download must reach the hub, the offline env is lifted *only* for
that step; the vLLM server still runs fully offline against the cached dir.

> ⚠️ **Meta-Llama-3 is a GATED HuggingFace repo.** Before the first run, on an
> ONLINE node: request access on the model page, then either `export HF_TOKEN=...`
> (passed through to the download) or run `huggingface-cli login`. Subsequent
> runs need no network for the model.

Override the model: `MODEL_PATH=/models/<dir>` to pin a local one, or
`PAPER_MODEL_REPO=<hf/id>` to download a different repo into `~/models`.

### Alignment with the P0 runners (and intentional differences)

Inherited from the P0 runners: `~/models` mount + resolution, offline
HF/vLLM env, SIF auto-discovery, isolated port (`18000 + JOB_SEED % N`),
health-wait loop, `cleanup()` trap, thread caps, `dgxa100`/A100 directives.

Intentionally dropped for this baseline:
- **Gorse recommender** (port 8088) — Information Spreading uses stock OASIS
  `twhin-bert` recsys, not the P0 Gorse/LightGCN hybrid.
- **`--enable-auto-tool-choice --tool-call-parser hermes`** — those serve
  CAMEL's multi-turn tool-calling path; JEV hits raw `/v1/completions` with
  `logit_bias` + `max_tokens=1`, so tool-choice flags are not applicable.
- **The 10-run CIB matrix** — this is a single-baseline reproduction, not the
  bot-ratio/topology sweep.

### Hermetic smoke test (no GPU, no vLLM)

Set `inference.jev_backend: mock` in the YAML (or run with it) to exercise the
full pipeline with the `MockJEVClassifierClient` — useful for CI and for
checking the harness before burning GPU hours.

## Outputs to send back for parity interpretation

- `data/simu_db/jev_information_spreading.db` — the simulation SQLite DB
  (post table, trace table, user relations). This is the propagation record.
- `slurm-<jobid>.out` / `.err` — per-step action distributions + timing.
- `vllm-<jobid>.log` — classifier server log (throughput, any logit-bias
  fallbacks relevant to bug T-06).

Attach these and I'll compute scale / depth / max-breadth and the Normalized
RMSE against the real propagation to assess feature parity.

## Scope / assumptions

- **Only** Information Spreading is implemented. Other baselines (Group
  Polarization, Herd Effect) need `dislike_post`, comment-engagement, and
  `search/trend/refresh` actions — deferred.
- `recsys.py` and `platform.py` are left **100% upstream** — the paper baseline
  uses stock OASIS recsys, not the P0 Gorse/LightGCN hybrid.
- `do_nothing` ≡ JEV Skip (`S`). `max_actions_per_agent=1` makes JEV's
  per-post-parallel-then-prune behavior equivalent to the paper's
  one-decision-per-activated-agent-per-step model.
- Belief updates are **disabled** for this baseline (no stance dynamics in
  Information Spreading).
