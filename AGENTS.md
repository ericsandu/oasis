# 🤖 System Context & Project State for AI Assistants

**To any AI agent or LLM assistant reading this file:**
This document contains the critical architectural context, experimental history, and technical roadmap of this repository. Read this entirely before assisting the user with new code generation or data analysis.

______________________________________________________________________

## 1. Project Overview

This repository uses the **OASIS** framework to simulate **Coordinated Inauthentic Behavior (CIB)** and social contagion within modern Neural and Behavioral Recommender Systems (evaluating a hybrid of **Gorse** and **TWIN-BERT**).

- **The Goal:** Understand how autonomous botnets and coordinated actors bypass algorithmic quarantines to propagate out-of-context propaganda across disparate demographic clusters (e.g., `#tech` propaganda into a `#sports` community).
- **The Architecture:** A Twitter-style multi-objective hybrid recommender combining:
  1. **In-Network Graph (SQL):** Direct follows and 2-hop social proof ("liked by friend").
  2. **Behavioral & Interaction Graph (Gorse):** Collaborative filtering, popularity velocity, and recency exploration.
  3. **Semantic Content Engine (TwinBERT):** Dense sentence-transformer embeddings mapping user interest vectors against tweet text.

## 2. Repository Structure & Modification Constraints

- **`oasis/` (Core Engine):** Contains OASIS platform logic. `platform.py` coordinates environment ticks and feed assembly. `oasis/social_platform/gorse_client.py` handles delta-syncing state to Gorse.
- **`gorse_config.toml`:** Configuration for the embedded Gorse engine. Configured with:
  `explore_recommend = { popular = 0.3, latest = 0.2, collaborative = 0.5 }`.
- **`cib_zoo/` (Phase 2 - LLM & Behavioral Attacks):** Contains modular botnet attack implementations (Like Farms, Comment Raids, Hashtag Hijackers, Repost Botnets, Sleeper Cells, and Co-engagement Poisoning).
- **`data/`:** Contains generated synthetic JSON profiles (`synthetic_bridge.json`) and SQLite simulation artifacts.
- **`scratch/`:** Empirical test harnesses, diagnostic scripts, and time-series telemetry.

## 3. Key Findings & Empirical Discoveries

1. **The TwinBERT Veto (Phase 1):** In pure semantic two-tower models, content is quarantined if topological or keyword proximity is missing. However, relying purely on TwinBERT creates aggressive echo chambers.
2. **The Pure-Gorse Recency Trap:** When testing pure Gorse without semantic constraints, brand-new posts achieve a 100% breach of foreign clusters due to Gorse's default `latest` cold-start exploration fallback, rendering naive like-farming artificially overpowered.
3. **Hashtag Dynamics (TwinBERT vs. Gorse):**
   - *TwinBERT:* Resists hashtag hijacking because it embeds the entire sentence semantics; appending `#sports` to a crypto pitch only shifts the 768-d vector by $\Delta \approx 0.03$.
   - *Gorse:* Highly vulnerable to hashtags via its Factorization Machine (CTR model) and category exploration, because Gorse treats `Item.Labels` as discrete categorical tokens without evaluating contextual text dissonance.
4. **The 60/40 Twitter-Like Hybrid Solution:**
   To accurately mirror Twitter's open-sourced "For You" architecture, the platform combines Gorse (60%) and TwinBERT (40%):
   $$\text{Total Score} = 0.40 \cdot S_{\text{TwinBERT}} + 0.30 \cdot S_{\text{CF}} + 0.18 \cdot S_{\text{Pop}} + 0.12 \cdot S_{\text{Rec}}$$
   This prevents trivial recency spam while allowing authentic viral or bridged crossover content to propagate.

## 4. Technical Methodology & HPC (FEP / UPB Grid) Deployment

### Institutional Infrastructure
Simulations are targeted for the National University of Science and Technology POLITEHNICA Bucharest (UPB) institutional grid (`fep8.grid.pub.ro`), utilizing Slurm scheduling and Apptainer (Singularity) container execution.

* **Target GPU Partitions:** `ucsx` (Tesla A100 3x/node, 512GB RAM), `ml` (Tesla A100 2x/node, 128GB RAM), `hd` (Tesla A100 10x/node).
* **Container Workflow:** Images cannot be compiled on worker nodes. The workflow requires building Docker images locally (`camel-oasis:latest`), pushing to DockerHub, and pulling on the login node:
  ```bash
  apptainer pull docker://<dockerhub-user>/camel-oasis:latest
  ```
* **Execution Flags:** Must specify `--nv` for CUDA GPU acceleration and bind local host storage:
  ```bash
  apptainer run --nv --bind ./data:/app/data --bind ./experiments:/app/experiments camel-oasis_latest.sif
  ```

### High-Throughput 8-bit LLM Inference Architecture
To support cohorts of **1,000 to 5,000+ active agents** without compute bottlenecks:
* **Recommended Models:** `Meta-Llama-3.1-8B-Instruct-FP8` (or `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8`).
* **VRAM Efficiency:** 8-bit quantization reduces model weight footprint to **~8.5 GB**, freeing >30 GB (on 40GB A100) or >70 GB (on 80GB A100) strictly for vLLM PagedAttention KV-cache.
* **vLLM Optimizations:**
  - `--max-model-len 2048`: Capping context length prevents memory fragmentation and multiplies concurrent agent batching capacity by 10x.
  - `--kv-cache-dtype fp8`: Halves KV-cache memory usage per token.
* **Multi-GPU Parallelism:**
  Run 1 vLLM server process per GPU (e.g. GPU 0 on port 8000, GPU 1 on port 8001). OASIS automatically round-robins agent prompt generation across ports via its `server_url` config list.

### Database Architecture: Preserving Experiment Bundles
OASIS writes platform state to `twitter_simulation.db`, while Gorse writes interaction matrices to `gorse_data.db` and precomputed caches to `gorse_cache.db`. **Do not attempt to merge these into a single SQLite file**; doing so causes multi-process write-lock crashes and destroys compatibility with Gorse's native web dashboard. Store them together in a run folder for local post-hoc visualization via `gorse-in-one -c gorse_config.toml`.

## 5. Active Roadmap (Phase 2: Complex Contagion & LLM Botnets)

- **Objective A: Implement the 60/40 Blended Ranker:** Finalize the unified candidate scoring pipeline merging SQL in-network candidates with Gorse out-of-network candidates evaluated by TwinBERT.
- **Objective B: Deploy Advanced CIB Botnets (`cib_zoo/`):**
  - **The Semantic Smuggler:** LLM-driven adversarial rewriting of `#tech` payloads to align with `#sports` TwinBERT vectors.
  - **The Organic Bridger / Sleeper Cell:** Bots that spend time establishing genuine collaborative filtering overlap in the target community before dropping propaganda.
  - **Co-Engagement Poisoning:** Artificially corrupting Gorse's item-to-item Jaccard matrix by co-liking target community anchor posts.
  - **Astroturfing Reply Raids:** Bypassing recommender feeds entirely by flooding the comment threads of high-reach organic posts.
  - **Closed-Loop Stepping-Stone Diffusion:** Using sentinel accounts to trigger multi-hop cascades based on organic adoption thresholds.
- **Objective C: Scaled Simulation on FEP:** Execute multi-step cascade experiments on the HPC cluster across Topologies A, B, C, and D using the containerized image and export full telemetry bundles.
