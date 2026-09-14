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
- **`cib_zoo/` (Phase 2 - LLM & Behavioral Attacks):** Contains modular botnet attack implementations (Like Farms, Comment Raids, Hashtag Hijackers, Repost Botnets, and Sleeper Cells).
- **`data/`:** Contains generated synthetic JSON profiles (`synthetic_bridge.json`) and SQLite simulation artifacts.
- **`scratch/`:** Empirical test harnesses, diagnostic scripts, and time-series telemetry.

## 3. Key Findings & Empirical Discoveries

1. **The TwinBERT Veto (Phase 1):** In pure semantic two-tower models, content is quarantined if topological or keyword proximity is missing. However, relying purely on TwinBERT creates aggressive echo chambers.
2. **The Pure-Gorse Recency Trap:** When testing pure Gorse without semantic constraints, brand-new posts achieve a 100% breach of foreign clusters due to Gorse's default `latest` cold-start exploration fallback, rendering naive like-farming artificially overpowered.
3. **The 60/40 Twitter-Like Hybrid Solution:**
   To accurately mirror Twitter's open-sourced "For You" architecture, the platform combines Gorse (60%) and TwinBERT (40%):
   $$\text{Total Score} = 0.40 \cdot S_{\text{TwinBERT}} + 0.30 \cdot S_{\text{CF}} + 0.18 \cdot S_{\text{Pop}} + 0.12 \cdot S_{\text{Rec}}$$
   This prevents trivial recency spam while allowing authentic viral or bridged crossover content to propagate.

## 4. Technical Methodology & HPC (FEP) Workflow

1. **Monolithic Docker Deployment:**
   The `Dockerfile` embeds `gorse-in-one` into the container's entrypoint (`entrypoint.sh`). When the container runs, Gorse automatically boots in the background at `http://127.0.0.1:8088` before the Python simulation begins.
2. **Preserving Dual Databases as an Experiment Bundle:**
   OASIS writes platform state to `twitter_simulation.db`, while Gorse writes interaction matrices to `gorse_data.db` and precomputed caches to `gorse_cache.db`. **Do not attempt to merge these into a single SQLite file**; doing so causes multi-process write-lock crashes and destroys compatibility with Gorse's native web dashboard. Store them together in a run folder for local post-hoc visualization via `gorse-in-one -c gorse_config.toml`.
3. **CI/Linting Pipeline:**
   The CI pipeline relies strictly on **Ruff** (`ruff check .`, `ruff format --check .`) and **yamllint** with a custom `.yamllint` ignoring binary virtual environments. Strict pre-commit hooks and flake8 have been deprecated.

## 5. Active Roadmap (Phase 2: Complex Contagion & LLM Botnets)

- **Objective A: Implement the 60/40 Blended Ranker:** Finalize the unified candidate scoring pipeline merging SQL in-network candidates with Gorse out-of-network candidates evaluated by TwinBERT.
- **Objective B: Deploy Advanced CIB Botnets (`cib_zoo/`):**
  - **The Semantic Smuggler:** LLM-driven adversarial rewriting of `#tech` payloads to align with `#sports` TwinBERT vectors.
  - **The Organic Bridger / Sleeper Cell:** Bots that spend time establishing genuine collaborative filtering overlap in the target community before dropping propaganda.
  - **2-Hop Social Proof Hijackers:** Bots engineering engagement from mutual bridge accounts to exploit in-network feed injection.
- **Objective C: Scaled Simulation on FEP:** Execute multi-step cascade experiments on the HPC cluster using the containerized image and export full telemetry bundles.
