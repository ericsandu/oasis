# 🏛️ Architecture: Twitter-Like Hybrid Recommender & CIB Attack Catalog

This document details the architectural specification of the **60/40 Hybrid Recommender System** designed for OASIS, along with a theoretical and empirical taxonomy of **Coordinated Inauthentic Behavior (CIB) attack vectors**, their predicted outcomes under a **Black-Box Threat Model**, and the deployment plan for the UPB High-Performance Computing (HPC) grid.

______________________________________________________________________

## Part 1: The Hybrid Recommender Architecture

To model modern social media platforms (specifically Twitter / X) without inheriting either the **monolithic echo chambers** of pure NLP models or the **cold-start vulnerabilities** of pure collaborative filtering, we utilize a multi-objective, candidate-sourced hybrid.

### 1. The Core Scoring Equation

For any user $u$ and candidate post $i$, the final feed score is computed as:

$$\text{Score}(u, i) = w_{\text{network}} + 0.40 \cdot S_{\text{TwinBERT}}(u, i) + 0.60 \cdot S_{\text{Gorse}}(u, i)$$

Where the Gorse behavioral component decomposes into:

$$S_{\text{Gorse}}(u, i) = 0.50 \cdot S_{\text{CF}}(u, i) + 0.30 \cdot S_{\text{Pop}}(i) + 0.20 \cdot S_{\text{Rec}}(i)$$

Expanding the full equation yields the **Net Effective Weights**:

$$\text{Score}(u, i) = w_{\text{network}} + \mathbf{0.40} \cdot S_{\text{TwinBERT}} + \mathbf{0.30} \cdot S_{\text{CF}} + \mathbf{0.18} \cdot S_{\text{Pop}} + \mathbf{0.12} \cdot S_{\text{Rec}}$$

| Term | Weight | Functional Role in Platform | Defense Function Against CIB |
| :--- | :---: | :--- | :--- |
| **$S_{\text{TwinBERT}}$** | **40%** | Semantic content relevance between user bio/history and post text | **Content Quarantine:** Penalizes off-topic payloads with an immediate 40-point deficit. |
| **$S_{\text{CF}}$** | **30%** | Bipartite user-item Collaborative Filtering (Matrix Factorization) | **Community Boundary:** Restricts dissemination to clusters with verified co-interaction history. |
| **$S_{\text{Pop}}$** | **18%** | Engagement velocity (likes, reposts, CTR) | **Trending Dynamics:** Allows viral posts to compete, but caps raw bot volume at 18% influence. |
| **$S_{\text{Rec}}$** | **12%** | Freshness exploration / age decay | **Discovery:** Gives new posts exploratory exposure without allowing recency monopolies. |
| **$w_{\text{network}}$** | *Additive* | In-network social affinity bonus ($+0.25$ if followed or 2-hop friend liked) | **Social Relevance:** Reflects direct relationships without completely bypassing quality scoring. |

---

### 2. Candidate Sourcing & Feed Quotas

Feed generation uses a **Two-Stage Candidate Funnel**:

1. **Candidate Pool A (In-Network & Social Proof via SQLite):**
   - **1-Hop Direct Follows:** Posts authored by accounts the user follows, ranked by gravity-decayed likes:
     $$\text{Rank} = \frac{\text{likes} + 1.0}{1.0 + 0.05 \cdot (t_{\text{now}} - t_{\text{created}})}$$
   - **2-Hop Social Proof (GraphJet analogue):** Posts liked by accounts the user follows (`"User X liked this"`).
2. **Candidate Pool B (Out-of-Network Discovery via Gorse):**
   - Top-20 candidates nominated by Gorse's collaborative filtering, trending, and exploratory indices.
3. **Unified Scoring & 50/50 Quota Assembly:**
   - Both candidate pools are pooled and scored via the 60/40 equation.
   - The final feed enforces a **balanced ratio**: max 50% in-network, min 50% out-of-network discovery. This guarantees discovery potential while honoring personal follow graphs.

---

### 3. The Black-Box Threat Model & Portfolio Theory of CIB

In real-world security scenarios, attackers operate with zero white-box access to the recommender weights:

$$\text{Score} = w_1 \cdot S_{\text{Semantics}} + w_2 \cdot S_{\text{Graph}} + w_3 \cdot S_{\text{Velocity}} \quad (\text{where } \sum w_i = 1)$$

* **The Single-Vector Trap:** If an attacker bets 100% of their compute budget on volume ($S_{\text{Velocity}} = 1.0$), but the platform weighs velocity at only $18\%$, the attack collapses ($0.18$).
* **Orthogonal Hedging (Portfolio Diversification):** If the attacker distributes their budget across all three orthogonal dimensions simultaneously:
  $$\vec{S}_{\text{attack}} = (S_{\text{Semantics}} \approx 0.8, \, S_{\text{Graph}} \approx 0.8, \, S_{\text{Velocity}} \approx 0.8)$$
  Because the dimensions are independent, **no matter what the platform's internal weights $(w_1, w_2, w_3)$ are**, the dot product is mathematically guaranteed to be high:
  $$\text{Score} = w_1(0.8) + w_2(0.8) + w_3(0.8) = 0.8 \sum w_i = \mathbf{0.80}$$

---

### 4. Hashtag Vulnerability: TwinBERT vs. Gorse

A key finding of this research is the stark difference in how dense NLP models and categorical recommenders handle hashtags:

1. **TwinBERT (Dense Whole-Sentence Semantics):**
   * Embeds the entire text into a continuous 768-dimensional latent space.
   * Appending `#sports` to a 200-word cryptocurrency pitch only shifts the sentence embedding vector by a negligible margin ($\Delta \approx 0.03$). The payload is still classified as `#tech` and quarantined.
2. **Gorse (Factorization Machines & Discrete Labels):**
   * Gorse's CTR predictor treats `Item.Labels` as **discrete categorical one-hot tokens**.
   * Appending `#sports` creates an un-diluted feature match: $\langle \mathbf{v}_{\text{user}}, \mathbf{v}_{\text{sports}} \rangle > 0$.
   * **Result:** Gorse is significantly *more* vulnerable to naive hashtag hijacking than TwinBERT, because it lacks contextual semantics to recognize off-topic dissonance.

---

## Part 2: Comprehensive Taxonomy of CIB Attacks & Predicted Outcomes

```
                                      [ CIB Attack Catalog ]
                                                 │
      ┌───────────────────────────┬──────────────┴────────────┬───────────────────────────┐
      ▼                           ▼                           ▼                           ▼
[ Content / Semantic ]    [ Engagement Velocity ]     [ Graph & Topology ]        [ Multi-Vector Hybrid ]
• Bio-Scraping Copyattack • Like-Farming Bursts       • Sleeper Organic Bridger   • Orthogonal Hedging
• Hashtag Hijacking       • Reply Raid / Astroturf    • Co-Engagement Poisoning   • Sentinel-Tuned Bandit
                          • Repost Cascade Botnets    • 2-Hop Social Proof Hijack • Trojan Horse (Apex)
```

---

### Tier 1: Content & Semantic Manipulation Vectors

#### ATK-1: Bio-Scraping Copyattack (Profile Mirroring)
* **Mechanism:** Bot scrapes target community bios and embeds matching high-density n-grams directly into the payload text or metadata.
* **Math Profile:** $S_{\text{TwinBERT}} \to 1.0$, $S_{\text{CF}} = 0.0$, $S_{\text{Pop}} = 0.0$.
* **Total Score:** $0.40 \times 1.0 + 0.60 \times 0.0 = \mathbf{0.40}$ (plus 0.12 recency = $\mathbf{0.52}$).
* **Predicted Outcome:** **QUARANTINED (< 10% reach)**.
* **Why:** In-domain organic posts have both semantic alignment and collaborative filtering overlap ($> 0.70$), out-ranking the copyattack.

#### ATK-2: Hashtag Hijacking & Keyword Co-opting
* **Mechanism:** Prepending trending target community hashtags (`#SuperBowl`, `#WorldCup`) to out-of-domain payloads.
* **Target Objective:** Exploits Gorse's Factorization Machine labels and category exploration.
* **Predicted Outcome:** **MARGINAL (5% – 15% reach)**. Neutralized by TwinBERT's whole-sentence semantic check.

---

### Tier 2: Engagement Velocity & Momentum Vectors

#### ATK-3: Like-Farming Bursts (`like_farm.py`)
* **Mechanism:** $N$ bots synchronously like a target payload within 1 simulation step.
* **Math Profile:** $S_{\text{Pop}} = 1.0$, $S_{\text{Rec}} = 1.0$, $S_{\text{TwinBERT}} = 0.0$, $S_{\text{CF}} = 0.0$.
* **Total Score:** $0.18 + 0.12 = \mathbf{0.30}$.
* **Predicted Outcome:** **QUARANTINED (< 5% reach)**. Volume alone cannot overcome the 70% penalty from missing semantics and collaborative history.

#### ATK-4: Astroturfing Comment Spam & Reply Raids (`comment_raid.py`)
* **Mechanism:** Bots bypass feed algorithms entirely by flooding the reply sections of the top 5 highest-reach organic posts with pre-scripted propaganda templates.
* **Target Objective:** Parasitizes established organic viral inventory.
* **Predicted Outcome:** **HIGH EXPOSURE (50% – 80% thread visibility)**. Extremely effective because reply threads are rendered chronologically without recommender candidate filtering.

#### ATK-5: Repost Cascade Botnets (`repost_botnet.py`)
* **Mechanism:** Multi-tiered hierarchy of bots quote-tweeting and retweeting the seed payload at staggered intervals.
* **Target Objective:** Inflates $S_{\text{Pop}}$ while triggering in-network feed injection across the bot follower sub-network.

---

### Tier 3: Graph Topology & Collaborative Filtering Vectors

#### ATK-6: Co-Engagement Poisoning (Jaccard Association Attack)
* **Mechanism:** Bots systematically co-like a popular organic sports post *and* the target tech payload in the same session.
* **Target Objective:** Forces high Jaccard similarity in Gorse's Item-to-Item matrix:
  $$\text{Similarity}(\text{Payload}, \text{SportsPost}) = \frac{|U_{\text{payload}} \cap U_{\text{sports}}|}{|U_{\text{payload}} \cup U_{\text{sports}}|}$$
* **Predicted Outcome:** **MODERATE-HIGH BREACH (35% – 50% reach)**. Triggers Gorse's item-to-item fallback, serving the payload under sports recommendations.

#### ATK-7: The Sleeper Organic Bridger (`sleeper_cell_cib.py`)
* **Mechanism:** 3-phase lifecycle: (1) Burn-in imitation of target community $\to$ (2) Latent CF vector formation ($S_{\text{CF}} \to 1.0$) $\to$ (3) Coordinated strike.
* **Math Profile:** $S_{\text{CF}} = 0.90 \to 0.27$, $S_{\text{Pop}} = 0.18$, $S_{\text{Rec}} = 0.12$, $S_{\text{TwinBERT}} = 0.0$.
* **Total Score:** $0.27 + 0.18 + 0.12 = \mathbf{0.57}$.
* **Predicted Outcome:** **HIGH PENETRATION (40% – 65% reach)**. Corrupts Gorse's collaborative filtering into treating bots as authentic community members.

#### ATK-8: 2-Hop Social Proof Hijack (GraphJet Exploit)
* **Mechanism:** Socially engineering a single mutual organic micro-influencer to like or retweet the payload.
* **Math Profile:** Enters In-Network Candidate Pool A (`friend_likes >= 1`), receiving the $+0.25$ social affinity bonus.
* **Predicted Outcome:** **LOCALIZED CLUSTER SATURATION (70% – 90% reach among that influencer's followers)**.

---

### Tier 4: Black-Box Adaptive & Apex Hybrids

#### ATK-9: Orthogonal Portfolio Hedging (The Generalist Attack)
* **Mechanism:** Allocates budget equally: 33% bio-scraping (ATK-1), 33% sleeper bridging (ATK-7), and 33% burst velocity (ATK-3).
* **Predicted Outcome:** **ROBUST BREACH ACROSS ALL CONFIGURATIONS (55% – 75% reach)**. Proves that multi-vector attacks render recommender parameter tuning obsolete.

#### ATK-10: Closed-Loop Sentinel-Tuned Bandit (Adaptive Controller)
* **Mechanism:** Autonomous LLM operating 2 Sentinel observer accounts in the target community. It reads public feed feedback at each step and runs Thompson Sampling to dynamically shift bot actions between like-farming, bio-scraping, and co-engagement poisoning.

#### ATK-11: The "Trojan Horse" (Apex Multi-Vector)
* **Mechanism:** Simultaneously deploys Bio-Scraping + Co-Engagement Poisoning + Sleeper Bridging + Burst Likes.
* **Total Score:** $0.34 (\text{Sem}) + 0.27 (\text{CF}) + 0.18 (\text{Pop}) + 0.12 (\text{Rec}) = \mathbf{0.91}$.
* **Predicted Outcome:** **TOTAL BREACH (> 90% reach)**. Mathematically defeats the entire hybrid defense stack.

---

## Part 3: Institutional HPC (UPB Grid) Deployment Blueprint

### 1. Compute Infrastructure (POLITEHNICA Bucharest Grid)
* **Access Point:** `fep8.grid.pub.ro`
* **Target Partitions:** `ucsx` (Tesla A100 3x/node, 512GB RAM) and `ml` (Tesla A100 2x/node, 128GB RAM).
* **Target Scale:** **1,000 to 5,000+ active agents** across multi-community topologies.

### 2. High-Throughput 8-bit vLLM Inference
* **Primary Model:** `Meta-Llama-3.1-8B-Instruct-FP8` (Footprint: ~8.5 GB VRAM).
* **Throughput Flags:** `--max-model-len 2048 --kv-cache-dtype fp8`.
* **Multi-GPU Parallelism:** Launch independent vLLM instances per GPU (e.g. ports 8000, 8001); OASIS automatically load-balances agent prompt requests across ports via `server_url`.

### 3. Slurm Submission Script (`run_fep_matrix.sh`)
```bash
#!/bin/bash
#SBATCH --job-name=cib_oasis_hpc
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

echo "Node: $(hostname) | CUDA: $CUDA_VISIBLE_DEVICES"

mkdir -p ./data ./experiments

apptainer run --nv \
  --bind $(pwd)/data:/app/data \
  --bind $(pwd)/experiments:/app/experiments \
  --bind $(pwd)/gorse_config.toml:/app/gorse_config.toml \
  camel-oasis_latest.sif \
  bash -c "PYTHONPATH=/app poetry run python scratch/hpc_matrix_runner.py"
```

### 4. Telemetry & Experiment Bundles
Each run outputs an isolated directory:
```
experiments/run_<timestamp>_<topology>_<recsys>_<attack>/
├── twitter_simulation.db   <-- Agent opinions, follow graph, full tweet texts
├── gorse_data.db           <-- Raw interactions & items
├── gorse_cache.db          <-- Precomputed recommendation caches & precision/recall curves
└── telemetry_summary.json  <-- DIR, IDR, and hop-latency metrics
```
This bundle preserves full compatibility with Gorse's native web dashboard (`gorse-in-one -c gorse_config.toml`) for post-hoc interactive analysis on local workstations.
