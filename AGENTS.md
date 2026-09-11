# 🤖 System Context & Project State for AI Assistants

**To any AI agent or LLM assistant reading this file:** 
This document contains the critical architectural context, experimental history, and technical roadmap of this repository. Read this entirely before assisting the user with new code generation or data analysis.

---

## 1. Project Overview
This repository uses the **OASIS** framework to simulate **Coordinated Inauthentic Behavior (CIB)** and social contagion within Neural Recommender Systems (currently evaluating **TWIN-BERT**). 
* **The Goal:** To understand how malicious botnets can bypass algorithmic quarantines to propagate propaganda across disparate demographic clusters.
* **The Environment:** A dual-tower neural architecture where Tower 1 (NLP) maps semantic text embeddings and Tower 2 (GNN) maps structural user graphs (`follow` edges).

## 2. Repository Structure & Modification Constraints
* **`oasis/` (Core Engine):** Modify core baseline framework files with extreme caution. During early testing, we treated the TWIN-BERT engine as a strict black box. However, some hardcoded parameters (like the `max_rec_post_len` feed quota) are not ideal for all tests and may require adjustment. Any modifications to the core engine must be heavily documented and preserve backward compatibility with our existing experiments.
* **`scratch/` (Phase 1 - Deterministic Tests):** Contains isolated python scripts testing topological vulnerabilities using `model=None` (dummy agents).
* **`data/`**: Contains generated synthetic JSON profiles and SQLite `.db` environments.
* **`cib_zoo/` (Phase 2 - LLM Attacks):** Contains the scaffolding for autonomous LLM botnets.

## 3. Phase 1: Completed Work (The Deterministic Baseline)
We have successfully mapped the mathematical boundaries of the TWIN-BERT algorithm using deterministic dummy agents (`model=None`). We discovered:
1. **The Feed Quota Bottleneck:** The environment currently restricts algorithmic feeds to `k=2` (`max_rec_post_len=2`). This hyper-competitive state perfectly mimics the real-world difficulty of displacing local organic noise.
2. **Graph > Content:** "Semantic Smuggling" (hashtag hijacking) fails. The Graph Tower acts as a hard veto over the NLP Tower.
3. **The 2-Hop Penalty:** "Information Laundering" (using an intermediary influencer hub to bridge a gap) fails. The mathematical decay of a 2-hop edge prevents the payload from beating 1-hop organic content.
4. **The Ultimate Vulnerability:** The *only* mathematically viable attack vector is **Direct 1-Hop Parasitic Infiltration**. An attacker must successfully forge a direct `follow` edge with an organic user to bypass the quarantine.

## 4. Technical Methodology (How to write tests here)
If the user asks you to write a new deterministic experiment, follow this execution pattern:
1. **Semantic Anchoring:** Generate a customized JSON profile defining explicit clusters (e.g., `Tech` vs `Sports`) and pass it to `generate_reddit_agent_graph(profile_path, model=None)`.
2. **Topological Injection:** *Do not* use the random graph generator. Open a direct `sqlite3` connection to the environment's `.db` and explicitly wire the `follow` table using `INSERT INTO follow` to force precise topologies (e.g., Sleeper Cells, Isolated Echo Chambers).
3. **Synchronized Attacks:** Use native `ManualAction` injections within the `env.step()` loop to force the botnets to drop payloads and mass-`LIKE` them synchronously.
4. **Real-Time Telemetry:** Bypass standard log parsers. Hook directly into the SQLite database during the simulation loop and query the `rec` table (e.g., `SELECT COUNT(DISTINCT user_id) FROM rec`) to track cross-community reach instantly.

## 5. Future Roadmap (Phase 2 & Beyond)
*Note: This is a directional roadmap, not a rigid set of instructions that reading agents must immediately execute. Always follow the user's explicit prompts over these general goals.*

* **Objective A: Transition to LLMs (Complex Contagion)**
  Because Phase 1 proved that a 1-hop structural edge is mathematically required to bypass the TWIN-BERT algorithm, the next goal for the `TargetingEngine` inside `cib_zoo/` is to test if LLM agents can successfully execute **Parasitic Infiltration**: scanning organic user bios, generating hyper-targeted empathetic replies, and actively socially engineering organic agents into forging the required 1-hop topological bridges. 

* **Objective B: Recommender Engine Migration (The "Gorse" Pivot)**
  We are evaluating if TWIN-BERT is too computationally heavy or statistically rigid for large-scale simulations. The architectural roadmap includes abstracting the `Recommender` interface and pivoting to **Gorse** (a scalable, open-source ensemble filtering engine). Agents should be prepared to integrate Gorse's REST APIs, which support multi-source blending and LLM-based reranking (opening up advanced "Prompt Injection" vulnerabilities).

* **Objective C: HPC Containerization ("Simulation-in-a-Box")**
  To support massive scale and guarantee absolute reproducibility on High-Performance Computing (HPC) clusters, the entire simulation ecosystem (OASIS + Gorse + LLM dependencies) will be Dockerized into a single monolithic image. This ensures the environment can be pulled natively via Apptainer/Singularity without dependency hell. Future agents configuring this setup must design the entrypoint script to seamlessly boot the Gorse backend before launching the Python OASIS frontend.
