# Architectural Specification & Implementation Plan: JEV Mode Optimization for OASIS

**Target Framework**: OASIS Social Simulation Platform  
**Target Component**: Organic Agent Inference Engine & Temporal Action Scheduling  
**Author**: Antigravity  
**Status**: Proposal / Draft  

---

## 1. Executive Summary

In OASIS simulations, simulating organic user behavior is the primary computational bottleneck. Under the existing CAMEL-AI architecture, every organic agent generates 150–400 tokens of verbose JSON function-calling schema per step just to emit binary feed actions (e.g., `like_post`, `repost`, or `do_nothing`). Autoregressive decoding is memory-bandwidth bound, limiting simulations on standard compute nodes to 20–50 concurrent organic agents.

Furthermore, in Twitter mode, organic users **purely interact with their recommendation feed** (served via `refresh()`), with no native post or hashtag search actions. 

**JEV Mode** (Joint Evaluation Vector mode) replaces per-agent autoregressive JSON generation with:
1. **Prefix KV-Cached Batched Feed Evaluation**: Leveraging vLLM Automatic Prefix Caching (APC) or SGLang RadixAttention over shared feed item representations.
2. **1-Token Constrained Action Classification**: Projecting agent decisions into a single output token (`L` for Like, `R` for Repost, `C` for Comment, `S` for Skip) via logit masking, eliminating 90–95% of decoding steps.
3. **Conditional Comment Generation Fallback**: Autoregressively generating text *only* for the small fraction (<10%) of agents that select `C`.
4. **Poisson Virtual Micro-Time Scheduling**: De-synchronizing agent step actions into realistic continuous timestamps across a simulation step window $\Delta T$ before committing to SQLite.

---

## 2. Baseline Bottleneck Analysis

### 2.1 The CAMEL-AI Autoregressive Overhead
Currently, each agent executes:
```python
# oasis/social_agent/agent.py
env_prompt = await self.env.to_text_prompt()
user_msg = BaseMessage.make_user_message(role_name="User", content=...)
response = await self.astep(user_msg)  # CAMEL ChatAgent step
for tool_call in response.info['tool_calls']:
    action_name = tool_call.tool_name
    args = tool_call.args
```
For 1,000 agents:
- Every agent generates a full OpenAI tool call: `{"name": "like_post", "arguments": "{"post_id": 42}"}`.
- Decode cost: $\approx 1{,}000 \times 200 \text{ tokens} = 200{,}000 \text{ decode steps/step}$.
- Decoder bandwidth saturation occurs rapidly, spiking GPU latency and memory consumption.
- Shared feed tokens across agents are repeatedly prefilled without KV-cache sharing.

### 2.2 Discrete Timestamp Collision
In `oasis/environment/env.py`:
- `asyncio.gather(*tasks)` fires all agent LLM coroutines concurrently.
- In Twitter mode, all actions in step t are recorded in SQLite (`post`, `trace`, `likes`) with identical discrete string timestamp `str(t)`.
- This creates unrealistic synchronization waves and prevents realistic sequential cascade modeling.

---

## 3. JEV Architecture Specification

```
                                   [ Recommendation System ]
                                               │
                                               ▼
                                 [ Common Feed Snapshot F_t ]
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
             [ Shared Prefix KV Cache ]                    [ Shared Prefix KV Cache ]
            Agent 1 Prompt: P_1 = (Bio_1, F_t)            Agent N Prompt: P_N = (Bio_N, F_t)
                        │                                             │
                        ▼                                             ▼
                 (vLLM / SGLang)                               (vLLM / SGLang)
             Logit-Constrained Decode                      Logit-Constrained Decode
                  (max_tokens=1)                                (max_tokens=1)
                        │                                             │
               Token: 'L' (Like)                             Token: 'C' (Comment)
                        │                                             │
                        │                                             ▼
                        │                                [ Autoregressive Fallback ]
                        │                                "Draft comment for Post X..."
                        │                                             │
                        │                                     Comment Text String
                        │                                             │
                        └──────────────────────┬──────────────────────┘
                                               │
                                               ▼
                              [ Virtual Micro-Time Scheduler ]
                               Assign tau_i ~ Exp(lambda_i)
                                   t_i = T_step + tau_i
                                               │
                                               ▼
                              [ Chronological Placement Queue ]
                                    Sort actions by t_i
                                               │
                                               ▼
                             [ SQLite DB & Platform Trace ]
```

### 3.1 Phase 1: 1-Token Action Classification & Prefix Caching

#### Action Vocabulary Projection
For feed post p_k in feed:
$\mathcal{V}_{\text{action}}$ = \{ 	ext{'L'}, 	ext{'R'}, 	ext{'C'}, 	ext{'S'} \}
- `L`: `LIKE_POST` (target: p_k)
- `R`: `REPOST` (target: p_k)
- `C`: `CREATE_COMMENT` (triggers Phase 2)
- `S`: `DO_NOTHING` / Skip

#### Prompt Structure (Cache Optimized)
```text
System: You are Twitter user @{user_name} ({bio}). You are scrolling your feed.
[POST ID: {post_id}] Author: @{author} | Content: "{content}"
Choose action: [L]ike, [R]epost, [C]omment, [S]kip. Output exactly one letter.
Action:
```
- Placing feed content in a uniform prefix enables **RadixAttention** (SGLang) or **Automatic Prefix Caching** (vLLM) to cache the feed representation across all agents reading the same item.

### 3.2 Phase 2: Conditional Comment Fallback
Only agents whose output token is `'C'` enter the autoregressive decoding stage.
- Typical interaction ratios on social feeds: Skip (~70%), Like (~20%), Repost (~5%), Comment (~5%).
- Out of 1,000 agents, only ~50 agents enter multi-token decoding.
- Prompt for Phase 2:
  ```text
  You decided to comment on post {post_id}: "{content}".
  Write a short, realistic tweet reply matching your persona:
  ```
- Max tokens set to 40–60 tokens.

### 3.3 Phase 3: Poisson Virtual Micro-Time Scheduling

To prevent artificial simultaneous bursts and correctly simulate dynamic cascade propagation:

1. **Window Definition**:
   Each global step represents simulation interval $\Delta T$ (e.g. 15 minutes = 900 seconds).

2. **Inter-Arrival Delay Sampling**:
   For agent i with user activity rate \lambda_i (derived from `UserInfo.activity_level_frequency`):
   	au_i \sim \min($\Delta T$, 	ext{Exponential}(\lambda_i))
   Alternatively, for uniform distribution across the active step:
   	au_i \sim \mathcal{U}(0, $\Delta T$)

3. **Virtual Timestamp Calculation**:
   t_{i} = T_{	ext{current}} + 	au_i
   where t_{i} is formatted as a continuous ISO-8601 timestamp or fractional simulation second.

4. **Chronological Placement Queue**:
   All batched actions (1-token actions and completed comment fallbacks) are collected into a memory buffer and sorted:
   $\mathcal{Q} = \text{sort}(\{(a_i, t_i)\}_{i=1}^N, \text{key}=t_i)$
   The simulation platform writes actions into SQLite sequentially according to this order.

---

## 4. Algorithmic Amplification & CIB Integration

- **CIB Bots (`CIBAgent`)**:
  - Continue using `NoOpModelBackend` with 0 GPU cost.
  - Bots generate actions deterministically or through local Thompson Sampling bandits (`cib_zoo/modifiers/bandit.py`).
  - Bot actions are assigned virtual timestamps following their configured `PulsedWaveModifier` duty cycles and inserted into the same chronological queue \mathcal{Q}.
- **RecSys Feedback**:
  - With sorted micro-timestamps, recsys recency scoring:
    	ext{date\_score} = \log\left(rac{271.8 - (t_{	ext{now}} - t_{	ext{created}})}{100}
ight)
    reflects genuine sub-step propagation dynamics rather than degenerate step-level ties.

---

## 5. Performance Comparison & ROI Analysis

| Metric | CAMEL-AI Baseline | JEV Mode | Improvement |
| :--- | :--- | :--- | :--- |
| **Decode Tokens / Agent** | 200 tokens (JSON schema) | 1 token (95% agents) / 50 tokens (5% agents) | **~98% reduction** |
| **KV Cache Hit Rate** | ~0% (isolated chat sessions) | 60% – 85% (prefix sharing) | **Significant memory saving** |
| **Step Latency (1k agents)**| ~120s – 300s (or OOM) | ~2s – 6s | **20x – 50x speedup** |
| **GPU Requirement** | Multi-GPU cluster required | Single workstation GPU (e.g. RTX 4090 / A100) | **Massive cost reduction** |
| **Simulation Realism** | Clustered discrete step stamps | Continuous Poisson temporal arrival | **Higher fidelity** |

---

## 6. Implementation Milestones

### Milestone 1: JEV Action Formatter & Logit Masking
- Implement `JEVPromptBuilder` formatting feed posts into 1-token query templates.
- Define logit-bias dictionary constraining logits to token IDs for `['L', 'R', 'C', 'S']`.

### Milestone 2: Batched Inference Runner
- Implement `BatchedInferenceEngine` wrapping vLLM / SGLang client APIs.
- Support batch submissions of all active organic agent prompts at step initiation.

### Milestone 3: Selective Comment Fallback Worker
- Filter outputs matching token `'C'`.
- Run secondary batched text-generation query for comments.

### Milestone 4: Virtual Micro-Time Queue
- Implement `MicroTimeScheduler` sampling 	au_i \sim 	ext{Exp}(\lambda_i).
- Implement `ChronologicalActionQueue` sorting actions before dispatching to OASIS `Channel`.

### Milestone 5: Verification & Benchmarking
- Add unit tests verifying 1-token output parsing and chronological ordering.
- Benchmark 1,000-agent step execution time vs. standard baseline.
