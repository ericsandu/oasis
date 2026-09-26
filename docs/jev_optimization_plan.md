# Architectural Specification & Team Dispatch Plan: JEV Mode Optimization for OASIS

**Target Framework**: OASIS Social Simulation Platform  
**Target Component**: Organic Agent Inference Engine, Belief Dynamics & Temporal Action Scheduling  
**Author**: Antigravity & Research Team  
**Status**: Ready for Implementation / Dispatch  

---

## 1. Executive Summary & Core Paradigm Shift

In large-scale social simulations, organic user decision-making represents 95%+ of total computational overhead. Under the default CAMEL-AI architecture, every organic agent generates 150–300 tokens of verbose JSON function-calling schema per step just to emit binary interactions (e.g. `like_post`, `repost`, `skip`). Autoregressive decoding is memory-bandwidth bound, capping single-GPU simulations at only 50–150 concurrent organic agents.

Furthermore, CAMEL-AI's chat history accumulation produces exponential context bloat (3,000–5,000 tokens per agent after 10 steps), which **destroys KV-cache reuse**, degrades persona adherence, and fails to model authentic psychological persuasion.

**JEV Mode** (Joint Evaluation Vectorization) fundamentally restructures social agent inference around four pillars:
1. **Inverted Prefix KV-Caching**: The post content serves as the static prefix and the user persona as the suffix, allowing vLLM/SGLang to compute post attention **once** and reuse it across all recipient agents.
2. **Item-Level Decoupled Parallelism (Zero Omnipotence)**: Agents evaluate their personalized feed items as independent concurrent micro-decisions, preserving strict information locality without global feed leakage.
3. **1-Token Constrained Logit Classification**: Collapses 150–300 decode steps into a **single GPU forward pass** for 95% of social interactions (`L` for Like, `R` for Repost, `C` for Comment, `S` for Skip).
4. **Three-Level Memory & Opinion Dynamics**: Replaces raw conversational history dumps with a lightweight **Dynamic Stance State** ($\text{Stance} \in [-1.0, +1.0]$) that tracks real mathematical persuasion curves under CIB exposure.

---

## 2. Bottleneck Analysis: The Flaws in Default OASIS / CAMEL-AI

### 2.1 The 200-Token JSON Decode Tax
Under `oasis/social_agent/agent.py`:
```python
env_prompt = await self.env.to_text_prompt()
user_msg = BaseMessage.make_user_message(content=f"Please perform social media actions: {env_prompt}")
response = await self.astep(user_msg)  # Generates 200+ tokens of JSON function calling
```
- In LLM serving, generation latency is governed by sequential decode steps:
  $$\text{Latency} = T_{\text{prefill}} + N_{\text{tokens}} \times T_{\text{decode}}$$
- 100 agents generating 200 tokens each require **200 sequential matrix-vector multiplications** on the GPU, taking 6–8 seconds per step.

### 2.2 Chat History Bloat & KV-Cache Thrashing
In CAMEL-AI, `self.memory` appends each step's full feed string and response.
- **Step 1**: Prompt = 350 tokens.
- **Step 5**: Prompt = 1,800 tokens.
- **Step 10**: Prompt = 3,500+ tokens.
- **The Result**: Because the prompt history changes continuously for every agent, **KV-cache hit rate is 0%**. The GPU spends massive compute re-prefilling thousands of irrelevant tokens of long-forgotten feeds, eventually causing Out-Of-Memory (OOM) crashes.

---

## 3. JEV Architectural Specification

```
                                      [ Gorse Recommendation System ]
                                                     │
                                                     ▼
                             Personalized Feeds (e.g. Agent 1 gets P42, P10; Agent 2 gets P42, P99)
                                                     │
                         ┌───────────────────────────┴───────────────────────────┐
                         ▼                                                       ▼
      [ Item Evaluation: Post 42 ]                            [ Item Evaluation: Post 10 ]
   Prefix: [POST 42: "Breaking AI..."]                     Prefix: [POST 10: "New open source..."]
   (Computed ONCE in vLLM Radix Tree)                      (Computed ONCE in vLLM Radix Tree)
             │                       │                                       │
             ▼                       ▼                                       ▼
  Suffix: [USER 1]        Suffix: [USER 2]                        Suffix: [USER 1]
  Bio + Stance Vector     Bio + Stance Vector                     Bio + Stance Vector
  Output: 'L' (Like)      Output: 'S' (Skip)                      Output: 'L' (Like)
  (1 Forward Pass)        (1 Forward Pass)                        (1 Forward Pass)
             │                       │                                       │
             └───────────────────────┼───────────────────────────────────────┘
                                     ▼
                      [ Conditional Comment Fallback ]
                      (Only if token == 'C' ──► ~5% branch)
                                     │
                                     ▼
                       [ Intra-Feed Action Resolver ]
                       Softmax confidence & agent budget filter
                                     │
                                     ▼
                      [ Virtual Micro-Time Scheduler ]
                       tau_i ~ Exp(lambda_i) ──► Continuous ISO timestamp
                                     │
                                     ▼
                       [ SQLite DB & Gorse Feedback ]
```

---

## 4. Key Architectural Mechanisms

### 4.1 Inverted Prefix Formatting (Maximizing KV Cache)

In standard autoregressive attention, tokens attend strictly to earlier tokens. If the prompt begins with the user profile, cache sharing between different users is impossible:
$$\text{Conventional (Bad): } [\underbrace{\text{User Bio}}_{\text{Unique per user}}, \underbrace{\text{Post Content}}_{\text{Shared}}] \implies \mathbf{0\% \text{ Cache Sharing}}$$

JEV **inverts the prompt order**:
$$\text{JEV Inverted (Optimal): } [\underbrace{\text{Post Content}}_{\text{Shared Prefix (100 tokens)}}, \underbrace{\text{User Profile + Stance}}_{\text{Personalized Suffix (30 tokens)}}] \implies \mathbf{70–90\% \text{ Cache Hit Rate}}$$

#### Prompt Template:
```text
[POST ID: {post_id}] Author: @{author} | Topic: #{topic}
Content: "{content}"

[OBSERVER]: @{user_name} | Traits: {mbti}, {country} | Bio: {bio}
[STANCE]: #{topic}: {stance_label} ({stance_score:+.2f})
[TASK]: Choose single reaction: [L]ike, [R]epost, [C]omment, [S]kip.
Action:
```
- When 80 agents receive the same viral post (Post 42) in their feed, vLLM's RadixAttention computes the attention keys and values for Post 42 **exactly once**.
- The 80 agent queries are executed as parallel suffixes against that single cached KV tensor.

---

### 4.2 Resolving the "Omnipotence Problem" via Item-Level Parallelism

- **The Problem**: Concatenating all platform posts into one shared prefix would create an **omnipotence violation**—agents would "know" about posts that were never served in their personal recommendation feed.
- **The JEV Solution**: Feed evaluation is **decoupled into per-item pairs**:
  $$\text{Feed}_u = \{p_{1}, p_{2}, \dots, p_{K}\} \implies \text{Batch evaluate } \{(p_1, u), (p_2, u), \dots, (p_K, u)\}$$
- **Information Locality**: Agent $u$ is evaluated **strictly and only** on the $K$ items returned by its personal Gorse recommendation query.
- **Parallel Execution**: All $K$ items for an agent are dispatched **concurrently within the same vLLM inference batch**. The agent evaluates its entire feed simultaneously on the GPU.

---

### 4.3 Intra-Feed Action Resolution & Budget Filtering

When an agent evaluates $K$ posts in parallel, it produces $K$ action tokens (e.g. `['L', 'S', 'L']`).
1. **Unconstrained Actions**: If the agent's budget allows, both likes are committed.
2. **Constrained Actions**: If the agent has a budget of 1 action per step:
   - Extract the softmax logit probability for each action token:
     $$P(L \mid p_k) = \frac{\exp(z_L^{(k)})}{\sum_{a \in \{L, R, C, S\}} \exp(z_a^{(k)})}$$
   - Select the interaction with the highest confidence score $\arg\max_k P(L \mid p_k)$.

---

### 4.4 The 3-Level Memory & Organic Influence Model

To study CIB propagation and disinformation, agents must exhibit **genuine psychological susceptibility and opinion drift** without prompt bloat:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: PLATFORM ALGORITHMIC MEMORY                                        │
│ • Handled by Gorse SQLite feedback tables (like, read, comment).            │
│ • When Agent A likes a bot payload, Gorse updates Agent A's CF embedding,   │
│   naturally pulling more related content into future feeds.                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 2: COMPACT EPISODIC ACTION LOG                                        │
│ • Replaces raw chat dumps with a lightweight summary of actual actions:     │
│   "Recent: Liked P12 (#tech), Liked P42 (#target), Commented on P42"        │
│ • Fixed footprint: ~30-40 tokens. Older events roll off automatically.      │
├─────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 3: DYNAMIC BELIEF STATE (OPINION DYNAMICS)                            │
│ • Each agent maintains a topic stance scalar: s_i(topic) in [-1.0, +1.0]     │
│ • Stance Update Function (Bounded Confidence / Bounded Persuasion):         │
│   When exposed to post with stance s_p and high peer engagement (likes):    │
│       delta_s = alpha * (s_p - s_i) * sigmoid(peer_engagement)              │
│       s_i(t+1) = clip(s_i(t) + delta_s, -1.0, 1.0)                          │
│ • Stance is mapped to descriptive text injected into the prompt suffix:     │
│   s > 0.6  -> "Strongly supportive of #topic"                               │
│   |s| <= 0.2 -> "Neutral regarding #topic"                                  │
│   s < -0.6 -> "Hostile / skeptical of #topic"                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Scientific Impact**: Researchers can directly measure and plot mathematical persuasion curves $\Delta \text{Stance}(t)$ across communities, proving whether CIB campaigns achieved cognitive conversion vs. mere exposure.

---

### 4.5 Virtual Micro-Time Poisson Scheduling

In standard OASIS, all step actions receive an identical discrete timestamp string (`str(step)`), causing unphysical step-level synchronization waves.

JEV introduces continuous Poisson process scheduling:
1. **Activity Arrival Sampling**:
   $$\tau_i \sim \text{Exponential}(\lambda_i), \quad \tau_i \in [0, \Delta T]$$
   where $\lambda_i$ is derived from `UserInfo.activity_level_frequency` and $\Delta T$ is the step duration (e.g. 900 seconds).
2. **Continuous Timestamp Generation**:
   $$t_{\text{action}} = T_{\text{step\_start}} + \tau_i \quad \text{(Formatted as ISO-8601 UTC)}$$
3. **Chronological Placement Queue**:
   All batched decisions (1-token actions and completed comment texts) are collected into a priority queue sorted by $t_{\text{action}}$:
   $$\mathcal{Q} = \text{sort}\Big(\big\{(a_i, t_{\text{action}}^{(i)})\big\}, \text{ key}=t_{\text{action}}\Big)$$
4. Actions are committed to SQLite sequentially according to this order, ensuring Gorse recency scoring (`date_score`) reflects true causal propagation order.

---

## 5. Performance & Resource Comparison

| Metric | Current CAMEL-AI Pipeline | JEV Mode Pipeline |
| :--- | :--- | :--- |
| **Decode Tokens / Agent** | $150–300$ tokens (verbose JSON) | **1 token** ($95\%$ agents) / $40$ tokens ($5\%$ comments) |
| **Decode GPU Passes** | $150–300$ sequential matrix steps | **1 matrix step** |
| **KV-Cache Hit Rate** | $0\%$ (Prompt differs every step) | **$70\%–90\%$** (Inverted prefix reuse) |
| **Prompt Size at Step 15**| $3{,}000–5{,}000$ tokens (chat bloat) | **$120–150$ tokens** (Constant $O(1)$) |
| **Step Latency (150 agents)**| $\approx 6.0–8.5\text{ s}$ | $\approx 0.3–0.5\text{ s}$ ($15\times$ speedup) |
| **Population Capacity** | $\approx 150$ agents per A100 | **$1{,}500–3{,}000+$ agents per A100** |
| **Opinion Tracking** | Implicit / accidental | **Explicit, mathematical $\Delta \text{Stance}(t)$** |
| **Timestamps** | Coarse discrete integers | Continuous Poisson micro-timestamps |

---

## 6. Team Dispatch & Implementation Roadmap

```
                    ┌──────────────────────────────────────────────┐
                    │      JEV TEAM DISPATCH ROADMAP (5 PHASES)    │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴───────┬───────────────────┐
         ▼                   ▼                     ▼                   ▼
    [Track 1]           [Track 2]             [Track 3]           [Track 4]
 JEV Prompt Builder   Dynamic Stance &      vLLM Batched        Micro-Time
 (Inverted Prefix)     Memory Engine       Logit Classifier      Scheduler
         │                   │                     │                   │
         └───────────────────┴─────────────┬───────┴───────────────────┘
                                           ▼
                                       [Track 5]
                                 OasisEnv Integration
                                 & Benchmark Test Suite
```

### Track 1: JEV Inverted Prompt Engine
- **File**: `oasis/social_agent/jev_prompt_builder.py`
- **Deliverables**:
  - `build_inverted_prefix(post)`: Formats standardized post header (target for KV caching).
  - `build_agent_suffix(agent, stance)`: Formats compact agent profile and dynamic stance.
  - Token serialization verification ensuring exact byte-level match across shared post prefixes.

### Track 2: Dynamic Stance & 3-Level Memory
- **File**: `oasis/social_agent/belief_state.py`
- **Deliverables**:
  - `BeliefState`: Dataclass tracking topic stances $s_i \in [-1.0, 1.0]$ and action history.
  - `update_stance(post, engagement)`: Implements bounded persuasion update function.
  - Serialization into prompt suffix label (`"Supportive"`, `"Neutral"`, `"Skeptical"`).

### Track 3: vLLM Batched Logit Classifier
- **File**: `oasis/inference/jev_classifier.py`
- **Deliverables**:
  - `classify_actions_batch(pairs)`: Submits prompt pairs to vLLM with `max_tokens=1` and `logit_bias={'L': 100, 'R': 100, 'C': 100, 'S': 100}`.
  - `generate_comments_batch(comment_requests)`: Secondary worker for `'C'` selections ($<10\%$ of actions).
  - Intra-feed resolver selecting highest confidence action under budget constraints.

### Track 4: Micro-Time Poisson Scheduler
- **File**: `oasis/clock/micro_time_scheduler.py`
- **Deliverables**:
  - `sample_arrival_times(agents, step_duration)`: Generates $\tau_i \sim \text{Exp}(\lambda_i)$.
  - `ChronologicalActionQueue`: Priority queue sorting actions by ISO-8601 timestamp before platform dispatch.

### Track 5: OasisEnv Integration & Benchmarking
- **File**: `oasis/environment/jev_env.py` and `cib_zoo/tests/test_jev_engine.py`
- **Deliverables**:
  - New execution mode in `OasisEnv.step()` toggling between CAMEL-AI and JEV mode.
  - Hermetic unit tests validating 1-token output parsing, cache hit rates, stance drift, and chronological queue commits.
  - Benchmark comparing 150-agent and 1,000-agent execution latency.

---

## 7. Team Layout & Specialized Skills Matrix

To ensure clean separation of concerns, high code quality, and hermetic verification, the implementation is partitioned across 5 specialized subagent personas:

| Role | Domain / Focus | Key Skills | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Lead Systems Architect** | Architectural decoupling, schema design, public APIs | `architecture-patterns`, `ai-agents-architect` | Oversees Track 1 & 5 integration; ensures zero global feed leakage and backward compatibility with `OasisEnv`. |
| **Inference & Batching Engineer** | vLLM/SGLang integration, prompt KV prefixing, 1-token logit classifier | `python-pro`, `parallel-agents` | Implements Tracks 1 & 3; formats exact byte-matching inverted prefixes, constrained logits, and fallback comment workers. |
| **Cognitive Dynamics Specialist** | 3-Level memory, opinion dynamics, stance modeling | `ai-agents-architect`, `python-pro` | Implements Track 2; develops `BeliefState`, bounded confidence math $\Delta s_i$, and Gorse algorithmic memory feedback. |
| **Concurrency & Temporal Engineer** | Micro-time clock, Poisson scheduling, priority queue | `python-pro`, `clean-code` | Implements Track 4; continuous arrival generation $\tau_i \sim \text{Exp}(\lambda_i)$ and chronological SQLite commit ordering. |
| **QA & Hermetic Verification Auditor** | Unit testing, test fixtures, benchmark suite | `quinn`, `pytest-skill`, `python-testing-patterns` | Implements Track 5 test suite; develops hermetic mock LLM/logit backends, enforces 100% pytest pass with max 4 CPU threads. |

---

## 8. Concrete Interface Specifications

### 8.1 Track 1: `oasis/social_agent/jev_prompt_builder.py`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PostPrefixData:
    post_id: int
    author_name: str
    topic: str
    content: str

@dataclass(frozen=True)
class AgentSuffixData:
    user_id: int
    user_name: str
    mbti: str
    country: str
    bio: str
    stance_label: str
    stance_score: float
    recent_actions: str = ""

class JEVPromptBuilder:
    @staticmethod
    def build_post_prefix(post: PostPrefixData) -> str:
        """Returns standard static prefix for KV caching."""
        ...

    @staticmethod
    def build_agent_suffix(agent: AgentSuffixData) -> str:
        """Returns agent persona suffix and reaction prompt."""
        ...

    @classmethod
    def assemble_eval_prompt(cls, post: PostPrefixData, agent: AgentSuffixData) -> str:
        """Concatenates prefix + suffix into single evaluation prompt."""
        ...
```

### 8.2 Track 2: `oasis/social_agent/belief_state.py`
```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class ActionLogItem:
    action_type: str
    post_id: int
    topic: str
    timestamp_iso: str

@dataclass
class BeliefState:
    user_id: int
    # topic -> stance scalar in [-1.0, +1.0]
    stances: Dict[str, float] = field(default_factory=dict)
    # Compact episodic log: max N items
    recent_actions: List[ActionLogItem] = field(default_factory=list)
    max_history_items: int = 5
    persuasion_rate_alpha: float = 0.15

    def get_stance(self, topic: str) -> float: ...
    def get_stance_label(self, topic: str) -> str: ...
    def update_stance(self, topic: str, post_stance: float, peer_engagement: int) -> float: ...
    def record_action(self, action_type: str, post_id: int, topic: str, timestamp_iso: str) -> None: ...
    def get_episodic_summary(self) -> str: ...
```

### 8.3 Track 3: `oasis/inference/jev_classifier.py`
```python
from typing import Protocol, List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class EvalItem:
    user_id: int
    post_id: int
    topic: str
    full_prompt: str

@dataclass
class ClassificationResult:
    user_id: int
    post_id: int
    action_char: str  # 'L', 'R', 'C', 'S'
    confidence: float
    logits: Dict[str, float]

class JEVClassifierClient(Protocol):
    async def classify_batch(self, items: List[EvalItem]) -> List[ClassificationResult]: ...
    async def generate_comment(self, user_prompt: str, post_content: str) -> str: ...
```

### 8.4 Track 4: `oasis/clock/micro_time_scheduler.py`
```python
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from datetime import datetime

@dataclass(order=True)
class ScheduledAction:
    sort_timestamp: float
    iso_timestamp: str = field(compare=False)
    user_id: int = field(compare=False)
    action_dict: Dict[str, Any] = field(compare=False)

class MicroTimeScheduler:
    def __init__(self, step_duration_seconds: float = 900.0, default_lambda: float = 1.0): ...
    def sample_offset(self, activity_frequency: float) -> float: ...
    def schedule_actions(
        self,
        step_index: int,
        base_time: datetime,
        agent_actions: List[Tuple[int, Dict[str, Any], float]]
    ) -> List[ScheduledAction]: ...
```

---

## 9. Verification, Resource Limits & Acceptance Gates

### Resource Guardrails:
1. **Thread Caps**: All pytest runs and Python executions must enforce `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` to strictly respect cluster and local host guidelines.
2. **Hermeticity**: Test suites must mock LLM/vLLM endpoints completely (`MockJEVClassifierClient`), generating deterministic logit distributions without requiring GPU hardware or live network APIs.
3. **Branch Isolation**: All implementation files and unit tests must be committed exclusively to branch `feat/jev-optimization`. No commits on `feat/cib-modular-engine` or `main`.

