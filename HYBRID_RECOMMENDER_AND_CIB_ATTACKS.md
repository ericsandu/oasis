# 🏛️ Architecture: Twitter-Like Hybrid Recommender & CIB Attack Catalog

This document details the architectural specification of the **60/40 Hybrid Recommender System** designed for OASIS, along with a theoretical and empirical taxonomy of **Coordinated Inauthentic Behavior (CIB) attack vectors** and their predicted outcomes.

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

## Part 2: Taxonomy of CIB Attacks & Likely Outcomes

Below is the theoretical analysis of how distinct botnet strategies will perform against this hybrid defense.

```
                                  [ CIB Attack Taxonomy ]
                                             │
      ┌──────────────────────┬───────────────┴──────────────┬──────────────────────┐
      ▼                      ▼                              ▼                      ▼
1. Blunt-Force Farm    2. Recency Exploit             3. Semantic Smuggler   4. Organic Bridger
(Volume Only)          (Timing Only)                  (Content Optimization) (Graph Manipulation)
[Outcome: QUARANTINED] [Outcome: HEAVILY DILUTED]     [Outcome: PARTIAL]     [Outcome: HIGH BREACH]
```

---

### Attack 1: The Naive Blunt-Force Like Farm
* **Mechanism:** 20 bots immediately spam-like an out-of-domain `#tech` payload to game the algorithm into serving it to `#sports` users.
* **Math Profile:**
  * $S_{\text{TwinBERT}} = 0.0$ (Zero semantic relevance to sports)
  * $S_{\text{CF}} = 0.0$ (Zero co-interaction overlap with sports users)
  * $S_{\text{Pop}} = 1.0$ (Maximized by 20 bot likes $\rightarrow$ captures 18%)
  * $S_{\text{Rec}} = 1.0$ (Fresh post $\rightarrow$ captures 12%)
  * **Total Score:** $0.0 + 0.0 + 0.18 + 0.12 = \mathbf{0.30}$
* **Predicted Outcome:** **QUARANTINED (0% to 5% Reach)**.
* **Why:** In-domain organic sports posts easily achieve $> 0.60$ (40% semantic + 20-30% CF). A score of 0.30 is mathematically insufficient to break into the Top-10 feed.

---

### Attack 2: The Chronological / Cold-Start Exploit
* **Mechanism:** An attacker drops a payload during a lull in organic publishing, attempting to exploit Gorse's recency fallback.
* **Math Profile:**
  * $S_{\text{Rec}} = 1.0$ (Captures 12%)
  * $S_{\text{Pop}} = 0.0$
  * $S_{\text{TwinBERT}} = 0.0$
  * $S_{\text{CF}} = 0.0$
  * **Total Score:** $\mathbf{0.12}$
* **Predicted Outcome:** **FAILED / HEAVILY SUPPRESSED (< 1% Reach)**.
* **Why:** Under our previous uncalibrated Gorse configuration (`latest = 0.3` without TwinBERT), this achieved a 100% breach. Under the 60/40 hybrid, a 12% freshness score cannot overcome the 40% semantic penalty.

---

### Attack 3: The Semantic Smuggler (LLM Adversarial Rewriting)
* **Mechanism:** An LLM-powered bot rewrites the `#tech` propaganda payload to embed `#sports` vocabulary, metaphors, and hashtags, seeking to maximize the TwinBERT score while preserving the underlying political/tech payload.
* **Math Profile:**
  * $S_{\text{TwinBERT}} \approx 0.85$ (Captures $0.40 \times 0.85 = \mathbf{0.34}$)
  * $S_{\text{Pop}} \approx 0.50$ (Captures $0.18 \times 0.50 = \mathbf{0.09}$)
  * $S_{\text{Rec}} = 1.0$ (Captures $\mathbf{0.12}$)
  * $S_{\text{CF}} = 0.0$ (Bots have no organic history in sports)
  * **Total Score:** $0.34 + 0.09 + 0.12 + 0.0 = \mathbf{0.55}$
* **Predicted Outcome:** **PARTIAL PENETRATION (20% to 45% Reach)**.
* **Why:** By neutralizing the semantic firewall, the payload competes directly on the bubble. It displaces mediocre organic posts, achieving moderate exposure among fringe users who lack dense collaborative filtering ties.

---

### Attack 4: The Organic Bridger / Sleeper Cell (CF Matrix Corruption)
* **Mechanism:** Bots spend a "burn-in" phase acting like authentic `#sports` fans (liking sports content, following sports users) to establish strong latent vectors in Gorse's Matrix Factorization. Once trusted, they pivot and like the `#tech` payload.
* **Math Profile:**
  * $S_{\text{CF}} \approx 0.90$ (Gorse treats bots as sports community members $\rightarrow$ captures $0.30 \times 0.90 = \mathbf{0.27}$)
  * $S_{\text{Pop}} = 1.0$ (Captures $\mathbf{0.18}$)
  * $S_{\text{Rec}} = 1.0$ (Captures $\mathbf{0.12}$)
  * $S_{\text{TwinBERT}} = 0.0$ (Text remains raw `#tech`)
  * **Total Score:** $0.27 + 0.18 + 0.12 + 0.0 = \mathbf{0.57}$
* **Predicted Outcome:** **HIGH PENETRATION (40% to 65% Reach)**.
* **Why:** Collaborative filtering is powerful. By corrupting the interaction matrix, Gorse logically infers that `#sports` users want this post because their "peers" (the sleeper bots) engaged with it.

---

### Attack 5: The 2-Hop Social Proof Hijack (GraphJet Exploit)
* **Mechanism:** Bots target a peripheral organic `#sports` micro-influencer with high follower overlap, baiting them into liking or retweeting the payload.
* **Math Profile:**
  * Candidate Pool: Enters **Candidate Pool A (In-Network)** via the 2-hop SQL query (`friend_likes >= 1`).
  * $w_{\text{network}} = +0.25$
  * $S_{\text{Pop}} = 0.18$
  * $S_{\text{Rec}} = 0.12$
  * **Total Score:** $0.25 + 0.18 + 0.12 = \mathbf{0.55}$
* **Predicted Outcome:** **TARGETED LOCAL BREACH (70% to 90% reach within that influencer's sub-cluster)**.
* **Why:** In-network social proof acts as a local multiplier. Even with low semantics, the social endorsement allows the post to penetrate the followers of the compromised host.

---

### Attack 6: The "Trojan Horse" (Dual-Vector: Semantic Smuggling + Organic Bridging)
* **Mechanism:** The apex attack. Sleeper bots establish authentic sports history (capturing CF), coordinated bots provide engagement velocity (capturing Popularity), and the payload is rewritten via LLM to adopt sports vernacular (capturing TwinBERT).
* **Math Profile:**
  * $S_{\text{TwinBERT}} = 0.85 \rightarrow \mathbf{0.34}$
  * $S_{\text{CF}} = 0.90 \rightarrow \mathbf{0.27}$
  * $S_{\text{Pop}} = 1.0 \rightarrow \mathbf{0.18}$
  * $S_{\text{Rec}} = 1.0 \rightarrow \mathbf{0.12}$
  * **Total Score:** $0.34 + 0.27 + 0.18 + 0.12 = \mathbf{0.91}$
* **Predicted Outcome:** **TOTAL BREACH (> 90% Saturation)**.
* **Why:** The attack satisfies every objective of the recommendation engine simultaneously. It is mathematically indistinguishable from authentic viral crossover content.

---

## Part 3: Summary Comparison Matrix

| Attack Strategy | Target Vector | Required Compute/Friction | Predicted Reach in Target Cluster | Algorithmic Defeated |
| :--- | :--- | :--- | :---: | :--- |
| **1. Blunt-Force Farm** | Raw Popularity | Low (simple bot spam) | **0% – 5%** | Neutralized by TwinBERT & CF |
| **2. Recency Timing** | Cold-Start Fallback | Minimal (time of post) | **< 1%** | Neutralized by TwinBERT |
| **3. Semantic Smuggler** | NLP TwinBERT Tower | Medium (LLM generation) | **20% – 45%** | Contained by Gorse CF |
| **4. Organic Bridger** | Gorse CF Matrix | High (burn-in period) | **40% – 65%** | Breaches Gorse; dampened by TwinBERT |
| **5. 2-Hop Social Hijack** | SQLite In-Network | High (social engineering) | **70% – 90% (Local)** | Exploits Social Proof bypass |
| **6. Trojan Horse (Dual)** | All Objectives | Very High (LLM + Sleeper) | **> 90% (Systemic)** | **Bypasses Entire Hybrid Stack** |

This framework establishes the empirical baseline for our Phase 2 experiments on the FEP cluster.
