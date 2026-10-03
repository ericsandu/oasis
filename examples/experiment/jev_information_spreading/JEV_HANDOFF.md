# JEV Engine — Comprehensive Handoff (branch `feat/jev-engine-core`)

> Audience: another AI agent (or engineer) picking up this work cold.
> Scope: everything done on this branch about the **JEV action-decision mechanism**
> and the OASIS "Information Spreading in X" (paper §F.2.2) parity study.
> Last updated: 2026-10-03. Repo: `ericsandu/oasis` (our fork = **origin**;
> `camel-ai/oasis` = upstream). Current HEAD: `5b1ee97`.

---

## 0. TL;DR for the impatient

- **JEV** = "Joint Evaluation Vectorization", **our own engine** (not an external
  cited method; do not confuse with TypeSafe's identically-named *Jev* product —
  pure name collision). It accelerates OASIS by replacing per-agent generative
  tool-calling with a **per-post single-token action classification** under a
  logit bias, enabling KV-cache prefix reuse.
- The experiment: reproduce the OASIS paper's **Information Spreading** cascade on
  `False_Business_0`, comparing **classic** OASIS (`use_jev:false`) vs **JEV**
  (`use_jev:true`), and both vs **real-world** cascade data.
- **Central finding of this session:** JEV's single-token classification
  **structurally under-produces Repost** (the cascade-driving action) — it was
  producing ~0–16 reposts vs classic's ~107. We ruled out, with controlled runs,
  every "cheap" explanation (prompt wording, token-id map, decode temperature,
  feed size, audience context) and showed the true axis is
  **single-token classification ≠ generating the action**. Switching JEV to
  **native generation** of the action reached **near-parity (105 reposts,
  NRMSE 0.08 vs classic, matches real data)** — at the cost of JEV's speedup
  (2.3× → 1.1×).
- **Open decision:** keep the fast single-token engine (and make it *trustworthy*
  via an AnyJev-L0 permutation-debias layer, training-free), vs accept generation
  as the faithful engine. Both are legitimate; it's a speed/fidelity fork.

---

## 1. The experiment & how to run it

### 1.1 What is being measured
OASIS paper §F.2.2 "Information Spreading in X": seed a (mis)information post,
run 50 timesteps, measure the **cascade** via three curves over time:
- **scale** — total nodes reached
- **depth** — longest repost chain (multi-hop)
- **max_breadth** — widest single level

Topic file: `False_Business_0`. Real-world ground-truth pkl is shipped; the paper
reports engines land ~0.30 NRMSE vs reality.

### 1.2 The two engines (one switch)
`simulation.use_jev` in the config:
- `false` → **classic** OASIS: CAMEL `ChatAgent` per agent, native tool-calling
  (`<tool_call>{...}</tool_call>`) on `/v1/chat/completions`, parsed by the
  hermes tool-call parser.
- `true` → **JEV**: `oasis/environment/jev_env.py` `step_jev()` pipeline —
  per-(agent,post) single-token classification on `/v1/completions` with a logit
  bias over action letters, Poisson micro-time scheduling, belief updates.

### 1.3 HPC facts (read before running anything)
- **Login:** `ssh teodor_daniel.milea@fep.grid.pub.ro -F /dev/null -i ~/.ssh/id_ed25519`
  - NOTE: the FIRST ssh to a dotted hostname in a fresh gateway process may be
    refused with a "sandbox-escape-ssh-self / DNS classification PENDING" message —
    **just retry the identical command**, it succeeds once DNS resolves off-loop.
- **Repo on HPC:** `~/eric_sandu/oasis` (checkout of this branch).
- **Container (SIF):** `examples/experiment/jev_information_spreading/jev_oasis.sif`
  (6.9 GB, the sbatch's default `$SIF`). The filesystem **inside** the SIF is
  **read-only at runtime** — `pip install` into the venv does NOT persist. Any
  new Python dep must be added to `jev_oasis.def` and the SIF rebuilt
  (`srun --partition=haswell --cpus-per-task=32 --mem=32GB apptainer build --force`,
  ~8–10 min, dominated by the vllm/torch pip resolve). **BUT** `oasis/` is
  bind-mounted at `/app`, so pure-Python edits to `oasis/` need **NO rebuild**.
- **vLLM:** served inside the SIF, Llama-3-8B-Instruct at `/models`, port derived
  per job. `--max-model-len 8192`, `--gpu-memory-utilization 0.90`,
  `VLLM_USE_FLASHINFER_SAMPLER=0` (required or EngineCore dies JIT-compiling
  flashinfer with no nvcc). vLLM version is **0.23.0**.
- **GPU partition:** `dgxa100`, single A100. **Do NOT submit two 0.90-VRAM GPU
  jobs at once** — they collide/OOM (we lost job 267858 this way). Run one at a
  time or chain with `--dependency=afterany:<jobid>`.
- **SLURM "FAILED" ≠ sim failed.** In almost every run the *simulation* finished
  and printed PERFORMANCE/PARITY; the FAILED state is the post-hoc analysis
  script's exit code (a `ModuleNotFoundError` / `pdb` in an except). Always
  `grep` the `.out` for the PERFORMANCE/PARITY block before concluding failure.

### 1.4 The compare harness
`examples/experiment/jev_information_spreading/run_compare.sbatch`:
starts vLLM, runs classic leg, runs JEV leg, prints PERFORMANCE (wall-clock +
speedup) and PARITY (action counts, cascade curves, NRMSE vs classic AND vs real
via `compare_jev_parity.py <db1> <db2> <topic>`).

**Experiment toggles are env vars threaded into the JEV leg** (all default off, so
the baseline is unchanged):
```
OASIS_JEV_FEED_MODE=1       # whole-feed guided-JSON instead of per-post
OASIS_JEV_VERBATIM=1        # exact upstream OASIS prompt (system+user+env JSON)
OASIS_JEV_COMPETITIVE=1     # scarcity framing + cross-feed full-logits selection
OASIS_JEV_GUIDED_CHOICE=1   # vLLM structured_outputs CHOICE masking (not logit-bias)
OASIS_JEV_GENERATIVE=1      # native generation of {"action":"..."} per post
OASIS_JEV_INSTRUCT_FRAME=1  # wrap raw completion prompt in Llama-3 instruct template
OASIS_JEV_L0=1              # AnyJev L0 debias (per-letter batch-mean, prior='none') — VALIDATED
OASIS_JEV_L0_GROUP=1        # L0 mean per-user instead of whole-batch
OASIS_JEV_CONF_DUMP=<path>  # per-post P(like)/P(repost) CSV dump
OASIS_JEV_CLASSIFY_TEMP=1.0 # classifier sampling temperature (default now 1.0)
OASIS_CLASSIC_TOOL_CHOICE=auto  # classic leg tool_choice (auto = paper-faithful)
```
Launch example:
```
sbatch --export=ALL,OASIS_JEV_GENERATIVE=1 \
  examples/experiment/jev_information_spreading/run_compare.sbatch
```
Single-leg k=1 test harness: `run_classic_k1.sbatch` (classic only, one GPU).

---

## 2. Architecture of the JEV engine (where the code lives)

| File | Role |
|---|---|
| `oasis/environment/jev_env.py` | `OasisEnv` + `JEVExecutionConfig` + `step_jev()` — the whole JEV simulation loop (feed build → classify → budget → schedule → dispatch). All experiment-mode branches live here (Stage d/e). |
| `oasis/inference/jev_classifier.py` | `VLLMJEVClassifierClient` (the classifier), `ClassificationResult`, `resolve_intra_feed_budget`, `resolve_competitive_full_logits`, `dump_action_confidence_gap`, `select_best_action`, `MockJEVClassifierClient`, `DEFAULT_ACTION_TOKEN_MAP`, `LLAMA3_INSTRUCT_FRAME`. |
| `oasis/social_agent/jev_prompt_builder.py` | `JEVPromptBuilder` — assembles the cacheable prompt: `build_task_instruction` (root-cached), `build_post_prefix` (per-post, cross-agent shared), `build_agent_suffix` (per-agent), `assemble_eval_prompt`, `build_feed_prompt`, `build_verbatim_prompt`. `ACTION_DESCRIPTIONS` = single source of truth for action char↔tool-name↔description. |
| `oasis/clock/micro_time_scheduler.py` | Poisson micro-time arrival scheduling for continuous within-step action timing. |
| `oasis/social_agent/belief_state.py` | Bounded-confidence stance dynamics (disabled in align runs). |
| `examples/.../twitter_simulation_large.py` | The driver: reads config, builds `logit_bias`/`token_id_map`/`guided_choice`/`generative`/`instruct_frame` from env, constructs `JEVExecutionConfig`, runs the step loop with sim-only timing. |
| `examples/.../align_with_real_world/{classic,jev,classic_k1}_align.yaml` | Experiment configs. |
| `.../compare_jev_parity.py` | Post-hoc: action counts + cascade curves + NRMSE. |

### 2.1 The KV-cache design (JEV's reason to exist)
Prompt layout for maximal RadixAttention prefix reuse:
```
[task instruction]  run-constant  -> caches once at the trie ROOT
[post prefix]       per-post       -> cached per post, shared across ALL agents
[agent suffix]      per-agent      -> the only varying tail
```
This is why JEV is fast: 1 post × N agents = 1 cached post-prefix + N cheap
suffixes. **Any change that makes the bulk of the prompt per-agent (e.g. whole-feed
mode) collapses this win.** Keep the layout invariant when editing.

### 2.2 The classifier mechanism (as currently built)
`/v1/completions`, `max_tokens=1`, `logprobs=5`, `temperature` (now default 1.0),
plus a `+50` `logit_bias` on the action-letter token ids from
`DEFAULT_ACTION_TOKEN_MAP`. The generated token is parsed to an action char; a
softmax over the top-5 action logprobs gives `confidence` + `logits`.
`resolve_intra_feed_budget` then enforces `max_actions_per_agent=1` by keeping the
highest-confidence non-skip action per agent.

---

## 3. The investigation — chronological, with evidence

Each row is a controlled run; "repost" is on `False_Business_0`, classic ≈ 107.

| # | Change tested | repost | scale | NRMSE vs classic | verdict |
|---|---|---|---|---|---|
| baseline | logit-bias, greedy (temp 0) | ~0 | 1 | 0.76 | 0-repost collapse |
| temp fix | classifier temp 0 → 1.0 (match classic's vLLM-default sampling) | 4 | 5 | — | unfroze argmax, didn't fix |
| audience | add "I have N followers" broadcaster framing to suffix | 1 | 2 | — | **worse**; spiked Follow 20→49 |
| whole-feed | one guided-JSON call/agent over full feed | **83** | 82 | **0.29** | big jump — but it GENERATES |
| verbatim per-post | exact upstream OASIS prompt, per-post | 24 | 25 | 0.67 | richer prompt helps a little |
| verbatim whole-feed | exact prompt + whole feed | 0 | — | — | 8192 ctx OVERFLOW (both engines) |
| k=1 CLASSIC | classic feed forced to ONE post | **109** | 107 | — | **falsifies feed/selection hypothesis** |
| guided-choice | vLLM `structured_outputs:{choice}` masking (correct tokens, no bias) | **16** | 17 | 0.73 | token-map fix helped 4→16, not enough |
| **generative** | **native `{"action":"repost"}` generation per post** | **105** | **104** | **0.08** | **near-parity; matches real data** |
| instruct-frame | guided-choice + Llama-3 chat template on completions | 11 | 12 | 0.67 | flipped collapse Like→**Follow 254** |
| **guided-choice + AnyJev L0** | per-letter batch-mean debias (prior='none') | **86** | **87** | **0.242** | **de-collapses; NRMSE-vs-real 0.325 ≈ paper's ~0.30; speedup held 1.95×** |

L0 result detail (commit `7f7673c`, job 267874): action mix repost 86 / like 79 /
follow 69 / do_nothing 0 — a realistic three-way SPREAD instead of a single-action
collapse. **This is the validated fast-path engine.** NRMSE 0.73→0.24 vs classic,
0.325 vs real (paper-level), at 1.95× speedup. KNOWN SIDE EFFECT: L0 **over-flattens**
a genuinely skewed marginal — classic's "repost-dominant, little else" became an even
split, so repost dropped 107→86 and like/follow/do_nothing shifted up (do_nothing
0 → nearly every agent now acts every step; follow(table) 94→154 = 69 sim-created
follows vs classic's 9, on the SAME 85 pre-seeded edges). The AnyJev study's fix is
`prior_strength=0.75` (subtract 0.75× the profile, not 1.0×) to keep repost's
legitimate majority — a one-line knob we have not yet added. Generative remains the
max-fidelity mode (0.08); L0 is the max-*speed-with-acceptable-fidelity* mode.
1. **Not decode temperature.** Temp 1.0 only moved reposts 0→4.
2. **Not audience/broadcaster context.** Adding it dropped reposts and spiked Follow.
3. **Not feed size / selection.** k=1 classic (one isolated post) reposts 109 —
   the SAME isolation JEV has — so "classic reposts because it selects among many
   posts" is **false**. This was the pivotal experiment.
4. **Not the token-id map alone.** The old `DEFAULT_ACTION_TOKEN_MAP` is polluted
   (boosts junk subwords `'de'`,`'ata'`,`'em'`; cross-action collisions e.g. token
   82 in both R and S — verified by decoding against the Llama-3 tokenizer). The
   `[0]` ids ARE correct (L=43,R=49,F=37,S=50). Fixing it via guided-choice helped
   4→16 but did not reach parity.
5. **It IS classification-vs-generation.** Native generation reached 105. The
   whole-feed run worked (83) for the same reason — it *generates* via guided-JSON,
   not because it sees many posts.
6. **Single-token classification is BRITTLE / frame-dependent.** The instruct-frame
   run swung the collapse from all-Like to **all-Follow (254)** on a framing change
   alone — a different degenerate attractor, never parity. A single masked
   next-token is a lossy, surface-sensitive projection of the model's computation,
   not the action decision.

### 3.2 The confidence dump caveat (don't repeat this mistake)
`dump_action_confidence_gap` currently logs ONLY `p_like` and `p_repost` (not
`p_follow`/`p_skip`). In the instruct-frame run the dump showed "P(R)>P(L) on 100%
of items" which LOOKED like Repost-favoring — but the executed actions were 254
Follow, because the unlogged **P(F) dominated**. **Fix the dump to log all four
action probabilities** before trusting it. (The dump is wired to fire whenever
`OASIS_JEV_CONF_DUMP` is set, in any per-post mode — commit `d2c90fa`.)

---

## 4. Literature grounding (done this session)

- **OASIS paper:** arXiv 2411.11581 (the thing we're reproducing).
- **TypeSafe "Jev"** (Sept 2026): a REAL, RL-trained "System One" **decision model**
  that does single-pass calibrated typed classification and *beats* generative
  judges on classification (F1 77.8 vs 74.1). Proves single-pass classification is
  legitimate **for judgments with known answers** — NOT for behavioral simulation.
  (Name collision with our JEV; unrelated.)
- **arXiv 2606.12369** ("Should LLM Agents Decide in Social Simulations?"):
  "LLM-based action selection is **not a direct replacement** for explicit decision
  policies: it can **alter the intended behavior**." = our divergence, named.
- **Nokia AnyJev** (Apache-2.0, `pip install anyjev`, supports vLLM, **training-free**):
  turns any open LLM into a calibrated decision layer. Two separable corrections:
  - **`perm`** (cyclic option-shift marginalization) — the SAFE half; cancels
    position bias; gain predicted zero-label by the raw order-flip rate (ρ≈0.6).
  - **`bc`** (batch/label prior division) — the RISKY half; HURTS when the true
    label marginal is skewed (our case: repost-dominant). The AnyJev L0 study
    explicitly says: for known-skewed marginals use `Decider(prior="none")`
    (perm only).
  - On Qwen3-8B/BANKING77, raw label-token logits (= our mechanism) have a **23%
    option-order flip rate** — AnyJev perm cuts it to ~7%. Our flip is far worse
    (total Like→Follow collapse).
- **Fine-tuned alternatives** (Jeff, decider, reflex, kev, eve-rlcd): actually
  TRAIN Qwen into a decision model (SFT + RLCD). Higher ceiling, needs a training
  pipeline + labels. **Not required** given AnyJev is training-free.

**Net:** the single-token paradigm is sound *for calibrated judgment*; our bug is
running a decision-model *interface* on an untrained generative model (Llama-3) via
raw logit-bias. AnyJev `perm` is the correct, training-free version of what we were
approximating. But even a perfectly calibrated action *classifier* may still produce
different *cascades* than generation (behavior-divergence paper) — calibration fixes
brittleness, not necessarily emergent fidelity.

---

## 4b. TRANSITION PLAN — move the engine onto AnyJev as the decision layer

> **Intent.** Stop maintaining our own hand-rolled single-token decision mechanism
> (`DEFAULT_ACTION_TOKEN_MAP` logit-bias + ad-hoc parse) and adopt **AnyJev** as the
> decision layer wherever possible. AnyJev is the correct, library-maintained,
> training-free version of exactly what our classifier approximates. Our engine
> (`step_jev`, scheduler, belief, prompt-cache architecture, harness) stays; only the
> **classifier backend** is swapped. The goal is: our JEV = `<our engine>` + `<AnyJev
> as the Decider>`, with generation kept as the separate max-fidelity mode.

### Why transition (the case)
- Our raw logit readout IS AnyJev's `raw` level (23% flip rate, the brittleness we
  hit). Our inline `apply_l0_debias` (commit `7f7673c`) is a hand-reimplementation of
  AnyJev's `perm`/L0 that already works (repost 16→86, NRMSE 0.73→0.24) — i.e. we have
  independently rebuilt a slice of AnyJev. Adopting the library gives us the rest for
  free: calibrated L1 (temperature scaling), L2 (closed-form head), the adaptive
  rotation budget (certified 1% disagreement at ~2.2× fewer prefills), and a
  maintained API matching TypeSafe's System One.
- It is Apache-2.0, `pip install anyjev`, supports vLLM (which we already serve), and
  needs **no fine-tuning** for L0/L1.

### What AnyJev replaces vs what stays (scope — keep this honest)
REPLACES (≈150 lines in `oasis/inference/jev_classifier.py`):
- `DEFAULT_ACTION_TOKEN_MAP`, `_format_logit_bias_payload`, `_ensure_token_bias`,
  `auto_discover_token_ids` — the hand-picked token-id machinery.
- the single-token readout + top-5 parse in `classify_batch`.
- our inline `apply_l0_debias` (superseded by AnyJev's own L0, keep ours as fallback).
STAYS (the ~98% that is the actual engine):
- `jev_env.py` `step_jev()`, `micro_time_scheduler.py`, `belief_state.py`,
  `jev_prompt_builder.py` (the KV-cache prompt architecture), all configs/harness,
  the classic-leg fixes, generative mode.

### Target architecture
Introduce an `AnyJevClassifierClient` that satisfies the SAME interface our engine
already calls (`classify_batch(items) -> list[ClassificationResult]`), so `step_jev`
is UNCHANGED. Internally it maps each `EvalItem` to an AnyJev call:
```python
from anyjev import Decider, Question
from anyjev.backends.vllm import VLLMBackend

# one Decider per run, pointed at the vLLM we already start in the sbatch
d = Decider(VLLMBackend(jev_url, model_name), level="L0", prior="none")
#   prior="none" is REQUIRED for us: our action marginal is repost-skewed and the
#   AnyJev batch prior HURTS skewed marginals (see §4). L0 = perm-only debias.
q = Question.choice(
        "Which single action does this agent take on the post?",
        options=[ACTION_DESCRIPTIONS[c][0] for c in allowed_chars],  # tool names
        name="action")
# state = our existing assembled per-post prompt (persona+post), as the AnyJev "state"
dist = d.decide(state, [q])["action"].distribution   # {tool_name: prob}
# map tool_name -> action_char, fill ClassificationResult(logits=dist, ...)
```
Key mapping notes:
- `ACTION_DESCRIPTIONS` (in `jev_prompt_builder.py`) is already the single source of
  truth char↔tool-name↔description — reuse it so options stay config-derived.
- Preserve the `logits`/`confidence` fields on `ClassificationResult` from AnyJev's
  returned `distribution` so the confidence dump + budget resolution are unchanged.
- Our prompt-cache layering (task/post/agent) still applies — AnyJev's `state` is just
  our assembled prompt; keep it byte-stable for RadixAttention reuse.

### Deployment reality (the friction, documented)
- `anyjev` is **NOT in the SIF** (verified: `ModuleNotFoundError`), and the SIF
  filesystem is **read-only at runtime**. Adopting the library therefore needs a
  **SIF rebuild**: add `anyjev` (and `anyjev[hf]` if using L2) to `jev_oasis.def`,
  rebuild via `srun --partition=haswell ... apptainer build --force` (~8–10 min).
  Audit dep compatibility with the pinned vllm 0.23 / torch first (the full oasis
  import set is 11 third-party pkgs; add anyjev's without breaking the vllm pin).
- L0/L1 run against our existing `--task generate` vLLM server. **L2** (closed-form
  head, +6–8 acc points) needs an **embed-pooler** server
  (`vllm serve ... --task embed --override-pooler-config ...`) and 100–300 labels per
  question — a bigger change; defer unless L0/L1 prove insufficient.

### Phased plan
1. **Phase 0 (no rebuild, DONE):** inline `apply_l0_debias` proves L0 helps on our
   task (repost 16→86). This de-risks the whole transition — we KNOW the paradigm works
   here before paying the rebuild cost.
2. **Phase 1 — strength knob (no rebuild, 1 line):** add `prior_strength` (0.5/0.75/1.0)
   to our inline `apply_l0_debias` and sweep it to fix the over-flatten (repost should
   climb back toward classic's 107 at ~0.75). This also tells us the *target* behavior
   before swapping to the library.
3. **Phase 2 — adopt the library:** add `anyjev` to the `.def`, rebuild SIF, implement
   `AnyJevClassifierClient` (above), gate behind `OASIS_JEV_ANYJEV=1`. A/B it against
   our inline L0 on the same topic/seed — they should match within noise (both are
   perm-only L0); if they diverge, our inline version has a bug the library exposes.
4. **Phase 3 — calibration (optional):** if thresholded confidence matters (e.g. for a
   decision rule), add AnyJev **L1** (temperature scaling, 100–500 labels bootstrapped
   from classic/generation runs as ground truth). ECE 0.24→0.095.
5. **Phase 4 — retire the hand-rolled path:** once `AnyJevClassifierClient` matches or
   beats our inline L0 across seeds, delete `DEFAULT_ACTION_TOKEN_MAP` + token-bias
   machinery (keep guided-choice as the masking primitive AnyJev can also use).

### The caveat that bounds this transition (do not forget)
AnyJev makes the *classifier* calibrated and stable; it does **not** make
classification equal to generation. Our own ablation (k=1 classic + generative run)
shows generation reaches NRMSE 0.08 while even debiased classification sits at ~0.24.
So the transition's ceiling for *cascade fidelity* is "good, fast, paper-level"
(≈0.3 vs real), NOT "parity with generation." Position AnyJev-L0 as the **fast engine**
and generation as the **faithful engine**; the transition is about making the fast
engine correct and maintainable, not about beating generation.

---

## 5. Current branch state (commits this session)

```
<newer: this handoff update + AnyJev transition plan>
7f7673c feat(jev): AnyJev L0 debias backend (training-free, drop-in)  <-- VALIDATED
07ca0e3 docs(jev): comprehensive handoff — findings, ablation ladder, next steps
5b1ee97 feat(jev): instruct-frame option for single-token classification
655a15c feat(jev): native-generation per-post mode (compose action, like classic)
d2c90fa fix(jev): run confidence dump in any per-post mode, not only competitive
54bff9a feat(jev): guided-choice enum decoding (correct action->token mapping)
31e7d53 test(align): k=1 classic feed-size experiment
0463e9c feat(jev): competitive per-post mode + P(R)-vs-P(L) confidence dump
3885af7 feat(jev): verbatim upstream-OASIS prompt mode (per-post and whole-feed)
00c3b82 feat(jev): whole-feed mode (one guided-JSON call/agent over full feed)
7b58bef feat(jev): add follower/audience context to agent suffix
39d41e7 fix(jev): sample classifier at temp 1.0 to match upstream classic
75b3327 fix(jev): reproduce base-OASIS anti-just-like steer (both upstream lines)
1c706fe refactor(jev): derive task prompt from available_actions, match OASIS 1:1
```
All experiment modes are **gated behind env flags, default off** — the baseline
per-post logit-bias path is intact and unchanged. Modes are mutually composable
except feed_mode (which supersedes the per-post branches).

### 5.1 Known correctness notes / preferences baked in
- **Commit style:** modular by concern, scoped conventional-commit messages.
- **Fork terminology:** our fork = "origin"; "upstream" = camel-ai/oasis.
- **Merge strategy:** keep changes to upstream minimal (essentials-only).
- **Parity definition:** match how the paper RAN it, not codebase defaults.
  Authoritative parity test = JEV-sim vs stock-OASIS-sim on same topic/seed/
  threshold; real_data pkls are a secondary sanity check.
- Upstream-as-shipped **crashes** on align (`KeyError: active_threshold`, generator
  leaves it commented). Our configs populate it; `[0.1]*24` is the paper-implied value.
- Classic's whole-feed tool prompt **overflows 8192 ctx** as the feed grows → some
  classic agents 400 and no-op. This slightly *under*counts classic, so JEV-vs-classic
  comparisons are fair-to-conservative on classic's side.
- `.gitignore` excludes HPC artifacts (SIF, slurm out/err, sim DBs, caches). Never
  leave multi-GB artifacts untracked inside the repo tree (a `git stash -u` once
  swept the 6.9 GB SIF and broke a run).

---

## 6. Where to go next (prioritized, with rationale)

> **Primary direction (decided this session): transition the decision layer onto
> AnyJev — see §4b for the full phased plan.** Phase 0 (inline L0) is DONE and
> VALIDATED (repost 16→86, NRMSE 0.73→0.24, 0.325 vs real ≈ paper, 1.95× speed).
> The steps below are ordered to execute that transition.

### P0 — Fix L0 over-flatten with a strength knob (Phase 1; no rebuild, ~1 line)
L0 at full strength over-flattened our skewed marginal (repost 107→86, like/follow
inflated, do_nothing→0, follow(table) 94→154). Add `prior_strength` (scale the
subtracted per-letter profile by 0.5/0.75/1.0) to `apply_l0_debias` and sweep it.
The AnyJev study's default is **0.75** precisely to preserve a legitimate majority on
skewed marginals — expect repost to climb back toward classic's ~107 at 0.75. Match
classic's *action MIX*, not just cascade scale. This also fixes the issue a reader
asked about (the "near-doubling": over-flatten converted do-nothing/default turns
into active like/follow actions).

### P1 — Adopt the AnyJev library as the backend (Phase 2; needs SIF rebuild)
Implement `AnyJevClassifierClient` behind the existing `classify_batch` interface
(code sketch in §4b), gated by `OASIS_JEV_ANYJEV=1`. Add `anyjev` to `jev_oasis.def`,
rebuild the SIF (audit vllm-0.23/torch dep compatibility first). A/B against our
inline L0 — they should match within noise (both perm-only L0). Use `prior="none"`.

### P1b — Cheap diagnostics worth having (one job each)
1. **Fix `dump_action_confidence_gap` to log all 4 actions** (currently only
   p_like/p_repost — the instruct-frame caveat §3.2 showed this hid P(F) dominance).
2. **Measure `order_flip_raw`** (reverse the action list, re-run) — the zero-label
   predictor of perm gain; documents how brittle raw was vs L0-fixed.

### P2 — Calibration + retire hand-rolled path (Phases 3–4)
If thresholded confidence matters, add AnyJev **L1** (temperature scaling, labels
bootstrapped from classic/generation runs). Once `AnyJevClassifierClient` matches/beats
inline L0 across seeds, delete `DEFAULT_ACTION_TOKEN_MAP` + token-bias machinery.

### P3 — The depth gap (orthogonal, still open)
Even at scale/breadth parity, BOTH engines produce **depth-1 flat-star** cascades
while real data is **depth-2**. Root cause (previously diagnosed): the align harness
hardcodes a tiny recsys cache so reposts rarely get re-recommended → no multi-hop
chains. This is a recsys-fidelity issue (config: `max_rec_post_len`/
`refresh_rec_post_count`), independent of the action mechanism. `jev_align.yaml`
already bumps these to 10; depth is still 1 — needs its own investigation (reposts
must enter feeds for A→B→C chains to form). AnyJev does NOT touch this.

### P4 — Benchmark & write-up
- 5× benchmark of each engine mode (`run_bench5.sbatch`) for mean±std NRMSE/speedup:
  classic / generative / guided-choice+L0(0.75) / guided-choice+AnyJev-lib.
- Two genuine, separable findings worth writing up:
  1. *"single-token action classification under raw logit-bias does not reproduce
     LLM-agent social cascades; the action must be generated OR the classifier must
     be debiased"* — k=1 control + generation ablation as evidence.
  2. *"a training-free AnyJev-L0 debias recovers paper-level cascade fidelity at
     ~2× throughput, trading a controlled over-flatten tuned by prior_strength."*

---

## 7. Hard-won lessons (operational, don't relearn these)

- **Probes must be readable.** Never base64-encode a remote probe to dodge SSH
  quoting; use a plain here-doc (`cat > f.py <<"PY" … PY`) or a file-editing tool.
- **One GPU job at a time** on dgxa100 (two 0.90-VRAM jobs collide/OOM).
- **"FAILED" is usually the analysis script, not the sim** — grep the `.out` for
  the PERFORMANCE/PARITY block first.
- **The SIF is read-only at runtime**; `oasis/` is bind-mounted (edit freely, no
  rebuild); new deps need a `.def` edit + rebuild.
- **vLLM 0.23**: `guided_choice` is gone; use `structured_outputs={"choice":[...]}`.
  Needs `VLLM_USE_FLASHINFER_SAMPLER=0`. `llama3_json` tool parser crashes on plain
  Llama-3 (no `<|python_tag|>`); use `hermes`.
- **Plain Llama-3-8B's chat template has NO tools block** — passing `tools=` is
  silently dropped. The classic leg needs the `tool_chat_template_llama3_min.jinja`
  (adds a tools block) for `tool_choice=auto` to emit calls.
- **Verify a number before interpreting it.** Several times a "finding" was an
  artifact (over-steered prompt → fake 110v110 parity; dump logging only 2 of 4
  actions → false "Repost-favoring"). Always check the endpoint 200/400 split and
  that the leg didn't silently Skip-fallback before trusting counts.

---

## 8. Glossary
- **classic** — stock OASIS path (`use_jev:false`), CAMEL tool-calling.
- **JEV** — our engine (`use_jev:true`), per-post single-token classification.
- **guided-choice** — vLLM `structured_outputs:{choice:[...]}` masking (correct,
  tokenizer-agnostic) — replaces the hand-rolled logit-bias.
- **generative mode** — JEV generates `{"action":"..."}` per post instead of a letter.
- **AnyJev / L0 / perm / bc** — Nokia's training-free decision layer; perm =
  cyclic-shift debias (safe), bc = batch-prior division (risky on skewed marginals).
- **flip rate / order_flip_raw** — how often reversing the option list changes the
  answer; a zero-label brittleness/predictor metric.
- **NRMSE** — normalized RMSE of a cascade curve vs a reference (classic or real).
