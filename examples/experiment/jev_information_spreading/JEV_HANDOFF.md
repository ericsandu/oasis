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

### 3.1 What each result PROVES (the logic chain)
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

## 5. Current branch state (commits this session)

```
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

### P0 — Decide the engine's identity (human/research call, not code)
The core result is settled and defensible: **single-token classification is a
speed-optimized approximation that does not reproduce cascades; native generation
reaches parity at ~1.1× speed.** Decide which the engine *is*:
- **(a) Faithful engine = generation.** Position single-token JEV honestly as a
  fast approximation valid only for coarse signals. Benchmark generative mode 5×
  for mean NRMSE + speedup (`run_bench5.sbatch` exists; point it at generative).
- **(b) Fast engine, made trustworthy.** Add **AnyJev perm-L0** (training-free,
  `prior="none"` given our skewed marginal) to stabilize the single-token path.
  Expectation from the L0 study: fixes the flip brittleness (23%→~7%, ours worse),
  **but may not close the cascade gap** — calibration ≠ generative behavior.

### P1 — Cheap diagnostics to run FIRST (one job each)
1. **Measure `order_flip_raw`.** Reverse the action list (S/F/R/L) and re-run
   guided-choice; compare action distributions. The AnyJev study says this
   zero-label number predicts perm-L0 gain (ρ≈0.6). If our flip rate is as high as
   the Like→Follow collapse implies, perm-L0 is worth building; if low, the gap is
   purely classification-vs-generation and L0 won't help.
2. **Fix `dump_action_confidence_gap` to log all 4 actions** (p_like, p_repost,
   p_follow, p_skip) and re-run — gets the honest per-post distribution the
   instruct-frame caveat (§3.2) showed we're currently missing.

### P2 — If pursuing the fast engine
- `pip install anyjev` requires adding it to `jev_oasis.def` + SIF rebuild (it's a
  new dep; runtime fs is read-only). OR reimplement perm-only L0 inline (it's just
  K cyclic-rotation prefills + averaging — cheap, no new dep). Prefer the inline
  reimplementation to avoid a rebuild; it's ~40 lines over the existing
  `classify_batch`. Use `prior="none"` (our marginal is repost-skewed; the batch
  prior would HURT — see §4).

### P3 — The depth gap (orthogonal, still open)
Even at scale/breadth parity, BOTH engines produce **depth-1 flat-star** cascades
while real data is **depth-2**. Root cause (previously diagnosed): the align harness
hardcodes a tiny recsys cache so reposts rarely get re-recommended → no multi-hop
chains. This is a recsys-fidelity issue (config: `max_rec_post_len`/
`refresh_rec_post_count`), independent of the action mechanism. `jev_align.yaml`
already bumps these to 10; depth is still 1 — needs its own investigation (reposts
must enter feeds for A→B→C chains to form).

### P4 — Benchmark & write-up
- 5× benchmark of the chosen engine (`run_bench5.sbatch`) for mean±std NRMSE/speedup.
- The ablation ladder (§3) is a genuine, publishable finding on its own:
  *"single-token action classification under logit bias does not reproduce
  LLM-agent social cascades; the action must be generated"* — with the k=1 control
  and the generation-vs-classification ablation as the evidence.

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
