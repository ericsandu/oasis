# JEV Action-Parity Roadmap

Goal: extend JEV so it reproduces base-OASIS behaviour across the **full action
surface**, not just post-targeted reactions, while keeping JEV's two structural
wins — single-token logit-bias classification and inverted RadixAttention
KV-cache (root TASK preamble → per-post prefix → per-agent suffix).

North star: **parity with how base OASIS actually works**, not an invented
mechanism. Where a faithful mechanism is also cheaper, prefer it; where it is
not, match base OASIS first and optimise second.

---

## 0. Current state (baseline for this plan)

- **Done:** post-targeted *reactions* via 1-token classify over `L R Q C F S`
  (Like / Repost / Quote / Comment / Follow / Skip). Parity with classic on the
  Information-Spreading experiment: scale/breadth NRMSE < 0.08, ~2× faster
  (recsys-bound ceiling, see scaling note).
- **Partly wired:** `generate_comment / generate_quote / generate_post` exist in
  `jev_classifier.py`; `classify_batch(generate_comments=True)` triggers a
  secondary generation call for `C`/`Q`. Quote and Comment *content* is
  generated but the action pipeline past that (threading, open-comments) is not
  faithful yet.
- **Out of scope for current research (deferred):** recsys/orchestration
  scaling (user-bio embedding cache, vectorised rec loop, recsys↔classify
  pipelining). Tracked separately — do NOT pursue now.

Base OASIS full action surface (from `agent_action.py` / `ActionType`):
post-targeted (like/unlike/dislike/undo/repost/quote/report/create_comment),
comment-targeted (like/dislike/undo on comments), feed-independent
(create_post, search_posts, search_user, follow/unfollow, mute, trend,
purchase, groups).

---

## 1. Action taxonomy (determines the JEV mechanism per action)

| Class | Actions | JEV mechanism |
|---|---|---|
| **A. Post-targeted reaction** | like, repost, follow-author, (skip) | ✅ 1-token classify (done) |
| **B. Post-targeted + generation** | quote, comment | classify picks Q/C → secondary generation on the regular endpoint, content cached after the post prefix |
| **C. Comment-targeted** | open-comments → like/dislike/reply on comments | NEW: two-stage "open comments" action (see §3) |
| **D. Feed-independent** | create_post, search_posts/user, trend, purchase | NEW: per-action confidence score, compared across actions, pick argmax (see §4) |

---

## 2. Class B — Quote & Comment (easier; post-targeted)

These are the natural next step: still anchored to a specific post, so they fit
the existing inverted-prefix cache.

- Flow: 1-token classify yields `Q` or `C` → fire a **secondary generation
  request** on the normal completions endpoint for the content, reusing the
  already-cached `[TASK][post prefix][persona]` as the generation prompt prefix
  (so the quote/comment generation also gets KV-cache hits off the post).
- **Comment caching:** cache generated comments keyed to their post, placed in
  the trie **after** the post prefix (so `[post prefix][comment prefix]` is a
  shared extension — a comment is "a post hanging off a post").
- **Scaling worry (noted by Eric):** issuing a generation request per comment of
  per post may not scale. Mitigations to evaluate:
  - batch all `C`/`Q` generations for a step into one `asyncio.gather` (already
    the classify pattern) so vLLM batches them server-side;
  - only generate content for comments that will actually be dispatched
    (post-schedule), not for every candidate;
  - cap generated-comment length (short tokens → cheap).

---

## 3. Class C — Comment interaction via an "open comments" action (faithful model)

Eric's preferred, most-realistic design: don't pre-request every comment of
every post. Instead model how a real user drills in.

1. Add an **`open_comments`** pseudo-action to the classify menu for a post
   (e.g. extend the letter set, or a second-stage classify). The TASK preamble
   clarifies: *"if you want to read or reply to the discussion on a post, choose
   Open-Comments; this reveals the post's comments next step so you can act on
   them."*
2. When an agent picks `open_comments` on post P:
   - that agent's **next step** receives P's comments **as feed items**, almost
     like a mini-refresh scoped to P's comment thread (comments arrive "like
     posts");
   - the agent can then Like / Dislike / Reply to those comments using the SAME
     1-token classify machinery (comments are just post-like eval items with the
     comment prefix cached after the post).
3. This keeps requests **demand-driven**: comments are only materialised/fetched
   for posts an agent actually opened, instead of N×P×(comments) up front.
   Scales with *interest*, like the real platform.

Parity check: compare against base OASIS `create_comment` + `like_comment` /
`dislike_comment` outcomes — do agents comment/engage on comments at comparable
rates and on comparable threads?

---

## 4. Class D — Feed-independent actions via confidence scoring

Actions like `create_post`, `search_posts`, `search_user`, `trend`,
`purchase_product` are not tied to a single feed post, so the per-post classify
does not apply.

- For each feed-independent action, issue **one query per step per agent asking
  for a confidence/propensity score** (0–1) that the agent would take it now,
  given persona + recent context.
- Compare those scores against each other (and against the best post-targeted
  reaction's implied score) and **pick the argmax**, respecting
  `max_actions_per_agent`.
- Keep it cache-friendly: the feed-independent prompt prefix is also run-constant
  per action type → cache at the trie root alongside TASK_INSTRUCTION; only the
  per-agent persona tail varies.
- `create_post` content, when chosen, uses the existing `generate_post` path.

Parity check: base-OASIS agents post/search at some rate under the free-form
prompt; the confidence-threshold must be calibrated so JEV's feed-independent
action rates match.

---

## 5. Cross-cutting: KV-cache discipline (keep JEV's win)

Every new prompt must preserve the inverted structure:

```
[run-constant instruction]      → trie root (TASK_INSTRUCTION + any per-action-class preamble)
[per-post / per-comment prefix] → shared across agents
[per-agent persona + cue]       → the only varying tail
```

Never put run-constant text in the per-agent suffix (it defeats caching — this
was the lesson from the action-steer calibration).

---

## 6. Validation protocol (per action class)

For each class added:
1. Run classic (base OASIS, tool-aware chat template, `tool_choice=auto`) vs JEV
   on the same topic/seed, matched prompts.
2. Compare per-action counts AND cascade metrics (scale/depth/max_breadth NRMSE),
   plus any action-specific metric (e.g. comment-thread depth for Class C).
3. Target: action-count ratios and metric NRMSE in the same band achieved for
   reactions (< ~0.1).
4. Record wall-clock, but treat the speedup as recsys-bound (see scaling note)
   — do not optimise recsys as part of this work.

---

## 7. Deferred / out of scope (tracked, not now)

- Recsys scaling: cache static user-bio embeddings (currently re-encoded every
  step), vectorise the O(N·P) per-user recommendation Python loop, and overlap
  `update_rec_table()` with the classify burst so the GPU is not idle between
  steps. This is JEV's true wall-clock ceiling at large N, but it is shared with
  classic and orthogonal to action parity.
- Sub-step micro-time `created_at` persistence (actions currently share the
  step's integer minute; cross-step ordering is correct).
- Standalone `jev_information_spreading.py` driver still double-calls
  `update_rec_table` (align driver is fixed); fix for consistency.
