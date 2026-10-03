# =========== Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. ===========
"""JEV Batched Logit Classifier and Intra-Feed Action Resolver.

Implements high-speed 1-token logit classification for OASIS social agent simulation.
Reduces conventional 150-300 decode steps down to a single GPU forward pass for 95% of
social interactions ([L]ike, [R]epost, [C]omment, [S]kip), coupled with a secondary
worker for conditional comment generation and intra-feed action budget resolution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

import httpx

from oasis.social_agent.jev_prompt_builder import JEVPromptBuilder

# Enforce resource guardrails
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

logger = logging.getLogger(__name__)

DEFAULT_VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_VLLM_MODEL = os.environ.get(
    "VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
)

DEFAULT_ACTION_TOKENS = ("L", "R", "Q", "C", "F", "S")

# Comprehensive token ID fallback maps across cl100k, o200k, Qwen, and LLaMA
# for both bare single characters ('L') and leading-space tokens (' L')
DEFAULT_ACTION_TOKEN_MAP: dict[str, list[int]] = {
    "L": [43, 445, 451, 75, 76],
    "R": [49, 432, 460, 81, 82],
    "Q": [48, 1229, 1486, 80, 81],
    "C": [34, 356, 363, 66, 67],
    "F": [37, 435, 434, 69, 70],
    "S": [50, 328, 336, 82, 83],
}

# Llama-3 INSTRUCT chat-template wrapper (verified byte-exact against the served
# tokenizer's apply_chat_template(add_generation_prompt=True)). Used by
# instruct_frame to wrap a raw completion prompt so /v1/completions asks the
# question of the instruct-aligned model (inside the user/assistant header
# frame) instead of the raw continuation model. {body} = the raw prompt.
LLAMA3_INSTRUCT_FRAME: str = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{body}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Strict JSON Schema Grammar definition for structured output fallbacks
ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["L", "R", "Q", "C", "F", "S"],
                    "description": "Selected social interaction action: L (Like), R (Repost), Q (Quote), C (Comment), F (Follow post author), S (Skip)",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


def _parse_json_action(raw_text: str) -> str:
    """Defensively extracts and validates single action character from JSON or text responses.

    Handles strict JSON objects, markdown code blocks, regex key-value extraction,
    and defaults to JEVPromptBuilder.parse_action_char.
    """
    if not raw_text or not raw_text.strip():
        return "S"

    text = raw_text.strip()

    # 1. Direct JSON deserialization
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for k in ("action", "reaction", "decision", "choice"):
                val = data.get(k)
                if val is not None:
                    parsed = JEVPromptBuilder.parse_action_char(str(val))
                    if parsed in JEVPromptBuilder.VALID_ACTIONS:
                        return parsed
    except Exception:
        pass

    # 2. Extract JSON code blocks if present
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            if isinstance(data, dict):
                for k in ("action", "reaction", "decision", "choice"):
                    val = data.get(k)
                    if val is not None:
                        parsed = JEVPromptBuilder.parse_action_char(str(val))
                        if parsed in JEVPromptBuilder.VALID_ACTIONS:
                            return parsed
        except Exception:
            pass

    # 3. Regex key-value matching: "action": "L" or 'action': 'Like'
    kv_match = re.search(
        r'["\'](?:action|reaction|decision|choice)["\']\s*:\s*["\']([LRQCFS])["\']',
        text,
        re.IGNORECASE,
    )
    if kv_match:
        return kv_match.group(1).upper()

    kv_word_match = re.search(
        r'["\'](?:action|reaction|decision|choice)["\']\s*:\s*["\'](\w+)["\']',
        text,
        re.IGNORECASE,
    )
    if kv_word_match:
        parsed = JEVPromptBuilder.parse_action_char(kv_word_match.group(1))
        if parsed in JEVPromptBuilder.VALID_ACTIONS:
            return parsed

    # 4. Standard action parsing fallback
    return JEVPromptBuilder.parse_action_char(text)


@dataclass
class EvalItem:
    """Represents a single item evaluation request for an agent evaluating a post.

    Attributes:
        user_id: Unique integer identifier for the evaluating agent.
        post_id: Unique integer identifier for the candidate post.
        topic: Topic category of the post.
        full_prompt: Assembled prompt string (prefix + suffix).
        post_content: Optional raw content of the post for comment fallback workers.
    """

    user_id: int
    post_id: int
    topic: str
    full_prompt: str
    post_content: str = ""


@dataclass
class ClassificationResult:
    """Result of 1-token action classification and intra-feed scoring.

    Attributes:
        user_id: Evaluating agent's integer ID.
        post_id: Evaluated post's integer ID.
        action_char: Predicted interaction choice ('L', 'R', 'Q', 'C', 'S').
        confidence: Probability score in [0.0, 1.0] for the chosen action.
        logits: Logit / probability distribution across {'L', 'R', 'Q', 'C', 'S'}.
        comment_text: Generated comment text if action_char == 'C' and comment worker ran.
        quote_text: Generated quote commentary text if action_char == 'Q' and quote worker ran.
    """

    user_id: int
    post_id: int
    action_char: str
    confidence: float
    logits: dict[str, float] = field(default_factory=dict)
    comment_text: str | None = None
    quote_text: str | None = None


@runtime_checkable
class JEVClassifierClient(Protocol):
    """Protocol interface for JEV action classification backends."""

    async def classify_batch(
        self,
        items: list[EvalItem],
        generate_comments: bool = False,
    ) -> list[ClassificationResult]:
        """Classify a batch of evaluation items into 1-token social actions.

        Args:
            items: List of EvalItem instances to evaluate.
            generate_comments: Whether to trigger secondary comment generation
                               for items where action_char == 'C'.

        Returns:
            List of ClassificationResult objects in corresponding order.
        """
        ...

    async def generate_comment(
        self,
        user_prompt: str,
        post_content: str,
    ) -> str:
        """Generate a realistic comment string for an agent commenting on a post.

        Args:
            user_prompt: Assembled user persona prompt or context.
            post_content: Post text being commented on.

        Returns:
            Generated comment text.
        """
        ...

    async def generate_comments_batch(
        self,
        comment_requests: list[tuple[str, str]],
    ) -> list[str]:
        """Batch generate comments for multiple (user_prompt, post_content) pairs.

        Args:
            comment_requests: List of (user_prompt, post_content) tuples.

        Returns:
            List of generated comment strings.
        """
        ...

    async def generate_quote(
        self,
        user_prompt: str,
        post_content: str,
    ) -> str:
        """Generate a realistic quote commentary string for an agent quote-tweeting a post.

        Args:
            user_prompt: Assembled user persona prompt or context.
            post_content: Post text being quoted.

        Returns:
            Generated quote commentary text.
        """
        ...

    async def generate_quotes_batch(
        self,
        quote_requests: list[tuple[str, str]],
    ) -> list[str]:
        """Batch generate quote commentaries for multiple (user_prompt, post_content) pairs.

        Args:
            quote_requests: List of (user_prompt, post_content) tuples.

        Returns:
            List of generated quote commentary strings.
        """
        ...

    async def generate_post(
        self,
        agent_context: str,
        topic: str,
        stance_label: str = "Neutral",
    ) -> str:
        """Generate a realistic spontaneous root post for an agent given persona context and topic.

        Args:
            agent_context: Assembled agent persona, traits, and bio context.
            topic: Primary topic or community category (e.g. 'tech', 'sports', 'politics').
            stance_label: Qualitative stance descriptor (e.g. 'Supportive', 'Neutral').

        Returns:
            Generated post content string with relevant hashtag.
        """
        ...

    async def generate_posts_batch(
        self,
        post_requests: list[tuple[str, str, str]],
    ) -> list[str]:
        """Batch generate spontaneous root posts for multiple (agent_context, topic, stance_label) tuples.

        Args:
            post_requests: List of (agent_context, topic, stance_label) tuples.

        Returns:
            List of generated root post content strings.
        """
        ...


def compute_softmax(logits: dict[str, float]) -> dict[str, float]:
    """Computes numerically stable softmax probabilities over a dictionary of logits.

    Args:
        logits: Dictionary mapping action keys to float logit scores.

    Returns:
        Dictionary mapping action keys to normalized probabilities summing to 1.0.
    """
    if not logits:
        return {}

    max_val = max(logits.values())
    exp_vals = {k: math.exp(v - max_val) for k, v in logits.items()}
    sum_exp = sum(exp_vals.values())

    if sum_exp <= 0.0 or math.isinf(sum_exp):
        uniform = 1.0 / len(logits)
        return {k: uniform for k in logits}

    return {k: v / sum_exp for k, v in exp_vals.items()}


def resolve_intra_feed_budget(
    results: list[ClassificationResult],
    budget: int = 1,
    downgrade_to_skip: bool = True,
) -> list[ClassificationResult]:
    """Resolves intra-feed action budget constraints across evaluated items.

    When an agent evaluates multiple items in their personalized feed and generates
    more non-skip actions ('L', 'R', 'C') than permitted by their step budget,
    this function selects the non-skip actions with the highest softmax confidence.
    Excess non-skip actions are converted to 'S' (Skip) if downgrade_to_skip=True,
    preserving full feed result length and ordering.

    Supports batches containing items from single or multiple agents (grouped by user_id).

    Args:
        results: Evaluated results across one or more users.
        budget: Maximum number of non-skip actions allowed per user (default: 1).
        downgrade_to_skip: If True, excess non-skip items are converted to 'S' (Skip).
                          If False, excess non-skip items are omitted from output.

    Returns:
        List of ClassificationResult objects respecting budget constraints.
    """
    if not results or budget < 0:
        return list(results)

    # Group results by user_id preserving original item positions
    user_items: dict[int, list[tuple[int, ClassificationResult]]] = defaultdict(
        list
    )
    for idx, res in enumerate(results):
        user_items[res.user_id].append((idx, res))

    resolved_map: dict[int, ClassificationResult] = {}

    for items in user_items.values():
        # Separate non-skips ('L', 'R', 'C') from skips ('S')
        non_skips = [
            (idx, res) for idx, res in items if res.action_char != "S"
        ]

        if len(non_skips) <= budget:
            for idx, res in items:
                resolved_map[idx] = res
        else:
            # Sort non-skips by confidence descending (argmax confidence selection)
            sorted_non_skips = sorted(
                non_skips, key=lambda pair: pair[1].confidence, reverse=True
            )
            allowed_indices = {pair[0] for pair in sorted_non_skips[:budget]}

            for idx, res in items:
                if res.action_char == "S" or idx in allowed_indices:
                    resolved_map[idx] = res
                else:
                    if downgrade_to_skip:
                        skip_confidence = (
                            res.logits.get("S", 0.0) if res.logits else 0.0
                        )
                        resolved_map[idx] = ClassificationResult(
                            user_id=res.user_id,
                            post_id=res.post_id,
                            action_char="S",
                            confidence=skip_confidence,
                            logits=dict(res.logits),
                            comment_text=None,
                            quote_text=None,
                        )
                    # If not downgrade_to_skip, excess item is dropped from resolved_map

    # Return results in original index order
    return [
        resolved_map[idx]
        for idx in range(len(results))
        if idx in resolved_map
    ]


def select_best_action(
    results: list[ClassificationResult],
) -> ClassificationResult | None:
    """Selects the single highest-confidence non-skip action from a list of results.

    Args:
        results: Evaluated results for an agent's feed.

    Returns:
        ClassificationResult with the highest confidence non-skip action,
        or None if all evaluated items resulted in 'S' (Skip).
    """
    non_skips = [r for r in results if r.action_char != "S"]
    if not non_skips:
        return None
    return max(non_skips, key=lambda r: r.confidence)


def resolve_competitive_full_logits(
    results: list[ClassificationResult],
    budget: int = 1,
    skip_char: str = "S",
    downgrade_to_skip: bool = True,
) -> list[ClassificationResult]:
    """Competitive per-post selection over the FULL per-action logits of a feed.

    The default pipeline (resolve_intra_feed_budget) first collapses each post to
    its argmax action, THEN picks the agent's one action by comparing those
    argmax confidences. Under a uniform action bias + greedy/low-mass Repost,
    every post collapses to Like, so the budget stage only ever compares
    "best Like vs best Like" -- a confident Repost is discarded per-post before
    the comparison ever runs.

    This resolver instead keeps every post's full per-action logit dict and, for
    each agent, picks the single (post, action) pair whose per-action softmax
    probability is highest ACROSS the whole feed and across all NON-SKIP actions.
    So a post where Repost is the model's second choice but still high-probability
    can win the agent's one slot over a post where Like barely edged everything.

    NO action weighting: the score is the raw model probability P(action|post)
    read from softmax(result.logits). A 0.60 Like still beats a 0.40 Repost; the
    only change from the default is that we no longer throw the Repost signal away
    before comparing. Pairs the SCARCITY-FRAMED prompt (which tells the model only
    its single strongest action across all posts executes) with a selection rule
    that can actually act on that framing.

    Args:
        results: Evaluated results across one or more users (full logits populated).
        budget: Max non-skip actions per user (default 1).
        skip_char: The no-op action character.
        downgrade_to_skip: Convert non-selected items to skip (preserves ordering).

    Returns:
        List of ClassificationResult respecting budget, where each surviving
        non-skip item's action_char/confidence is set to the WINNING
        (post, action) chosen by cross-feed full-logits comparison.
    """
    if not results or budget < 0:
        return list(results)

    user_items: dict[int, list[tuple[int, ClassificationResult]]] = defaultdict(
        list
    )
    for idx, res in enumerate(results):
        user_items[res.user_id].append((idx, res))

    resolved_map: dict[int, ClassificationResult] = {}

    for items in user_items.values():
        # Build the candidate set: for every post, every NON-SKIP action with a
        # captured logit, scored by its per-post softmax probability.
        # candidate = (score, idx, action_char)
        candidates: list[tuple[float, int, str]] = []
        for idx, res in items:
            probs = compute_softmax(res.logits) if res.logits else {}
            for act, p in probs.items():
                if act == skip_char:
                    continue
                candidates.append((p, idx, act))

        if not candidates:
            # Nothing actionable -> keep originals (all skips / empty logits)
            for idx, res in items:
                resolved_map[idx] = res
            continue

        # Highest raw probability across the whole feed wins the agent's slot(s).
        candidates.sort(key=lambda c: c[0], reverse=True)
        winners: dict[int, tuple[str, float]] = {}
        for score, idx, act in candidates:
            if len(winners) >= budget:
                break
            if idx not in winners:  # one winning action per post slot
                winners[idx] = (act, score)

        for idx, res in items:
            if idx in winners:
                win_act, win_score = winners[idx]
                resolved_map[idx] = ClassificationResult(
                    user_id=res.user_id,
                    post_id=res.post_id,
                    action_char=win_act,
                    confidence=win_score,
                    logits=dict(res.logits),
                    comment_text=res.comment_text,
                    quote_text=res.quote_text,
                )
            else:
                if downgrade_to_skip:
                    skip_conf = (
                        compute_softmax(res.logits).get(skip_char, 0.0)
                        if res.logits else 0.0
                    )
                    resolved_map[idx] = ClassificationResult(
                        user_id=res.user_id,
                        post_id=res.post_id,
                        action_char=skip_char,
                        confidence=skip_conf,
                        logits=dict(res.logits),
                    )
                # else: dropped

    return [
        resolved_map[idx]
        for idx in range(len(results))
        if idx in resolved_map
    ]


def dump_action_confidence_gap(
    results: list[ClassificationResult],
    like_char: str = "L",
    repost_char: str = "R",
) -> list[dict[str, float]]:
    """Per-post P(Repost) vs P(Like) comparison across every evaluated item.

    Reads the raw per-action logits captured during classification and reports,
    for each (agent, post), the model's softmax probability for Like and Repost
    and their gap P(R) - P(L). This is the headline measurement: whether Repost
    is genuinely tiny on every post, or second-but-close on some -- the thing
    that decides whether competitive feed-level selection can surface it.

    No weighting, no bias correction beyond the uniform action-bias which cancels
    in the softmax: these are the model's own relative preferences.

    Args:
        results: Raw (pre-budget) classification results with full logits.
        like_char: Action char for Like.
        repost_char: Action char for Repost.

    Returns:
        One row per item: {user_id, post_id, p_like, p_repost, gap_r_minus_l,
        argmax_action, r_present (1/0 whether Repost appeared in top-logprobs)}.
    """
    rows: list[dict[str, float]] = []
    for res in results:
        probs = compute_softmax(res.logits) if res.logits else {}
        p_like = probs.get(like_char, 0.0)
        p_repost = probs.get(repost_char, 0.0)
        rows.append(
            {
                "user_id": float(res.user_id),
                "post_id": float(res.post_id),
                "p_like": p_like,
                "p_repost": p_repost,
                "gap_r_minus_l": p_repost - p_like,
                "r_present": 1.0 if repost_char in (res.logits or {}) else 0.0,
                "l_present": 1.0 if like_char in (res.logits or {}) else 0.0,
            }
        )
    return rows


class MockJEVClassifierClient:
    """Deterministic, hermetic JEV classifier client for unit tests without GPU or network.

    Enforces reproducible action predictions based on explicit action mappings,
    agent topic stance parsing, or pseudo-random deterministic hashing.
    """

    def __init__(
        self,
        action_mapping: dict[tuple[int, int], str] | None = None,
        default_action: str | None = None,
        mock_comment: str = "Interesting perspective on this topic.",
        mock_quote: str = "Thought-provoking perspective on this topic.",
        default_confidence: float = 0.95,
        seed: int = 42,
    ) -> None:
        """Initializes the mock classifier.

        Args:
            action_mapping: Mapping of (user_id, post_id) -> action character ('L', 'R', 'Q', 'C', 'S').
            default_action: Global fallback action if pair not in action_mapping.
            mock_comment: Default comment text returned for comment generation requests.
            mock_quote: Default quote text returned for quote commentary requests.
            default_confidence: Default confidence assigned to deterministic choices.
            seed: Seed value used for deterministic fallback generation.
        """
        self.action_mapping = dict(action_mapping or {})
        self.default_action = default_action
        self.mock_comment = mock_comment
        self.mock_quote = mock_quote
        self.default_confidence = default_confidence
        self.seed = seed

    def set_action_mapping(
        self, user_id: int, post_id: int, action_char: str
    ) -> None:
        """Configures a specific action response for an agent-post pair."""
        assert action_char in JEVPromptBuilder.VALID_ACTIONS
        self.action_mapping[(user_id, post_id)] = action_char

    def _determine_action(self, item: EvalItem) -> tuple[str, float, dict[str, float]]:
        """Determines the action character and simulated logit distribution for an item."""
        pair = (item.user_id, item.post_id)

        # 1. Explicit mapping override
        if pair in self.action_mapping:
            char = self.action_mapping[pair]
            logits = {"L": 0.05, "R": 0.05, "Q": 0.05, "C": 0.05, "S": 0.05}
            logits[char] = 4.0
            probs = compute_softmax(logits)
            return char, probs[char], probs

        # 2. Fixed default action override
        if self.default_action is not None:
            char = self.default_action
            logits = {"L": 0.05, "R": 0.05, "Q": 0.05, "C": 0.05, "S": 0.05}
            logits[char] = 4.0
            probs = compute_softmax(logits)
            return char, probs[char], probs

        # 3. Deterministic heuristic based on prompt stance parsing
        prompt = item.full_prompt
        stance_score = 0.0
        stance_match = re.search(r"\[STANCE\]:.*?([+-]\d+\.\d{2})", prompt)
        if stance_match:
            try:
                stance_score = float(stance_match.group(1))
            except ValueError:
                stance_score = 0.0

        # Hash-based pseudo-random choice deterministic on user, post, and seed
        h = (
            abs(
                hash(
                    (
                        item.user_id,
                        item.post_id,
                        item.topic,
                        self.seed,
                    )
                )
            )
            % 100
        )

        if stance_score > 0.3 or "Supportive" in prompt:
            # Positive stance: strongly biased to Like, occasionally Quote, Repost or Comment
            if h < 10:
                char = "C"
            elif h < 20:
                char = "Q"
            elif h < 35:
                char = "R"
            elif h < 90:
                char = "L"
            else:
                char = "S"
            base_logits = {"L": 3.0, "R": 1.5, "Q": 1.5, "C": 1.2, "S": -0.5}
        elif stance_score < -0.3 or "Hostile" in prompt or "Skeptical" in prompt:
            # Negative stance: biased to Skip, occasionally critical Comment or Quote
            if h < 10:
                char = "C"
            elif h < 20:
                char = "Q"
            else:
                char = "S"
            base_logits = {"L": -1.0, "R": -1.5, "Q": 1.2, "C": 1.0, "S": 3.5}
        else:
            # Neutral stance
            if h < 35:
                char = "L"
            elif h < 45:
                char = "R"
            elif h < 55:
                char = "Q"
            elif h < 65:
                char = "C"
            else:
                char = "S"
            base_logits = {"L": 1.0, "R": 0.8, "Q": 0.8, "C": 0.8, "S": 1.5}

        # Boost the chosen action logit
        base_logits[char] += 2.0
        probs = compute_softmax(base_logits)
        return char, probs[char], probs

    async def classify_batch(
        self,
        items: list[EvalItem],
        generate_comments: bool = False,
    ) -> list[ClassificationResult]:
        """Hermetically evaluates items and returns ClassificationResult instances."""
        results: list[ClassificationResult] = []

        for item in items:
            char, conf, logits = self._determine_action(item)
            comment_text = None
            quote_text = None
            if generate_comments and char == "C":
                comment_text = await self.generate_comment(
                    item.full_prompt, item.post_content
                )
            elif generate_comments and char == "Q":
                quote_text = await self.generate_quote(
                    item.full_prompt, item.post_content
                )

            results.append(
                ClassificationResult(
                    user_id=item.user_id,
                    post_id=item.post_id,
                    action_char=char,
                    confidence=conf,
                    logits=logits,
                    comment_text=comment_text,
                    quote_text=quote_text,
                )
            )

        return results

    async def classify_batch_generative(
        self,
        items: list[EvalItem],
        allowed_chars: list | None = None,
        generate_comments: bool = False,
    ) -> list[ClassificationResult]:
        """Mock parity for the native-generation path: same deterministic
        action mapping as classify_batch (the mock has no real generation)."""
        return await self.classify_batch(items, generate_comments=generate_comments)

    # Alias matching plan nomenclature
    classify_actions_batch = classify_batch

    async def generate_comment(
        self,
        user_prompt: str,
        post_content: str,
    ) -> str:
        """Returns deterministic mock comment string."""
        if post_content:
            clean_content = post_content.strip()
            if len(clean_content) > 30:
                clean_content = clean_content[:27] + "..."
            return f"Regarding '{clean_content}': {self.mock_comment}"
        return self.mock_comment

    async def generate_comments_batch(
        self,
        comment_requests: list[tuple[str, str]],
    ) -> list[str]:
        """Batch generates mock comments."""
        return [
            await self.generate_comment(prompt, content)
            for prompt, content in comment_requests
        ]

    async def generate_quote(
        self,
        user_prompt: str,
        post_content: str,
    ) -> str:
        """Returns deterministic mock quote commentary string."""
        if post_content:
            clean_content = post_content.strip()
            if len(clean_content) > 30:
                clean_content = clean_content[:27] + "..."
            return f"Quoting '{clean_content}': {self.mock_quote}"
        return self.mock_quote

    async def generate_quotes_batch(
        self,
        quote_requests: list[tuple[str, str]],
    ) -> list[str]:
        """Batch generates mock quote commentaries."""
        return [
            await self.generate_quote(prompt, content)
            for prompt, content in quote_requests
        ]

    async def generate_post(
        self,
        agent_context: str,
        topic: str,
        stance_label: str = "Neutral",
    ) -> str:
        """Returns deterministic mock post string reflecting topic and stance."""
        clean_topic = topic.strip().lstrip("#")
        templates: dict[str, list[str]] = {
            "tech": [
                f"Excited to test out the new systems and optimization benchmarks today! #{clean_topic}",
                f"Deep dive into compiler architecture and distributed scaling: thoughts? #{clean_topic}",
                f"Modern developer workflows are evolving fast. Great time to build. #{clean_topic}",
            ],
            "sports": [
                f"What a phenomenal match yesterday! Tactical masterclass on the pitch. #{clean_topic}",
                f"Training cycle in full swing. Looking forward to the championship rounds! #{clean_topic}",
                f"Analytics in sports continue to revolutionize game strategies. #{clean_topic}",
            ],
            "politics": [
                f"Key legislative updates today on policy and governance reform. #{clean_topic}",
                f"Constructive dialogue and civic participation are essential for progress. #{clean_topic}",
                f"Examining economic data and market implications for the upcoming quarter. #{clean_topic}",
            ],
        }
        topic_lower = clean_topic.lower()
        pool = templates.get(
            topic_lower,
            [
                f"Fascinating developments happening across the ecosystem today. #{clean_topic}",
                f"Sharing some reflections and observations on current trends. #{clean_topic}",
            ],
        )
        h = abs(hash((agent_context, clean_topic, stance_label, self.seed))) % len(pool)
        return pool[h]

    async def generate_posts_batch(
        self,
        post_requests: list[tuple[str, str, str]],
    ) -> list[str]:
        """Batch generates mock spontaneous root posts."""
        return [
            await self.generate_post(ctx, topic, stance)
            for ctx, topic, stance in post_requests
        ]

    def resolve_budget(
        self,
        results: list[ClassificationResult],
        budget: int = 1,
    ) -> list[ClassificationResult]:
        """Applies intra-feed budget resolution to results."""
        return resolve_intra_feed_budget(results, budget=budget)


class VLLMJEVClassifierClient:
    """Batched 1-token logit classifier client interfacing with vLLM endpoints.

    Utilizes vLLM /v1/completions (with prefix KV-cache reuse and max_tokens=1)
    to perform ultra-low-latency classification, backed by a secondary worker
    for conditional comment generation when 'C' is emitted.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        logit_bias: dict[int | str, float] | None = None,
        token_id_map: dict[str, int] | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        temperature: float = 0.0,
        classify_max_tokens: int = 1,
        stop: list[str] | None = None,
        auto_generate_comments: bool = False,
        max_retries: int = 2,
        comment_temperature: float = 0.7,
        comment_max_tokens: int = 256,
        post_max_tokens: int | None = 512,
        max_concurrent_comments: int = 8,
        auto_discover_token_ids: bool = False,
        guided_choice_actions: Sequence[str] | None = None,
        instruct_frame: bool = False,
    ) -> None:
        """Initializes the vLLM classifier client.

        Args:
            base_url: vLLM OpenAI-compatible base URL (default: VLLM_BASE_URL env or 127.0.0.1:8000/v1).
            model_name: Registered model identifier (default: VLLM_MODEL env).
            logit_bias: Optional logit bias dictionary passed to vLLM.
            token_id_map: Optional mapping from action chars ('L', 'R', 'C', 'S') to vocab token IDs.
            timeout: HTTP request timeout in seconds.
            client: Optional shared httpx.AsyncClient instance.
            temperature: Sampling temperature for action classification.
            classify_max_tokens: Maximum tokens for action classification (default: 4, to allow '[L]', 'Like', etc.).
            stop: Optional stop sequences for completion (default: ['\n']).
            auto_generate_comments: If True, automatically trigger comment generation on 'C'.
            max_retries: Max retry attempts on transient HTTP/connection errors.
            comment_temperature: Temperature for secondary comment generation.
            comment_max_tokens: Maximum tokens for secondary comment generation (default: 256).
            post_max_tokens: Maximum tokens for spontaneous root post generation (default: 512).
            max_concurrent_comments: Semaphore concurrency limit for secondary comment requests.
            auto_discover_token_ids: If True, dynamically queries /tokenize to resolve action token IDs.
        """
        raw_url = base_url or DEFAULT_VLLM_URL
        self.base_url = raw_url.rstrip("/")
        self.model_name = model_name or DEFAULT_VLLM_MODEL
        self.raw_logit_bias = logit_bias
        self.token_id_map = dict(token_id_map or {})
        self.timeout = timeout
        self.temperature = temperature
        self.classify_max_tokens = classify_max_tokens
        self.stop = stop or ["\n"]
        self.auto_generate_comments = auto_generate_comments
        self.max_retries = max_retries
        self.comment_temperature = comment_temperature
        self.comment_max_tokens = comment_max_tokens
        self.post_max_tokens = (
            post_max_tokens if post_max_tokens is not None else 512
        )
        self._comment_semaphore = asyncio.Semaphore(max_concurrent_comments)
        self.auto_discover_token_ids = auto_discover_token_ids
        # When set, classify_batch uses vLLM structured-outputs CHOICE masking
        # (structured_outputs={"choice":[...]}) instead of logit_bias+argmax.
        # The server masks to exactly these letters over the REAL tokenizer, so
        # the model's own (temperature-sampled) preference among the actions
        # decides -- the faithful analogue of classic's tool-call generation,
        # with no hand-picked token ids. None => legacy logit-bias path.
        self.guided_choice_actions = (
            list(guided_choice_actions) if guided_choice_actions else None
        )
        # When True, wrap each raw completion prompt in the Llama-3 INSTRUCT chat
        # template before sending to /v1/completions, so single-token
        # classification asks its question of the instruct-aligned model (inside
        # the user/assistant header frame) rather than the raw continuation
        # model. Isolates "instruct frame" from "generation": tests whether a
        # correctly-framed single-token classifier matches the generative path.
        self.instruct_frame = instruct_frame
        self._token_bias_initialized = False

        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            headers={"Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        """Closes the underlying HTTP client if owned."""
        if self._own_client and self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.aclose()

    async def _ensure_token_bias(self) -> None:
        """Dynamically discovers token IDs from vLLM /tokenize endpoint if available,
        or configures default cl100k/Qwen logit biases for 1-token action classification."""
        if self._token_bias_initialized:
            return

        self._token_bias_initialized = True

        if self.raw_logit_bias:
            return

        discovered_ids: dict[int, str] = {}
        candidate_strings = [
            "L", "R", "Q", "C", "S",
            " L", " R", " Q", " C", " S",
            "l", "r", "q", "c", "s",
        ]

        # 1. Attempt dynamic query to vLLM /tokenize endpoint
        base_clean = self.base_url.rstrip("/")
        tokenize_url = (
            f"{base_clean[:-3]}/tokenize"
            if base_clean.endswith("/v1")
            else f"{base_clean}/tokenize"
        )

        try:
            for s in candidate_strings:
                resp = await self._client.post(
                    tokenize_url,
                    json={"model": self.model_name, "prompt": s},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    toks = data.get("tokens", [])
                    if len(toks) == 1:
                        char_key = s.strip().upper()
                        discovered_ids[int(toks[0])] = char_key
                        if char_key not in self.token_id_map:
                            self.token_id_map[char_key] = int(toks[0])
            if discovered_ids:
                logger.info(
                    "✓ Discovered %d action token IDs from vLLM /tokenize: %s",
                    len(discovered_ids),
                    discovered_ids,
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to query vLLM /tokenize: %s", e)

        # 2. Fallback to standard cl100k/Qwen token IDs if dynamic discovery failed
        if not discovered_ids:
            fallback_map = {
                43: "L", 49: "R", 48: "Q", 34: "C", 50: "S",
                445: "L", 432: "R", 1229: "Q", 356: "C", 328: "S",
                75: "L", 81: "R", 80: "Q", 66: "C", 82: "S",
            }
            discovered_ids = fallback_map
            for tid, act in fallback_map.items():
                if act not in self.token_id_map:
                    self.token_id_map[act] = tid
            logger.info(
                "Configured default cl100k/Qwen token IDs for logit biasing: %s",
                self.token_id_map,
            )

        # Set uniform +50.0 positive logit bias across all valid action tokens
        self.raw_logit_bias = {tid: 50.0 for tid in discovered_ids}

    def _format_logit_bias_payload(self) -> dict[str, float] | None:
        """Formats logit bias dictionary into API-compliant string token ID mapping."""
        if not self.raw_logit_bias:
            return None

        formatted: dict[str, float] = {}
        for k, v in self.raw_logit_bias.items():
            try:
                bias_val = max(-100.0, min(100.0, float(v)))
            except (ValueError, TypeError):
                bias_val = 50.0

            if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                formatted[str(k)] = bias_val
            elif isinstance(k, str):
                char_key = k.strip().upper()
                if char_key in self.token_id_map:
                    val_id = self.token_id_map[char_key]
                    if isinstance(val_id, (list, tuple)):
                        for sub_id in val_id:
                            formatted[str(sub_id)] = bias_val
                    else:
                        formatted[str(val_id)] = bias_val
                elif char_key in DEFAULT_ACTION_TOKEN_MAP:
                    for tid in DEFAULT_ACTION_TOKEN_MAP[char_key]:
                        formatted[str(tid)] = bias_val

        return formatted if formatted else None

    async def classify_batch(
        self,
        items: list[EvalItem],
        generate_comments: bool = False,
    ) -> list[ClassificationResult]:
        """Classifies a batch of EvalItem prompts using vLLM /v1/completions.

        Passes classify_max_tokens and logprobs=5 to perform item-level parallel forward passes
        leveraging vLLM RadixAttention shared prefix KV-cache reuse.
        """
        if not items:
            return []

        if self.auto_discover_token_ids and not self._token_bias_initialized:
            await self._ensure_token_bias()

        prompts = [item.full_prompt for item in items]
        if self.instruct_frame:
            prompts = [
                LLAMA3_INSTRUCT_FRAME.format(body=p) for p in prompts
            ]
        endpoint = f"{self.base_url}/completions"

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": self.classify_max_tokens,
            "temperature": self.temperature,
            "logprobs": 5,
        }
        if self.stop and self.classify_max_tokens > 1:
            payload["stop"] = self.stop

        if self.guided_choice_actions:
            # vLLM structured-outputs CHOICE masking: the server restricts the
            # output to exactly these strings over the real tokenizer, so the
            # model's own (temperature-sampled) preference among the actions
            # decides. No logit_bias / hand-picked token ids. logprobs:5 is kept
            # so the per-post P(action) confidence dump still works.
            payload["structured_outputs"] = {
                "choice": list(self.guided_choice_actions)
            }
        else:
            formatted_bias = self._format_logit_bias_payload()
            if formatted_bias:
                payload["logit_bias"] = formatted_bias

        # Attempt batched completions endpoint
        response_data: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(endpoint, json=payload)
                if resp.status_code == 200:
                    response_data = resp.json()
                    break
                elif resp.status_code == 404:
                    # Endpoint /completions not mounted; fallback to chat completions
                    return await self._fallback_chat_classify_batch(
                        items, generate_comments
                    )
                else:
                    logger.warning(
                        "vLLM /completions returned HTTP %d: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries:
                    logger.error("Failed to query vLLM classifier: %s", e)
                    # Return safe Skip fallback on network error
                    return [
                        ClassificationResult(
                            user_id=item.user_id,
                            post_id=item.post_id,
                            action_char="S",
                            confidence=0.0,
                            logits={"S": 1.0},
                        )
                        for item in items
                    ]
                await asyncio.sleep(0.1 * (2**attempt))

        if not response_data:
            return [
                ClassificationResult(
                    user_id=item.user_id,
                    post_id=item.post_id,
                    action_char="S",
                    confidence=0.0,
                    logits={"S": 1.0},
                )
                for item in items
            ]

        choices = response_data.get("choices", [])
        results: list[ClassificationResult] = []

        for idx, item in enumerate(items):
            if idx < len(choices):
                choice = choices[idx]
                raw_text = choice.get("text", "")
                action_char = JEVPromptBuilder.parse_action_char(raw_text)

                # Extract top_logprobs if available
                logits_dict: dict[str, float] = {}
                logprobs_obj = choice.get("logprobs")
                if logprobs_obj and isinstance(logprobs_obj, dict):
                    top_logprobs_list = logprobs_obj.get("top_logprobs")
                    if top_logprobs_list and len(top_logprobs_list) > 0:
                        first_top = top_logprobs_list[0] or {}
                        for tok_str, lp in first_top.items():
                            clean_tok = tok_str.strip()
                            if clean_tok in {"[", "]", "", ":", "-", ">", "(", ")", "*"}:
                                continue
                            char_candidate = (
                                JEVPromptBuilder.parse_action_char(tok_str)
                            )
                            if (
                                char_candidate in JEVPromptBuilder.VALID_ACTIONS
                                and (
                                    char_candidate not in logits_dict
                                    or lp > logits_dict[char_candidate]
                                )
                            ):
                                logits_dict[char_candidate] = float(lp)

                # Safeguard: if raw_text was unparseable punctuation (e.g. '[' or '('),
                # recover the intended action from the highest logit action in first_top
                if raw_text.strip() in {"[", "]", "", ":", "(", ")", "*", "-"} and logits_dict:
                    best_action = max(logits_dict.items(), key=lambda kv: kv[1])[0]
                    action_char = best_action

                if logits_dict and action_char in logits_dict:
                    probs = compute_softmax(logits_dict)
                    confidence = probs.get(action_char, 0.5)
                elif logits_dict:
                    probs = compute_softmax(logits_dict)
                    confidence = max(probs.values()) if probs else 0.5
                else:
                    logits_dict = {action_char: 1.0}
                    confidence = 1.0

                results.append(
                    ClassificationResult(
                        user_id=item.user_id,
                        post_id=item.post_id,
                        action_char=action_char,
                        confidence=confidence,
                        logits=logits_dict,
                    )
                )
            else:
                results.append(
                    ClassificationResult(
                        user_id=item.user_id,
                        post_id=item.post_id,
                        action_char="S",
                        confidence=0.0,
                        logits={"S": 1.0},
                    )
                )

        if logger.isEnabledFor(logging.INFO) and choices:
            sample_texts = [
                choices[i].get("text", "")
                for i in range(min(5, len(choices)))
            ]
            action_dist = defaultdict(int)
            for r in results:
                action_dist[r.action_char] += 1
            logger.info(
                f"[JEV vLLM Batch] Evaluated {len(items)} items | "
                f"Raw outputs sample: {sample_texts!r} | "
                f"Actions: {dict(action_dist)}"
            )

        # Trigger secondary comment / quote worker if requested
        if generate_comments or self.auto_generate_comments:
            comment_indices = [
                i for i, r in enumerate(results) if r.action_char == "C"
            ]
            if comment_indices:
                requests = [
                    (items[i].full_prompt, items[i].post_content)
                    for i in comment_indices
                ]
                comments = await self.generate_comments_batch(requests)
                for res_idx, comment in zip(comment_indices, comments):
                    results[res_idx].comment_text = comment

            quote_indices = [
                i for i, r in enumerate(results) if r.action_char == "Q"
            ]
            if quote_indices:
                q_requests = [
                    (items[i].full_prompt, items[i].post_content)
                    for i in quote_indices
                ]
                quotes = await self.generate_quotes_batch(q_requests)
                for res_idx, quote in zip(quote_indices, quotes):
                    results[res_idx].quote_text = quote

        return results

    async def classify_batch_generative(
        self,
        items: list[EvalItem],
        allowed_chars: list | None = None,
        generate_comments: bool = False,
    ) -> list[ClassificationResult]:
        """NATIVE-GENERATION per-post path: instead of masking a single action
        LETTER (classification), the model GENERATES the action as multi-token
        guided-JSON output -- {"action": "repost"} -- per (agent, post). This is
        the faithful analogue of classic OASIS's tool-call generation, where the
        model deliberately composes `repost(...)` rather than having a single
        next-token read off. k=1 classic showed the model reposts an isolated
        post readily when it GENERATES the act; single-token classification does
        not capture that. This path tests whether generation (not the feed, not
        the token mask) is the axis.

        Keeps the per-post structure (one call per agent-post) so item-level
        concurrency and the eval pipeline downstream are unchanged. Uses the
        action NAME enum (like_post/repost/...) so the model composes a word, not
        a letter. Returns ClassificationResult list in item order.

        Args:
            items: EvalItem list (full_prompt already assembled per post).
            allowed_chars: enabled action letters; mapped to tool names for the
                           JSON enum via JEVPromptBuilder.ACTION_DESCRIPTIONS.
            generate_comments: trigger comment/quote workers for C/Q choices.

        Returns:
            list[ClassificationResult] in corresponding order.
        """
        if not items:
            return []

        chars = [c for c in (allowed_chars or ["L", "R", "F", "S"]) if c]
        if "S" not in chars:
            chars.append("S")
        # name<->char maps from the single source of truth (ACTION_DESCRIPTIONS).
        name_by_char = {
            c: JEVPromptBuilder.ACTION_DESCRIPTIONS[c][0]
            for c in chars if c in JEVPromptBuilder.ACTION_DESCRIPTIONS
        }
        char_by_name = {v: k for k, v in name_by_char.items()}
        action_names = list(name_by_char.values())

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "post_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": action_names},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }
        endpoint = f"{self.base_url}/chat/completions"

        async def _one(idx: int, item: EvalItem) -> ClassificationResult:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": item.full_prompt}],
                "max_tokens": 16,
                "temperature": self.temperature,
                "response_format": schema,
            }
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        content = (data.get("choices", [{}])[0]
                                   .get("message", {}).get("content", ""))
                        name = str(json.loads(content).get("action", "")).strip()
                        ch = char_by_name.get(name, "S")
                        return ClassificationResult(
                            user_id=item.user_id, post_id=item.post_id,
                            action_char=ch, confidence=1.0, logits={ch: 1.0},
                        )
                    elif resp.status_code == 400:
                        # drop guided schema and retry free-form once
                        payload.pop("response_format", None)
                    else:
                        logger.warning(
                            "generative /chat returned HTTP %d: %s",
                            resp.status_code, resp.text[:160])
                except Exception as e:  # noqa: BLE001
                    if attempt == self.max_retries:
                        logger.debug(
                            "classify_batch_generative failed (uid=%s pid=%s): %s",
                            item.user_id, item.post_id, e)
                    await asyncio.sleep(0.1 * (2 ** attempt))
            return ClassificationResult(
                user_id=item.user_id, post_id=item.post_id,
                action_char="S", confidence=0.0, logits={"S": 1.0},
            )

        results = await asyncio.gather(*[
            _one(i, it) for i, it in enumerate(items)
        ])

        if generate_comments or self.auto_generate_comments:
            for res, it in zip(results, items):
                if res.action_char == "C" and not res.comment_text:
                    res.comment_text = await self.generate_comment(
                        it.full_prompt, it.post_content)
                elif res.action_char == "Q" and not res.quote_text:
                    res.quote_text = await self.generate_quote(
                        it.full_prompt, it.post_content)

        return list(results)

    async def classify_feed(
        self,
        feed_items: list,
        allowed_chars: list | None = None,
    ) -> list:
        """WHOLE-FEED variant: one guided-JSON call per agent over their ENTIRE
        feed, instead of one 1-token call per (agent,post). This mirrors base
        OASIS classic (agent sees the whole feed and SELECTS one post + action),
        testing whether comparative/feed-level context restores reposting that
        the per-post isolated framing suppresses.

        CACHING NOTE: loses the per-post cross-agent prefix sharing (each agent's
        feed is a distinct post set); only the root TASK instruction stays
        shared. This is the deliberate speed/fidelity trade of this experiment.

        Args:
            feed_items: list of (user_id, feed_prompt, valid_post_ids) tuples.
            allowed_chars: enabled action letters (for the JSON enum).

        Returns:
            list of (user_id, chosen_post_id|None, action_char) tuples.
        """
        if not feed_items:
            return []
        actions = [c for c in (allowed_chars or ["L", "R", "F", "S"]) if c]
        if "S" not in actions:
            actions.append("S")
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "feed_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "post_id": {"type": "integer"},
                        "action": {"type": "string", "enum": actions},
                    },
                    "required": ["post_id", "action"],
                    "additionalProperties": False,
                },
            },
        }
        endpoint = f"{self.base_url}/chat/completions"

        async def _one(user_id, prompt, valid_ids):
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": self.temperature,
                "response_format": schema,
            }
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        content = (data.get("choices", [{}])[0]
                                   .get("message", {}).get("content", ""))
                        obj = json.loads(content)
                        pid = obj.get("post_id")
                        act = str(obj.get("action", "S")).strip().upper()[:1]
                        if act not in actions:
                            act = "S"
                        if valid_ids and pid not in valid_ids:
                            pid = valid_ids[0] if valid_ids else None
                        return (user_id, pid, act)
                    elif resp.status_code == 400:
                        payload.pop("response_format", None)
                except Exception as e:  # noqa: BLE001
                    if attempt == self.max_retries:
                        logger.debug("classify_feed failed for %s: %s", user_id, e)
                    await asyncio.sleep(0.1 * (2 ** attempt))
            return (user_id, None, "S")

        return await asyncio.gather(*[
            _one(uid, prompt, vids) for uid, prompt, vids in feed_items
        ])

    # Alias matching plan nomenclature
    classify_actions_batch = classify_batch

    async def _fallback_chat_classify_batch(
        self,
        items: list[EvalItem],
        generate_comments: bool = False,
    ) -> list[ClassificationResult]:
        """Fallback implementation using concurrent /v1/chat/completions requests.

        Enforces strict 1-token logit biasing, defensive action parsing, and robust
        JSON schema grammar fallback to guarantee organic actions are preserved and
        never collapse to 100% Skip ('S').
        """
        if not items:
            return []

        # Enforce logit bias formatting: if raw_logit_bias is unset, build default action bias
        formatted_bias = self._format_logit_bias_payload()
        if not formatted_bias:
            formatted_bias = {}
            for act, tids in DEFAULT_ACTION_TOKEN_MAP.items():
                for tid in tids:
                    formatted_bias[str(tid)] = 50.0

        async def _classify_single(item: EvalItem) -> ClassificationResult:
            endpoint = f"{self.base_url}/chat/completions"

            # 1. Primary Strategy: Chat completions with strict logit biasing
            payload_with_bias: dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": item.full_prompt}],
                "max_tokens": max(4, self.classify_max_tokens),
                "temperature": self.temperature,
                "logit_bias": formatted_bias,
            }

            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._client.post(endpoint, json=payload_with_bias)
                    if resp.status_code == 200:
                        data = resp.json()
                        choice = (
                            data.get("choices", [{}])[0]
                            if data.get("choices")
                            else {}
                        )
                        raw_content = choice.get("message", {}).get("content", "")
                        action = _parse_json_action(raw_content)
                        if action in JEVPromptBuilder.VALID_ACTIONS:
                            confidence = 0.95 if action != "S" else 0.5
                            return ClassificationResult(
                                user_id=item.user_id,
                                post_id=item.post_id,
                                action_char=action,
                                confidence=confidence,
                                logits={action: 1.0},
                            )
                    elif resp.status_code == 400:
                        # 2. Secondary Strategy: Fallback to structured JSON Schema Grammar
                        logger.debug(
                            "Chat completions rejected logit_bias (HTTP 400); falling back to JSON schema grammar"
                        )
                        json_payload: dict[str, Any] = {
                            "model": self.model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a social media interaction classifier. Output a JSON object with the "
                                        "'action' field containing one of: 'L', 'R', 'Q', 'C', 'S'."
                                    ),
                                },
                                {"role": "user", "content": item.full_prompt},
                            ],
                            "max_tokens": 16,
                            "temperature": self.temperature,
                            "response_format": ACTION_JSON_SCHEMA,
                        }
                        resp_json = await self._client.post(endpoint, json=json_payload)
                        if resp_json.status_code == 200:
                            data = resp_json.json()
                            choice = (
                                data.get("choices", [{}])[0]
                                if data.get("choices")
                                else {}
                            )
                            raw_content = choice.get("message", {}).get("content", "")
                            action = _parse_json_action(raw_content)
                            if action in JEVPromptBuilder.VALID_ACTIONS:
                                confidence = 0.95 if action != "S" else 0.5
                                return ClassificationResult(
                                    user_id=item.user_id,
                                    post_id=item.post_id,
                                    action_char=action,
                                    confidence=confidence,
                                    logits={action: 1.0},
                                )

                        # 3. Tertiary Strategy: Standard chat completion without logit_bias or response_format
                        plain_payload: dict[str, Any] = {
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": item.full_prompt}],
                            "max_tokens": max(4, self.classify_max_tokens),
                            "temperature": self.temperature,
                        }
                        resp_plain = await self._client.post(endpoint, json=plain_payload)
                        if resp_plain.status_code == 200:
                            data = resp_plain.json()
                            choice = (
                                data.get("choices", [{}])[0]
                                if data.get("choices")
                                else {}
                            )
                            raw_content = choice.get("message", {}).get("content", "")
                            action = _parse_json_action(raw_content)
                            if action in JEVPromptBuilder.VALID_ACTIONS:
                                confidence = 0.95 if action != "S" else 0.5
                                return ClassificationResult(
                                    user_id=item.user_id,
                                    post_id=item.post_id,
                                    action_char=action,
                                    confidence=confidence,
                                    logits={action: 1.0},
                                )
                except Exception as e:  # noqa: BLE001
                    if attempt == self.max_retries:
                        logger.warning(
                            "Fallback chat classify error for post %d: %s",
                            item.post_id,
                            e,
                        )
                    else:
                        await asyncio.sleep(0.05 * (2**attempt))

            # Defensive safe fallback
            return ClassificationResult(
                user_id=item.user_id,
                post_id=item.post_id,
                action_char="S",
                confidence=0.0,
                logits={"S": 1.0},
            )

        results = await asyncio.gather(*[_classify_single(item) for item in items])

        if generate_comments or self.auto_generate_comments:
            comment_indices = [
                i for i, r in enumerate(results) if r.action_char == "C"
            ]
            if comment_indices:
                requests = [
                    (items[i].full_prompt, items[i].post_content)
                    for i in comment_indices
                ]
                comments = await self.generate_comments_batch(requests)
                for res_idx, comment in zip(comment_indices, comments):
                    results[res_idx].comment_text = comment

            quote_indices = [
                i for i, r in enumerate(results) if r.action_char == "Q"
            ]
            if quote_indices:
                q_requests = [
                    (items[i].full_prompt, items[i].post_content)
                    for i in quote_indices
                ]
                quotes = await self.generate_quotes_batch(q_requests)
                for res_idx, quote in zip(quote_indices, quotes):
                    results[res_idx].quote_text = quote

        return list(results)

    async def generate_comment(
        self,
        user_prompt: str,
        post_content: str,
    ) -> str:
        """Generates a realistic social media comment reacting to a post."""
        clean_user_prompt = user_prompt
        if "[TASK]:" in clean_user_prompt:
            clean_user_prompt = clean_user_prompt.split("[TASK]:")[0].strip()

        async with self._comment_semaphore:
            endpoint = f"{self.base_url}/chat/completions"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are simulating an authentic social media user. "
                        "Write a single concise comment (1-2 sentences) reacting to the post. "
                        "Output ONLY the comment text without quotes or explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{clean_user_prompt}\n\n"
                        f'Target Post Content: "{post_content}"\n\n'
                        "Comment:"
                    ),
                },
            ]

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.comment_max_tokens,
                "temperature": self.comment_temperature,
            }

            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_comment = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        clean_comment = raw_comment.strip().strip('"\'')
                        return clean_comment if clean_comment else "Interesting post."
                except Exception as e:  # noqa: BLE001
                    if attempt == self.max_retries:
                        logger.warning("Comment generation error: %s", e)
                        return "Interesting post."
                    await asyncio.sleep(0.1 * (2**attempt))

            return "Interesting post."

    async def generate_comments_batch(
        self,
        comment_requests: list[tuple[str, str]],
    ) -> list[str]:
        """Concurrently generates comments for multiple requests."""
        tasks = [
            self.generate_comment(prompt, content)
            for prompt, content in comment_requests
        ]
        return await asyncio.gather(*tasks)

    async def generate_quote(
        self,
        user_prompt: str,
        post_content: str,
    ) -> str:
        """Generates a realistic social media quote commentary reacting to a post."""
        clean_user_prompt = user_prompt
        if "[TASK]:" in clean_user_prompt:
            clean_user_prompt = clean_user_prompt.split("[TASK]:")[0].strip()

        async with self._comment_semaphore:
            endpoint = f"{self.base_url}/chat/completions"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are simulating an authentic social media user quote-tweeting a post. "
                        "Write a single concise quote commentary (1-2 sentences) sharing your perspective. "
                        "Output ONLY the quote commentary text without quotes or explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{clean_user_prompt}\n\n"
                        f'Target Post Content: "{post_content}"\n\n'
                        "Quote Commentary:"
                    ),
                },
            ]

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.comment_max_tokens,
                "temperature": self.comment_temperature,
            }

            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_quote = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        clean_quote = raw_quote.strip().strip('"\'')
                        return clean_quote if clean_quote else "Interesting post, sharing with my followers."
                except Exception as e:  # noqa: BLE001
                    if attempt == self.max_retries:
                        logger.warning("Quote generation error: %s", e)
                        return "Interesting post, sharing with my followers."
                    await asyncio.sleep(0.1 * (2**attempt))

            return "Interesting post, sharing with my followers."

    async def generate_quotes_batch(
        self,
        quote_requests: list[tuple[str, str]],
    ) -> list[str]:
        """Concurrently generates quote commentaries for multiple requests."""
        tasks = [
            self.generate_quote(prompt, content)
            for prompt, content in quote_requests
        ]
        return await asyncio.gather(*tasks)

    async def generate_post(
        self,
        agent_context: str,
        topic: str,
        stance_label: str = "Neutral",
    ) -> str:
        """Generates a realistic social media root post given agent persona and topic."""
        clean_context = agent_context
        if "[TASK]:" in clean_context:
            clean_context = clean_context.split("[TASK]:")[0].strip()

        clean_topic = topic.strip().lstrip("#")
        async with self._comment_semaphore:
            endpoint = f"{self.base_url}/chat/completions"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are simulating an authentic social media user. "
                        f"Write a single concise, engaging original post (1-2 sentences) discussing #{clean_topic}. "
                        f"Reflect your personality, traits, and perspective. Always include the #{clean_topic} hashtag. "
                        "Output ONLY the post text without quotes, explanation, or prefixes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{clean_context}\n\n"
                        f"Topic: #{clean_topic} (Stance: {stance_label})\n\n"
                        "Original Post:"
                    ),
                },
            ]

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.post_max_tokens,
                "temperature": self.comment_temperature,
            }

            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_post = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        clean_post = raw_post.strip().strip('"\'')
                        if clean_post:
                            if f"#{clean_topic}" not in clean_post:
                                clean_post = f"{clean_post} #{clean_topic}"
                            return clean_post
                except Exception as e:  # noqa: BLE001
                    if attempt == self.max_retries:
                        logger.warning("Post generation error: %s", e)
                        return f"Sharing some thoughts on #{clean_topic} today."
                    await asyncio.sleep(0.1 * (2**attempt))

            return f"Sharing some thoughts on #{clean_topic} today."

    async def generate_posts_batch(
        self,
        post_requests: list[tuple[str, str, str]],
    ) -> list[str]:
        """Concurrently generates spontaneous root posts for multiple requests."""
        tasks = [
            self.generate_post(ctx, topic, stance)
            for ctx, topic, stance in post_requests
        ]
        return await asyncio.gather(*tasks)

    def resolve_budget(
        self,
        results: list[ClassificationResult],
        budget: int = 1,
    ) -> list[ClassificationResult]:
        """Applies intra-feed budget resolution to results."""
        return resolve_intra_feed_budget(results, budget=budget)
