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
import logging
import math
import os
import re
from collections import defaultdict
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
        action_char: Predicted interaction choice ('L', 'R', 'C', 'S').
        confidence: Probability score in [0.0, 1.0] for the chosen action.
        logits: Logit / probability distribution across {'L', 'R', 'C', 'S'}.
        comment_text: Generated comment text if action_char == 'C' and comment worker ran.
    """

    user_id: int
    post_id: int
    action_char: str
    confidence: float
    logits: dict[str, float] = field(default_factory=dict)
    comment_text: str | None = None


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
        default_confidence: float = 0.95,
        seed: int = 42,
    ) -> None:
        """Initializes the mock classifier.

        Args:
            action_mapping: Mapping of (user_id, post_id) -> action character ('L', 'R', 'C', 'S').
            default_action: Global fallback action if pair not in action_mapping.
            mock_comment: Default comment text returned for comment generation requests.
            default_confidence: Default confidence assigned to deterministic choices.
            seed: Seed value used for deterministic fallback generation.
        """
        self.action_mapping = dict(action_mapping or {})
        self.default_action = default_action
        self.mock_comment = mock_comment
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
            logits = {"L": 0.05, "R": 0.05, "C": 0.05, "S": 0.05}
            logits[char] = 4.0
            probs = compute_softmax(logits)
            return char, probs[char], probs

        # 2. Fixed default action override
        if self.default_action is not None:
            char = self.default_action
            logits = {"L": 0.05, "R": 0.05, "C": 0.05, "S": 0.05}
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
            # Positive stance: strongly biased to Like, occasionally Repost or Comment
            if h < 10:
                char = "C"
            elif h < 30:
                char = "R"
            elif h < 90:
                char = "L"
            else:
                char = "S"
            base_logits = {"L": 3.0, "R": 1.5, "C": 1.2, "S": -0.5}
        elif stance_score < -0.3 or "Hostile" in prompt or "Skeptical" in prompt:
            # Negative stance: biased to Skip, occasionally critical Comment
            if h < 15:
                char = "C"
            else:
                char = "S"
            base_logits = {"L": -1.0, "R": -1.5, "C": 1.0, "S": 3.5}
        else:
            # Neutral stance
            if h < 40:
                char = "L"
            elif h < 50:
                char = "R"
            elif h < 60:
                char = "C"
            else:
                char = "S"
            base_logits = {"L": 1.0, "R": 0.8, "C": 0.8, "S": 1.5}

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
            if generate_comments and char == "C":
                comment_text = await self.generate_comment(
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
                )
            )

        return results

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
        auto_generate_comments: bool = False,
        max_retries: int = 2,
        comment_temperature: float = 0.7,
        comment_max_tokens: int = 64,
        max_concurrent_comments: int = 8,
    ) -> None:
        """Initializes the vLLM classifier client.

        Args:
            base_url: vLLM OpenAI-compatible base URL (default: VLLM_BASE_URL env or 127.0.0.1:8000/v1).
            model_name: Registered model identifier (default: VLLM_MODEL env).
            logit_bias: Optional logit bias dictionary passed to vLLM.
            token_id_map: Optional mapping from action chars ('L', 'R', 'C', 'S') to vocab token IDs.
            timeout: HTTP request timeout in seconds.
            client: Optional shared httpx.AsyncClient instance.
            temperature: Sampling temperature for 1-token classification (0.0 = greedy).
            auto_generate_comments: If True, automatically trigger comment generation on 'C'.
            max_retries: Max retry attempts on transient HTTP/connection errors.
            comment_temperature: Temperature for secondary comment generation.
            comment_max_tokens: Maximum tokens for secondary comment generation.
            max_concurrent_comments: Semaphore concurrency limit for secondary comment requests.
        """
        raw_url = base_url or DEFAULT_VLLM_URL
        self.base_url = raw_url.rstrip("/")
        self.model_name = model_name or DEFAULT_VLLM_MODEL
        self.raw_logit_bias = logit_bias
        self.token_id_map = dict(token_id_map or {})
        self.timeout = timeout
        self.temperature = temperature
        self.auto_generate_comments = auto_generate_comments
        self.max_retries = max_retries
        self.comment_temperature = comment_temperature
        self.comment_max_tokens = comment_max_tokens
        self._comment_semaphore = asyncio.Semaphore(max_concurrent_comments)

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

    def _format_logit_bias_payload(self) -> dict[str, float] | None:
        """Formats logit bias dictionary into API-compliant string token ID mapping."""
        if not self.raw_logit_bias:
            return None

        formatted: dict[str, float] = {}
        for k, v in self.raw_logit_bias.items():
            if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                formatted[str(k)] = float(v)
            elif isinstance(k, str) and k.upper() in self.token_id_map:
                token_id = self.token_id_map[k.upper()]
                formatted[str(token_id)] = float(v)

        return formatted if formatted else None

    async def classify_batch(
        self,
        items: list[EvalItem],
        generate_comments: bool = False,
    ) -> list[ClassificationResult]:
        """Classifies a batch of EvalItem prompts using vLLM /v1/completions.

        Passes max_tokens=1 and logprobs=5 to perform item-level parallel forward passes
        leveraging vLLM RadixAttention shared prefix KV-cache reuse.
        """
        if not items:
            return []

        prompts = [item.full_prompt for item in items]
        endpoint = f"{self.base_url}/completions"

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": self.temperature,
            "logprobs": 5,
        }

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

                if logits_dict:
                    probs = compute_softmax(logits_dict)
                    confidence = probs.get(action_char, 0.5)
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

        # Trigger secondary comment worker if requested
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

        return results

    # Alias matching plan nomenclature
    classify_actions_batch = classify_batch

    async def _fallback_chat_classify_batch(
        self,
        items: list[EvalItem],
        generate_comments: bool,
    ) -> list[ClassificationResult]:
        """Fallback implementation using concurrent /v1/chat/completions requests."""
        async def _classify_single(item: EvalItem) -> ClassificationResult:
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": item.full_prompt}],
                "max_tokens": 1,
                "temperature": self.temperature,
            }
            try:
                resp = await self._client.post(endpoint, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    action = JEVPromptBuilder.parse_action_char(content)
                    return ClassificationResult(
                        user_id=item.user_id,
                        post_id=item.post_id,
                        action_char=action,
                        confidence=0.9,
                        logits={action: 1.0},
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("Fallback chat classify error for post %d: %s", item.post_id, e)
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

        return list(results)

    async def generate_comment(
        self,
        user_prompt: str,
        post_content: str,
    ) -> str:
        """Generates a realistic social media comment reacting to a post."""
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
                        f"{user_prompt}\n\n"
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

    def resolve_budget(
        self,
        results: list[ClassificationResult],
        budget: int = 1,
    ) -> list[ClassificationResult]:
        """Applies intra-feed budget resolution to results."""
        return resolve_intra_feed_budget(results, budget=budget)
