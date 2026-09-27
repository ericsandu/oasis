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
"""Comprehensive unit tests for JEV Inverted Prompt Engine (Track 1) and Batched Logit Classifier (Track 3).

Validates:
1. Byte-for-byte reproducibility of static post prefixes across identical posts.
2. Inverted prompt prefix caching guarantees (prompt.startswith(prefix)).
3. Agent suffix persona, dynamic stance formatting, and Level 2 episodic action log injection.
4. Robust 1-token action parsing ('L', 'R', 'C', 'S').
5. Numerically stable softmax logit normalization.
6. Intra-feed budget resolution (argmax confidence selection and multi-agent grouping).
7. MockJEVClassifierClient deterministic execution and conditional comment generation.
8. VLLMJEVClassifierClient payload formatting, logit bias handling, and secondary comment worker.
9. AST guardrails enforcing zero sqlite3 imports and zero raw SQL statements.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Enforce resource caps
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ[var] = "4"

from oasis.inference.jev_classifier import (
    ClassificationResult,
    EvalItem,
    JEVClassifierClient,
    MockJEVClassifierClient,
    VLLMJEVClassifierClient,
    compute_softmax,
    resolve_intra_feed_budget,
    select_best_action,
)
from oasis.social_agent.jev_prompt_builder import (
    AgentSuffixData,
    JEVPromptBuilder,
    PostPrefixData,
)

# ==============================================================================
# Track 1: JEV Inverted Prompt Builder Tests
# ==============================================================================


class TestJEVPromptBuilder:
    """Test suite for JEVPromptBuilder and Inverted Prefix KV-caching serialization."""

    def test_post_prefix_exact_byte_reproducibility(self) -> None:
        """Verify that identical post data yields 100% byte-for-byte identical prefix strings."""
        post1 = PostPrefixData(
            post_id=42,
            author_name="alice_wonder",
            topic="ai_alignment",
            content="Breakthrough in constitutional AI governance frameworks.",
        )
        post2 = PostPrefixData(
            post_id=42,
            author_name="@alice_wonder",  # leading @ should be normalized
            topic="#ai_alignment",  # leading # should be normalized
            content="Breakthrough in constitutional AI governance frameworks.",
        )

        prefix1 = JEVPromptBuilder.build_post_prefix(post1)
        prefix2 = JEVPromptBuilder.build_post_prefix(post2)

        # Byte-level equality
        assert prefix1.encode("utf-8") == prefix2.encode("utf-8")
        assert prefix1 == prefix2
        assert prefix1.endswith("\n\n")
        assert "[POST ID: 42] Author: @alice_wonder | Topic: #ai_alignment\n" in prefix1
        assert 'Content: "Breakthrough in constitutional AI governance frameworks."\n\n' in prefix1

    def test_post_prefix_alias_build_inverted_prefix(self) -> None:
        """Verify build_inverted_prefix is an exact alias for build_post_prefix."""
        post = PostPrefixData(
            post_id=101,
            author_name="tech_lead",
            topic="crypto",
            content="Decentralized consensus protocols.",
        )
        assert JEVPromptBuilder.build_inverted_prefix(post) == JEVPromptBuilder.build_post_prefix(post)

    def test_post_prefix_with_quote_and_metrics(self) -> None:
        """Verify post prefix formatting puts original post first for KV cache reuse, followed by quote."""
        quote_post = PostPrefixData(
            post_id=45,
            author_name="reviewer_01",
            topic="safety",
            content="Original AI safety framework.",
            quote_content="Essential read for all researchers.",
            original_author="original_author_42",
            original_post_id=42,
            num_likes=12,
            num_shares=3,
        )
        prefix = JEVPromptBuilder.build_post_prefix(quote_post)
        assert prefix.startswith(
            "[POST ID: 42] Author: @original_author_42 | Topic: #safety\n"
            'Content: "Original AI safety framework."\n\n'
        )
        assert "[QUOTE POST ID: 45] Author: @reviewer_01 | Topic: #safety | Likes: 12 | Reposts: 3\n" in prefix
        assert 'Quote Commentary: "Essential read for all researchers."\n\n' in prefix

    def test_quote_post_prefix_starts_with_original_post_prefix(self) -> None:
        """Verify quote post prefix strictly starts with original post prefix for 100% RadixAttention KV hit."""
        orig_post = PostPrefixData(
            post_id=101,
            author_name="lead_dev",
            topic="performance",
            content="vLLM prefix caching cuts latency by 80%.",
        )
        orig_prefix = JEVPromptBuilder.build_post_prefix(orig_post)

        quote_post = PostPrefixData(
            post_id=202,
            author_name="evaluator_99",
            topic="performance",
            content="vLLM prefix caching cuts latency by 80%.",
            quote_content="Confirmed in our latest cluster benchmarks!",
            original_author="lead_dev",
            original_post_id=101,
            num_likes=5,
            num_shares=2,
        )
        quote_prefix = JEVPromptBuilder.build_post_prefix(quote_post)

        # Crucial architectural proof: quote_prefix starts with the exact byte sequence of orig_prefix
        assert quote_prefix.startswith(orig_prefix)

    def test_agent_suffix_formatting_and_stance(self) -> None:
        """Verify agent suffix formatting including persona, traits, and formatted stance score."""
        agent = AgentSuffixData(
            user_id=1,
            user_name="bob_analyst",
            mbti="INTJ",
            country="US",
            bio="Data scientist exploring synthetic media.",
            stance_label="Supportive",
            stance_score=0.82,
        )

        suffix = JEVPromptBuilder.build_agent_suffix(agent, topic="ai_alignment")

        assert "[OBSERVER]: @bob_analyst | Traits: INTJ, US | Bio: Data scientist exploring synthetic media." in suffix
        assert "[STANCE]: #ai_alignment: Supportive (+0.82)" in suffix
        assert (
            "[TASK]: Choose single reaction: L (Like), R (Repost), Q (Quote), C (Comment), S (Skip). Output ONLY the letter."
            in suffix
        )
        assert suffix.endswith("Action: ")

    def test_agent_suffix_negative_and_zero_stance_score(self) -> None:
        """Verify signed float formatting for negative and neutral stance scores."""
        agent_neg = AgentSuffixData(
            user_id=2,
            user_name="@skeptic_user",
            mbti="ENTP",
            country="DE",
            bio="Security auditor",
            stance_label="Skeptical",
            stance_score=-0.45,
        )
        suffix_neg = JEVPromptBuilder.build_agent_suffix(agent_neg, topic="malware")
        assert "[STANCE]: #malware: Skeptical (-0.45)" in suffix_neg

        agent_zero = AgentSuffixData(
            user_id=3,
            user_name="neutral_bot",
            mbti="ISTJ",
            country="JP",
            bio="Archivist",
            stance_label="Neutral",
            stance_score=0.0,
        )
        suffix_zero = JEVPromptBuilder.build_agent_suffix(agent_zero, topic="news")
        assert "[STANCE]: #news: Neutral (+0.00)" in suffix_zero

    def test_agent_suffix_with_recent_actions_memory(self) -> None:
        """Verify Level 2 episodic memory insertion into agent suffix."""
        agent = AgentSuffixData(
            user_id=4,
            user_name="active_voter",
            mbti="INFJ",
            country="FR",
            bio="Community organizer",
            stance_label="Supportive",
            stance_score=0.60,
            recent_actions="Liked P10 (#tech), Reposted P15 (#ai)",
        )
        suffix = JEVPromptBuilder.build_agent_suffix(agent, topic="tech")
        assert "[RECENT ACTIONS]: Liked P10 (#tech), Reposted P15 (#ai)" in suffix

        # Test stripping of 'Recent:' prefix if present
        agent_with_prefix = AgentSuffixData(
            user_id=5,
            user_name="user5",
            mbti="INFP",
            country="UK",
            bio="Writer",
            stance_label="Neutral",
            stance_score=0.05,
            recent_actions="Recent: Commented on P99",
        )
        suffix_p = JEVPromptBuilder.build_agent_suffix(agent_with_prefix)
        assert "[RECENT ACTIONS]: Commented on P99" in suffix_p

    def test_assemble_eval_prompt_radix_cache_prefix_hit_invariant(self) -> None:
        """Verify that prompts assembled for 50 distinct agents share the exact identical prefix slice."""
        post = PostPrefixData(
            post_id=999,
            author_name="viral_source",
            topic="breaking_tech",
            content="New frontier AI model weights released publicly under Apache 2.0.",
        )
        static_prefix = JEVPromptBuilder.build_post_prefix(post)

        assembled_prompts: list[str] = []
        for i in range(50):
            agent = AgentSuffixData(
                user_id=i,
                user_name=f"agent_{i:03d}",
                mbti="INTJ" if i % 2 == 0 else "ENFP",
                country="US" if i % 3 == 0 else "EU",
                bio=f"Persona description for agent {i}",
                stance_label="Supportive" if i % 2 == 0 else "Skeptical",
                stance_score=0.80 if i % 2 == 0 else -0.30,
            )
            prompt = JEVPromptBuilder.assemble_eval_prompt(post, agent)
            assembled_prompts.append(prompt)

            # Invariant 1: Prompt strictly begins with static_prefix
            assert prompt.startswith(static_prefix)
            # Invariant 2: Prefix slice byte-for-byte identical
            assert prompt[: len(static_prefix)] == static_prefix

        # Invariant 3: All 50 agents share the exact same prefix tokens / bytes
        for p in assembled_prompts:
            assert p[: len(static_prefix)].encode("utf-8") == static_prefix.encode("utf-8")

    @pytest.mark.parametrize(
        ("raw_output", "expected_char"),
        [
            ("L", "L"),
            ("R", "R"),
            ("Q", "Q"),
            ("C", "C"),
            ("S", "S"),
            (" l ", "L"),
            (" r\n", "R"),
            (" q ", "Q"),
            ("Like", "L"),
            ("LIKE", "L"),
            ("Repost", "R"),
            ("Quote", "Q"),
            ("QUOTE", "Q"),
            ("Comment", "C"),
            ("Skip", "S"),
            ("[L]", "L"),
            ("[R]", "R"),
            ("[Q]", "Q"),
            ("[C]", "C"),
            ("[S]", "S"),
            ("Action: L", "L"),
            ("Action: [R]", "R"),
            ("Action: [Q]", "Q"),
            ("Action: Quote", "Q"),
            ("Action: Comment", "C"),
            ("[Action]: S", "S"),
            ("I would choose [L] because I like it", "L"),
            ("unknown nonsense 123", "S"),  # safe fallback to Skip
            ("", "S"),
            ("   ", "S"),
        ],
    )
    def test_parse_action_char_robustness(
        self, raw_output: str, expected_char: str
    ) -> None:
        """Verify robust parsing of raw model outputs into single canonical reaction chars."""
        assert JEVPromptBuilder.parse_action_char(raw_output) == expected_char


# ==============================================================================
# Track 3: JEV Classifier, Softmax & Budget Resolution Tests
# ==============================================================================


class TestJEVClassifierUtilities:
    """Test suite for softmax computation and intra-feed budget resolution."""

    def test_compute_softmax_numerically_stable(self) -> None:
        """Verify compute_softmax produces valid probability distribution and avoids overflow."""
        logits = {"L": 1000.0, "R": 998.0, "C": 995.0, "S": 990.0}
        probs = compute_softmax(logits)

        # Probabilities sum to 1.0
        assert pytest.approx(sum(probs.values()), abs=1e-6) == 1.0
        # Largest logit has highest probability
        assert probs["L"] > probs["R"] > probs["C"] > probs["S"]
        assert probs["L"] > 0.8

    def test_compute_softmax_empty(self) -> None:
        """Verify compute_softmax handles empty dictionary gracefully."""
        assert compute_softmax({}) == {}

    def test_resolve_intra_feed_budget_single_user(self) -> None:
        """Verify intra-feed budget selects highest confidence non-skip action."""
        results = [
            ClassificationResult(user_id=1, post_id=10, action_char="L", confidence=0.75, logits={"L": 0.75, "S": 0.25}),
            ClassificationResult(user_id=1, post_id=20, action_char="C", confidence=0.92, logits={"C": 0.92, "S": 0.08}),
            ClassificationResult(user_id=1, post_id=30, action_char="S", confidence=0.88, logits={"S": 0.88, "L": 0.12}),
            ClassificationResult(user_id=1, post_id=40, action_char="L", confidence=0.60, logits={"L": 0.60, "S": 0.40}),
        ]

        # Budget = 1: Post 20 (C with 0.92) is highest non-skip; Post 10 and 40 should be downgraded to S
        resolved = resolve_intra_feed_budget(results, budget=1, downgrade_to_skip=True)

        assert len(resolved) == 4
        # Post 10 downgraded to S
        assert resolved[0].post_id == 10 and resolved[0].action_char == "S"
        # Post 20 retained as C
        assert resolved[1].post_id == 20 and resolved[1].action_char == "C" and resolved[1].confidence == 0.92
        # Post 30 kept as original S
        assert resolved[2].post_id == 30 and resolved[2].action_char == "S"
        # Post 40 downgraded to S
        assert resolved[3].post_id == 40 and resolved[3].action_char == "S"

    def test_resolve_intra_feed_budget_all_skips(self) -> None:
        """Verify budget resolver handles feeds where all items are 'S'."""
        results = [
            ClassificationResult(user_id=1, post_id=1, action_char="S", confidence=0.9, logits={"S": 0.9}),
            ClassificationResult(user_id=1, post_id=2, action_char="S", confidence=0.8, logits={"S": 0.8}),
        ]
        resolved = resolve_intra_feed_budget(results, budget=1)
        assert len(resolved) == 2
        assert all(r.action_char == "S" for r in resolved)

    def test_resolve_intra_feed_budget_multiple_users(self) -> None:
        """Verify multi-user batch preserves independent budget resolution per agent."""
        results = [
            # User 1 items
            ClassificationResult(user_id=1, post_id=10, action_char="L", confidence=0.70, logits={"L": 0.70, "S": 0.30}),
            ClassificationResult(user_id=1, post_id=20, action_char="L", confidence=0.85, logits={"L": 0.85, "S": 0.15}),
            # User 2 items
            ClassificationResult(user_id=2, post_id=10, action_char="R", confidence=0.90, logits={"R": 0.90, "S": 0.10}),
            ClassificationResult(user_id=2, post_id=30, action_char="L", confidence=0.95, logits={"L": 0.95, "S": 0.05}),
        ]

        resolved = resolve_intra_feed_budget(results, budget=1, downgrade_to_skip=True)

        assert len(resolved) == 4
        # User 1: Post 20 wins (0.85 > 0.70)
        assert resolved[0].action_char == "S"
        assert resolved[1].action_char == "L" and resolved[1].post_id == 20
        # User 2: Post 30 wins (0.95 > 0.90)
        assert resolved[2].action_char == "S"
        assert resolved[3].action_char == "L" and resolved[3].post_id == 30

    def test_select_best_action(self) -> None:
        """Verify select_best_action picks top non-skip item or returns None."""
        results = [
            ClassificationResult(user_id=1, post_id=1, action_char="S", confidence=0.99, logits={"S": 0.99}),
            ClassificationResult(user_id=1, post_id=2, action_char="L", confidence=0.72, logits={"L": 0.72}),
            ClassificationResult(user_id=1, post_id=3, action_char="C", confidence=0.88, logits={"C": 0.88}),
        ]
        best = select_best_action(results)
        assert best is not None
        assert best.post_id == 3
        assert best.action_char == "C"

        all_skip = [
            ClassificationResult(user_id=1, post_id=1, action_char="S", confidence=0.9, logits={"S": 0.9}),
            ClassificationResult(user_id=1, post_id=2, action_char="S", confidence=0.8, logits={"S": 0.8}),
        ]
        assert select_best_action(all_skip) is None


# ==============================================================================
# Track 3: MockJEVClassifierClient Tests
# ==============================================================================


class TestMockJEVClassifierClient:
    """Test suite for hermetic MockJEVClassifierClient."""

    def test_implements_jev_classifier_client_protocol(self) -> None:
        """Verify MockJEVClassifierClient conforms to JEVClassifierClient Protocol."""
        client = MockJEVClassifierClient()
        assert isinstance(client, JEVClassifierClient)

    @pytest.mark.asyncio
    async def test_deterministic_action_mapping(self) -> None:
        """Verify configured action mapping produces exact requested actions."""
        mapping = {
            (1, 101): "L",
            (1, 102): "R",
            (2, 101): "C",
            (2, 102): "S",
        }
        client = MockJEVClassifierClient(action_mapping=mapping)

        items = [
            EvalItem(user_id=1, post_id=101, topic="tech", full_prompt="Prompt 1"),
            EvalItem(user_id=1, post_id=102, topic="tech", full_prompt="Prompt 2"),
            EvalItem(user_id=2, post_id=101, topic="tech", full_prompt="Prompt 3"),
            EvalItem(user_id=2, post_id=102, topic="tech", full_prompt="Prompt 4"),
        ]

        results = await client.classify_batch(items)
        assert len(results) == 4
        assert [r.action_char for r in results] == ["L", "R", "C", "S"]
        assert all(r.confidence > 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_conditional_comment_generation(self) -> None:
        """Verify comment fallback worker triggers when 'C' is emitted and generate_comments=True."""
        client = MockJEVClassifierClient(
            action_mapping={(1, 50): "C", (1, 51): "L"},
            mock_comment="I think this is an important contribution to AI.",
        )

        items = [
            EvalItem(user_id=1, post_id=50, topic="ai", full_prompt="Prompt C", post_content="Paper on safety"),
            EvalItem(user_id=1, post_id=51, topic="ai", full_prompt="Prompt L", post_content="Paper on speed"),
        ]

        results = await client.classify_batch(items, generate_comments=True)
        assert results[0].action_char == "C"
        assert results[0].comment_text is not None
        assert "Paper on safety" in results[0].comment_text or "AI" in results[0].comment_text

        # Item 1 was Like: comment_text should remain None
        assert results[1].action_char == "L"
        assert results[1].comment_text is None

    @pytest.mark.asyncio
    async def test_batch_comment_generation(self) -> None:
        """Verify generate_comments_batch generates comments for multiple requests."""
        client = MockJEVClassifierClient(mock_comment="Nice post!")
        requests = [("user_ctx1", "post content 1"), ("user_ctx2", "post content 2")]
        comments = await client.generate_comments_batch(requests)
        assert len(comments) == 2
        assert all("Nice post!" in c for c in comments)

    @pytest.mark.asyncio
    async def test_conditional_quote_generation(self) -> None:
        """Verify quote fallback worker triggers when 'Q' is emitted and generate_comments=True."""
        client = MockJEVClassifierClient(
            action_mapping={(1, 60): "Q", (1, 61): "S"},
            mock_quote="Great breakthrough in LLM optimization.",
        )

        items = [
            EvalItem(user_id=1, post_id=60, topic="ai", full_prompt="Prompt Q", post_content="Paper on JEV"),
            EvalItem(user_id=1, post_id=61, topic="ai", full_prompt="Prompt S", post_content="Paper on caching"),
        ]

        results = await client.classify_batch(items, generate_comments=True)
        assert results[0].action_char == "Q"
        assert results[0].quote_text is not None
        assert "Paper on JEV" in results[0].quote_text or "Great breakthrough" in results[0].quote_text

        # Item 1 was Skip: quote_text should remain None
        assert results[1].action_char == "S"
        assert results[1].quote_text is None

    @pytest.mark.asyncio
    async def test_batch_quote_generation(self) -> None:
        """Verify generate_quotes_batch generates quotes for multiple requests."""
        client = MockJEVClassifierClient(mock_quote="Compelling quote!")
        requests = [("user_ctx1", "post content 1"), ("user_ctx2", "post content 2")]
        quotes = await client.generate_quotes_batch(requests)
        assert len(quotes) == 2
        assert all("Compelling quote!" in q for q in quotes)


# ==============================================================================
# Track 3: VLLMJEVClassifierClient Tests
# ==============================================================================


class TestVLLMJEVClassifierClient:
    """Test suite for VLLMJEVClassifierClient with mocked HTTP interactions."""

    def test_implements_jev_classifier_client_protocol(self) -> None:
        """Verify VLLMJEVClassifierClient conforms to JEVClassifierClient Protocol."""
        client = VLLMJEVClassifierClient(base_url="http://mock-vllm:8000/v1")
        assert isinstance(client, JEVClassifierClient)

    @pytest.mark.asyncio
    async def test_classify_batch_vllm_completions(self) -> None:
        """Verify VLLMJEVClassifierClient correctly queries /v1/completions with max_tokens=1 and parses logprobs."""
        mock_response_json = {
            "choices": [
                {
                    "index": 0,
                    "text": "L",
                    "logprobs": {
                        "top_logprobs": [
                            {"L": -0.05, "S": -3.5, "C": -4.0, "R": -5.0}
                        ]
                    },
                },
                {
                    "index": 1,
                    "text": "S",
                    "logprobs": {
                        "top_logprobs": [
                            {"S": -0.10, "L": -2.8, "C": -4.2, "R": -6.0}
                        ]
                    },
                },
            ]
        }

        mock_http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_json
        mock_http_client.post.return_value = mock_resp

        client = VLLMJEVClassifierClient(
            base_url="http://mock-vllm:8000/v1",
            model_name="Qwen/Qwen2.5-32B-Instruct",
            client=mock_http_client,
        )

        items = [
            EvalItem(user_id=1, post_id=10, topic="tech", full_prompt="Post 10 Prompt"),
            EvalItem(user_id=1, post_id=20, topic="tech", full_prompt="Post 20 Prompt"),
        ]

        results = await client.classify_batch(items)

        # Assert correct request structure was dispatched
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        endpoint = call_args[0][0]
        payload = call_args[1]["json"]

        assert endpoint == "http://mock-vllm:8000/v1/completions"
        assert payload["model"] == "Qwen/Qwen2.5-32B-Instruct"
        assert payload["prompt"] == ["Post 10 Prompt", "Post 20 Prompt"]
        assert payload["max_tokens"] == 1
        assert payload["logprobs"] == 5

        # Assert results were parsed accurately
        assert len(results) == 2
        assert results[0].action_char == "L"
        assert results[0].confidence > 0.85
        assert results[1].action_char == "S"
        assert results[1].confidence > 0.85

    @pytest.mark.asyncio
    async def test_classify_batch_bracket_recovery_safeguard(self) -> None:
        """Verify that when raw_text is '[' (e.g. from 1-token output), the intended action is recovered from logprobs."""
        mock_response_json = {
            "choices": [
                {
                    "index": 0,
                    "text": "[",
                    "logprobs": {
                        "top_logprobs": [
                            {"[": -0.1, "L": -1.2, "S": -2.5, "C": -4.0}
                        ]
                    },
                }
            ]
        }
        mock_http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_json
        mock_http_client.post.return_value = mock_resp

        client = VLLMJEVClassifierClient(
            base_url="http://mock-vllm:8000/v1",
            client=mock_http_client,
        )
        items = [EvalItem(user_id=1, post_id=10, topic="tech", full_prompt="Post Prompt")]
        results = await client.classify_batch(items)
        assert len(results) == 1
        assert results[0].action_char == "L"
        assert results[0].confidence > 0.70

    @pytest.mark.asyncio
    async def test_classify_batch_logit_bias_formatting(self) -> None:
        """Verify numeric and token_id_map logit bias keys are converted to string token IDs in payload."""
        mock_http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"text": "L", "logprobs": {}}]
        }
        mock_http_client.post.return_value = mock_resp

        # token ID map: L -> 44, R -> 55, C -> 66, S -> 77
        token_map = {"L": 44, "R": 55, "C": 66, "S": 77}
        logit_bias = {"L": 100.0, "R": 100.0, "C": 100.0, "S": 100.0}

        client = VLLMJEVClassifierClient(
            base_url="http://mock-vllm:8000/v1",
            logit_bias=logit_bias,
            token_id_map=token_map,
            client=mock_http_client,
        )

        items = [EvalItem(user_id=1, post_id=1, topic="tech", full_prompt="P1")]
        await client.classify_batch(items)

        call_args = mock_http_client.post.call_args
        payload = call_args[1]["json"]
        assert "logit_bias" in payload
        assert payload["logit_bias"] == {"44": 100.0, "55": 100.0, "66": 100.0, "77": 100.0}

    @pytest.mark.asyncio
    async def test_classify_batch_auto_discover_token_ids(self) -> None:
        """Verify client queries /tokenize endpoint to discover action token IDs and applies logit bias."""
        mock_http_client = AsyncMock()

        def mock_post_handler(url: str, **kwargs: Any) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            if "tokenize" in url:
                payload = kwargs.get("json", {})
                prompt_str = payload.get("prompt", "")
                token_map = {"L": [43], "R": [49], "Q": [48], "C": [34], "S": [50]}
                clean_char = prompt_str.strip().upper()
                resp.json.return_value = {"tokens": token_map.get(clean_char, [999])}
            else:
                resp.json.return_value = {
                    "choices": [{"text": "L", "logprobs": {}}]
                }
            return resp

        mock_http_client.post.side_effect = mock_post_handler

        client = VLLMJEVClassifierClient(
            base_url="http://mock-vllm:8000/v1",
            auto_discover_token_ids=True,
            client=mock_http_client,
        )

        items = [EvalItem(user_id=1, post_id=10, topic="tech", full_prompt="Post Prompt")]
        results = await client.classify_batch(items)

        assert len(results) == 1
        assert results[0].action_char == "L"
        # Verify logit bias was constructed and sent in completions call
        completions_calls = [
            c for c in mock_http_client.post.call_args_list
            if "completions" in c[0][0] and "tokenize" not in c[0][0]
        ]
        assert len(completions_calls) == 1
        c_payload = completions_calls[0][1]["json"]
        assert "logit_bias" in c_payload
        assert c_payload["logit_bias"]["43"] == 50.0

    @pytest.mark.asyncio
    async def test_fallback_to_chat_completions_on_404(self) -> None:
        """Verify client falls back to /chat/completions if /completions returns 404."""
        mock_http_client = AsyncMock()

        # First call to /completions -> 404
        resp_404 = MagicMock()
        resp_404.status_code = 404

        # Subsequent call to /chat/completions -> 200
        resp_chat = MagicMock()
        resp_chat.status_code = 200
        resp_chat.json.return_value = {
            "choices": [{"message": {"content": "R"}}]
        }

        mock_http_client.post.side_effect = [resp_404, resp_chat]

        client = VLLMJEVClassifierClient(
            base_url="http://mock-vllm:8000/v1",
            client=mock_http_client,
        )

        items = [EvalItem(user_id=1, post_id=10, topic="tech", full_prompt="Chat fallback test")]
        results = await client.classify_batch(items)

        assert len(results) == 1
        assert results[0].action_char == "R"

    @pytest.mark.asyncio
    async def test_secondary_comment_worker_trigger(self) -> None:
        """Verify secondary comment generation worker is invoked for 'C' action choices."""
        mock_http_client = AsyncMock()

        # Completions response returns 'C'
        comp_resp = MagicMock()
        comp_resp.status_code = 200
        comp_resp.json.return_value = {
            "choices": [{"text": "C", "logprobs": {}}]
        }

        # Chat completions response for comment worker returns generated comment
        chat_resp = MagicMock()
        chat_resp.status_code = 200
        chat_resp.json.return_value = {
            "choices": [{"message": {"content": "This is a profound insight on decentralized agents."}}]
        }

        mock_http_client.post.side_effect = [comp_resp, chat_resp]

        client = VLLMJEVClassifierClient(
            base_url="http://mock-vllm:8000/v1",
            client=mock_http_client,
        )

        items = [
            EvalItem(
                user_id=1,
                post_id=42,
                topic="ai",
                full_prompt="Prompt for C",
                post_content="Decentralized multi-agent consensus",
            )
        ]

        results = await client.classify_batch(items, generate_comments=True)

        assert len(results) == 1
        assert results[0].action_char == "C"
        assert results[0].comment_text == "This is a profound insight on decentralized agents."

    @pytest.mark.asyncio
    async def test_aclose_and_context_manager(self) -> None:
        """Verify proper async client lifecycle cleanup."""
        mock_client = AsyncMock()
        mock_client.is_closed = False

        client = VLLMJEVClassifierClient(client=mock_client)
        client._own_client = True  # simulate ownership
        await client.aclose()
        mock_client.aclose.assert_awaited_once()


# ==============================================================================
# Architectural AST Guardrail Tests
# ==============================================================================


class TestASTGuardrailsForTrack1And3:
    """Enforces zero sqlite3 imports and zero raw SQL statements in JEV engine files."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "oasis/social_agent/jev_prompt_builder.py",
            "oasis/social_agent/belief_state.py",
            "oasis/inference/jev_classifier.py",
        ],
    )
    def test_no_sqlite3_import_in_jev_files(self, rel_path: str) -> None:
        """Verify that Track 1 & 3 modules strictly do not import sqlite3."""
        oasis_root = Path(__file__).resolve().parents[2]
        file_path = oasis_root / rel_path
        assert file_path.exists(), f"Target file {file_path} does not exist."

        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "sqlite3", (
                        f"Forbidden 'import sqlite3' found in {rel_path}:{node.lineno}."
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "sqlite3" and not (node.module or "").startswith("sqlite3."), (
                    f"Forbidden 'from sqlite3 import ...' found in {rel_path}:{node.lineno}."
                )

    @pytest.mark.parametrize(
        "rel_path",
        [
            "oasis/social_agent/jev_prompt_builder.py",
            "oasis/social_agent/belief_state.py",
            "oasis/inference/jev_classifier.py",
        ],
    )
    def test_no_raw_sql_queries_in_jev_files(self, rel_path: str) -> None:
        """Verify that no raw SQL statements exist in JEV engine files."""
        oasis_root = Path(__file__).resolve().parents[2]
        file_path = oasis_root / rel_path

        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        forbidden_sql = [
            "SELECT ", "INSERT INTO ", "DELETE FROM ", "UPDATE ", "DROP TABLE", "CREATE TABLE"
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val_upper = node.value.strip().upper()
                for fragment in forbidden_sql:
                    assert fragment not in val_upper, (
                        f"Forbidden raw SQL query literal '{fragment}' detected in {rel_path}:{node.lineno}."
                    )
