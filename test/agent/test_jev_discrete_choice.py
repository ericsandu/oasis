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
"""Comprehensive Test Suite for TypeSafe Jev Discrete Choice Decision Model (A, B, C, D).

Verifies:
1. Dynamic action injection: Task instructions, guided-choice tokens, and decoders
   are dynamically constructed based on the experiment's active actions.
2. 100% Prefix Hash Invariance: Root preamble (Token 0) + post prefix remain byte-identical
   and SHA-256 invariant across candidate agents evaluating the same post (RadixAttention KV cache).
3. Token Map Disjointness: DEFAULT_MCQ_TOKEN_MAP is strictly pairwise disjoint with zero collisions.
4. Robust Option Parsing: Option letters ('A', 'B', 'C', 'D') map dynamically to action characters
   ('R', 'L', 'F', 'S') and semantic names ('repost', 'like_post', 'follow', 'do_nothing').
5. Behavioral Parity: Supportive cascade items produce realistic cascade actions (Repost, Like)
   matching empirical Information Spreading diffusion dynamics.
6. End-to-End Environment Interoperability: Execution through JEVEnvironment step pipeline.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oasis.environment.jev_env import (
    JEVEnvironment,
    JEVExecutionConfig,
    JEVStepResult,
)
from oasis.inference.jev_classifier import (
    DEFAULT_MCQ_TOKEN_MAP,
    ClassificationResult,
    EvalItem,
    MockJEVClassifierClient,
    VLLMJEVClassifierClient,
    validate_action_token_map,
)
from oasis.social_agent.jev_prompt_builder import (
    AgentSuffixData,
    JEVPromptBuilder,
    PostPrefixData,
)


# ==============================================================================
# 1. Dynamic Action Injection Tests
# ==============================================================================

def test_resolve_action_specs_canonical_priority():
    """Verifies that resolve_action_specs orders actions in canonical cascade priority:
    Repost -> Like -> Follow -> Skip -> Quote -> Comment."""
    actions = ["like_post", "repost", "follow", "do_nothing"]
    specs = JEVPromptBuilder.resolve_action_specs(actions)

    chars = [s[0] for s in specs]
    names = [s[1] for s in specs]

    assert chars == ["R", "L", "F", "S"]
    assert names == ["repost", "like_post", "follow", "do_nothing"]


def test_resolve_action_specs_custom_subset():
    """Verifies dynamic resolution for minimal action sets."""
    # Experiment with only like and do_nothing
    specs_like_only = JEVPromptBuilder.resolve_action_specs(["like_post", "do_nothing"])
    chars = [s[0] for s in specs_like_only]
    assert chars == ["L", "S"]

    # Experiment with character symbols
    specs_chars = JEVPromptBuilder.resolve_action_specs(["R", "L", "S"])
    chars_resolved = [s[0] for s in specs_chars]
    assert chars_resolved == ["R", "L", "S"]


def test_get_mcq_mappings_dynamic_assignment():
    """Verifies option letters A, B, C... are dynamically assigned to active specs."""
    actions = ["like_post", "repost", "follow", "do_nothing"]
    letter_to_action, action_to_letter, option_letters = (
        JEVPromptBuilder.get_mcq_mappings(actions)
    )

    assert option_letters == ["A", "B", "C", "D"]
    # Canonical priority: A is Repost, B is Like, C is Follow, D is Skip
    assert letter_to_action["A"] == ("R", "repost")
    assert letter_to_action["B"] == ("L", "like_post")
    assert letter_to_action["C"] == ("F", "follow")
    assert letter_to_action["D"] == ("S", "do_nothing")

    assert action_to_letter["R"] == "A"
    assert action_to_letter["repost"] == "A"
    assert action_to_letter["L"] == "B"
    assert action_to_letter["like_post"] == "B"


def test_build_task_instruction_dynamically_injected():
    """Verifies task instruction renders active MCQ options dynamically into prompt preamble."""
    actions = ["like_post", "repost", "do_nothing"]
    instruction = JEVPromptBuilder.build_task_instruction(actions)

    assert "(A) Repost the post to your followers" in instruction
    assert "(B) Like the post" in instruction
    assert "(C) Do nothing (skip without interacting)" in instruction
    assert "(D)" not in instruction  # Only 3 actions enabled
    assert "choose exactly ONE action and output ONLY its option letter (A, B, C):" in instruction


# ==============================================================================
# 2. 100% Prefix Hash Invariance (RadixAttention KV-Cache Guarantee)
# ==============================================================================

def test_prefix_hash_invariance_across_diverse_agents():
    """Verifies that the shared [Task][Post Prefix] is 100% byte-identical and has identical
    SHA-256 hash across 50+ diverse agents evaluating the same post."""
    post = PostPrefixData(
        post_id=101,
        author_name="@tech_insider",
        topic="neuromorphic",
        content="Groundbreaking neuromorphic chip architecture released today with 10x energy efficiency.",
        num_likes=145,
        num_shares=62,
    )

    allowed_actions = ["like_post", "repost", "follow", "do_nothing"]
    shared_prefix = JEVPromptBuilder.build_shared_prefix(
        post, allowed_chars=allowed_actions
    )
    shared_hash = JEVPromptBuilder.get_shared_prefix_hash(
        post, allowed_chars=allowed_actions
    )

    expected_hash = hashlib.sha256(shared_prefix.encode("utf-8")).hexdigest()
    assert shared_hash == expected_hash

    # Generate 50 diverse agents with varied traits, stances, memory, and audience
    for i in range(50):
        agent = AgentSuffixData(
            user_id=1000 + i,
            user_name=f"agent_{i:03d}",
            mbti=["INTJ", "ENFP", "ISTP", "ESFJ", "INTP"][i % 5],
            country=["US", "DE", "JP", "FR", "RO"][i % 5],
            bio=f"Researcher specialized in domain {i} exploring decentralized algorithms.",
            stance_label=["Supportive", "Neutral", "Skeptical"][i % 3],
            stance_score=float((i % 20 - 10) / 10.0),
            recent_actions=f"liked post {100 + i}" if i % 2 == 0 else "",
            num_followers=10 * i,
            num_follows=5 * i,
        )

        full_prompt = JEVPromptBuilder.assemble_eval_prompt(
            post,
            agent,
            allowed_chars=allowed_actions,
        )

        # 1. Prefix must match exactly at character and byte level
        assert full_prompt.startswith(shared_prefix), f"Agent {i} prompt diverged from shared prefix"

        # 2. Leading slice hash must match shared_hash
        prefix_slice = full_prompt[:len(shared_prefix)]
        slice_hash = hashlib.sha256(prefix_slice.encode("utf-8")).hexdigest()
        assert slice_hash == shared_hash

        # 3. Trailing suffix contains dynamic agent persona and 'Action: ' cue
        suffix_part = full_prompt[len(shared_prefix):]
        assert f"agent_{i:03d}" in suffix_part
        assert "Action: " in suffix_part


# ==============================================================================
# 3. Disjoint Token Map Tests
# ==============================================================================

def test_mcq_token_map_pairwise_disjoint():
    """Verifies that DEFAULT_MCQ_TOKEN_MAP is strictly pairwise disjoint with zero collisions."""
    assert validate_action_token_map(DEFAULT_MCQ_TOKEN_MAP) is True

    seen_tokens: dict[int, str] = {}
    for letter, token_ids in DEFAULT_MCQ_TOKEN_MAP.items():
        assert len(token_ids) > 0, f"Option {letter} has empty token ID list"
        for tid in token_ids:
            assert isinstance(tid, int) and tid > 0
            assert tid not in seen_tokens, (
                f"Collision detected: Token {tid} is assigned to both "
                f"'{seen_tokens[tid]}' and '{letter}'"
            )
            seen_tokens[tid] = letter


# ==============================================================================
# 4. Robust Dynamic Option Parsing Tests
# ==============================================================================

def test_parse_option_letter_to_action_char_and_name():
    """Verifies parsing of single option letters, bracketed letters, and prefixes."""
    allowed = ["like_post", "repost", "follow", "do_nothing"]

    # Option A -> Repost ('R')
    assert JEVPromptBuilder.parse_action_char("A", allowed_chars=allowed) == "R"
    assert JEVPromptBuilder.parse_action_char("(A)", allowed_chars=allowed) == "R"
    assert JEVPromptBuilder.parse_action_char("[A]", allowed_chars=allowed) == "R"
    assert JEVPromptBuilder.parse_action_char("Action: A", allowed_chars=allowed) == "R"
    assert JEVPromptBuilder.parse_action_char("A.", allowed_chars=allowed) == "R"
    assert JEVPromptBuilder.parse_semantic_action("A", allowed_chars=allowed) == "repost"

    # Option B -> Like ('L')
    assert JEVPromptBuilder.parse_action_char("B", allowed_chars=allowed) == "L"
    assert JEVPromptBuilder.parse_action_char("(B)", allowed_chars=allowed) == "L"
    assert JEVPromptBuilder.parse_semantic_action("B", allowed_chars=allowed) == "like_post"

    # Option C -> Follow ('F')
    assert JEVPromptBuilder.parse_action_char("C", allowed_chars=allowed) == "F"
    assert JEVPromptBuilder.parse_semantic_action("C", allowed_chars=allowed) == "follow"

    # Option D -> Skip ('S')
    assert JEVPromptBuilder.parse_action_char("D", allowed_chars=allowed) == "S"
    assert JEVPromptBuilder.parse_semantic_action("D", allowed_chars=allowed) == "do_nothing"

    # Direct word fallback
    assert JEVPromptBuilder.parse_action_char("repost", allowed_chars=allowed) == "R"
    assert JEVPromptBuilder.parse_semantic_action("like", allowed_chars=allowed) == "like_post"


# ==============================================================================
# 5. Mock Classifier Parity & Distribution Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_mock_classifier_cascade_supportive_distribution():
    """Verifies that supportive items in MockJEVClassifierClient produce realistic cascade
    actions (Option A Repost and Option B Like dominance) matching Information Spreading."""
    mock_client = MockJEVClassifierClient(seed=42)
    allowed = ["like_post", "repost", "follow", "do_nothing"]

    items = [
        EvalItem(
            user_id=100 + i,
            post_id=200,
            topic="business",
            full_prompt=(
                "For the post below, choose exactly ONE action and output ONLY its option letter (A, B, C, D):\n"
                "(A) Repost the post to your followers\n"
                "(B) Like the post\n"
                "(C) Follow the post author\n"
                "(D) Do nothing (skip without interacting).\n\n"
                "[POST ID: 200] Author: @tech_ceo | Topic: #business | Likes: 50 | Reposts: 20\n"
                'Content: "Major breakthrough announced."\n\n'
                f"[OBSERVER]: @user_{i} | Traits: INTJ, US | Bio: Researcher\n"
                "[STANCE]: #business: Supportive (+0.80)\n"
                "[AUDIENCE]: I have 100 followers. I have 50 follows.\n"
                "Action: "
            ),
        )
        for i in range(100)
    ]

    results = await mock_client.classify_batch(items, allowed_actions=allowed)
    assert len(results) == 100

    action_counts = {}
    for r in results:
        action_counts[r.action_char] = action_counts.get(r.action_char, 0) + 1
        assert r.action_name in ("repost", "like_post", "follow", "do_nothing", "quote_post", "create_comment")
        # Ensure enriched logits contain option letters and names
        assert "A" in r.logits or "R" in r.logits
        assert "repost" in r.logits or "like_post" in r.logits

    # Reposts should be dominant on supportive cascade post
    assert action_counts.get("R", 0) > 40, f"Expected >40 Reposts, got {action_counts}"
    assert action_counts.get("L", 0) > 20, f"Expected >20 Likes, got {action_counts}"


# ==============================================================================
# 6. VLLM Classifier Client Request Validation
# ==============================================================================

@pytest.mark.asyncio
async def test_vllm_client_discrete_choice_request_payload():
    """Verifies that VLLMJEVClassifierClient properly constructs max_tokens=1 and
    guided_choice=['A', 'B', 'C', 'D'] in the request payload."""
    allowed = ["like_post", "repost", "follow", "do_nothing"]
    client = VLLMJEVClassifierClient(
        base_url="http://mock-vllm:8000/v1",
        model_name="Meta-Llama-3-8B-Instruct",
        temperature=1.0,
        guided_choice_actions=allowed,
    )

    item = EvalItem(
        user_id=1,
        post_id=101,
        topic="tech",
        full_prompt="Pre-rendered prompt with Action: ",
    )

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "text": "A",
                "logprobs": {
                    "top_logprobs": [
                        {"A": -0.2, "B": -1.8, "C": -3.5, "D": -4.0}
                    ]
                },
            }
        ]
    }

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        results = await client.classify_batch([item], allowed_actions=allowed)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]

        # 1. 1-token output
        assert payload["max_tokens"] == 1
        # 2. Guided choice with option letters
        assert payload["guided_choice"] == ["A", "B", "C", "D"]
        assert payload["temperature"] == 1.0

        # 3. Decoded result
        assert len(results) == 1
        res = results[0]
        assert res.action_char == "R"
        assert res.action_name == "repost"
        assert res.confidence > 0.7
        assert "R" in res.logits
        assert "repost" in res.logits


# ==============================================================================
# 7. End-to-End Environment Interoperability
# ==============================================================================

@pytest.mark.asyncio
async def test_jev_environment_discrete_choice_step():
    """Verifies end-to-end step_jev execution with dynamic discrete choice decision model."""
    mock_client = MockJEVClassifierClient(seed=123)
    allowed = ["like_post", "repost", "follow", "do_nothing"]

    config = JEVExecutionConfig(
        classifier_client=mock_client,
        allowed_actions=allowed,
        default_topic="tech",
    )

    env = JEVEnvironment(config=config)

    # Mock agent and feed retrieval
    mock_agent = AsyncMock()
    mock_agent.social_agent_id = 1
    mock_agent.user_name = "alice"
    mock_agent.agent_char = "INTJ"
    mock_agent.country = "US"
    mock_agent.bio = "Tech researcher"
    mock_agent.topic = "tech"

    mock_agent_graph = MagicMock()
    mock_agent_graph.agent_mappings = {1: mock_agent}
    env.agent_graph = mock_agent_graph

    feed_post = {
        "post_id": 500,
        "user_id": 2,
        "content": "Exciting AI breakthrough today!",
        "topic": "tech",
        "num_likes": 10,
        "num_shares": 5,
        "stance": 0.85,
    }

    with patch.object(env, "_retrieve_agent_feed", new_callable=AsyncMock, return_value=[feed_post]), \
         patch.object(env, "_fetch_follower_map", return_value={1: (10, 5)}):

        result = await env.step_jev(step_index=1, active_agent_ids=[1])

        assert isinstance(result, JEVStepResult)
        assert result.step_index == 1
        assert result.total_evaluations == 1
        assert sum(result.action_counts.values()) >= 1
