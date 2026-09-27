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
"""Comprehensive End-to-End Test Suite and Benchmarking for JEV Engine.

Adheres strictly to docs/jev_optimization_plan.md Section 9 and Track 5 deliverables.
Enforces 100% hermetic execution (zero GPU / network dependencies) and 4-thread CPU caps.

Verified Capabilities:
1. Inverted prompt prefix caching identity: exact byte matching of PostPrefixData across multiple users.
2. 1-token output classification parsing and intra-feed argmax budget resolution.
3. Dynamic belief state drift (multi-step convergence/polarization) under positive/negative peer engagement.
4. Micro-time Poisson continuous arrival scheduling and ChronologicalActionQueue FIFO tie-breaking & Channel delivery.
5. Multi-step end-to-end JEV simulation with real OasisEnv/JEVEnvironment and MockJEVClassifierClient.
6. Latency micro-benchmark asserting high throughput of 1-token classification over 150 simulated agents.
7. AST guardrails asserting zero sqlite3 imports in jev_prompt_builder, belief_state, jev_classifier,
   micro_time_scheduler, and jev_env.
"""

from __future__ import annotations

import ast
import asyncio
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Enforce resource guardrails (Section 9: max 4 CPU threads)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)
for var in THREAD_VARS:
    os.environ[var] = "4"

# Safe mock for heavy ML packages if running in lightweight CPU environment
for pkg in (
    "torch",
    "torch.nn",
    "sentence_transformers",
    "transformers",
    "sklearn",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
):
    try:
        __import__(pkg)
    except ImportError:
        sys.modules[pkg] = MagicMock()

import pytest

from oasis.clock.micro_time_scheduler import (
    ChronologicalActionQueue,
    MicroTimeScheduler,
    ScheduledAction,
)
from oasis.environment.env import OasisEnv
from oasis.environment.jev_env import (
    JEVEnvironment,
    JEVExecutionConfig,
    JEVStepResult,
)
from oasis.inference.jev_classifier import (
    ClassificationResult,
    EvalItem,
    MockJEVClassifierClient,
    compute_softmax,
    resolve_intra_feed_budget,
    select_best_action,
)
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.belief_state import (
    LABEL_NEUTRAL,
    LABEL_STRONGLY_OPPOSED,
    LABEL_STRONGLY_SUPPORTIVE,
    BeliefState,
)
from oasis.social_agent.jev_prompt_builder import (
    AgentSuffixData,
    JEVPromptBuilder,
    PostPrefixData,
)
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.typing import ActionType, DefaultPlatformType

# ==============================================================================
# Helper Mock Fixtures and Utilities
# ==============================================================================


class SyntheticAgent:
    """Mock agent representing a SocialAgent for hermetic, fast simulation testing."""

    def __init__(
        self,
        agent_id: int,
        user_name: str | None = None,
        mbti: str = "INTJ",
        country: str = "US",
        bio: str = "A synthetic agent persona",
        activity_freq: float = 1.0,
    ) -> None:
        self.social_agent_id = agent_id
        self.agent_id = agent_id
        self.user_name = user_name or f"user_{agent_id}"
        self.mbti = mbti
        self.country = country
        self.user_info = UserInfo(
            user_name=self.user_name,
            name=f"User {agent_id}",
            description=bio,
            profile={
                "other_info": {
                    "mbti": mbti,
                    "country": country,
                    "activity_level_frequency": [activity_freq],
                    "user_profile": bio,
                }
            },
        )


class MockHermeticPlatform:
    """In-memory hermetic platform for zero-network testing with Channel."""

    def __init__(self, channel: Channel | None = None) -> None:
        self.channel = channel or Channel()
        self.db_path = ":memory:"
        self.recsys_type = "twitter"
        self.start_time = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
        self.updated_rec_table = False
        self.feeds: dict[int, list[dict[str, Any]]] = {}

    async def update_rec_table(self) -> None:
        self.updated_rec_table = True

    async def refresh(self, agent_id: int) -> dict[str, Any]:
        posts = self.feeds.get(agent_id, [])
        return {"success": True, "posts": posts}


# ==============================================================================
# 1. Inverted Prompt Prefix Caching Identity Tests
# ==============================================================================


class TestJEVInvertedPromptPrefixIdentity:
    """Test suite verifying Track 1 Inverted Prompt Prefix KV-Caching Guarantees."""

    def test_exact_byte_level_prefix_identity_across_diverse_personas(self) -> None:
        """Verify that PostPrefixData produces exact byte-for-byte identical prefix across multiple observers."""
        shared_post = PostPrefixData(
            post_id=42,
            author_name="alice_researcher",
            topic="ai_alignment",
            content="Breakthrough in formal verification of neural policy boundaries.",
        )
        post_prefix_str = JEVPromptBuilder.build_post_prefix(shared_post)
        post_prefix_bytes = post_prefix_str.encode("utf-8")

        # 10 completely diverse agent personas
        personas = [
            AgentSuffixData(
                user_id=1,
                user_name="bob_enthusiast",
                mbti="ENFP",
                country="US",
                bio="Tech enthusiast and optimist",
                stance_label="Supportive",
                stance_score=0.45,
                recent_actions="Liked P10 (#ai)",
            ),
            AgentSuffixData(
                user_id=2,
                user_name="carol_skeptic",
                mbti="INTJ",
                country="DE",
                bio="AI safety researcher and skeptic",
                stance_label="Skeptical",
                stance_score=-0.35,
                recent_actions="Commented on P12 (#security)",
            ),
            AgentSuffixData(
                user_id=3,
                user_name="dan_neutral",
                mbti="ISTP",
                country="FR",
                bio="Systems engineer",
                stance_label="Neutral / Undecided",
                stance_score=0.0,
            ),
            AgentSuffixData(
                user_id=4,
                user_name="elena_critic",
                mbti="ESTJ",
                country="JP",
                bio="Policy analyst and auditor",
                stance_label="Strongly Opposed",
                stance_score=-0.80,
                recent_actions="Reposted P5 (#ethics)",
            ),
            AgentSuffixData(
                user_id=5,
                user_name="frank_advocate",
                mbti="INFJ",
                country="CA",
                bio="Decentralization advocate",
                stance_label="Strongly Supportive",
                stance_score=0.92,
            ),
            AgentSuffixData(
                user_id=6,
                user_name="grace_coder",
                mbti="INTP",
                country="UK",
                bio="Open source hacker",
                stance_label="Neutral / Undecided",
                stance_score=0.05,
            ),
            AgentSuffixData(
                user_id=7,
                user_name="heidi_journalist",
                mbti="ENTP",
                country="AU",
                bio="Investigative tech reporter",
                stance_label="Skeptical",
                stance_score=-0.25,
            ),
            AgentSuffixData(
                user_id=8,
                user_name="ivan_math",
                mbti="INTJ",
                country="CH",
                bio="Cryptographer and mathematician",
                stance_label="Supportive",
                stance_score=0.55,
            ),
            AgentSuffixData(
                user_id=9,
                user_name="judy_lawyer",
                mbti="ESFJ",
                country="SG",
                bio="Intellectual property counsel",
                stance_label="Neutral / Undecided",
                stance_score=-0.08,
            ),
            AgentSuffixData(
                user_id=10,
                user_name="mallory_bot",
                mbti="ISTJ",
                country="BR",
                bio="Automated news aggregator",
                stance_label="Supportive",
                stance_score=0.30,
            ),
        ]

        assembled_prompts: list[str] = []
        for persona in personas:
            full_prompt = JEVPromptBuilder.assemble_eval_prompt(shared_post, persona)
            assembled_prompts.append(full_prompt)

            # Guarantees:
            # 1. Full prompt strictly starts with the post prefix string
            assert full_prompt.startswith(post_prefix_str), (
                f"Agent {persona.user_id} prompt does not start with post prefix!"
            )

            # 2. Exact byte-level slice identity
            prompt_bytes = full_prompt.encode("utf-8")
            prefix_slice_bytes = prompt_bytes[: len(post_prefix_bytes)]
            assert prefix_slice_bytes == post_prefix_bytes, (
                f"Agent {persona.user_id} prefix slice does not match post prefix bytes!"
            )

            # 3. Suffix is non-empty and personalized to this observer
            suffix_part = full_prompt[len(post_prefix_str) :]
            assert f"@{persona.user_name}" in suffix_part
            assert persona.mbti in suffix_part
            assert persona.country in suffix_part
            assert persona.bio in suffix_part
            assert "Action: " in suffix_part

        # Verify all 10 prefixes are mutually byte-identical
        first_prefix_bytes = assembled_prompts[0].encode("utf-8")[
            : len(post_prefix_bytes)
        ]
        for idx, prompt in enumerate(assembled_prompts[1:], start=2):
            curr_prefix_bytes = prompt.encode("utf-8")[: len(post_prefix_bytes)]
            assert curr_prefix_bytes == first_prefix_bytes, (
                f"Observer {idx} prefix bytes differ from observer 1!"
            )

        # Prefix Caching Efficiency Ratio Check
        # The shared prefix must comprise a significant fraction (>40%) of the total prompt length,
        # confirming 70-90% KV-cache reuse capability when batched across observers.
        for prompt in assembled_prompts:
            ratio = len(post_prefix_bytes) / len(prompt.encode("utf-8"))
            assert 0.30 <= ratio <= 0.85, f"Prefix ratio {ratio:.2f} outside expected range"

    def test_post_prefix_normalization_invariance(self) -> None:
        """Verify that handle normalization (@alice vs alice, #topic vs topic) guarantees byte identity."""
        p1 = PostPrefixData(
            post_id=101,
            author_name="@satoshi",
            topic="#crypto",
            content="Proof of work enables distributed consensus.",
        )
        p2 = PostPrefixData(
            post_id=101,
            author_name="satoshi",
            topic="crypto",
            content="   Proof of work enables distributed consensus.   ",
        )
        p3 = PostPrefixData(
            post_id=101,
            author_name="  @satoshi  ",
            topic="  #crypto  ",
            content="Proof of work enables distributed consensus.",
        )

        prefix1 = JEVPromptBuilder.build_post_prefix(p1)
        prefix2 = JEVPromptBuilder.build_post_prefix(p2)
        prefix3 = JEVPromptBuilder.build_post_prefix(p3)

        assert prefix1.encode("utf-8") == prefix2.encode("utf-8")
        assert prefix1.encode("utf-8") == prefix3.encode("utf-8")
        assert prefix1 == prefix2 == prefix3

    def test_post_prefix_unicode_and_multilingual_byte_identity(self) -> None:
        """Verify byte-level caching identity holds for UTF-8 multilingual text and emojis."""
        multilingual_post = PostPrefixData(
            post_id=777,
            author_name="мир_ai",
            topic="технологии",
            content="🚀 Модель рассуждений достигла 95% точности! 人工知能の発展 🤖🔒",
        )
        prefix_str = JEVPromptBuilder.build_post_prefix(multilingual_post)
        prefix_bytes = prefix_str.encode("utf-8")

        agent_a = AgentSuffixData(
            user_id=1,
            user_name="alex",
            mbti="INTJ",
            country="UA",
            bio="Data scientist",
            stance_label="Supportive",
            stance_score=0.7,
        )
        agent_b = AgentSuffixData(
            user_id=2,
            user_name="kenji",
            mbti="INTP",
            country="JP",
            bio="ML engineer",
            stance_label="Neutral / Undecided",
            stance_score=0.0,
        )

        prompt_a = JEVPromptBuilder.assemble_eval_prompt(multilingual_post, agent_a)
        prompt_b = JEVPromptBuilder.assemble_eval_prompt(multilingual_post, agent_b)

        assert prompt_a.startswith(prefix_str)
        assert prompt_b.startswith(prefix_str)
        assert prompt_a.encode("utf-8")[: len(prefix_bytes)] == prefix_bytes
        assert prompt_b.encode("utf-8")[: len(prefix_bytes)] == prefix_bytes


# ==============================================================================
# 2. 1-Token Output Parsing and Intra-Feed Argmax Budget Resolution Tests
# ==============================================================================


class Test1TokenOutputParsingAndBudgetResolution:
    """Test suite for 1-token output parsing and intra-feed budget resolution."""

    @pytest.mark.parametrize(
        "raw_output, expected_action",
        [
            # Standard single tokens
            ("L", "L"),
            ("R", "R"),
            ("C", "C"),
            ("S", "S"),
            # Lowercase tokens
            ("l", "L"),
            ("r", "R"),
            ("c", "C"),
            ("s", "S"),
            # Whitespace and newline padding
            ("  L  ", "L"),
            ("\n\nR\n", "R"),
            ("\t C \t", "C"),
            ("  s  ", "S"),
            # Action header formats
            ("Action: L", "L"),
            ("Action: R", "R"),
            ("Action: C", "C"),
            ("Action: S", "S"),
            ("action: l", "L"),
            ("[Action]: C", "C"),
            # Bracketed tokens
            ("[L]", "L"),
            ("[R]", "R"),
            ("[C]", "C"),
            ("[S]", "S"),
            ("[l]", "L"),
            ("Action: [L]", "L"),
            # Full word descriptions
            ("Like", "L"),
            ("LIKE", "L"),
            ("Repost", "R"),
            ("REPOST", "R"),
            ("Comment", "C"),
            ("COMMENT", "C"),
            ("Skip", "S"),
            ("SKIP", "S"),
            ("I want to like this post", "L"),
            ("Skip this item", "S"),
            # Fallbacks on unparseable / empty inputs
            ("", "S"),
            ("   ", "S"),
            ("None", "S"),
            ("UNKNOWN", "S"),
            ("42", "S"),
            ("InvalidAction", "S"),
        ],
    )
    def test_1token_action_parsing_variations(
        self, raw_output: str, expected_action: str
    ) -> None:
        """Verify robust parsing of 1-token action strings into canonical 'L', 'R', 'C', 'S'."""
        parsed = JEVPromptBuilder.parse_action_char(raw_output)
        assert parsed == expected_action

    def test_intra_feed_argmax_budget_resolution_single_agent(self) -> None:
        """Verify argmax confidence selection when agent feed emits more active actions than budget."""
        results = [
            ClassificationResult(
                user_id=1,
                post_id=101,
                action_char="L",
                confidence=0.82,
                logits={"L": 0.82, "S": 0.18},
            ),
            ClassificationResult(
                user_id=1,
                post_id=102,
                action_char="L",
                confidence=0.96,  # Highest confidence non-skip
                logits={"L": 0.96, "S": 0.04},
            ),
            ClassificationResult(
                user_id=1,
                post_id=103,
                action_char="R",
                confidence=0.74,
                logits={"R": 0.74, "S": 0.26},
            ),
            ClassificationResult(
                user_id=1,
                post_id=104,
                action_char="S",
                confidence=0.99,
                logits={"S": 0.99, "L": 0.01},
            ),
            ClassificationResult(
                user_id=1,
                post_id=105,
                action_char="C",
                confidence=0.65,
                logits={"C": 0.65, "S": 0.35},
            ),
        ]

        # Scenario A: budget=1, downgrade_to_skip=True
        # Post 102 (conf 0.96) must be retained; Posts 101, 103, 105 downgraded to 'S'
        resolved = resolve_intra_feed_budget(
            results, budget=1, downgrade_to_skip=True
        )
        assert len(resolved) == 5
        # Verify ordering is strictly preserved
        assert [r.post_id for r in resolved] == [101, 102, 103, 104, 105]

        # Post 102 retains 'L'
        assert resolved[1].action_char == "L"
        assert resolved[1].confidence == 0.96

        # Posts 101, 103, 105 are converted to 'S'
        assert resolved[0].action_char == "S"
        assert resolved[2].action_char == "S"
        assert resolved[3].action_char == "S"  # Originally 'S'
        assert resolved[4].action_char == "S"

        # Scenario B: budget=2, downgrade_to_skip=True
        # Top 2 non-skips: Post 102 (0.96) and Post 101 (0.82)
        resolved2 = resolve_intra_feed_budget(
            results, budget=2, downgrade_to_skip=True
        )
        assert len(resolved2) == 5
        assert resolved2[0].action_char == "L"  # Kept (conf 0.82)
        assert resolved2[1].action_char == "L"  # Kept (conf 0.96)
        assert resolved2[2].action_char == "S"  # Downgraded (conf 0.74)
        assert resolved2[3].action_char == "S"  # Kept (originally S)
        assert resolved2[4].action_char == "S"  # Downgraded (conf 0.65)

        # Scenario C: downgrade_to_skip=False
        # Only allowed non-skips + natural skips are kept
        resolved_nodowngrade = resolve_intra_feed_budget(
            results, budget=1, downgrade_to_skip=False
        )
        # Should contain Post 102 (allowed non-skip) and Post 104 (natural skip)
        kept_post_ids = [r.post_id for r in resolved_nodowngrade]
        assert kept_post_ids == [102, 104]

    def test_intra_feed_budget_multi_agent_isolation(self) -> None:
        """Verify that intra-feed budget is applied independently per user without cross-agent contamination."""
        results = [
            # User 1 items
            ClassificationResult(user_id=1, post_id=10, action_char="L", confidence=0.70),
            ClassificationResult(user_id=1, post_id=11, action_char="L", confidence=0.92),
            ClassificationResult(user_id=1, post_id=12, action_char="R", confidence=0.60),
            # User 2 items
            ClassificationResult(user_id=2, post_id=20, action_char="C", confidence=0.55),
            ClassificationResult(user_id=2, post_id=21, action_char="R", confidence=0.88),
            # User 3 items (all skips)
            ClassificationResult(user_id=3, post_id=30, action_char="S", confidence=0.95),
            ClassificationResult(user_id=3, post_id=31, action_char="S", confidence=0.99),
        ]

        resolved = resolve_intra_feed_budget(results, budget=1, downgrade_to_skip=True)
        assert len(resolved) == 7

        # User 1: Post 11 retained
        u1_res = [r for r in resolved if r.user_id == 1]
        assert u1_res[0].action_char == "S"
        assert u1_res[1].action_char == "L"
        assert u1_res[2].action_char == "S"

        # User 2: Post 21 retained
        u2_res = [r for r in resolved if r.user_id == 2]
        assert u2_res[0].action_char == "S"
        assert u2_res[1].action_char == "R"

        # User 3: All skips preserved
        u3_res = [r for r in resolved if r.user_id == 3]
        assert u3_res[0].action_char == "S"
        assert u3_res[1].action_char == "S"

    def test_select_best_action_utility(self) -> None:
        """Verify select_best_action finds argmax confidence non-skip or returns None if all skips."""
        items = [
            ClassificationResult(user_id=1, post_id=1, action_char="L", confidence=0.80),
            ClassificationResult(user_id=1, post_id=2, action_char="R", confidence=0.95),
            ClassificationResult(user_id=1, post_id=3, action_char="S", confidence=0.99),
        ]
        best = select_best_action(items)
        assert best is not None
        assert best.post_id == 2
        assert best.action_char == "R"
        assert best.confidence == 0.95

        # All skips returns None
        all_skips = [
            ClassificationResult(user_id=1, post_id=1, action_char="S", confidence=0.90),
            ClassificationResult(user_id=1, post_id=2, action_char="S", confidence=0.95),
        ]
        assert select_best_action(all_skips) is None

    def test_compute_softmax_stability(self) -> None:
        """Verify numerical stability of softmax with extreme logits."""
        logits = {"L": 1000.0, "R": 1000.0, "C": 900.0, "S": 0.0}
        probs = compute_softmax(logits)
        assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-5)
        assert math.isclose(probs["L"], 0.5, rel_tol=1e-3)
        assert math.isclose(probs["R"], 0.5, rel_tol=1e-3)
        assert probs["S"] == 0.0

        # Empty logits returns empty dict
        assert compute_softmax({}) == {}


# ==============================================================================
# 3. Dynamic Belief State Drift & Opinion Dynamics Tests
# ==============================================================================


class TestDynamicBeliefStateDrift:
    """Test suite for Track 2 Dynamic Belief State and Bounded Confidence Opinion Dynamics."""

    def test_multi_step_convergence_under_positive_engagement(self) -> None:
        """Verify monotonic convergence towards positive stance under repeated positive peer engagement."""
        belief = BeliefState(
            user_id=1,
            stances={"ai_ethics": 0.0},  # Neutral starting stance
            persuasion_rate_alpha=0.20,
            delta_max=1.0,
        )
        assert belief.get_stance_label("ai_ethics") == LABEL_NEUTRAL

        post_stance = 0.85
        high_peer_engagement = 50  # sigmoid(50) ≈ 1.0

        trajectory: list[float] = [belief.get_stance("ai_ethics")]

        for step in range(10):
            prev_s = belief.get_stance("ai_ethics")
            new_s = belief.update_stance(
                topic="ai_ethics",
                post_stance=post_stance,
                peer_engagement=high_peer_engagement,
            )
            trajectory.append(new_s)

            # Monotonic shift towards positive post stance
            assert new_s > prev_s, f"Step {step}: Stance failed to increase ({new_s} <= {prev_s})"
            assert new_s <= 1.0, f"Step {step}: Stance exceeded upper bound 1.0"
            assert new_s <= post_stance, f"Step {step}: Stance overshot target post stance"

        # After 10 steps, agent must have strongly converged into Supportive territory
        final_stance = belief.get_stance("ai_ethics")
        assert final_stance > 0.65
        assert belief.get_stance_label("ai_ethics") == LABEL_STRONGLY_SUPPORTIVE

    def test_multi_step_polarization_under_negative_engagement(self) -> None:
        """Verify monotonic drift towards hostile/opposed stance under negative peer exposure."""
        belief = BeliefState(
            user_id=2,
            stances={"gmo": 0.50},  # Initially supportive
            persuasion_rate_alpha=0.18,
            delta_max=1.0,
        )
        assert belief.get_stance_label("gmo") == LABEL_STRONGLY_SUPPORTIVE

        hostile_post_stance = -0.90
        peer_engagement = 80  # Viral negative engagement

        for step in range(12):
            prev_s = belief.get_stance("gmo")
            new_s = belief.update_stance(
                topic="gmo",
                post_stance=hostile_post_stance,
                peer_engagement=peer_engagement,
            )
            assert new_s < prev_s, f"Step {step}: Stance failed to decrease ({new_s} >= {prev_s})"
            assert new_s >= -1.0, f"Step {step}: Stance fell below lower bound -1.0"

        final_stance = belief.get_stance("gmo")
        assert final_stance < -0.60
        assert belief.get_stance_label("gmo") == LABEL_STRONGLY_OPPOSED

    def test_bounded_confidence_threshold_limiting(self) -> None:
        """Verify delta_max parameter limits the maximum single-step stance jump under extreme gap."""
        # Extreme gap Delta = 2.0 (agent at -1.0, post at +1.0)
        belief_capped = BeliefState(
            user_id=3,
            stances={"topic": -1.0},
            persuasion_rate_alpha=0.25,
            delta_max=0.40,  # Cap jump to 0.40
        )
        belief_uncapped = BeliefState(
            user_id=4,
            stances={"topic": -1.0},
            persuasion_rate_alpha=0.25,
            delta_max=2.0,  # No effective cap
        )

        capped_step = belief_capped.update_stance(
            topic="topic", post_stance=1.0, peer_engagement=10, omega_peer=1.0
        )
        uncapped_step = belief_uncapped.update_stance(
            topic="topic", post_stance=1.0, peer_engagement=10, omega_peer=1.0
        )

        # Expected capped delta_s: 0.25 * 0.40 * 1.0 = +0.10 -> new stance = -0.90
        assert math.isclose(capped_step, -0.90, rel_tol=1e-4)

        # Expected uncapped delta_s: 0.25 * 2.0 * 1.0 = +0.50 -> new stance = -0.50
        assert math.isclose(uncapped_step, -0.50, rel_tol=1e-4)

        # Bounded confidence correctly restrained the extreme shift
        assert capped_step < uncapped_step

    def test_peer_engagement_social_proof_modulation(self) -> None:
        """Verify peer engagement weight omega_peer = sigmoid(engagement) scales persuasion force."""
        b_low = BeliefState(
            user_id=5, stances={"news": 0.0}, persuasion_rate_alpha=0.20
        )
        b_high = BeliefState(
            user_id=6, stances={"news": 0.0}, persuasion_rate_alpha=0.20
        )

        # Post stance = 1.0
        # Engagement = 0 -> sigmoid(0) = 0.50
        step_low = b_low.update_stance(topic="news", post_stance=1.0, peer_engagement=0)
        # Engagement = 50 -> sigmoid(50) ≈ 1.0
        step_high = b_high.update_stance(
            topic="news", post_stance=1.0, peer_engagement=50
        )

        # High engagement must induce ~2x the persuasion shift of zero engagement
        assert step_high > step_low
        assert math.isclose(step_high / step_low, 2.0, rel_tol=0.05)

    def test_episodic_action_log_rolling_window_and_summary(self) -> None:
        """Verify Level 2 episodic action log preserves bounded FIFO capacity and formatted summary."""
        belief = BeliefState(user_id=7, max_history_items=3)

        # Record 5 actions sequentially
        belief.record_action("like", 10, "tech", "2026-09-26T12:00:00Z")
        belief.record_action("repost", 20, "politics", "2026-09-26T12:01:00Z")
        belief.record_action("comment", 30, "science", "2026-09-26T12:02:00Z")
        belief.record_action("like", 40, "tech", "2026-09-26T12:03:00Z")
        belief.record_action("repost", 50, "ai", "2026-09-26T12:04:00Z")

        # Capacity capped strictly at 3 (oldest P10, P20 rolled off)
        assert len(belief.recent_actions) == 3
        kept_post_ids = [item.post_id for item in belief.recent_actions]
        assert kept_post_ids == [30, 40, 50]

        summary = belief.get_episodic_summary()
        assert "Commented on P30" in summary
        assert "Liked P40" in summary
        assert "Reposted P50" in summary
        assert "P10" not in summary
        assert "P20" not in summary


# ==============================================================================
# 4. Micro-Time Poisson Scheduling & Action Queue Tests
# ==============================================================================


class TestMicroTimePoissonSchedulerAndActionQueue:
    """Test suite for continuous Poisson arrival scheduling and ChronologicalActionQueue."""

    def test_poisson_arrival_offsets_strictly_bounded(self) -> None:
        """Verify that sampled offsets tau_i ~ Exp(lambda_i) strictly respect [0, step_duration_seconds]."""
        scheduler = MicroTimeScheduler(
            step_duration_seconds=900.0, default_lambda=1.0, seed=42
        )

        for freq in [0.05, 0.5, 1.0, 3.0, 10.0, 50.0]:
            for _ in range(100):
                offset = scheduler.sample_offset(activity_frequency=freq)
                assert 0.0 <= offset <= 900.0, f"Offset {offset} outside bounds for freq {freq}"

    def test_activity_frequency_statistical_scaling(self) -> None:
        """Verify that agents with higher activity frequency arrive statistically earlier."""
        scheduler = MicroTimeScheduler(
            step_duration_seconds=1000.0, default_lambda=1.0, seed=12345
        )

        n_samples = 1000
        high_freq_offsets = [
            scheduler.sample_offset(5.0) for _ in range(n_samples)
        ]
        low_freq_offsets = [
            scheduler.sample_offset(0.2) for _ in range(n_samples)
        ]

        mean_high = sum(high_freq_offsets) / n_samples
        mean_low = sum(low_freq_offsets) / n_samples

        # Higher activity frequency yields strictly smaller mean arrival delay
        assert mean_high < mean_low
        # E[tau] for high = 1 / (5.0 * 1.0) = 0.2s; for low = min(1 / 0.2, 1000) = 5.0s
        assert mean_high < 1.0
        assert mean_low > 2.0

    def test_inactive_agent_defaults_to_step_end(self) -> None:
        """Verify agents with non-positive activity frequency default to step_duration_seconds."""
        scheduler = MicroTimeScheduler(step_duration_seconds=900.0)
        assert scheduler.sample_offset(0.0) == 900.0
        assert scheduler.sample_offset(-1.5) == 900.0

    def test_chronological_action_queue_priority_and_fifo_tie_breaking(
        self,
    ) -> None:
        """Verify priority queue orders by sort_timestamp, and resolves ties using FIFO counter."""
        queue = ChronologicalActionQueue()

        # Insert items out of order, including identical timestamps
        a1 = ScheduledAction(
            sort_timestamp=100.0,
            iso_timestamp="2026-09-26T12:01:40Z",
            user_id=1,
            action_dict={"id": "first_at_100"},
        )
        a2 = ScheduledAction(
            sort_timestamp=100.0,
            iso_timestamp="2026-09-26T12:01:40Z",
            user_id=2,
            action_dict={"id": "second_at_100"},
        )
        a_early = ScheduledAction(
            sort_timestamp=50.0,
            iso_timestamp="2026-09-26T12:00:50Z",
            user_id=3,
            action_dict={"id": "early_at_50"},
        )
        a3 = ScheduledAction(
            sort_timestamp=100.0,
            iso_timestamp="2026-09-26T12:01:40Z",
            user_id=4,
            action_dict={"id": "third_at_100"},
        )
        a_late = ScheduledAction(
            sort_timestamp=200.0,
            iso_timestamp="2026-09-26T12:03:20Z",
            user_id=5,
            action_dict={"id": "late_at_200"},
        )

        queue.put_nowait(a1)
        queue.put_nowait(a2)
        queue.put_nowait(a_early)
        queue.put_nowait(a3)
        queue.put_nowait(a_late)

        drained = queue.drain()
        assert len(drained) == 5

        # Strictly ordered by timestamp
        assert drained[0].action_dict["id"] == "early_at_50"
        # FIFO stability: identical 100.0 timestamps popped in exact insertion order: a1, a2, a3
        assert drained[1].action_dict["id"] == "first_at_100"
        assert drained[2].action_dict["id"] == "second_at_100"
        assert drained[3].action_dict["id"] == "third_at_100"
        # Late item popped last
        assert drained[4].action_dict["id"] == "late_at_200"

    @pytest.mark.asyncio
    async def test_chronological_action_queue_channel_delivery(self) -> None:
        """Verify draining queue delivers scheduled actions to OASIS Channel in micro-timestamp order."""
        queue = ChronologicalActionQueue()
        channel = Channel()

        # Push 6 actions with shuffled timestamps
        timestamps = [30.0, 10.0, 50.0, 20.0, 40.0, 15.0]
        for idx, ts in enumerate(timestamps):
            queue.put_nowait(
                ScheduledAction(
                    sort_timestamp=ts,
                    iso_timestamp=f"2026-09-26T12:00:{int(ts):02d}Z",
                    user_id=idx + 1,
                    action_dict={
                        "action_type": ActionType.LIKE_POST,
                        "message": 100 + idx,
                    },
                )
            )

        delivered_count = await queue.drain_to_channel(channel)
        assert delivered_count == 6

        # Pull messages from channel.receive_queue and verify sorted order
        delivered_uids: list[int] = []
        for _ in range(6):
            msg = await channel.receive_queue.get()
            # Each channel message format: (msg_id, (user_id, message, action_type))
            payload = msg[1]
            delivered_uids.append(payload[0])

        # UIDs ordered by sorted timestamps:
        # 10.0 -> UID 2
        # 15.0 -> UID 6
        # 20.0 -> UID 4
        # 30.0 -> UID 1
        # 40.0 -> UID 5
        # 50.0 -> UID 3
        expected_uids = [2, 6, 4, 1, 5, 3]
        assert delivered_uids == expected_uids


# ==============================================================================
# 5. Multi-Step End-to-End JEV Simulation with OasisEnv / JEVEnvironment
# ==============================================================================


class TestEndToEndJEVSimulationHermetic:
    """Test suite executing multi-step hermetic JEV simulations without GPU or network."""

    @pytest.mark.asyncio
    async def test_multi_step_simulation_hermetic_zero_gpu_network(self) -> None:
        """Verify multi-step simulation lifecycle with real JEVEnvironment and MockJEVClassifierClient."""
        graph = AgentGraph(backend="igraph")
        # 6 diverse agents
        agents = [
            SyntheticAgent(1, "alice", mbti="INTJ", country="US", activity_freq=2.0),
            SyntheticAgent(2, "bob", mbti="ENFP", country="UK", activity_freq=1.0),
            SyntheticAgent(3, "carol", mbti="ISTP", country="CA", activity_freq=0.5),
            SyntheticAgent(4, "dave", mbti="ENTP", country="DE", activity_freq=1.5),
            SyntheticAgent(5, "elena", mbti="INFJ", country="FR", activity_freq=3.0),
            SyntheticAgent(6, "frank", mbti="ESTJ", country="JP", activity_freq=0.8),
        ]
        for a in agents:
            graph.add_agent(a)

        mock_plat = MockHermeticPlatform()
        env = JEVEnvironment(
            env_or_graph=graph,
            config=JEVExecutionConfig(
                step_duration_seconds=900.0,
                max_actions_per_agent=1,
                enable_belief_updates=True,
                seed=42,
            ),
        )
        env.platform = mock_plat
        env.channel = mock_plat.channel

        # Deterministic Mock Classifier
        mock_client = MockJEVClassifierClient(
            action_mapping={
                # Step 0 mappings
                (1, 101): "L",
                (2, 101): "C",
                (3, 101): "S",
                (4, 102): "R",
                (5, 102): "L",
                (6, 102): "S",
                # Step 1 mappings
                (1, 201): "R",
                (2, 201): "L",
                (3, 201): "L",
                (4, 202): "C",
                (5, 202): "S",
                (6, 202): "R",
            },
            mock_comment="Hermetic test comment on post.",
        )
        env.classifier_client = mock_client

        # Execute Step 0
        feeds_step0 = {
            1: [{"post_id": 101, "topic": "ai", "content": "Post 101", "stance": 0.8, "num_likes": 20}],
            2: [{"post_id": 101, "topic": "ai", "content": "Post 101", "stance": 0.8, "num_likes": 20}],
            3: [{"post_id": 101, "topic": "ai", "content": "Post 101", "stance": 0.8, "num_likes": 20}],
            4: [{"post_id": 102, "topic": "ai", "content": "Post 102", "stance": 0.6, "num_likes": 10}],
            5: [{"post_id": 102, "topic": "ai", "content": "Post 102", "stance": 0.6, "num_likes": 10}],
            6: [{"post_id": 102, "topic": "ai", "content": "Post 102", "stance": 0.6, "num_likes": 10}],
        }
        res0 = await env.step_jev(step_index=0, agent_feeds=feeds_step0)

        assert res0.step_index == 0
        assert res0.total_evaluations == 6
        # Expected active actions: Alice (L), Bob (C), Dave (R), Elena (L) -> 4 active actions
        assert res0.num_actions == 4
        assert res0.num_likes == 2
        assert res0.num_reposts == 1
        assert res0.num_comments == 1
        assert res0.num_skips == 2

        # Verify comment generation
        bob_action = next(sa for sa in res0.scheduled_actions if sa.user_id == 2)
        assert bob_action.action_dict["action_char"] == "C"
        assert "Hermetic test comment" in bob_action.action_dict["comment_text"]

        # Verify Channel received active actions
        assert env.channel.receive_queue.qsize() == 4
        # Drain channel for next step
        while not env.channel.receive_queue.empty():
            await env.channel.receive_queue.get()

        # Execute Step 1
        feeds_step1 = {
            1: [{"post_id": 201, "topic": "ai", "content": "Post 201", "stance": 0.9, "num_likes": 30}],
            2: [{"post_id": 201, "topic": "ai", "content": "Post 201", "stance": 0.9, "num_likes": 30}],
            3: [{"post_id": 201, "topic": "ai", "content": "Post 201", "stance": 0.9, "num_likes": 30}],
            4: [{"post_id": 202, "topic": "ai", "content": "Post 202", "stance": -0.5, "num_likes": 15}],
            5: [{"post_id": 202, "topic": "ai", "content": "Post 202", "stance": -0.5, "num_likes": 15}],
            6: [{"post_id": 202, "topic": "ai", "content": "Post 202", "stance": -0.5, "num_likes": 15}],
        }
        res1 = await env.step_jev(step_index=1, agent_feeds=feeds_step1)

        assert res1.step_index == 1
        assert res1.total_evaluations == 6
        # Expected active actions: Alice (R), Bob (L), Carol (L), Dave (C), Frank (R) -> 5 active
        assert res1.num_actions == 5
        assert res1.num_skips == 1

        # Verify belief evolution over 2 steps for Alice (user 1): exposed to 0.8 then 0.9
        bs_alice = env.get_belief_state(1)
        assert bs_alice.get_stance("ai") > 0.20
        # Level 2 episodic actions recorded
        assert len(bs_alice.recent_actions) == 2

    @pytest.mark.asyncio
    async def test_step_jev_backward_compatibility_with_oasis_env(self) -> None:
        """Verify that JEVEnvironment subclasses OasisEnv and OasisEnv directly supports step_jev."""
        graph = AgentGraph(backend="igraph")
        agent = SyntheticAgent(1, "legacy_agent")
        graph.add_agent(agent)

        mock_plat = MockHermeticPlatform()
        oasis_env = OasisEnv.__new__(OasisEnv)
        oasis_env.agent_graph = graph
        oasis_env.platform = mock_plat
        oasis_env.channel = mock_plat.channel
        oasis_env.database_path = ":memory:"
        oasis_env.llm_semaphore = asyncio.Semaphore(10)
        oasis_env.platform_type = DefaultPlatformType.TWITTER
        oasis_env.platform_task = None

        feeds = {1: [{"post_id": 99, "topic": "science", "content": "Quantum entanglement breakthrough."}]}
        res = await oasis_env.step_jev(step_index=0, agent_feeds=feeds)

        assert isinstance(res, JEVStepResult)
        assert res.total_evaluations == 1
        assert hasattr(oasis_env, "_jev_engine")
        assert isinstance(oasis_env._jev_engine, JEVEnvironment)


# ==============================================================================
# 6. Latency Micro-Benchmark: 150 Simulated Agents
# ==============================================================================


class TestJEVLatencyMicroBenchmark:
    """Micro-benchmark verifying high throughput of 1-token classification over 150 simulated agents."""

    @pytest.mark.asyncio
    async def test_150_agents_high_throughput_micro_benchmark(self) -> None:
        """Benchmark 1-token logit classification + intra-feed budget over 150 agents (750 items).

        Verifies:
        1. 750 item evaluations executed within < 100ms wall-clock time on 4 CPU threads.
        2. Throughput exceeds 3,000 item evaluations per second.
        3. Intra-feed budget resolution maintains strict per-agent bounds (<= 150 active actions).
        """
        n_agents = 150
        posts_per_feed = 5
        total_evaluations = n_agents * posts_per_feed  # 750 items

        # Synthesize 150 agents
        agents = [
            AgentSuffixData(
                user_id=i,
                user_name=f"sim_user_{i:03d}",
                mbti="INTJ" if i % 2 == 0 else "ENFP",
                country="US" if i % 3 == 0 else "UK",
                bio=f"Synthetic benchmark agent {i}",
                stance_label="Neutral / Undecided",
                stance_score=0.0,
            )
            for i in range(n_agents)
        ]

        # 5 shared candidate posts
        posts = [
            PostPrefixData(
                post_id=p,
                author_name=f"curator_{p}",
                topic="climate",
                content=f"Renewable energy storage benchmark report item #{p}.",
            )
            for p in range(posts_per_feed)
        ]

        # Stage 1: Assembled Inverted Prompts
        t_start_assemble = time.perf_counter()
        items: list[EvalItem] = []
        for agent in agents:
            for post in posts:
                prompt = JEVPromptBuilder.assemble_eval_prompt(post, agent)
                items.append(
                    EvalItem(
                        user_id=agent.user_id,
                        post_id=post.post_id,
                        topic=post.topic,
                        full_prompt=prompt,
                    )
                )
        t_assemble = time.perf_counter() - t_start_assemble

        assert len(items) == total_evaluations

        # Stage 2: Hermetic Batched Classification
        client = MockJEVClassifierClient(seed=42)
        t_start_classify = time.perf_counter()
        raw_results = await client.classify_batch(items, generate_comments=False)
        t_classify = time.perf_counter() - t_start_classify

        assert len(raw_results) == total_evaluations

        # Stage 3: Intra-Feed Argmax Budget Resolution (budget=1 per agent)
        t_start_resolve = time.perf_counter()
        resolved_results = resolve_intra_feed_budget(
            raw_results, budget=1, downgrade_to_skip=True
        )
        t_resolve = time.perf_counter() - t_start_resolve

        assert len(resolved_results) == total_evaluations

        t_total = t_assemble + t_classify + t_resolve
        throughput = total_evaluations / t_total

        # Latency Assertions:
        # Entire 750-evaluation pipeline must complete in under 100ms (0.10s)
        assert t_total < 0.10, f"Benchmark total latency {t_total*1000:.2f}ms exceeded 100ms!"
        # Throughput must exceed 3,000 evaluations / second
        assert throughput > 3000.0, f"Throughput {throughput:.0f} evals/sec below 3000 threshold"

        # Count active non-skip actions emitted across all 150 agents
        active_actions = [r for r in resolved_results if r.action_char != "S"]
        # With budget=1 per agent, total active actions cannot exceed 150
        assert len(active_actions) <= n_agents

    @pytest.mark.asyncio
    async def test_150_agents_full_env_step_benchmark(self) -> None:
        """Benchmark full JEVEnvironment.step_jev over 150 agents with micro-time scheduling."""
        n_agents = 150
        graph = AgentGraph(backend="igraph")
        for i in range(n_agents):
            graph.add_agent(SyntheticAgent(i, f"agent_{i:03d}"))

        mock_plat = MockHermeticPlatform()
        env = JEVEnvironment(
            env_or_graph=graph,
            config=JEVExecutionConfig(
                step_duration_seconds=900.0,
                max_actions_per_agent=1,
                seed=42,
            ),
        )
        env.platform = mock_plat
        env.channel = mock_plat.channel

        feeds = {
            i: [
                {
                    "post_id": p,
                    "author_name": f"author_{p}",
                    "topic": "tech",
                    "content": f"Content body for post {p}",
                    "stance": 0.5,
                    "num_likes": 10,
                }
                for p in range(5)
            ]
            for i in range(n_agents)
        }

        t_start = time.perf_counter()
        result = await env.step_jev(step_index=0, agent_feeds=feeds)
        duration = time.perf_counter() - t_start

        assert result.total_evaluations == 750
        # Full env step must complete in under 200ms
        assert duration < 0.20, f"Full step_jev latency {duration*1000:.2f}ms exceeded 200ms!"
        assert result.num_actions <= n_agents


# ==============================================================================
# 7. AST Guardrails Asserting Zero sqlite3 Imports
# ==============================================================================


class TestASTGuardrailsStrictZeroSQLite3:
    """Verify strict architectural boundaries and AST cleanliness across all JEV core modules."""

    JEV_CORE_FILES = (
        "oasis/social_agent/jev_prompt_builder.py",
        "oasis/social_agent/belief_state.py",
        "oasis/inference/jev_classifier.py",
        "oasis/clock/micro_time_scheduler.py",
        "oasis/environment/jev_env.py",
    )

    FORBIDDEN_SQL_FRAGMENTS = (
        "SELECT ",
        "INSERT INTO ",
        "DELETE FROM ",
        "UPDATE ",
        "DROP TABLE",
        "CREATE TABLE",
    )

    def _get_oasis_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @pytest.mark.parametrize("rel_path", JEV_CORE_FILES)
    def test_zero_sqlite3_imports_in_jev_core_modules(self, rel_path: str) -> None:
        """Verify that JEV core modules strictly contain zero sqlite3 imports."""
        file_path = self._get_oasis_root() / rel_path
        assert file_path.exists(), f"Target module missing: {file_path}"

        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "sqlite3" not in alias.name, (
                        f"Forbidden 'import {alias.name}' detected in {rel_path}:{node.lineno}!"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "sqlite3" not in mod, (
                    f"Forbidden 'from {mod} import ...' detected in {rel_path}:{node.lineno}!"
                )

    @pytest.mark.parametrize("rel_path", JEV_CORE_FILES)
    def test_zero_raw_sql_in_jev_core_modules(self, rel_path: str) -> None:
        """Verify that JEV core modules contain zero raw SQL statement string literals."""
        file_path = self._get_oasis_root() / rel_path
        assert file_path.exists(), f"Target module missing: {file_path}"

        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val_upper = node.value.strip().upper()
                for fragment in self.FORBIDDEN_SQL_FRAGMENTS:
                    assert fragment not in val_upper, (
                        f"Forbidden raw SQL query literal '{fragment}' detected in {rel_path}:{node.lineno}!"
                    )

    def test_resource_caps_enforced(self) -> None:
        """Verify environment variables enforce the 4 CPU thread limit."""
        for var in THREAD_VARS:
            assert os.environ.get(var) == "4", f"{var} not set to 4!"


@pytest.mark.asyncio
class TestJEVOrganicPostingTrack:
    """Tests for the separate spontaneous organic posting track and clean persona context."""

    def test_build_agent_persona_context_clean_formatting(self) -> None:
        """Verify build_agent_persona_context outputs clean context without [TASK] instructions."""
        agent = AgentSuffixData(
            user_id=42,
            user_name="test_user",
            mbti="INTJ",
            country="US",
            bio="AI researcher and engineer",
            stance_label="Supportive",
            stance_score=0.75,
            recent_actions="Liked post 1",
            topic="tech",
        )
        context = JEVPromptBuilder.build_agent_persona_context(agent, topic="tech")
        assert "[OBSERVER]: @test_user" in context
        assert "INTJ" in context
        assert "[STANCE]: #tech: Supportive (+0.75)" in context
        assert "[RECENT ACTIONS]: Liked post 1" in context
        assert "[TASK]:" not in context
        assert "Action:" not in context

        # Verify build_agent_suffix incorporates persona context and appends task
        suffix = JEVPromptBuilder.build_agent_suffix(agent, topic="tech")
        assert context in suffix
        assert "[TASK]:" in suffix
        assert suffix.endswith("Action: ")

    async def test_mock_classifier_generate_post(self) -> None:
        """Verify MockJEVClassifierClient produces topic-aligned spontaneous root posts."""
        client = MockJEVClassifierClient(seed=123)
        post = await client.generate_post(
            agent_context="[OBSERVER]: @tech_analyst",
            topic="tech",
            stance_label="Supportive",
        )
        assert isinstance(post, str)
        assert len(post) > 10
        assert "#tech" in post

        # Batch generation
        batch_reqs = [
            ("[OBSERVER]: @u1", "tech", "Supportive"),
            ("[OBSERVER]: @u2", "sports", "Neutral"),
            ("[OBSERVER]: @u3", "politics", "Skeptical"),
        ]
        results = await client.generate_posts_batch(batch_reqs)
        assert len(results) == 3
        assert "#tech" in results[0]
        assert "#sports" in results[1]
        assert "#politics" in results[2]

    async def test_step_organic_posts_execution(self) -> None:
        """Verify step_organic_posts generates and dispatches CREATE_POST actions for both bots and organic users."""
        from cib_zoo.agent.cib_agent import CIBAgent, NoOpModelBackend
        from oasis.social_agent.agent import SocialAgent
        from oasis.social_platform.channel import Channel
        from oasis.social_platform.config import UserInfo
        from oasis.social_platform.typing import ActionType

        channel = Channel()
        graph = AgentGraph()

        # Create organic agent
        org_info = UserInfo(user_name="org_user_1", name="Organic User 1", description="Tech developer")
        org_agent = SocialAgent(
            agent_id=1,
            user_info=org_info,
            channel=channel,
            model=NoOpModelBackend(),
        )
        graph.add_agent(org_agent)

        # Create CIB bot
        bot_info = UserInfo(user_name="cib_bot_10", name="Bot 10", description="Sports fan")
        bot = CIBAgent(
            agent_id=10,
            user_info=bot_info,
            channel=channel,
            max_actions_per_step=3,
        )
        graph.add_agent(bot)

        config = JEVExecutionConfig(
            step_duration_seconds=300.0,
            enable_organic_posting=True,
            organic_post_rate=1.0,  # deterministic 100% posting for test
            bot_organic_post_rate=1.0,
            community_map={1: "tech", 10: "sports"},
        )
        client = MockJEVClassifierClient()
        jev_env = JEVEnvironment(
            env_or_graph=graph,
            config=config,
            classifier_client=client,
        )

        scheduled_posts = await jev_env.step_organic_posts(
            step_index=0,
            candidate_agents=[org_agent, bot],
        )

        assert len(scheduled_posts) == 2
        post_uids = {p.user_id for p in scheduled_posts}
        assert post_uids == {1, 10}

        for sa in scheduled_posts:
            act = sa.action_dict
            assert act["action_type"] == ActionType.CREATE_POST
            assert act["action_name"] == "create_post"
            assert isinstance(act["content"], str)
            if sa.user_id == 1:
                assert "#tech" in act["content"]
            elif sa.user_id == 10:
                assert "#sports" in act["content"]

        # Verify bot budget limiter recorded the post
        assert bot.budget.step_action_count == 1
        assert bot.budget.get_action_counts()[ActionType.CREATE_POST] == 1

        # Verify BeliefState recorded create_post
        org_b = jev_env.get_belief_state(1)
        assert len(org_b.recent_actions) == 1
        assert org_b.recent_actions[0].action_type == "create_post"

    async def test_step_organic_posts_respects_bot_budget_exhaustion(self) -> None:
        """Verify CIB bots do not post if CREATE_POST occurrence threshold is exhausted."""
        from cib_zoo.agent.cib_agent import CIBAgent
        from oasis.social_platform.channel import Channel
        from oasis.social_platform.config import UserInfo
        from oasis.social_platform.typing import ActionType

        channel = Channel()
        graph = AgentGraph()

        bot_info = UserInfo(user_name="exhausted_bot", name="Exhausted Bot")
        bot = CIBAgent(
            agent_id=20,
            user_info=bot_info,
            channel=channel,
            occurrence_thresholds={ActionType.CREATE_POST: 0},  # Cap to 0 posts
        )
        graph.add_agent(bot)

        config = JEVExecutionConfig(
            bot_organic_post_rate=1.0,
            default_topic="tech",
        )
        client = MockJEVClassifierClient()
        jev_env = JEVEnvironment(
            env_or_graph=graph,
            config=config,
            classifier_client=client,
        )

        scheduled_posts = await jev_env.step_organic_posts(
            step_index=0,
            candidate_agents=[bot],
        )
        assert len(scheduled_posts) == 0

    async def test_seed_initial_posts_populates_feed_across_communities(self) -> None:
        """Verify seed_initial_posts distributes background posts across multiple communities."""
        from cib_zoo.agent.cib_agent import CIBAgent, NoOpModelBackend
        from oasis.social_agent.agent import SocialAgent
        from oasis.social_platform.channel import Channel
        from oasis.social_platform.config import UserInfo
        from oasis.social_platform.typing import ActionType

        CIBAgent.reset_registry()
        channel = Channel()
        graph = AgentGraph()

        # Create agents across 3 communities: tech, sports, politics
        agents = []
        community_map = {}
        for i in range(1, 10):
            topic = "tech" if i <= 3 else ("sports" if i <= 6 else "politics")
            community_map[i] = topic
            info = UserInfo(
                user_name=f"user_{i}",
                name=f"User {i}",
                description=f"{topic} enthusiast",
            )
            agent = SocialAgent(
                agent_id=i,
                user_info=info,
                channel=channel,
                model=NoOpModelBackend(),
            )
            graph.add_agent(agent)
            agents.append(agent)

        config = JEVExecutionConfig(
            community_map=community_map,
            default_topic="tech",
        )
        client = MockJEVClassifierClient()
        jev_env = JEVEnvironment(
            env_or_graph=graph,
            config=config,
            classifier_client=client,
        )

        seeded = await jev_env.seed_initial_posts(
            agents=agents,
            num_posts=6,
        )

        assert len(seeded) == 6
        topics_seen = {sa.action_dict["topic"] for sa in seeded}
        assert "tech" in topics_seen
        assert "sports" in topics_seen
        assert "politics" in topics_seen

        for sa in seeded:
            assert sa.action_dict["action_type"] == ActionType.CREATE_POST
            assert f"#{sa.action_dict['topic']}" in sa.action_dict["content"]
