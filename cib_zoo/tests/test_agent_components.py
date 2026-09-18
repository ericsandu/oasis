"""Hermetic component tests for CIBAgent, BudgetLimiter, and PerceptionCache."""

from __future__ import annotations

import unittest.mock as mock
from typing import Any

import pytest

from oasis.environment.env_action import ManualAction
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.typing import ActionType
from cib_zoo.agent.budget import BudgetLimiter
from cib_zoo.agent.cib_agent import CIBAgent, NoOpModelBackend
from cib_zoo.agent.perception import CachedPost, CachedUser, PerceptionCache


# ============================================================================
# 1. Budget & Lifetime Exhaustion Tests
# ============================================================================


class TestCIBAgentBudgetExhaustion:
    """Verifies that CIBAgent and BudgetLimiter strictly halt once total_budget is reached."""

    def test_agent_budget_exhaustion(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        """Requirement R1/R5: Agent stops generating actions once total_budget is reached."""
        agent = CIBAgent(
            agent_id=1,
            user_info=mock_user_info,
            channel=mock_channel,
            max_actions_per_step=2,
            total_budget=3,
        )

        assert agent.budget.is_exhausted is False
        assert agent.budget.remaining_total_budget == 3

        # Execute Action 1
        assert agent.can_execute(ActionType.LIKE_POST) is True
        agent.record_action(ActionType.LIKE_POST)
        assert agent.budget.remaining_total_budget == 2

        # Execute Action 2
        assert agent.can_execute(ActionType.CREATE_COMMENT) is True
        agent.record_action(ActionType.CREATE_COMMENT)
        assert agent.budget.remaining_total_budget == 1

        agent.reset_step_budget()

        # Execute Action 3 (Exhausting total budget)
        assert agent.can_execute(ActionType.LIKE_POST) is True
        agent.record_action(ActionType.LIKE_POST)
        assert agent.budget.remaining_total_budget == 0
        assert agent.budget.is_exhausted is True

        # Any subsequent active action must be gated
        assert agent.can_execute(ActionType.LIKE_POST) is False
        assert agent.can_execute(ActionType.CREATE_POST) is False
        assert agent.can_execute(ActionType.FOLLOW) is False

        # filter_actions must filter out all candidate actions
        candidates = [
            ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 10}),
            ManualAction(
                action_type=ActionType.CREATE_POST, action_args={"content": "spam"}
            ),
        ]
        filtered = agent.filter_actions(candidates)
        assert filtered == []

    def test_agent_do_nothing_allowed_on_exhaustion(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        """DO_NOTHING is an idle action and must remain executable even when budget is 0."""
        agent = CIBAgent(
            agent_id=2,
            user_info=mock_user_info,
            channel=mock_channel,
            total_budget=0,
        )
        assert agent.budget.is_exhausted is True
        assert agent.can_execute(ActionType.DO_NOTHING) is True

        # Emitting DO_NOTHING on exhaustion when requested
        candidates = [
            ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 1})
        ]
        admitted = agent.budget.filter_actions(
            candidates, emit_do_nothing_on_exhaust=True
        )
        assert len(admitted) == 1
        assert admitted[0].action_type == ActionType.DO_NOTHING


# ============================================================================
# 2. Step Rate Limiting Tests
# ============================================================================


class TestCIBAgentMaxActionsPerStep:
    """Verifies that CIBAgent clamps actions to max_actions_per_step."""

    def test_agent_max_actions_per_step(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        """Requirement R1/R5: Verifies agent clamps candidate actions to max_actions_per_step."""
        agent = CIBAgent(
            agent_id=3,
            user_info=mock_user_info,
            channel=mock_channel,
            max_actions_per_step=2,
            total_budget=10,
        )

        candidates = [
            ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 1}),
            ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 2}),
            ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 3}),
            ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 4}),
            ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 5}),
        ]

        filtered = agent.filter_actions(candidates)
        assert len(filtered) == 2
        assert filtered[0].action_args["post_id"] == 1
        assert filtered[1].action_args["post_id"] == 2

        # Record the 2 admitted actions
        for act in filtered:
            agent.record_action(act.action_type)

        # Step budget is now saturated
        assert agent.can_execute(ActionType.LIKE_POST) is False
        assert agent.budget.remaining_step_budget == 0

        # Step boundary reset restores capacity
        agent.reset_step_budget()
        assert agent.budget.remaining_step_budget == 2
        assert agent.can_execute(ActionType.LIKE_POST) is True


# ============================================================================
# 3. Occurrence Threshold Gating Tests
# ============================================================================


class TestCIBAgentOccurrenceThresholds:
    """Verifies action-type specific quota enforcement."""

    def test_agent_occurrence_thresholds(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        """Requirement R1/R5: Verifies specific action types are gated when quota is reached."""
        agent = CIBAgent(
            agent_id=4,
            user_info=mock_user_info,
            channel=mock_channel,
            max_actions_per_step=5,
            total_budget=20,
            occurrence_thresholds={
                ActionType.LIKE_POST: 2,
                ActionType.CREATE_POST: 1,
            },
        )

        # Unconstrained action types can execute
        assert agent.can_execute(ActionType.FOLLOW) is True

        # Execute 2 LIKE_POST actions
        assert agent.can_execute(ActionType.LIKE_POST) is True
        agent.record_action(ActionType.LIKE_POST)
        assert agent.can_execute(ActionType.LIKE_POST) is True
        agent.record_action(ActionType.LIKE_POST)

        # LIKE_POST quota is reached
        assert agent.can_execute(ActionType.LIKE_POST) is False
        assert agent.budget.remaining_occurrence_quota(ActionType.LIKE_POST) == 0

        # CREATE_POST still has quota 1
        assert agent.can_execute(ActionType.CREATE_POST) is True
        assert agent.budget.remaining_occurrence_quota(ActionType.CREATE_POST) == 1

        # Test filtering mixed candidate actions
        candidates = [
            ManualAction(
                action_type=ActionType.LIKE_POST, action_args={"post_id": 99}
            ),  # Gated
            ManualAction(
                action_type=ActionType.CREATE_POST, action_args={"content": "hi"}
            ),  # Allowed
            ManualAction(
                action_type=ActionType.FOLLOW, action_args={"followee_id": 42}
            ),  # Allowed
        ]
        filtered = agent.filter_actions(candidates)
        assert len(filtered) == 2
        assert filtered[0].action_type == ActionType.CREATE_POST
        assert filtered[1].action_type == ActionType.FOLLOW

    def test_occurrence_thresholds_string_compatibility(self) -> None:
        """Verifies occurrence thresholds accept both Enum and string keys."""
        limiter = BudgetLimiter(
            max_actions_per_step=5,
            occurrence_thresholds={"like_post": 1, "follow": 2},
        )
        assert limiter.can_execute(ActionType.LIKE_POST) is True
        limiter.record_action("like_post")
        assert limiter.can_execute(ActionType.LIKE_POST) is False
        assert limiter.can_execute("follow") is True


# ============================================================================
# 4. Perception Cache Update and Query Tests
# ============================================================================


class TestCIBAgentPerceptionCache:
    """Verifies in-memory caching of REFRESH, SEARCH_POSTS, and SEARCH_USER observations."""

    def test_perception_cache_update_and_query(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        """Requirement R1/R5: Verifies posts and users are cached and queried correctly."""
        agent = CIBAgent(
            agent_id=5,
            user_info=mock_user_info,
            channel=mock_channel,
        )

        # 1. Update from REFRESH feed
        feed_payload = {
            "posts": [
                {
                    "post_id": 101,
                    "user_id": 12,
                    "content": "Breaking news on AI #tech",
                    "created_at": "2026-09-18T12:00:00Z",
                    "num_likes": 10,
                    "num_dislikes": 1,
                    "num_shares": 3,
                },
                {
                    "post_id": 102,
                    "user_id": 15,
                    "content": "A beautiful morning!",
                    "created_at": "2026-09-18T12:05:00Z",
                    "num_likes": 2,
                    "num_dislikes": 0,
                    "num_shares": 0,
                },
            ]
        }
        agent.update_perception(ActionType.REFRESH, feed_payload)

        cached_posts = agent.perception.get_cached_posts()
        assert len(cached_posts) == 2
        p101 = next(p for p in cached_posts if p["post_id"] == 101)
        assert p101["content"] == "Breaking news on AI #tech"
        assert p101["num_likes"] == 10

        # 2. Update from SEARCH_POSTS
        search_posts_payload = {
            "posts": [
                {
                    "post_id": 201,
                    "user_id": 88,
                    "content": "Target topic analysis",
                    "num_likes": 50,
                }
            ]
        }
        agent.update_perception(ActionType.SEARCH_POSTS, search_posts_payload)
        all_posts = agent.perception.get_cached_posts()
        assert len(all_posts) == 3

        # Query with keyword filter
        filtered_posts = agent.perception.get_cached_posts(query="Target topic")
        assert len(filtered_posts) == 1
        assert filtered_posts[0]["post_id"] == 201

        # 3. Update from SEARCH_USER
        search_user_payload = {
            "users": [
                {
                    "user_id": 500,
                    "user_name": "influencer_one",
                    "name": "Influencer One",
                    "bio": "Key opinion leader",
                    "num_followers": 25000,
                    "num_followings": 150,
                }
            ]
        }
        agent.update_perception(ActionType.SEARCH_USER, search_user_payload)
        cached_users = agent.perception.get_cached_users()
        assert len(cached_users) == 1
        assert cached_users[0]["user_name"] == "influencer_one"
        assert cached_users[0]["num_followers"] == 25000

    def test_perception_cache_deduplication(self) -> None:
        """Verifies duplicate post observations update in-place rather than duplicating."""
        cache = PerceptionCache()
        cache.update_from_feed([{"post_id": 1, "content": "Original", "num_likes": 5}])
        assert len(cache.get_cached_posts()) == 1
        assert cache.get_cached_posts()[0]["num_likes"] == 5

        cache.update_from_feed([{"post_id": 1, "content": "Updated", "num_likes": 20}])
        assert len(cache.get_cached_posts()) == 1
        assert cache.get_cached_posts()[0]["num_likes"] == 20
        assert cache.get_cached_posts()[0]["content"] == "Updated"

    def test_perception_cache_step_ttl_invalidation(self) -> None:
        """Verifies step-based TTL evicts or ignores stale posts."""
        cache = PerceptionCache(ttl_steps=2, current_step=0)
        cache.update_from_feed([{"post_id": 1, "content": "Old post"}], step=0)

        # Step 1: post age = 1 (valid)
        cache.advance_step(1)
        assert len(cache.get_cached_posts()) == 1

        # Step 3: post age = 3 (exceeds ttl_steps=2)
        cache.advance_step(3)
        assert len(cache.get_cached_posts()) == 0


# ============================================================================
# 5. Zero Backdoor SQLite Access Guardrail Tests
# ============================================================================


class TestCIBAgentNoDirectSQLiteAccess:
    """Verifies that CIBAgent methods execute without touching sqlite3 connections."""

    def test_no_direct_sqlite_access(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        """Requirement R1/R5: CIBAgent decision methods execute without touching sqlite3 connections."""
        agent = CIBAgent(
            agent_id=6,
            user_info=mock_user_info,
            channel=mock_channel,
            max_actions_per_step=2,
            total_budget=10,
        )

        # Intercept any sqlite3.connect invocation
        with mock.patch(
            "sqlite3.connect",
            side_effect=AssertionError("Backdoor SQLite connection detected!"),
        ) as mock_connect:
            # 1. Budget checks
            assert agent.can_execute(ActionType.LIKE_POST) is True
            agent.record_action(ActionType.LIKE_POST)
            agent.reset_step_budget()

            # 2. Perception cache operations
            agent.update_perception(
                ActionType.REFRESH,
                {"posts": [{"post_id": 999, "content": "In-memory test"}]},
            )
            posts = agent.perception.get_cached_posts()
            assert len(posts) == 1

            # 3. Action filtering
            candidates = [
                ManualAction(
                    action_type=ActionType.LIKE_POST, action_args={"post_id": 999}
                )
            ]
            admitted = agent.filter_actions(candidates)
            assert len(admitted) == 1

            # Verify sqlite3 was never called
            mock_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_cib_environment_no_direct_sqlite_access(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        """Verifies CIBEnvironment prompt and stats generation do not connect to sqlite3."""
        agent = CIBAgent(
            agent_id=7,
            user_info=mock_user_info,
            channel=mock_channel,
        )
        with mock.patch(
            "sqlite3.connect",
            side_effect=AssertionError("Backdoor SQLite connection detected!"),
        ) as mock_connect:
            prompt = await agent.env.to_text_prompt()
            assert "CIB Agent 7" in prompt
            followers_str = await agent.env.get_followers_env()
            assert "0 followers" in followers_str
            follows_str = await agent.env.get_follows_env()
            assert "0 follows" in follows_str
            mock_connect.assert_not_called()


# ============================================================================
# 6. Additional Agent Robustness & Model Backend Tests
# ============================================================================


class TestCIBAgentRobustness:
    """Verifies error handling, no-op backend, and budget validation."""

    def test_budget_limiter_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="max_actions_per_step must be >= 1"):
            BudgetLimiter(max_actions_per_step=0)

        with pytest.raises(ValueError, match="total_budget must be >= 0"):
            BudgetLimiter(total_budget=-1)

        with pytest.raises(ValueError, match="Occurrence quota for .* must be >= 0"):
            BudgetLimiter(occurrence_thresholds={"like_post": -1})

        limiter = BudgetLimiter(max_actions_per_step=1, total_budget=1)
        limiter.record_action(ActionType.LIKE_POST)
        with pytest.raises(RuntimeError, match="exceeds budget limit"):
            limiter.record_action(ActionType.LIKE_POST)

    def test_noop_model_backend(self) -> None:
        backend = NoOpModelBackend()
        assert backend.token_limit == 4096
        with pytest.raises(RuntimeError, match="does not use LLM inference"):
            backend.run("test")

    @pytest.mark.asyncio
    async def test_perform_action_by_llm_raises(
        self,
        mock_channel: Channel,
        mock_user_info: UserInfo,
    ) -> None:
        agent = CIBAgent(
            agent_id=8,
            user_info=mock_user_info,
            channel=mock_channel,
        )
        with pytest.raises(NotImplementedError, match="deterministic/pattern"):
            await agent.perform_action_by_llm()
