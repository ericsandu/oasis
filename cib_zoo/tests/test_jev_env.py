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
"""Comprehensive unit tests for Track 5: Integrated JEV Environment (oasis/environment/jev_env.py).

Validates:
1. JEVExecutionConfig and JEVStepResult dataclass contracts and analytics properties.
2. JEVEnvironment initialization wrapping OasisEnv or directly with AgentGraph.
3. Complete JEV execution loop (step_jev):
   - Inverted prompt assembly & KV prefix sharing across multiple observers.
   - 1-token logit classification with MockJEVClassifierClient.
   - Intra-feed budget resolution with argmax confidence selection.
   - Secondary comment generation worker for 'C' reactions.
   - Dynamic BeliefState bounded persuasion opinion drift & Level 2 episodic action logging.
   - Continuous Poisson arrival time scheduling.
   - ChronologicalActionQueue draining to OASIS Channel in micro-timestamp order.
4. Backward compatibility with standard OasisEnv (CAMEL action loop and step_jev method).
5. AST Guardrails: strictly zero sqlite3 imports and zero raw SQL statements in jev_env.py.
"""

from __future__ import annotations

import ast
import asyncio
import os
from datetime import datetime, timezone
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

from oasis.clock.micro_time_scheduler import ScheduledAction
from oasis.environment import (
    JEVEnvironment,
    JEVExecutionConfig,
    JEVStepResult,
    ManualAction,
    OasisEnv,
)
from oasis.environment.jev_env import default_oasis_channel_formatter
from oasis.inference.jev_classifier import (
    MockJEVClassifierClient,
)
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.belief_state import BeliefState
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.typing import ActionType, DefaultPlatformType

# ==============================================================================
# Helper Mock Fixtures
# ==============================================================================


class DummyAgent:
    """Mock agent representing a SocialAgent for lightweight testing."""

    def __init__(
        self,
        agent_id: int,
        user_name: str | None = None,
        mbti: str = "INTJ",
        country: str = "US",
        bio: str = "A synthetic agent",
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


class MockPlatform:
    """Mock platform simulating platform recsys and channel interactions."""

    def __init__(self, channel: Channel | None = None) -> None:
        self.channel = channel or Channel()
        self.db_path = ":memory:"
        self.recsys_type = "twitter"
        self.start_time = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
        self.updated_rec_table = False
        self.feeds: dict[int, list[dict]] = {}

    async def update_rec_table(self) -> None:
        self.updated_rec_table = True

    async def refresh(self, agent_id: int) -> dict:
        posts = self.feeds.get(agent_id, [])
        return {"success": True, "posts": posts}


# ==============================================================================
# 1. Config & StepResult Contracts
# ==============================================================================


class TestJEVConfigAndResult:
    """Tests for JEVExecutionConfig and JEVStepResult dataclasses."""

    def test_jev_execution_config_defaults(self) -> None:
        config = JEVExecutionConfig()
        assert config.batch_size == 64
        assert config.step_duration_seconds == 900.0
        assert config.default_lambda == 1.0
        assert config.enable_belief_updates is True
        assert config.max_actions_per_agent == 1
        assert config.classifier_client is None
        assert config.default_topic == "general"
        assert config.downgrade_to_skip is True
        assert config.send_skips_to_platform is False

    def test_jev_execution_config_custom_values(self) -> None:
        config = JEVExecutionConfig(
            batch_size=128,
            step_duration_seconds=1800.0,
            default_lambda=2.5,
            max_actions_per_agent=3,
            default_topic="climate",
            seed=123,
            send_skips_to_platform=True,
        )
        assert config.batch_size == 128
        assert config.step_duration_seconds == 1800.0
        assert config.default_lambda == 2.5
        assert config.max_actions_per_agent == 3
        assert config.default_topic == "climate"
        assert config.seed == 123
        assert config.send_skips_to_platform is True

    def test_jev_step_result_properties_and_serialization(self) -> None:
        action1 = ScheduledAction(
            sort_timestamp=100.0,
            iso_timestamp="2026-09-26T12:01:40Z",
            user_id=1,
            action_dict={"action_char": "L", "post_id": 10},
        )
        action2 = ScheduledAction(
            sort_timestamp=105.0,
            iso_timestamp="2026-09-26T12:01:45Z",
            user_id=2,
            action_dict={"action_char": "C", "post_id": 10},
        )
        action3 = ScheduledAction(
            sort_timestamp=110.0,
            iso_timestamp="2026-09-26T12:01:50Z",
            user_id=3,
            action_dict={"action_char": "S", "post_id": 10},
        )

        result = JEVStepResult(
            step_index=0,
            total_evaluations=3,
            action_counts={"L": 1, "R": 0, "C": 1, "S": 1},
            execution_time_seconds=0.045,
            scheduled_actions=[action1, action2, action3],
        )

        assert result.num_likes == 1
        assert result.num_reposts == 0
        assert result.num_comments == 1
        assert result.num_skips == 1
        assert result.num_actions == 2  # likes + reposts + comments

        data = result.to_dict()
        assert data["step_index"] == 0
        assert data["total_evaluations"] == 3
        assert data["num_actions"] == 2
        assert len(data["scheduled_actions"]) == 3
        assert data["scheduled_actions"][0]["user_id"] == 1


# ==============================================================================
# 2. Environment Initialization & Belief State Management
# ==============================================================================


class TestJEVEnvironmentInitialization:
    """Tests for JEVEnvironment construction and state initialization."""

    def test_init_with_agent_graph(self) -> None:
        graph = AgentGraph(backend="igraph")
        agent1 = DummyAgent(1, "alice")
        agent2 = DummyAgent(2, "bob")
        graph.add_agent(agent1)
        graph.add_agent(agent2)

        env = JEVEnvironment(
            env_or_graph=graph,
            config=JEVExecutionConfig(default_topic="ai"),
        )

        assert env.agent_graph is graph
        assert len(env.belief_states) == 2
        assert 1 in env.belief_states
        assert 2 in env.belief_states
        assert env.get_belief_state(1).user_id == 1
        assert env.get_belief_state(2).user_id == 2
        assert isinstance(env.classifier_client, MockJEVClassifierClient)

    def test_init_wrapping_oasis_env(self) -> None:
        graph = AgentGraph(backend="igraph")
        agent = DummyAgent(42, "carol")
        graph.add_agent(agent)

        mock_plat = MockPlatform()
        oasis_env = OasisEnv.__new__(OasisEnv)
        oasis_env.agent_graph = graph
        oasis_env.platform = mock_plat
        oasis_env.channel = mock_plat.channel
        oasis_env.database_path = ":memory:"
        oasis_env.llm_semaphore = asyncio.Semaphore(10)
        oasis_env.platform_type = DefaultPlatformType.TWITTER
        oasis_env.platform_task = None

        jev_env = JEVEnvironment(env_or_graph=oasis_env)
        assert jev_env.agent_graph is graph
        assert jev_env.platform is mock_plat
        assert jev_env.channel is mock_plat.channel
        assert 42 in jev_env.belief_states

    def test_belief_state_get_and_set(self) -> None:
        graph = AgentGraph(backend="igraph")
        env = JEVEnvironment(env_or_graph=graph)

        # Non-existent user automatically initialized with defaults
        state = env.get_belief_state(99)
        assert state.user_id == 99
        assert state.get_stance("ai") == 0.0

        # Custom stance assignment
        custom = BeliefState(user_id=99, stances={"ai": 0.75})
        env.set_belief_state(99, custom)
        assert env.get_belief_state(99).get_stance("ai") == 0.75


# ==============================================================================
# 3. Complete JEV Execution Loop (step_jev)
# ==============================================================================


class TestJEVStepExecution:
    """Tests verifying the complete JEV pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_step_jev_with_custom_feeds_and_mock_classifier(self) -> None:
        """Verify 1-token classification, intra-feed budget, comments, and micro-time dispatch."""
        graph = AgentGraph(backend="igraph")
        agent1 = DummyAgent(1, "alice", activity_freq=2.0)
        agent2 = DummyAgent(2, "bob", activity_freq=1.0)
        graph.add_agent(agent1)
        graph.add_agent(agent2)

        mock_plat = MockPlatform()
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

        # Configure deterministic mock classifier
        # Agent 1 evaluates Post 101 -> 'L' (Like), Post 102 -> 'R' (Repost)
        # Agent 2 evaluates Post 101 -> 'C' (Comment), Post 102 -> 'S' (Skip)
        mock_client = MockJEVClassifierClient(
            action_mapping={
                (1, 101): "L",
                (1, 102): "R",
                (2, 101): "C",
                (2, 102): "S",
            },
            mock_comment="Fascinating discovery in AI!",
        )
        env.classifier_client = mock_client

        # Feeds for both agents
        feeds = {
            1: [
                {
                    "post_id": 101,
                    "author_name": "charlie",
                    "topic": "tech",
                    "content": "New breakthrough in AI alignment algorithms.",
                    "stance": 0.8,
                    "num_likes": 15,
                },
                {
                    "post_id": 102,
                    "author_name": "dave",
                    "topic": "tech",
                    "content": "Open weights models released today.",
                    "stance": 0.4,
                    "num_likes": 5,
                },
            ],
            2: [
                {
                    "post_id": 101,
                    "author_name": "charlie",
                    "topic": "tech",
                    "content": "New breakthrough in AI alignment algorithms.",
                    "stance": 0.8,
                    "num_likes": 15,
                },
                {
                    "post_id": 102,
                    "author_name": "dave",
                    "topic": "tech",
                    "content": "Open weights models released today.",
                    "stance": 0.4,
                    "num_likes": 5,
                },
            ],
        }

        base_time = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
        result = await env.step_jev(
            step_index=0,
            base_time=base_time,
            agent_feeds=feeds,
        )

        assert mock_plat.updated_rec_table is True
        assert result.step_index == 0
        assert result.total_evaluations == 4  # 2 agents * 2 posts
        # Agent 1 budget=1: one non-skip kept, one downgraded to skip
        # Agent 2 budget=1: comment kept, skip kept
        assert result.num_actions == 2
        assert result.num_likes + result.num_reposts == 1
        assert result.num_comments == 1
        assert result.num_skips == 2

        # Verify comment text was generated for 'C' action
        comment_action = next(
            sa
            for sa in result.scheduled_actions
            if sa.user_id == 2 and sa.action_dict["action_char"] == "C"
        )
        assert comment_action.action_dict["action_char"] == "C"
        assert "Fascinating discovery in AI!" in comment_action.action_dict["comment_text"]
        assert comment_action.action_dict["message"][0] == 101
        assert "Fascinating discovery in AI!" in comment_action.action_dict["message"][1]
        assert comment_action.action_dict["action_type"] == ActionType.CREATE_COMMENT

        # Verify BeliefState was updated for agent 1
        bs1 = env.get_belief_state(1)
        assert bs1.get_stance("tech") > 0.0  # shifted towards post stance 0.8
        assert len(bs1.recent_actions) > 0  # episodic action recorded

        # Verify BeliefState for agent 2
        bs2 = env.get_belief_state(2)
        assert bs2.get_stance("tech") > 0.0
        assert "Commented on P101" in bs2.get_episodic_summary()

        # Verify timestamps are strictly ordered
        timestamps = [sa.sort_timestamp for sa in result.scheduled_actions]
        assert timestamps == sorted(timestamps)

        # Verify channel draining: 2 active actions dispatched
        channel_queue = env.channel.receive_queue
        assert channel_queue.qsize() == 2

        msg1 = await channel_queue.get()
        msg2 = await channel_queue.get()
        # Each message on channel receive_queue is (msg_id, (user_id, message, action_type))
        assert len(msg1[1]) == 3
        assert len(msg2[1]) == 3
        user_ids = {msg1[1][0], msg2[1][0]}
        assert user_ids == {1, 2}

    @pytest.mark.asyncio
    async def test_step_jev_inverted_prefix_byte_identity(self) -> None:
        """Verify that when 2 agents evaluate the same post, the prefix slice is byte-identical."""
        graph = AgentGraph(backend="igraph")
        a1 = DummyAgent(1, "alice")
        a2 = DummyAgent(2, "bob")
        graph.add_agent(a1)
        graph.add_agent(a2)

        env = JEVEnvironment(env_or_graph=graph)

        post = {
            "post_id": 999,
            "author_name": "eve",
            "topic": "security",
            "content": "Zero-day vulnerability patched in openSSL.",
        }
        feeds = {1: [post], 2: [post]}

        # Intercept EvalItems submitted to classifier
        captured_items = []
        original_classify = env.classifier_client.classify_batch

        async def capture_classify(items, generate_comments=False):
            captured_items.extend(items)
            return await original_classify(items, generate_comments)

        env.classifier_client.classify_batch = capture_classify  # type: ignore

        await env.step_jev(step_index=1, agent_feeds=feeds)

        assert len(captured_items) == 2
        item1, item2 = captured_items[0], captured_items[1]
        assert item1.post_id == 999
        assert item2.post_id == 999

        # Post prefix part must be 100% byte-for-byte identical
        prefix1 = item1.full_prompt.split("[OBSERVER]:")[0]
        prefix2 = item2.full_prompt.split("[OBSERVER]:")[0]
        assert prefix1.encode("utf-8") == prefix2.encode("utf-8")
        assert prefix1 == prefix2
        assert "[POST ID: 999]" in prefix1

    @pytest.mark.asyncio
    async def test_step_jev_empty_agent_graph_and_feeds(self) -> None:
        """Verify robust handling of empty graphs and empty feeds without errors."""
        empty_graph = AgentGraph(backend="igraph")
        env = JEVEnvironment(env_or_graph=empty_graph)

        res = await env.step_jev(step_index=0)
        assert res.total_evaluations == 0
        assert res.num_actions == 0
        assert len(res.scheduled_actions) == 0

        # Graph with agents but empty feeds
        agent = DummyAgent(10, "empty_user")
        empty_graph.add_agent(agent)
        res2 = await env.step_jev(step_index=1, agent_feeds={10: []})
        assert res2.total_evaluations == 0
        assert res2.num_actions == 0

    @pytest.mark.asyncio
    async def test_step_jev_active_agent_ids_filtering(self) -> None:
        """Verify that passing active_agent_ids restricts evaluation strictly to that subset."""
        graph = AgentGraph(backend="igraph")
        for i in range(5):
            graph.add_agent(DummyAgent(i, f"agent_{i}"))

        env = JEVEnvironment(env_or_graph=graph)
        feeds = {
            i: [{"post_id": 1, "topic": "test", "content": "post"}]
            for i in range(5)
        }

        # Filter to only agents 1 and 3
        res = await env.step_jev(
            step_index=0,
            agent_feeds=feeds,
            active_agent_ids=[1, 3],
        )

        assert res.total_evaluations == 2
        evaluated_uids = {sa.user_id for sa in res.scheduled_actions}
        assert evaluated_uids == {1, 3}

    @pytest.mark.asyncio
    async def test_step_jev_batch_chunking(self) -> None:
        """Verify that when batch_size < total items, items are correctly chunked."""
        graph = AgentGraph(backend="igraph")
        for i in range(10):
            graph.add_agent(DummyAgent(i))

        env = JEVEnvironment(
            env_or_graph=graph,
            config=JEVExecutionConfig(batch_size=3),
        )
        feeds = {
            i: [{"post_id": 100 + i, "topic": "t", "content": "c"}]
            for i in range(10)
        }

        chunk_sizes = []
        original_classify = env.classifier_client.classify_batch

        async def track_chunks(items, generate_comments=False):
            chunk_sizes.append(len(items))
            return await original_classify(items, generate_comments)

        env.classifier_client.classify_batch = track_chunks  # type: ignore

        res = await env.step_jev(step_index=0, agent_feeds=feeds)
        assert res.total_evaluations == 10
        # 10 items chunked by 3 -> 3, 3, 3, 1
        assert chunk_sizes == [3, 3, 3, 1]


# ==============================================================================
# 4. Backward Compatibility with OasisEnv
# ==============================================================================


class TestOasisEnvBackwardCompatibility:
    """Verify seamless backward compatibility with OasisEnv."""

    @pytest.mark.asyncio
    async def test_oasis_env_step_jev_convenience_method(self) -> None:
        """Verify OasisEnv instance directly supports step_jev."""
        graph = AgentGraph(backend="igraph")
        graph.add_agent(DummyAgent(1, "direct_agent"))

        mock_plat = MockPlatform()
        oasis_env = OasisEnv.__new__(OasisEnv)
        oasis_env.agent_graph = graph
        oasis_env.platform = mock_plat
        oasis_env.channel = mock_plat.channel
        oasis_env.database_path = ":memory:"
        oasis_env.llm_semaphore = asyncio.Semaphore(10)
        oasis_env.platform_type = DefaultPlatformType.TWITTER
        oasis_env.platform_task = None

        feeds = {1: [{"post_id": 5, "topic": "news", "content": "Breaking news"}]}
        result = await oasis_env.step_jev(step_index=0, agent_feeds=feeds)

        assert isinstance(result, JEVStepResult)
        assert result.total_evaluations == 1
        assert hasattr(oasis_env, "_jev_engine")

    @pytest.mark.asyncio
    async def test_jev_environment_step_routes_camel_actions(self) -> None:
        """Verify JEVEnvironment.step routes to CAMEL loop when actions dict is provided."""
        graph = AgentGraph(backend="igraph")
        env = JEVEnvironment(env_or_graph=graph)

        mock_camel_step = AsyncMock()
        env._oasis_env = MagicMock()
        env._oasis_env.step = mock_camel_step

        actions = {MagicMock(): ManualAction(ActionType.DO_NOTHING, {})}
        await env.step(actions)

        mock_camel_step.assert_awaited_once_with(actions)

    @pytest.mark.asyncio
    async def test_default_oasis_channel_formatter(self) -> None:
        """Verify default_oasis_channel_formatter yields (user_id, message, action_type)."""
        sa = ScheduledAction(
            sort_timestamp=10.0,
            iso_timestamp="2026-09-26T00:00:10Z",
            user_id=7,
            action_dict={
                "action_type": ActionType.LIKE_POST,
                "message": 42,
            },
        )
        formatted = default_oasis_channel_formatter(sa)
        assert formatted == (7, 42, ActionType.LIKE_POST)


# ==============================================================================
# 5. AST & Guardrail Verification (Zero SQLite Imports)
# ==============================================================================


class TestASTGuardrailsForTrack5:
    """Verify hermeticity, thread limits, zero sqlite3 imports, and clean AST."""

    def test_zero_sqlite3_imports_in_jev_env_source(self) -> None:
        """Ensure oasis/environment/jev_env.py has strictly zero sqlite3 imports."""
        oasis_root = Path(__file__).resolve().parents[2]
        file_path = oasis_root / "oasis/environment/jev_env.py"

        assert file_path.exists(), f"Track 5 file missing: {file_path}"
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "sqlite3" not in alias.name, (
                        f"Forbidden 'import {alias.name}' detected in {file_path}:{node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "sqlite3" not in mod, (
                    f"Forbidden 'from {mod} import ...' detected in {file_path}:{node.lineno}"
                )

    def test_zero_raw_sql_in_jev_env_source(self) -> None:
        """Verify no raw SQL query string literals exist in jev_env.py."""
        oasis_root = Path(__file__).resolve().parents[2]
        file_path = oasis_root / "oasis/environment/jev_env.py"

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
                        f"Forbidden raw SQL query literal '{fragment}' detected in {file_path}:{node.lineno}"
                    )
