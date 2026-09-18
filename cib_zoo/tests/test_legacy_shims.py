"""Unit tests for backward-compatible legacy attack script shims."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from oasis.social_platform.typing import ActionType

from cib_zoo.comment_raid import execute_comment_raid
from cib_zoo.hashtag_hijacking import execute_hashtag_hijacking
from cib_zoo.like_farm import execute_like_farm
from cib_zoo.repost_botnet import execute_repost_botnet


class MockAgent:
    def __init__(self, agent_id: int):
        self.social_agent_id = agent_id

    def filter_actions(self, actions):
        return actions


@pytest.fixture
def mock_env():
    env = MagicMock()
    agents = {1: MockAgent(1), 2: MockAgent(2)}
    env.agent_graph.get_agent.side_effect = lambda aid: agents[aid]
    env.step = AsyncMock()
    return env


@pytest.mark.asyncio
async def test_like_farm_shim(mock_env) -> None:
    with pytest.deprecated_call(match="execute_like_farm is deprecated"):
        await execute_like_farm(mock_env, bot_ids=[1, 2], target_post_ids=[10, 20])

    mock_env.step.assert_awaited_once()
    actions_dict = mock_env.step.call_args[0][0]
    assert len(actions_dict) == 2
    for agent, acts in actions_dict.items():
        assert len(acts) == 2
        assert acts[0].action_type == ActionType.LIKE_POST
        assert acts[0].action_args["post_id"] == 10
        assert acts[1].action_args["post_id"] == 20


@pytest.mark.asyncio
async def test_comment_raid_shim(mock_env) -> None:
    with pytest.deprecated_call(match="execute_comment_raid is deprecated"):
        await execute_comment_raid(
            mock_env,
            bot_ids=[1],
            target_post_ids=[15],
            templates=["Raid content"],
        )

    mock_env.step.assert_awaited_once()
    actions_dict = mock_env.step.call_args[0][0]
    agent = list(actions_dict.keys())[0]
    assert actions_dict[agent][0].action_type == ActionType.CREATE_COMMENT
    assert actions_dict[agent][0].action_args["content"] == "Raid content"


@pytest.mark.asyncio
async def test_repost_botnet_shim(mock_env) -> None:
    with pytest.deprecated_call(match="execute_repost_botnet is deprecated"):
        await execute_repost_botnet(mock_env, bot_ids=[1], seed_post_id=30, depth=2)

    mock_env.step.assert_awaited_once()
    actions_dict = mock_env.step.call_args[0][0]
    agent = list(actions_dict.keys())[0]
    assert len(actions_dict[agent]) == 2
    assert actions_dict[agent][0].action_type == ActionType.QUOTE_POST
    assert actions_dict[agent][0].action_args["post_id"] == 30


@pytest.mark.asyncio
async def test_hashtag_hijacking_shim(mock_env) -> None:
    with pytest.deprecated_call(match="execute_hashtag_hijacking is deprecated"):
        await execute_hashtag_hijacking(
            mock_env,
            bot_ids=[1],
            target_hashtag="#crypto",
            num_posts=3,
        )

    mock_env.step.assert_awaited_once()
    actions_dict = mock_env.step.call_args[0][0]
    agent = list(actions_dict.keys())[0]
    assert len(actions_dict[agent]) == 3
    for act in actions_dict[agent]:
        assert act.action_type == ActionType.CREATE_POST
        assert "#crypto" in act.action_args["content"]
