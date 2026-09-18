"""Unit tests for composite CIB attack patterns."""

import pytest
from oasis.social_platform.typing import ActionType

from cib_zoo.patterns import (
    AstroturfPattern,
    BridgingPattern,
    CoEngagementPattern,
    ReplyRaidPattern,
    SleeperAgingPattern,
)


class TestCoEngagementPattern:
    def test_co_engagement_generates_pairs(self, cib_agent_factory) -> None:
        bots = [cib_agent_factory(agent_id=i, max_actions_per_step=5) for i in range(1, 4)]
        pattern = CoEngagementPattern(anchor_post_id=10, payload_post_id=20)
        
        actions = pattern.generate_step_actions(step=0, squad_bots=bots, context={})
        assert len(actions) == 3
        for bot in bots:
            bot_actions = actions[bot.social_agent_id]
            assert len(bot_actions) == 2
            assert bot_actions[0].action_type == ActionType.LIKE_POST
            assert bot_actions[0].action_args["post_id"] == 10
            assert bot_actions[1].action_type == ActionType.LIKE_POST
            assert bot_actions[1].action_args["post_id"] == 20

    def test_co_engagement_respects_agent_step_budget(self, cib_agent_factory) -> None:
        # Bot limited to 1 action per step: should only get the first action
        bot = cib_agent_factory(agent_id=1, max_actions_per_step=1)
        pattern = CoEngagementPattern(anchor_post_id=10, payload_post_id=20)
        actions = pattern.generate_step_actions(step=0, squad_bots=[bot], context={})
        assert len(actions[1]) == 1
        assert actions[1][0].action_args["post_id"] == 10


class TestReplyRaidPattern:
    def test_reply_raid_distribution(self, cib_agent_factory) -> None:
        bots = [cib_agent_factory(agent_id=i) for i in range(1, 4)]
        templates = ["Template A", "Template B", "Template C"]
        pattern = ReplyRaidPattern(target_post_id=50, templates=templates, distribution="round_robin")

        actions = pattern.generate_step_actions(step=0, squad_bots=bots, context={})
        contents = [actions[b.social_agent_id][0].action_args["content"] for b in bots]
        assert contents == ["Template A", "Template B", "Template C"]


class TestAstroturfPattern:
    def test_astroturf_post_generation(self, cib_agent_factory) -> None:
        bots = [cib_agent_factory(agent_id=1)]
        pattern = AstroturfPattern(hashtag="#viralnews", payload_template="Breaking narrative details!")

        actions = pattern.generate_step_actions(step=0, squad_bots=bots, context={})
        action = actions[1][0]
        assert action.action_type == ActionType.CREATE_POST
        assert "#viralnews" in action.action_args["content"]
        assert "Breaking narrative details!" in action.action_args["content"]


class TestBridgingPattern:
    def test_bridging_influencer_mode(self, cib_agent_factory) -> None:
        bots = [cib_agent_factory(agent_id=i) for i in range(1, 3)]
        pattern = BridgingPattern(mode="influencer", influencer_ids=[999, 888])

        actions = pattern.generate_step_actions(step=0, squad_bots=bots, context={})
        assert actions[1][0].action_type == ActionType.FOLLOW
        assert actions[1][0].action_args["followee_id"] in [999, 888]

    def test_bridging_inter_squad_mode(self, cib_agent_factory) -> None:
        squad_a_bots = [cib_agent_factory(agent_id=1)]
        pattern = BridgingPattern(mode="inter_squad", target_squad_bot_ids=[100, 101])

        actions = pattern.generate_step_actions(step=0, squad_bots=squad_a_bots, context={})
        assert actions[1][0].action_type == ActionType.FOLLOW
        assert actions[1][0].action_args["followee_id"] in [100, 101]


class TestSleeperAgingPattern:
    def test_sleeper_aging_transition(self, cib_agent_factory) -> None:
        bot = cib_agent_factory(agent_id=1, max_actions_per_step=5)
        warmup = AstroturfPattern(hashtag="general", payload_template="Nice weather today.")
        strike = CoEngagementPattern(anchor_post_id=1, payload_post_id=2)
        pattern = SleeperAgingPattern(
            warmup_steps=3,
            strike_steps=2,
            warmup_behavior=warmup,
            strike_behavior=strike,
        )

        assert not pattern.is_in_strike_phase(step=0)
        assert not pattern.is_in_strike_phase(step=2)
        assert pattern.is_in_strike_phase(step=3)

        # Step 1 (warmup) emits CREATE_POST
        act_warmup = pattern.generate_step_actions(step=1, squad_bots=[bot], context={})
        assert act_warmup[1][0].action_type == ActionType.CREATE_POST

        # Step 3 (strike) emits LIKE_POST
        act_strike = pattern.generate_step_actions(step=3, squad_bots=[bot], context={})
        assert act_strike[1][0].action_type == ActionType.LIKE_POST
