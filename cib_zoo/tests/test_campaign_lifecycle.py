"""Tests for multi-phase campaign lifecycles (15+ steps) and budget/policy state transitions."""

import pytest
from cib_zoo.agent.budget import BudgetLimiter
from cib_zoo.modifiers.bandit import ThompsonSamplingBanditModifier
from cib_zoo.patterns.astroturf import AstroturfPattern
from cib_zoo.patterns.co_engagement import CoEngagementPattern
from cib_zoo.patterns.sleeper_aging import SleeperAgingPattern
from cib_zoo.presets.s1_retrieval_poisoning import create_s1_campaign
from oasis.social_platform.typing import ActionType


def test_s1_campaign_15_step_phase_transitions():
    """Verify that Preset S1 correctly executes Warmup (steps 0-4) then transitions to Strike (steps 5-14)."""
    campaign = create_s1_campaign(
        warmup_bot_ids=[1, 2, 3],
        strike_bot_ids=[4, 5, 6, 7],
        influencer_ids=[100, 101],
        anchor_post_id=10,
        payload_post_id=20,
        payload_template="Test payload #breaking",
        warmup_steps=5,
        strike_steps=10,
    )

    assert campaign.total_steps == 15

    # Phase 1: Warmup (steps 0 to 4)
    for step in range(0, 5):
        phase, rel_step = campaign.get_active_phase(step)
        assert phase is not None
        assert phase.phase_name == "Warmup_Phase"
        assert rel_step == step
        squad_names = [s.squad_id for s in phase.squads]
        assert "squad_warmup_bridging" in squad_names

    # Phase 2: Strike (steps 5 to 14)
    for step in range(5, 15):
        phase, rel_step = campaign.get_active_phase(step)
        assert phase is not None
        assert phase.phase_name == "Strike_Phase"
        assert rel_step == step - 5
        squad_names = [s.squad_id for s in phase.squads]
        assert "squad_co_engagement" in squad_names
        assert "squad_astroturf" in squad_names


def test_budget_limiter_across_15_steps():
    """Verify BudgetLimiter correctly resets per-step budget each step while decrementing total lifetime budget."""
    max_per_step = 2
    total_budget = 20
    limiter = BudgetLimiter(max_actions_per_step=max_per_step, total_budget=total_budget)

    actions_executed = 0

    for step in range(15):
        limiter.reset_step()
        assert limiter.remaining_step_budget == min(max_per_step, limiter.remaining_total_budget or 0)

        # Attempt 3 actions each step (exceeding step cap)
        for _ in range(3):
            if limiter.can_execute(ActionType.LIKE_POST):
                limiter.record_action(ActionType.LIKE_POST)
                actions_executed += 1

        # Must never exceed max_per_step in any single step
        assert limiter.step_action_count <= max_per_step

    # Total actions executed must be exactly capped by total_budget
    assert actions_executed == total_budget
    assert limiter.is_exhausted


def test_sleeper_aging_pattern_state_switch(cib_agent_factory):
    """Verify SleeperAgingPattern generates benign actions in warmup and strike actions after switch_step."""
    bot = cib_agent_factory(agent_id=1, max_actions_per_step=5)
    warmup = AstroturfPattern(hashtag="general", payload_template="Nice weather today.")
    strike = CoEngagementPattern(anchor_post_id=1, payload_post_id=2)

    pattern = SleeperAgingPattern(
        warmup_steps=6,
        strike_steps=6,
        warmup_behavior=warmup,
        strike_behavior=strike,
    )

    # Pre-switch steps (0 to 5)
    for step in range(6):
        assert not pattern.is_in_strike_phase(step=step)
        actions = pattern.generate_step_actions(step=step, squad_bots=[bot], context={})
        assert actions[1][0].action_type == ActionType.CREATE_POST

    # Post-switch steps (6 to 11)
    for step in range(6, 12):
        assert pattern.is_in_strike_phase(step=step)
        actions = pattern.generate_step_actions(step=step, squad_bots=[bot], context={})
        assert actions[1][0].action_type == ActionType.LIKE_POST


def test_bandit_dynamic_convergence_across_15_steps():
    """Verify ThompsonSamplingBanditModifier dynamically updates posteriors across 15 steps when one arm yields higher exposure."""
    arm_names = [ActionType.LIKE_POST.value, ActionType.REPOST.value, ActionType.CREATE_COMMENT.value]
    bandit = ThompsonSamplingBanditModifier(arms=arm_names, random_state=42)

    # Arm 0 (LIKE_POST) is given consistent positive reward
    for step in range(15):
        sampled_arm = bandit.sample_arm()
        assert sampled_arm in arm_names

        # Simulate reward: high reward for LIKE_POST, zero for others
        reward = 1.0 if sampled_arm == ActionType.LIKE_POST.value else 0.0
        bandit.update(sampled_arm, reward)

    # After 15 steps of positive reinforcement, LIKE_POST alpha should dominate
    stats = bandit.get_arm_stats()
    like_stats = stats[ActionType.LIKE_POST.value]
    repost_stats = stats[ActionType.REPOST.value]

    assert like_stats["alpha"] > repost_stats["alpha"]
    assert like_stats["expected_value"] > repost_stats["expected_value"]
