"""Preset S2: Adaptive Ranking Subversion Campaign.

Phase 1: Sentinel Discovery (bots query feeds/search to establish ranking baselines).
Phase 2: Adaptive Strike with Thompson Sampling Multi-Armed Bandit tuning across vectors.
"""

from __future__ import annotations

from typing import Optional

from cib_zoo.modifiers.bandit import ThompsonSamplingBanditModifier
from cib_zoo.patterns.astroturf import AstroturfPattern
from cib_zoo.patterns.reply_raid import ReplyRaidPattern
from cib_zoo.primitives.search import SearchPrimitive
from cib_zoo.presets.campaign_builder import CampaignBuilder
from cib_zoo.presets.schema import CampaignConfig


def create_s2_campaign(
    sentinel_bot_ids: list[int],
    strike_bot_ids: list[int],
    target_post_id: int,
    raid_templates: list[str],
    astroturf_hashtag: str = "counter_narrative",
    astroturf_template: str = "Important perspective on recent events.",
    discovery_steps: int = 3,
    strike_steps: int = 7,
    bandit_modifier: Optional[ThompsonSamplingBanditModifier] = None,
) -> CampaignConfig:
    """Build Preset S2: Adaptive Ranking Subversion Campaign."""
    builder = CampaignBuilder.create("S2_AdaptiveRankingSubversion")

    # Phase 1: Sentinel Observation & Perception Gathering
    # Sentinel bots refresh feeds and search users/posts to track rank trajectory
    builder.add_phase("Sentinel_Discovery", duration_steps=discovery_steps)
    builder.add_squad(
        squad_id="squad_sentinels",
        bot_ids=sentinel_bot_ids,
        pattern=ReplyRaidPattern(
            target_post_id=target_post_id,
            templates=["Observing discussion context..."],
        ),
        max_actions_per_step=1,
    )

    # Phase 2: Adaptive Strike
    # Modifiers dynamically tune arm allocation between reply raid and astroturf
    effective_bandit = bandit_modifier or ThompsonSamplingBanditModifier(
        arms=["reply_raid", "astroturf"]
    )

    builder.add_phase("Adaptive_Strike", duration_steps=strike_steps)
    builder.add_squad(
        squad_id="squad_adaptive_raid",
        bot_ids=strike_bot_ids,
        pattern=ReplyRaidPattern(
            target_post_id=target_post_id,
            templates=raid_templates,
        ),
        modifier=effective_bandit,
        max_actions_per_step=1,
    )

    return builder.build()
