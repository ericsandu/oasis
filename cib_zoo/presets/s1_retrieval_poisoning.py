"""Preset S1: Multi-Channel Retrieval Poisoning Campaign.

Phase 1: Warmup & Bridging (sleeper bots follow key micro-influencers and age organically).
Phase 2: Strike (synchronized co-engagement poisoning CF items + astroturf payload broadcast).
"""

from __future__ import annotations

from typing import Optional

from cib_zoo.patterns.astroturf import AstroturfPattern
from cib_zoo.patterns.bridging import BridgingPattern
from cib_zoo.patterns.co_engagement import CoEngagementPattern
from cib_zoo.presets.campaign_builder import CampaignBuilder
from cib_zoo.presets.schema import CampaignConfig


def create_s1_campaign(
    warmup_bot_ids: list[int],
    strike_bot_ids: list[int],
    influencer_ids: list[int],
    anchor_post_id: int,
    payload_post_id: int,
    payload_template: str,
    hashtag: str = "breaking",
    warmup_steps: int = 5,
    strike_steps: int = 5,
) -> CampaignConfig:
    """Build Preset S1: Multi-Channel Retrieval Poisoning Campaign."""
    builder = CampaignBuilder.create("S1_MultiChannelRetrievalPoisoning")

    # Phase 1: Warmup & Bridging
    builder.add_phase("Warmup_Phase", duration_steps=warmup_steps)
    builder.add_squad(
        squad_id="squad_warmup_bridging",
        bot_ids=warmup_bot_ids,
        pattern=BridgingPattern(mode="influencer", influencer_ids=influencer_ids),
        max_actions_per_step=1,
    )

    # Phase 2: Synchronized Strike
    builder.add_phase("Strike_Phase", duration_steps=strike_steps)
    # Co-engagement squad poisons collaborative filtering
    builder.add_squad(
        squad_id="squad_co_engagement",
        bot_ids=strike_bot_ids[: len(strike_bot_ids) // 2 or 1],
        pattern=CoEngagementPattern(
            anchor_post_id=anchor_post_id,
            payload_post_id=payload_post_id,
        ),
        max_actions_per_step=2,
    )
    # Astroturf squad broadcasts narrative
    builder.add_squad(
        squad_id="squad_astroturf",
        bot_ids=strike_bot_ids[len(strike_bot_ids) // 2 or 1 :],
        pattern=AstroturfPattern(
            hashtag=hashtag,
            payload_template=payload_template,
        ),
        max_actions_per_step=1,
    )

    return builder.build()
