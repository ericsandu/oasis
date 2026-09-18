"""Preset S3: Cascaded Presentation Capture Campaign.

Three-phase orchestration:
Phase 1: Bridge squad establishes 2-hop topological proximity to organic hubs.
Phase 2: Raid squad floods high-ranking discussion slots.
Phase 3: Astroturf squad generates volume burst to capture discovery feeds.
"""

from __future__ import annotations

from typing import Optional

from cib_zoo.patterns.astroturf import AstroturfPattern
from cib_zoo.patterns.bridging import BridgingPattern
from cib_zoo.patterns.reply_raid import ReplyRaidPattern
from cib_zoo.presets.campaign_builder import CampaignBuilder
from cib_zoo.presets.schema import CampaignConfig


def create_s3_campaign(
    bridge_bot_ids: list[int],
    raid_bot_ids: list[int],
    astroturf_bot_ids: list[int],
    target_influencer_ids: list[int],
    target_post_id: int,
    raid_templates: list[str],
    astroturf_hashtag: str,
    astroturf_template: str,
    bridge_steps: int = 3,
    raid_steps: int = 4,
    astroturf_steps: int = 5,
) -> CampaignConfig:
    """Build Preset S3: Cascaded Presentation Capture Campaign."""
    builder = CampaignBuilder.create("S3_CascadedPresentationCapture")

    # Phase 1: Bridge Squad
    builder.add_phase("Bridge_Phase", duration_steps=bridge_steps)
    builder.add_squad(
        squad_id="squad_bridge",
        bot_ids=bridge_bot_ids,
        pattern=BridgingPattern(mode="influencer", influencer_ids=target_influencer_ids),
        max_actions_per_step=1,
    )

    # Phase 2: Raid Squad
    builder.add_phase("Raid_Phase", duration_steps=raid_steps)
    builder.add_squad(
        squad_id="squad_raid",
        bot_ids=raid_bot_ids,
        pattern=ReplyRaidPattern(target_post_id=target_post_id, templates=raid_templates),
        max_actions_per_step=1,
    )

    # Phase 3: Astroturf Squad
    builder.add_phase("Astroturf_Phase", duration_steps=astroturf_steps)
    builder.add_squad(
        squad_id="squad_astroturf",
        bot_ids=astroturf_bot_ids,
        pattern=AstroturfPattern(hashtag=astroturf_hashtag, payload_template=astroturf_template),
        max_actions_per_step=1,
    )

    return builder.build()
