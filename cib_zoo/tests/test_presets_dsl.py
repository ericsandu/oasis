"""Unit tests for declarative campaign DSL, builder, and pre-built presets S1, S2, S3."""

import pytest
from oasis.social_platform.typing import ActionType

from cib_zoo.patterns import (
    AstroturfPattern,
    BridgingPattern,
    CoEngagementPattern,
    ReplyRaidPattern,
)
from cib_zoo.presets import (
    CampaignBuilder,
    CampaignConfig,
    PhaseConfig,
    SquadConfig,
    create_s1_campaign,
    create_s2_campaign,
    create_s3_campaign,
)


class TestCampaignSchema:
    def test_squad_config_validation(self) -> None:
        pattern = CoEngagementPattern(anchor_post_id=1, payload_post_id=2)
        squad = SquadConfig(squad_id="alpha", bot_ids=[1, 2, 3], pattern=pattern)
        assert squad.squad_id == "alpha"
        assert squad.bot_ids == [1, 2, 3]

        with pytest.raises(ValueError, match="squad_id must be non-empty"):
            SquadConfig(squad_id="", bot_ids=[1], pattern=pattern)

        with pytest.raises(ValueError, match="bot_ids cannot be empty"):
            SquadConfig(squad_id="alpha", bot_ids=[], pattern=pattern)

    def test_campaign_phase_navigation(self) -> None:
        builder = CampaignBuilder.create("TestNavigation")
        pattern = CoEngagementPattern(anchor_post_id=1, payload_post_id=2)

        # Phase 1: 3 steps [0, 1, 2]
        builder.add_phase("Phase_1", duration_steps=3)
        builder.add_squad("s1", bot_ids=[1, 2], pattern=pattern)

        # Phase 2: 4 steps [3, 4, 5, 6]
        builder.add_phase("Phase_2", duration_steps=4)
        builder.add_squad("s2", bot_ids=[3, 4], pattern=pattern)

        campaign = builder.build()
        assert campaign.total_steps == 7
        assert campaign.get_all_bot_ids() == {1, 2, 3, 4}

        # Step 0 -> Phase 1, relative step 0
        p, rel = campaign.get_active_phase(0)
        assert p.phase_name == "Phase_1"
        assert rel == 0

        # Step 2 -> Phase 1, relative step 2
        p, rel = campaign.get_active_phase(2)
        assert p.phase_name == "Phase_1"
        assert rel == 2

        # Step 3 -> Phase 2, relative step 0
        p, rel = campaign.get_active_phase(3)
        assert p.phase_name == "Phase_2"
        assert rel == 0

        # Step 6 -> Phase 2, relative step 3
        p, rel = campaign.get_active_phase(6)
        assert p.phase_name == "Phase_2"
        assert rel == 3

        with pytest.raises(IndexError, match="out of campaign bounds"):
            campaign.get_active_phase(7)


class TestPresets:
    def test_preset_s1_structure(self) -> None:
        campaign = create_s1_campaign(
            warmup_bot_ids=[1, 2],
            strike_bot_ids=[3, 4, 5, 6],
            influencer_ids=[99],
            anchor_post_id=10,
            payload_post_id=20,
            payload_template="Coordinated strike payload",
            warmup_steps=3,
            strike_steps=4,
        )
        assert campaign.campaign_name == "S1_MultiChannelRetrievalPoisoning"
        assert campaign.total_steps == 7
        assert len(campaign.phases) == 2
        assert campaign.phases[0].phase_name == "Warmup_Phase"
        assert campaign.phases[1].phase_name == "Strike_Phase"
        assert len(campaign.phases[1].squads) == 2

    def test_preset_s2_structure(self) -> None:
        campaign = create_s2_campaign(
            sentinel_bot_ids=[1],
            strike_bot_ids=[2, 3],
            target_post_id=42,
            raid_templates=["Comment A", "Comment B"],
            discovery_steps=2,
            strike_steps=5,
        )
        assert campaign.campaign_name == "S2_AdaptiveRankingSubversion"
        assert campaign.total_steps == 7
        assert campaign.phases[1].squads[0].modifier is not None

    def test_preset_s3_structure(self) -> None:
        campaign = create_s3_campaign(
            bridge_bot_ids=[1],
            raid_bot_ids=[2],
            astroturf_bot_ids=[3],
            target_influencer_ids=[88],
            target_post_id=100,
            raid_templates=["Raid reply"],
            astroturf_hashtag="trending_topic",
            astroturf_template="Astroturf message",
            bridge_steps=2,
            raid_steps=3,
            astroturf_steps=4,
        )
        assert campaign.campaign_name == "S3_CascadedPresentationCapture"
        assert campaign.total_steps == 9
        assert len(campaign.phases) == 3
        assert campaign.phases[0].phase_name == "Bridge_Phase"
        assert campaign.phases[1].phase_name == "Raid_Phase"
        assert campaign.phases[2].phase_name == "Astroturf_Phase"
