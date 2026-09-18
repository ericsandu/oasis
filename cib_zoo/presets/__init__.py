"""CIB Declarative Campaign DSL and Presets."""

from cib_zoo.presets.campaign_builder import CampaignBuilder
from cib_zoo.presets.s1_retrieval_poisoning import create_s1_campaign
from cib_zoo.presets.s2_adaptive_ranking import create_s2_campaign
from cib_zoo.presets.s3_cascaded_presentation import create_s3_campaign
from cib_zoo.presets.schema import CampaignConfig, PhaseConfig, SquadConfig

__all__ = [
    "CampaignConfig",
    "PhaseConfig",
    "SquadConfig",
    "CampaignBuilder",
    "create_s1_campaign",
    "create_s2_campaign",
    "create_s3_campaign",
]
