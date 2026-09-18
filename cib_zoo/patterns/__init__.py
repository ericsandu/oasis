"""CIB Composite Attack Patterns."""

from cib_zoo.patterns.astroturf import AstroturfPattern
from cib_zoo.patterns.base import BasePattern
from cib_zoo.patterns.bridging import BridgingPattern
from cib_zoo.patterns.co_engagement import CoEngagementPattern
from cib_zoo.patterns.reply_raid import ReplyRaidPattern
from cib_zoo.patterns.sleeper_aging import SleeperAgingPattern

__all__ = [
    "BasePattern",
    "CoEngagementPattern",
    "ReplyRaidPattern",
    "AstroturfPattern",
    "BridgingPattern",
    "SleeperAgingPattern",
]
