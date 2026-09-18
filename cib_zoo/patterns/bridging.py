"""Bridging pattern for graph connectivity manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

from oasis.environment.env_action import ManualAction

from cib_zoo.patterns.base import BasePattern
from cib_zoo.primitives.follow import FollowPrimitive
from cib_zoo.primitives.like import LikePrimitive

if TYPE_CHECKING:
    from cib_zoo.agent.cib_agent import CIBAgent


class BridgingPattern(BasePattern):
    """Bridging attack pattern supporting both micro-influencer targeting and inter-squad bridging.

    Modes:
    1. 'influencer': Targets key organic micro-influencers to pull their followers
       into 2-hop graph recommendations (e.g. TwHIN-BERT / SimRank).
    2. 'inter_squad': Coordinates cross-follows and cross-engagement between distinct bot squads
       to evade community detection algorithms (e.g. Infomap, Louvain) and lower modularity.
    """

    def __init__(
        self,
        mode: Literal["influencer", "inter_squad"] = "influencer",
        influencer_ids: Optional[list[int]] = None,
        target_squad_bot_ids: Optional[list[int]] = None,
        interact_post_ids: Optional[list[int]] = None,
    ) -> None:
        if mode not in ("influencer", "inter_squad"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'influencer' or 'inter_squad'.")

        if mode == "influencer":
            if not influencer_ids:
                raise ValueError("influencer_ids list cannot be empty in 'influencer' mode.")
            self.influencer_ids = [int(x) for x in influencer_ids if int(x) > 0]
        else:
            if not target_squad_bot_ids:
                raise ValueError("target_squad_bot_ids cannot be empty in 'inter_squad' mode.")
            self.target_squad_bot_ids = [int(x) for x in target_squad_bot_ids if int(x) > 0]

        self.mode = mode
        self.influencer_ids = influencer_ids or []
        self.target_squad_bot_ids = target_squad_bot_ids or []
        self.interact_post_ids = interact_post_ids or []
        self._follow_primitive = FollowPrimitive()

    def generate_step_actions(
        self,
        step: int,
        squad_bots: list[CIBAgent],
        context: dict[str, Any],
    ) -> dict[int, list[ManualAction]]:
        raw: dict[int, list[ManualAction]] = {}

        if self.mode == "influencer":
            for i, bot in enumerate(squad_bots):
                target_inf = self.influencer_ids[(step + i) % len(self.influencer_ids)]
                actions = [self._follow_primitive.generate(followee_id=target_inf)]
                if self.interact_post_ids:
                    target_post = self.interact_post_ids[(step + i) % len(self.interact_post_ids)]
                    actions.append(LikePrimitive(post_id=target_post).generate())
                raw[bot.social_agent_id] = actions

        elif self.mode == "inter_squad":
            for i, bot in enumerate(squad_bots):
                target_peer = self.target_squad_bot_ids[(step + i) % len(self.target_squad_bot_ids)]
                actions = [self._follow_primitive.generate(followee_id=target_peer)]
                if self.interact_post_ids:
                    target_post = self.interact_post_ids[(step + i) % len(self.interact_post_ids)]
                    actions.append(LikePrimitive(post_id=target_post).generate())
                raw[bot.social_agent_id] = actions

        return self.sanitize_actions_for_bots(raw, squad_bots)
