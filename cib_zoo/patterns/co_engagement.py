"""Co-engagement attack pattern for collaborative filtering poisoning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oasis.environment.env_action import ManualAction

from cib_zoo.patterns.base import BasePattern
from cib_zoo.primitives.like import LikePrimitive

if TYPE_CHECKING:
    from cib_zoo.agent.cib_agent import CIBAgent


class CoEngagementPattern(BasePattern):
    """Coordinates squad bots to simultaneously co-like anchor and payload posts.

    Targeting collaborative filtering recommender systems (e.g. Gorse, item-item CF),
    co-liking elevates the co-occurrence weight between high-affinity organic anchor
    content and the target payload.
    """

    def __init__(
        self,
        anchor_post_id: int,
        payload_post_id: int,
        ratio: float = 1.0,
    ) -> None:
        if anchor_post_id <= 0:
            raise ValueError(f"anchor_post_id must be positive, got {anchor_post_id}")
        if payload_post_id <= 0:
            raise ValueError(f"payload_post_id must be positive, got {payload_post_id}")
        if anchor_post_id == payload_post_id:
            raise ValueError("anchor_post_id and payload_post_id must be distinct.")
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0.0, 1.0], got {ratio}")

        self.anchor_post_id = int(anchor_post_id)
        self.payload_post_id = int(payload_post_id)
        self.ratio = ratio
        self._like_anchor = LikePrimitive(post_id=self.anchor_post_id)
        self._like_payload = LikePrimitive(post_id=self.payload_post_id)

    def generate_step_actions(
        self,
        step: int,
        squad_bots: list[CIBAgent],
        context: dict[str, Any],
    ) -> dict[int, list[ManualAction]]:
        raw: dict[int, list[ManualAction]] = {}
        active_count = int(len(squad_bots) * self.ratio)

        for i, bot in enumerate(squad_bots):
            if i >= active_count:
                raw[bot.social_agent_id] = []
                continue

            actions = [
                self._like_anchor.generate(),
                self._like_payload.generate(),
            ]
            raw[bot.social_agent_id] = actions

        return self.sanitize_actions_for_bots(raw, squad_bots)
