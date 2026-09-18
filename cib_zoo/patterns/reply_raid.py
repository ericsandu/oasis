"""Reply raid pattern for conversation and visual slot hijacking."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal

from oasis.environment.env_action import ManualAction

from cib_zoo.patterns.base import BasePattern
from cib_zoo.primitives.comment import CommentPrimitive

if TYPE_CHECKING:
    from cib_zoo.agent.cib_agent import CIBAgent


class ReplyRaidPattern(BasePattern):
    """Floods comments on target organic posts to capture conversation threads."""

    def __init__(
        self,
        target_post_id: int,
        templates: list[str],
        distribution: Literal["round_robin", "random"] = "round_robin",
    ) -> None:
        if target_post_id <= 0:
            raise ValueError(f"target_post_id must be positive, got {target_post_id}")
        if not templates:
            raise ValueError("templates list cannot be empty.")

        self.target_post_id = int(target_post_id)
        self.templates = list(templates)
        self.distribution = distribution
        self._comment_primitive = CommentPrimitive(post_id=self.target_post_id)

    def generate_step_actions(
        self,
        step: int,
        squad_bots: list[CIBAgent],
        context: dict[str, Any],
    ) -> dict[int, list[ManualAction]]:
        raw: dict[int, list[ManualAction]] = {}

        for i, bot in enumerate(squad_bots):
            if self.distribution == "round_robin":
                template = self.templates[(step * len(squad_bots) + i) % len(self.templates)]
            else:
                template = random.choice(self.templates)

            action = self._comment_primitive.generate(
                content=template,
                bot_id=bot.social_agent_id,
                step=step,
            )
            raw[bot.social_agent_id] = [action]

        return self.sanitize_actions_for_bots(raw, squad_bots)
