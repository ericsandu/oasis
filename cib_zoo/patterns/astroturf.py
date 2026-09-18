"""Astroturf pattern for synthetic trending narrative push."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from oasis.environment.env_action import ManualAction

from cib_zoo.patterns.base import BasePattern
from cib_zoo.primitives.post import PostPrimitive

if TYPE_CHECKING:
    from cib_zoo.agent.cib_agent import CIBAgent


class AstroturfPattern(BasePattern):
    """Coordinates squad bots to post synthetic narrative messages around hashtags."""

    def __init__(
        self,
        hashtag: str,
        payload_template: str | list[str],
        ephemeral_ttl: Optional[int] = None,
    ) -> None:
        clean_tag = hashtag.strip().lstrip("#")
        if not clean_tag:
            raise ValueError("hashtag must be non-empty.")

        if isinstance(payload_template, str):
            if not payload_template.strip():
                raise ValueError("payload_template must be non-empty.")
            self.templates = [payload_template]
        elif isinstance(payload_template, list):
            if not payload_template:
                raise ValueError("payload_template list cannot be empty.")
            self.templates = list(payload_template)
        else:
            raise TypeError("payload_template must be a string or list of strings.")

        self.hashtag = clean_tag
        self.ephemeral_ttl = ephemeral_ttl
        self._post_primitive = PostPrimitive(hashtags=[self.hashtag])

    def generate_step_actions(
        self,
        step: int,
        squad_bots: list[CIBAgent],
        context: dict[str, Any],
    ) -> dict[int, list[ManualAction]]:
        raw: dict[int, list[ManualAction]] = {}

        for i, bot in enumerate(squad_bots):
            template = self.templates[(step + i) % len(self.templates)]
            action = self._post_primitive.generate(
                content=template,
                bot_id=bot.social_agent_id,
                step=step,
            )
            raw[bot.social_agent_id] = [action]

        return self.sanitize_actions_for_bots(raw, squad_bots)
