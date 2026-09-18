"""Base class for composite CIB attack patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from oasis.environment.env_action import ManualAction

if TYPE_CHECKING:
    from cib_zoo.agent.cib_agent import CIBAgent


class BasePattern(ABC):
    """Abstract base class for CIB attack patterns.

    A pattern coordinates multiple bots within a squad over simulation steps,
    emitting mapped ManualActions that adhere strictly to bot budgets and platform APIs.
    """

    @abstractmethod
    def generate_step_actions(
        self,
        step: int,
        squad_bots: list[CIBAgent],
        context: dict[str, Any],
    ) -> dict[int, list[ManualAction]]:
        """Generate actions for squad bots at a specific simulation step.

        Args:
            step: Current simulation step (0-indexed).
            squad_bots: List of CIBAgent instances in this squad.
            context: Shared simulation context (e.g. active posts, feeds, discovery results).

        Returns:
            dict mapping agent_id -> list of ManualAction instances.
        """
        raise NotImplementedError

    def sanitize_actions_for_bots(
        self,
        raw_actions: dict[int, list[ManualAction]],
        squad_bots: list[CIBAgent],
    ) -> dict[int, list[ManualAction]]:
        """Filter and clamp generated actions according to each bot's budget and quotas.

        Args:
            raw_actions: Mapping of agent_id -> proposed list of ManualActions.
            squad_bots: List of CIBAgent instances in this squad.

        Returns:
            dict mapping agent_id -> filtered list of executable ManualActions.
        """
        bot_map = {bot.social_agent_id: bot for bot in squad_bots}
        sanitized: dict[int, list[ManualAction]] = {}

        for agent_id, actions in raw_actions.items():
            bot = bot_map.get(agent_id)
            if bot is not None and hasattr(bot, "filter_actions"):
                sanitized[agent_id] = bot.filter_actions(actions)
            else:
                sanitized[agent_id] = actions

        return sanitized

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
