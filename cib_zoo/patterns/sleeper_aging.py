"""Sleeper aging two-phase pattern (warmup burn-in -> synchronized strike)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oasis.environment.env_action import ManualAction

from cib_zoo.patterns.base import BasePattern

if TYPE_CHECKING:
    from cib_zoo.agent.cib_agent import CIBAgent


class SleeperAgingPattern(BasePattern):
    """Executes a two-phase campaign: benign warmup aging, then high-impact synchronized strike."""

    def __init__(
        self,
        warmup_steps: int,
        strike_steps: int,
        warmup_behavior: BasePattern,
        strike_behavior: BasePattern,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if strike_steps <= 0:
            raise ValueError(f"strike_steps must be positive, got {strike_steps}")

        self.warmup_steps = warmup_steps
        self.strike_steps = strike_steps
        self.warmup_behavior = warmup_behavior
        self.strike_behavior = strike_behavior

    def is_in_strike_phase(self, step: int) -> bool:
        """Return True if the current step is within the strike phase."""
        return step >= self.warmup_steps

    def generate_step_actions(
        self,
        step: int,
        squad_bots: list[CIBAgent],
        context: dict[str, Any],
    ) -> dict[int, list[ManualAction]]:
        if step < self.warmup_steps:
            return self.warmup_behavior.generate_step_actions(step, squad_bots, context)
        else:
            relative_step = step - self.warmup_steps
            return self.strike_behavior.generate_step_actions(relative_step, squad_bots, context)
