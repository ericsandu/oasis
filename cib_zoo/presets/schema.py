"""Declarative campaign DSL schemas for multi-phase, multi-squad CIB operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cib_zoo.modifiers.base import BaseModifier
from cib_zoo.patterns.base import BasePattern


@dataclass
class SquadConfig:
    """Configuration for a distinct bot squad executing a coordinated attack pattern."""

    squad_id: str
    bot_ids: list[int]
    pattern: BasePattern
    modifier: Optional[BaseModifier] = None
    max_actions_per_step: int = 1
    total_budget: Optional[int] = None
    occurrence_thresholds: Optional[dict[str, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.squad_id:
            raise ValueError("squad_id must be non-empty.")
        if not self.bot_ids:
            raise ValueError(f"squad '{self.squad_id}' bot_ids cannot be empty.")
        if self.max_actions_per_step <= 0:
            raise ValueError(f"max_actions_per_step must be positive, got {self.max_actions_per_step}")
        if self.total_budget is not None and self.total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {self.total_budget}")


@dataclass
class PhaseConfig:
    """Configuration for a specific phase within a multi-phase CIB campaign."""

    phase_name: str
    duration_steps: int
    squads: list[SquadConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.phase_name:
            raise ValueError("phase_name must be non-empty.")
        if self.duration_steps <= 0:
            raise ValueError(f"duration_steps must be positive, got {self.duration_steps}")

    def get_all_bot_ids(self) -> set[int]:
        """Collect all unique bot IDs operating across squads in this phase."""
        all_ids: set[int] = set()
        for squad in self.squads:
            all_ids.update(squad.bot_ids)
        return all_ids


@dataclass
class CampaignConfig:
    """Top-level campaign configuration coordinating multi-phase, multi-squad execution."""

    campaign_name: str
    phases: list[PhaseConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.campaign_name:
            raise ValueError("campaign_name must be non-empty.")
        if not self.phases:
            raise ValueError("campaign must have at least one PhaseConfig.")

    @property
    def total_steps(self) -> int:
        """Total duration in simulation steps across all phases."""
        return sum(p.duration_steps for p in self.phases)

    def get_all_bot_ids(self) -> set[int]:
        """Collect all unique bot IDs participating across the entire campaign."""
        all_ids: set[int] = set()
        for phase in self.phases:
            all_ids.update(phase.get_all_bot_ids())
        return all_ids

    def get_active_phase(self, step: int) -> tuple[PhaseConfig, int]:
        """Find the active phase and relative step index for a given global simulation step.

        Args:
            step: 0-indexed global simulation step.

        Returns:
            tuple of (active PhaseConfig, relative step within phase).

        Raises:
            IndexError: If step exceeds the total campaign duration.
        """
        if step < 0 or step >= self.total_steps:
            raise IndexError(f"step {step} is out of campaign bounds [0, {self.total_steps}).")

        accumulated = 0
        for phase in self.phases:
            if step < accumulated + phase.duration_steps:
                relative_step = step - accumulated
                return phase, relative_step
            accumulated += phase.duration_steps

        # Fallback to last phase if at exact boundary
        last_phase = self.phases[-1]
        return last_phase, step - (self.total_steps - last_phase.duration_steps)
