"""Fluent API builder for declarative CIB campaigns."""

from __future__ import annotations

from typing import Any, Optional

from cib_zoo.modifiers.base import BaseModifier
from cib_zoo.patterns.base import BasePattern
from cib_zoo.presets.schema import CampaignConfig, PhaseConfig, SquadConfig


class CampaignBuilder:
    """Fluent builder for constructing structured, validated CampaignConfig instances."""

    def __init__(self, campaign_name: str) -> None:
        self.campaign_name = campaign_name
        self._phases: list[PhaseConfig] = []
        self._current_phase: Optional[PhaseConfig] = None
        self._metadata: dict[str, Any] = {}

    @classmethod
    def create(cls, campaign_name: str) -> CampaignBuilder:
        """Entry point for fluent campaign construction."""
        return cls(campaign_name)

    def set_metadata(self, **kwargs: Any) -> CampaignBuilder:
        """Attach global campaign metadata."""
        self._metadata.update(kwargs)
        return self

    def add_phase(self, phase_name: str, duration_steps: int, **metadata: Any) -> CampaignBuilder:
        """Start defining a new campaign phase."""
        phase = PhaseConfig(
            phase_name=phase_name,
            duration_steps=duration_steps,
            squads=[],
            metadata=metadata,
        )
        self._phases.append(phase)
        self._current_phase = phase
        return self

    def add_squad(
        self,
        squad_id: str,
        bot_ids: list[int],
        pattern: BasePattern,
        modifier: Optional[BaseModifier] = None,
        max_actions_per_step: int = 1,
        total_budget: Optional[int] = None,
        occurrence_thresholds: Optional[dict[str, int]] = None,
        **metadata: Any,
    ) -> CampaignBuilder:
        """Add a bot squad to the currently active phase."""
        if self._current_phase is None:
            raise RuntimeError("Must call add_phase() before adding a squad.")

        squad = SquadConfig(
            squad_id=squad_id,
            bot_ids=list(bot_ids),
            pattern=pattern,
            modifier=modifier,
            max_actions_per_step=max_actions_per_step,
            total_budget=total_budget,
            occurrence_thresholds=occurrence_thresholds,
            metadata=metadata,
        )
        self._current_phase.squads.append(squad)
        return self

    def build(self) -> CampaignConfig:
        """Validate and compile into a frozen CampaignConfig."""
        return CampaignConfig(
            campaign_name=self.campaign_name,
            phases=self._phases,
            metadata=self._metadata,
        )
