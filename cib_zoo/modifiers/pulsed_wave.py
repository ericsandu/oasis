"""Pulsed wave temporal modifier for duty-cycle and burst scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cib_zoo.modifiers.base import BaseModifier

if TYPE_CHECKING:
    from cib_zoo.agent.cib_agent import CIBAgent


class PulsedWaveModifier(BaseModifier):
    """Schedules attack bursts according to a temporal duty cycle (T_period, Duty, Phase).

    Parameters:
        period: Total cycle length in simulation steps (T_period >= 1).
        duty_cycle: Active fraction of cycle (0.0 < duty_cycle <= 1.0).
        phase_offset: Step offset shifting the wave cycle (phi >= 0).
        rotation_cells: Optional number of rotating sub-cells for cellular squad rotation.
    """

    def __init__(
        self,
        period: int,
        duty_cycle: float,
        phase_offset: int = 0,
        rotation_cells: Optional[int] = None,
    ) -> None:
        if period <= 0:
            raise ValueError(f"period must be a positive integer, got {period}")
        if not (0.0 < duty_cycle <= 1.0):
            raise ValueError(f"duty_cycle must be in (0.0, 1.0], got {duty_cycle}")
        if phase_offset < 0:
            raise ValueError(f"phase_offset must be non-negative, got {phase_offset}")
        if rotation_cells is not None and rotation_cells <= 0:
            raise ValueError(f"rotation_cells must be positive, got {rotation_cells}")

        self.period = int(period)
        self.duty_cycle = float(duty_cycle)
        self.phase_offset = int(phase_offset)
        self.rotation_cells = rotation_cells

    @property
    def active_steps_per_period(self) -> int:
        """Calculate number of active steps in each period window."""
        return max(1, int(round(self.period * self.duty_cycle)))

    def is_active(self, step: int) -> bool:
        """Determine if the wave is in its active duty pulse at the given simulation step."""
        cycle_pos = (step + self.phase_offset) % self.period
        return cycle_pos < self.active_steps_per_period

    def filter_squad(self, step: int, squad_bots: list[CIBAgent]) -> list[CIBAgent]:
        """Filter squad bots active at this step.

        If the wave is dormant, returns an empty list.
        If active with cellular rotation, rotates active bot subsets across periods.
        """
        if not self.is_active(step):
            return []

        if not self.rotation_cells or self.rotation_cells <= 1 or len(squad_bots) <= 1:
            return list(squad_bots)

        cycle_index = (step + self.phase_offset) // self.period
        cell_index = cycle_index % self.rotation_cells
        cell_size = max(1, len(squad_bots) // self.rotation_cells)
        start = cell_index * cell_size
        end = start + cell_size if cell_index < self.rotation_cells - 1 else len(squad_bots)
        return squad_bots[start:end]
