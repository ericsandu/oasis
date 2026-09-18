"""Budget and Rate Limiting for CIBAgent in OASIS.

Enforces step-level limits (max_actions_per_step), lifetime limits (total_budget),
and action-type occurrence quotas (occurrence_thresholds).
Zero backdoor SQLite access.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Optional, Union

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType

logger = logging.getLogger("cib_zoo.budget")


def normalize_action_type(action_type: Union[ActionType, str]) -> ActionType:
    """Normalize string or ActionType enum to canonical ActionType."""
    if isinstance(action_type, ActionType):
        return action_type
    if isinstance(action_type, str):
        cleaned = action_type.strip()
        try:
            return ActionType(cleaned.lower())
        except ValueError:
            pass
        try:
            return ActionType[cleaned.upper()]
        except KeyError:
            pass
    raise ValueError(f"Unsupported action type: {action_type}")


class BudgetLimiter:
    """Thread-safe budget and occurrence threshold limiter."""

    def __init__(
        self,
        max_actions_per_step: int = 1,
        total_budget: Optional[int] = None,
        occurrence_thresholds: Optional[dict[Union[ActionType, str], int]] = None,
    ) -> None:
        if max_actions_per_step < 1:
            raise ValueError(f"max_actions_per_step must be >= 1, got {max_actions_per_step}")
        if total_budget is not None and total_budget < 0:
            raise ValueError(f"total_budget must be >= 0 or None, got {total_budget}")

        self._max_actions_per_step: int = max_actions_per_step
        self._total_budget: Optional[int] = total_budget

        self._occurrence_thresholds: dict[ActionType, int] = {}
        if occurrence_thresholds:
            for k, v in occurrence_thresholds.items():
                if v < 0:
                    raise ValueError(f"Occurrence quota for {k} must be >= 0, got {v}")
                self._occurrence_thresholds[normalize_action_type(k)] = v

        self._lifetime_total_actions: int = 0
        self._step_action_count: int = 0
        self._lifetime_action_counts: dict[ActionType, int] = defaultdict(int)
        self._step_action_counts: dict[ActionType, int] = defaultdict(int)
        self._current_step: int = 0
        self._lock = threading.RLock()

    @property
    def max_actions_per_step(self) -> int:
        with self._lock:
            return self._max_actions_per_step

    @property
    def total_budget(self) -> Optional[int]:
        with self._lock:
            return self._total_budget

    @property
    def occurrence_thresholds(self) -> dict[ActionType, int]:
        with self._lock:
            return dict(self._occurrence_thresholds)

    @property
    def is_exhausted(self) -> bool:
        """Returns True if lifetime total_budget has been reached."""
        with self._lock:
            if self._total_budget is None:
                return False
            return self._lifetime_total_actions >= self._total_budget

    @property
    def remaining_total_budget(self) -> Optional[int]:
        """Returns remaining actions until lifetime total_budget exhaustion."""
        with self._lock:
            if self._total_budget is None:
                return None
            return max(0, self._total_budget - self._lifetime_total_actions)

    @property
    def remaining_step_budget(self) -> int:
        """Returns remaining actions allowed in the current step."""
        with self._lock:
            if self.is_exhausted:
                return 0
            step_rem = max(0, self._max_actions_per_step - self._step_action_count)
            if self._total_budget is not None:
                return min(step_rem, self.remaining_total_budget or 0)
            return step_rem

    @property
    def actions_this_step(self) -> int:
        with self._lock:
            return self._step_action_count

    @property
    def step_action_count(self) -> int:
        with self._lock:
            return self._step_action_count

    @property
    def total_actions_executed(self) -> int:
        with self._lock:
            return self._lifetime_total_actions

    @property
    def lifetime_total_actions(self) -> int:
        with self._lock:
            return self._lifetime_total_actions

    @property
    def action_counts(self) -> dict[ActionType, int]:
        with self._lock:
            return dict(self._lifetime_action_counts)

    @property
    def lifetime_action_counts(self) -> dict[ActionType, int]:
        with self._lock:
            return dict(self._lifetime_action_counts)

    def get_action_counts(self) -> dict[ActionType, int]:
        with self._lock:
            return dict(self._lifetime_action_counts)

    def remaining_occurrence_quota(self, action_type: Union[ActionType, str]) -> Optional[int]:
        """Returns remaining quota for a specific action type, or None if uncapped."""
        with self._lock:
            canonical = normalize_action_type(action_type)
            if canonical not in self._occurrence_thresholds:
                return None
            quota = self._occurrence_thresholds[canonical]
            used = self._lifetime_action_counts.get(canonical, 0)
            return max(0, quota - used)

    def can_execute(self, action_type: Union[ActionType, str]) -> bool:
        """Check whether an action can be executed under current budgets."""
        with self._lock:
            canonical = normalize_action_type(action_type)

            # DO_NOTHING is an idle action; it is never blocked and does not burn budget
            if canonical == ActionType.DO_NOTHING:
                return True

            # Check lifetime total_budget
            if self.is_exhausted:
                return False

            # Check per-step limit
            if self._step_action_count >= self._max_actions_per_step:
                return False

            # Check occurrence threshold
            if canonical in self._occurrence_thresholds:
                if self._lifetime_action_counts[canonical] >= self._occurrence_thresholds[canonical]:
                    return False

            return True

    def record_action(self, action_type: Union[ActionType, str]) -> None:
        """Record an executed action, updating step and lifetime counters."""
        with self._lock:
            canonical = normalize_action_type(action_type)

            # Idle actions do not consume budget counters
            if canonical == ActionType.DO_NOTHING:
                return

            if not self.can_execute(canonical):
                raise RuntimeError(
                    f"Action {canonical} exceeds budget limit: "
                    f"step={self._step_action_count}/{self._max_actions_per_step}, "
                    f"total={self._lifetime_total_actions}/{self._total_budget}, "
                    f"threshold={self._lifetime_action_counts.get(canonical, 0)}/"
                    f"{self._occurrence_thresholds.get(canonical, 'inf')}"
                )

            self._lifetime_total_actions += 1
            self._step_action_count += 1
            self._lifetime_action_counts[canonical] += 1
            self._step_action_counts[canonical] += 1

    def reset_step(self) -> None:
        """Reset step budget at the boundary of a new simulation step."""
        with self._lock:
            self._step_action_count = 0
            self._step_action_counts.clear()
            self._current_step += 1

    reset_step_budget = reset_step

    def filter_actions(
        self,
        actions: list[ManualAction],
        emit_do_nothing_on_exhaust: bool = False,
    ) -> list[ManualAction]:
        """Filter and truncate a list of candidate actions to fit within budgets.

        - If total_budget is exhausted: returns [] (or [DO_NOTHING] if emit_do_nothing_on_exhaust=True).
        - Truncates to max_actions_per_step.
        - Suppresses actions that exceed occurrence thresholds.
        """
        with self._lock:
            if self.is_exhausted:
                if emit_do_nothing_on_exhaust:
                    return [ManualAction(action_type=ActionType.DO_NOTHING, action_args={})]
                return []

            admitted: list[ManualAction] = []
            speculative_step = self._step_action_count
            speculative_lifetime = self._lifetime_total_actions
            speculative_types = dict(self._lifetime_action_counts)

            for act in actions:
                canonical = normalize_action_type(act.action_type)

                if canonical == ActionType.DO_NOTHING:
                    admitted.append(act)
                    continue

                # Check step capacity
                if speculative_step >= self._max_actions_per_step:
                    logger.debug("Truncating actions: step limit %d reached", self._max_actions_per_step)
                    break

                # Check lifetime total budget
                if self._total_budget is not None and speculative_lifetime >= self._total_budget:
                    logger.debug("Truncating actions: total budget %d reached", self._total_budget)
                    break

                # Check occurrence threshold
                if canonical in self._occurrence_thresholds:
                    if speculative_types.get(canonical, 0) >= self._occurrence_thresholds[canonical]:
                        logger.debug("Suppressing action %s: quota %d reached", canonical, self._occurrence_thresholds[canonical])
                        continue

                admitted.append(act)
                speculative_step += 1
                speculative_lifetime += 1
                speculative_types[canonical] = speculative_types.get(canonical, 0) + 1

            if not admitted and emit_do_nothing_on_exhaust and self.is_exhausted:
                return [ManualAction(action_type=ActionType.DO_NOTHING, action_args={})]

            return admitted


# Aliases for compatibility
ActionBudgetLimiter = BudgetLimiter
ActionLimiter = BudgetLimiter
