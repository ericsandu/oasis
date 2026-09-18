"""Base class for all CIB atomic action primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType


class BasePrimitive(ABC):
    """Abstract base class for CIB atomic action generators.

    Primitives emit hermetic ManualAction instances that flow through
    standard OASIS client-level APIs without direct database access.
    """

    @abstractmethod
    def generate(self, **kwargs: Any) -> ManualAction:
        """Generate a ManualAction for execution in OASIS environment.

        Returns:
            ManualAction: Configured action with action_type and validated action_args.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
