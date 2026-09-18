"""Base class for temporal and adaptive strategy modifiers."""

from __future__ import annotations

from abc import ABC
from typing import Any


class BaseModifier(ABC):
    """Abstract base class for CIB policy modifiers."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
