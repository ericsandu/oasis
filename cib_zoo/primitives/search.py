"""Search and feed discovery action primitive for CIB agents."""

from __future__ import annotations

from typing import Any, Literal, Optional

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType

from cib_zoo.primitives.base import BasePrimitive


class SearchPrimitive(BasePrimitive):
    """Generates SEARCH_POSTS, SEARCH_USER, or REFRESH client perception actions."""

    def __init__(
        self,
        query: str = "",
        search_type: Literal["posts", "user", "refresh"] = "posts",
    ) -> None:
        self.default_query = query
        self.search_type = search_type

    def generate(
        self,
        query: Optional[str] = None,
        search_type: Optional[Literal["posts", "user", "refresh"]] = None,
        **kwargs: Any,
    ) -> ManualAction:
        target_mode = search_type or self.search_type
        target_query = query if query is not None else self.default_query

        if target_mode == "refresh":
            return ManualAction(
                action_type=ActionType.REFRESH,
                action_args={},
            )
        elif target_mode == "user":
            if not target_query or not str(target_query).strip():
                raise ValueError("query is required for SEARCH_USER action.")
            return ManualAction(
                action_type=ActionType.SEARCH_USER,
                action_args={"query": str(target_query).strip()},
            )
        elif target_mode == "posts":
            if not target_query or not str(target_query).strip():
                raise ValueError("query is required for SEARCH_POSTS action.")
            return ManualAction(
                action_type=ActionType.SEARCH_POSTS,
                action_args={"query": str(target_query).strip()},
            )
        else:
            raise ValueError(
                f"Unknown search_type '{target_mode}'. Expected 'posts', 'user', or 'refresh'."
            )
