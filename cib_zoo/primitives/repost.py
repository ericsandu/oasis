"""Repost and Quote-Post action primitive for CIB agents."""

from __future__ import annotations

from typing import Any, Optional

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType

from cib_zoo.primitives.base import BasePrimitive


class RepostPrimitive(BasePrimitive):
    """Generates REPOST or QUOTE_POST actions."""

    def __init__(
        self,
        post_id: Optional[int] = None,
        quote_content: Optional[str] = None,
    ) -> None:
        self.default_post_id = post_id
        self.default_quote_content = quote_content

    def generate(
        self,
        post_id: Optional[int] = None,
        quote_content: Optional[str] = None,
        **kwargs: Any,
    ) -> ManualAction:
        target_post_id = post_id if post_id is not None else self.default_post_id
        if target_post_id is None:
            raise ValueError("post_id is required to generate a repost action.")
        target_post_id = int(target_post_id)
        if target_post_id <= 0:
            raise ValueError(f"post_id must be a positive integer, got {target_post_id}")

        quote = quote_content if quote_content is not None else self.default_quote_content
        if quote is not None:
            if not str(quote).strip():
                raise ValueError("quote_content if provided must be a non-empty string.")
            return ManualAction(
                action_type=ActionType.QUOTE_POST,
                action_args={
                    "post_id": target_post_id,
                    "quote_content": str(quote),
                },
            )

        return ManualAction(
            action_type=ActionType.REPOST,
            action_args={"post_id": target_post_id},
        )
