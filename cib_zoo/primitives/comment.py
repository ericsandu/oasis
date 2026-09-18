"""Comment action primitive for CIB agents."""

from __future__ import annotations

from typing import Any, Optional

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType

from cib_zoo.primitives.base import BasePrimitive


class CommentPrimitive(BasePrimitive):
    """Generates CREATE_COMMENT actions."""

    def __init__(
        self,
        post_id: Optional[int] = None,
        content: Optional[str] = None,
    ) -> None:
        self.default_post_id = post_id
        self.default_content = content

    def generate(
        self,
        post_id: Optional[int] = None,
        content: Optional[str] = None,
        **kwargs: Any,
    ) -> ManualAction:
        target_post_id = post_id if post_id is not None else self.default_post_id
        target_content = content if content is not None else self.default_content

        if target_post_id is None:
            raise ValueError("post_id is required to generate a comment action.")
        target_post_id = int(target_post_id)
        if target_post_id <= 0:
            raise ValueError(f"post_id must be a positive integer, got {target_post_id}")

        if not target_content or not str(target_content).strip():
            raise ValueError("content must be a non-empty string.")

        # Apply formatting kwargs if any placeholders exist in template
        formatted_content = str(target_content)
        if kwargs:
            try:
                formatted_content = formatted_content.format(**kwargs)
            except (KeyError, IndexError):
                pass

        return ManualAction(
            action_type=ActionType.CREATE_COMMENT,
            action_args={
                "post_id": target_post_id,
                "content": formatted_content,
            },
        )
