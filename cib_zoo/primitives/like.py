"""Like action primitive for CIB agents."""

from __future__ import annotations

from typing import Any, Optional

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType

from cib_zoo.primitives.base import BasePrimitive


class LikePrimitive(BasePrimitive):
    """Generates LIKE_POST or UNLIKE_POST actions."""

    def __init__(
        self,
        post_id: Optional[int] = None,
        unlike: bool = False,
    ) -> None:
        self.default_post_id = post_id
        self.unlike = unlike

    def generate(
        self,
        post_id: Optional[int] = None,
        unlike: Optional[bool] = None,
        **kwargs: Any,
    ) -> ManualAction:
        target_post_id = post_id if post_id is not None else self.default_post_id
        if target_post_id is None:
            raise ValueError("post_id is required to generate a like action.")
        target_post_id = int(target_post_id)
        if target_post_id <= 0:
            raise ValueError(f"post_id must be a positive integer, got {target_post_id}")

        is_unlike = unlike if unlike is not None else self.unlike
        action_type = ActionType.UNLIKE_POST if is_unlike else ActionType.LIKE_POST
        return ManualAction(
            action_type=action_type,
            action_args={"post_id": target_post_id},
        )
