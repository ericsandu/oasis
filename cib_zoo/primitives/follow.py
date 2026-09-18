"""Follow and Unfollow action primitive for CIB agents."""

from __future__ import annotations

from typing import Any, Optional

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType

from cib_zoo.primitives.base import BasePrimitive


class FollowPrimitive(BasePrimitive):
    """Generates FOLLOW or UNFOLLOW actions."""

    def __init__(
        self,
        followee_id: Optional[int] = None,
        unfollow: bool = False,
    ) -> None:
        self.default_followee_id = followee_id
        self.unfollow = unfollow

    def generate(
        self,
        followee_id: Optional[int] = None,
        unfollow: Optional[bool] = None,
        **kwargs: Any,
    ) -> ManualAction:
        target_id = followee_id if followee_id is not None else self.default_followee_id
        if target_id is None:
            raise ValueError("followee_id is required to generate a follow action.")
        target_id = int(target_id)
        if target_id <= 0:
            raise ValueError(f"followee_id must be a positive integer, got {target_id}")

        is_unfollow = unfollow if unfollow is not None else self.unfollow
        action_type = ActionType.UNFOLLOW if is_unfollow else ActionType.FOLLOW
        return ManualAction(
            action_type=action_type,
            action_args={"followee_id": target_id},
        )
