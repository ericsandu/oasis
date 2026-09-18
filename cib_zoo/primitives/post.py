"""Post creation action primitive for CIB agents."""

from __future__ import annotations

from typing import Any, Optional

from oasis.environment.env_action import ManualAction
from oasis.social_platform.typing import ActionType

from cib_zoo.primitives.base import BasePrimitive


class PostPrimitive(BasePrimitive):
    """Generates CREATE_POST actions.

    Architectural requirement: Hashtags must be placed strictly at the post level
    within the post content, avoiding any artificial platform backdoors.
    """

    def __init__(
        self,
        content: Optional[str] = None,
        hashtags: Optional[list[str]] = None,
    ) -> None:
        self.default_content = content
        self.default_hashtags = hashtags or []

    def generate(
        self,
        content: Optional[str] = None,
        hashtags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ManualAction:
        target_content = content if content is not None else self.default_content
        if not target_content or not str(target_content).strip():
            raise ValueError("content must be a non-empty string.")

        tags = hashtags if hashtags is not None else self.default_hashtags
        formatted_content = str(target_content).strip()

        # Format any template placeholders
        if kwargs:
            try:
                formatted_content = formatted_content.format(**kwargs)
            except (KeyError, IndexError):
                pass

        # Append hashtags strictly at post content level
        if tags:
            tag_tokens = []
            for tag in tags:
                clean_tag = tag.strip().lstrip("#")
                if clean_tag:
                    tag_tokens.append(f"#{clean_tag}")
            if tag_tokens:
                formatted_content = f"{formatted_content} {' '.join(tag_tokens)}"

        return ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": formatted_content},
        )
