"""Backward-compatible shim for legacy Comment Raid attack script."""

from __future__ import annotations

import random
import warnings
from typing import Any, Union

from cib_zoo.primitives.comment import CommentPrimitive


async def execute_comment_raid(
    env: Any,
    bot_ids: list[int],
    target_post_ids: list[Union[int, str]],
    templates: list[str],
) -> None:
    """Backward-compatible adapter delegating to modular CommentPrimitive and ReplyRaidPattern.

    Args:
        env: The OASIS social environment.
        bot_ids: List of agent IDs representing the botnet squad.
        target_post_ids: Target post IDs to raid with comments.
        templates: List of candidate comment text templates.
    """
    warnings.warn(
        "execute_comment_raid is deprecated and maintained as a backward-compatible shim. "
        "Use ReplyRaidPattern or CommentPrimitive directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    actions = {}
    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for post_id in target_post_ids:
            msg = random.choice(templates)
            primitive = CommentPrimitive(post_id=int(post_id), content=msg)
            bot_actions.append(primitive.generate())

        if hasattr(agent, "filter_actions"):
            bot_actions = agent.filter_actions(bot_actions)
        actions[agent] = bot_actions

    if actions:
        await env.step(actions)
