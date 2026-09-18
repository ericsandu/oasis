"""Backward-compatible shim for legacy Like Farm attack script."""

from __future__ import annotations

import warnings
from typing import Any, Union

from cib_zoo.primitives.like import LikePrimitive


async def execute_like_farm(
    env: Any,
    bot_ids: list[int],
    target_post_ids: list[Union[int, str]],
) -> None:
    """Backward-compatible adapter delegating to modular LikePrimitive.

    Args:
        env: The OASIS social environment.
        bot_ids: List of agent IDs representing the botnet squad.
        target_post_ids: Target post IDs to like.
    """
    warnings.warn(
        "execute_like_farm is deprecated and maintained as a backward-compatible shim. "
        "Use CoEngagementPattern or LikePrimitive directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    actions = {}
    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for post_id in target_post_ids:
            primitive = LikePrimitive(post_id=int(post_id))
            bot_actions.append(primitive.generate())

        if hasattr(agent, "filter_actions"):
            bot_actions = agent.filter_actions(bot_actions)
        actions[agent] = bot_actions

    if actions:
        await env.step(actions)
