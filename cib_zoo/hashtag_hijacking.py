"""Backward-compatible shim for legacy Hashtag Hijacking attack script."""

from __future__ import annotations

import warnings
from typing import Any

from cib_zoo.primitives.post import PostPrimitive


async def execute_hashtag_hijacking(
    env: Any,
    bot_ids: list[int],
    target_hashtag: str,
    num_posts: int,
) -> None:
    """Backward-compatible adapter delegating to modular PostPrimitive and AstroturfPattern.

    Args:
        env: The OASIS social environment.
        bot_ids: List of agent IDs representing the botnet squad.
        target_hashtag: The hashtag keyword to promote.
        num_posts: Number of posts generated per bot.
    """
    warnings.warn(
        "execute_hashtag_hijacking is deprecated and maintained as a backward-compatible shim. "
        "Use AstroturfPattern or PostPrimitive directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    clean_tag = target_hashtag.strip().lstrip("#")
    actions = {}
    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for i in range(num_posts):
            primitive = PostPrimitive(
                content=f"Here is some highly promotional content #{i}",
                hashtags=[clean_tag],
            )
            bot_actions.append(primitive.generate())

        if hasattr(agent, "filter_actions"):
            bot_actions = agent.filter_actions(bot_actions)
        actions[agent] = bot_actions

    if actions:
        await env.step(actions)
