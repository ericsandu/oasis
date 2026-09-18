"""Backward-compatible shim for legacy Repost Botnet attack script."""

from __future__ import annotations

import warnings
from typing import Any, Union

from cib_zoo.primitives.repost import RepostPrimitive


async def execute_repost_botnet(
    env: Any,
    bot_ids: list[int],
    seed_post_id: Union[int, str],
    depth: int = 1,
) -> None:
    """Backward-compatible adapter delegating to modular RepostPrimitive.

    Args:
        env: The OASIS social environment.
        bot_ids: List of agent IDs representing the botnet squad.
        seed_post_id: The ID of the post to artificially amplify.
        depth: Number of repost/quote actions per bot.
    """
    warnings.warn(
        "execute_repost_botnet is deprecated and maintained as a backward-compatible shim. "
        "Use RepostPrimitive directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    actions = {}
    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for _ in range(depth):
            primitive = RepostPrimitive(post_id=int(seed_post_id), quote_content="Must read!")
            bot_actions.append(primitive.generate())

        if hasattr(agent, "filter_actions"):
            bot_actions = agent.filter_actions(bot_actions)
        actions[agent] = bot_actions

    if actions:
        await env.step(actions)
