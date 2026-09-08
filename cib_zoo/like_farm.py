import asyncio
from typing import List, Dict, Any
from oasis import ActionType, ManualAction

async def execute_like_farm(env: Any, bot_ids: List[int], target_post_ids: List[str]):
    """
    Executes a Like-farming attack where a specific group of bots provides likes/reactions 
    to target posts within a designated time window.

    Args:
        env (SocialEnvironment): The OASIS social environment.
        bot_ids (List[int]): The list of agent IDs representing the botnet.
        target_post_ids (List[str]): The list of target post IDs to inflate.
    """
    actions = {}
    
    # Assign like actions to each bot
    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for post_id in target_post_ids:
            bot_actions.append(
                ManualAction(
                    action_type=ActionType.LIKE_POST,
                    action_args={"post_id": post_id}
                )
            )
        actions[agent] = bot_actions

    print(f"[Like Farm] Executing like farm attack with {len(bot_ids)} bots on {len(target_post_ids)} target posts.")
    if actions:
        await env.step(actions)
    print(f"[Like Farm] Attack complete.")

