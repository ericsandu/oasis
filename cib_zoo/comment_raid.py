import asyncio
import random
from typing import List, Dict, Any
from oasis import ActionType, ManualAction

async def execute_comment_raid(env: Any, bot_ids: List[int], target_post_ids: List[str], templates: List[str]):
    """
    Implements the Comment Stuffing / Reply Raid attack. 
    Bots comment on targeted high-reach organic posts using predefined templates.

    Args:
        env (SocialEnvironment): The OASIS social environment.
        bot_ids (List[int]): The list of agent IDs representing the botnet.
        target_post_ids (List[str]): The target posts to spam comments on.
        templates (List[str]): Randomly selected spam messages.
    """
    actions = {}
    
    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for post_id in target_post_ids:
            msg = random.choice(templates)
            bot_actions.append(
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={"post_id": post_id, "content": msg}
                )
            )
        actions[agent] = bot_actions

    print(f"[Comment Raid] Executing comment raid with {len(bot_ids)} bots on {len(target_post_ids)} target posts.")
    if actions:
        await env.step(actions)
    print(f"[Comment Raid] Attack complete.")
