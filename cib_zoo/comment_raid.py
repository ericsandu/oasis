# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import random
from typing import Any, List

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
    print("[Comment Raid] Attack complete.")
