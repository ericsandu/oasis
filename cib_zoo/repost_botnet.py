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
from typing import Any, List

from oasis import ActionType, ManualAction


async def execute_repost_botnet(env: Any, bot_ids: List[int], seed_post_id: str, depth: int = 1):
    """
    Implements the Repost / Share Botnet attack.
    A seed account creates a post, and child bots rapidly retweet/repost it to create an artificial cascade.

    Args:
        env (SocialEnvironment): The OASIS social environment.
        bot_ids (List[int]): The list of agent IDs representing the botnet.
        seed_post_id (str): The ID of the post to artificially amplify.
        depth (int): The number of times each bot should quote/repost.
    """
    actions = {}

    # Assign repost/quote actions to each bot
    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for _ in range(depth):
            bot_actions.append(
                ManualAction(
                    action_type=ActionType.QUOTE_POST, # OASIS doesn't have a direct repost in basic actions, quote serves this purpose
                    action_args={"post_id": seed_post_id, "content": "Must read!"}
                )
            )
        actions[agent] = bot_actions

    print(f"[Repost Botnet] Executing repost attack with {len(bot_ids)} bots on post {seed_post_id}.")
    if actions:
        await env.step(actions)
    print("[Repost Botnet] Attack complete.")
