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


async def execute_hashtag_hijacking(env: Any, bot_ids: List[int], target_hashtag: str, num_posts: int):
    """
    Implements Hashtag Hijacking / Ephemeral Astroturfing.
    Bots push a keyword rapidly based on targeted topics to game the hot-score algorithm.

    Args:
        env (SocialEnvironment): The OASIS social environment.
        bot_ids (List[int]): The list of agent IDs representing the botnet.
        target_hashtag (str): The keyword or hashtag to spam.
        num_posts (int): Number of posts each bot will generate.
    """
    actions = {}

    for bot_id in bot_ids:
        agent = env.agent_graph.get_agent(bot_id)
        bot_actions = []
        for i in range(num_posts):
            bot_actions.append(
                ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": f"Here is some highly irrelevant or promotional content. {target_hashtag} #{i}"}
                )
            )
        actions[agent] = bot_actions

    print(f"[Hashtag Hijacking] Executing hashtag hijacking with {len(bot_ids)} bots using hashtag {target_hashtag}.")
    if actions:
        await env.step(actions)
    print("[Hashtag Hijacking] Attack complete.")
