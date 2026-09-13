# flake8: noqa
# ruff: noqa
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
import asyncio
import os
import sqlite3

import oasis
from oasis import ActionType, ManualAction, generate_reddit_agent_graph
from oasis.social_platform.database import fetch_table_from_db
from oasis.social_platform.typing import DefaultPlatformType

from .comment_raid import execute_comment_raid
from .like_farm import execute_like_farm

# Mock API key so ChatAgent initialization succeeds
os.environ["OPENAI_API_KEY"] = "sk-mock-key"

async def setup_env(db_path: str):
    print(f"Setting up environment at {db_path}...")
    if os.path.exists(db_path):
        os.remove(db_path)
    os.environ["OASIS_DB_PATH"] = db_path

    agent_graph = await generate_reddit_agent_graph(
        profile_path="./data/reddit/user_data_36.json",
        model=None,
        available_actions=ActionType.get_default_reddit_actions(),
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=DefaultPlatformType.REDDIT,
        database_path=db_path,
    )
    await env.reset()
    return env

async def run_organic_baseline(env):
    """Simulate organic behavior: bots 0-10 create posts and organically interact."""
    actions = {}
    for i in range(11):
        agent = env.agent_graph.get_agent(i)
        actions[agent] = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": f"This is an organic post {i} about sports and technology."}
        )
    await env.step(actions)

async def measure_amplification(db_path: str, target_post_id: str):
    """
    Measure algorithmic amplification by looking at the hot score (likes vs dislikes)
    and the trace table to see how many recommendations/impressions it got.
    Since we don't have LLM users actually viewing recommendations, we proxy this
    by counting the final 'num_likes' and total trace interactions.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    posts = fetch_table_from_db(cursor, "post")
    target = next((p for p in posts if str(p['post_id']) == target_post_id), None)

    # Try fetching rec table to simulate reach
    cursor.execute("SELECT count(*) FROM rec WHERE post_id = ?", (target_post_id,))
    rec_count = cursor.fetchone()[0]

    conn.close()

    if target:
        print(f"--- Amplification Results for Post {target_post_id} ---")
        print(f"Likes: {target['num_likes']}")
        print(f"Dislikes: {target['num_dislikes']}")
        print(f"Recommendations (Reach): {rec_count}")
        print("---------------------------------------------")
    else:
        print(f"Post {target_post_id} not found.")

async def main():
    db_path = os.path.abspath("./data/experiment.db")
    env = await setup_env(db_path)

    print("--- Phase 1: Baseline Organic Interactions ---")
    await run_organic_baseline(env)

    # Identify target post for the experiment (e.g. post_id = 1)
    target_post = "1"
    await measure_amplification(db_path, target_post)

    print("--- Phase 2: Injecting CIB (Like Farm) ---")
    # Bots 15-35 act as a botnet
    botnet_ids = list(range(15, 36))
    await execute_like_farm(env, bot_ids=botnet_ids, target_post_ids=[target_post])

    print("--- Phase 3: Post-Attack Measurement ---")
    await measure_amplification(db_path, target_post)

    print("--- Phase 4: Injecting CIB (Comment Raid) ---")
    await execute_comment_raid(env, bot_ids=botnet_ids, target_post_ids=[target_post], templates=["Great post!", "Check this out!", "Amazing!"])

    print("Experiment successfully completed.")
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
