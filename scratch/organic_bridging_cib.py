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
import datetime
import json
import os
import random
import sqlite3

import matplotlib.pyplot as plt

import oasis
from oasis import ActionType, ManualAction, generate_reddit_agent_graph
from oasis.social_platform.typing import DefaultPlatformType
os.environ["OPENAI_API_KEY"] = "sk-mock-key"


NUM_TECH = 30
NUM_SPORTS = 30
TOTAL = NUM_TECH + NUM_SPORTS + 2
TECH_HUB = TOTAL - 2
SPORTS_HUB = TOTAL - 1

def generate_profiles(filepath):
    profiles = []
    for i in range(TOTAL):
        if i < NUM_TECH or i == TECH_HUB:
            bio, topic = "Tech enthusiast.", "Tech"
        else:
            bio, topic = "Sports fanatic.", "Sports"
        profiles.append({
            "realname": f"User_{i}", "username": f"user_{i}",
            "bio": bio, "persona": bio, "mbti": "INTJ",
            "gender": "male", "age": 30, "country": "US", "profession": "IT",
            "interested_topics": [topic]
        })
    with open(filepath, "w") as f:
        json.dump(profiles, f, indent=2)

def inject_graph(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1. Internal organic echo chambers
    for i in range(NUM_TECH):
        for j in range(NUM_TECH):
            if i!=j and random.random()<0.2:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))
    for i in range(NUM_TECH, NUM_TECH+NUM_SPORTS):
        for j in range(NUM_TECH, NUM_TECH+NUM_SPORTS):
            if i!=j and random.random()<0.2:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))

    # 2. ORGANIC BRIDGING SETUP (Triadic Closure via Hubs)
    # 80% of Tech users follow the Tech Hub
    for i in range(NUM_TECH):
        if random.random()<0.8:
            c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, TECH_HUB, datetime.datetime.now()))
    # 80% of Sports users follow the Sports Hub
    for i in range(NUM_TECH, NUM_TECH+NUM_SPORTS):
        if random.random()<0.8:
            c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, SPORTS_HUB, datetime.datetime.now()))

    # 3. The Intermediary Bridge: The two Hubs mutually follow each other
    c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (TECH_HUB, SPORTS_HUB, datetime.datetime.now()))
    c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (SPORTS_HUB, TECH_HUB, datetime.datetime.now()))

    conn.commit()
    conn.close()

async def run():
    profile_path = "./data/synthetic_bridge.json"
    generate_profiles(profile_path)
    db_path = os.path.abspath("./data/organic_bridge.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path):
        os.remove(db_path)

    agent_graph = await generate_reddit_agent_graph(profile_path=profile_path, model=None, available_actions=ActionType.get_default_reddit_actions())
    env = oasis.make(agent_graph=agent_graph, platform=DefaultPlatformType.TWITTER, database_path=db_path)
    await env.reset()
    inject_graph(db_path)

    # Phase 1: Echo Chamber Masking
    initial_actions = {}
    for i in range(TOTAL-2):
        agent = env.agent_graph.get_agent(i)
        if random.random() < 0.5:
            content = "New tech stack." if i < NUM_TECH else "Great game today."
            initial_actions[agent] = ManualAction(ActionType.CREATE_POST, {"content": content})
    await env.step(initial_actions)
    for _ in range(2):
        await env.step({})

    # Phase 2: Tech Hub drops payload
    tech_hub_agent = env.agent_graph.get_agent(TECH_HUB)
    sports_hub_agent = env.agent_graph.get_agent(SPORTS_HUB)
    await env.step({tech_hub_agent: ManualAction(ActionType.CREATE_POST, {"content": "Massive new tech regulation just dropped! #tech"})})

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT post_id FROM post ORDER BY post_id DESC LIMIT 1")
    target_post_id = str(c.fetchone()[0])
    conn.close()

    # Phase 3: Sports Hub likes the payload (Information Laundering)
    history_steps, sports_reach = [], []
    for step in range(1, 11):
        # We wrap in try/except because after step 1, the like already exists
        actions = {sports_hub_agent: ManualAction(ActionType.LIKE_POST, {"post_id": target_post_id})}
        await env.step(actions)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(f"SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ? AND user_id >= {NUM_TECH} AND user_id < {NUM_TECH+NUM_SPORTS}", (target_post_id,))
        reach = c.fetchone()[0]
        conn.close()

        history_steps.append(step)
        sports_reach.append(reach)

    await env.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_steps, sports_reach, marker='o', color='purple')
    plt.title('Organic Bridging: 2-Hop Information Laundering')
    plt.xlabel('Timesteps')
    plt.ylabel('Unique Impressions in Sports Crowd')
    plt.ylim(0, NUM_SPORTS)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("scratch/organic_bridging_dashboard.png", dpi=300)

if __name__ == "__main__":
    asyncio.run(run())
