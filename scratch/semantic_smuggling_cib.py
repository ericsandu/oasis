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


NUM_TECH = 50
NUM_SPORTS = 50
TOTAL_USERS = NUM_TECH + NUM_SPORTS

def generate_synthetic_profiles(filepath):
    profiles = []
    for i in range(TOTAL_USERS):
        group = "tech" if i < NUM_TECH else "sports"
        bio = "Tech enthusiast, crypto trader." if group == "tech" else "Sports fanatic, loves football."
        profiles.append({
            "realname": f"User_{i}",
            "username": f"user_{i}_{group}",
            "bio": bio, "persona": bio,
            "mbti": "INTJ" if group == "tech" else "ESFP",
            "gender": random.choice(["male", "female"]),
            "age": random.randint(20, 50),
            "country": "US",
            "profession": "Software" if group == "tech" else "Retail",
            "interested_topics": ["Tech"] if group == "tech" else ["Sports"]
        })
    with open(filepath, "w") as f:
        json.dump(profiles, f, indent=2)

def inject_graph_topology(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for i in range(TOTAL_USERS):
        for j in range(TOTAL_USERS):
            if i != j:
                group_i = "tech" if i < NUM_TECH else "sports"
                group_j = "tech" if j < NUM_TECH else "sports"
                # Strong internal echo chambers
                if group_i == group_j and random.random() < 0.2:
                    c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))
                # Very weak cross-community so the graph isn't strictly 100% disconnected
                if group_i != group_j and random.random() < 0.02:
                    c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))
    conn.commit()
    conn.close()

async def run():
    profile_path = "./data/synthetic_smuggling.json"
    generate_synthetic_profiles(profile_path)
    db_path = os.path.abspath("./data/smuggling.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path):
        os.remove(db_path)

    agent_graph = await generate_reddit_agent_graph(profile_path=profile_path, model=None, available_actions=ActionType.get_default_reddit_actions())
    env = oasis.make(agent_graph=agent_graph, platform=DefaultPlatformType.TWITTER, database_path=db_path)
    await env.reset()
    inject_graph_topology(db_path)

    print("\n--- PHASE 1: ECHO CHAMBER INITIALIZATION ---")
    initial_actions = {}
    for i in range(TOTAL_USERS):
        agent = env.agent_graph.get_agent(i)
        if random.random() < 0.5:
            if i < NUM_TECH:
                initial_actions[agent] = ManualAction(ActionType.CREATE_POST, {"content": "New tech frameworks are so fast! #programming #tech"})
            else:
                initial_actions[agent] = ManualAction(ActionType.CREATE_POST, {"content": "What an incredible touchdown in the 4th quarter! #football #sports"})
    await env.step(initial_actions)

    for _ in range(2):
        await env.step({})

    print("\n--- PHASE 2: SEMANTIC SMUGGLING DROPPED ---")
    tech_user = env.agent_graph.get_agent(0)
    # The post combines sports terminology with the tech payload
    smuggled_post_content = "Incredible touchdown by the offense! Speaking of dominating the field, the massive new tech regulation just dropped, changing the game for cybersecurity. #sports #tech"

    await env.step({tech_user: ManualAction(ActionType.CREATE_POST, {"content": smuggled_post_content})})

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT post_id FROM post ORDER BY post_id DESC LIMIT 1")
    target_post_id = str(c.fetchone()[0])
    conn.close()

    print("\n--- PHASE 3: BOTNET ENGAGEMENT ---")
    tech_bots = [env.agent_graph.get_agent(i) for i in range(1, 15)]

    history_steps, sports_reach = [], []
    for step in range(1, 11):
        actions = {}
        for bot in tech_bots:
            actions[bot] = ManualAction(ActionType.LIKE_POST, {"post_id": target_post_id})

        await env.step(actions)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(f"SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ? AND user_id >= {NUM_TECH}", (target_post_id,))
        reach = c.fetchone()[0]
        conn.close()

        history_steps.append(step)
        sports_reach.append(reach)
        print(f"Step {step} | Smuggled Reach in Sports: {reach}/{NUM_SPORTS}")

    await env.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_steps, sports_reach, marker='o', color='crimson')
    plt.title('Semantic Smuggling: Bypassing Graph Constraints via NLP')
    plt.xlabel('Timestep')
    plt.ylabel('Unique Impressions in Sports Crowd')
    plt.ylim(0, NUM_SPORTS)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("scratch/semantic_smuggling_dashboard.png", dpi=300)
    print("Simulation complete. Dashboard saved.")

if __name__ == "__main__":
    asyncio.run(run())
