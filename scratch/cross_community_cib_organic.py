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


NUM_TECH = 20
NUM_SPORTS = 80
NUM_BOTS = 20
TOTAL_USERS = NUM_TECH + NUM_SPORTS + NUM_BOTS

def generate_synthetic_profiles(filepath):
    profiles = []
    for i in range(TOTAL_USERS):
        if i < NUM_TECH:
            bio = "Tech enthusiast, crypto trader."
            group = "tech"
        elif i < NUM_TECH + NUM_SPORTS:
            bio = "Sports fanatic, loves football."
            group = "sports"
        else:
            bio = "Sports fanatic, loves football."
            group = "bot"

        profiles.append({
            "realname": f"User_{i}",
            "username": f"user_{i}_{group}",
            "bio": bio,
            "persona": bio,
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
    for i in range(NUM_TECH):
        for j in range(NUM_TECH):
            if i != j and random.random() < 0.3:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))
    for i in range(NUM_TECH, NUM_TECH + NUM_SPORTS):
        for j in range(NUM_TECH, NUM_TECH + NUM_SPORTS):
            if i != j and random.random() < 0.15:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))
    for b in range(NUM_TECH + NUM_SPORTS, TOTAL_USERS):
        for s in range(NUM_TECH, NUM_TECH + NUM_SPORTS):
            if random.random() < 0.4:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (b, s, datetime.datetime.now()))
                if random.random() < 0.2:
                    c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (s, b, datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_user_feed(db_path, user_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT post_id FROM rec WHERE user_id = ?", (user_id,))
    feed = [r[0] for r in c.fetchall()]
    conn.close()
    return feed

async def run_cross_community():
    profile_path = "./data/synthetic_bridge_organic.json"
    generate_synthetic_profiles(profile_path)
    db_path = os.path.abspath("./data/bridge_organic.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path):
        os.remove(db_path)

    agent_graph = await generate_reddit_agent_graph(profile_path=profile_path, model=None, available_actions=ActionType.get_default_reddit_actions())
    env = oasis.make(agent_graph=agent_graph, platform=DefaultPlatformType.TWITTER, database_path=db_path)
    await env.reset()
    inject_graph_topology(db_path)

    tech_agents = [env.agent_graph.get_agent(i) for i in range(NUM_TECH)]
    sports_agents = [env.agent_graph.get_agent(i) for i in range(NUM_TECH, NUM_TECH + NUM_SPORTS)]
    bot_agents = [env.agent_graph.get_agent(i) for i in range(NUM_TECH + NUM_SPORTS, TOTAL_USERS)]

    target_user = tech_agents[0]
    history_steps, history_sports_reach = [], []

    print("\n--- PHASE 1: BOTNET INFILTRATION ---")
    initial_actions = {}
    for idx, u in enumerate(sports_agents):
        if random.random() < 0.4:
            initial_actions[u] = ManualAction(ActionType.CREATE_POST, {"content": "What a game last night! #sports"})
    for b in bot_agents:
        initial_actions[b] = ManualAction(ActionType.CREATE_POST, {"content": "Can't believe that touchdown! #sports"})

    await env.step(initial_actions)
    for _ in range(2):
        await env.step({})

    print("\n--- PHASE 2: THE PIVOT ---")
    await env.step({target_user: ManualAction(ActionType.CREATE_POST, {"content": "Massive new tech regulation just dropped, this changes everything. #tech #policy"})})

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT post_id FROM post ORDER BY post_id DESC LIMIT 1")
    target_post_id = str(c.fetchone()[0])
    conn.close()

    print("\n--- PHASE 3: ORGANIC SECONDARY CONTAGION ---")
    for step in range(1, 15):
        actions = {}
        # Bots act
        acting_bots = random.sample(bot_agents, 3)
        for bot in acting_bots:
            actions[bot] = ManualAction(ActionType.LIKE_POST, {"post_id": target_post_id})

        # Organic Users Randomly interact with feeds
        for idx, u in enumerate(sports_agents):
            feed = get_user_feed(db_path, NUM_TECH + idx)
            if feed and random.random() < 0.3: # 30% chance to like a post they see
                post_to_like = random.choice(feed)
                actions[u] = ManualAction(ActionType.LIKE_POST, {"post_id": str(post_to_like)})

        await env.step(actions)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(f"SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ? AND user_id >= {NUM_TECH} AND user_id < {NUM_TECH + NUM_SPORTS}", (target_post_id,))
        sports_reach = c.fetchone()[0]
        conn.close()

        history_steps.append(step)
        history_sports_reach.append(sports_reach)
        print(f"Step {step}: Tech Propaganda Reach in Sports Crowd = {sports_reach}")

    await env.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_steps, history_sports_reach, marker='o', color='purple', linewidth=2)
    plt.title('Secondary Contagion: Organic Amplification')
    plt.xlabel('Timestep')
    plt.ylabel('Unique Impressions in Disjunct Crowd')
    plt.ylim(0, NUM_SPORTS)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig("scratch/organic_contagion_dashboard.png", dpi=300)

if __name__ == "__main__":
    asyncio.run(run_cross_community())
