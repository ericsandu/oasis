import asyncio
import sqlite3
import os
import random
import datetime
import json
import matplotlib.pyplot as plt

os.environ["OPENAI_API_KEY"] = "sk-mock-key"

import oasis
from oasis import ActionType, ManualAction, generate_reddit_agent_graph
from oasis.social_platform.typing import DefaultPlatformType

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
            bio = "Sports fanatic, loves football." # BOTS MASK AS SPORTS FANS
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
    
    # 1. Tech Echo Chamber (Tech follows Tech)
    for i in range(NUM_TECH):
        for j in range(NUM_TECH):
            if i != j and random.random() < 0.3:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))
                
    # 2. Sports Echo Chamber (Sports follows Sports)
    for i in range(NUM_TECH, NUM_TECH + NUM_SPORTS):
        for j in range(NUM_TECH, NUM_TECH + NUM_SPORTS):
            if i != j and random.random() < 0.15:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (i, j, datetime.datetime.now()))
                
    # 3. TIER 4: PARASITIC INFILTRATION (Bots bridge the graph)
    for b in range(NUM_TECH + NUM_SPORTS, TOTAL_USERS):
        # Bots aggressively follow sports fans
        for s in range(NUM_TECH, NUM_TECH + NUM_SPORTS):
            if random.random() < 0.4:
                c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (b, s, datetime.datetime.now()))
                # Trick them into mutuals
                if random.random() < 0.2:
                    c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (s, b, datetime.datetime.now()))
                    
    conn.commit()
    conn.close()

async def run_cross_community():
    profile_path = "./data/synthetic_bridge.json"
    generate_synthetic_profiles(profile_path)
    
    db_path = os.path.abspath("./data/bridge.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path): os.remove(db_path)

    agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=None,
        available_actions=ActionType.get_default_reddit_actions(),
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=DefaultPlatformType.TWITTER, recsys_type="gorse", 
        database_path=db_path,
    )
    await env.reset()
    inject_graph_topology(db_path)
    
    tech_agents = [env.agent_graph.get_agent(i) for i in range(NUM_TECH)]
    sports_agents = [env.agent_graph.get_agent(i) for i in range(NUM_TECH, NUM_TECH + NUM_SPORTS)]
    bot_agents = [env.agent_graph.get_agent(i) for i in range(NUM_TECH + NUM_SPORTS, TOTAL_USERS)]
    
    target_user = tech_agents[0]
    target_post_id = None
    
    history_steps = []
    history_sports_reach = []

    print("\n--- PHASE 1: BOTNET INFILTRATION (Semantic Masking) ---")
    initial_actions = {}
    for u in sports_agents:
        if random.random() < 0.4:
            initial_actions[u] = ManualAction(ActionType.CREATE_POST, {"content": "What a game last night! #sports"})
    
    # Bots create sports posts to align their NLP embedding
    for b in bot_agents:
        initial_actions[b] = ManualAction(ActionType.CREATE_POST, {"content": "Can't believe that touchdown! #sports"})
        
    await env.step(initial_actions)
    
    # Settle
    for _ in range(2): await env.step({})

    print("\n--- PHASE 2: THE PIVOT (Tech Propaganda) ---")
    await env.step({
        target_user: ManualAction(ActionType.CREATE_POST, {"content": "Massive new tech regulation just dropped, this changes everything. #tech #policy"})
    })
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT post_id FROM post ORDER BY post_id DESC LIMIT 1")
    target_post_id = str(c.fetchone()[0])
    conn.close()

    print("\n--- PHASE 3: CROSS-COMMUNITY SPILLOVER ---")
    for step in range(1, 12):
        bot_actions = {}
        acting_bots = random.sample(bot_agents, 3)
        for bot in acting_bots:
            bot_actions[bot] = ManualAction(ActionType.LIKE_POST, {"post_id": target_post_id})
            
        await env.step(bot_actions)
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        # Calculate how many SPORTS fans saw the tech post
        c.execute(f"SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ? AND user_id >= {NUM_TECH} AND user_id < {NUM_TECH + NUM_SPORTS}", (target_post_id,))
        sports_reach = c.fetchone()[0]
        conn.close()
        
        history_steps.append(step)
        history_sports_reach.append(sports_reach)
        print(f"Step {step}: Tech Propaganda Reach in Sports Crowd = {sports_reach}")

    await env.close()
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(history_steps, history_sports_reach, marker='o', color='red', linewidth=2)
    plt.title('Tier 4: Cross-Community Spillover (Tech Post -> Sports Feed)')
    plt.xlabel('Bot Pivot Actions (Steps)')
    plt.ylabel('Unique Impressions in Disjunct Crowd')
    plt.ylim(0, NUM_SPORTS)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig("scratch/cross_community_dashboard.png", dpi=300)

if __name__ == "__main__":
    asyncio.run(run_cross_community())
