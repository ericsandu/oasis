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

NUM_TECH = 30
NUM_SPORTS = 60
NUM_MOVIES = 30
NUM_TECH_BOTS = 10
NUM_SPORTS_BOTS = 10
NUM_MOVIE_BOTS = 10

TOTAL_USERS = NUM_TECH + NUM_SPORTS + NUM_MOVIES + NUM_TECH_BOTS + NUM_SPORTS_BOTS + NUM_MOVIE_BOTS

def get_group(i):
    if i < 30: return "tech_organic"
    if i < 90: return "sports_organic"
    if i < 120: return "movies_organic"
    if i < 130: return "tech_bot"
    if i < 140: return "sports_bot"
    return "movies_bot"

def generate_synthetic_profiles(filepath):
    profiles = []
    for i in range(TOTAL_USERS):
        group = get_group(i)
        
        if "tech" in group:
            bio, mbti, prof, topic = "Tech enthusiast, crypto trader.", "INTJ", "Software", "Tech"
        elif "sports" in group:
            bio, mbti, prof, topic = "Sports fanatic, loves football.", "ESFP", "Retail", "Sports"
        else:
            bio, mbti, prof, topic = "Indie film buff, loves cinema.", "INFP", "Art", "Movies"
            
        profiles.append({
            "realname": f"User_{i}",
            "username": f"user_{i}_{group}",
            "bio": bio,
            "persona": bio,
            "mbti": mbti,
            "gender": random.choice(["male", "female"]),
            "age": random.randint(20, 50),
            "country": "US",
            "profession": prof,
            "interested_topics": [topic]
        })
    with open(filepath, "w") as f:
        json.dump(profiles, f, indent=2)

def inject_graph_topology(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    def add_edge(u, v):
        c.execute("INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)", (u, v, datetime.datetime.now()))
        
    for i in range(TOTAL_USERS):
        for j in range(TOTAL_USERS):
            if i == j: continue
            gi, gj = get_group(i), get_group(j)
            
            # Intra-cluster organic edges
            if gi == gj and "organic" in gi:
                if random.random() < 0.15: add_edge(i, j)
                
            # Sleeper bots embedding themselves in their target clusters
            if "bot" in gi and "organic" in gj and gi.split('_')[0] == gj.split('_')[0]:
                if random.random() < 0.4: add_edge(i, j)  # Bot follows organic
                if random.random() < 0.25: add_edge(j, i) # Organic trusts bot back
                
    conn.commit()
    conn.close()

async def run_sleeper_cell():
    profile_path = "./data/synthetic_sleeper.json"
    generate_synthetic_profiles(profile_path)
    db_path = os.path.abspath("./data/sleeper.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path): os.remove(db_path)

    agent_graph = await generate_reddit_agent_graph(profile_path=profile_path, model=None, available_actions=ActionType.get_default_reddit_actions())
    env = oasis.make(agent_graph=agent_graph, platform=DefaultPlatformType.TWITTER, database_path=db_path)
    await env.reset()
    inject_graph_topology(db_path)
    
    all_bots = [env.agent_graph.get_agent(i) for i in range(120, TOTAL_USERS)]
    target_user = env.agent_graph.get_agent(0)
    
    history_steps = []
    reach_sports = []
    reach_movies = []

    print("\n--- PHASE 1: SEMANTIC MASKING ---")
    initial_actions = {}
    for i in range(TOTAL_USERS):
        agent = env.agent_graph.get_agent(i)
        group = get_group(i)
        if random.random() < 0.3:
            if "sports" in group:
                initial_actions[agent] = ManualAction(ActionType.CREATE_POST, {"content": "Incredible game today! #sports"})
            elif "movies" in group:
                initial_actions[agent] = ManualAction(ActionType.CREATE_POST, {"content": "Just watched a great indie film. #cinema"})
            elif "tech" in group:
                initial_actions[agent] = ManualAction(ActionType.CREATE_POST, {"content": "New framework looks solid. #programming"})
                
    await env.step(initial_actions)
    for _ in range(2): await env.step({})

    print("\n--- PHASE 2: THE PAYLOAD DROP ---")
    await env.step({target_user: ManualAction(ActionType.CREATE_POST, {"content": "Massive new tech regulation just dropped, this changes everything for cybersecurity. #tech #policy"})})
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT post_id FROM post ORDER BY post_id DESC LIMIT 1")
    target_post_id = str(c.fetchone()[0])
    conn.close()

    print("\n--- PHASE 3: DISTRIBUTED SYBIL STRIKE ---")
    for step in range(1, 11):
        actions = {}
        # Synchronized cross-cluster bot likes
        acting_bots = all_bots
        for bot in acting_bots:
            actions[bot] = ManualAction(ActionType.LIKE_POST, {"post_id": target_post_id})
                
        await env.step(actions)
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute(f"SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ? AND user_id >= 30 AND user_id < 90", (target_post_id,))
        s_reach = c.fetchone()[0]
        
        c.execute(f"SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ? AND user_id >= 90 AND user_id < 120", (target_post_id,))
        m_reach = c.fetchone()[0]
        conn.close()
        
        history_steps.append(step)
        reach_sports.append(s_reach)
        reach_movies.append(m_reach)
        
        print(f"Step {step} | Sports Reach: {s_reach}/60 | Movies Reach: {m_reach}/30")

    await env.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_steps, reach_sports, marker='o', color='purple', label='Reach in Sports (Target 60)')
    plt.plot(history_steps, reach_movies, marker='s', color='orange', label='Reach in Movies (Target 30)')
    plt.title('Distributed Sybil Strike: The "Wormhole" Effect')
    plt.xlabel('Bot Pivot Actions (Timesteps)')
    plt.ylabel('Unique Impressions per Demographic')
    plt.ylim(0, 65)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("scratch/sleeper_cell_dashboard.png", dpi=300)

if __name__ == "__main__":
    asyncio.run(run_sleeper_cell())
