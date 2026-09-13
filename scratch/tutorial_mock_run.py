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
import math
import os
import random
import sqlite3

import matplotlib.pyplot as plt

# Mock API key to bypass camel-ai checks

import oasis
from oasis import ActionType, ManualAction, generate_reddit_agent_graph
from oasis.social_platform.typing import DefaultPlatformType
os.environ["OPENAI_API_KEY"] = "sk-mock-key"



async def run_tutorial():
    print("Initializing OASIS Environment (Twitter/TWIN-BERT CPU Mode)...")
    db_path = os.path.abspath("./data/twitter_simulation.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path):
        os.remove(db_path)

    # We use 36 agents to give TWIN-BERT a decent latent space to map
    agent_graph = await generate_reddit_agent_graph(
        profile_path="./data/reddit/user_data_36.json",
        model=None,
        available_actions=ActionType.get_default_reddit_actions(),
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=DefaultPlatformType.TWITTER, # This triggers the TWIN-BERT neural ranker
        database_path=db_path,
    )
    await env.reset()

    # 1. Define our participants
    target_organic_user = env.agent_graph.get_agent(0)
    other_organic_users = [env.agent_graph.get_agent(i) for i in range(1, 20)]
    bot_network = [env.agent_graph.get_agent(i) for i in range(20, 36)]

    # 2. Initialize Telemetry (STRICT THRESHOLDS)
    # Narrow normal distribution so they are tightly clustered around neutral initially
    organic_opinions = {u.agent_id: max(-1.0, min(1.0, random.gauss(0.0, 0.05))) for u in other_organic_users}
    organic_exposures = {u.agent_id: set() for u in other_organic_users}

    # STRICT BOUNDS: highly resistant to extreme propaganda
    tolerance_bounds = {u.agent_id: random.uniform(0.1, 0.25) for u in other_organic_users}

    history_steps = []
    history_reach = []
    history_avg_opinion = []
    history_action_cost = []
    total_bot_actions = 0
    target_post_id = "1"

    print("\n--- PHASE 1: ORGANIC BASELINE (Burn-In) ---")
    await env.step({
        target_organic_user: ManualAction(ActionType.CREATE_POST, {"content": "Check out this controversial new tech policy!"})
    })

    # Burn-in for TWIN-BERT to establish initial embeddings
    for step in range(1, 4):
        await env.step({})
        reach = get_post_reach(db_path, target_post_id)
        history_steps.append(step)
        history_reach.append(reach)
        history_avg_opinion.append(sum(organic_opinions.values()) / len(organic_opinions))
        history_action_cost.append(total_bot_actions)
        print(f"Step {step} (Organic): Post Reach = {reach}")

    print("\n--- PHASE 2: CIB ATTACK (Targeted Like-Farm) ---")
    # Extended attack window to watch the neural network curve evolve
    for step in range(4, 16):
        bot_actions = {}
        # Trickle attack: only 2 bots act per step to show gradual algorithmic manipulation
        acting_bots = random.sample(bot_network, 2)
        for bot in acting_bots:
            bot_actions[bot] = ManualAction(ActionType.LIKE_POST, {"post_id": target_post_id})
            total_bot_actions += 1

        print(f"Step {step}: {len(acting_bots)} Bots are liking the post (TWIN-BERT Forward Pass)...")
        await env.step(bot_actions)

        reach = get_post_reach(db_path, target_post_id)
        update_opinion_drift(db_path, target_post_id, bot_network, organic_opinions, organic_exposures, tolerance_bounds)

        history_steps.append(step)
        history_reach.append(reach)
        history_avg_opinion.append(sum(organic_opinions.values()) / len(organic_opinions))
        history_action_cost.append(total_bot_actions)
        print(f"   -> Reach = {reach} | Avg Opinion = {history_avg_opinion[-1]:.3f} | Cost = {total_bot_actions}")

    await env.close()
    generate_plots(history_steps, history_reach, history_avg_opinion, history_action_cost)

def get_post_reach(db_path, post_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ?", (post_id,))
    reach = c.fetchone()[0]
    conn.close()
    return reach

def update_opinion_drift(db_path, post_id, bot_network, opinions, exposures, bounds):
    """The Dynamic Bounded Confidence Model (Strict)"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT user_id FROM rec WHERE post_id = ?", (post_id,))
    exposed_users = [row[0] for row in c.fetchall() if row[0] in opinions]

    c.execute("SELECT num_likes FROM post WHERE post_id = ?", (post_id,))
    row = c.fetchone()
    likes = row[0] if row else 0
    conn.close()

    tau = 1.0 # Propaganda stance

    for u in exposed_users:
        x_i = opinions[u]
        eps = bounds[u]
        dist = abs(tau - x_i)

        # Only update if the propaganda is within their strict tolerance boundary
        if dist <= eps:
            base_mu = 0.05 # Very slow base susceptibility

            # Dynamic Reasoning: Confirmation Bias + Logarithmic Social Proof
            confirmation_multiplier = max(0.0, 1.0 - (dist / eps))
            social_proof = 1.0 + math.log1p(likes)

            dynamic_mu = base_mu * confirmation_multiplier * social_proof
            dynamic_mu = min(dynamic_mu, 0.4) # Cap it

            opinions[u] = x_i + dynamic_mu * (tau - x_i)

def generate_plots(steps, reach, opinions, cost):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(steps, reach, marker='o', color='blue', linewidth=2)
    ax1.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5, label='Attack Starts')
    ax1.set_title('TWIN-BERT Amplification (Reach)')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Organic Impressions')
    ax1.legend()

    ax2.plot(steps, opinions, marker='s', color='green', linewidth=2)
    ax2.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Internal Opinion Drift ($\Delta E$)')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Avg Societal Belief')

    ax3.plot(cost, reach, marker='^', color='orange', linewidth=2)
    ax3.set_title('Attack Efficiency (Reach vs Cost)')
    ax3.set_xlabel('Cumulative Bot Actions (Budget)')
    ax3.set_ylabel('Algorithmic Reach')

    plt.tight_layout()
    plt.savefig("scratch/twinbert_dashboard.png", dpi=300)
    print("\nSuccess! Telemetry dashboard saved to scratch/twinbert_dashboard.png")

if __name__ == "__main__":
    asyncio.run(run_tutorial())
