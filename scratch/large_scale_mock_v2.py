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

import oasis
from oasis import ActionType, ManualAction, generate_reddit_agent_graph
from oasis.social_platform.typing import DefaultPlatformType

os.environ["OPENAI_API_KEY"] = "sk-mock-key"


NUM_ALIGNED = 30
NUM_DISJUNCT = 170
NUM_BOTS = 50
TOTAL_USERS = NUM_ALIGNED + NUM_DISJUNCT + NUM_BOTS

async def run_large_scale():
    profile_path = "./data/synthetic_250.json"
    db_path = os.path.abspath("./data/large_scale_v2.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path):
        os.remove(db_path)

    agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=None,
        available_actions=ActionType.get_default_reddit_actions(),
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=DefaultPlatformType.TWITTER,
        database_path=db_path,
    )
    await env.reset()

    target_user = env.agent_graph.get_agent(0)
    aligned_agents = [env.agent_graph.get_agent(i) for i in range(1, NUM_ALIGNED)]
    disjunct_agents = [env.agent_graph.get_agent(i) for i in range(NUM_ALIGNED, NUM_ALIGNED + NUM_DISJUNCT)]
    bot_agents = [env.agent_graph.get_agent(i) for i in range(NUM_ALIGNED + NUM_DISJUNCT, TOTAL_USERS)]

    opinions = {u.agent_id: random.gauss(0.2, 0.1) for u in aligned_agents}
    bounds = {u.agent_id: random.uniform(0.3, 0.6) for u in aligned_agents}
    opinions.update({u.agent_id: random.gauss(0.0, 0.05) for u in disjunct_agents})
    bounds.update({u.agent_id: random.uniform(0.05, 0.15) for u in disjunct_agents})
    exposures = {u.agent_id: set() for u in aligned_agents + disjunct_agents}

    history_steps, history_reach, history_action_cost = [], [], []
    history_aligned_opinion, history_disjunct_opinion = [], []
    total_bot_actions = 0
    target_post_id = None # We will find this after generation

    print("\n--- PHASE 1: ORGANIC CONTENT COMPETITION ---")
    # We flood the platform with noise so TWIN-BERT has to actually rank things
    initial_actions = {}
    for u in aligned_agents:
        if random.random() < 0.6:
            initial_actions[u] = ManualAction(ActionType.CREATE_POST, {"content": "Tech stocks are wild today! #crypto"})
    for u in disjunct_agents:
        if random.random() < 0.8:
            topics = ["Did you see the game last night?", "Just baked a fresh batch of cookies!", "Best indie movie of the year.", "Hitting the gym!"]
            initial_actions[u] = ManualAction(ActionType.CREATE_POST, {"content": random.choice(topics)})

    # The target payload
    initial_actions[target_user] = ManualAction(ActionType.CREATE_POST, {"content": "Check out this controversial new tech policy! #tech #policy"})

    await env.step(initial_actions)

    # Find the target post ID (it will be the one created by target_user)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT post_id FROM post WHERE user_id = ?", (target_user.agent_id,))
    c.execute("SELECT post_id FROM post ORDER BY post_id DESC LIMIT 1")
    target_post_id = str(c.fetchone()[0])
    conn.close()

    # Let TWIN-BERT settle feeds for 2 steps organically
    for step in range(1, 3):
        await env.step({})
        reach = get_post_reach(db_path, target_post_id)
        print(f"Step {step}: Organic Reach = {reach}")
        log_telemetry(step, reach, total_bot_actions, aligned_agents, disjunct_agents, opinions, history_steps, history_reach, history_action_cost, history_aligned_opinion, history_disjunct_opinion)

    print("\n--- PHASE 2: CIB ATTACK ---")
    for step in range(3, 15):
        bot_actions = {}
        acting_bots = random.sample(bot_agents, 6) # Bots push harder to fight the noise
        for bot in acting_bots:
            bot_actions[bot] = ManualAction(ActionType.LIKE_POST, {"post_id": target_post_id})
            total_bot_actions += 1

        await env.step(bot_actions)

        reach = get_post_reach(db_path, target_post_id)
        update_opinion_drift(db_path, target_post_id, bot_agents, opinions, exposures, bounds)
        log_telemetry(step, reach, total_bot_actions, aligned_agents, disjunct_agents, opinions, history_steps, history_reach, history_action_cost, history_aligned_opinion, history_disjunct_opinion)
        print(f"   -> Reach = {reach} | Cost = {total_bot_actions}")

    await env.close()
    generate_plots(history_steps, history_reach, history_aligned_opinion, history_disjunct_opinion, history_action_cost)

def log_telemetry(step, reach, cost, aligned, disjunct, opinions, h_steps, h_reach, h_cost, h_align, h_disj):
    h_steps.append(step)
    h_reach.append(reach)
    h_cost.append(cost)
    h_align.append(sum([opinions[u.agent_id] for u in aligned]) / len(aligned))
    h_disj.append(sum([opinions[u.agent_id] for u in disjunct]) / len(disjunct))

def get_post_reach(db_path, post_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT user_id) FROM rec WHERE post_id = ?", (post_id,))
    reach = c.fetchone()[0]
    conn.close()
    return reach

def update_opinion_drift(db_path, post_id, bot_network, opinions, exposures, bounds):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT user_id FROM rec WHERE post_id = ?", (post_id,))
    exposed_users = [row[0] for row in c.fetchall() if row[0] in opinions]
    c.execute("SELECT num_likes FROM post WHERE post_id = ?", (post_id,))
    row = c.fetchone()
    likes = row[0] if row else 0
    conn.close()

    tau = 1.0
    for u in exposed_users:
        x_i = opinions[u]
        eps = bounds[u]
        dist = abs(tau - x_i)

        if dist <= eps:
            base_mu = 0.05
            confirmation = max(0.0, 1.0 - (dist / eps))
            social_proof = 1.0 + math.log1p(likes)
            dynamic_mu = min(base_mu * confirmation * social_proof, 0.4)
            opinions[u] = x_i + dynamic_mu * (tau - x_i)

def generate_plots(steps, reach, align_op, disj_op, cost):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(steps, reach, marker='o', color='blue', linewidth=2)
    ax1.axvline(x=2.5, color='gray', linestyle='--')
    ax1.set_title('Content Competition Reach (250 Users)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Impressions (Fighting Noise)')
    ax1.set_ylim(0, 250)

    ax2.plot(steps, align_op, marker='s', color='#00ffcc', linewidth=2, label='Aligned Sub-graph')
    ax2.plot(steps, disj_op, marker='x', color='#ff3366', linewidth=2, label='Disjunct Sub-graph')
    ax2.axvline(x=2.5, color='gray', linestyle='--')
    ax2.set_title(r'Polarized Opinion Drift ($\Delta E$)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Avg Belief')
    ax2.set_ylim(-0.2, 1.0)
    ax2.legend()

    ax3.plot(cost, reach, marker='^', color='orange', linewidth=2)
    ax3.set_title('Efficiency vs Algorithmic Resistance')
    ax3.set_xlabel('Budget (Bot Actions)')
    ax3.set_ylabel('Reach')

    plt.tight_layout()
    plt.savefig("scratch/large_scale_dashboard_v2.png", dpi=300)

if __name__ == "__main__":
    asyncio.run(run_large_scale())
