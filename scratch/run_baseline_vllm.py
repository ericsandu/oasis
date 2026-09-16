#!/usr/bin/env python3
"""
Baseline OASIS Simulation with vLLM (Qwen-2.5-32B-Instruct) and Gorse Recommender.
Runs pure organic user interactions across multiple steps without any CIB attacks.
"""

import asyncio
import os
import sys
import time
import argparse
from typing import Optional

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import oasis
from oasis import ActionType, LLMAction, ManualAction, generate_twitter_agent_graph
from oasis.inference import get_vllm_model, check_vllm_health
from camel.logger import get_logger

logger = get_logger("baseline_vllm_simulation")


async def run_baseline_simulation(
    num_steps: int = 5,
    db_path: str = "./data/baseline_simulation.db",
    profile_path: str = "data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv",
    vllm_url: str = "http://127.0.0.1:8000/v1",
    model_name: str = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    active_agent_ratio: float = 0.5,
):
    print("=" * 60)
    print("OASIS Baseline Simulation (Organic / Non-CIB)")
    print(f"Model: {model_name}")
    print(f"vLLM Endpoint: {vllm_url}")
    print(f"Steps: {num_steps} | DB: {db_path}")
    print("=" * 60)

    # 1. Health check for vLLM
    print(f"Checking connectivity to vLLM server at {vllm_url}...")
    if check_vllm_health(url=vllm_url, timeout=4.0):
        print("✓ vLLM central node is reachable and healthy.")
    else:
        print(f"⚠ Warning: vLLM server at {vllm_url} did not respond to healthcheck.")
        print("  Proceeding assuming vLLM is initializing or accessible via OpenAI API route.")

    # 2. Database preparation
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path} for clean run...")
        os.remove(db_path)
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    # 3. Create Model Backend
    model = get_vllm_model(
        model_type=model_name,
        url=vllm_url,
        temperature=0.7,
        max_tokens=512,
    )

    # 4. Generate Agent Graph
    available_actions = ActionType.get_default_twitter_actions()
    resolved_profile = os.path.join(REPO_ROOT, profile_path) if not os.path.isabs(profile_path) else profile_path
    
    print(f"Loading agent profiles from: {resolved_profile}...")
    agent_graph = await generate_twitter_agent_graph(
        profile_path=resolved_profile,
        model=model,
        available_actions=available_actions,
    )

    total_agents = len(agent_graph.get_agents())
    print(f"✓ Agent graph constructed with {total_agents} organic agents.")

    # 5. Initialize OASIS Environment with Gorse Recommender
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )

    print("Resetting simulation environment...")
    await env.reset()

    # Step 0: Seed initial organic discussion
    print("\n--- Seeding Initial Organic Posts ---")
    seed_actions = {}
    seed_agents = [agent_graph.get_agent(i) for i in range(min(5, total_agents))]
    sample_topics = [
        "Excited to share our new research on high-throughput multi-agent systems!",
        "Incredible game last night, what a finish in extra time!",
        "Tech stocks showing strong momentum heading into Q3 earnings season.",
        "Beautiful morning for a run in the park, enjoying the autumn weather.",
        "Looking forward to the upcoming AI conference next month in Bucharest.",
    ]
    for idx, agent in enumerate(seed_agents):
        if agent is not None:
            seed_actions[agent] = ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": sample_topics[idx % len(sample_topics)]},
            )
    
    await env.step(seed_actions)
    print(f"Seeded {len(seed_actions)} initial organic posts.")

    # Main Simulation Loop
    for step_num in range(1, num_steps + 1):
        step_start = time.time()
        print(f"\n>>> Executing Step {step_num} / {num_steps}...")

        # Activate a proportion of agents dynamically
        num_active = max(1, int(total_agents * active_agent_ratio))
        active_agents = [
            agent for _, agent in agent_graph.get_agents()
        ][:num_active]

        actions = {agent: LLMAction() for agent in active_agents}
        print(f"  Dispatching {len(actions)} LLM actions to vLLM (Qwen-2.5-32B)...")

        try:
            await env.step(actions)
            elapsed = time.time() - step_start
            print(f"  Step {step_num} completed in {elapsed:.2f}s ({elapsed / len(actions):.2f}s/agent).")
        except Exception as e:
            print(f"  Step {step_num} encountered an error: {e}")
            raise e

    # Close Environment
    await env.close()
    print("\n" + "=" * 60)
    print(f"Simulation completed successfully! Database stored at: {db_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Baseline OASIS Simulation with vLLM")
    parser.add_argument("--steps", type=int, default=3, help="Number of simulation steps")
    parser.add_argument("--db", type=str, default="./data/baseline_simulation.db", help="Path to output SQLite DB")
    parser.add_argument("--profile", type=str, default="data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv", help="Profile CSV path")
    parser.add_argument("--vllm-url", type=str, default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"), help="vLLM endpoint")
    parser.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"), help="HuggingFace model ID")
    parser.add_argument("--ratio", type=float, default=0.3, help="Active agent ratio per step")

    args = parser.parse_args()

    asyncio.run(
        run_baseline_simulation(
            num_steps=args.steps,
            db_path=args.db,
            profile_path=args.profile,
            vllm_url=args.vllm_url,
            model_name=args.model,
            active_agent_ratio=args.ratio,
        )
    )


if __name__ == "__main__":
    main()
