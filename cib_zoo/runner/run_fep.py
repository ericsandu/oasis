"""CLI Runner script for executing CIB experiments locally or on UPB Grid clusters."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time

# Ensure oasis root is on sys.path
_oasis_root = Path(__file__).resolve().parents[2]
if str(_oasis_root) not in sys.path:
    sys.path.insert(0, str(_oasis_root))

# Enforce thread limit (4 max) for cluster and local execution
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(var, "4")

# Lightweight CPU mocks for heavy ML dependencies if running in non-GPU environment
for pkg in (
    "torch", "torch.nn", "sentence_transformers", "transformers",
    "sklearn", "sklearn.feature_extraction", "sklearn.feature_extraction.text",
    "sklearn.metrics", "sklearn.metrics.pairwise"
):
    try:
        __import__(pkg)
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules[pkg] = MagicMock()

from oasis.environment.env import OasisEnv
from oasis.environment.env_action import ManualAction
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.database import create_db
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

from cib_zoo.agent.cib_agent import CIBAgent, NoOpModelBackend
from cib_zoo.metrics.amplification import (
    calculate_causal_amplification,
    calculate_differential_amplification,
)
from cib_zoo.presets import create_s1_campaign, create_s2_campaign, create_s3_campaign

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("cib_zoo.runner")


def make_user_info(agent_id: int, name: str) -> UserInfo:
    return UserInfo(
        user_name=f"user_{agent_id:05d}",
        name=name,
        description=f"Persona description for {name}",
        profile={
            "nodes": [],
            "edges": [],
            "other_info": {
                "user_profile": f"Profile for {name}",
                "mbti": "INTJ",
                "gender": "non-binary",
                "age": 25,
                "country": "US",
                "activity_level": ["active"] * 24,
                "activity_level_frequency": [1] * 24,
                "active_threshold": [0.5] * 24,
            },
        },
        recsys_type="reddit",
        is_controllable=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIB attack campaigns in OASIS simulation.")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["s1", "s2", "s3"],
        default="s1",
        help="Campaign preset to execute (s1, s2, s3).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./cib_experiment.db",
        help="Path to SQLite simulation database.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./cib_results.json",
        help="Path to save experiment evaluation telemetry.",
    )
    parser.add_argument(
        "--num-bots",
        type=int,
        default=4,
        help="Total number of coordinated bot accounts.",
    )
    parser.add_argument(
        "--num-organic",
        type=int,
        default=4,
        help="Number of background organic agents (IDs 1..N).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum simulation steps to run.",
    )
    parser.add_argument(
        "--recsys-type",
        type=str,
        default="reddit",
        help="Recsys type to initialize (reddit, twitter, random).",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    logger.info(
        f"Initializing CIB runner: preset={args.preset}, bots={args.num_bots}, "
        f"organic={args.num_organic}, max_steps={args.max_steps}, recsys={args.recsys_type}"
    )

    db_path = Path(args.db_path).resolve()
    if db_path.exists():
        db_path.unlink()

    channel = Channel()
    platform = Platform(
        db_path=str(db_path),
        channel=channel,
        recsys_type=args.recsys_type,
        show_score=False,
        refresh_rec_post_count=5,
        max_rec_post_len=10,
    )
    platform_task = asyncio.create_task(platform.running())

    try:
        CIBAgent.reset_registry()
        agent_graph = AgentGraph()

        # 1. Instantiate Organic Agents (IDs 1 .. num_organic) as standard SocialAgent
        organic_agents = []
        for i in range(1, args.num_organic + 1):
            agent = SocialAgent(
                agent_id=i,
                user_info=make_user_info(i, f"Organic User {i}"),
                channel=channel,
                model=NoOpModelBackend(),
            )
            agent_graph.add_agent(agent)
            organic_agents.append(agent)

        # 2. Instantiate CIB Bots sequentially right after organic agents
        bot_start_id = args.num_organic + 1
        bot_ids = list(range(bot_start_id, bot_start_id + args.num_bots))
        bot_agents: dict[int, CIBAgent] = {}
        for b_id in bot_ids:
            bot = CIBAgent(
                agent_id=b_id,
                user_info=make_user_info(b_id, f"CIB Squad Bot {b_id}"),
                channel=channel,
                max_actions_per_step=5,
            )
            agent_graph.add_agent(bot)
            bot_agents[b_id] = bot

        logger.info(
            f"Instantiated {len(organic_agents)} organic agents (IDs 1..{args.num_organic}) "
            f"and {CIBAgent.get_instance_count()} CIB bots (IDs {CIBAgent.get_bot_ids()})"
        )

        env = OasisEnv(
            agent_graph=agent_graph,
            platform=platform,
            database_path=str(db_path),
        )

        # 3. Seed initial baseline & target posts in Step 0
        seed_actions = {
            organic_agents[0]: [
                ManualAction(ActionType.CREATE_POST, {"content": "Organic topic discussion #baseline"}),
                ManualAction(ActionType.CREATE_POST, {"content": "Controversial payload narrative #target"}),
            ]
        }
        await env.step(seed_actions)

        # Retrieve seeded post IDs
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT post_id FROM post ORDER BY post_id ASC LIMIT 2")
        posts = cur.fetchall()
        conn.close()

        baseline_post_id = posts[0][0] if len(posts) > 0 else 1
        payload_post_id = posts[1][0] if len(posts) > 1 else 2

        # 4. Compile the Campaign Preset
        warmup_steps = max(1, args.max_steps // 2)
        strike_steps = max(1, args.max_steps - warmup_steps)

        if args.preset == "s1":
            campaign = create_s1_campaign(
                warmup_bot_ids=bot_ids[: len(bot_ids) // 2 or 1],
                strike_bot_ids=bot_ids,
                influencer_ids=[organic_agents[0].social_agent_id],
                anchor_post_id=baseline_post_id,
                payload_post_id=payload_post_id,
                payload_template="Amplifying payload narrative #target",
                warmup_steps=warmup_steps,
                strike_steps=strike_steps,
            )
        elif args.preset == "s2":
            campaign = create_s2_campaign(
                sentinel_bot_ids=[bot_ids[0]],
                strike_bot_ids=bot_ids[1:] if len(bot_ids) > 1 else bot_ids,
                target_post_id=payload_post_id,
                raid_templates=["Perspective A #target", "Perspective B #target"],
                discovery_steps=warmup_steps,
                strike_steps=strike_steps,
            )
        else:
            campaign = create_s3_campaign(
                bridge_bot_ids=[bot_ids[0]],
                raid_bot_ids=[bot_ids[1 % len(bot_ids)]],
                astroturf_bot_ids=bot_ids,
                target_influencer_ids=[organic_agents[0].social_agent_id],
                target_post_id=payload_post_id,
                raid_templates=["Raid commentary #target"],
                astroturf_hashtag="target",
                astroturf_template="Astroturf viral broadcast #target",
            )

        logger.info(f"Executing campaign '{campaign.campaign_name}' ({campaign.total_steps} total steps)...")

        # 5. Execute Simulation Steps
        for step in range(campaign.total_steps):
            phase, rel_step = campaign.get_active_phase(step)
            step_actions: dict = {}

            # Generate CIB actions for each squad active in this phase
            for squad in phase.squads:
                squad_bot_instances = [bot_agents[b_id] for b_id in squad.bot_ids if b_id in bot_agents]
                squad_actions = squad.pattern.generate_step_actions(
                    step=rel_step,
                    squad_bots=squad_bot_instances,
                    context={"global_step": step, "baseline_post_id": baseline_post_id, "payload_post_id": payload_post_id},
                )
                for b_id, action_list in squad_actions.items():
                    bot_agent = bot_agents.get(b_id)
                    if bot_agent:
                        step_actions[bot_agent] = action_list

            # Organic agents interact with feed or remain idle
            for org_agent in organic_agents[1:]:
                step_actions[org_agent] = [ManualAction(ActionType.DO_NOTHING, {})]

            await env.step(step_actions)
            logger.info(f"Completed Step {step + 1}/{campaign.total_steps} (Phase: '{phase.phase_name}', Actions: {len(step_actions)})")

        # 6. Compute Evaluation Metrics
        diff_amplification = calculate_differential_amplification(
            db_path=str(db_path),
            payload_post_id=payload_post_id,
            baseline_post_id=baseline_post_id,
        )

        ratio_amplification = calculate_causal_amplification(
            db_path=str(db_path),
            payload_post_id=payload_post_id,
            baseline_post_id=baseline_post_id,
            n_bots=CIBAgent.get_instance_count(),
            mode="ratio",
        )

        # Database Telemetry
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM post")
        total_posts = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM like")
        total_likes = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM comment")
        total_comments = cur.fetchone()[0]
        active_bot_ids = CIBAgent.get_bot_ids()
        if active_bot_ids:
            placeholders = ",".join("?" for _ in active_bot_ids)
            cur.execute(f"SELECT COUNT(*) FROM trace WHERE user_id IN ({placeholders})", tuple(active_bot_ids))
            total_bot_actions = cur.fetchone()[0]
        else:
            total_bot_actions = 0
        conn.close()

        results = {
            "status": "success",
            "preset": args.preset,
            "campaign_name": campaign.campaign_name,
            "num_bots": CIBAgent.get_instance_count(),
            "bot_ids": CIBAgent.get_bot_ids(),
            "total_steps": campaign.total_steps,
            "differential_amplification": diff_amplification,
            "ratio_amplification": ratio_amplification,
            "db_path": str(db_path),
            "telemetry": {
                "total_posts": total_posts,
                "total_likes": total_likes,
                "total_comments": total_comments,
                "total_bot_actions": total_bot_actions,
            },
            "timestamp": time.time(),
        }

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Campaign completed! Differential Lift: {diff_amplification:.4f} | "
            f"Ratio Amplification: {ratio_amplification:.4f} | Total Bot Actions: {total_bot_actions}"
        )
        logger.info(f"Results successfully saved to {output_path}")

    finally:
        platform_task.cancel()
        try:
            await platform_task
        except asyncio.CancelledError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
