"""CLI Runner script for executing CIB experiments locally or on UPB Grid clusters.

Supports both local hermetic CPU execution and distributed GPU execution with vLLM,
across multi-topic and ideologically polarized network topologies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

# Ensure oasis root is on sys.path
_oasis_root = Path(__file__).resolve().parents[2]
if str(_oasis_root) not in sys.path:
    sys.path.insert(0, str(_oasis_root))

# Enforce thread limit (4 max) for cluster and local execution
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(var, "4")

# Lightweight CPU mocks for heavy ML dependencies if running in non-GPU environment
for pkg in (
    "torch",
    "torch.nn",
    "sentence_transformers",
    "transformers",
    "sklearn",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
):
    try:
        __import__(pkg)
    except ImportError:
        from unittest.mock import MagicMock

        sys.modules[pkg] = MagicMock()

from cib_zoo.agent.cib_agent import CIBAgent, NoOpModelBackend
from cib_zoo.metrics.amplification import (
    calculate_causal_amplification,
    calculate_differential_amplification,
    calculate_exposure_from_db,
)
from cib_zoo.presets import create_s1_campaign, create_s2_campaign, create_s3_campaign
from cib_zoo.topology.network_builder import build_network
from oasis.environment.env import OasisEnv
from oasis.environment.env_action import LLMAction, ManualAction
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("cib_zoo.runner")


def make_user_info(agent_id: int, name: str) -> UserInfo:
    return UserInfo(
        user_name=f"bot_{agent_id:05d}",
        name=name,
        description=f"CIB coordinated squad persona for {name}",
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
        recsys_type="twitter",
        is_controllable=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CIB attack campaigns in OASIS simulation."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "s1", "s2", "s3"],
        default="s1",
        help="Campaign preset to execute: 'baseline' (0 bots control) or 's1', 's2', 's3'.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=["multitopic", "polarized"],
        default="multitopic",
        help="Network topology: 'multitopic' (Tech/Politics/Sports) or 'polarized' (Progressive/Conservative).",
    )
    parser.add_argument(
        "--topic-mode",
        type=str,
        choices=["existing", "cold_start"],
        default="existing",
        help="Topic regime: 'existing' (topic has baseline history) or 'cold_start' (brand new 0-baseline).",
    )
    parser.add_argument(
        "--num-organic",
        type=int,
        default=150,
        help="Number of organic agents in the network (default: 150).",
    )
    parser.add_argument(
        "--bot-ratio",
        type=float,
        default=None,
        help="Bot infiltration ratio relative to organic population (e.g. 0.10, 0.15, 0.20). Overrides --num-bots.",
    )
    parser.add_argument(
        "--num-bots",
        type=int,
        default=4,
        help="Absolute number of coordinated bot accounts (used if --bot-ratio is not set).",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="",
        help="URL of vLLM server (e.g. 'http://127.0.0.1:8000/v1'). If omitted, runs in local CPU mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        help="Model name identifier for vLLM.",
    )
    parser.add_argument(
        "--active-ratio",
        type=float,
        default=0.35,
        help="Proportion of organic agents sampled to take actions per step (default: 0.35).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum simulation steps to run.",
    )
    parser.add_argument(
        "--recsys-type",
        type=str,
        default="reddit",
        help="Recsys type to initialize (reddit, twitter, random).",
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
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    # Determine effective number of bots
    if args.preset == "baseline":
        num_bots = 0
    elif args.bot_ratio is not None:
        num_bots = max(1, int(args.num_organic * args.bot_ratio))
    else:
        num_bots = max(1, args.num_bots)

    logger.info(
        f"Initializing CIB runner: preset={args.preset}, topology={args.topology}, "
        f"topic_mode={args.topic_mode}, organic={args.num_organic}, bots={num_bots} "
        f"({(num_bots / args.num_organic * 100):.1f}%), max_steps={args.max_steps}, recsys={args.recsys_type}"
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

        # 1. Resolve Model Backend for Organic Agents
        is_llm_mode = False
        organic_model = None
        if args.vllm_url:
            from oasis.inference import check_vllm_health, get_vllm_model

            if check_vllm_health(args.vllm_url, timeout=4.0):
                logger.info(
                    f"✓ vLLM central server reachable at {args.vllm_url}. Using model: {args.model}"
                )
                organic_model = get_vllm_model(
                    model_type=args.model,
                    url=args.vllm_url,
                    temperature=0.7,
                    max_tokens=256,
                )
                is_llm_mode = True
            else:
                logger.warning(
                    f"⚠ vLLM server at {args.vllm_url} unreachable. Falling back to local hermetic mode."
                )
                organic_model = NoOpModelBackend()
        else:
            logger.info("Running in hermetic local CPU mode (No LLM API calls).")
            organic_model = NoOpModelBackend()

        # 2. Build Organic Network Topology
        agent_graph, organic_agents, community_map = build_network(
            topology=args.topology,
            num_users=args.num_organic,
            model=organic_model,
            channel=channel,
        )

        # 3. Instantiate CIB Bots (if preset != baseline)
        bot_agents: dict[int, CIBAgent] = {}
        bot_ids: list[int] = []
        if num_bots > 0:
            bot_start_id = args.num_organic + 1
            bot_ids = list(range(bot_start_id, bot_start_id + num_bots))
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
            f"Network assembled: {len(organic_agents)} organic agents ({args.topology}) | "
            f"{CIBAgent.get_instance_count()} CIB bots registered."
        )

        env = OasisEnv(
            agent_graph=agent_graph,
            platform=platform,
            database_path=str(db_path),
        )

        # 4. Step 0: Seed Posts according to topic mode
        seed_actions: dict = {}
        if args.topic_mode == "existing":
            # Topic has established baseline and target post circulating
            seed_actions[organic_agents[0]] = [
                ManualAction(
                    ActionType.CREATE_POST,
                    {"content": "Organic topic discussion #baseline"},
                ),
                ManualAction(
                    ActionType.CREATE_POST,
                    {"content": "Controversial payload narrative #target"},
                ),
            ]
        else:
            # Cold-start: Only baseline post exists at step 0; target post injected in step 1
            seed_actions[organic_agents[0]] = [
                ManualAction(
                    ActionType.CREATE_POST,
                    {"content": "Organic topic discussion #baseline"},
                ),
            ]

        await env.step(seed_actions)

        # Retrieve seeded post IDs
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT post_id FROM post ORDER BY post_id ASC")
        seeded_posts = [r[0] for r in cur.fetchall()]
        conn.close()

        baseline_post_id = seeded_posts[0] if len(seeded_posts) > 0 else 1
        payload_post_id = (
            seeded_posts[1] if len(seeded_posts) > 1 else (baseline_post_id + 1)
        )

        # In cold-start mode, if payload post was not seeded by organic agent, bot squad seeds it at step 1
        cold_start_seeded = len(seeded_posts) > 1

        # 5. Compile Campaign Preset (if bots present)
        campaign = None
        warmup_steps = max(1, args.max_steps // 2)
        strike_steps = max(1, args.max_steps - warmup_steps)

        if num_bots > 0 and args.preset != "baseline":
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
            elif args.preset == "s3":
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

        campaign_name = campaign.campaign_name if campaign else "OrganicControlBaseline"
        total_steps = campaign.total_steps if campaign else args.max_steps
        logger.info(f"Executing campaign '{campaign_name}' ({total_steps} steps)...")

        # 6. Main Simulation Loop
        exposure_timeline = []

        for step in range(total_steps):
            step_actions: dict = {}

            # Cold-start injection at step 0 if needed
            if args.topic_mode == "cold_start" and not cold_start_seeded and bot_agents:
                first_bot = bot_agents[bot_ids[0]]
                step_actions[first_bot] = [
                    ManualAction(
                        ActionType.CREATE_POST,
                        {"content": "Controversial payload narrative #target"},
                    )
                ]
                cold_start_seeded = True

            # Sample active organic agents
            num_active = max(1, int(len(organic_agents) * args.active_ratio))
            active_organic = random.sample(organic_agents, num_active)

            for org_agent in active_organic:
                if is_llm_mode:
                    step_actions[org_agent] = LLMAction()
                else:
                    # In local CPU mode: default to DO_NOTHING
                    step_actions[org_agent] = [ManualAction(ActionType.DO_NOTHING, {})]

            # Generate CIB actions for active squads
            if campaign is not None:
                phase, rel_step = campaign.get_active_phase(step)
                for squad in phase.squads:
                    squad_bot_instances = [
                        bot_agents[b_id] for b_id in squad.bot_ids if b_id in bot_agents
                    ]
                    squad_actions = squad.pattern.generate_step_actions(
                        step=rel_step,
                        squad_bots=squad_bot_instances,
                        context={
                            "global_step": step,
                            "baseline_post_id": baseline_post_id,
                            "payload_post_id": payload_post_id,
                        },
                    )
                    for b_id, action_list in squad_actions.items():
                        bot_agent = bot_agents.get(b_id)
                        if bot_agent:
                            step_actions[bot_agent] = action_list

            await env.step(step_actions)

            # Record step exposure telemetry
            e_base_t = calculate_exposure_from_db(str(db_path), baseline_post_id)
            e_pay_t = calculate_exposure_from_db(str(db_path), payload_post_id)
            exposure_timeline.append(
                {
                    "step": step + 1,
                    "baseline_exposure": e_base_t,
                    "payload_exposure": e_pay_t,
                    "net_lift": e_pay_t - e_base_t,
                    "actions_dispatched": len(step_actions),
                }
            )

            logger.info(
                f"Step {step + 1}/{total_steps} completed | "
                f"Payload Exposure: {e_pay_t:.1f}, Baseline: {e_base_t:.1f}, Lift: {(e_pay_t - e_base_t):.1f}"
            )

        # 7. Compute Global Evaluation Metrics
        e_baseline = calculate_exposure_from_db(str(db_path), baseline_post_id)
        e_payload = calculate_exposure_from_db(str(db_path), payload_post_id)

        if num_bots > 0:
            diff_amplification = calculate_differential_amplification(
                db_path=str(db_path),
                payload_post_id=payload_post_id,
                baseline_post_id=baseline_post_id,
            )
            ratio_amplification = calculate_causal_amplification(
                db_path=str(db_path),
                payload_post_id=payload_post_id,
                baseline_post_id=baseline_post_id,
                n_bots=num_bots,
                mode="ratio",
            )
        else:
            diff_amplification = 0.0
            ratio_amplification = 1.0

        # 8. Cross-Community Engagement Breakdown
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
            cur.execute(
                f"SELECT COUNT(*) FROM trace WHERE user_id IN ({placeholders})",
                tuple(active_bot_ids),
            )
            total_bot_actions = cur.fetchone()[0]
        else:
            total_bot_actions = 0

        # Community-level breakdown for payload post
        community_stats: dict[str, dict[str, int]] = {}
        unique_communities = set(community_map.values())
        for comm in unique_communities:
            members = [aid for aid, c in community_map.items() if c == comm]
            m_placeholders = ",".join("?" for _ in members)

            cur.execute(
                f"SELECT COUNT(*) FROM like WHERE post_id = ? AND user_id IN ({m_placeholders})",
                (payload_post_id, *members),
            )
            comm_likes = cur.fetchone()[0]

            cur.execute(
                f"SELECT COUNT(*) FROM comment WHERE post_id = ? AND user_id IN ({m_placeholders})",
                (payload_post_id, *members),
            )
            comm_comments = cur.fetchone()[0]

            cur.execute(
                f"SELECT COUNT(*) FROM rec WHERE post_id = ? AND user_id IN ({m_placeholders})",
                (payload_post_id, *members),
            )
            comm_recs = cur.fetchone()[0]

            community_stats[comm] = {
                "member_count": len(members),
                "payload_likes": comm_likes,
                "payload_comments": comm_comments,
                "payload_impressions": comm_recs,
            }

        conn.close()

        results = {
            "status": "success",
            "preset": args.preset,
            "campaign_name": campaign_name,
            "topology": args.topology,
            "topic_mode": args.topic_mode,
            "num_organic": args.num_organic,
            "num_bots": num_bots,
            "bot_ratio": num_bots / args.num_organic if args.num_organic > 0 else 0.0,
            "bot_ids": active_bot_ids,
            "total_steps": total_steps,
            "baseline_post_id": baseline_post_id,
            "payload_post_id": payload_post_id,
            "exposure_baseline": e_baseline,
            "exposure_payload": e_payload,
            "differential_amplification": diff_amplification,
            "ratio_amplification": ratio_amplification,
            "is_llm_mode": is_llm_mode,
            "model_name": args.model if is_llm_mode else "hermetic_noop",
            "db_path": str(db_path),
            "telemetry": {
                "total_posts": total_posts,
                "total_likes": total_likes,
                "total_comments": total_comments,
                "total_bot_actions": total_bot_actions,
            },
            "community_telemetry": community_stats,
            "exposure_timeline": exposure_timeline,
            "timestamp": time.time(),
        }

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump(results, f, indent=2)

        logger.info(
            f"Campaign completed! Differential Lift: {diff_amplification:.4f} | "
            f"Ratio: {ratio_amplification:.4f} | Total Bot Actions: {total_bot_actions}"
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
