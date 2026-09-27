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

import shutil

from cib_zoo.agent.cib_agent import CIBAgent, NoOpModelBackend
from cib_zoo.metrics.amplification import (
    calculate_causal_amplification,
    calculate_differential_amplification,
    calculate_exposure_from_db,
)
from cib_zoo.modifiers.bio_scraping import BioScrapingModifier
from cib_zoo.presets import create_s1_campaign, create_s2_campaign, create_s3_campaign
from cib_zoo.topology.network_builder import build_network, sync_network_to_db
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
logging.root.setLevel(logging.INFO)
for _h in logging.root.handlers:
    _h.setLevel(logging.INFO)
logger = logging.getLogger("cib_zoo.runner")
logger.setLevel(logging.INFO)
logging.getLogger("oasis.environment.jev_env").setLevel(logging.INFO)
logging.getLogger("oasis.inference.jev_classifier").setLevel(logging.INFO)


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


def compile_campaign_preset(
    preset: str,
    bot_ids: list[int],
    organic_agents: list[Any],
    baseline_post_id: int,
    payload_post_id: int,
    total_campaign_steps: int,
) -> Any:
    """Compile declarative campaign config for the specified preset."""
    if not bot_ids or preset == "baseline":
        return None
    warmup_steps = max(1, total_campaign_steps // 2)
    strike_steps = max(1, total_campaign_steps - warmup_steps)
    if preset == "s1":
        return create_s1_campaign(
            warmup_bot_ids=bot_ids[: len(bot_ids) // 2 or 1],
            strike_bot_ids=bot_ids,
            influencer_ids=[organic_agents[0].social_agent_id],
            anchor_post_id=baseline_post_id,
            payload_post_id=payload_post_id,
            payload_template="Amplifying payload narrative #target",
            warmup_steps=warmup_steps,
            strike_steps=strike_steps,
        )
    elif preset == "s2":
        return create_s2_campaign(
            sentinel_bot_ids=[bot_ids[0]],
            strike_bot_ids=bot_ids[1:] if len(bot_ids) > 1 else bot_ids,
            target_post_id=payload_post_id,
            raid_templates=["Perspective A #target", "Perspective B #target"],
            discovery_steps=warmup_steps,
            strike_steps=strike_steps,
        )
    elif preset == "s3":
        return create_s3_campaign(
            bridge_bot_ids=[bot_ids[0]],
            raid_bot_ids=[bot_ids[1 % len(bot_ids)]],
            astroturf_bot_ids=bot_ids,
            target_influencer_ids=[organic_agents[0].social_agent_id],
            target_post_id=payload_post_id,
            raid_templates=["Raid commentary #target"],
            astroturf_hashtag="target",
            astroturf_template="Astroturf viral broadcast #target",
        )
    return None


async def execute_narrative_injection(
    env: Any,
    platform: Any,
    organic_agents: list[Any],
    bot_agents: dict[int, Any],
    bot_ids: list[int],
    db_path: Path,
    topic_mode: str,
    preset: str,
    enable_bio_scraping: bool,
) -> tuple[int, int]:
    """Inject baseline and target narrative posts into the living platform environment."""
    injection_actions: dict = {
        organic_agents[0]: [
            ManualAction(ActionType.CREATE_POST, {"content": "Organic topic discussion #baseline"})
        ]
    }
    if topic_mode == "existing" or (preset != "baseline" and bot_agents):
        target_poster = bot_agents[bot_ids[0]] if bot_agents else organic_agents[1]
        injection_actions[target_poster] = [
            ManualAction(ActionType.CREATE_POST, {"content": "Controversial payload narrative #target"})
        ]
    await env.step(injection_actions)

    if hasattr(platform, "update_rec_table"):
        try:
            await platform.update_rec_table()
        except Exception:
            pass

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT post_id FROM post WHERE content LIKE '%#baseline%' ORDER BY post_id ASC LIMIT 1")
    row_base = cur.fetchone()
    base_id = row_base[0] if row_base else 1

    cur.execute("SELECT post_id FROM post WHERE content LIKE '%#target%' ORDER BY post_id ASC LIMIT 1")
    row_pay = cur.fetchone()
    pay_id = row_pay[0] if row_pay else (base_id + 1)

    if enable_bio_scraping and bot_agents:
        bio_mod = BioScrapingModifier()
        bio_mod.adapt_squad_personas(
            squad_bots=list(bot_agents.values()),
            candidate_organic_agents=organic_agents,
        )
        for bot in bot_agents.values():
            u = bot.user_info
            cur.execute(
                "UPDATE user SET user_name=?, name=?, bio=? WHERE user_id=?",
                (u.user_name, u.name, u.description, bot.social_agent_id),
            )
        conn.commit()
        logger.info("✓ BioScrapingModifier adapted bot squad to chameleon demographic personas.")

    conn.close()
    return base_id, pay_id


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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum output generation tokens per agent turn (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for LLM generation (default: 0.7).",
    )
    parser.add_argument(
        "--use-jev",
        action="store_true",
        help="Use Joint Evaluation Vectorization (JEV) engine for high-speed batched simulation.",
    )
    parser.add_argument(
        "--organic-post-rate",
        type=float,
        default=0.10,
        help="Probability that an active organic agent publishes a spontaneous root post per step (default: 0.10).",
    )
    parser.add_argument(
        "--bot-organic-post-rate",
        type=float,
        default=0.15,
        help="Probability that a CIB bot publishes an organic background post per step (default: 0.15).",
    )
    parser.add_argument(
        "--num-initial-posts",
        type=int,
        default=15,
        help="Number of initial organic background posts to populate the feed across communities at step 0 (default: 15). Set to 0 to disable.",
    )
    parser.add_argument(
        "--costart-steps",
        type=int,
        default=2,
        help="Number of shared initial burn-in steps where active organic agents post and interact before baseline/target injection (default: 2). Set to 0 to disable.",
    )
    parser.add_argument(
        "--costart-cache-dir",
        type=str,
        default=None,
        help="Directory to cache and load shared co-start SQLite checkpoints for multi-run efficiency (default: None).",
    )
    parser.add_argument(
        "--enable-bio-scraping",
        action="store_true",
        help="Enable Vector 7 Bio-Scraping Copyattack modifier for bots to adopt chameleon demographic personas.",
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
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
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

        # Synchronize user identities and initial social graph into SQLite
        all_sim_agents = organic_agents + list(bot_agents.values())
        sync_network_to_db(
            agents=all_sim_agents,
            db_path=str(db_path),
            community_map=community_map,
        )

        env = OasisEnv(
            agent_graph=agent_graph,
            platform=platform,
            database_path=str(db_path),
        )

        jev_env = None
        if args.use_jev:
            from oasis.environment.jev_env import (
                JEVEnvironment,
                JEVExecutionConfig,
            )

            if is_llm_mode:
                from oasis.inference.jev_classifier import VLLMJEVClassifierClient

                jev_client = VLLMJEVClassifierClient(
                    base_url=args.vllm_url,
                    model_name=args.model,
                    temperature=args.temperature,
                    classify_max_tokens=1,
                    comment_max_tokens=64,
                    comment_temperature=args.temperature,
                    auto_discover_token_ids=True,
                )
            else:
                from oasis.inference.jev_classifier import MockJEVClassifierClient

                jev_client = MockJEVClassifierClient()

            jev_config = JEVExecutionConfig(
                step_duration_seconds=300.0,
                max_actions_per_agent=1,
                seed=42,
                wait_for_platform=True,
                organic_post_rate=args.organic_post_rate,
                bot_organic_post_rate=args.bot_organic_post_rate,
                community_map=community_map,
            )
            jev_env = JEVEnvironment(
                env_or_graph=env,
                config=jev_config,
                classifier_client=jev_client,
            )

        # 4. Step 0: Initial feed population and co-start preparation
        initial_posts = []
        if args.num_initial_posts > 0 and jev_env is not None:
            initial_posts = await jev_env.seed_initial_posts(
                agents=organic_agents,
                num_posts=args.num_initial_posts,
            )
            logger.info(
                f"✓ Populated initial feed with {len(initial_posts)} organic background posts across communities."
            )

        # Refresh recommendation table so agents immediately have rich, populated feeds
        if hasattr(platform, "update_rec_table"):
            try:
                await platform.update_rec_table()
                logger.info("✓ Platform recommendation table refreshed with initial post inventory.")
            except Exception as e:
                logger.warning(f"Could not refresh recommendation table after seeding: {e}")

        # Checkpoint / Cache path for co-start
        costart_cache_file = None
        if args.costart_cache_dir and args.costart_steps > 0:
            cache_dir = Path(args.costart_cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            costart_cache_file = cache_dir / f"costart_{args.topology}_{args.num_organic}org_init{args.num_initial_posts}_cs{args.costart_steps}.db"

        start_step = 0
        total_steps = args.costart_steps + args.max_steps if args.costart_steps > 0 else args.max_steps
        baseline_post_id = 1
        payload_post_id = 2
        campaign = None

        if costart_cache_file and costart_cache_file.exists():
            shutil.copy(costart_cache_file, db_path)
            start_step = args.costart_steps
            logger.info(
                f"✓ Reusing cached co-start database from {costart_cache_file}. Fast-forwarding to step {start_step + 1}."
            )
        elif args.costart_steps == 0:
            # Legacy direct start: inject baseline and target at step 0
            baseline_post_id, payload_post_id = await execute_narrative_injection(
                env=env,
                platform=platform,
                organic_agents=organic_agents,
                bot_agents=bot_agents,
                bot_ids=bot_ids,
                db_path=db_path,
                topic_mode=args.topic_mode,
                preset=args.preset,
                enable_bio_scraping=args.enable_bio_scraping,
            )
            campaign = compile_campaign_preset(
                preset=args.preset,
                bot_ids=bot_ids,
                organic_agents=organic_agents,
                baseline_post_id=baseline_post_id,
                payload_post_id=payload_post_id,
                total_campaign_steps=args.max_steps,
            )

        campaign_name = campaign.campaign_name if campaign else f"Preset_{args.preset}"
        logger.info(f"Executing campaign '{campaign_name}' ({total_steps} total steps, {args.costart_steps} co-start)...")

        # 6. Main Simulation Loop
        exposure_timeline = []

        for step in range(start_step, total_steps):
            step_actions: dict = {}

            # Transition step: End of co-start -> Inject focal baseline & target narrative
            if args.costart_steps > 0 and step == args.costart_steps:
                baseline_post_id, payload_post_id = await execute_narrative_injection(
                    env=env,
                    platform=platform,
                    organic_agents=organic_agents,
                    bot_agents=bot_agents,
                    bot_ids=bot_ids,
                    db_path=db_path,
                    topic_mode=args.topic_mode,
                    preset=args.preset,
                    enable_bio_scraping=args.enable_bio_scraping,
                )
                campaign = compile_campaign_preset(
                    preset=args.preset,
                    bot_ids=bot_ids,
                    organic_agents=organic_agents,
                    baseline_post_id=baseline_post_id,
                    payload_post_id=payload_post_id,
                    total_campaign_steps=args.max_steps,
                )
                logger.info(
                    f"✓ Injected focal narratives at step {step + 1}: "
                    f"Baseline ID={baseline_post_id}, Payload ID={payload_post_id}. Campaign '{args.preset}' activated."
                )

            # Sample active organic agents
            num_active = max(1, int(len(organic_agents) * args.active_ratio))
            active_organic = random.sample(organic_agents, num_active)

            if not args.use_jev:
                for org_agent in active_organic:
                    if is_llm_mode:
                        step_actions[org_agent] = LLMAction()
                    else:
                        step_actions[org_agent] = [ManualAction(ActionType.DO_NOTHING, {})]

            # Generate CIB actions for active squads (only after co-start phase)
            if campaign is not None and step >= args.costart_steps:
                rel_campaign_step = step - args.costart_steps
                phase, rel_step = campaign.get_active_phase(rel_campaign_step)
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

            post_actions = []
            if args.use_jev and jev_env is not None:
                # 1. Execute CIB bot campaign actions if present
                if step_actions:
                    await env.step(step_actions)

                # 2. Track 1: Parallel Feed Interaction Pass for active organic agents
                active_org_ids = [a.social_agent_id for a in active_organic]
                jev_res = await jev_env.step_jev(
                    step_index=step,
                    active_agent_ids=active_org_ids,
                )
                logger.info(
                    f"[JEV] Step {step + 1} actions: L={jev_res.num_likes}, "
                    f"R={jev_res.num_reposts}, Q={jev_res.num_quotes}, "
                    f"C={jev_res.num_comments}, S={jev_res.num_skips} "
                    f"({jev_res.execution_time_seconds:.3f}s)"
                )

                # 3. Track 2: Separate Spontaneous Organic Posting Pass (Bots + Active Organic Agents)
                posting_candidates = list(active_organic) + list(bot_agents.values())
                post_actions = await jev_env.step_organic_posts(
                    step_index=step,
                    candidate_agents=posting_candidates,
                )
                if post_actions:
                    bot_post_count = sum(1 for a in post_actions if a.user_id in bot_agents)
                    org_post_count = len(post_actions) - bot_post_count
                    logger.info(
                        f"[JEV Organic Post Track] Step {step + 1}: Published {len(post_actions)} "
                        f"spontaneous root posts ({bot_post_count} bots, {org_post_count} organic)."
                    )
            else:
                await env.step(step_actions)

            # Checkpoint save at end of co-start burn-in
            if costart_cache_file and step == args.costart_steps - 1:
                shutil.copy(db_path, costart_cache_file)
                logger.info(f"✓ Cached shared co-start checkpoint database to {costart_cache_file}")

            # Record step exposure telemetry
            e_base_t = 0.0
            e_pay_t = 0.0
            if step >= args.costart_steps:
                e_base_t = calculate_exposure_from_db(str(db_path), baseline_post_id)
                e_pay_t = calculate_exposure_from_db(str(db_path), payload_post_id)

            dispatched_count = len(step_actions)
            if args.use_jev and jev_res is not None:
                dispatched_count += jev_res.num_actions + len(post_actions)

            phase_label = "co_start" if step < args.costart_steps else "strike"
            exposure_timeline.append(
                {
                    "step": step + 1,
                    "phase": phase_label,
                    "baseline_exposure": e_base_t,
                    "payload_exposure": e_pay_t,
                    "net_lift": e_pay_t - e_base_t,
                    "actions_dispatched": dispatched_count,
                }
            )

            logger.info(
                f"Step {step + 1}/{total_steps} [{phase_label}] | "
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
        cur.execute(
            "SELECT COUNT(*) FROM post WHERE quote_content IS NOT NULL AND quote_content != ''"
        )
        total_quotes = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM post WHERE original_post_id IS NOT NULL AND (quote_content IS NULL OR quote_content = '')"
        )
        total_reposts = cur.fetchone()[0]

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
            "use_jev": bool(args.use_jev),
            "model_name": args.model if is_llm_mode else "hermetic_noop",
            "db_path": str(db_path),
            "telemetry": {
                "total_posts": total_posts,
                "total_likes": total_likes,
                "total_comments": total_comments,
                "total_quotes": total_quotes,
                "total_reposts": total_reposts,
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
