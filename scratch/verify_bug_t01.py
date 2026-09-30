"""Empirical Verification Script for Bug T-01: Dual Clock Advancement.

Directly tests the clock advancement mechanics of OasisEnv.step() and
JEVEnvironment.step_jev() sharing the same Platform sandbox_clock under
both Twitter and Reddit platform modes.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Safe mock for heavy ML packages if running in lightweight CPU environment
for pkg in (
    "torch", "torch.nn", "sentence_transformers", "transformers",
    "sklearn", "sklearn.feature_extraction", "sklearn.feature_extraction.text",
    "sklearn.metrics", "sklearn.metrics.pairwise", "tqdm",
):
    try:
        __import__(pkg)
    except ImportError:
        sys.modules[pkg] = MagicMock()

from oasis.clock.clock import Clock
from oasis.environment import OasisEnv, JEVEnvironment, JEVExecutionConfig, ManualAction
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.database import create_db
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType, DefaultPlatformType
from cib_zoo.agent.cib_agent import CIBAgent


def make_user_info(agent_id: int, name: str) -> UserInfo:
    return UserInfo(
        user_name=f"user_{agent_id:03d}",
        name=name,
        description=f"User {agent_id}",
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


async def run_scenario(recsys_type: str):
    print(f"\n======================================================================")
    print(f"TESTING SCENARIO: recsys_type = '{recsys_type}'")
    print(f"======================================================================")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / f"test_clock_{recsys_type}.db"
        create_db(str(db_file))

        channel = Channel()
        platform = Platform(
            db_path=str(db_file),
            channel=channel,
            recsys_type=recsys_type,
            show_score=False,
            refresh_rec_post_count=5,
            max_rec_post_len=10,
        )
        platform_task = asyncio.create_task(platform.running())

        try:
            agent_graph = AgentGraph()

            # Create an organic agent
            organic_agent = CIBAgent(
                agent_id=1,
                user_info=make_user_info(1, "Organic User"),
                channel=channel,
                max_actions_per_step=1,
            )
            agent_graph.add_agent(organic_agent)

            # Create a bot agent
            bot_agent = CIBAgent(
                agent_id=2,
                user_info=make_user_info(2, "CIB Bot"),
                channel=channel,
                max_actions_per_step=1,
            )
            agent_graph.add_agent(bot_agent)

            # 2. Setup OasisEnv with the platform
            env = OasisEnv(
                agent_graph=agent_graph,
                platform=platform,
                database_path=str(db_file),
            )

            # 3. Setup JEVEnvironment wrapping env
            jev_config = JEVExecutionConfig(
                step_duration_seconds=300.0,
                max_actions_per_agent=1,
                wait_for_platform=True,
            )
            mock_classifier = AsyncMock()
            mock_classifier.classify_batch = AsyncMock(return_value=[])

            jev_env = JEVEnvironment(
                env_or_graph=env,
                config=jev_config,
                classifier_client=mock_classifier,
            )

            print(f"Platform type resolved in OasisEnv: {env.platform_type}")
            print(f"Shared clock verified: {env.platform.sandbox_clock is jev_env.platform.sandbox_clock}")

            feed_data = {
                1: [
                    {
                        "post_id": 101,
                        "author_name": "user_001",
                        "topic": "tech",
                        "content": "Sample post content for feed evaluation",
                        "stance": 0.5,
                        "num_likes": 0,
                    }
                ]
            }

            # Phase 1: Baseline step (step_actions = {})
            clock_before_baseline = env.platform.sandbox_clock.time_step
            step_actions = {}
            if step_actions:
                await env.step(step_actions)
            res_b = await jev_env.step_jev(step_index=0, active_agent_ids=[1], agent_feeds=feed_data)
            clock_after_baseline = env.platform.sandbox_clock.time_step
            delta_baseline = clock_after_baseline - clock_before_baseline
            print(f"Baseline step: before={clock_before_baseline}, after={clock_after_baseline}, delta=+{delta_baseline}")

            # Phase 2: Strike step (bot actions present)
            step_actions = {
                bot_agent: [ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": 101})]
            }
            clock_before_strike = env.platform.sandbox_clock.time_step
            if step_actions:
                await env.step(step_actions)
            res_s = await jev_env.step_jev(step_index=1, active_agent_ids=[1], agent_feeds=feed_data)
            clock_after_strike = env.platform.sandbox_clock.time_step
            delta_strike = clock_after_strike - clock_before_strike
            print(f"Strike step:   before={clock_before_strike}, after={clock_after_strike}, delta=+{delta_strike}")

            return {
                "recsys_type": recsys_type,
                "platform_type": str(env.platform_type),
                "delta_baseline": delta_baseline,
                "delta_strike": delta_strike,
                "has_dual_advancement": (delta_strike == 2 and delta_baseline == 1),
            }

        finally:
            platform_task.cancel()
            try:
                await platform_task
            except asyncio.CancelledError:
                pass


async def main():
    results = []
    # Test Twitter, Reddit, and Random
    for recsys in ["twitter", "random", "reddit"]:
        res = await run_scenario(recsys)
        results.append(res)

    print("\n" + "=" * 70)
    print("EMPIRICAL AUDIT SUMMARY FOR BUG T-01:")
    print("=" * 70)
    for r in results:
        status = "CONFIRMED BUG T-01 (Dual Clock Advancement)" if r["has_dual_advancement"] else "Single Clock Advancement (Reddit mode)"
        print(f"Recsys: {r['recsys_type']:<8} | PlatformType: {r['platform_type']:<28} | Baseline Delta: +{r['delta_baseline']} | Strike Delta: +{r['delta_strike']} | Result: {status}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
