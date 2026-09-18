"""End-to-end hermetic micro-simulation test with real OasisEnv.step() and SQLite logging."""

import asyncio
from pathlib import Path
import sqlite3
import pytest

from oasis.environment.env import OasisEnv
from oasis.environment.env_action import ManualAction
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.database import create_db
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

from cib_zoo.agent.cib_agent import CIBAgent
from cib_zoo.metrics.amplification import calculate_causal_amplification
from cib_zoo.patterns.astroturf import AstroturfPattern
from cib_zoo.patterns.co_engagement import CoEngagementPattern


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


@pytest.mark.asyncio
async def test_hermetic_5_step_micro_simulation(tmp_path: Path) -> None:
    """Run a 5-step hermetic OASIS simulation and verify SQLite persistence and A(s,r) metric."""
    db_file = tmp_path / "test_micro_sim.db"
    create_db(str(db_file))

    channel = Channel()
    platform = Platform(
        db_path=str(db_file),
        channel=channel,
        recsys_type="reddit",
        show_score=False,
        refresh_rec_post_count=5,
        max_rec_post_len=10,
    )
    platform_task = asyncio.create_task(platform.running())

    try:
        agent_graph = AgentGraph()

        # Agent 1: Organic creator
        organic_agent = CIBAgent(
            agent_id=1,
            user_info=make_user_info(1, "Organic Creator"),
            channel=channel,
            max_actions_per_step=5,
        )
        agent_graph.add_agent(organic_agent)

        # Agent 2 & 3: CIB Attack Squad
        cib_bot_2 = CIBAgent(
            agent_id=2,
            user_info=make_user_info(2, "CIB Bot 2"),
            channel=channel,
            max_actions_per_step=5,
        )
        cib_bot_3 = CIBAgent(
            agent_id=3,
            user_info=make_user_info(3, "CIB Bot 3"),
            channel=channel,
            max_actions_per_step=5,
        )
        agent_graph.add_agent(cib_bot_2)
        agent_graph.add_agent(cib_bot_3)

        env = OasisEnv(
            agent_graph=agent_graph,
            platform=platform,
            database_path=str(db_file),
        )

        # Step 0: Organic agent posts baseline and payload
        step0_actions = {
            organic_agent: [
                ManualAction(action_type=ActionType.CREATE_POST, action_args={"content": "Baseline organic post #science"}),
                ManualAction(action_type=ActionType.CREATE_POST, action_args={"content": "Target payload post #cib_campaign"}),
            ]
        }
        await env.step(step0_actions)

        # Retrieve generated post IDs from database
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT post_id, content FROM post ORDER BY post_id ASC")
        posts = cursor.fetchall()
        assert len(posts) >= 2
        baseline_post_id = posts[0][0]
        payload_post_id = posts[1][0]
        conn.close()

        # Step 1: Co-engagement squad likes payload post
        co_pattern = CoEngagementPattern(anchor_post_id=baseline_post_id, payload_post_id=payload_post_id)
        step1_actions_raw = co_pattern.generate_step_actions(step=0, squad_bots=[cib_bot_2, cib_bot_3], context={})
        step1_actions = {
            cib_bot_2: step1_actions_raw[2],
            cib_bot_3: step1_actions_raw[3],
        }
        await env.step(step1_actions)

        # Step 2: Astroturf burst
        astro_pattern = AstroturfPattern(hashtag="cib_campaign", payload_template="Amplifying payload post details")
        step2_actions_raw = astro_pattern.generate_step_actions(step=0, squad_bots=[cib_bot_2, cib_bot_3], context={})
        step2_actions = {
            cib_bot_2: step2_actions_raw[2],
            cib_bot_3: step2_actions_raw[3],
        }
        await env.step(step2_actions)

        # Step 3: Bot 2 comments on payload
        step3_actions = {
            cib_bot_2: [ManualAction(action_type=ActionType.CREATE_COMMENT, action_args={"post_id": payload_post_id, "content": "Critical discussion comment"})]
        }
        await env.step(step3_actions)

        # Step 4: Organic creator does nothing
        step4_actions = {
            organic_agent: [ManualAction(action_type=ActionType.DO_NOTHING, action_args={})]
        }
        await env.step(step4_actions)

        # Verification: Validate database records
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM post")
        post_count = cursor.fetchone()[0]
        assert post_count >= 4  # 2 organic + 2 astroturf

        cursor.execute("SELECT COUNT(*) FROM like")
        like_count = cursor.fetchone()[0]
        assert like_count >= 4  # 2 bots * 2 posts

        cursor.execute("SELECT COUNT(*) FROM comment")
        comment_count = cursor.fetchone()[0]
        assert comment_count >= 1

        conn.close()

        # Verification: Calculate causal amplification metric A(s, r)
        amplification = calculate_causal_amplification(
            db_path=str(db_file),
            payload_post_id=payload_post_id,
            baseline_post_id=baseline_post_id,
            n_bots=2,
            n_seed=1,
        )
        assert amplification > 0.0
        # Payload had both likes and comments, baseline had only likes, so payload exposure > baseline
        assert amplification >= 0.5

    finally:
        platform_task.cancel()
        try:
            await platform_task
        except asyncio.CancelledError:
            pass
        if hasattr(platform, "db") and platform.db:
            platform.db.close()
