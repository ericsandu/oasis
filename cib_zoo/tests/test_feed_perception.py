"""Regression and perception tests verifying user persistence and active feed delivery.

Ensures simulations never run with empty user tables, follow graphs are properly seeded,
and agents actively receive mock posts in their feeds during refresh.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3
import pytest

from cib_zoo.agent.cib_agent import CIBAgent, NoOpModelBackend
from cib_zoo.metrics.amplification import calculate_exposure_from_db
from cib_zoo.topology.network_builder import build_network, sync_network_to_db
from oasis.environment.env import OasisEnv
from oasis.environment.env_action import ManualAction
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.database import create_db
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType


def make_dummy_user_info(agent_id: int, name: str, comm: str = "tech") -> UserInfo:
    return UserInfo(
        user_name=f"usr_{agent_id:03d}",
        name=name,
        description=f"Bio description for {name} ({comm})",
        profile={
            "nodes": [],
            "edges": [],
            "other_info": {
                "user_profile": f"Profile of {name}",
                "mbti": "INTJ",
                "community": comm,
                "gender": "non-binary",
                "age": 28,
                "country": "US",
                "activity_level": ["active"] * 24,
                "activity_level_frequency": [1] * 24,
                "active_threshold": [0.5] * 24,
            },
        },
        recsys_type="twitter",
        is_controllable=False,
    )


def test_sync_network_populates_user_and_follow_tables(tmp_path: Path) -> None:
    """Verify sync_network_to_db populates user rows and intra-community follow edges."""
    db_file = tmp_path / "test_sync.db"
    create_db(str(db_file))

    # Build multitopic network with 15 users (5 tech, 5 politics, 5 sports)
    agent_graph, organic_agents, comm_map = build_network(
        topology="multitopic",
        num_users=15,
    )

    # Add 3 CIB bots
    bot_agents = [
        CIBAgent(
            agent_id=100 + i,
            user_info=make_dummy_user_info(100 + i, f"Bot {i}"),
            channel=Channel(),
            max_actions_per_step=5,
        )
        for i in range(3)
    ]
    all_agents = organic_agents + bot_agents

    sync_network_to_db(
        agents=all_agents,
        db_path=str(db_file),
        community_map=comm_map,
        min_follows_per_user=2,
        max_follows_per_user=3,
        seed=123,
    )

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Verify user table has exactly 18 entries
    cur.execute("SELECT COUNT(*) FROM user")
    total_users = cur.fetchone()[0]
    assert total_users == 18, f"Expected 18 users, got {total_users}"

    # Verify follow table has edges
    cur.execute("SELECT COUNT(*) FROM follow")
    total_follows = cur.fetchone()[0]
    assert total_follows > 0, "Follow table should have seeded edges"

    # Verify num_followers and num_followings are populated
    cur.execute("SELECT user_id, num_followings, num_followers FROM user WHERE user_id <= 15")
    rows = cur.fetchall()
    for uid, n_ing, n_er in rows:
        assert n_ing > 0, f"User {uid} should have non-zero followings"
        assert n_er >= 0

    conn.close()


def test_oasis_env_automatic_user_sync(tmp_path: Path) -> None:
    """Verify OasisEnv.__init__ automatically synchronizes all agents in agent_graph into SQLite."""
    db_file = tmp_path / "test_env_sync.db"
    create_db(str(db_file))

    channel = Channel()
    platform = Platform(
        db_path=str(db_file),
        channel=channel,
        recsys_type="reddit",
        show_score=False,
    )

    agent_graph = AgentGraph()
    for i in range(1, 6):
        agent = SocialAgent(
            agent_id=i,
            user_info=make_dummy_user_info(i, f"Organic {i}"),
            channel=channel,
            model=NoOpModelBackend(),
        )
        agent_graph.add_agent(agent)

    # Bot agent
    bot = CIBAgent(
        agent_id=99,
        user_info=make_dummy_user_info(99, "Bot 99"),
        channel=channel,
        model=NoOpModelBackend(),
    )
    agent_graph.add_agent(bot)

    # Initialize OasisEnv - should automatically trigger sync_agents_to_db
    _ = OasisEnv(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(db_file),
    )

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT user_id, user_name, bio FROM user ORDER BY user_id ASC")
    users = cur.fetchall()
    conn.close()

    assert len(users) == 6, f"Expected 6 users in DB, got {len(users)}"
    user_ids = [u[0] for u in users]
    assert user_ids == [1, 2, 3, 4, 5, 99]


@pytest.mark.asyncio
async def test_mock_posts_feed_receipt_and_refresh(tmp_path: Path) -> None:
    """Verify that seeded mock posts appear in agent feeds and refresh returns active content."""
    db_file = tmp_path / "test_feed_delivery.db"
    create_db(str(db_file))

    channel = Channel()
    platform = Platform(
        db_path=str(db_file),
        channel=channel,
        recsys_type="twitter",
        show_score=False,
        refresh_rec_post_count=5,
        max_rec_post_len=10,
    )
    platform_task = asyncio.create_task(platform.running())

    try:
        # Build 10-user network and sync to database
        agent_graph, organic_agents, comm_map = build_network(
            topology="multitopic",
            num_users=10,
            model=NoOpModelBackend(),
            channel=channel,
        )

        sync_network_to_db(
            agents=organic_agents,
            db_path=str(db_path:=str(db_file)),
            community_map=comm_map,
            min_follows_per_user=2,
            max_follows_per_user=4,
            seed=42,
        )

        env = OasisEnv(
            agent_graph=agent_graph,
            platform=platform,
            database_path=str(db_file),
        )

        # Step 0: User 1 creates baseline post, User 2 creates payload post
        step0_actions = {
            organic_agents[0]: [
                ManualAction(
                    ActionType.CREATE_POST,
                    {"content": "Tech discussion on distributed systems #tech"},
                )
            ],
            organic_agents[1]: [
                ManualAction(
                    ActionType.CREATE_POST,
                    {"content": "Controversial payload narrative #target"},
                )
            ],
        }
        await env.step(step0_actions)

        # Verify posts exist in DB
        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()
        cur.execute("SELECT post_id, content FROM post")
        posts = cur.fetchall()
        conn.close()
        assert len(posts) == 2

        # Update recommendation system cache
        await platform.update_rec_table()

        # Call refresh for a user who follows User 1 or User 2, or via recs
        recipient = organic_agents[2]
        refresh_result = await recipient.env.action.refresh()
        assert refresh_result["success"] is True, f"Refresh failed: {refresh_result}"
        assert len(refresh_result["posts"]) > 0, "Recipient should have received posts in feed"

        # Check prompt string generated by agent environment (calls refresh internally)
        prompt_str = await recipient.env.get_posts_env()
        assert "After refreshing, there are no existing posts." not in prompt_str
        assert "After refreshing, you see some posts" in prompt_str

        # Verify exposure calculation uses exact matching without collisions
        exp1 = calculate_exposure_from_db(str(db_file), post_id=1)
        exp2 = calculate_exposure_from_db(str(db_file), post_id=2)
        assert exp1 >= 1.0
        assert exp2 >= 1.0

    finally:
        platform_task.cancel()
        try:
            await platform_task
        except asyncio.CancelledError:
            pass
