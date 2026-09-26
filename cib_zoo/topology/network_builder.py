"""Network builder for structured multi-topic and ideologically polarized agent graphs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cib_zoo.agent.cib_agent import NoOpModelBackend
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.typing import ActionType

logger = logging.getLogger("cib_zoo.topology")

# Curated persona templates for multi-topic community generation
TECH_TEMPLATES = [
    (
        "Alice Chen",
        "alice_tech",
        "Senior ML engineer & open-source contributor. Obsessed with distributed training and compilers.",
        "INTJ",
        "US",
    ),
    (
        "Marcus Bell",
        "mbell_dev",
        "Rust & Linux systems programmer. Building high-throughput vector storage engines.",
        "ISTP",
        "UK",
    ),
    (
        "Elena Rostova",
        "elena_quantum",
        "Quantum computing researcher & physicist exploring superconducting qubit coherence.",
        "INTP",
        "DE",
    ),
    (
        "David Kim",
        "dkim_ai",
        "Autonomous agent researcher working on reasoning, planning, and tool use.",
        "ENTJ",
        "CA",
    ),
    (
        "Sophia Patel",
        "sophia_cyber",
        "Cybersecurity researcher focused on formal verification and zero-knowledge systems.",
        "INFJ",
        "US",
    ),
]

POLITICS_TEMPLATES = [
    (
        "Arthur Vance",
        "avance_policy",
        "Senior policy analyst covering federal budget oversight and regulatory reform.",
        "ISTJ",
        "US",
    ),
    (
        "Chloe Martin",
        "cmartin_gov",
        "Municipal governance advocate tracking urban zoning and public transit spending.",
        "ENFJ",
        "US",
    ),
    (
        "Julian Morales",
        "jmorales_econ",
        "Macroeconomist & trade researcher monitoring central bank interest rate policies.",
        "ENTP",
        "CA",
    ),
    (
        "Sarah Jenkins",
        "sjenkins_news",
        "Investigative reporter following congressional committee hearings and integrity laws.",
        "ISFJ",
        "UK",
    ),
    (
        "Tariq Al-Mansoor",
        "tariq_diplomacy",
        "International affairs scholar writing on multilateral trade and diplomatic security.",
        "INFP",
        "CH",
    ),
]

SPORTS_TEMPLATES = [
    (
        "Liam O'Connor",
        "loconnor_sports",
        "Premier league football analyst & tactics enthusiast. High press or bust.",
        "ESTP",
        "UK",
    ),
    (
        "Maya Tanaka",
        "maya_run",
        "Marathon runner, coach, and physiological endurance researcher.",
        "ESFP",
        "JP",
    ),
    (
        "Carlos Mendez",
        "carlos_hoops",
        "Basketball analytics coach tracking spacing, shot charts, and pick-and-roll efficiency.",
        "ESFJ",
        "US",
    ),
    (
        "Greta Lindholm",
        "greta_alpine",
        "Skiing and alpine winter sports enthusiast documenting cross-country trails.",
        "ISFP",
        "SE",
    ),
    (
        "Samir Patel",
        "samir_racing",
        "Motorsport aerodynamicist and race strategy analyst following Formula 1.",
        "ESTJ",
        "UK",
    ),
]


def build_multitopic_network(
    num_users: int = 150,
    model: Any = None,
    channel: Channel | None = None,
) -> tuple[AgentGraph, list[SocialAgent], dict[int, str]]:
    """Build a multi-community agent network partitioned across Tech, Politics, and Sports.

    Args:
        num_users: Total number of organic agents.
        model: Optional LLM model backend. Defaults to NoOpModelBackend if None.
        channel: Optional Channel instance.

    Returns:
        tuple of (AgentGraph, list[SocialAgent], dict[agent_id -> community_name])
    """
    effective_channel = channel or Channel()
    effective_model = model if model is not None else NoOpModelBackend()

    agent_graph = AgentGraph()
    agents: list[SocialAgent] = []
    community_map: dict[int, str] = {}

    n_tech = num_users // 3
    n_pol = num_users // 3
    n_sports = num_users - (n_tech + n_pol)

    communities = (
        [("tech", TECH_TEMPLATES)] * n_tech
        + [("politics", POLITICS_TEMPLATES)] * n_pol
        + [("sports", SPORTS_TEMPLATES)] * n_sports
    )

    for idx, (comm_name, templates) in enumerate(communities, start=1):
        tpl = templates[(idx - 1) % len(templates)]
        name, user_prefix, desc, mbti, country = tpl
        username = f"{user_prefix}_{idx:03d}"

        user_info = UserInfo(
            user_name=username,
            name=f"{name} ({idx})",
            description=desc,
            profile={
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": f"{name} is a Twitter user interested in {comm_name}. {desc}",
                    "mbti": mbti,
                    "gender": "non-binary",
                    "age": 25 + (idx % 35),
                    "country": country,
                    "community": comm_name,
                    "activity_level": ["active"] * 24,
                    "activity_level_frequency": [1] * 24,
                    "active_threshold": [0.5] * 24,
                },
            },
            recsys_type="twitter",
            is_controllable=False,
        )

        agent = SocialAgent(
            agent_id=idx,
            user_info=user_info,
            channel=effective_channel,
            model=effective_model,
            available_actions=ActionType.get_default_twitter_actions(),
        )

        agent_graph.add_agent(agent)
        agents.append(agent)
        community_map[idx] = comm_name

    logger.info(
        f"Built Multi-Topic Network with {len(agents)} agents: "
        f"Tech={n_tech}, Politics={n_pol}, Sports={n_sports}"
    )
    return agent_graph, agents, community_map


def build_polarized_network(
    num_users: int = 150,
    model: Any = None,
    channel: Channel | None = None,
    data_dir: str | None = None,
) -> tuple[AgentGraph, list[SocialAgent], dict[int, str]]:
    """Build a bi-partisan ideologically polarized network (Progressive vs. Conservative).

    Reads real user personas and follower networks from Twitter polarization datasets.

    Args:
        num_users: Total number of organic agents.
        model: Optional LLM model backend.
        channel: Optional Channel instance.
        data_dir: Directory containing 197_progressive.csv and 197_baoshou.csv.

    Returns:
        tuple of (AgentGraph, list[SocialAgent], dict[agent_id -> community_name])
    """
    effective_channel = channel or Channel()
    effective_model = model if model is not None else NoOpModelBackend()

    if data_dir is None:
        oasis_root = Path(__file__).resolve().parents[2]
        data_dir = str(oasis_root / "data" / "twitter_dataset" / "group_polarization")

    prog_path = Path(data_dir) / "197_progressive.csv"
    cons_path = Path(data_dir) / "197_baoshou.csv"

    if not prog_path.exists() or not cons_path.exists():
        raise FileNotFoundError(
            f"Polarization datasets missing at {data_dir}. Expected 197_progressive.csv and 197_baoshou.csv"
        )

    df_prog = pd.read_csv(prog_path)
    df_cons = pd.read_csv(cons_path)

    half = num_users // 2
    n_prog = half
    n_cons = num_users - half

    df_prog_sample = df_prog.iloc[:n_prog]
    df_cons_sample = df_cons.iloc[:n_cons]

    agent_graph = AgentGraph()
    agents: list[SocialAgent] = []
    community_map: dict[int, str] = {}

    current_id = 1

    # Progressive Agents
    for _, row in df_prog_sample.iterrows():
        username = str(row.get("username", f"prog_{current_id}"))
        name = str(row.get("name", f"Progressive User {current_id}"))
        description = str(
            row.get("description", "Progressive social media participant.")
        )
        user_char = str(row.get("user_char", description))

        user_info = UserInfo(
            user_name=f"{username}_{current_id:03d}",
            name=name,
            description=description,
            profile={
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": user_char,
                    "community": "progressive",
                    "ideology": "progressive",
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

        agent = SocialAgent(
            agent_id=current_id,
            user_info=user_info,
            channel=effective_channel,
            model=effective_model,
            available_actions=ActionType.get_default_twitter_actions(),
        )
        agent_graph.add_agent(agent)
        agents.append(agent)
        community_map[current_id] = "progressive"
        current_id += 1

    # Conservative Agents
    for _, row in df_cons_sample.iterrows():
        username = str(row.get("username", f"cons_{current_id}"))
        name = str(row.get("name", f"Conservative User {current_id}"))
        description = str(
            row.get("description", "Conservative social media participant.")
        )
        user_char = str(row.get("user_char", description))

        user_info = UserInfo(
            user_name=f"{username}_{current_id:03d}",
            name=name,
            description=description,
            profile={
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": user_char,
                    "community": "conservative",
                    "ideology": "conservative",
                    "gender": "non-binary",
                    "age": 34,
                    "country": "US",
                    "activity_level": ["active"] * 24,
                    "activity_level_frequency": [1] * 24,
                    "active_threshold": [0.5] * 24,
                },
            },
            recsys_type="twitter",
            is_controllable=False,
        )

        agent = SocialAgent(
            agent_id=current_id,
            user_info=user_info,
            channel=effective_channel,
            model=effective_model,
            available_actions=ActionType.get_default_twitter_actions(),
        )
        agent_graph.add_agent(agent)
        agents.append(agent)
        community_map[current_id] = "conservative"
        current_id += 1

    logger.info(
        f"Built Polarized Network with {len(agents)} agents: "
        f"Progressive={n_prog}, Conservative={n_cons}"
    )
    return agent_graph, agents, community_map


def build_network(
    topology: str = "multitopic",
    num_users: int = 150,
    model: Any = None,
    channel: Channel | None = None,
) -> tuple[AgentGraph, list[SocialAgent], dict[int, str]]:
    """Factory router to construct requested network topology.

    Args:
        topology: 'multitopic' or 'polarized'
        num_users: Number of organic users.
        model: Optional model backend.
        channel: Optional Channel instance.

    Returns:
        tuple of (AgentGraph, list[SocialAgent], dict[agent_id -> community_name])
    """
    if topology == "polarized":
        return build_polarized_network(
            num_users=num_users, model=model, channel=channel
        )
    elif topology == "multitopic":
        return build_multitopic_network(
            num_users=num_users, model=model, channel=channel
        )
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. Expected 'multitopic' or 'polarized'."
        )


def sync_network_to_db(
    agents: list[SocialAgent],
    db_path: str,
    community_map: dict[int, str] | None = None,
    min_follows_per_user: int = 5,
    max_follows_per_user: int = 8,
    seed: int = 42,
) -> None:
    """Persist all agents into SQLite user table and seed intra-community homophily follow edges.

    Args:
        agents: List of all SocialAgent / CIBAgent instances participating in the simulation.
        db_path: Path to target SQLite database.
        community_map: Optional dict mapping agent_id -> community_name.
        min_follows_per_user: Minimum initial followings to seed per user.
        max_follows_per_user: Maximum initial followings to seed per user.
        seed: Random seed for reproducible network graph generation.
    """
    import random
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Populate user table
    user_rows = []
    for agent in agents:
        if hasattr(agent, "to_db_user_row"):
            user_rows.append(agent.to_db_user_row())
        else:
            info = agent.user_info
            aid = getattr(agent, "social_agent_id", getattr(agent, "agent_id", 0))
            user_rows.append((
                aid,
                aid,
                getattr(info, "user_name", f"user_{aid}"),
                getattr(info, "name", f"User {aid}"),
                getattr(info, "description", None) or f"Profile for {aid}",
                "2026-09-26 00:00:00",
                0,
                0,
            ))

    cur.executemany(
        "INSERT OR REPLACE INTO user (user_id, agent_id, user_name, name, bio, created_at, num_followings, num_followers) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        user_rows,
    )

    # 2. Seed intra-community follow edges (Homophily)
    if community_map:
        rng = random.Random(seed)
        follow_rows = []
        communities = set(community_map.values())
        for comm in communities:
            comm_members = [uid for uid, c in community_map.items() if c == comm]
            for uid in comm_members:
                k = min(
                    len(comm_members) - 1,
                    rng.randint(min_follows_per_user, max_follows_per_user),
                )
                peers = [p for p in comm_members if p != uid]
                if peers and k > 0:
                    for followee in rng.sample(peers, k):
                        follow_rows.append((uid, followee, "2026-09-26 00:00:00"))

        if follow_rows:
            cur.executemany(
                "INSERT OR IGNORE INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)",
                follow_rows,
            )

            # 3. Synchronize follower/following count caches in user table
            cur.execute("""
                UPDATE user SET 
                    num_followings = (SELECT COUNT(*) FROM follow WHERE follower_id = user.user_id),
                    num_followers = (SELECT COUNT(*) FROM follow WHERE followee_id = user.user_id)
            """)

    conn.commit()
    conn.close()

