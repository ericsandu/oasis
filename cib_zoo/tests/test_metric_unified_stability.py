"""Tests for unified causal differential amplification metric and stability under edge-case distributions."""

import sqlite3
import pytest
from cib_zoo.agent.cib_agent import CIBAgent
from cib_zoo.metrics import (
    calculate_causal_amplification,
    calculate_community_partitioned_exposure,
    calculate_differential_amplification,
    calculate_dual_bubble_amplification,
    calculate_exposure_from_db,
    resolve_posts_by_narrative,
)


def _init_test_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            content TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE like (
            like_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            post_id INTEGER,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE comment (
            comment_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            post_id INTEGER,
            content TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE rec (
            rec_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            post_id INTEGER,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE dislike (
            dislike_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            post_id INTEGER,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE report (
            report_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            post_id INTEGER,
            reason TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE trace (
            trace_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            action TEXT,
            info TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def test_differential_amplification_existing_topic(tmp_path):
    """Test unified differential metric on an established topic with substantial organic baseline."""
    db_file = tmp_path / "existing_topic.db"
    _init_test_db(str(db_file))

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Baseline post 100: 50 recs, 20 likes, 10 comments
    cur.execute("INSERT INTO post VALUES (100, 1, 'Organic baseline topic', '0')")
    for u in range(50):
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 100, '0')", (u,))
    for u in range(20):
        cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (?, 100, '0')", (u,))
    for u in range(10):
        cur.execute("INSERT INTO comment (user_id, post_id, content, created_at) VALUES (?, 100, 'Great', '0')", (u,))

    # Payload post 101: 80 recs, 40 likes, 25 comments
    cur.execute("INSERT INTO post VALUES (101, 2, 'CIB amplified topic', '0')")
    for u in range(80):
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 101, '0')", (u,))
    for u in range(40):
        cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (?, 101, '0')", (u,))
    for u in range(25):
        cur.execute("INSERT INTO comment (user_id, post_id, content, created_at) VALUES (?, 101, 'Agreed', '0')", (u,))

    conn.commit()
    conn.close()

    # Exposure(Baseline) = 1.0 + (50*2) + (20*1.5) + (10*2) = 1 + 100 + 30 + 20 = 151.0
    # Exposure(Payload)  = 1.0 + (80*2) + (40*1.5) + (25*2) = 1 + 160 + 60 + 50 = 271.0
    # Delta = 271.0 - 151.0 = 120.0
    # With 10 bots, causal lift should be exactly 120.0 / 10 = 12.0 net impressions per bot
    lift = calculate_differential_amplification(
        db_path=str(db_file),
        payload_post_id=101,
        baseline_post_id=100,
        n_bots=10,
    )
    assert pytest.approx(lift, 0.01) == 12.0


def test_differential_amplification_cold_start_new_topic(tmp_path):
    """Test unified differential metric on cold-start topic starting at zero baseline."""
    db_file = tmp_path / "cold_start.db"
    _init_test_db(str(db_file))

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Baseline post 200 has 0 likes, 0 recs, 0 comments (just the seed)
    cur.execute("INSERT INTO post VALUES (200, 1, 'Brand new hashtag', '0')")

    # Payload post 201 has 30 recs and 10 likes generated through astroturf
    cur.execute("INSERT INTO post VALUES (201, 2, 'Brand new hashtag amplified', '0')")
    for u in range(30):
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 201, '0')", (u,))
    for u in range(10):
        cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (?, 201, '0')", (u,))

    conn.commit()
    conn.close()

    # Exposure(Baseline) = 1.0
    # Exposure(Payload)  = 1.0 + (30*2) + (10*1.5) = 76.0
    # Delta = 75.0, with 5 bots => 15.0 net exposure per bot
    lift = calculate_differential_amplification(
        db_path=str(db_file),
        payload_post_id=201,
        baseline_post_id=200,
        n_bots=5,
    )
    assert pytest.approx(lift, 0.01) == 15.0


def test_differential_amplification_backfire_chilling_effect(tmp_path):
    """Test negative causal lift when obnoxious bot swarm causes organic backlash/dislikes."""
    db_file = tmp_path / "backfire.db"
    _init_test_db(str(db_file))

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Baseline post 300: 40 recs, 20 likes
    cur.execute("INSERT INTO post VALUES (300, 1, 'Organic debate', '0')")
    for u in range(40):
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 300, '0')", (u,))
    for u in range(20):
        cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (?, 300, '0')", (u,))

    # Payload post 301: 20 recs, 5 likes, but 50 dislikes and 10 reports due to spam
    cur.execute("INSERT INTO post VALUES (301, 2, 'Obnoxious spam raid', '0')")
    for u in range(20):
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 301, '0')", (u,))
    for u in range(5):
        cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (?, 301, '0')", (u,))
    for u in range(50):
        cur.execute("INSERT INTO dislike (user_id, post_id, created_at) VALUES (?, 301, '0')", (u,))
    for u in range(10):
        cur.execute("INSERT INTO report (user_id, post_id, reason, created_at) VALUES (?, 301, 'spam', '0')", (u,))

    conn.commit()
    conn.close()

    # Exposure(Baseline) = 1.0 + 80 + 30 = 111.0
    # Exposure(Payload) = 1.0 + 40 + 7.5 - (50*1.5) - (10*2.0) = 48.5 - 75 - 20 = -46.5
    # Delta = -46.5 - 111.0 = -157.5
    # Lift should be negative!
    lift = calculate_differential_amplification(
        db_path=str(db_file),
        payload_post_id=301,
        baseline_post_id=300,
        n_bots=10,
    )
    assert lift < 0.0
    assert pytest.approx(lift, 0.01) == -15.75


def test_metric_assertions(tmp_path):
    """Test safety asserts on bot counts and activity thresholds."""
    db_file = tmp_path / "asserts.db"
    _init_test_db(str(db_file))

    with pytest.raises(AssertionError, match="n_bots must be positive"):
        calculate_differential_amplification(
            db_path=str(db_file),
            payload_post_id=1,
            baseline_post_id=2,
            n_bots=0,
        )

    # Test minimum bot actions enforcement in trace table
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("INSERT INTO post VALUES (1, 1, 'Post 1', '0')")
    cur.execute("INSERT INTO post VALUES (2, 2, 'Post 2', '0')")
    # Only 2 actions in trace
    cur.execute("INSERT INTO trace VALUES (1, 10, 'like', 'post:1', '0')")
    cur.execute("INSERT INTO trace VALUES (2, 11, 'like', 'post:1', '0')")
    conn.commit()
    conn.close()

    with pytest.raises(AssertionError, match="Expected at least 10 bot actions"):
        calculate_differential_amplification(
            db_path=str(db_file),
            payload_post_id=1,
            baseline_post_id=2,
            n_bots=5,
            min_expected_bot_actions=10,
        )

    # Test filtering by bot_ids explicitly
    with pytest.raises(AssertionError, match="Expected at least 5 bot actions"):
        calculate_differential_amplification(
            db_path=str(db_file),
            payload_post_id=1,
            baseline_post_id=2,
            n_bots=5,
            min_expected_bot_actions=5,
            bot_ids=[10, 11],
        )

    # Test dynamic CIBAgent registry resolution without arbitrary magic-number caps
    CIBAgent.reset_registry()
    assert CIBAgent.get_instance_count() == 0
    assert CIBAgent.get_bot_ids() == []

    CIBAgent._registry.add(50)
    CIBAgent._registry.add(51)
    assert CIBAgent.get_instance_count() == 2
    assert CIBAgent.get_bot_ids() == [50, 51]
    assert CIBAgent.is_bot(50) is True
    assert CIBAgent.is_bot(10) is False

    # The existing trace actions were performed by user_id 10 and 11, not bots (50, 51)
    # So dynamic bot action count is 0, raising error when min_expected_bot_actions >= 1
    with pytest.raises(AssertionError, match="Expected at least 1 bot actions"):
        calculate_differential_amplification(
            db_path=str(db_file),
            payload_post_id=1,
            baseline_post_id=2,
            min_expected_bot_actions=1,
        )

    # Add bot action with registered bot ID 50 and verify dynamic resolution recognizes it
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("INSERT INTO trace VALUES (3, 50, 'like', 'post:1', '0')")
    conn.commit()
    conn.close()

    res = calculate_differential_amplification(
        db_path=str(db_file),
        payload_post_id=1,
        baseline_post_id=2,
        min_expected_bot_actions=1,
    )
    assert isinstance(res, float)

    # Clean up registry
    CIBAgent.reset_registry()
    assert CIBAgent.get_instance_count() == 0


def test_exact_post_trace_matching_no_substring_collision(tmp_path):
    """Verify that calculate_exposure_from_db uses exact post_id matching without substring collisions."""
    import json
    from cib_zoo.metrics.amplification import calculate_exposure_from_db, is_post_trace_match

    # 1. Test helper matching directly
    assert is_post_trace_match(json.dumps({"post_id": 1, "action": "like"}), 1) is True
    assert is_post_trace_match(json.dumps({"post_id": 10, "action": "like"}), 1) is False
    assert is_post_trace_match(json.dumps({"user_id": 101, "content": "15 points scored in F1"}), 1) is False
    assert is_post_trace_match("post:1", 1) is True
    assert is_post_trace_match("post:10", 1) is False
    assert is_post_trace_match(json.dumps({"post_id": 2}), 2) is True
    assert is_post_trace_match(json.dumps({"post_id": 20}), 2) is False

    # 2. Test in database context
    db_file = tmp_path / "test_collision.db"
    _init_test_db(str(db_file))

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("INSERT INTO post VALUES (1, 1, 'Target Post 1', '0')")
    cur.execute("INSERT INTO post VALUES (2, 2, 'Target Post 2', '0')")

    # Injects traces that could collide under unconstrained LIKE '%1%'
    # User 105 posts about 15 minutes, with post_id 10
    cur.execute("INSERT INTO trace VALUES (1, 105, 'create_post', ?, '0')", (json.dumps({"content": "15 minutes in F1", "post_id": 10}),))
    # User 102 likes post 12
    cur.execute("INSERT INTO trace VALUES (2, 102, 'like_post', ?, '0')", (json.dumps({"post_id": 12}),))
    # User 103 likes post 1
    cur.execute("INSERT INTO trace VALUES (3, 103, 'like_post', ?, '0')", (json.dumps({"post_id": 1}),))
    # User 104 likes post 2
    cur.execute("INSERT INTO trace VALUES (4, 104, 'like_post', ?, '0')", (json.dumps({"post_id": 2}),))
    conn.commit()
    conn.close()

    # Post 1 has base 1.0 + 1 trace match * 0.5 = 1.5
    exp1 = calculate_exposure_from_db(str(db_file), post_id=1)
    # Post 2 has base 1.0 + 1 trace match * 0.5 = 1.5
    exp2 = calculate_exposure_from_db(str(db_file), post_id=2)

    assert exp1 == 1.5, f"Expected 1.5, got {exp1} (spurious substring match detected!)"
    assert exp2 == 1.5, f"Expected 1.5, got {exp2} (spurious substring match detected!)"
    assert exp1 == exp2


def test_resolve_posts_by_narrative(tmp_path):
    """Verify narrative resolution for single IDs, collections, and hashtag search patterns."""
    db_file = tmp_path / "test_resolve.db"
    _init_test_db(str(db_file))

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("INSERT INTO post VALUES (10, 1, 'Distributed cloud #baseline', '0')")
    cur.execute("INSERT INTO post VALUES (20, 2, 'Neuromorphic spike computing #target #neuromorphic', '0')")
    cur.execute("INSERT INTO post VALUES (21, 2, 'Analog memristor chips #target #neuromorphic', '0')")
    cur.execute("INSERT INTO post VALUES (30, 3, 'Unrelated sports news', '0')")
    conn.commit()
    conn.close()

    # Direct integer and set resolution
    assert resolve_posts_by_narrative(str(db_file), 10) == [10]
    assert resolve_posts_by_narrative(str(db_file), [20, 21]) == [20, 21]

    # Tag search pattern resolution
    assert resolve_posts_by_narrative(str(db_file), "%#baseline%") == [10]
    assert resolve_posts_by_narrative(str(db_file), "%#target%") == [20, 21]
    assert resolve_posts_by_narrative(str(db_file), "#neuromorphic") == [20, 21]
    assert resolve_posts_by_narrative(str(db_file), "%#nonexistent%") == []


def test_dual_bubble_amplification_metrics(tmp_path):
    """Verify in-bubble vs out-of-bubble partitioned exposure and causal amplification."""
    db_file = tmp_path / "test_dual_bubble.db"
    _init_test_db(str(db_file))

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Baseline post 100: Tech community discussion
    cur.execute("INSERT INTO post VALUES (100, 1, 'Mainstream cloud post #baseline', '0')")
    # In-group tech users (1..5) engage with baseline post: 5 recs, 2 likes
    for u in [1, 2, 3, 4, 5]:
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 100, '0')", (u,))
    for u in [1, 2]:
        cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (?, 100, '0')", (u,))

    # Payload post 200: Neuromorphic tech post
    cur.execute("INSERT INTO post VALUES (200, 2, 'Neuromorphic computing #target #neuromorphic', '0')")
    # In-group tech users (1..5) engage heavily: 5 recs, 5 likes, 3 comments
    for u in [1, 2, 3, 4, 5]:
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 200, '0')", (u,))
        cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (?, 200, '0')", (u,))
    for u in [1, 2, 3]:
        cur.execute("INSERT INTO comment (user_id, post_id, content, created_at) VALUES (?, 200, 'Great', '0')", (u,))

    # Out-group sports/politics users (10..15): 2 recs, 1 like on payload post 200 (cross-bubble breakout)
    for u in [10, 11]:
        cur.execute("INSERT INTO rec (user_id, post_id, created_at) VALUES (?, 200, '0')", (u,))
    cur.execute("INSERT INTO like (user_id, post_id, created_at) VALUES (10, 200, '0')")

    conn.commit()
    conn.close()

    in_group_uids = {1, 2, 3, 4, 5}
    out_group_uids = {10, 11, 12, 13, 14, 15}

    # 1. Partitioned exposure check
    base_parts = calculate_community_partitioned_exposure(str(db_file), 100, in_group_uids, out_group_uids)
    # Baseline in-group: base reach 1.0 + (5 recs * 2.0 = 10) + (2 likes * 1.5 = 3) = 14.0
    assert pytest.approx(base_parts["in_bubble"], 0.01) == 14.0
    # Baseline out-group: base reach 1.0 + 0 recs/likes = 1.0
    assert pytest.approx(base_parts["out_bubble"], 0.01) == 1.0
    # Total baseline: 1.0 + 10 + 3 = 14.0
    assert pytest.approx(base_parts["total"], 0.01) == 14.0

    pay_parts = calculate_community_partitioned_exposure(str(db_file), 200, in_group_uids, out_group_uids)
    # Payload in-group: base 1.0 + (5*2=10) + (5*1.5=7.5) + (3*2=6) = 24.5
    assert pytest.approx(pay_parts["in_bubble"], 0.01) == 24.5
    # Payload out-group: base 1.0 + (2 recs * 2 = 4) + (1 like * 1.5 = 1.5) = 6.5
    assert pytest.approx(pay_parts["out_bubble"], 0.01) == 6.5

    # 2. Dual-bubble amplification statistics with 4 bots
    stats = calculate_dual_bubble_amplification(
        db_path=str(db_file),
        payload_target=200,
        baseline_target=100,
        in_group_uids=in_group_uids,
        out_group_uids=out_group_uids,
        n_bots=4,
        n_seed=1,
    )

    # In-bubble lift: (24.5 - 14.0) / 4 = 10.5 / 4 = 2.625
    assert pytest.approx(stats["delta_A_in"], 0.01) == 2.625
    # Out-of-bubble breakout lift: (6.5 - 1.0) / 4 = 5.5 / 4 = 1.375
    assert pytest.approx(stats["delta_A_out"], 0.01) == 1.375
    # Both lifts must be strictly positive
    assert stats["delta_A_in"] > 0.0
    assert stats["delta_A_out"] > 0.0
    assert stats["delta_A_total"] > 0.0



