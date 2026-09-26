"""Tests for unified causal differential amplification metric and stability under edge-case distributions."""

import sqlite3
import pytest
from cib_zoo.metrics.amplification import (
    calculate_causal_amplification,
    calculate_differential_amplification,
    calculate_exposure_from_db,
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

    with pytest.raises(AssertionError, match="Expected at least 10 actions"):
        calculate_differential_amplification(
            db_path=str(db_file),
            payload_post_id=1,
            baseline_post_id=2,
            n_bots=5,
            min_expected_bot_actions=10,
        )
