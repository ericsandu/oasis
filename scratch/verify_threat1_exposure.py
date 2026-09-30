"""Empirical Verification Script for Threat 1 (T-04): Circular Exposure Metric Inflation.

Directly tests calculate_exposure_from_db() in oasis/cib_zoo/metrics/amplification.py
under pure bot self-engagement scenarios with zero organic reach.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

from cib_zoo.metrics.amplification import calculate_exposure_from_db


def create_schema(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            content TEXT,
            created_at TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE rec (
            user_id INTEGER,
            post_id INTEGER,
            PRIMARY KEY(user_id, post_id)
        );
    """)
    cur.execute("""
        CREATE TABLE [like] (
            like_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            post_id INTEGER
        );
    """)
    cur.execute("""
        CREATE TABLE comment (
            comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            post_id INTEGER,
            content TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE trace (
            trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            created_at TEXT,
            action TEXT,
            info TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE dislike (
            dislike_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            post_id INTEGER
        );
    """)
    cur.execute("""
        CREATE TABLE report (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            post_id INTEGER
        );
    """)
    conn.commit()
    conn.close()


def test_threat_1_circular_exposure():
    print("======================================================================")
    print("EMPIRICAL TEST: Threat 1 / T-04 (Circular Exposure Metric Inflation)")
    print("======================================================================")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_threat1.db")
        create_schema(db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # 1. Insert payload post (post_id = 100) created by Bot 151
        cur.execute("INSERT INTO post (post_id, user_id, content, created_at) VALUES (100, 151, 'Payload CIB Post #target_tag', '2026-09-29 10:00:00')")

        # 2. Insert baseline post (post_id = 200) created by Organic User 1
        cur.execute("INSERT INTO post (post_id, user_id, content, created_at) VALUES (200, 1, 'Baseline Organic Post #baseline_tag', '2026-09-29 10:00:00')")

        # 3. Simulate Organic Cohort (Users 1..150) and Bot Squad (Users 151..180, 30 bots)
        organic_uids = list(range(1, 151))
        bot_uids = list(range(151, 181))

        # Baseline post has 50 organic impressions in rec table:
        for uid in organic_uids[:50]:
            cur.execute("INSERT INTO rec (user_id, post_id) VALUES (?, 200)", (uid,))

        # PAYLOAD POST HAS ZERO ORGANIC REACH:
        # - 0 organic impressions in rec
        # - 0 organic likes
        # - 0 organic comments
        # - 0 organic traces

        # BUT BOTS ENGAGE HEAVILY IN SELF-ENGAGEMENT:
        # 30 bots produce 150 comments on payload post
        for i in range(150):
            bot_id = bot_uids[i % len(bot_uids)]
            cur.execute("INSERT INTO comment (user_id, post_id, content) VALUES (?, 100, ?)", (bot_id, f"Bot comment {i}"))
            cur.execute("INSERT INTO trace (user_id, created_at, action, info) VALUES (?, '2026-09-29 10:05:00', 'create_comment', ?)", (bot_id, "{'post_id': 100}"))

        # 30 bots produce 30 likes on payload post
        for bot_id in bot_uids:
            cur.execute("INSERT INTO [like] (user_id, post_id) VALUES (?, 100)", (bot_id,))
            cur.execute("INSERT INTO trace (user_id, created_at, action, info) VALUES (?, '2026-09-29 10:06:00', 'like_post', ?)", (bot_id, "{'post_id': 100}"))

        conn.commit()
        conn.close()

        # 4. Measure exposure using calculate_exposure_from_db
        # Case A: Global exposure as called in run_fep.py:904-905 (user_ids=None)
        exp_payload_global = calculate_exposure_from_db(db_path, post_id=100, user_ids=None)
        exp_baseline_global = calculate_exposure_from_db(db_path, post_id=200, user_ids=None)

        print(f"\n1. Global Exposure Calculation (user_ids=None):")
        print(f"   Payload Post Exposure (0 organic impressions, 150 bot comments, 30 bot likes): {exp_payload_global}")
        print(f"   Baseline Post Exposure (50 organic impressions, 0 bot actions):                 {exp_baseline_global}")
        
        # Verify manual arithmetic:
        # base_reach = 1.0
        # rec_impressions = 0
        # likes_count = 30 * 1.5 = 45.0
        # comments_count = 150 * 2.0 = 300.0
        # trace_count = 180 * 0.5 = 90.0
        # Expected total = 1.0 + 45.0 + 300.0 + 90.0 = 436.0
        expected_payload_exp = 1.0 + (30 * 1.5) + (150 * 2.0) + (180 * 0.5)
        print(f"   Exact theoretical formula result: {expected_payload_exp}")
        assert abs(exp_payload_global - expected_payload_exp) < 1e-3, f"Expected {expected_payload_exp}, got {exp_payload_global}"

        # 5. Measure Differential Amplification:
        net_lift = exp_payload_global - exp_baseline_global
        num_bots = len(bot_uids)
        delta_A = net_lift / num_bots
        print(f"\n2. Resulting Metric Inflation:")
        print(f"   Net Lift:                    {net_lift:.1f}")
        print(f"   Differential Amplification:  {delta_A:.2f}")

        # 6. Check what happens if we inspect organic exposure only
        # With user_ids=organic_uids:
        exp_payload_organic = calculate_exposure_from_db(db_path, post_id=100, user_ids=organic_uids)
        print(f"\n3. Organic Exposure when filtered by user_ids=organic_uids:")
        print(f"   Payload Organic Exposure: {exp_payload_organic} (reflects only base_reach=1.0, 0 actual engagement)")

        # Verify that bot rows are counted in like, comment, and trace
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM [like] WHERE post_id = 100")
        like_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM comment WHERE post_id = 100")
        comment_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM trace WHERE info LIKE '%100%'")
        trace_count = cur.fetchone()[0]
        conn.close()

        print(f"\n4. Database Row Confirmation:")
        print(f"   Bot Likes counted:     {like_count} (all from bot IDs {min(bot_uids)}-{max(bot_uids)})")
        print(f"   Bot Comments counted:  {comment_count} (all from bot IDs {min(bot_uids)}-{max(bot_uids)})")
        print(f"   Bot Traces counted:    {trace_count} (all from bot IDs {min(bot_uids)}-{max(bot_uids)})")

        assert exp_payload_global > 400.0, "Bot actions failed to inflate exposure metric"
        assert exp_payload_organic == 1.0, "Organic exposure should only be base_reach=1.0"

        print("\n======================================================================")
        print("EMPIRICALLY CONFIRMED THREAT 1 (T-04):")
        print(f"- calculate_exposure_from_db() blindly sums bot likes, comments, and traces.")
        print(f"- When 30 bots talk to themselves (150 comments, 30 likes) with 0 organic reach,")
        print(f"  the metric reports an exposure of {exp_payload_global:.1f} and Delta A = {delta_A:.2f}!")
        print(f"- The reported amplification is 100% self-generated phantom contamination.")
        print("======================================================================")


if __name__ == "__main__":
    test_threat_1_circular_exposure()
