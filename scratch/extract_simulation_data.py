#!/usr/bin/env python3
"""
Comprehensive Data Extraction & Telemetry Exporter for OASIS Simulations.
Pure standard-library implementation (sqlite3, csv, json) with zero external dependencies.
Parses SQLite database files (user, post, trace, follow, comment) and exports:
  1. summary_metrics.json (Global KPIs, action distributions, top-viral posts)
  2. posts_telemetry.csv  (Per-post metrics: likes, reposts, impressions, CTR)
  3. user_engagement.csv  (Per-user activity: posts, likes given/received, graph degrees)
"""

import os
import sys
import json
import csv
import sqlite3
import argparse
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter, defaultdict


def extract_simulation_data(db_path: str, output_dir: str) -> Dict[str, Any]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Simulation database not found: {db_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Discover available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = set(r[0] for r in cursor.fetchall())
    print(f"Found tables in DB: {sorted(list(tables))}")

    # 1. Trace / Actions Analysis
    action_counts = Counter()
    total_traces = 0
    impressions_map = defaultdict(int)  # post_id -> count of times delivered in REFRESH

    if "trace" in tables:
        cursor.execute("SELECT * FROM trace")
        traces = cursor.fetchall()
        total_traces = len(traces)
        for t in traces:
            action = t["action"] if "action" in t.keys() else None
            if action:
                action_counts[action] += 1

            if action == "refresh":
                info = t["info"] if "info" in t.keys() else None
                if info:
                    try:
                        data = json.loads(info) if isinstance(info, str) else info
                        post_ids = data.get("post_ids", []) if isinstance(data, dict) else []
                        for pid in post_ids:
                            impressions_map[pid] += 1
                    except Exception:
                        pass
        print(f"Extracted {total_traces} trace records across {len(action_counts)} action types.")

    # 2. Posts & Performance Analysis
    posts_data = []
    user_post_counts = Counter()
    user_likes_received = Counter()

    if "post" in tables:
        cursor.execute("SELECT * FROM post")
        posts = cursor.fetchall()
        print(f"Extracted {len(posts)} post records.")
        
        for row in posts:
            keys = row.keys()
            pid = row["post_id"]
            author_id = row["user_id"] if "user_id" in keys else None
            content = row["content"] if "content" in keys and row["content"] else ""
            created_at = str(row["created_at"]) if "created_at" in keys and row["created_at"] else ""
            likes = int(row["num_likes"] or 0) if "num_likes" in keys else 0
            dislikes = int(row["num_dislikes"] or 0) if "num_dislikes" in keys else 0
            shares = int(row["num_shares"] or 0) if "num_shares" in keys else 0
            reports = int(row["num_reports"] or 0) if "num_reports" in keys else 0
            impressions = impressions_map.get(pid, 0)
            
            ctr = (likes + shares) / max(1, impressions) if impressions > 0 else 0.0

            if author_id is not None:
                user_post_counts[author_id] += 1
                user_likes_received[author_id] += likes

            posts_data.append({
                "post_id": pid,
                "author_id": author_id,
                "content": content[:200].replace("\n", " "),
                "full_content_length": len(content),
                "created_at": created_at,
                "num_likes": likes,
                "num_dislikes": dislikes,
                "num_shares": shares,
                "num_reports": reports,
                "impressions": impressions,
                "ctr": round(ctr, 4),
                "total_engagement": likes + shares
            })
    
    posts_csv_path = os.path.join(output_dir, "posts_telemetry.csv")
    if posts_data:
        with open(posts_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(posts_data[0].keys()))
            writer.writeheader()
            writer.writerows(posts_data)
        print(f"Saved posts telemetry to: {posts_csv_path}")

    # 3. User Engagement & Social Graph Analysis
    users_data = []
    if "user" in tables:
        in_degrees = Counter()
        out_degrees = Counter()
        if "follow" in tables:
            cursor.execute("SELECT follower_id, followee_id FROM follow")
            for f_row in cursor.fetchall():
                out_degrees[f_row["follower_id"]] += 1
                in_degrees[f_row["followee_id"]] += 1

        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        print(f"Extracted {len(users)} user records.")

        for urow in users:
            ukeys = urow.keys()
            uid = urow["user_id"]
            uname = urow["user_name"] if "user_name" in ukeys and urow["user_name"] else f"user_{uid}"
            bio = ""
            if "bio" in ukeys and urow["bio"]:
                bio = urow["bio"]
            elif "profile" in ukeys and urow["profile"]:
                bio = urow["profile"]

            users_data.append({
                "user_id": uid,
                "user_name": uname,
                "bio_preview": str(bio)[:150].replace("\n", " "),
                "posts_created": user_post_counts.get(uid, 0),
                "likes_received": user_likes_received.get(uid, 0),
                "followers_count": in_degrees.get(uid, int(urow["num_followers"] or 0) if "num_followers" in ukeys else 0),
                "following_count": out_degrees.get(uid, int(urow["num_followings"] or 0) if "num_followings" in ukeys else 0),
            })

    users_csv_path = os.path.join(output_dir, "user_engagement.csv")
    if users_data:
        with open(users_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(users_data[0].keys()))
            writer.writeheader()
            writer.writerows(users_data)
        print(f"Saved user engagement metrics to: {users_csv_path}")

    # 4. Global Summary Metrics
    total_posts = len(posts_data)
    total_likes = sum(p["num_likes"] for p in posts_data)
    total_shares = sum(p["num_shares"] for p in posts_data)
    total_impressions = sum(p["impressions"] for p in posts_data)
    avg_ctr = (sum(p["ctr"] for p in posts_data) / total_posts) if total_posts > 0 else 0.0

    sorted_posts = sorted(posts_data, key=lambda x: (x["total_engagement"], x["num_likes"]), reverse=True)[:5]
    top_posts = [
        {
            "post_id": p["post_id"],
            "author_id": p["author_id"],
            "content": p["content"],
            "likes": p["num_likes"],
            "shares": p["num_shares"],
            "impressions": p["impressions"],
            "ctr": p["ctr"],
        }
        for p in sorted_posts
    ]

    summary = {
        "extraction_timestamp": datetime.now().isoformat(),
        "database_path": os.path.abspath(db_path),
        "total_users": len(users_data),
        "total_posts": total_posts,
        "total_traces": total_traces,
        "total_likes_recorded": total_likes,
        "total_shares_recorded": total_shares,
        "total_impressions_recorded": total_impressions,
        "average_post_ctr": round(avg_ctr, 4),
        "action_distribution": dict(action_counts),
        "top_viral_posts": top_posts,
    }

    summary_json_path = os.path.join(output_dir, "summary_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to: {summary_json_path}")

    conn.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract telemetry and metrics from an OASIS simulation SQLite DB.")
    parser.add_argument("--db", type=str, required=True, help="Path to input simulation SQLite DB file")
    parser.add_argument("--out", type=str, default=None, help="Output directory (defaults to ./extracted_<dbname>)")

    args = parser.parse_args()
    
    if args.out is None:
        db_stem = os.path.splitext(os.path.basename(args.db))[0]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.db)), f"extracted_{db_stem}")
    else:
        out_dir = args.out

    summary = extract_simulation_data(args.db, out_dir)
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"Total Users: {summary['total_users']}")
    print(f"Total Posts: {summary['total_posts']}")
    print(f"Total Traces: {summary['total_traces']}")
    print(f"Action Counts: {summary['action_distribution']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
