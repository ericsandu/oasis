---
name: oasis-db-forensics
description: 'Authoritative guide for querying, auditing, and forensic analysis of OASIS CIB simulation SQLite databases (simulation.db). Covers table schemas, JSON info extraction from traces, impression log auditing, bot-filtering patterns, and SQL threat detection queries for Threats T-01 through T-08.'
risk: low
source: internal/oasis-research
source_type: specialized-skill
date_added: 2026-09-30
---

# OASIS Database Forensics & Audit Skill

## Overview & Architecture

OASIS CIB simulations persist all agents, social actions, recommendations, and platform states into SQLite databases (`simulation.db`), typically located within run subdirectories (e.g. `experiments/batch_.../R07_.../simulation.db`).

### Safe Connection Rules (WAL Mode & Concurrency)
Simulations write asynchronously using SQLite WAL mode (`simulation.db-wal` and `simulation.db-shm`). When querying databases during or after runs, **always open read-only connections** with URI filenames to eliminate lock contention (`sqlite3.OperationalError: database is locked`):

```python
import sqlite3

# Always connect in read-only mode via URI
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
```

---

## Authoritative Database Schema & Table Reference

| Table | Description | Critical Columns & Forensic Notes |
| :--- | :--- | :--- |
| `user` | Agent profiles & identities | `user_id`, `user_name`, `name`, `bio`, `created_at`. Organic IDs are typically `1..N`, bots are `N+1..N+B`. |
| `post` | Root posts & quotes | `post_id`, `user_id`, `content`, `stance` (**REAL DEFAULT 0.0, added Phase P0**), `created_at`, `num_likes`, `num_dislikes`, `num_shares`, `num_reports`. |
| `rec` | **Transient** feed buffer | `user_id`, `post_id`, `step_index`, `rank`, `score`. **WARNING**: Upstream OASIS executes `DELETE FROM rec` every step! Only contains the latest step's feed. |
| `rec_impression_log` | **Persistent** longitudinal impressions | `log_id`, `step_index`, `user_id`, `post_id`, `rank`, `score`, `timestamp`. **Added in Phase P0** to preserve all impressions across all simulation steps. |
| `like` | Post like actions | `like_id`, `user_id`, `post_id`, `created_at`. |
| `comment` | Post comment actions | `comment_id`, `user_id`, `post_id`, `content`, `created_at`. |
| `trace` | Chronological event stream | `trace_id`, `user_id`, `action`, `info` (**JSON encoded!**), `created_at`. |
| `follow` | Social graph edges | `follow_id`, `follower_id`, `followee_id`, `created_at`. |

---

## Critical Forensic Pitfalls & Extraction Rules

### 1. The Trace Table `post_id` Trap (Threat T-02)
In the `trace` table, `post_id` is **NOT a root column**. It is embedded inside a JSON string in the `info` column:
```sql
-- FAILS in older code: SELECT post_id FROM trace; (KeyError / NoSuchColumn)

-- CORRECT SQLite extraction:
SELECT user_id, action, json_extract(info, '$.post_id') AS post_id, created_at 
FROM trace 
WHERE action IN ('like_post', 'create_comment', 'repost_post');
```

In Python, defensively extract `post_id`:
```python
import json

def extract_post_id_from_trace(row: dict) -> int | None:
    if "post_id" in row and row["post_id"] is not None:
        return int(row["post_id"])
    info_raw = row.get("info")
    if isinstance(info_raw, str):
        try:
            return json.loads(info_raw).get("post_id")
        except Exception:
            return None
    elif isinstance(info_raw, dict):
        return info_raw.get("post_id")
    return None
```

### 2. Bot Self-Engagement Contamination (Threat T-01)
Adversarial bot accounts coordinate by liking and commenting on each other's posts. If metric queries do not exclude bot IDs, circular bot interactions are falsely credited as "organic amplification":
```sql
-- CONTAMINATED QUERY (Includes bot echo chamber):
SELECT COUNT(*) FROM like WHERE post_id = :payload_id;

-- FORENSICALLY SOUND ORGANIC ENGAGEMENT QUERY:
SELECT COUNT(*) FROM like 
WHERE post_id = :payload_id 
  AND user_id NOT IN (SELECT user_id FROM user WHERE user_id >= :min_bot_id);
```
**Zero-Exposure Invariant**: If all engagements on a post are authored by excluded bots (`organic_impressions + organic_likes + organic_comments == 0`), organic exposure **MUST** return `0.0`.

---

## Threat Detection SQL Catalog (The 20 Threats)

When auditing an OASIS database, run this forensic checklist:

### Threat T-01: Circular Bot Amplification Check
Compare total actions on the payload post versus organic actions:
```sql
SELECT 
    COUNT(*) AS total_engagements,
    SUM(CASE WHEN user_id IN (:bot_ids) THEN 1 ELSE 0 END) AS bot_engagements,
    SUM(CASE WHEN user_id NOT IN (:bot_ids) THEN 1 ELSE 0 END) AS organic_engagements
FROM (
    SELECT user_id FROM like WHERE post_id = :payload_id
    UNION ALL
    SELECT user_id FROM comment WHERE post_id = :payload_id
);
```
*Verdict*: If `organic_engagements == 0` while the experiment claims $\Delta \mathcal{A} > 1.0$, the run suffers from Threat T-01 contamination.

### Threat T-05: Feed Impression Wipeout Audit
Check whether historical recommendation impressions were preserved or wiped out:
```sql
SELECT 
    (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='rec_impression_log') AS has_log_table,
    (SELECT COUNT(*) FROM rec) AS transient_rec_rows,
    (SELECT COUNT(*) FROM rec_impression_log) AS persistent_rec_rows;
```
*Verdict*: If `has_log_table == 0` or `persistent_rec_rows == 0` (with `transient_rec_rows == 0`), historical feed telemetry was obliterated by `DELETE FROM rec`.

### Threat T-06: Zombie Skip Collapse Audit
Check the action distribution among organic users:
```sql
SELECT 
    action, 
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM trace WHERE user_id NOT IN (:bot_ids)), 2) AS pct
FROM trace
WHERE user_id NOT IN (:bot_ids)
GROUP BY action
ORDER BY count DESC;
```
*Verdict*: If `action IN ('do_nothing', 'skip')` accounts for $\ge 98\%$ of organic traces, organic agents suffered from the 1-token fallback zombie collapse.

### Threat T-07: Post Stance Monotonic Collapse Audit
Verify whether posts have dynamic ideological stances ($s \in [-1.0, 1.0]$):
```sql
SELECT 
    COUNT(*) AS total_posts,
    SUM(CASE WHEN stance != 0.0 THEN 1 ELSE 0 END) AS non_zero_stance_posts,
    MIN(stance) AS min_stance,
    AVG(stance) AS avg_stance,
    MAX(stance) AS max_stance
FROM post;
```
*Verdict*: If `non_zero_stance_posts == 0`, all posts are strictly neutral ($0.0$). Under Deffuant dynamics, agent beliefs will monotonically collapse to apathy.

---

## Python Forensic Extraction Recipes

### 1. Single Run Quick Audit Script
```python
import sqlite3, json

def audit_simulation_db(db_path: str, bot_ids: list[int]) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    ph = ",".join("?" for _ in bot_ids) if bot_ids else "NULL"
    
    report = {}
    
    # 1. Table row counts
    for tbl in ("user", "post", "like", "comment", "trace", "rec", "rec_impression_log"):
        cur.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'")
        if cur.fetchone()[0] > 0:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            report[tbl] = cur.fetchone()[0]
        else:
            report[tbl] = 0
            
    # 2. Organic vs bot comments
    if bot_ids:
        cur.execute(f"SELECT COUNT(*) FROM comment WHERE user_id NOT IN ({ph})", bot_ids)
        report["organic_comments"] = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM comment WHERE user_id IN ({ph})", bot_ids)
        report["bot_comments"] = cur.fetchone()[0]
        
    # 3. Stance check
    cur.execute("PRAGMA table_info(post)")
    cols = [c[1] for c in cur.fetchall()]
    if "stance" in cols:
        cur.execute("SELECT COUNT(*) FROM post WHERE stance != 0.0")
        report["non_zero_stance_posts"] = cur.fetchone()[0]
    else:
        report["stance_missing"] = True
        
    conn.close()
    return report
```

### 2. Multi-Run Batch Traversal Recipe
```python
from pathlib import Path
import json

def traverse_batch_directory(batch_dir_path: str) -> list[dict]:
    batch_dir = Path(batch_dir_path)
    summary_data = []
    
    for sub in sorted(batch_dir.iterdir()):
        if not sub.is_dir():
            continue
        db_file = sub / "simulation.db"
        json_files = list(sub.glob("*.json"))
        
        if db_file.exists() and json_files:
            with open(json_files[0], "r") as f:
                res = json.load(f)
            bot_ids = res.get("bot_ids", [])
            db_stats = audit_simulation_db(str(db_file), bot_ids)
            
            summary_data.append({
                "run": sub.name,
                "preset": res.get("preset"),
                "diff_amp": res.get("differential_amplification"),
                "db_stats": db_stats,
            })
    return summary_data
```

---

## Hermes Agent Diagnostic Interpretation Rules

When Hermes Agent audits simulation outputs:
1. **Never trust `differential_amplification` in isolation**: Always cross-reference against `organic_likes`, `organic_comments`, and `rec_impression_log`.
2. **If `organic_comments == 0` and `organic_likes == 0`**, any reported amplification $> 0.0$ is an artifact of bot self-engagement contamination (Threat T-01).
3. **If `rec_impression_log` has 0 rows**, the recommendation system telemetry was erased prior to metric evaluation (Threat T-05).
4. **If all post stances are 0.0**, belief evolution graphs cannot be interpreted as ideological polarization or echo-chamber breach (Threat T-07).
