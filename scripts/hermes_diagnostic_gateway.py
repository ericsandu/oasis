#!/usr/bin/env python3
"""Hermes Simulation Diagnostic Gateway & Discord Webhook Notifier.

Automatically ingests OASIS CIB simulation telemetry, logs, and database metrics,
queries Hermes Agent (backed by local vLLM serving Qwen 3.8 27B) to diagnose anomalies,
failure modes, or fatal crashes, and dispatches structured diagnostic reports to Discord.

Supports both:
1. Single-run audits (--results-json, --db-path, --log-file)
2. Full batch directory audits (--batch-dir pointing to overall summary + individual run folders & databases)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hermes_gateway")


def extract_db_summary(
    db_path: str | None,
    bot_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Extract forensic sanity metrics directly from an OASIS simulation SQLite database."""
    if not db_path or not os.path.exists(db_path):
        return {"status": "Database not found"}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()

        tables: dict[str, Any] = {}
        for tbl in (
            "user",
            "post",
            "like",
            "comment",
            "trace",
            "rec",
            "rec_impression_log",
            "follow",
        ):
            cur.execute(
                f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'"
            )
            if cur.fetchone()[0] > 0:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                tables[tbl] = cur.fetchone()[0]
            else:
                tables[tbl] = 0

        # Check for stance column and non-zero distribution in post table
        cur.execute("PRAGMA table_info(post)")
        post_cols = [c[1] for c in cur.fetchall()]
        if "stance" in post_cols:
            cur.execute("SELECT COUNT(*) FROM post WHERE stance != 0.0")
            tables["posts_with_non_zero_stance"] = cur.fetchone()[0]
            cur.execute("SELECT MIN(stance), AVG(stance), MAX(stance) FROM post")
            min_s, avg_s, max_s = cur.fetchone()
            tables["post_stance_range"] = {
                "min": round(min_s, 3) if min_s is not None else 0.0,
                "avg": round(avg_s, 3) if avg_s is not None else 0.0,
                "max": round(max_s, 3) if max_s is not None else 0.0,
            }
        else:
            tables["posts_with_non_zero_stance"] = 0
            tables["stance_column_missing"] = True

        # Action breakdown in trace
        cur.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='trace'"
        )
        if cur.fetchone()[0] > 0:
            cur.execute("SELECT action, COUNT(*) FROM trace GROUP BY action")
            tables["trace_actions"] = dict(cur.fetchall())

            # Bot vs organic trace breakdown if bot_ids available
            if bot_ids:
                ph = ",".join("?" for _ in bot_ids)
                cur.execute(
                    f"SELECT COUNT(*) FROM trace WHERE user_id IN ({ph})", bot_ids
                )
                tables["bot_traces"] = cur.fetchone()[0]
                cur.execute(
                    f"SELECT COUNT(*) FROM trace WHERE user_id NOT IN ({ph})", bot_ids
                )
                tables["organic_traces"] = cur.fetchone()[0]

        # Like and comment organic vs bot attribution
        if bot_ids:
            ph = ",".join("?" for _ in bot_ids)
            for tbl in ("like", "comment"):
                cur.execute(
                    f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'"
                )
                if cur.fetchone()[0] > 0:
                    cur.execute(
                        f"SELECT COUNT(*) FROM {tbl} WHERE user_id NOT IN ({ph})",
                        bot_ids,
                    )
                    tables[f"organic_{tbl}s"] = cur.fetchone()[0]
                    cur.execute(
                        f"SELECT COUNT(*) FROM {tbl} WHERE user_id IN ({ph})", bot_ids
                    )
                    tables[f"bot_{tbl}s"] = cur.fetchone()[0]

        conn.close()
        return tables
    except Exception as e:
        return {"error": f"Failed to query database: {e}"}


def extract_batch_forensics(batch_dir_path: str) -> dict[str, Any]:
    """Scan and ingest an entire batch run folder, individual run JSONs, and SQLite databases."""
    batch_dir = Path(batch_dir_path)
    if not batch_dir.is_dir():
        return {"error": f"Batch directory not found: {batch_dir_path}"}

    # 1. Locate top-level summary JSON
    summary_file = None
    for candidate in ("cib_batch_summary.json", "batch_summary.json"):
        p = batch_dir / candidate
        if p.is_file():
            summary_file = p
            break
    if not summary_file:
        summaries = list(batch_dir.glob("*summary*.json"))
        if summaries:
            summary_file = summaries[0]

    batch_meta: dict[str, Any] = {}
    if summary_file:
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                batch_meta = json.load(f)
        except Exception as e:
            batch_meta = {"json_read_error": str(e)}

    # 2. Discover run subdirectories
    runs: dict[str, Any] = {}
    subdirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
    for sub in subdirs:
        res_json = None
        db_file = None
        for f in sub.glob("*.json"):
            if "results" in f.name or f.name.startswith("run_"):
                res_json = f
                break
        if not res_json:
            jsons = list(sub.glob("*.json"))
            if jsons:
                res_json = jsons[0]

        db_candidates = list(sub.glob("*.db"))
        if db_candidates:
            db_file = db_candidates[0]

        if res_json or db_file:
            run_data: dict[str, Any] = {}
            if res_json and res_json.is_file():
                try:
                    with open(res_json, "r", encoding="utf-8") as f:
                        run_data = json.load(f)
                except Exception as e:
                    run_data = {"error": str(e)}

            bot_ids = run_data.get("bot_ids", [])
            db_stats = extract_db_summary(
                str(db_file) if db_file else None, bot_ids=bot_ids
            )

            runs[sub.name] = {
                "results": run_data,
                "db_summary": db_stats,
                "db_path": str(db_file) if db_file else None,
                "json_path": str(res_json) if res_json else None,
            }

    # 3. Aggregate batch-level forensic flags
    aggregate: dict[str, Any] = {
        "total_runs_discovered": len(runs),
        "presets_tested": list(
            sorted(
                set(
                    r["results"].get("preset", "unknown")
                    for r in runs.values()
                    if isinstance(r.get("results"), dict)
                )
            )
        ),
        "runs_with_zero_organic_actions": 0,
        "runs_with_t01_contamination": 0,
        "runs_with_empty_rec_tables": 0,
        "runs_with_missing_stance": 0,
    }

    per_run_table: list[dict[str, Any]] = []
    for r_name, r_info in runs.items():
        res = r_info.get("results", {})
        db_s = r_info.get("db_summary", {})
        org_likes = db_s.get("organic_likes", 0)
        org_comments = db_s.get("organic_comments", 0)
        org_traces = db_s.get("organic_traces", 0)
        bot_actions = res.get("telemetry", {}).get("total_bot_actions", 0)
        diff_amp = res.get("differential_amplification", 0.0)

        is_t01 = False
        if org_likes == 0 and org_comments == 0:
            aggregate["runs_with_zero_organic_actions"] += 1
            if diff_amp > 1.0:
                aggregate["runs_with_t01_contamination"] += 1
                is_t01 = True

        is_t05 = False
        if db_s.get("rec") == 0 and db_s.get("rec_impression_log", 0) == 0:
            aggregate["runs_with_empty_rec_tables"] += 1
            is_t05 = True

        is_t07 = False
        if (
            db_s.get("stance_column_missing")
            or db_s.get("posts_with_non_zero_stance") == 0
        ):
            aggregate["runs_with_missing_stance"] += 1
            is_t07 = True

        per_run_table.append(
            {
                "run_id": r_name,
                "preset": res.get("preset", "N/A"),
                "topology": res.get("topology", "N/A"),
                "diff_amp": round(diff_amp, 4) if isinstance(diff_amp, (int, float)) else diff_amp,
                "bot_actions": bot_actions,
                "org_comments": org_comments,
                "org_likes": org_likes,
                "db_posts": db_s.get("post", 0),
                "db_traces": db_s.get("trace", 0),
                "t01_flag": is_t01,
                "t05_flag": is_t05,
                "t07_flag": is_t07,
            }
        )

    return {
        "batch_dir": str(batch_dir),
        "batch_summary": batch_meta,
        "aggregate_forensics": aggregate,
        "runs_summary_table": per_run_table,
        "runs": runs,
    }


def query_hermes_cli(
    prompt: str,
    base_url: str | None = None,
    model: str | None = None,
) -> str | None:
    """Attempt diagnosis via Hermes Agent CLI in headless oneshot mode."""
    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        return None

    try:
        logger.info("Invoking Hermes Agent CLI for forensic simulation diagnosis...")
        env = os.environ.copy()
        env["HERMES_YOLO_MODE"] = "1"
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
            env["VLLM_BASE_URL"] = base_url
        if model:
            env["HERMES_MODEL"] = model
            env["OPENAI_MODEL_NAME"] = model
        res = subprocess.run(
            [hermes_bin, "chat", "--oneshot", "-q", prompt],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
        logger.warning(
            "Hermes CLI returned non-zero (%d): %s", res.returncode, res.stderr
        )
    except Exception as e:
        logger.warning("Hermes CLI execution error: %s", e)
    return None


def query_vllm_direct(
    prompt: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen3.8-27B",
) -> str:
    """Direct fallback to local vLLM OpenAI-compatible endpoint."""
    logger.info("Querying vLLM endpoint directly (%s / %s)...", base_url, model)
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Hermes, an expert forensic AI auditor for the OASIS CIB simulation sandbox. "
                    "Analyze simulation telemetry, databases, and logs against AGENTS.md requirements. "
                    "Detect algorithmic anomalies (e.g. Threat T-01 bot self-engagement contamination, "
                    "Threat T-03 time-decay inversion, Threat T-05 recommendation wipeout, "
                    "Threat T-06 zombie skip collapse, Threat T-07 stance collapse) and provide clear technical verdicts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("Direct vLLM query failed: %s", e)
        return f"[Diagnosis Generation Error: {e}]"


def run_single_diagnosis(
    results_summary: dict[str, Any],
    log_tail: str,
    db_summary: dict[str, Any],
    error_trace: str | None,
    base_url: str,
    model: str,
) -> str:
    """Construct prompt and generate Hermes forensic analysis for a single run."""
    prompt = f"""### OASIS Simulation Single-Run Diagnostic Request
Please perform a forensic audit of the following OASIS CIB simulation execution:

1. **Results JSON Telemetry**:
```json
{json.dumps(results_summary, indent=2)}
```

2. **Database Row Counts & Metrics**:
```json
{json.dumps(db_summary, indent=2)}
```

3. **Recent Execution Logs / Error Traces**:
```text
{error_trace if error_trace else log_tail[-3000:]}
```

### Audit Instructions:
- Evaluate whether the results represent a genuine, realistic simulation or suffer from known failure modes documented in AGENTS.md:
  * Threat T-01: Bot self-engagement contamination / false amplification.
  * Threat T-03: Time-decay ranking inversion.
  * Threat T-05: Feed impression wipeout (empty rec / missing rec_impression_log).
  * Threat T-06: 100% Skip zombie collapse of organic agents.
  * Threat T-07: Belief state drift collapse to 0.0 (missing stance column or all zero).
  * Threat T-08: Dual clock advancement (+2 per step).
- If an error occurred, explain the exact line and root cause in the codebase.
- Provide a 3-5 bullet point executive summary and concrete remediation actions.
"""
    diag = query_hermes_cli(prompt, base_url=base_url, model=model)
    if not diag:
        diag = query_vllm_direct(prompt, base_url=base_url, model=model)
    return diag


def run_batch_diagnosis(
    batch_forensics: dict[str, Any],
    base_url: str,
    model: str,
) -> str:
    """Construct prompt and generate Hermes forensic analysis for a multi-run batch suite."""
    agg = batch_forensics.get("aggregate_forensics", {})
    runs_table = batch_forensics.get("runs_summary_table", [])
    batch_meta = batch_forensics.get("batch_summary", {})

    prompt = f"""### OASIS Batch Simulation Suite Diagnostic Request
Please perform an authoritative forensic audit of the following multi-run OASIS CIB experiment batch:

1. **Batch Aggregate Overview**:
- Total Runs Discovered: {agg.get('total_runs_discovered')}
- Presets Evaluated: {agg.get('presets_tested')}
- Runs with Zero Organic Actions: {agg.get('runs_with_zero_organic_actions')} / {agg.get('total_runs_discovered')}
- Runs Flagged for Threat T-01 (False Bot Amplification): {agg.get('runs_with_t01_contamination')}
- Runs with Empty Recommendation Tables (Threat T-05): {agg.get('runs_with_empty_rec_tables')}
- Runs with Missing / Zero Stances (Threat T-07): {agg.get('runs_with_missing_stance')}

2. **Per-Run Breakdown & Database Telemetry Table**:
```json
{json.dumps(runs_table, indent=2)}
```

3. **Top-Level Batch Telemetry Summary**:
```json
{json.dumps(batch_meta, indent=2)[:3500]}
```

### Forensic Audit Instructions (against AGENTS.md):
1. **Veracity of Amplification Claims**: Examine whether high differential amplification (e.g. S2 reporting ΔA > 10.0) is genuine cross-bubble organic persuasion, or purely circular bot self-engagement contamination (Threat T-01).
2. **Organic Social Dynamics**: Assess whether organic agents actively engaged or succumbed to 100% Skip zombie collapse (Threat T-06).
3. **Database Integrity**: Check if historical recommendation impressions were preserved in `rec_impression_log` or wiped out (Threat T-05).
4. **Ideological Dynamics**: Verify whether stances diversified across topics or suffered neutral collapse (Threat T-07).
5. **Executive Recommendations**: Deliver a concise executive summary comparing Baseline vs S1 vs S2 vs S3, highlighting scientific anomalies and recommending forward fixes.
"""
    diag = query_hermes_cli(prompt, base_url=base_url, model=model)
    if not diag:
        diag = query_vllm_direct(prompt, base_url=base_url, model=model)
    return diag


def send_discord_notification(
    webhook_url: str,
    status: str,
    diagnosis_text: str,
    summary_data: dict[str, Any],
    is_batch: bool = False,
    error_trace: str | None = None,
) -> bool:
    """Dispatch structured Embed message to Discord Webhook."""
    if not webhook_url:
        logger.warning("No DISCORD_WEBHOOK_URL configured. Skipping Discord dispatch.")
        return False

    now_iso = datetime.now(timezone.utc).isoformat()
    if status == "SUCCESS":
        color = 0x2ECC71  # Green
        title = "✅ OASIS Simulation Batch Audited Successfully" if is_batch else "✅ OASIS Simulation Completed"
    elif status == "WARNING":
        color = 0xF1C40F  # Yellow
        title = "⚠️ OASIS Simulation Batch: Algorithmic Anomalies Detected" if is_batch else "⚠️ OASIS Simulation: Anomaly Warnings"
    else:
        color = 0xE74C3C  # Red
        title = "❌ OASIS Simulation Failed / Error Encountered"

    truncated_diagnosis = diagnosis_text[:3800] + (
        "\n...[truncated]" if len(diagnosis_text) > 3800 else ""
    )

    fields = []
    if is_batch:
        agg = summary_data.get("aggregate_forensics", {})
        fields = [
            {"name": "Total Runs", "value": str(agg.get("total_runs_discovered", "N/A")), "inline": True},
            {"name": "Presets", "value": ", ".join(agg.get("presets_tested", [])) or "N/A", "inline": True},
            {"name": "T-01 Bot Contamination", "value": f"{agg.get('runs_with_t01_contamination', 0)} runs", "inline": True},
            {"name": "Zero Organic Actions", "value": f"{agg.get('runs_with_zero_organic_actions', 0)} runs", "inline": True},
            {"name": "Rec Table Wipeouts", "value": f"{agg.get('runs_with_empty_rec_tables', 0)} runs", "inline": True},
            {"name": "Missing Stances", "value": f"{agg.get('runs_with_missing_stance', 0)} runs", "inline": True},
        ]
    else:
        fields = [
            {"name": "Preset / Campaign", "value": str(summary_data.get("preset", "N/A")), "inline": True},
            {"name": "Bots / Agents", "value": f"{summary_data.get('num_bots', 'N/A')} bots", "inline": True},
            {"name": "Total Steps", "value": str(summary_data.get("max_steps", "N/A")), "inline": True},
        ]
        amp = summary_data.get("differential_amplification")
        if amp is not None:
            fields.append({"name": "ΔA(s,r) Amplification", "value": f"{amp:.4f}", "inline": True})

    embed = {
        "title": title,
        "description": truncated_diagnosis,
        "color": color,
        "fields": fields,
        "footer": {
            "text": "Hermes Agent Autonomous Diagnostic Gateway • Qwen 3.8 27B vLLM",
        },
        "timestamp": now_iso,
    }

    payload = {
        "username": "Hermes Simulation Sentinel",
        "avatar_url": "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/assets/hermes-logo.png",
        "embeds": [embed],
    }

    try:
        logger.info("Posting diagnostic report to Discord webhook...")
        resp = requests.post(webhook_url, json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("✓ Diagnostic report dispatched to Discord successfully.")
        return True
    except Exception as e:
        logger.error("Failed to post to Discord webhook: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Simulation Diagnostic Gateway")
    parser.add_argument(
        "--batch-dir",
        default=None,
        help="Path to folder containing multi-run batch results and databases",
    )
    parser.add_argument(
        "--results-json",
        default="cib_results.json",
        help="Path to single-run simulation results JSON",
    )
    parser.add_argument("--log-file", default=None, help="Path to simulation log file")
    parser.add_argument(
        "--db-path", default=None, help="Path to simulation SQLite database"
    )
    parser.add_argument(
        "--webhook-url",
        default=os.getenv("DISCORD_WEBHOOK_URL"),
        help="Discord Webhook URL",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--model", default=os.getenv("HERMES_MODEL", "Qwen/Qwen3.8-27B")
    )
    parser.add_argument(
        "--error-trace",
        default=None,
        help="Explicit error trace if simulation crashed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without posting to Discord and mock vLLM if offline",
    )
    args = parser.parse_args()

    is_batch_mode = args.batch_dir is not None

    if is_batch_mode:
        logger.info("Executing Batch Mode Forensic Audit on: %s", args.batch_dir)
        batch_forensics = extract_batch_forensics(args.batch_dir)
        if "error" in batch_forensics:
            logger.error("Batch extraction failed: %s", batch_forensics["error"])
            sys.exit(1)

        agg = batch_forensics["aggregate_forensics"]
        status = "SUCCESS"
        if agg["runs_with_t01_contamination"] > 0 or agg["runs_with_zero_organic_actions"] > 0:
            status = "WARNING"

        logger.info(
            "Analyzing %d batch runs with Hermes Agent...",
            agg["total_runs_discovered"],
        )
        diagnosis = run_batch_diagnosis(
            batch_forensics=batch_forensics,
            base_url=args.vllm_url,
            model=args.model,
        )

        if args.dry_run and diagnosis.startswith("[Diagnosis Generation Error:"):
            diagnosis = (
                "**[DRY-RUN BATCH AUDIT REPORT]**\n"
                f"- **Runs Evaluated**: {agg['total_runs_discovered']} across presets {agg['presets_tested']}.\n"
                f"- **Threat T-01 Contamination**: {agg['runs_with_t01_contamination']} runs showed bot-inflated ΔA with 0 organic reach.\n"
                f"- **Threat T-05 Telemetry**: {agg['runs_with_empty_rec_tables']} runs lacked impression log persistence.\n"
                f"- **Threat T-07 Beliefs**: {agg['runs_with_missing_stance']} runs lacked dynamic post stance enrichment.\n"
                "- **Verdict**: Historical baseline verified; Phase P0 remediation required for valid causal inference."
            )

        print("\n" + "=" * 65)
        print("HERMES BATCH FORENSIC DIAGNOSTIC REPORT")
        print("=" * 65)
        print(diagnosis)
        print("=" * 65 + "\n")

        if args.dry_run:
            print("[DRY-RUN] Discord batch payload preview:")
            print(
                json.dumps(
                    {
                        "status": status,
                        "batch_dir": args.batch_dir,
                        "total_runs": agg["total_runs_discovered"],
                        "diagnosis_preview": diagnosis[:200] + "...",
                    },
                    indent=2,
                )
            )
        elif args.webhook_url:
            send_discord_notification(
                webhook_url=args.webhook_url,
                status=status,
                diagnosis_text=diagnosis,
                summary_data=batch_forensics,
                is_batch=True,
            )

    else:
        # Single run audit flow
        results: dict[str, Any] = {}
        if os.path.exists(args.results_json):
            try:
                with open(args.results_json, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception as e:
                results = {"json_read_error": str(e)}

        log_tail = ""
        if args.log_file and os.path.exists(args.log_file):
            try:
                with open(args.log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    log_tail = "".join(lines[-150:])
            except Exception as e:
                log_tail = f"[Log read error: {e}]"

        bot_ids = results.get("bot_ids", [])
        db_summary = extract_db_summary(args.db_path, bot_ids=bot_ids)

        if args.error_trace:
            status = "ERROR"
        elif results.get("error"):
            status = "ERROR"
        else:
            status = "SUCCESS"

        logger.info("Analyzing simulation results with Hermes Agent...")
        diagnosis = run_single_diagnosis(
            results_summary=results,
            log_tail=log_tail,
            db_summary=db_summary,
            error_trace=args.error_trace,
            base_url=args.vllm_url,
            model=args.model,
        )

        if args.dry_run and diagnosis.startswith("[Diagnosis Generation Error:"):
            diagnosis = (
                "**[DRY-RUN SIMULATION EVALUATION]**\n"
                "- **Status**: Verified telemetry structure and SQLite health.\n"
                "- **Algorithmic Review**: Exposure metric correctly filters bot accounts (Threat T-01).\n"
                "- **Convergence**: Exponential half-life decay function maintains monotonic order (Threat T-03).\n"
                "- **Action Profile**: Healthy distribution across Like, Repost, Comment; zero zombie Skips observed."
            )

        if status == "SUCCESS" and any(
            w in diagnosis.lower()
            for w in (
                "anomaly",
                "unrealistic",
                "zombie",
                "circular",
                "contamination",
                "inversion",
            )
        ):
            status = "WARNING"

        print("\n" + "=" * 60)
        print("HERMES FORENSIC DIAGNOSTIC REPORT")
        print("=" * 60)
        print(diagnosis)
        print("=" * 60 + "\n")

        if args.dry_run:
            print("[DRY-RUN] Discord payload preview:")
            print(
                json.dumps(
                    {
                        "status": status,
                        "results": results,
                        "diagnosis_preview": diagnosis[:200] + "...",
                    },
                    indent=2,
                )
            )
        elif args.webhook_url:
            send_discord_notification(
                webhook_url=args.webhook_url,
                status=status,
                diagnosis_text=diagnosis,
                summary_data=results,
                is_batch=False,
                error_trace=args.error_trace,
            )


if __name__ == "__main__":
    main()
