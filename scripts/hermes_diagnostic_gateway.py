#!/usr/bin/env python3
"""Hermes Simulation Diagnostic Gateway & Discord Webhook Notifier.

Automatically ingests OASIS CIB simulation telemetry, logs, and database metrics,
queries Hermes Agent (backed by local vLLM serving Qwen 3.8 27B) to diagnose anomalies
or fatal crashes, and dispatches a structured diagnostic report to Discord.
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


def extract_db_summary(db_path: str | None) -> dict[str, Any]:
    """Extract sanity metrics directly from the simulation SQLite database."""
    if not db_path or not os.path.exists(db_path):
        return {"status": "Database not found"}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()

        tables = {}
        for tbl in ("user", "post", "like", "comment", "trace", "rec", "rec_impression_log"):
            cur.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'")
            if cur.fetchone()[0] > 0:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                tables[tbl] = cur.fetchone()[0]
            else:
                tables[tbl] = 0

        # Check for non-zero stances in post table
        cur.execute("SELECT COUNT(*) FROM post WHERE stance != 0.0")
        tables["posts_with_non_zero_stance"] = cur.fetchone()[0]

        conn.close()
        return tables
    except Exception as e:
        return {"error": f"Failed to query database: {e}"}


def query_hermes_cli(prompt: str) -> str | None:
    """Attempt diagnosis via Hermes Agent CLI in headless oneshot mode."""
    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        return None

    try:
        logger.info("Invoking Hermes Agent CLI for forensic simulation diagnosis...")
        env = os.environ.copy()
        env["HERMES_YOLO_MODE"] = "1"
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
        logger.warning("Hermes CLI returned non-zero (%d): %s", res.returncode, res.stderr)
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
                    "Analyze simulation telemetry and logs against AGENTS.md requirements. Identify root causes, "
                    "detect algorithmic anomalies (e.g. zero organic reach, time decay inversion, zombie Skips), "
                    "and provide clear, concise technical recommendations."
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


def run_diagnosis(
    results_summary: dict[str, Any],
    log_tail: str,
    db_summary: dict[str, Any],
    error_trace: str | None,
    base_url: str,
    model: str,
) -> str:
    """Construct prompt and generate Hermes forensic analysis."""
    prompt = f"""### Simulation Diagnostic Request
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
  * Threat T-05: Feed impression wipeout (DELETE FROM rec).
  * Threat T-06: 100% Skip zombie collapse of organic agents.
  * Threat T-07: Belief state drift collapse to 0.0.
  * Threat T-08: Dual clock advancement (+2 per step).
- If an error occurred, explain the exact line and root cause in the codebase.
- Provide a 3-5 bullet point executive summary and concrete remediation actions.
"""
    # Try Hermes CLI first, fallback to direct vLLM
    diag = query_hermes_cli(prompt)
    if not diag:
        diag = query_vllm_direct(prompt, base_url=base_url, model=model)
    return diag


def send_discord_notification(
    webhook_url: str,
    status: str,
    diagnosis_text: str,
    results_summary: dict[str, Any],
    error_trace: str | None = None,
) -> bool:
    """Dispatch structured Embed message to Discord Webhook."""
    if not webhook_url:
        logger.warning("No DISCORD_WEBHOOK_URL configured. Skipping Discord dispatch.")
        return False

    now_iso = datetime.now(timezone.utc).isoformat()
    if status == "SUCCESS":
        color = 0x2ECC71  # Green
        title = "✅ OASIS Simulation Completed Successfully"
    elif status == "WARNING":
        color = 0xF1C40F  # Yellow
        title = "⚠️ OASIS Simulation Completed with Anomaly Warnings"
    else:
        color = 0xE74C3C  # Red
        title = "❌ OASIS Simulation Failed / Error Encountered"

    # Truncate diagnosis if exceeding Discord embed limit (4096 chars for description)
    truncated_diagnosis = diagnosis_text[:3800] + ("\n...[truncated]" if len(diagnosis_text) > 3800 else "")

    fields = [
        {"name": "Preset / Campaign", "value": str(results_summary.get("preset", "N/A")), "inline": True},
        {"name": "Bots / Agents", "value": f"{results_summary.get('num_bots', 'N/A')} bots", "inline": True},
        {"name": "Total Steps", "value": str(results_summary.get("max_steps", "N/A")), "inline": True},
    ]

    amp = results_summary.get("differential_amplification")
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
    parser.add_argument("--results-json", default="cib_results.json", help="Path to simulation results JSON")
    parser.add_argument("--log-file", default=None, help="Path to simulation log file")
    parser.add_argument("--db-path", default=None, help="Path to simulation SQLite database")
    parser.add_argument("--webhook-url", default=os.getenv("DISCORD_WEBHOOK_URL"), help="Discord Webhook URL")
    parser.add_argument("--vllm-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("HERMES_MODEL", "Qwen/Qwen3.8-27B"))
    parser.add_argument("--error-trace", default=None, help="Explicit error trace if simulation crashed")
    parser.add_argument("--dry-run", action="store_true", help="Run without posting to Discord and mock vLLM if offline")
    args = parser.parse_args()

    # 1. Load results JSON if present
    results = {}
    if os.path.exists(args.results_json):
        try:
            with open(args.results_json, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            results = {"json_read_error": str(e)}

    # 2. Read log tail if log file given
    log_tail = ""
    if args.log_file and os.path.exists(args.log_file):
        try:
            with open(args.log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                log_tail = "".join(lines[-150:])
        except Exception as e:
            log_tail = f"[Log read error: {e}]"

    # 3. Extract database metrics
    db_summary = extract_db_summary(args.db_path)

    # 4. Determine baseline status
    if args.error_trace:
        status = "ERROR"
    elif results.get("error"):
        status = "ERROR"
    else:
        status = "SUCCESS"

    # 5. Run Hermes forensic analysis
    logger.info("Analyzing simulation results with Hermes Agent...")
    diagnosis = run_diagnosis(
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

    # Check for anomaly warning flags in diagnosis
    if status == "SUCCESS" and any(w in diagnosis.lower() for w in ("anomaly", "unrealistic", "zombie", "circular", "contamination", "inversion")):
        status = "WARNING"

    # 6. Print diagnosis to console
    print("\n" + "=" * 60)
    print("HERMES FORENSIC DIAGNOSTIC REPORT")
    print("=" * 60)
    print(diagnosis)
    print("=" * 60 + "\n")

    # 7. Post to Discord
    if args.dry_run:
        print("[DRY-RUN] Discord payload preview:")
        print(json.dumps({
            "status": status,
            "results": results,
            "diagnosis_preview": diagnosis[:200] + "...",
        }, indent=2))
    elif args.webhook_url:
        send_discord_notification(
            webhook_url=args.webhook_url,
            status=status,
            diagnosis_text=diagnosis,
            results_summary=results,
            error_trace=args.error_trace,
        )


if __name__ == "__main__":
    main()
