"""Causal amplification metric A(s, r) calculation for CIB research evaluation."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Optional


def is_post_trace_match(info_str: str, post_id: int) -> bool:
    """Check if a trace.info string refers specifically to the target post_id.

    Avoids raw substring matching bugs where post_id=1 would match user_id=105
    or conversational text like '15 minutes' or post_id=10.
    """
    if not info_str:
        return False
    try:
        data = json.loads(info_str)
        if isinstance(data, dict):
            if data.get("post_id") == post_id:
                return True
            # Also check nested post_id if present
            if "post" in data and isinstance(data["post"], dict):
                if data["post"].get("post_id") == post_id:
                    return True
    except (ValueError, TypeError):
        pass

    # Regex fallback for non-JSON or shorthand formats (e.g. 'post:1', 'post_id: 1')
    pattern = rf'(?:post_id["\':\s]+|post:)\s*{post_id}\b'
    return bool(re.search(pattern, info_str))


def calculate_exposure_from_db(
    db_path: str,
    post_id: int,
    epsilon: float = 1e-6,
) -> float:
    """Calculate cumulative visibility and exposure of a post across simulation steps.

    Combines algorithmic recommendations (rec table), engagement signals (likes, comments),
    and trace logs.

    Args:
        db_path: Path to SQLite simulation database.
        post_id: Target post ID.
        epsilon: Regularization constant preventing division by zero.

    Returns:
        Exposure score scalar >= 0.0.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check available tables in database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        rec_impressions = 0
        if "rec" in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM rec WHERE post_id = ?", (post_id,))
            rec_impressions = cursor.fetchone()[0]

        likes_count = 0
        if "like" in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM like WHERE post_id = ?", (post_id,))
            likes_count = cursor.fetchone()[0]

        comments_count = 0
        if "comment" in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM comment WHERE post_id = ?", (post_id,))
            comments_count = cursor.fetchone()[0]

        trace_count = 0
        if "trace" in existing_tables:
            cursor.execute("SELECT info FROM trace")
            for (info_str,) in cursor.fetchall():
                if is_post_trace_match(info_str, post_id):
                    trace_count += 1

        # Base exposure plus weighted algorithmic and interaction reach
        # Negative signals (dislikes/mutes/reports) can optionally reduce net exposure
        negative_penalty = 0.0
        dislikes_count = 0
        if "dislike" in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM dislike WHERE post_id = ?", (post_id,))
            dislikes_count = cursor.fetchone()[0]

        mutes_reports_count = 0
        if "report" in existing_tables:
            cursor.execute("SELECT COUNT(*) FROM report WHERE post_id = ?", (post_id,))
            mutes_reports_count += cursor.fetchone()[0]

        total_exposure = float(
            1.0
            + (rec_impressions * 2.0)
            + (likes_count * 1.5)
            + (comments_count * 2.0)
            + (trace_count * 0.5)
            - (dislikes_count * 1.5)
            - (mutes_reports_count * 2.0)
        )
        return total_exposure

    finally:
        conn.close()


def calculate_causal_amplification(
    db_path: str,
    payload_post_id: int,
    baseline_post_id: int,
    n_bots: int,
    n_seed: int = 1,
    epsilon: float = 1e-6,
    mode: str = "ratio",
) -> float:
    """Calculate the Causal Algorithmic Amplification metric A(s, r).

    Supports two modes:
    - "ratio": Classical multiplier relative to baseline:
          A(s, r) = (N_seed / N_bots) * (Exposure(Payload_s) / (Exposure(Baseline_org) + epsilon))
    - "difference": Additive causal treatment effect normalized by bot cost:
          A(s, r) = (Exposure(Payload_s) - Exposure(Baseline_org)) / N_bots

    Args:
        db_path: Path to the SQLite simulation database.
        payload_post_id: Post ID of the coordinated attack payload.
        baseline_post_id: Post ID of the organic control baseline post.
        n_bots: Number of coordinated bots participating in the attack squad.
        n_seed: Number of initial seed authors (default 1).
        epsilon: Small epsilon preventing division by zero.
        mode: Metric computation mode ('ratio' or 'difference').

    Returns:
        Amplification score scalar (can be negative in 'difference' mode if attack backfires).
    """
    assert n_bots > 0, f"n_bots must be positive, got {n_bots}"
    assert n_seed > 0, f"n_seed must be positive, got {n_seed}"

    exposure_payload = calculate_exposure_from_db(db_path, payload_post_id, epsilon)
    exposure_baseline = calculate_exposure_from_db(db_path, baseline_post_id, epsilon)

    if mode == "difference":
        return float(exposure_payload - exposure_baseline) / float(n_bots)

    resource_ratio = float(n_seed) / float(n_bots)
    relative_exposure = exposure_payload / (exposure_baseline + epsilon)

    return resource_ratio * relative_exposure


def calculate_differential_amplification(
    db_path: str,
    payload_post_id: int,
    baseline_post_id: int,
    n_bots: Optional[int] = None,
    n_seed: int = 1,
    min_expected_bot_actions: int = 0,
    bot_ids: Optional[list[int]] = None,
) -> float:
    """Calculate the unified additive causal lift per bot unit:

        Delta A(s, r) = (Exposure(Payload) - Exposure(Baseline)) / N_bots

    Dynamically resolves bot identities and count from the CIBAgent registry,
    eliminating arbitrary ID thresholds or magic-number caps on user count.

    Args:
        db_path: Path to simulation database.
        payload_post_id: Post ID of the CIB attack payload.
        baseline_post_id: Post ID of the organic control post.
        n_bots: Optional number of bots. If omitted, resolved dynamically from CIBAgent.get_instance_count().
        n_seed: Number of seed authors.
        min_expected_bot_actions: Minimum threshold of bot actions expected in trace.
        bot_ids: Optional explicit list of bot user IDs. If omitted, resolved from CIBAgent.get_bot_ids().

    Returns:
        Scalar differential amplification.
    """
    from cib_zoo.agent.cib_agent import CIBAgent

    resolved_bot_ids = list(bot_ids) if bot_ids is not None else CIBAgent.get_bot_ids()
    effective_n_bots = n_bots if n_bots is not None else CIBAgent.get_instance_count()

    assert effective_n_bots > 0, f"n_bots must be positive, got {effective_n_bots}"
    assert n_seed > 0, f"n_seed must be positive, got {n_seed}"

    # Verify bot squad activity if required
    if min_expected_bot_actions > 0:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trace'")
            if cursor.fetchone():
                if resolved_bot_ids:
                    placeholders = ",".join("?" for _ in resolved_bot_ids)
                    cursor.execute(
                        f"SELECT COUNT(*) FROM trace WHERE user_id IN ({placeholders})",
                        tuple(resolved_bot_ids),
                    )
                else:
                    cursor.execute("SELECT COUNT(*) FROM trace WHERE user_id >= 0")

                total_actions = cursor.fetchone()[0]
                assert total_actions >= min_expected_bot_actions, (
                    f"Expected at least {min_expected_bot_actions} bot actions in trace, found {total_actions}"
                )
        finally:
            conn.close()

    return calculate_causal_amplification(
        db_path=db_path,
        payload_post_id=payload_post_id,
        baseline_post_id=baseline_post_id,
        n_bots=effective_n_bots,
        n_seed=n_seed,
        mode="difference",
    )
