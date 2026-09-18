"""Causal amplification metric A(s, r) calculation for CIB research evaluation."""

from __future__ import annotations

import math
import sqlite3
from typing import Optional


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
            cursor.execute("SELECT COUNT(*) FROM trace WHERE info LIKE ?", (f"%{post_id}%",))
            trace_count = cursor.fetchone()[0]

        # Base exposure plus weighted algorithmic and interaction reach
        total_exposure = float(1.0 + (rec_impressions * 2.0) + (likes_count * 1.5) + (comments_count * 2.0) + (trace_count * 0.5))
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
) -> float:
    """Calculate the Causal Algorithmic Amplification metric A(s, r).

    Formula:
        A(s, r) = (N_seed / N_bots) * (Exposure(Payload_s | r) / (Exposure(Baseline_org | r) + epsilon))

    Args:
        db_path: Path to the SQLite simulation database.
        payload_post_id: Post ID of the coordinated attack payload.
        baseline_post_id: Post ID of the organic control baseline post.
        n_bots: Number of coordinated bots participating in the attack squad.
        n_seed: Number of initial seed authors (default 1).
        epsilon: Small epsilon preventing division by zero.

    Returns:
        A(s, r) amplification ratio scalar.
    """
    if n_bots <= 0:
        raise ValueError(f"n_bots must be positive, got {n_bots}")
    if n_seed <= 0:
        raise ValueError(f"n_seed must be positive, got {n_seed}")

    exposure_payload = calculate_exposure_from_db(db_path, payload_post_id, epsilon)
    exposure_baseline = calculate_exposure_from_db(db_path, baseline_post_id, epsilon)

    resource_ratio = float(n_seed) / float(n_bots)
    relative_exposure = exposure_payload / (exposure_baseline + epsilon)

    return resource_ratio * relative_exposure
