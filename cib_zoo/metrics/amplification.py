"""Causal amplification metric A(s, r) calculation for CIB research evaluation.

Supports both single-post tracking and multi-post topic narrative aggregation,
as well as dual in-bubble vs. out-of-bubble community partitioned amplification.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Any, Collection, Optional, Union


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


def resolve_posts_by_narrative(
    db_path: str,
    target: Union[int, list[int], set[int], str],
) -> list[int]:
    """Resolve post IDs matching a specific integer ID, list of IDs, or content tag pattern.

    Args:
        db_path: Path to SQLite simulation database.
        target: Integer post ID, iterable of post IDs, or substring/tag pattern (e.g. '%#target%').

    Returns:
        Sorted list of matching post IDs.
    """
    if isinstance(target, int):
        return [target]
    if isinstance(target, (list, set, tuple)):
        return sorted({int(x) for x in target if int(x) > 0})
    if isinstance(target, str):
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            pattern = target if "%" in target else f"%{target}%"
            cur.execute("SELECT post_id FROM post WHERE content LIKE ? ORDER BY post_id ASC", (pattern,))
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
    return []


def calculate_exposure_from_db(
    db_path: str,
    post_id: Union[int, list[int], set[int], str],
    epsilon: float = 1e-6,
    user_ids: Optional[Collection[int]] = None,
    exclude_user_ids: Optional[Collection[int]] = None,
) -> float:
    """Calculate cumulative visibility and exposure of a post or topic narrative.

    Combines algorithmic recommendations (rec table / rec_impression_log),
    engagement signals (likes, comments), and trace logs. Can optionally be
    partitioned to a specific subset of users (in-bubble or out-of-bubble) or
    exclude specific users (such as bots) to isolate organic platform reach.

    Args:
        db_path: Path to SQLite simulation database.
        post_id: Target post ID, list of post IDs, or topic/tag pattern string.
        epsilon: Regularization constant preventing division by zero.
        user_ids: Optional filter restricting engagement to a specific cohort of users.
        exclude_user_ids: Optional filter excluding specific user IDs (e.g. bots).

    Returns:
        Exposure score scalar >= 0.0.
    """
    pids = resolve_posts_by_narrative(db_path, post_id)
    if not pids:
        return 0.0

    exclude_uids_list = (
        sorted({int(x) for x in exclude_user_ids})
        if exclude_user_ids is not None
        else []
    )
    exclude_uids_set = set(exclude_uids_list)

    if user_ids is not None:
        target_uids_list = sorted(
            {int(x) for x in user_ids if int(x) not in exclude_uids_set}
        )
        target_uids_set = set(target_uids_list)
        if len(user_ids) > 0 and not target_uids_list:
            # Target cohort was non-empty, but all target users were excluded
            return 0.0
    else:
        target_uids_list = []
        target_uids_set = set()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        placeholders_p = ",".join("?" for _ in pids)
        filter_user = bool(target_uids_list)
        filter_exclude = bool(exclude_uids_list)
        placeholders_u = ",".join("?" for _ in target_uids_list) if filter_user else ""
        placeholders_ex = ",".join("?" for _ in exclude_uids_list) if filter_exclude else ""

        def count_table_rows(table_name: str) -> int:
            if table_name not in existing_tables:
                return 0
            query = f"SELECT COUNT(*) FROM [{table_name}] WHERE post_id IN ({placeholders_p})"
            params = list(pids)
            if filter_user:
                query += f" AND user_id IN ({placeholders_u})"
                params.extend(target_uids_list)
            if filter_exclude:
                query += f" AND user_id NOT IN ({placeholders_ex})"
                params.extend(exclude_uids_list)
            cursor.execute(query, tuple(params))
            return cursor.fetchone()[0]

        # Check rec_impression_log first if available and populated (longitudinal impressions)
        rec_impressions = 0
        has_rec_log = False
        if "rec_impression_log" in existing_tables:
            cursor.execute("SELECT 1 FROM rec_impression_log LIMIT 1")
            if cursor.fetchone() is not None:
                has_rec_log = True
                rec_impressions = count_table_rows("rec_impression_log")

        if not has_rec_log and "rec" in existing_tables:
            rec_impressions = count_table_rows("rec")

        likes_count = count_table_rows("like")
        comments_count = count_table_rows("comment")
        dislikes_count = count_table_rows("dislike")
        mutes_reports_count = count_table_rows("report")

        # Trace count (evaluating user_id filters if provided)
        trace_count = 0
        if "trace" in existing_tables:
            query = "SELECT user_id, info FROM trace"
            clauses = []
            params = []
            if filter_user:
                clauses.append(f"user_id IN ({placeholders_u})")
                params.extend(target_uids_list)
            if filter_exclude:
                clauses.append(f"user_id NOT IN ({placeholders_ex})")
                params.extend(exclude_uids_list)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            cursor.execute(query, tuple(params))

            target_pids_set = set(pids)
            for _, info_str in cursor.fetchall():
                for pid in target_pids_set:
                    if is_post_trace_match(info_str, pid):
                        trace_count += 1
                        break

        # Base reach per post item + weighted interaction volume
        if filter_exclude:
            # If strict bot exclusion is active and zero organic interactions occurred
            # (no impressions, no likes, no comments, no traces), organic reach is 0.0.
            if (rec_impressions + likes_count + comments_count + trace_count) == 0:
                return 0.0

            # Posts authored by excluded users (bots) do not grant organic base reach.
            organic_author_count = 0
            if "post" in existing_tables:
                cursor.execute(
                    f"SELECT post_id, user_id FROM post WHERE post_id IN ({placeholders_p})",
                    tuple(pids),
                )
                for pid, uid in cursor.fetchall():
                    if uid is not None and uid not in exclude_uids_set:
                        if not filter_user or uid in target_uids_set:
                            organic_author_count += 1
            base_reach = float(organic_author_count)
        else:
            base_reach = float(len(pids))

        total_exposure = float(
            base_reach
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


def calculate_community_partitioned_exposure(
    db_path: str,
    target: Union[int, list[int], set[int], str],
    in_group_uids: set[int],
    out_group_uids: set[int],
    epsilon: float = 1e-6,
    exclude_user_ids: Optional[Collection[int]] = None,
) -> dict[str, float]:
    """Calculate narrative exposure partitioned into in-bubble, out-of-bubble, and total.

    Args:
        db_path: Path to SQLite simulation database.
        target: Target post ID, list of IDs, or tag/topic string.
        in_group_uids: Set of user IDs residing within the target community bubble.
        out_group_uids: Set of user IDs residing outside the target bubble.
        epsilon: Small epsilon for numerical stability.
        exclude_user_ids: Optional filter excluding specific user IDs (e.g. bots).

    Returns:
        Dictionary with keys 'in_bubble', 'out_bubble', and 'total'.
    """
    exp_in = calculate_exposure_from_db(
        db_path, target, epsilon, user_ids=in_group_uids, exclude_user_ids=exclude_user_ids
    )
    exp_out = calculate_exposure_from_db(
        db_path, target, epsilon, user_ids=out_group_uids, exclude_user_ids=exclude_user_ids
    )
    exp_total = calculate_exposure_from_db(
        db_path, target, epsilon, user_ids=None, exclude_user_ids=exclude_user_ids
    )
    return {
        "in_bubble": exp_in,
        "out_bubble": exp_out,
        "total": exp_total,
    }


def calculate_dual_bubble_amplification(
    db_path: str,
    payload_target: Union[int, list[int], set[int], str],
    baseline_target: Union[int, list[int], set[int], str],
    in_group_uids: set[int],
    out_group_uids: set[int],
    n_bots: int,
    n_seed: int = 1,
    epsilon: float = 1e-6,
    exclude_user_ids: Optional[Collection[int]] = None,
) -> dict[str, float]:
    """Calculate dual in-bubble vs. out-of-bubble causal amplification statistics.

    Allows measuring whether an attack penetrates only its immediate echo chamber (in-bubble)
    or successfully breaks out into cross-community recommendation feeds (out-of-bubble).

    Args:
        db_path: Path to SQLite simulation database.
        payload_target: Payload post ID(s) or tag pattern.
        baseline_target: Baseline control post ID(s) or tag pattern.
        in_group_uids: User IDs in target community bubble (e.g. tech).
        out_group_uids: User IDs outside target community bubble (e.g. sports, politics).
        n_bots: Number of coordinated bots.
        n_seed: Number of seed authors (default 1).
        epsilon: Epsilon preventing division by zero.
        exclude_user_ids: Optional filter excluding specific user IDs (e.g. bots).

    Returns:
        Dictionary containing in-bubble, out-bubble, and total amplification scores.
    """
    assert n_bots > 0, f"n_bots must be positive, got {n_bots}"
    assert n_seed > 0, f"n_seed must be positive, got {n_seed}"

    pay_parts = calculate_community_partitioned_exposure(
        db_path, payload_target, in_group_uids, out_group_uids, epsilon, exclude_user_ids=exclude_user_ids
    )
    base_parts = calculate_community_partitioned_exposure(
        db_path, baseline_target, in_group_uids, out_group_uids, epsilon, exclude_user_ids=exclude_user_ids
    )

    resource_ratio = float(n_seed) / float(n_bots)

    # In-bubble metrics
    delta_A_in = float(pay_parts["in_bubble"] - base_parts["in_bubble"]) / float(n_bots)
    ratio_A_in = resource_ratio * (pay_parts["in_bubble"] / (base_parts["in_bubble"] + epsilon))

    # Out-of-bubble metrics (breakout reach)
    delta_A_out = float(pay_parts["out_bubble"] - base_parts["out_bubble"]) / float(n_bots)
    ratio_A_out = resource_ratio * (pay_parts["out_bubble"] / (base_parts["out_bubble"] + epsilon))

    # Total global metrics
    delta_A_total = float(pay_parts["total"] - base_parts["total"]) / float(n_bots)
    ratio_A_total = resource_ratio * (pay_parts["total"] / (base_parts["total"] + epsilon))

    return {
        "exposure_payload_in": pay_parts["in_bubble"],
        "exposure_baseline_in": base_parts["in_bubble"],
        "delta_A_in": delta_A_in,
        "ratio_A_in": ratio_A_in,
        "exposure_payload_out": pay_parts["out_bubble"],
        "exposure_baseline_out": base_parts["out_bubble"],
        "delta_A_out": delta_A_out,
        "ratio_A_out": ratio_A_out,
        "exposure_payload_total": pay_parts["total"],
        "exposure_baseline_total": base_parts["total"],
        "delta_A_total": delta_A_total,
        "ratio_A_total": ratio_A_total,
    }


def calculate_causal_amplification(
    db_path: str,
    payload_post_id: Union[int, list[int], set[int], str],
    baseline_post_id: Union[int, list[int], set[int], str],
    n_bots: int,
    n_seed: int = 1,
    epsilon: float = 1e-6,
    mode: str = "ratio",
    exclude_user_ids: Optional[Collection[int]] = None,
) -> float:
    """Calculate the Causal Algorithmic Amplification metric A(s, r).

    Supports both single post IDs and multi-post topic narrative targets.

    Supports two modes:
    - "ratio": Classical multiplier relative to baseline:
          A(s, r) = (N_seed / N_bots) * (Exposure(Payload_s) / (Exposure(Baseline_org) + epsilon))
    - "difference": Additive causal treatment effect normalized by bot cost:
          A(s, r) = (Exposure(Payload_s) - Exposure(Baseline_org)) / N_bots
    """
    assert n_bots > 0, f"n_bots must be positive, got {n_bots}"
    assert n_seed > 0, f"n_seed must be positive, got {n_seed}"

    exposure_payload = calculate_exposure_from_db(
        db_path, payload_post_id, epsilon, exclude_user_ids=exclude_user_ids
    )
    exposure_baseline = calculate_exposure_from_db(
        db_path, baseline_post_id, epsilon, exclude_user_ids=exclude_user_ids
    )

    if mode == "difference":
        return float(exposure_payload - exposure_baseline) / float(n_bots)

    resource_ratio = float(n_seed) / float(n_bots)
    relative_exposure = exposure_payload / (exposure_baseline + epsilon)

    return resource_ratio * relative_exposure


def calculate_differential_amplification(
    db_path: str,
    payload_post_id: Union[int, list[int], set[int], str],
    baseline_post_id: Union[int, list[int], set[int], str],
    n_bots: Optional[int] = None,
    n_seed: int = 1,
    min_expected_bot_actions: int = 0,
    bot_ids: Optional[Collection[int]] = None,
    exclude_user_ids: Optional[Collection[int]] = None,
) -> float:
    """Calculate the unified additive causal lift per bot unit:

        Delta A(s, r) = (Exposure(Payload) - Exposure(Baseline)) / N_bots
    """
    from cib_zoo.agent.cib_agent import CIBAgent

    resolved_bot_ids = list(bot_ids) if bot_ids is not None else CIBAgent.get_bot_ids()
    effective_n_bots = n_bots if n_bots is not None else CIBAgent.get_instance_count()

    assert effective_n_bots > 0, f"n_bots must be positive, got {effective_n_bots}"
    assert n_seed > 0, f"n_seed must be positive, got {n_seed}"

    # Determine exclude_user_ids: if explicitly provided, use it;
    # otherwise, if resolved_bot_ids is available, default to excluding bot self-actions
    effective_exclude = (
        exclude_user_ids
        if exclude_user_ids is not None
        else (resolved_bot_ids if resolved_bot_ids else None)
    )

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
        exclude_user_ids=effective_exclude,
    )
