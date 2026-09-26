"""Tests for recommendation system sensitivity to CIB manipulation and time-horizon stability."""

import math
import numpy as np
import pytest
from oasis.social_platform.recsys import (
    normalize_similarity_adjustments,
    get_trace_contents,
    reset_globals,
)
from oasis.social_platform.typing import ActionType


def test_recsys_date_score_horizon_stability():
    """Verify that the date_score calculation remains finite and non-NaN across long horizons (t > 90, t > 271)."""
    post_creation_time = 0
    test_current_times = [0, 10, 50, 90, 150, 270, 271, 272, 300, 500, 1000]

    for current_time in test_current_times:
        time_delta = min(270.0, max(0.0, float(current_time - post_creation_time)))
        score = np.log((271.8 - time_delta) / 100.0)

        assert not np.isnan(score), f"Score is NaN at current_time={current_time}"
        assert not np.isinf(score), f"Score is Inf at current_time={current_time}"
        assert score <= np.log(271.8 / 100.0), "Score exceeded max recency value"


def test_recsys_similarity_adjustment_sensitivity():
    """Verify that liking a target/anchor post increases the adjusted similarity score in the ranker."""
    post_scores = [
        (1, 0.2),
        (2, 0.5),
        (3, 0.8),
    ]
    base_similarity = 0.5

    # Case 1: High like similarity (e.g. user liked content closely aligned with payload)
    like_sim_high = 0.9
    dislike_sim_none = 0.0
    adjusted_high = normalize_similarity_adjustments(
        post_scores=post_scores,
        base_similarity=base_similarity,
        like_similarity=like_sim_high,
        dislike_similarity=dislike_sim_none,
    )
    assert adjusted_high > base_similarity, "Liking aligned content should increase adjusted similarity score"

    # Case 2: High dislike similarity (user disliked content aligned with payload)
    dislike_sim_high = 0.9
    like_sim_none = 0.0
    adjusted_low = normalize_similarity_adjustments(
        post_scores=post_scores,
        base_similarity=base_similarity,
        like_similarity=like_sim_none,
        dislike_similarity=dislike_sim_high,
    )
    assert adjusted_low < base_similarity, "Disliking aligned content should decrease adjusted similarity score"
    assert adjusted_high > adjusted_low, "Positive engagement must rank higher than negative engagement"


def test_recsys_trace_retrieval_integrity():
    """Verify trace table filtering correctly isolates liked post contents for similarity adjustments."""
    post_table = [
        {"post_id": 1, "user_id": 10, "content": "Organic sports post"},
        {"post_id": 2, "user_id": 20, "content": "CIB anchor post about finance"},
        {"post_id": 3, "user_id": 30, "content": "CIB payload narrative"},
    ]
    trace_table = [
        {"user_id": 100, "action": ActionType.LIKE_POST.value, "post_id": 2},
        {"user_id": 100, "action": ActionType.UNLIKE_POST.value, "post_id": 1},
        {"user_id": 101, "action": ActionType.LIKE_POST.value, "post_id": 3},
    ]

    # Test user 100 likes
    likes_100 = get_trace_contents(
        user_id=100,
        action=ActionType.LIKE_POST.value,
        post_table=post_table,
        trace_table=trace_table,
    )
    assert len(likes_100) == 1
    assert "CIB anchor post about finance" in likes_100

    # Test user 100 unlikes
    unlikes_100 = get_trace_contents(
        user_id=100,
        action=ActionType.UNLIKE_POST.value,
        post_table=post_table,
        trace_table=trace_table,
    )
    assert len(unlikes_100) == 1
    assert "Organic sports post" in unlikes_100
