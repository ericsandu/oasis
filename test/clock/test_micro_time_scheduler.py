"""Core OASIS Clock tests for MicroTimeScheduler and ChronologicalActionQueue."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest

from oasis.clock import (
    ChronologicalActionQueue,
    MicroTimeScheduler,
    QueueEmpty,
    QueueFull,
    ScheduledAction,
)


def test_scheduled_action_ordering():
    """Verify ScheduledAction order strictly follows sort_timestamp."""
    a1 = ScheduledAction(10.0, "2026-01-01T00:00:10Z", 1, {"type": "like"})
    a2 = ScheduledAction(5.0, "2026-01-01T00:00:05Z", 2, {"type": "repost"})
    assert a2 < a1
    assert sorted([a1, a2]) == [a2, a1]


def test_micro_time_scheduler_sampling():
    """Verify MicroTimeScheduler generates continuous offsets within step duration."""
    scheduler = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0, seed=42)
    offset = scheduler.sample_offset(activity_frequency=1.0)
    assert 0.0 <= offset <= 900.0


def test_micro_time_scheduler_schedule_actions():
    """Verify schedule_actions returns sorted ScheduledAction list."""
    scheduler = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0, seed=42)
    base_time = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
    actions = [
        (1, {"action": "like"}, 1.0),
        (2, {"action": "post"}, 2.0),
    ]
    scheduled = scheduler.schedule_actions(step_index=1, base_time=base_time, agent_actions=actions)
    assert len(scheduled) == 2
    assert scheduled[0].sort_timestamp <= scheduled[1].sort_timestamp
    for sa in scheduled:
        assert sa.iso_timestamp.startswith("2026-09-26T12:15:")


@pytest.mark.asyncio
async def test_chronological_action_queue_async():
    """Verify async chronological ordering in ChronologicalActionQueue."""
    queue = ChronologicalActionQueue()
    await queue.put(ScheduledAction(20.0, "t20", 2, {}))
    await queue.put(ScheduledAction(10.0, "t10", 1, {}))

    first = await queue.get()
    second = await queue.get()
    assert first.user_id == 1
    assert second.user_id == 2
