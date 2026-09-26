"""Unit test suite for Track 4: Micro-Time Poisson Scheduler (oasis/clock/micro_time_scheduler.py).

Verifies Section 8.4 of docs/jev_optimization_plan.md:
- ScheduledAction ordering and comparison invariance
- MicroTimeScheduler Poisson arrival generation and ISO-8601 scheduling
- ChronologicalActionQueue async-safe and thread-safe chronological dispatch
- Zero sqlite3 imports and AST guardrail compliance
"""

from __future__ import annotations

import ast
import asyncio
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from oasis.clock.micro_time_scheduler import (
    ChronologicalActionQueue,
    MicroTimeScheduler,
    QueueEmpty,
    QueueFull,
    ScheduledAction,
)

# ==============================================================================
# 1. ScheduledAction Dataclass Verification
# ==============================================================================

class TestScheduledAction:
    """Verify ScheduledAction dataclass ordering, comparisons, and helper methods."""

    def test_scheduled_action_field_initialization(self) -> None:
        """Verify field assignment and types."""
        action = ScheduledAction(
            sort_timestamp=1700000000.5,
            iso_timestamp="2023-11-14T22:13:20.500000+00:00",
            user_id=42,
            action_dict={"action": "like", "post_id": 101},
        )
        assert action.sort_timestamp == 1700000000.5
        assert action.iso_timestamp == "2023-11-14T22:13:20.500000+00:00"
        assert action.user_id == 42
        assert action.action_dict == {"action": "like", "post_id": 101}

    def test_scheduled_action_order_by_sort_timestamp_strictly(self) -> None:
        """Verify comparison order is strictly dictated by sort_timestamp."""
        a1 = ScheduledAction(10.0, "2026-01-01T00:00:10Z", 99, {"type": "repost"})
        a2 = ScheduledAction(20.0, "2026-01-01T00:00:20Z", 1, {"type": "like"})
        a3 = ScheduledAction(5.0, "2026-01-01T00:00:05Z", 50, {"type": "comment"})

        assert a3 < a1 < a2
        assert a2 > a1 > a3
        assert a1 <= a2
        assert a2 >= a1

        actions = [a1, a2, a3]
        sorted_actions = sorted(actions)
        assert sorted_actions == [a3, a1, a2]

    def test_scheduled_action_compare_false_fields(self) -> None:
        """Verify iso_timestamp, user_id, and action_dict are ignored in comparisons."""
        # Even with different user_id, iso_timestamp, and action_dict:
        a1 = ScheduledAction(100.0, "2026-01-01T00:00:00Z", 1, {"payload": "first"})
        a2 = ScheduledAction(100.0, "2026-01-01T00:01:00Z", 2, {"payload": "second"})

        assert a1 == a2
        assert not (a1 < a2)
        assert not (a2 < a1)

    def test_scheduled_action_unhashable_uncomparable_action_dict(self) -> None:
        """Verify complex non-orderable objects in action_dict do not crash comparisons."""
        class NonOrderable:
            pass

        a1 = ScheduledAction(50.0, "t1", 1, {"complex": NonOrderable()})
        a2 = ScheduledAction(60.0, "t2", 2, {"complex": NonOrderable()})
        # Should compare solely on sort_timestamp without inspecting action_dict
        assert a1 < a2

    def test_scheduled_action_tuple_unpacking_and_indexing(self) -> None:
        """Verify tuple unpacking and subscripting for seamless backwards compatibility."""
        action = ScheduledAction(123.45, "2026-01-01T12:00:00Z", 7, {"key": "val"})

        ts, iso, uid, act = action
        assert ts == 123.45
        assert iso == "2026-01-01T12:00:00Z"
        assert uid == 7
        assert act == {"key": "val"}

        assert action[0] == 123.45
        assert action[1] == "2026-01-01T12:00:00Z"
        assert action[2] == 7
        assert action[3] == {"key": "val"}
        assert len(action) == 4

    def test_scheduled_action_dt_property_and_to_dict(self) -> None:
        """Verify dt property parses ISO string and to_dict formats dictionary."""
        iso_str = "2026-09-26T12:00:00+00:00"
        action = ScheduledAction(1000.0, iso_str, 5, {"action": "post"})

        dt = action.dt
        assert dt == datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)

        d = action.to_dict()
        assert d == {
            "sort_timestamp": 1000.0,
            "iso_timestamp": iso_str,
            "user_id": 5,
            "action_dict": {"action": "post"},
        }


# ==============================================================================
# 2. MicroTimeScheduler Poisson Process Verification
# ==============================================================================

class TestMicroTimeScheduler:
    """Verify continuous Poisson arrival generation and ISO-8601 action scheduling."""

    def test_scheduler_init_validation(self) -> None:
        """Verify parameter bounds in scheduler construction."""
        scheduler = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0)
        assert scheduler.step_duration_seconds == 900.0
        assert scheduler.default_lambda == 1.0

        with pytest.raises(ValueError, match="step_duration_seconds must be positive"):
            MicroTimeScheduler(step_duration_seconds=0.0)

        with pytest.raises(ValueError, match="step_duration_seconds must be positive"):
            MicroTimeScheduler(step_duration_seconds=-10.0)

        with pytest.raises(ValueError, match="default_lambda must be positive"):
            MicroTimeScheduler(default_lambda=0.0)

        with pytest.raises(ValueError, match="default_lambda must be positive"):
            MicroTimeScheduler(default_lambda=-1.0)

    def test_scheduler_deterministic_seeding(self) -> None:
        """Verify reproducible pseudo-random sampling when seed is fixed."""
        s1 = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0, seed=42)
        s2 = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0, seed=42)

        samples1 = [s1.sample_offset(1.5) for _ in range(50)]
        samples2 = [s2.sample_offset(1.5) for _ in range(50)]
        assert samples1 == samples2

        # Resetting seed produces identical sequence
        s1.set_seed(42)
        samples1_reset = [s1.sample_offset(1.5) for _ in range(50)]
        assert samples1_reset == samples1

    def test_scheduler_sample_offset_bounds(self) -> None:
        """Verify sampled offsets are strictly bounded within [0, step_duration_seconds]."""
        scheduler = MicroTimeScheduler(step_duration_seconds=300.0, default_lambda=2.0, seed=123)

        for freq in [0.01, 0.1, 0.5, 1.0, 5.0, 50.0, 500.0]:
            for _ in range(100):
                offset = scheduler.sample_offset(freq)
                assert 0.0 <= offset <= 300.0

    def test_scheduler_sample_offset_inactive_agent(self) -> None:
        """Verify agents with activity_frequency <= 0 default to step_duration_seconds."""
        scheduler = MicroTimeScheduler(step_duration_seconds=600.0, default_lambda=1.0)
        assert scheduler.sample_offset(0.0) == 600.0
        assert scheduler.sample_offset(-1.5) == 600.0

    def test_scheduler_statistical_arrival_rate_scaling(self) -> None:
        """Verify higher activity_frequency yields shorter delays (faster arrivals)."""
        scheduler = MicroTimeScheduler(
            step_duration_seconds=10000.0, default_lambda=1.0, seed=999
        )
        n_samples = 1500

        low_freq_offsets = [scheduler.sample_offset(0.2) for _ in range(n_samples)]
        high_freq_offsets = [scheduler.sample_offset(5.0) for _ in range(n_samples)]

        mean_low = sum(low_freq_offsets) / len(low_freq_offsets)
        mean_high = sum(high_freq_offsets) / len(high_freq_offsets)

        # Expected delay is 1 / lambda = 1 / (freq * 1.0)
        # For freq=0.2, expected ~ 5.0; for freq=5.0, expected ~ 0.2
        assert mean_high < mean_low
        assert mean_high < 1.0
        assert mean_low > 2.0

    def test_scheduler_sample_arrival_times_batch(self) -> None:
        """Verify sample_arrival_times handles lists of floats and agent-like objects."""
        scheduler = MicroTimeScheduler(step_duration_seconds=500.0, default_lambda=1.0, seed=10)

        class MockAgent:
            def __init__(self, freq: float):
                self.activity_frequency = freq

        class MockLegacyAgent:
            def __init__(self, freq_list: list[int]):
                self.activity_level_frequency = freq_list

        agents = [1.0, 2.0, MockAgent(3.0), MockLegacyAgent([4, 4, 4])]
        offsets = scheduler.sample_arrival_times(agents, step_duration=250.0)

        assert len(offsets) == 4
        for off in offsets:
            assert 0.0 <= off <= 250.0

    def test_scheduler_schedule_actions_basic_and_sorting(self) -> None:
        """Verify schedule_actions returns chronologically sorted ScheduledActions with valid ISO-8601."""
        scheduler = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0, seed=42)
        base_time = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)

        actions = [
            (10, {"action": "like", "post_id": 1}, 1.0),
            (20, {"action": "repost", "post_id": 2}, 2.0),
            (30, {"action": "comment", "content": "hi"}, 0.5),
        ]

        scheduled = scheduler.schedule_actions(step_index=0, base_time=base_time, agent_actions=actions)
        assert len(scheduled) == 3

        # Verify sorted ascending
        assert scheduled[0].sort_timestamp <= scheduled[1].sort_timestamp <= scheduled[2].sort_timestamp

        # Verify ISO-8601 formatting in UTC
        for sa in scheduled:
            dt = datetime.fromisoformat(sa.iso_timestamp)
            assert dt.tzinfo is not None
            assert base_time <= dt <= base_time + timedelta(seconds=900.0)
            assert sa.sort_timestamp == pytest.approx(dt.timestamp(), rel=1e-5)

    def test_scheduler_schedule_actions_step_offset_calculation(self) -> None:
        """Verify step offset T_base = base_time + step_index * step_duration_seconds."""
        scheduler = MicroTimeScheduler(step_duration_seconds=600.0, default_lambda=1.0, seed=7)
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        step_index = 3
        # Base offset for step 3: 3 * 600 = 1800s (00:30:00)
        expected_min_dt = datetime(2026, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
        expected_max_dt = datetime(2026, 1, 1, 0, 40, 0, tzinfo=timezone.utc)

        actions = [(1, {"action": "post"}, 1.0)]
        scheduled = scheduler.schedule_actions(step_index=step_index, base_time=base_time, agent_actions=actions)

        dt = scheduled[0].dt
        assert expected_min_dt <= dt <= expected_max_dt

    def test_scheduler_schedule_actions_raw_and_precomputed_offsets(self) -> None:
        """Verify precomputed offset overrides and use_raw_offsets flag."""
        scheduler = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0)
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Using use_raw_offsets=True
        raw_actions = [
            (1, {"id": "late"}, 300.0),
            (2, {"id": "early"}, 50.0),
            (3, {"id": "middle"}, 150.0),
        ]
        scheduled = scheduler.schedule_actions(
            step_index=0, base_time=base_time, agent_actions=raw_actions, use_raw_offsets=True
        )
        assert [sa.user_id for sa in scheduled] == [2, 3, 1]
        assert scheduled[0].sort_timestamp == pytest.approx(base_time.timestamp() + 50.0)
        assert scheduled[1].sort_timestamp == pytest.approx(base_time.timestamp() + 150.0)
        assert scheduled[2].sort_timestamp == pytest.approx(base_time.timestamp() + 300.0)

        # 2. Using action_dict["_offset"]
        dict_actions = [
            (10, {"action": "like", "_offset": 12.5}, 99.0),
            (20, {"action": "repost", "_offset": 2.5}, 0.1),
        ]
        scheduled_dict = scheduler.schedule_actions(
            step_index=0, base_time=base_time, agent_actions=dict_actions
        )
        assert scheduled_dict[0].user_id == 20
        assert scheduled_dict[1].user_id == 10
        assert scheduled_dict[0].sort_timestamp == pytest.approx(base_time.timestamp() + 2.5)
        assert scheduled_dict[1].sort_timestamp == pytest.approx(base_time.timestamp() + 12.5)

    def test_scheduler_schedule_actions_input_types(self) -> None:
        """Verify base_time accepts naive datetime, aware datetime, ISO string, and epoch float."""
        scheduler = MicroTimeScheduler(step_duration_seconds=900.0, default_lambda=1.0, seed=1)
        actions = [(1, {"type": "ping"}, 1.0)]

        # Naive datetime
        res1 = scheduler.schedule_actions(0, datetime(2026, 5, 1, 10, 0, 0), actions)  # noqa: DTZ001
        assert res1[0].iso_timestamp.startswith("2026-05-01T10:")

        # ISO string
        res2 = scheduler.schedule_actions(0, "2026-05-01T10:00:00+00:00", actions)
        assert res2[0].iso_timestamp.startswith("2026-05-01T10:")

        # POSIX float
        ts_float = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        res3 = scheduler.schedule_actions(0, ts_float, actions)
        assert res3[0].iso_timestamp.startswith("2026-05-01T10:")

    def test_scheduler_schedule_actions_empty_and_error(self) -> None:
        """Verify empty input handling and invalid step_index rejection."""
        scheduler = MicroTimeScheduler()
        base_time = datetime.now(timezone.utc)

        assert scheduler.schedule_actions(0, base_time, []) == []

        with pytest.raises(ValueError, match="step_index must be non-negative"):
            scheduler.schedule_actions(-1, base_time, [])

        with pytest.raises(TypeError):
            scheduler.schedule_actions(0, None, [])  # type: ignore


# ==============================================================================
# 3. ChronologicalActionQueue Concurrency & Priority Verification
# ==============================================================================

class TestChronologicalActionQueue:
    """Verify thread-safety, async-safety, chronological ordering, and Channel dispatch."""

    def test_queue_synchronous_priority_ordering(self) -> None:
        """Verify sync push/pop extracts actions in strict chronological timestamp order."""
        q = ChronologicalActionQueue()
        a_late = ScheduledAction(30.0, "t30", 1, {"a": "late"})
        a_early = ScheduledAction(10.0, "t10", 2, {"a": "early"})
        a_mid = ScheduledAction(20.0, "t20", 3, {"a": "mid"})

        q.push_nowait(a_late)
        q.push_nowait(a_early)
        q.push_nowait(a_mid)

        assert len(q) == 3
        assert q.pop_nowait() == a_early
        assert q.pop_nowait() == a_mid
        assert q.pop_nowait() == a_late
        assert q.empty()

    def test_queue_fifo_stability_for_equal_timestamps(self) -> None:
        """Verify FIFO tie-breaking preserves insertion order when micro-timestamps match."""
        q = ChronologicalActionQueue()
        a1 = ScheduledAction(10.0, "t10_1", 101, {"seq": 1})
        a2 = ScheduledAction(10.0, "t10_2", 102, {"seq": 2})
        a3 = ScheduledAction(10.0, "t10_3", 103, {"seq": 3})

        q.push_batch([a1, a2, a3])

        popped = [q.pop_nowait() for _ in range(3)]
        assert [x.user_id for x in popped] == [101, 102, 103]

    def test_queue_empty_and_full_exceptions(self) -> None:
        """Verify QueueEmpty and QueueFull inherit from both asyncio and queue standard exceptions."""
        q = ChronologicalActionQueue(maxsize=2)

        # Pop from empty queue
        with pytest.raises(QueueEmpty) as exc_info:
            q.pop_nowait()
        assert isinstance(exc_info.value, asyncio.QueueEmpty)
        assert isinstance(exc_info.value, queue.Empty)

        # Fill queue
        a1 = ScheduledAction(1.0, "t1", 1, {})
        a2 = ScheduledAction(2.0, "t2", 2, {})
        a3 = ScheduledAction(3.0, "t3", 3, {})
        q.push_nowait(a1)
        q.push_nowait(a2)
        assert q.full()

        with pytest.raises(QueueFull) as exc_info_full:
            q.push_nowait(a3)
        assert isinstance(exc_info_full.value, asyncio.QueueFull)
        assert isinstance(exc_info_full.value, queue.Full)

    def test_queue_peek_and_drain(self) -> None:
        """Verify peek non-destructively inspects earliest item and drain empties in order."""
        q = ChronologicalActionQueue()
        assert q.peek() is None

        actions = [
            ScheduledAction(float(i), f"t{i}", i, {})
            for i in [40, 10, 30, 20]
        ]
        q.push_batch(actions)

        assert q.peek().user_id == 10  # type: ignore
        assert len(q) == 4

        drained = q.drain()
        assert len(drained) == 4
        assert [x.user_id for x in drained] == [10, 20, 30, 40]
        assert q.empty()

    @pytest.mark.asyncio
    async def test_queue_async_put_get(self) -> None:
        """Verify asynchronous put and get operations with priority ordering and empty-queue wakeup."""
        q = ChronologicalActionQueue()

        # 1. Verify priority ordering when multiple items are pushed asynchronously
        await q.put(ScheduledAction(30.0, "t30", 3, {}))
        await q.put(ScheduledAction(10.0, "t10", 1, {}))
        await q.put(ScheduledAction(20.0, "t20", 2, {}))

        results = [
            (await q.get()).user_id,
            (await q.get()).user_id,
            (await q.get()).user_id,
        ]
        assert results == [1, 2, 3]

        # 2. Verify coroutine waiting on empty queue wakes up when an item arrives
        async def delayed_consumer() -> int:
            action = await q.get()
            return action.user_id

        task = asyncio.create_task(delayed_consumer())
        await asyncio.sleep(0.01)  # Allow consumer to suspend on empty queue
        await q.put(ScheduledAction(50.0, "t50", 42, {}))

        result_delayed = await task
        assert result_delayed == 42

    @pytest.mark.asyncio
    async def test_queue_async_concurrent_producers_and_consumers(self) -> None:
        """Verify high concurrency with multiple coroutine producers and consumers."""
        q = ChronologicalActionQueue()
        n_items_per_producer = 25
        n_producers = 4
        total_items = n_producers * n_items_per_producer

        async def producer(p_id: int) -> None:
            for i in range(n_items_per_producer):
                ts = float((p_id * 100) + (n_items_per_producer - i))
                await q.put(ScheduledAction(ts, f"iso_{ts}", p_id, {"idx": i}))
                await asyncio.sleep(0.001)

        async def consumer(collector: list[ScheduledAction]) -> None:
            while len(collector) < total_items:
                item = await q.get()
                collector.append(item)
                q.task_done()

        consumed: list[ScheduledAction] = []
        producers = [asyncio.create_task(producer(p)) for p in range(n_producers)]
        consumer_task = asyncio.create_task(consumer(consumed))

        await asyncio.gather(*producers)
        await q.join()
        await consumer_task

        assert len(consumed) == total_items
        # Ensure all tasks processed
        assert q.empty()

    @pytest.mark.asyncio
    async def test_queue_drain_to_channel_dispatch(self) -> None:
        """Verify drain_to_channel dispatches actions in chronological order to OASIS Channel."""
        class MockChannel:
            def __init__(self):
                self.dispatched = []

            async def write_to_receive_queue(self, payload: Any) -> str:
                self.dispatched.append(payload)
                return "msg_123"

        channel = MockChannel()
        q = ChronologicalActionQueue()

        a1 = ScheduledAction(10.0, "2026-01-01T00:00:10Z", 1, {"action": "like"})
        a2 = ScheduledAction(5.0, "2026-01-01T00:00:05Z", 2, {"action": "repost"})
        q.push_batch([a1, a2])

        count = await q.drain_to_channel(channel)
        assert count == 2
        assert q.empty()

        # Dispatched in order of sort_timestamp: a2 first (5.0), then a1 (10.0)
        assert channel.dispatched[0] == (2, {"action": "repost"}, "2026-01-01T00:00:05Z")
        assert channel.dispatched[1] == (1, {"action": "like"}, "2026-01-01T00:00:10Z")

    def test_queue_thread_safe_producer_and_sync_get(self) -> None:
        """Verify thread-safe ingestion across multiple OS threads with blocking get_sync."""
        q = ChronologicalActionQueue()
        n_threads = 4
        items_per_thread = 20
        total_items = n_threads * items_per_thread

        def producer_worker(t_id: int) -> None:
            for i in range(items_per_thread):
                time.sleep(0.0005)
                ts = float((t_id * 1000) + i)
                q.push_nowait(ScheduledAction(ts, f"t_{ts}", t_id, {"i": i}))

        threads = [
            threading.Thread(target=producer_worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()

        popped: list[ScheduledAction] = []
        for _ in range(total_items):
            item = q.get_sync(timeout=3.0)
            popped.append(item)
            q.task_done()

        for t in threads:
            t.join()

        assert len(popped) == total_items
        assert q.join_sync(timeout=1.0)
        assert q.empty()

    def test_queue_get_sync_timeout(self) -> None:
        """Verify get_sync raises QueueEmpty upon timeout on empty queue."""
        q = ChronologicalActionQueue()
        with pytest.raises(QueueEmpty, match="timed out"):
            q.get_sync(timeout=0.05)


# ==============================================================================
# 4. AST & Guardrail Verification (Zero SQLite Imports)
# ==============================================================================

class TestASTGuardrailsForTrack4:
    """Verify hermeticity, thread limits, zero sqlite3 imports, and clean AST."""

    def test_zero_sqlite3_imports_in_track_4_source(self) -> None:
        """Ensure micro_time_scheduler.py has strictly zero sqlite3 imports."""
        oasis_root = Path(__file__).resolve().parents[2]
        file_path = oasis_root / "oasis/clock/micro_time_scheduler.py"

        assert file_path.exists(), f"Track 4 file missing: {file_path}"
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "sqlite3" not in alias.name, (
                        f"Forbidden 'import {alias.name}' detected in {file_path}:{node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "sqlite3" not in mod, (
                    f"Forbidden 'from {mod} import ...' detected in {file_path}:{node.lineno}"
                )

    def test_zero_sqlite3_imports_in_track_4_tests(self) -> None:
        """Ensure test_micro_time_scheduler.py has strictly zero sqlite3 imports."""
        file_path = Path(__file__).resolve()
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "sqlite3" not in alias.name, (
                        f"Forbidden 'import {alias.name}' detected in {file_path}:{node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "sqlite3" not in mod, (
                    f"Forbidden 'from {mod} import ...' detected in {file_path}:{node.lineno}"
                )

    def test_no_forbidden_sql_statements_in_track_4(self) -> None:
        """Ensure no raw SQL query literals exist in Track 4 code."""
        oasis_root = Path(__file__).resolve().parents[2]
        file_path = oasis_root / "oasis/clock/micro_time_scheduler.py"

        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))

        forbidden_sql = [
            "SELECT ", "INSERT INTO ", "DELETE FROM ", "UPDATE ", "DROP TABLE", "CREATE TABLE"
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val_upper = node.value.strip().upper()
                for fragment in forbidden_sql:
                    assert fragment not in val_upper, (
                        f"Forbidden raw SQL query literal '{fragment}' detected in {file_path}:{node.lineno}."
                    )
