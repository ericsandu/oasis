# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""Micro-Time Poisson Arrival Scheduler & Chronological Action Queue for OASIS.

Implements Track 4 of the JEV Optimization Architecture (docs/jev_optimization_plan.md Section 8.4):
- ScheduledAction: Dataclass ordered strictly by continuous micro-time float timestamp.
- MicroTimeScheduler: Continuous Poisson arrival generator (tau_i ~ Exp(lambda_i)).
- ChronologicalActionQueue: Thread-safe and async-safe priority queue ensuring causal dispatch order.
"""

from __future__ import annotations

import asyncio
import collections
import heapq
import queue
import random
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


class QueueEmpty(asyncio.QueueEmpty, queue.Empty):
    """Raised when attempting to retrieve an action from an empty queue non-blockingly."""


class QueueFull(asyncio.QueueFull, queue.Full):
    """Raised when attempting to insert an action into a full queue non-blockingly."""


@dataclass(order=True)
class ScheduledAction:
    """Dataclass representing an agent action scheduled at a continuous micro-timestamp.

    Attributes:
        sort_timestamp: Float representation of the continuous action timestamp (POSIX epoch).
            Used as the primary and only ordering key for chronological comparisons.
        iso_timestamp: ISO-8601 formatted string representation in UTC.
            Excluded from equality and ordering comparisons (compare=False).
        user_id: Unique agent identifier. Excluded from comparison (compare=False).
        action_dict: Dictionary containing the action payload, type, and parameters.
            Excluded from comparison (compare=False).
    """

    sort_timestamp: float
    iso_timestamp: str = field(compare=False)
    user_id: int = field(compare=False)
    action_dict: dict[str, Any] = field(compare=False)

    @property
    def dt(self) -> datetime:
        """Parse and return the ISO-8601 timestamp as a timezone-aware datetime object."""
        return datetime.fromisoformat(self.iso_timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the scheduled action into a dictionary format."""
        return {
            "sort_timestamp": self.sort_timestamp,
            "iso_timestamp": self.iso_timestamp,
            "user_id": self.user_id,
            "action_dict": self.action_dict,
        }

    def __iter__(self):
        """Enable tuple unpacking: sort_timestamp, iso_timestamp, user_id, action_dict = action."""
        yield self.sort_timestamp
        yield self.iso_timestamp
        yield self.user_id
        yield self.action_dict

    def __getitem__(self, index: int) -> Any:
        """Enable index-based access."""
        return (self.sort_timestamp, self.iso_timestamp, self.user_id, self.action_dict)[index]

    def __len__(self) -> int:
        """Return the number of fields in the action (4)."""
        return 4


class MicroTimeScheduler:
    """Continuous micro-time arrival scheduler based on Poisson processes.

    Replaces discrete, step-synchronized action execution waves with realistic,
    continuous arrival intervals sampled from exponential distributions:
        tau_i ~ Exponential(lambda_i), where lambda_i = activity_frequency * default_lambda
    bounded within [0.0, step_duration_seconds].

    Args:
        step_duration_seconds: Virtual duration of a discrete simulation step in seconds
            (default: 900.0, i.e. 15 minutes).
        default_lambda: Baseline arrival rate parameter lambda_0 (default: 1.0).
        seed: Optional integer seed for deterministic pseudo-random sampling.
    """

    def __init__(
        self,
        step_duration_seconds: float = 900.0,
        default_lambda: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if step_duration_seconds <= 0.0:
            raise ValueError(
                f"step_duration_seconds must be positive, got {step_duration_seconds}"
            )
        if default_lambda <= 0.0:
            raise ValueError(
                f"default_lambda must be positive, got {default_lambda}"
            )

        self.step_duration_seconds: float = float(step_duration_seconds)
        self.default_lambda: float = float(default_lambda)
        self._seed: int | None = seed
        self._rng: random.Random = random.Random(seed)

    def set_seed(self, seed: int | None = None) -> None:
        """Reset or modify the PRNG seed for deterministic simulation experiments.

        Args:
            seed: Integer seed value or None to reset to non-deterministic random state.
        """
        self._seed = seed
        self._rng = random.Random(seed)

    def sample_offset(self, activity_frequency: float) -> float:
        """Sample a continuous delay tau_i ~ Exp(lambda_i) bounded within [0, step_duration_seconds].

        The effective arrival rate is:
            lambda_i = activity_frequency * default_lambda
        If activity_frequency <= 0, the agent is inactive and delay defaults to the end of
        the step (step_duration_seconds).

        Args:
            activity_frequency: Agent activity frequency multiplier (from UserInfo).

        Returns:
            Continuous time delay in seconds, guaranteed within [0.0, step_duration_seconds].
        """
        if activity_frequency <= 0.0:
            return float(self.step_duration_seconds)

        lambda_i = float(activity_frequency) * self.default_lambda
        if lambda_i <= 0.0:
            return float(self.step_duration_seconds)

        tau = self._rng.expovariate(lambda_i)
        bounded_tau = min(max(0.0, tau), self.step_duration_seconds)
        return float(bounded_tau)

    def sample_arrival_times(
        self,
        agents_or_frequencies: Iterable[Any],
        step_duration: float | None = None,
    ) -> list[float]:
        """Sample continuous arrival delays for a collection of agents or activity frequencies.

        Args:
            agents_or_frequencies: Iterable of frequency scalars or agent-like objects
                exposing 'activity_frequency' or 'activity_level_frequency'.
            step_duration: Optional temporary override for step duration bound in seconds.

        Returns:
            List of sampled delays in seconds.
        """
        max_duration = (
            float(step_duration)
            if step_duration is not None
            else self.step_duration_seconds
        )
        offsets: list[float] = []

        for item in agents_or_frequencies:
            if isinstance(item, (int, float)):
                freq = float(item)
            elif hasattr(item, "activity_frequency"):
                freq = float(item.activity_frequency)
            elif hasattr(item, "activity_level_frequency"):
                freq_attr = item.activity_level_frequency
                freq = float(
                    freq_attr[0] if isinstance(freq_attr, (list, tuple)) else freq_attr
                )
            else:
                freq = 1.0

            raw_offset = self.sample_offset(freq)
            if step_duration is not None and max_duration != self.step_duration_seconds:
                clamped = min(max(0.0, raw_offset), max_duration)
                offsets.append(float(clamped))
            else:
                offsets.append(raw_offset)

        return offsets

    def schedule_actions(
        self,
        step_index: int,
        base_time: datetime | str | float,
        agent_actions: list[tuple[int, dict[str, Any], float]],
        *,
        use_raw_offsets: bool = False,
    ) -> list[ScheduledAction]:
        """Compute exact micro-timestamps and return chronologically sorted ScheduledActions.

        1. Computes the base timestamp for this step:
               T_base = base_time + step_index * step_duration_seconds
        2. For each action (user_id, action_dict, activity_frequency_or_offset):
               Delta t_i = sample_offset(activity_frequency) (or precomputed offset)
               t_i = T_base + Delta t_i
               iso_timestamp = t_i.isoformat() (UTC)
        3. Returns the list of ScheduledActions sorted chronologically by sort_timestamp.

        Args:
            step_index: Integer index of the simulation step (must be >= 0).
            base_time: Base simulation start time (datetime object, ISO string, or POSIX float).
            agent_actions: List of tuples (user_id, action_dict, activity_frequency).
            use_raw_offsets: If True, uses the 3rd tuple element directly as seconds offset.

        Returns:
            List of ScheduledAction instances sorted chronologically.
        """
        if step_index < 0:
            raise ValueError(f"step_index must be non-negative, got {step_index}")

        # Normalize base_time to timezone-aware UTC datetime
        if isinstance(base_time, str):
            base_dt = datetime.fromisoformat(base_time)
        elif isinstance(base_time, (int, float)):
            base_dt = datetime.fromtimestamp(float(base_time), tz=timezone.utc)
        elif isinstance(base_time, datetime):
            base_dt = base_time
        else:
            raise TypeError(
                f"base_time must be datetime, str, or float, got {type(base_time).__name__}"
            )

        if base_dt.tzinfo is None:
            base_time_utc = base_dt.replace(tzinfo=timezone.utc)
        else:
            base_time_utc = base_dt.astimezone(timezone.utc)

        step_offset_seconds = float(step_index) * self.step_duration_seconds
        t_base = base_time_utc + timedelta(seconds=step_offset_seconds)

        scheduled_list: list[ScheduledAction] = []
        for user_id, action_dict, freq_or_offset in agent_actions:
            if use_raw_offsets:
                delta_t = min(
                    max(0.0, float(freq_or_offset)), self.step_duration_seconds
                )
            elif isinstance(action_dict, dict) and "_offset" in action_dict:
                delta_t = min(
                    max(0.0, float(action_dict["_offset"])), self.step_duration_seconds
                )
            elif (
                isinstance(action_dict, dict)
                and "_precomputed_offset" in action_dict
            ):
                delta_t = min(
                    max(0.0, float(action_dict["_precomputed_offset"])),
                    self.step_duration_seconds,
                )
            else:
                delta_t = self.sample_offset(float(freq_or_offset))

            t_action = t_base + timedelta(seconds=delta_t)
            sort_ts = t_action.timestamp()
            iso_ts = t_action.isoformat()

            act_payload = (
                dict(action_dict)
                if isinstance(action_dict, dict)
                else action_dict
            )
            scheduled = ScheduledAction(
                sort_timestamp=sort_ts,
                iso_timestamp=iso_ts,
                user_id=int(user_id),
                action_dict=act_payload,
            )
            scheduled_list.append(scheduled)

        # Sort chronologically (stable sort)
        scheduled_list.sort()
        return scheduled_list


class ChronologicalActionQueue:
    """Thread-safe and async-safe priority queue ordered chronologically by micro-time.

    Provides ordered FIFO dispatching of ScheduledAction objects based on their
    sort_timestamp attribute. Supports:
    - Async coroutines via put/get/join
    - Synchronous OS threads via put_nowait/get_nowait/get_sync
    - Batch pushing and complete chronological draining
    - Direct dispatch to OASIS Channel instances

    Args:
        maxsize: Maximum number of items allowed in the queue (default: 0, unbounded).
    """

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize: int = max(0, int(maxsize))
        self._heap: list[tuple[float, int, ScheduledAction]] = []
        self._counter: int = 0
        self._lock: threading.RLock = threading.RLock()
        self._cond: threading.Condition = threading.Condition(self._lock)
        self._getters: collections.deque[asyncio.Future] = collections.deque()
        self._putters: collections.deque[asyncio.Future] = collections.deque()
        self._join_waiters: collections.deque[asyncio.Future] = collections.deque()
        self._unfinished_tasks: int = 0
        self._finished_event: threading.Event = threading.Event()
        self._finished_event.set()

    @staticmethod
    def _extract_sort_timestamp(item: Any) -> float:
        """Extract continuous numerical timestamp from item."""
        if isinstance(item, ScheduledAction):
            return item.sort_timestamp
        if hasattr(item, "sort_timestamp"):
            return float(item.sort_timestamp)
        if (
            isinstance(item, (tuple, list))
            and len(item) > 0
            and isinstance(item[0], (int, float))
        ):
            return float(item[0])
        raise TypeError(
            f"Item must be ScheduledAction or have sort_timestamp, got {type(item).__name__}"
        )

    @staticmethod
    def _set_result_safe(future: asyncio.Future, result: Any) -> None:
        """Safely set result on an asyncio.Future if not already cancelled or done."""
        if not future.done():
            future.set_result(result)

    def _wakeup_next_getter(self) -> None:
        """Wake up the next waiting getter coroutine with the earliest item.

        Must be called while holding self._lock.
        """
        while self._getters and self._heap:
            getter = self._getters.popleft()
            if not getter.done():
                _, _, earliest = heapq.heappop(self._heap)
                loop = getter.get_loop()
                if loop.is_running() and not loop.is_closed():
                    loop.call_soon_threadsafe(self._set_result_safe, getter, earliest)
                else:
                    # Loop is closed, restore item to heap
                    ts = self._extract_sort_timestamp(earliest)
                    heapq.heappush(self._heap, (ts, self._counter, earliest))
                    self._counter += 1
                break

    def _wakeup_next_putter(self) -> None:
        """Wake up the next waiting putter coroutine when queue has space.

        Must be called while holding self._lock.
        """
        while self._putters and (self.maxsize <= 0 or len(self._heap) < self.maxsize):
            putter = self._putters.popleft()
            if not putter.done():
                loop = putter.get_loop()
                if loop.is_running() and not loop.is_closed():
                    loop.call_soon_threadsafe(self._set_result_safe, putter, None)
                break

    def _put_internal(self, item: Any) -> None:
        """Internal helper to insert an item into the priority heap.

        Must be called while holding self._lock.
        """
        if self.maxsize > 0 and len(self._heap) >= self.maxsize:
            raise QueueFull(
                f"ChronologicalActionQueue reached maxsize={self.maxsize}"
            )

        ts = self._extract_sort_timestamp(item)
        heapq.heappush(self._heap, (ts, self._counter, item))
        self._counter += 1
        self._unfinished_tasks += 1
        self._finished_event.clear()
        self._cond.notify()
        self._wakeup_next_getter()

    def put_nowait(self, item: ScheduledAction) -> None:
        """Insert a ScheduledAction into the queue synchronously without blocking.

        Args:
            item: ScheduledAction to insert.

        Raises:
            QueueFull: If the queue is bounded and currently at maxsize capacity.
        """
        with self._lock:
            self._put_internal(item)

    async def put(self, item: ScheduledAction) -> None:
        """Insert a ScheduledAction into the queue asynchronously, waiting if full.

        Args:
            item: ScheduledAction to insert.
        """
        while True:
            with self._lock:
                if self.maxsize <= 0 or len(self._heap) < self.maxsize:
                    self._put_internal(item)
                    return
                loop = asyncio.get_running_loop()
                putter = loop.create_future()
                self._putters.append(putter)

            try:
                await putter
            except asyncio.CancelledError:
                with self._lock:
                    try:
                        self._putters.remove(putter)
                    except ValueError:
                        pass
                raise

    # Aliases for put/put_nowait
    push = put
    push_nowait = put_nowait

    def push_batch(self, items: Iterable[ScheduledAction]) -> int:
        """Synchronously insert an iterable of ScheduledActions into the queue.

        Args:
            items: Iterable of ScheduledAction items.

        Returns:
            The number of items successfully pushed.
        """
        with self._lock:
            count = 0
            for item in items:
                self._put_internal(item)
                count += 1
            return count

    async def push_batch_async(self, items: Iterable[ScheduledAction]) -> int:
        """Asynchronously insert an iterable of ScheduledActions into the queue.

        Args:
            items: Iterable of ScheduledAction items.

        Returns:
            The number of items successfully pushed.
        """
        count = 0
        for item in items:
            await self.put(item)
            count += 1
        return count

    def get_nowait(self) -> ScheduledAction:
        """Retrieve and remove the earliest action in chronological order non-blockingly.

        Returns:
            ScheduledAction with the earliest sort_timestamp.

        Raises:
            QueueEmpty: If the queue contains no items.
        """
        with self._lock:
            if not self._heap:
                raise QueueEmpty("ChronologicalActionQueue is empty")
            _, _, item = heapq.heappop(self._heap)
            self._wakeup_next_putter()
            return item

    async def get(self) -> ScheduledAction:
        """Retrieve and remove the earliest action chronologically, awaiting if empty.

        Returns:
            ScheduledAction with the earliest sort_timestamp.
        """
        with self._lock:
            if self._heap:
                _, _, item = heapq.heappop(self._heap)
                self._wakeup_next_putter()
                return item

            loop = asyncio.get_running_loop()
            getter = loop.create_future()
            self._getters.append(getter)

        try:
            return await getter
        except asyncio.CancelledError:
            with self._lock:
                try:
                    self._getters.remove(getter)
                except ValueError:
                    pass
                if getter.done() and not getter.cancelled():
                    # Item was assigned just before cancellation was acknowledged
                    item = getter.result()
                    ts = self._extract_sort_timestamp(item)
                    heapq.heappush(self._heap, (ts, self._counter, item))
                    self._counter += 1
                    self._wakeup_next_getter()
            raise

    # Aliases for get/get_nowait
    pop = get
    pop_nowait = get_nowait

    def get_sync(self, timeout: float | None = None) -> ScheduledAction:
        """Thread-safe blocking dequeue with optional timeout.

        Args:
            timeout: Maximum seconds to wait. None waits indefinitely.

        Returns:
            ScheduledAction with earliest sort_timestamp.

        Raises:
            QueueEmpty: If timeout expires before an action becomes available.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._heap:
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise QueueEmpty(
                            "ChronologicalActionQueue is empty (timed out)"
                        )
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()

            _, _, item = heapq.heappop(self._heap)
            self._wakeup_next_putter()
            return item

    pop_sync = get_sync

    def peek(self) -> ScheduledAction | None:
        """Inspect the earliest scheduled action without removing it from the queue.

        Returns:
            The earliest ScheduledAction, or None if the queue is empty.
        """
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][2]

    def drain(self) -> list[ScheduledAction]:
        """Drain and return all actions currently in the queue in chronological order.

        Returns:
            List of ScheduledActions ordered by sort_timestamp ascending.
        """
        with self._lock:
            actions: list[ScheduledAction] = []
            while self._heap:
                _, _, item = heapq.heappop(self._heap)
                actions.append(item)
            self._wakeup_next_putter()
            return actions

    async def drain_to_channel(
        self,
        channel: Any,
        formatter: Callable[[ScheduledAction], Any] | None = None,
    ) -> int:
        """Drain all actions in chronological order and dispatch them to the OASIS Channel.

        Args:
            channel: Target OASIS Channel instance or async message sink.
            formatter: Optional callback to format ScheduledAction into custom channel payload.
                Defaults to (action.user_id, action.action_dict, action.iso_timestamp).

        Returns:
            Count of actions successfully dispatched.
        """
        actions = self.drain()
        for item in actions:
            if formatter is not None:
                payload = formatter(item)
            else:
                payload = (item.user_id, item.action_dict, item.iso_timestamp)

            if hasattr(channel, "write_to_receive_queue"):
                await channel.write_to_receive_queue(payload)
            elif hasattr(channel, "put"):
                await channel.put(payload)
            elif hasattr(channel, "send_to"):
                await channel.send_to(payload)
            else:
                raise TypeError(
                    f"Unsupported channel object type: {type(channel).__name__}"
                )
        return len(actions)

    def task_done(self) -> None:
        """Indicate that a formerly enqueued action has completed processing.

        Raises:
            ValueError: If task_done() is called more times than items were enqueued.
        """
        with self._lock:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._finished_event.set()
                while self._join_waiters:
                    waiter = self._join_waiters.popleft()
                    if not waiter.done():
                        loop = waiter.get_loop()
                        if loop.is_running() and not loop.is_closed():
                            loop.call_soon_threadsafe(
                                self._set_result_safe, waiter, None
                            )

    async def join(self) -> None:
        """Block asynchronously until all items in the queue have been processed."""
        with self._lock:
            if self._unfinished_tasks == 0:
                return
            loop = asyncio.get_running_loop()
            waiter = loop.create_future()
            self._join_waiters.append(waiter)

        await waiter

    def join_sync(self, timeout: float | None = None) -> bool:
        """Block synchronous thread until all items in the queue have been processed.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if all tasks were completed, False if timed out.
        """
        return self._finished_event.wait(timeout=timeout)

    def qsize(self) -> int:
        """Return the current number of items in the queue."""
        with self._lock:
            return len(self._heap)

    def empty(self) -> bool:
        """Return True if the queue is currently empty, False otherwise."""
        with self._lock:
            return len(self._heap) == 0

    def full(self) -> bool:
        """Return True if the queue has reached maxsize capacity, False otherwise."""
        with self._lock:
            return self.maxsize > 0 and len(self._heap) >= self.maxsize

    def clear(self) -> None:
        """Remove all items from the queue."""
        with self._lock:
            self._heap.clear()
            self._wakeup_next_putter()

    def __len__(self) -> int:
        """Return current queue size via len(queue)."""
        return self.qsize()

    def __bool__(self) -> bool:
        """Evaluate True if queue has items, False if empty."""
        return not self.empty()

    def __repr__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"ChronologicalActionQueue(qsize={self.qsize()}, maxsize={self.maxsize})"
        )
