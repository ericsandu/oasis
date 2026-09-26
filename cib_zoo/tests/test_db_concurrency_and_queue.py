"""Tests for SQLite concurrency, WAL mode resilience, and sequential write queue processing."""

import asyncio
import sqlite3
import threading
import time
import pytest
from oasis.social_platform.channel import Channel


def _init_wal_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA busy_timeout = 5000")
    cur.execute("""
        CREATE TABLE actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id INTEGER,
            action_type TEXT,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_channel_queue_sequential_writes(tmp_path):
    """Verify that multiple concurrent agents queuing writes to Channel process sequentially without DB locks."""
    db_file = tmp_path / "queue_test.db"
    _init_wal_db(str(db_file))

    channel = Channel()
    num_agents = 25
    actions_per_agent = 10
    total_actions = num_agents * actions_per_agent

    # Single-writer consumer loop simulating platform database writer
    async def writer_consumer():
        conn = sqlite3.connect(str(db_file))
        conn.execute("PRAGMA busy_timeout = 5000")
        processed = 0
        try:
            while processed < total_actions:
                message_id, data = await channel.receive_from()
                agent_id, action_type, payload = data
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO actions (agent_id, action_type, payload) VALUES (?, ?, ?)",
                    (agent_id, action_type, payload),
                )
                conn.commit()
                await channel.send_to((message_id, {"status": "ok"}))
                processed += 1
        finally:
            conn.close()

    # Agent producer task
    async def agent_worker(agent_id: int):
        for i in range(actions_per_agent):
            msg_id = await channel.write_to_receive_queue(
                (agent_id, "LIKE_POST", f"post_{i}")
            )
            # Await confirmation
            while True:
                resp = await channel.send_dict.get(msg_id)
                if resp is not None:
                    break
                await asyncio.sleep(0.001)

    # Launch consumer and all concurrent agent producers
    consumer_task = asyncio.create_task(writer_consumer())
    producer_tasks = [asyncio.create_task(agent_worker(a_id)) for a_id in range(num_agents)]

    await asyncio.gather(*producer_tasks)
    await consumer_task

    # Verify total rows written
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM actions")
    count = cur.fetchone()[0]
    conn.close()

    assert count == total_actions, f"Expected {total_actions} rows, found {count}"


def test_concurrent_readers_during_active_writes(tmp_path):
    """Verify that concurrent reader threads do not lock or crash during active database write transactions under WAL mode."""
    db_file = tmp_path / "readers_test.db"
    _init_wal_db(str(db_file))

    stop_event = threading.Event()
    read_errors = []
    successful_reads = [0]

    def reader_loop():
        conn = sqlite3.connect(str(db_file))
        conn.execute("PRAGMA busy_timeout = 5000")
        cur = conn.cursor()
        while not stop_event.is_set():
            try:
                cur.execute("SELECT COUNT(*) FROM actions")
                _ = cur.fetchone()[0]
                successful_reads[0] += 1
                time.sleep(0.002)
            except Exception as e:
                read_errors.append(e)
        conn.close()

    # Start 3 background reader threads
    readers = [threading.Thread(target=reader_loop) for _ in range(3)]
    for r in readers:
        r.start()

    # Main thread performs rapid batch writes
    conn_writer = sqlite3.connect(str(db_file))
    conn_writer.execute("PRAGMA busy_timeout = 5000")
    cur_writer = conn_writer.cursor()

    try:
        for batch in range(20):
            with conn_writer:
                for i in range(25):
                    cur_writer.execute(
                        "INSERT INTO actions (agent_id, action_type, payload) VALUES (?, ?, ?)",
                        (batch, "POST", f"payload_{i}"),
                    )
            time.sleep(0.005)
    finally:
        stop_event.set()
        for r in readers:
            r.join()
        conn_writer.close()

    assert len(read_errors) == 0, f"Encountered reader errors under WAL: {read_errors}"
    assert successful_reads[0] > 10, "Readers should have executed multiple concurrent read queries"
