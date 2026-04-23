"""Integration test for the atomic claim primitive.

Spins up a temp SQLite DB with N fake jobs, launches K in-process
"workers" in threads, and asserts each shard is claimed exactly once
(no double-processing, no missed shards).

Run with:
    python -m execution.opus_queue.tests.test_concurrent_claim
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from execution.opus_queue import db as queue_db
from execution.opus_queue.tests._tmp import workspace_temp_dir


def _seed_jobs(conn, n: int, model: str) -> None:
    for i in range(n):
        conn.execute(
            """
            INSERT OR IGNORE INTO directions
                (direction_key, model, src_lang, tgt_lang,
                 n_sentences, shard_size, est_hours, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))
            """,
            (f"dir_{i:04d}-tgt", model, f"dir_{i:04d}", "tgt", 100, 50, 1.0),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO jobs
                (direction_key, model, shard_id, start_idx, end_idx, status, attempts)
            VALUES (?, ?, 0, 0, 50, 'pending', 0)
            """,
            (f"dir_{i:04d}-tgt", model),
        )


def _worker(db_path: str, model: str, worker_id: str,
            seen: list[tuple[str, int]], lock: threading.Lock) -> None:
    conn = queue_db.connect(db_path)
    try:
        while True:
            job = queue_db.claim_next(conn, model, worker_id, retries=10)
            if job is None:
                return
            time.sleep(0.001)  # simulate a tiny bit of work
            result = queue_db.mark_done(
                conn,
                job["direction_key"],
                job["model"],
                job["shard_id"],
                worker_id,
                "/tmp/x",
            )
            assert result == "done", f"unexpected finalize result: {result}"
            with lock:
                seen.append((job["direction_key"], job["shard_id"]))
    finally:
        conn.close()


def run_test(n_jobs: int = 50, n_workers: int = 4) -> None:
    with workspace_temp_dir("concurrent_claim") as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = queue_db.connect(db_path)
        queue_db.initialize(conn)
        _seed_jobs(conn, n_jobs, model="fake_model")
        conn.close()

        seen: list[tuple[str, int]] = []
        lock = threading.Lock()
        threads = [
            threading.Thread(
                target=_worker,
                args=(db_path, "fake_model", f"w{i}", seen, lock),
            )
            for i in range(n_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(seen) == n_jobs, f"expected {n_jobs} claims, got {len(seen)}"
        assert len(set(seen)) == n_jobs, "duplicate shards claimed"

        conn = queue_db.connect(db_path)
        try:
            counts = queue_db.count_by_status(conn, model="fake_model")
        finally:
            conn.close()
        assert counts.get("done", 0) == n_jobs, f"not all done: {counts}"
        assert counts.get("pending", 0) == 0, f"leftover pending: {counts}"
        print(f"OK: {n_jobs} jobs claimed exactly once by {n_workers} workers.")


if __name__ == "__main__":
    run_test()
