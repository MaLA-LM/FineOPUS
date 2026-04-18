"""Regression tests for worker-owned shard finalization semantics.

Run with:
    python -m execution.opus_queue.tests.test_worker_state_transitions
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from execution.opus_queue import db as queue_db


def _seed_job(
    conn,
    *,
    direction_key: str = "eng_Latn-fra_Latn",
    model: str = "metricx24",
) -> None:
    src_lang, tgt_lang = direction_key.split("-", 1)
    conn.execute(
        """
        INSERT INTO directions
            (direction_key, model, src_lang, tgt_lang,
             n_sentences, shard_size, est_hours, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))
        """,
        (direction_key, model, src_lang, tgt_lang, 100, 50, 1.0),
    )
    conn.execute(
        """
        INSERT INTO jobs
            (direction_key, model, shard_id, start_idx, end_idx, status, attempts)
        VALUES (?, ?, 0, 0, 50, 'pending', 0)
        """,
        (direction_key, model),
    )


def _job_state(conn, direction_key: str = "eng_Latn-fra_Latn", model: str = "metricx24") -> dict:
    row = conn.execute(
        """
        SELECT status, worker_id, attempts, started_at, finished_at, out_path, last_error
          FROM jobs
         WHERE direction_key=? AND model=? AND shard_id=0
        """,
        (direction_key, model),
    ).fetchone()
    assert row is not None
    return dict(row)


def _requeue_job(conn, direction_key: str = "eng_Latn-fra_Latn", model: str = "metricx24") -> None:
    conn.execute(
        """
        UPDATE jobs
           SET status='pending', worker_id=NULL, started_at=NULL
         WHERE direction_key=? AND model=? AND shard_id=0
        """,
        (direction_key, model),
    )


def test_stale_worker_cannot_requeue_done_row() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)
            first = queue_db.claim_next(conn, "metricx24", "w1")
            assert first is not None
            _requeue_job(conn)
            second = queue_db.claim_next(conn, "metricx24", "w2")
            assert second is not None
            assert queue_db.mark_done(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w2", "/tmp/w2.jsonl"
            ) == "done"
            assert queue_db.mark_failed(
                conn,
                "eng_Latn-fra_Latn",
                "metricx24",
                0,
                "w1",
                "late failure",
                3,
            ) == "stale_noop"
            state = _job_state(conn)
            assert state["status"] == "done"
            assert state["worker_id"] == "w2"
            assert state["out_path"] == "/tmp/w2.jsonl"
            assert state["last_error"] is None
        finally:
            conn.close()


def test_stale_worker_cannot_mark_done_after_requeue() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)
            first = queue_db.claim_next(conn, "metricx24", "w1")
            assert first is not None
            _requeue_job(conn)
            second = queue_db.claim_next(conn, "metricx24", "w2")
            assert second is not None
            assert queue_db.mark_done(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", "/tmp/w1.jsonl"
            ) == "stale_noop"
            mid_state = _job_state(conn)
            assert mid_state["status"] == "running"
            assert mid_state["worker_id"] == "w2"
            assert mid_state["out_path"] is None

            assert queue_db.mark_done(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w2", "/tmp/w2.jsonl"
            ) == "done"
            final_state = _job_state(conn)
            assert final_state["status"] == "done"
            assert final_state["out_path"] == "/tmp/w2.jsonl"
        finally:
            conn.close()


def test_mark_failed_requeues_then_fails_at_threshold() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)
            first = queue_db.claim_next(conn, "metricx24", "w1")
            assert first is not None
            assert queue_db.mark_failed(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", "boom-1", 2
            ) == "requeued"
            state = _job_state(conn)
            assert state["status"] == "pending"
            assert state["worker_id"] is None
            assert state["finished_at"] is None
            assert state["last_error"] == "boom-1"

            second = queue_db.claim_next(conn, "metricx24", "w2")
            assert second is not None
            assert int(second["attempts"]) == 2
            assert queue_db.mark_failed(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w2", "boom-2", 2
            ) == "failed"
            state = _job_state(conn)
            assert state["status"] == "failed"
            assert state["worker_id"] is None
            assert state["finished_at"] is not None
            assert state["last_error"] == "boom-2"
        finally:
            conn.close()


def run_test() -> None:
    test_stale_worker_cannot_requeue_done_row()
    test_stale_worker_cannot_mark_done_after_requeue()
    test_mark_failed_requeues_then_fails_at_threshold()
    print("OK: worker-owned finalization prevents stale workers from clobbering shard state.")


if __name__ == "__main__":
    run_test()
