"""Regression tests for worker-owned shard finalization semantics.

Run with:
    python -m execution.opus_queue.tests.test_worker_state_transitions
"""

from __future__ import annotations

from pathlib import Path

from execution.opus_queue import db as queue_db
from execution.opus_queue.tests._tmp import workspace_temp_dir


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
        SELECT status, worker_id, attempts, started_at, finished_at, out_path, last_error,
               claim_gpu_count, gpu_seconds_total
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
    with workspace_temp_dir("worker_state") as tmp:
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
    with workspace_temp_dir("worker_state") as tmp:
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
    with workspace_temp_dir("worker_state") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)
            first = queue_db.claim_next(conn, "metricx24", "w1")
            assert first is not None
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 120
                 WHERE direction_key='eng_Latn-fra_Latn' AND model='metricx24' AND shard_id=0
                """
            )
            assert queue_db.mark_failed(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", "boom-1", 2
            ) == "requeued"
            state = _job_state(conn)
            assert state["status"] == "pending"
            assert state["worker_id"] is None
            assert state["started_at"] is None
            assert state["finished_at"] is None
            assert state["claim_gpu_count"] == 1
            assert state["last_error"] == "boom-1"
            assert 115 <= float(state["gpu_seconds_total"]) <= 125

            second = queue_db.claim_next(conn, "metricx24", "w2")
            assert second is not None
            assert int(second["attempts"]) == 2
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 30
                 WHERE direction_key='eng_Latn-fra_Latn' AND model='metricx24' AND shard_id=0
                """
            )
            assert queue_db.mark_failed(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w2", "boom-2", 2
            ) == "failed"
            state = _job_state(conn)
            assert state["status"] == "failed"
            assert state["worker_id"] is None
            assert state["started_at"] is not None
            assert state["finished_at"] is not None
            assert state["last_error"] == "boom-2"
            assert 145 <= float(state["gpu_seconds_total"]) <= 155
        finally:
            conn.close()


def test_mark_failed_preserves_traceback_tail_when_trimmed() -> None:
    with workspace_temp_dir("worker_state") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)
            claimed = queue_db.claim_next(conn, "metricx24", "w1")
            assert claimed is not None

            long_error = "Traceback head\n" + ("x" * 1200) + "\nRuntimeError: boom-tail"
            assert queue_db.mark_failed(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", long_error, 2
            ) == "requeued"

            state = _job_state(conn)
            assert state["last_error"] is not None
            assert state["last_error"].startswith("Traceback head")
            assert "\n...\n" in state["last_error"]
            assert state["last_error"].endswith("RuntimeError: boom-tail")
            assert len(state["last_error"]) <= 1000
        finally:
            conn.close()


def run_test() -> None:
    test_stale_worker_cannot_requeue_done_row()
    test_stale_worker_cannot_mark_done_after_requeue()
    test_mark_failed_requeues_then_fails_at_threshold()
    test_mark_failed_preserves_traceback_tail_when_trimmed()
    print("OK: worker-owned finalization prevents stale workers from clobbering shard state.")


if __name__ == "__main__":
    run_test()
