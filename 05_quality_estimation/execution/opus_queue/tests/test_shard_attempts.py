"""Regression tests for cumulative GPU accounting on `jobs`.

Each claim/finalize cycle must preserve queue semantics while accumulating
consumed GPU seconds on the owning shard row.

Run with:
    python -m execution.opus_queue.tests.test_shard_attempts
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
    shard_id: int = 0,
) -> None:
    src_lang, tgt_lang = direction_key.split("-", 1)
    conn.execute(
        """
        INSERT OR IGNORE INTO directions
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
        VALUES (?, ?, ?, 0, 50, 'pending', 0)
        """,
        (direction_key, model, shard_id),
    )


def _job_rows(conn, model: str = "metricx24") -> list[dict]:
    rows = conn.execute(
        """
        SELECT direction_key, model, shard_id, status, worker_id,
               started_at, finished_at, attempts, claim_gpu_count,
               gpu_seconds_total, out_path, last_error
          FROM jobs
         WHERE model = ?
         ORDER BY shard_id ASC
        """,
        (model,),
    ).fetchall()
    return [dict(row) for row in rows]


def test_done_accumulates_gpu_seconds() -> None:
    with workspace_temp_dir("gpu_accounting") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)
            claimed = queue_db.claim_next(
                conn,
                "metricx24",
                "w1",
                slurm_job_id="12345",
                slurm_array_task_id="0",
                node_host="node01",
                gpu_count=2,
            )
            assert claimed is not None
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 120
                 WHERE direction_key='eng_Latn-fra_Latn' AND model='metricx24' AND shard_id=0
                """
            )

            assert queue_db.mark_done(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", "/tmp/w1.jsonl"
            ) == "done"

            rows = _job_rows(conn)
            assert len(rows) == 1
            assert rows[0]["status"] == "done"
            assert rows[0]["worker_id"] == "w1"
            assert rows[0]["finished_at"] is not None
            assert rows[0]["claim_gpu_count"] == 2
            assert rows[0]["out_path"] == "/tmp/w1.jsonl"
            assert 235 <= float(rows[0]["gpu_seconds_total"]) <= 245
        finally:
            conn.close()


def test_requeued_then_done_accumulates_both_attempts() -> None:
    with workspace_temp_dir("gpu_accounting") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)

            assert queue_db.claim_next(conn, "metricx24", "w1", gpu_count=1) is not None
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 60
                 WHERE direction_key='eng_Latn-fra_Latn' AND model='metricx24' AND shard_id=0
                """
            )
            assert queue_db.mark_failed(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", "boom-1", 3
            ) == "requeued"

            assert queue_db.claim_next(conn, "metricx24", "w2", gpu_count=4) is not None
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 30
                 WHERE direction_key='eng_Latn-fra_Latn' AND model='metricx24' AND shard_id=0
                """
            )
            assert queue_db.mark_done(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w2", "/tmp/w2.jsonl"
            ) == "done"

            rows = _job_rows(conn)
            assert len(rows) == 1
            assert rows[0]["attempts"] == 2
            assert rows[0]["status"] == "done"
            assert rows[0]["worker_id"] == "w2"
            assert rows[0]["claim_gpu_count"] == 4
            assert 175 <= float(rows[0]["gpu_seconds_total"]) <= 185
        finally:
            conn.close()


def test_failed_at_threshold_accumulates_gpu_seconds() -> None:
    with workspace_temp_dir("gpu_accounting") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)
            assert queue_db.claim_next(conn, "metricx24", "w1", gpu_count=3) is not None
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 20
                 WHERE direction_key='eng_Latn-fra_Latn' AND model='metricx24' AND shard_id=0
                """
            )
            assert queue_db.mark_failed(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", "boom-1", 1
            ) == "failed"
            rows = _job_rows(conn)
            assert len(rows) == 1
            assert rows[0]["status"] == "failed"
            assert rows[0]["worker_id"] is None
            assert rows[0]["finished_at"] is not None
            assert rows[0]["claim_gpu_count"] == 3
            assert 55 <= float(rows[0]["gpu_seconds_total"]) <= 65
        finally:
            conn.close()


def test_reset_own_stale_accumulates_open_gpu_seconds() -> None:
    with workspace_temp_dir("gpu_accounting") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn, shard_id=0)
            _seed_job(conn, shard_id=1)
            assert queue_db.claim_next(conn, "metricx24", "w1", gpu_count=2) is not None
            assert queue_db.claim_next(conn, "metricx24", "w1", gpu_count=2) is not None
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 50
                 WHERE model='metricx24' AND worker_id='w1'
                """
            )

            count = queue_db.reset_own_stale(conn, "w1")
            assert count == 2

            rows = _job_rows(conn)
            assert len(rows) == 2
            assert all(row["status"] == "pending" for row in rows)
            assert all(row["worker_id"] is None for row in rows)
            assert all(row["started_at"] is None for row in rows)
            assert all(row["claim_gpu_count"] == 1 for row in rows)
            assert all(row["last_error"] == "reset_own_stale" for row in rows)
            total_gpu_seconds = sum(float(row["gpu_seconds_total"]) for row in rows)
            assert 195 <= total_gpu_seconds <= 205
        finally:
            conn.close()


def test_reaper_accumulates_open_gpu_seconds() -> None:
    with workspace_temp_dir("gpu_accounting") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn)

            assert queue_db.claim_next(conn, "metricx24", "w1", gpu_count=4) is not None
            conn.execute(
                """
                UPDATE jobs
                   SET started_at = started_at - 600
                 WHERE direction_key='eng_Latn-fra_Latn' AND model='metricx24' AND shard_id=0
                """
            )

            reclaimed = queue_db.reset_stale_rows(conn, {"metricx24": 30})
            assert len(reclaimed) == 1

            rows = _job_rows(conn)
            assert len(rows) == 1
            assert rows[0]["status"] == "pending"
            assert rows[0]["worker_id"] is None
            assert rows[0]["started_at"] is None
            assert rows[0]["claim_gpu_count"] == 1
            assert rows[0]["last_error"] == "reaped"
            assert 2395 <= float(rows[0]["gpu_seconds_total"]) <= 2405
        finally:
            conn.close()


def test_gpu_hour_aggregation() -> None:
    with workspace_temp_dir("gpu_accounting") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(conn, shard_id=0)
            _seed_job(conn, shard_id=1)

            assert queue_db.claim_next(
                conn, "metricx24", "w1", gpu_count=1
            ) is not None
            assert queue_db.claim_next(
                conn, "metricx24", "w2", gpu_count=4
            ) is not None

            conn.execute(
                """
                UPDATE jobs
                   SET started_at = CASE shard_id
                       WHEN 0 THEN started_at - 120
                       WHEN 1 THEN started_at - 120
                       ELSE started_at
                   END
                 WHERE model='metricx24'
                """
            )
            assert queue_db.mark_done(
                conn, "eng_Latn-fra_Latn", "metricx24", 0, "w1", "/tmp/a"
            ) == "done"
            assert queue_db.mark_done(
                conn, "eng_Latn-fra_Latn", "metricx24", 1, "w2", "/tmp/b"
            ) == "done"

            row = conn.execute(
                """
                SELECT SUM(gpu_seconds_total) AS gpu_seconds
                  FROM jobs
                 WHERE status = 'done'
                """
            ).fetchone()
            gpu_seconds = float(row["gpu_seconds"] or 0.0)
            assert 595 <= gpu_seconds <= 605, gpu_seconds
        finally:
            conn.close()


def run_test() -> None:
    test_done_accumulates_gpu_seconds()
    test_requeued_then_done_accumulates_both_attempts()
    test_failed_at_threshold_accumulates_gpu_seconds()
    test_reset_own_stale_accumulates_open_gpu_seconds()
    test_reaper_accumulates_open_gpu_seconds()
    test_gpu_hour_aggregation()
    print("OK: cumulative GPU accounting is preserved across done, retry, reset, and reap flows.")


if __name__ == "__main__":
    run_test()
