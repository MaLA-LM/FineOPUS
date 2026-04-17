"""Regression tests for stale-row reaping.

Run with:
    python -m execution.opus_queue.tests.test_reaper
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from execution.opus_queue import queue_db, reaper


def _seed_running_job(
    conn,
    *,
    direction_key: str,
    model: str,
    started_at: int,
    worker_id: str,
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
            (direction_key, model, shard_id, start_idx, end_idx, status, worker_id, started_at, attempts)
        VALUES (?, ?, ?, 0, 50, 'running', ?, ?, 1)
        """,
        (direction_key, model, shard_id, worker_id, started_at),
    )


def _seed_failed_job(
    conn,
    *,
    direction_key: str,
    model: str,
    finished_at: int,
    last_error: str,
    shard_id: int = 0,
    attempts: int = 3,
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
            (direction_key, model, shard_id, start_idx, end_idx, status, finished_at, attempts, last_error)
        VALUES (?, ?, ?, 0, 50, 'failed', ?, ?, ?)
        """,
        (direction_key, model, shard_id, finished_at, attempts, last_error),
    )


def run_test() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            now_ts = int(
                conn.execute(
                    "SELECT CAST(strftime('%s','now') AS INTEGER) AS ts"
                ).fetchone()["ts"]
            )
            _seed_running_job(
                conn,
                direction_key="eng_Latn-fra_Latn",
                model="metricx24",
                started_at=now_ts - 60,
                worker_id="metricx-old",
            )
            _seed_running_job(
                conn,
                direction_key="eng_Latn-deu_Latn",
                model="metricx24",
                started_at=now_ts - 10,
                worker_id="metricx-fresh",
                shard_id=1,
            )
            _seed_running_job(
                conn,
                direction_key="eng_Latn-spa_Latn",
                model="qwen3-4b-instruct-2507",
                started_at=now_ts - 400,
                worker_id="qwen-old",
            )
            _seed_running_job(
                conn,
                direction_key="eng_Latn-ita_Latn",
                model="qwen3-4b-instruct-2507",
                started_at=now_ts - 60,
                worker_id="qwen-fresh",
                shard_id=1,
            )
            _seed_failed_job(
                conn,
                direction_key="eng_Latn-por_Latn",
                model="metricx24",
                finished_at=now_ts - 120,
                last_error="metricx boom",
                shard_id=2,
            )
            _seed_failed_job(
                conn,
                direction_key="eng_Latn-nld_Latn",
                model="qwen3-4b-instruct-2507",
                finished_at=now_ts - 120,
                last_error="qwen boom",
                shard_id=2,
            )

            reclaimed = queue_db.reset_stale_rows(
                conn, {"metricx24": 30, "qwen3-4b-instruct-2507": 300}
            )
            reclaimed_keys = {
                (row["direction_key"], row["model"], row["shard_id"]) for row in reclaimed
            }
            assert reclaimed_keys == {
                ("eng_Latn-fra_Latn", "metricx24", 0),
                ("eng_Latn-spa_Latn", "qwen3-4b-instruct-2507", 0),
            }

            rows = conn.execute(
                """
                SELECT direction_key, model, shard_id, status, worker_id, last_error
                  FROM jobs
                 ORDER BY model, shard_id
                """
            ).fetchall()
            states = {
                (row["direction_key"], row["model"], row["shard_id"]): dict(row)
                for row in rows
            }
            assert states[("eng_Latn-fra_Latn", "metricx24", 0)]["status"] == "pending"
            assert states[("eng_Latn-fra_Latn", "metricx24", 0)]["worker_id"] is None
            assert states[("eng_Latn-fra_Latn", "metricx24", 0)]["last_error"] == "reaped"

            assert states[("eng_Latn-deu_Latn", "metricx24", 1)]["status"] == "running"
            assert states[("eng_Latn-deu_Latn", "metricx24", 1)]["worker_id"] == "metricx-fresh"
            assert states[("eng_Latn-por_Latn", "metricx24", 2)]["status"] == "failed"
            assert states[("eng_Latn-por_Latn", "metricx24", 2)]["last_error"] == "metricx boom"

            assert states[("eng_Latn-spa_Latn", "qwen3-4b-instruct-2507", 0)]["status"] == "pending"
            assert states[("eng_Latn-ita_Latn", "qwen3-4b-instruct-2507", 1)]["status"] == "running"
            assert states[("eng_Latn-nld_Latn", "qwen3-4b-instruct-2507", 2)]["status"] == "failed"

            revived = queue_db.reset_stale_rows(
                conn,
                {"metricx24": 30, "qwen3-4b-instruct-2507": 300},
                reset_failed=True,
            )
            revived_keys = {
                (row["direction_key"], row["model"], row["shard_id"], row["previous_status"])
                for row in revived
            }
            assert revived_keys == {
                ("eng_Latn-por_Latn", "metricx24", 2, "failed"),
            }

            rows = conn.execute(
                """
                SELECT direction_key, model, shard_id, status, attempts, finished_at, last_error
                  FROM jobs
                 WHERE shard_id = 2
                 ORDER BY model, shard_id
                """
            ).fetchall()
            failed_states = {
                (row["direction_key"], row["model"], row["shard_id"]): dict(row)
                for row in rows
            }
            assert failed_states[("eng_Latn-por_Latn", "metricx24", 2)]["status"] == "pending"
            assert failed_states[("eng_Latn-por_Latn", "metricx24", 2)]["attempts"] == 0
            assert failed_states[("eng_Latn-por_Latn", "metricx24", 2)]["finished_at"] is None
            assert failed_states[("eng_Latn-por_Latn", "metricx24", 2)]["last_error"] == "metricx boom"
            assert failed_states[("eng_Latn-nld_Latn", "qwen3-4b-instruct-2507", 2)]["status"] == "failed"
        finally:
            conn.close()

    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            now_ts = int(
                conn.execute(
                    "SELECT CAST(strftime('%s','now') AS INTEGER) AS ts"
                ).fetchone()["ts"]
            )
            _seed_failed_job(
                conn,
                direction_key="eng_Latn-swe_Latn",
                model="unknown-model",
                finished_at=now_ts - 120,
                last_error="transient infra issue",
            )
        finally:
            conn.close()

        reclaimed = reaper.run_once(
            argparse.Namespace(
                db=str(db_path),
                interval=0,
                timeout_multiplier=1.0,
                default_timeout_seconds=30,
                reset_failed=True,
            )
        )
        assert reclaimed == 1

        conn = queue_db.connect(db_path)
        try:
            row = conn.execute(
                """
                SELECT status, attempts, finished_at, last_error
                  FROM jobs
                 WHERE direction_key='eng_Latn-swe_Latn'
                   AND model='unknown-model'
                   AND shard_id=0
                """
            ).fetchone()
            assert row is not None
            state = dict(row)
            assert state["status"] == "pending"
            assert state["attempts"] == 0
            assert state["finished_at"] is None
            assert state["last_error"] == "transient infra issue"
        finally:
            conn.close()
    print("OK: reaper reclaims stale running rows by default and can opt-in to revive failed rows.")


if __name__ == "__main__":
    run_test()
