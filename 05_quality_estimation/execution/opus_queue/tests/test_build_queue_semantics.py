"""Regression tests for build_queue reset/reassign semantics.

Run with:
    python -m execution.opus_queue.tests.test_build_queue_semantics
"""

from __future__ import annotations

from pathlib import Path

from execution.opus_queue import db as queue_db
from execution.opus_queue.db import writes as queue_ops
from execution.opus_queue.tests._tmp import workspace_temp_dir


def _seed_job(conn, *, direction_key: str, model: str, shard_id: int, status: str) -> None:
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
        VALUES (?, ?, ?, ?, ?, ?, 1)
        """,
        (direction_key, model, shard_id, shard_id * 10, shard_id * 10 + 10, status),
    )


def _count_rows(conn, table: str, *, direction_key: str | None = None, model: str | None = None) -> int:
    where = []
    params: list[str] = []
    if direction_key is not None:
        where.append("direction_key=?")
        params.append(direction_key)
    if model is not None:
        where.append("model=?")
        params.append(model)
    sql = f"SELECT COUNT(*) AS n FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    row = conn.execute(sql, tuple(params)).fetchone()
    return int(row["n"])


def test_reset_pending_for_model_blocks_done_without_force() -> None:
    with workspace_temp_dir("build_queue") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(
                conn,
                direction_key="eng_Latn-fra_Latn",
                model="metricx24",
                shard_id=0,
                status="done",
            )
            _seed_job(
                conn,
                direction_key="eng_Latn-fra_Latn",
                model="metricx24",
                shard_id=1,
                status="pending",
            )
            try:
                queue_ops.reset_pending_for_model(conn, "metricx24", force=False)
            except SystemExit:
                pass
            else:
                raise AssertionError("expected SystemExit when done rows exist without --force")
            assert _count_rows(conn, "jobs", model="metricx24") == 2
            assert _count_rows(conn, "directions", model="metricx24") == 1
        finally:
            conn.close()


def test_reset_pending_for_model_force_drops_done_rows() -> None:
    with workspace_temp_dir("build_queue") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(
                conn,
                direction_key="eng_Latn-deu_Latn",
                model="metricx24",
                shard_id=0,
                status="done",
            )
            queue_ops.reset_pending_for_model(conn, "metricx24", force=True)
            assert _count_rows(conn, "jobs", model="metricx24") == 0
            assert _count_rows(conn, "directions", model="metricx24") == 0
        finally:
            conn.close()


def test_reset_pending_for_model_without_done_deletes_non_done_rows() -> None:
    with workspace_temp_dir("build_queue") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(
                conn,
                direction_key="eng_Latn-spa_Latn",
                model="xcomet-xl",
                shard_id=0,
                status="pending",
            )
            _seed_job(
                conn,
                direction_key="eng_Latn-spa_Latn",
                model="xcomet-xl",
                shard_id=1,
                status="failed",
            )
            queue_ops.reset_pending_for_model(conn, "xcomet-xl", force=False)
            assert _count_rows(conn, "jobs", model="xcomet-xl") == 0
            assert _count_rows(conn, "directions", model="xcomet-xl") == 0
        finally:
            conn.close()


def test_reassign_preserves_done_rows_without_force_and_drops_with_force() -> None:
    with workspace_temp_dir("build_queue") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(
                conn,
                direction_key="eng_Latn-ita_Latn",
                model="metricx24",
                shard_id=0,
                status="done",
            )
            _seed_job(
                conn,
                direction_key="eng_Latn-ita_Latn",
                model="metricx24",
                shard_id=1,
                status="pending",
            )
            assert (
                queue_ops.delete_pending_for_pair(
                    conn, "eng_Latn-ita_Latn", "metricx24", force=False
                )
                is True
            )
            assert _count_rows(conn, "jobs", direction_key="eng_Latn-ita_Latn", model="metricx24") == 2
            assert (
                queue_ops.delete_pending_for_pair(
                    conn, "eng_Latn-ita_Latn", "metricx24", force=True
                )
                is False
            )
            assert _count_rows(conn, "jobs", direction_key="eng_Latn-ita_Latn", model="metricx24") == 0
            assert _count_rows(conn, "directions", direction_key="eng_Latn-ita_Latn", model="metricx24") == 0
        finally:
            conn.close()


def test_reassign_without_done_deletes_only_old_non_done_pair() -> None:
    with workspace_temp_dir("build_queue") as tmp:
        db_path = Path(tmp) / "queue.db"
        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_job(
                conn,
                direction_key="eng_Latn-por_Latn",
                model="metricx24",
                shard_id=0,
                status="pending",
            )
            assert (
                queue_ops.delete_pending_for_pair(
                    conn, "eng_Latn-por_Latn", "metricx24", force=False
                )
                is False
            )
            assert _count_rows(conn, "jobs", direction_key="eng_Latn-por_Latn", model="metricx24") == 0
            assert _count_rows(conn, "directions", direction_key="eng_Latn-por_Latn", model="metricx24") == 0
        finally:
            conn.close()


def run_test() -> None:
    test_reset_pending_for_model_blocks_done_without_force()
    test_reset_pending_for_model_force_drops_done_rows()
    test_reset_pending_for_model_without_done_deletes_non_done_rows()
    test_reassign_preserves_done_rows_without_force_and_drops_with_force()
    test_reassign_without_done_deletes_only_old_non_done_pair()
    print("OK: build_queue preserves completed rows unless --force is explicit.")


if __name__ == "__main__":
    run_test()
