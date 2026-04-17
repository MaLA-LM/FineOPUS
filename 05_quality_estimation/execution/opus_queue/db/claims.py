from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from execution.opus_queue.db.retry import DEFAULT_CLAIM_RETRIES, execute_with_retry

FinalizeResult = Literal["done", "requeued", "failed", "stale_noop"]

__all__ = [
    "FinalizeResult",
    "claim_next",
    "mark_done",
    "mark_failed",
    "reset_own_stale",
    "reset_stale_rows",
]

_CLAIM_SQL = """
UPDATE jobs
   SET status = 'running',
       worker_id = ?,
       started_at = CAST(strftime('%s','now') AS INTEGER),
       attempts = attempts + 1
 WHERE rowid = (
           SELECT rowid FROM jobs
            WHERE status = 'pending' AND model = ?
            ORDER BY (end_idx - start_idx) DESC
            LIMIT 1
       )
RETURNING direction_key, model, shard_id, start_idx, end_idx, attempts
"""


def claim_next(
    conn: sqlite3.Connection,
    model: str,
    worker_id: str,
    *,
    retries: int = DEFAULT_CLAIM_RETRIES,
) -> dict | None:
    cursor = execute_with_retry(conn, _CLAIM_SQL, (worker_id, model), attempts=retries)
    row = cursor.fetchone()
    return dict(row) if row else None


def mark_done(
    conn: sqlite3.Connection,
    direction_key: str,
    model: str,
    shard_id: int,
    worker_id: str,
    out_path: str | Path,
) -> FinalizeResult:
    row = conn.execute(
        """
        UPDATE jobs
           SET status = 'done',
               finished_at = CAST(strftime('%s','now') AS INTEGER),
               out_path = ?,
               last_error = NULL
         WHERE direction_key = ?
           AND model = ?
           AND shard_id = ?
           AND status = 'running'
           AND worker_id = ?
        RETURNING status
        """,
        (str(out_path), direction_key, model, shard_id, worker_id),
    ).fetchone()
    if row is None:
        return "stale_noop"
    return "done"


def mark_failed(
    conn: sqlite3.Connection,
    direction_key: str,
    model: str,
    shard_id: int,
    worker_id: str,
    error: str | None,
    max_attempts: int,
) -> FinalizeResult:
    trimmed = (error or "")[:1000] or None
    row = conn.execute(
        """
        UPDATE jobs
           SET status = CASE
                            WHEN attempts >= ? THEN 'failed'
                            ELSE 'pending'
                        END,
               finished_at = CASE
                                 WHEN attempts >= ? THEN CAST(strftime('%s','now') AS INTEGER)
                                 ELSE NULL
                             END,
               worker_id = NULL,
               out_path = NULL,
               last_error = ?
         WHERE direction_key = ?
           AND model = ?
           AND shard_id = ?
           AND status = 'running'
           AND worker_id = ?
        RETURNING status
        """,
        (
            max_attempts,
            max_attempts,
            trimmed,
            direction_key,
            model,
            shard_id,
            worker_id,
        ),
    ).fetchone()
    if row is None:
        return "stale_noop"
    return "failed" if row["status"] == "failed" else "requeued"


def reset_own_stale(conn: sqlite3.Connection, worker_id: str) -> int:
    cursor = conn.execute(
        """
        UPDATE jobs
           SET status = 'pending',
               worker_id = NULL,
               last_error = 'reset_own_stale'
         WHERE status = 'running' AND worker_id = ?
        """,
        (worker_id,),
    )
    return int(cursor.rowcount or 0)


def reset_stale_rows(
    conn: sqlite3.Connection,
    per_model_cutoff_seconds: dict[str, int],
    *,
    reset_failed: bool = False,
) -> list[dict]:
    reclaimed: list[dict] = []
    if not per_model_cutoff_seconds:
        return reclaimed
    now_row = conn.execute(
        "SELECT CAST(strftime('%s','now') AS INTEGER) AS ts"
    ).fetchone()
    now_ts = int(now_row["ts"])
    for model, cutoff_seconds in per_model_cutoff_seconds.items():
        cutoff_ts = now_ts - int(cutoff_seconds)
        rows = conn.execute(
            """
            SELECT direction_key, model, shard_id, worker_id, started_at,
                   'running' AS previous_status
              FROM jobs
             WHERE status = 'running' AND model = ? AND started_at < ?
            """,
            (model, cutoff_ts),
        ).fetchall()
        if rows:
            conn.execute(
                """
                UPDATE jobs
                   SET status = 'pending',
                       worker_id = NULL,
                       last_error = 'reaped'
                 WHERE status = 'running' AND model = ? AND started_at < ?
                """,
                (model, cutoff_ts),
            )
            reclaimed.extend(dict(row) for row in rows)
        if not reset_failed:
            continue

        failed_rows = conn.execute(
            """
            SELECT direction_key, model, shard_id, worker_id, finished_at, last_error,
                   'failed' AS previous_status
              FROM jobs
             WHERE status = 'failed' AND model = ? AND finished_at < ?
            """,
            (model, cutoff_ts),
        ).fetchall()
        if not failed_rows:
            continue
        conn.execute(
            """
            UPDATE jobs
               SET status = 'pending',
                   attempts = 0,
                   worker_id = NULL,
                   finished_at = NULL
             WHERE status = 'failed' AND model = ? AND finished_at < ?
            """,
            (model, cutoff_ts),
        )
        reclaimed.extend(dict(row) for row in failed_rows)
    return reclaimed
