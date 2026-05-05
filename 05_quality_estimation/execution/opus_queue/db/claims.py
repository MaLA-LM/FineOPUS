from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from execution.opus_queue.db.retry import (
    DEFAULT_CLAIM_RETRIES,
    FINALIZE_RETRIES,
    execute_with_retry,
)

FinalizeResult = Literal["done", "requeued", "failed", "stale_noop"]

__all__ = [
    "FinalizeResult",
    "claim_next",
    "mark_done",
    "mark_failed",
    "reset_own_stale",
    "reset_stale_rows",
]

_MAX_ERROR_CHARS = 1000
_NOW_SQL = "CAST(strftime('%s','now') AS INTEGER)"
_GPU_SECONDS_EXPR = (
    "CASE "
    "WHEN started_at IS NULL THEN gpu_seconds_total "
    "ELSE gpu_seconds_total + "
    f"(({_NOW_SQL} - started_at) * "
    "CASE WHEN claim_gpu_count < 1 THEN 1 ELSE claim_gpu_count END) "
    "END"
)

_CLAIM_SQL = """
UPDATE jobs
   SET status = 'running',
       worker_id = ?,
       started_at = CAST(strftime('%s','now') AS INTEGER),
       finished_at = NULL,
       claim_gpu_count = ?,
       attempts = attempts + 1
 WHERE rowid = (
           SELECT rowid FROM jobs
            WHERE status = 'pending' AND model = ?
            ORDER BY (end_idx - start_idx) DESC
            LIMIT 1
       )
RETURNING direction_key, model, shard_id, start_idx, end_idx, attempts, started_at
"""

_CLAIM_DIRECTION_SQL = """
UPDATE jobs
   SET status = 'running',
       worker_id = ?,
       started_at = CAST(strftime('%s','now') AS INTEGER),
       finished_at = NULL,
       claim_gpu_count = ?,
       attempts = attempts + 1
 WHERE rowid = (
           SELECT rowid FROM jobs
            WHERE status = 'pending' AND model = ? AND direction_key = ?
            ORDER BY (end_idx - start_idx) DESC
            LIMIT 1
       )
RETURNING direction_key, model, shard_id, start_idx, end_idx, attempts, started_at
"""


def _claim_with_sql(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple,
    *,
    retries: int,
) -> dict | None:
    cursor = execute_with_retry(conn, sql, params, attempts=retries)
    row = cursor.fetchone()
    return None if row is None else dict(row)


def claim_next(
    conn: sqlite3.Connection,
    model: str,
    worker_id: str,
    *,
    retries: int = DEFAULT_CLAIM_RETRIES,
    slurm_job_id: str | None = None,
    slurm_array_task_id: str | None = None,
    node_host: str | None = None,
    gpu_count: int = 1,
    preferred_direction_key: str | None = None,
) -> dict | None:
    del slurm_job_id, slurm_array_task_id, node_host
    claim_gpu_count = max(1, int(gpu_count))
    if preferred_direction_key:
        row = _claim_with_sql(
            conn,
            _CLAIM_DIRECTION_SQL,
            (worker_id, claim_gpu_count, model, preferred_direction_key),
            retries=retries,
        )
        if row is not None:
            return row

    return _claim_with_sql(
        conn,
        _CLAIM_SQL,
        (worker_id, claim_gpu_count, model),
        retries=retries,
    )


def mark_done(
    conn: sqlite3.Connection,
    direction_key: str,
    model: str,
    shard_id: int,
    worker_id: str,
    out_path: str | Path,
) -> FinalizeResult:
    row = execute_with_retry(
        conn,
        f"""
        UPDATE jobs
           SET status = 'done',
               finished_at = {_NOW_SQL},
               gpu_seconds_total = {_GPU_SECONDS_EXPR},
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
        attempts=FINALIZE_RETRIES,
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
    trimmed = _trim_error(error)
    row = execute_with_retry(
        conn,
        f"""
        UPDATE jobs
           SET status = CASE
                            WHEN attempts >= ? THEN 'failed'
                            ELSE 'pending'
                        END,
               finished_at = CASE
                                 WHEN attempts >= ? THEN {_NOW_SQL}
                                 ELSE NULL
                             END,
               started_at = CASE
                                WHEN attempts >= ? THEN started_at
                                ELSE NULL
                            END,
               claim_gpu_count = CASE
                                     WHEN attempts >= ? THEN claim_gpu_count
                                     ELSE 1
                                 END,
               gpu_seconds_total = {_GPU_SECONDS_EXPR},
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
            max_attempts,
            max_attempts,
            trimmed,
            direction_key,
            model,
            shard_id,
            worker_id,
        ),
        attempts=FINALIZE_RETRIES,
    ).fetchone()
    if row is None:
        return "stale_noop"
    return "failed" if row["status"] == "failed" else "requeued"


def _trim_error(error: str | None) -> str | None:
    if not error:
        return None
    if len(error) <= _MAX_ERROR_CHARS:
        return error

    separator = "\n...\n"
    head_budget = 650
    tail_budget = _MAX_ERROR_CHARS - head_budget - len(separator)
    head = error[:head_budget].rstrip()
    tail = error[-tail_budget:].lstrip()
    return f"{head}{separator}{tail}"


def reset_own_stale(conn: sqlite3.Connection, worker_id: str) -> int:
    cursor = execute_with_retry(
        conn,
        f"""
        UPDATE jobs
           SET status = 'pending',
               worker_id = NULL,
               started_at = NULL,
               claim_gpu_count = 1,
               gpu_seconds_total = {_GPU_SECONDS_EXPR},
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
            execute_with_retry(
                conn,
                f"""
                UPDATE jobs
                   SET status = 'pending',
                       worker_id = NULL,
                       started_at = NULL,
                       claim_gpu_count = 1,
                       gpu_seconds_total = {_GPU_SECONDS_EXPR},
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
        execute_with_retry(
            conn,
            """
            UPDATE jobs
               SET status = 'pending',
                   attempts = 0,
                   worker_id = NULL,
                   started_at = NULL,
                   finished_at = NULL,
                   claim_gpu_count = 1
             WHERE status = 'failed' AND model = ? AND finished_at < ?
            """,
            (model, cutoff_ts),
        )
        reclaimed.extend(dict(row) for row in failed_rows)
    return reclaimed
