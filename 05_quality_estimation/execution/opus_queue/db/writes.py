"""Write-path DB operations used by build_queue.

These helpers manage direction/job lifecycle (insert, reset, reassign,
orphan cleanup) on top of the low-level primitives in ``queue_db``.
"""
from __future__ import annotations

import sqlite3


def fetch_existing_models(conn: sqlite3.Connection, direction_key: str) -> list[str]:
    rows = conn.execute(
        "SELECT model FROM directions WHERE direction_key=? ORDER BY model",
        (direction_key,),
    ).fetchall()
    return [str(row["model"]) for row in rows]


def count_done_jobs(
    conn: sqlite3.Connection,
    model: str,
    direction_key: str | None = None,
) -> int:
    if direction_key is None:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE model=? AND status='done'",
            (model,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs "
            "WHERE direction_key=? AND model=? AND status='done'",
            (direction_key, model),
        ).fetchone()
    return int(row["n"])


def _delete_jobs(
    conn: sqlite3.Connection,
    where_sql: str,
    params: tuple,
    *,
    force: bool,
) -> None:
    if force:
        conn.execute(f"DELETE FROM jobs WHERE {where_sql}", params)
    else:
        conn.execute(
            f"DELETE FROM jobs WHERE {where_sql} AND status != 'done'", params
        )


def _delete_orphan_directions(
    conn: sqlite3.Connection,
    model: str,
    direction_key: str | None = None,
) -> None:
    """Remove direction rows that no longer have any associated jobs."""
    if direction_key is None:
        conn.execute(
            """
            DELETE FROM directions
             WHERE model = ?
               AND NOT EXISTS (
                   SELECT 1 FROM jobs
                    WHERE jobs.direction_key = directions.direction_key
                      AND jobs.model = directions.model
               )
            """,
            (model,),
        )
    else:
        conn.execute(
            """
            DELETE FROM directions
             WHERE direction_key = ? AND model = ?
               AND NOT EXISTS (
                   SELECT 1 FROM jobs
                    WHERE jobs.direction_key = directions.direction_key
                      AND jobs.model = directions.model
               )
            """,
            (direction_key, model),
        )


def reset_pending_for_model(
    conn: sqlite3.Connection, model: str, *, force: bool
) -> None:
    done_count = count_done_jobs(conn, model)
    if done_count > 0 and not force:
        raise SystemExit(
            f"--reset-pending-for-model {model}: {done_count} 'done' rows exist. "
            "Re-run with --force to discard them."
        )
    _delete_jobs(conn, "model=?", (model,), force=force)
    _delete_orphan_directions(conn, model)


def delete_pending_for_pair(
    conn: sqlite3.Connection, direction_key: str, model: str, *, force: bool
) -> bool:
    """Delete non-done rows for a (direction, model). Returns True if blocked."""
    done_count = count_done_jobs(conn, model, direction_key=direction_key)
    if done_count > 0 and not force:
        return True
    _delete_jobs(
        conn,
        "direction_key=? AND model=?",
        (direction_key, model),
        force=force,
    )
    _delete_orphan_directions(conn, model, direction_key)
    return False
