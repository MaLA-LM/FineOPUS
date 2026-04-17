from __future__ import annotations

import sqlite3

__all__ = ["log_event"]


def log_event(
    conn: sqlite3.Connection,
    worker_id: str | None,
    event: str,
    *,
    direction_key: str | None = None,
    model: str | None = None,
    shard_id: int | None = None,
    detail: str | None = None,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO run_events
                (ts, worker_id, event, direction_key, model, shard_id, detail)
            VALUES (CAST(strftime('%s','now') AS INTEGER), ?, ?, ?, ?, ?, ?)
            """,
            (
                worker_id,
                event,
                direction_key,
                model,
                shard_id,
                detail[:1000] if detail else None,
            ),
        )
    except sqlite3.OperationalError:
        pass
