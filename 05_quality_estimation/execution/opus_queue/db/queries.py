from __future__ import annotations

import sqlite3

__all__ = ["count_by_status"]


def count_by_status(
    conn: sqlite3.Connection, model: str | None = None
) -> dict[str, int]:
    if model is not None:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE model=? GROUP BY status",
            (model,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
    return {row["status"]: int(row["n"]) for row in rows}
