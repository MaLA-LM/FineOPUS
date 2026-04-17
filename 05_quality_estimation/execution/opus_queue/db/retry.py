from __future__ import annotations

import random
import sqlite3
import time
from collections.abc import Iterable

DEFAULT_CLAIM_RETRIES = 5

__all__ = ["DEFAULT_CLAIM_RETRIES", "execute_with_retry"]


def execute_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    params: Iterable = (),
    *,
    attempts: int = DEFAULT_CLAIM_RETRIES,
) -> sqlite3.Cursor:
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return conn.execute(sql, tuple(params))
        except sqlite3.OperationalError as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            time.sleep(0.1 + random.random() * 0.4)
    assert last_exc is not None
    raise last_exc
