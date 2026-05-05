from __future__ import annotations

import os
import random
import sqlite3
import time
from collections.abc import Iterable


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, value)


DEFAULT_CLAIM_RETRIES = _env_int("OPUS_QUEUE_DB_RETRIES", 30)
FINALIZE_RETRIES = _env_int("OPUS_QUEUE_DB_FINALIZE_RETRIES", 120)
_BACKOFF_BASE = _env_float("OPUS_QUEUE_DB_BACKOFF_BASE", 0.25)
_BACKOFF_CAP = _env_float("OPUS_QUEUE_DB_BACKOFF_CAP", 5.0)

__all__ = [
    "DEFAULT_CLAIM_RETRIES",
    "FINALIZE_RETRIES",
    "execute_with_retry",
]


def execute_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    params: Iterable = (),
    *,
    attempts: int = DEFAULT_CLAIM_RETRIES,
    base_delay: float = _BACKOFF_BASE,
    max_delay: float = _BACKOFF_CAP,
) -> sqlite3.Cursor:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return conn.execute(sql, tuple(params))
        except sqlite3.OperationalError as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            delay_cap = min(max_delay, base_delay * (2 ** attempt))
            time.sleep(random.uniform(0.0, delay_cap))
    assert last_exc is not None
    raise last_exc
