from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 30_000

__all__ = ["BUSY_TIMEOUT_MS", "SCHEMA_PATH", "SCHEMA_VERSION", "connect", "initialize"]


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    current = int(conn.execute("PRAGMA user_version;").fetchone()[0])
    if current == 0:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    elif current != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported schema version {current} (expected {SCHEMA_VERSION}); "
            "migration required."
        )
