from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 2
BUSY_TIMEOUT_MS = 120_000

__all__ = ["BUSY_TIMEOUT_MS", "SCHEMA_PATH", "SCHEMA_VERSION", "connect", "initialize"]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {str(row[1]) for row in rows}


def _ensure_jobs_gpu_columns(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "jobs")
    if "claim_gpu_count" not in columns:
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN claim_gpu_count INTEGER NOT NULL DEFAULT 1"
        )
    if "gpu_seconds_total" not in columns:
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN gpu_seconds_total REAL NOT NULL DEFAULT 0"
        )


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=120.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    _ensure_jobs_gpu_columns(conn)
    current = int(conn.execute("PRAGMA user_version;").fetchone()[0])
    if current == 0:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    elif current == 1:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    elif current != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported schema version {current} (expected {SCHEMA_VERSION}); "
            "migration required."
        )
