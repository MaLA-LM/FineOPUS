"""
db.py
-----
DuckDB connection factory with HPC-appropriate resource settings.
"""

from pathlib import Path

import duckdb


def setup_connection(
    scratch_tmp: Path,
    threads: int = 64,
    memory_limit: str = "180GB",
    max_temp_size: str = "500GiB",
) -> duckdb.DuckDBPyConnection:
    """
    Create and configure a DuckDB in-memory connection for large-scale
    parquet processing on HPC scratch storage.
    """
    scratch_tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads};")
    con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    con.execute(f"SET temp_directory='{scratch_tmp}';")
    con.execute(f"SET max_temp_directory_size='{max_temp_size}';")
    return con
