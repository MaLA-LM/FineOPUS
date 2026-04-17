"""Row-count cache for OPUS direction parquet files.

Avoids re-scanning parquet metadata on every build_queue invocation by
persisting {direction_key: row_count} to a JSON sidecar file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from utils.logger import logger

_COUNT_CACHE_REL_PATH = Path("FineOPUS_test/.row_counts.json")


def resolve_path(count_cache_arg: str | None, opus_root: Path) -> Path:
    if count_cache_arg:
        return Path(count_cache_arg)
    return opus_root.parent / _COUNT_CACHE_REL_PATH


def load(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Row-count cache unreadable, ignoring: %s", path)
        return {}
    return {str(k): int(v) for k, v in data.items()}


def save(path: Path, cache: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def count_direction_rows(direction_dir: Path) -> int:
    import pyarrow.parquet as pq  # local import keeps CLI import light

    total = 0
    for parquet_path in sorted(direction_dir.glob("*.parquet")):
        if not parquet_path.is_file():
            continue
        meta = pq.ParquetFile(str(parquet_path)).metadata
        total += int(meta.num_rows)
    return total
