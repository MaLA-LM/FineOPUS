from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from utils.hashing import stable_hash_int

if TYPE_CHECKING:
    import pyarrow as pa


def compute_bucket_id(direction_key_value: str, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be > 0.")
    return stable_hash_int(direction_key_value) % num_buckets


def _iter_stage_files(stage_root: Path) -> list[Path]:
    if not stage_root.exists():
        return []
    return sorted(path for path in stage_root.rglob("*.parquet") if path.is_file())


def _split_table_by_bucket(table: pa.Table, num_buckets: int) -> dict[int, pa.Table]:
    import pyarrow as pa

    if "direction_key" not in table.column_names:
        raise ValueError("Input table is missing required 'direction_key' column.")

    index_map: dict[int, list[int]] = defaultdict(list)
    keys = table.column("direction_key").to_pylist()
    for idx, key in enumerate(keys):
        if key is None:
            raise ValueError("direction_key contains null values.")
        bucket_id = compute_bucket_id(str(key), num_buckets)
        index_map[bucket_id].append(idx)

    result: dict[int, pa.Table] = {}
    for bucket_id, indices in index_map.items():
        result[bucket_id] = table.take(pa.array(indices, type=pa.int64()))
    return result

