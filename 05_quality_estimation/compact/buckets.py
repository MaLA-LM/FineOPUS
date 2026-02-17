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


def _iter_stage_files(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted(
        path
        for path in dataset_root.rglob("*.parquet")
        if path.is_file() and "stage" in path.parts
    )


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


def _filter_committed(
    table: pa.Table, committed_keys: set[tuple[str, str, str]]
) -> pa.Table:
    """Remove rows whose (direction_key, model_name, split) is already committed."""
    if not committed_keys:
        return table
    import pyarrow as pa

    dkeys = table.column("direction_key").to_pylist()
    models = table.column("model_name").to_pylist()
    splits = table.column("split").to_pylist()

    keep_indices = [
        idx
        for idx in range(table.num_rows)
        if (str(dkeys[idx]), str(models[idx]), str(splits[idx])) not in committed_keys
    ]

    if len(keep_indices) == table.num_rows:
        return table
    if not keep_indices:
        return table.slice(0, 0)
    return table.take(pa.array(keep_indices, type=pa.int64()))


def _extract_summary_keys(table: pa.Table) -> set[tuple[str, str, str]]:
    """Extract (direction_key, model_name, split) tuples from summary rows."""
    row_types = table.column("row_type").to_pylist()
    dkeys = table.column("direction_key").to_pylist()
    models = table.column("model_name").to_pylist()
    splits = table.column("split").to_pylist()
    return {
        (str(dkeys[idx]), str(models[idx]), str(splits[idx]))
        for idx in range(table.num_rows)
        if row_types[idx] == "summary"
    }
