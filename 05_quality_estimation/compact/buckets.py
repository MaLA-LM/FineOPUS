from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from utils.hashing import stable_hash_int
from utils.io import ROW_TYPE_SUMMARY

if TYPE_CHECKING:
    import pyarrow as pa

StageCommitKey = tuple[str, str, str, int]


def compute_bucket_id(direction_key_value: str, num_buckets: int) -> int:
    return stable_hash_int(direction_key_value) % num_buckets


def _iter_stage_files(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted(
        path
        for path in dataset_root.rglob("part-*.jsonl")
        if path.is_file() and any(part.startswith("shard=") for part in path.parts)
    )


def _stage_checkpoint_path_for_part(stage_part_path: Path) -> Path:
    return stage_part_path.parent / "checkpoint.jsonl"


def _extract_stage_commit_key(record: Mapping[str, object]) -> StageCommitKey:
    return (
        record["direction_key"],
        record["model_name"],
        record["split"],
        int(record["shard_id"]),
    )


def _load_stage_checkpoint(checkpoint_path: Path) -> set[StageCommitKey]:
    if not checkpoint_path.exists():
        return set()

    committed: set[StageCommitKey] = set()
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("row_type") == ROW_TYPE_SUMMARY:
                committed.add(_extract_stage_commit_key(record))
    return committed


def _split_table_by_bucket(table: pa.Table, num_buckets: int) -> dict[int, pa.Table]:
    import pyarrow as pa

    index_map: dict[int, list[int]] = defaultdict(list)
    keys = table.column("direction_key").to_pylist()
    for idx, key in enumerate(keys):
        bucket_id = compute_bucket_id(key, num_buckets)
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
        if row_types[idx] == ROW_TYPE_SUMMARY
    }
