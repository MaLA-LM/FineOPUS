from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone

from dataset.mediator import DatasetAdapter
from execution.flores_array.hashing import compute_shard_id
from execution.flores_array.manifest import ManifestEntry
from execution.flores_array.shard_context import resolve_shard_context
from execution.flores_array.stage_writer import ShardStageWriter
from utils.hashing import direction_key
from utils.logger import logger

ScoreEntry = Callable[[ManifestEntry], object]

__all__ = ["run_scoring", "FloresArrayExecutor", "validate_flores_args"]


def _make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-pid{os.getpid()}"


def _resolve_entry_shard_id(
    entry: ManifestEntry, num_shards: int, direction_key_value: str
) -> int:
    if entry.shard_id is None:
        return compute_shard_id(direction_key_value, num_shards)
    if entry.shard_id < 0:
        raise SystemExit(
            "Manifest shard_id out of range for current shard count: "
            f"{direction_key_value} has shard_id={entry.shard_id}."
        )
    return entry.shard_id


def run_scoring(
    args,
    dataset: DatasetAdapter,
    directions: list[ManifestEntry],
    model_tag: str,
    score_entry: ScoreEntry,
) -> None:
    shard_context = resolve_shard_context(args)
    run_id = _make_run_id()
    writers: dict[str, ShardStageWriter] = {}

    total_rows = len(directions)
    assigned_rows = 0
    skipped_rows = 0
    processed_rows = 0

    def get_writer(split: str) -> ShardStageWriter:
        writer = writers.get(split)
        if writer is None:
            writer = ShardStageWriter(
                output_base=args.output_base,
                dataset=dataset.id,
                model_tag=model_tag,
                split=split,
                shard_id=shard_context.shard_id,
                run_id=run_id,
                max_directions_per_part=args.max_directions_per_part,
                target_part_bytes=args.target_part_bytes,
            )
            writers[split] = writer
        return writer

    try:
        for entry in directions:
            key = direction_key(entry.src_lang, entry.tgt_lang)
            entry_shard_id = _resolve_entry_shard_id(
                entry, shard_context.num_shards, key
            )
            if entry_shard_id != shard_context.shard_id:
                continue
            assigned_rows += 1

            writer = get_writer(entry.split)
            if key in writer.committed_direction_keys:
                logger.info("SKIP (checkpoint): %s split=%s", key, entry.split)
                skipped_rows += 1
                continue

            logger.info(
                "Scoring %s split=%s shard=%s/%s",
                key,
                entry.split,
                shard_context.shard_id,
                shard_context.num_shards,
            )
            frame = score_entry(entry)
            frame["direction_key"] = key
            frame["shard_id"] = shard_context.shard_id
            writer.add_direction(frame, key)
            processed_rows += 1
    finally:
        for writer in writers.values():
            writer.close()

    logger.info(
        "Worker complete: total=%s assigned=%s processed=%s skipped=%s",
        total_rows,
        assigned_rows,
        processed_rows,
        skipped_rows,
    )


from execution.flores_array.directions import validate_flores_args  # noqa: E402,F401
from execution.flores_array.executor import FloresArrayExecutor  # noqa: E402,F401
