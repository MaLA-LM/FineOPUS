from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from dataset.manifest import ManifestEntry, read_manifest_entries
from dataset.mediator import DatasetAdapter
from utils.hashing import compute_shard_id, direction_key
from utils.stage_writer import ShardStageWriter

ScoreEntry = Callable[[ManifestEntry], object]


@dataclass(frozen=True)
class ShardContext:
    shard_id: int
    num_shards: int


def generate_run_id() -> str:
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-pid{os.getpid()}"


def _parse_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer in {name}: {raw!r}") from exc


def resolve_shard_context(args) -> ShardContext:
    task_id = _parse_int_env("SLURM_ARRAY_TASK_ID")
    task_count = _parse_int_env("SLURM_ARRAY_TASK_COUNT")
    task_min = _parse_int_env("SLURM_ARRAY_TASK_MIN")

    if task_id is not None and task_count is not None:
        min_task_id = 0 if task_min is None else task_min
        shard_id = task_id - min_task_id
        num_shards = task_count
    else:
        if args.num_shards is None or args.shard_id is None:
            raise SystemExit(
                "Missing shard info. Provide --num-shards and --shard-id, "
                "or run inside a Slurm array with SLURM_ARRAY_TASK_ID/COUNT."
            )
        shard_id = args.shard_id
        num_shards = args.num_shards

    if num_shards <= 0:
        raise SystemExit("num_shards must be > 0.")
    if shard_id < 0 or shard_id >= num_shards:
        raise SystemExit(
            f"shard_id out of range: {shard_id} not in [0, {num_shards - 1}]"
        )
    return ShardContext(shard_id=shard_id, num_shards=num_shards)


def collect_directions(args, dataset: DatasetAdapter) -> list[ManifestEntry]:
    if not args.manifest:
        raise SystemExit("Worker mode requires --manifest.")
    directions = read_manifest_entries(args.manifest)
    valid_splits = set(dataset.split_values)
    for entry in directions:
        if entry.split not in valid_splits:
            supported = ", ".join(dataset.split_values)
            raise SystemExit(
                f"Unsupported split '{entry.split}' for dataset '{dataset.id}'. "
                f"Supported: {supported}."
            )
    return directions


def _resolve_entry_shard_id(
    entry: ManifestEntry, num_shards: int, direction_key_value: str
) -> int:
    if entry.shard_id is None:
        return compute_shard_id(direction_key_value, num_shards)
    if entry.shard_id < 0 or entry.shard_id >= num_shards:
        raise SystemExit(
            "Manifest shard_id out of range for current shard count: "
            f"{direction_key_value} has shard_id={entry.shard_id}, "
            f"num_shards={num_shards}."
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
    run_id = args.run_id or generate_run_id()
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
                max_seconds_per_part=args.max_seconds_per_part,
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
            if args.resume and key in writer.committed_direction_keys:
                print(f"SKIP (checkpoint): {key} split={entry.split}")
                skipped_rows += 1
                continue

            print(
                "Scoring "
                f"{key} split={entry.split} "
                f"shard={shard_context.shard_id}/{shard_context.num_shards}"
            )
            frame = score_entry(entry)
            frame = frame.copy()
            frame["direction_key"] = key
            frame["shard_id"] = shard_context.shard_id
            writer.add_direction(frame, key)
            processed_rows += 1
    finally:
        for writer in writers.values():
            writer.close()

    print(
        "Worker complete: "
        f"total={total_rows} assigned={assigned_rows} "
        f"processed={processed_rows} skipped={skipped_rows}"
    )
