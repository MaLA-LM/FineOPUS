from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from compact.buckets import (
    discover_models,
    extract_stage_commit_key,
    iter_stage_files_for_model,
    load_stage_checkpoint,
)
from compact.cli import parse_args
from compact.models import ModelStats, PendingRows
from compact.writer import ModelPartWriter
from utils.io import ROW_TYPE_SUMMARY
from utils.logger import logger

_PAIR_BATCH_SIZE = 100
_BUCKETS_PER_MODEL = 6


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-pid{os.getpid()}"


def _group_stage_files_by_shard(stage_files: list[Path]) -> dict[Path, list[Path]]:
    shards: dict[Path, list[Path]] = {}
    for path in stage_files:
        shards.setdefault(path.parent, []).append(path)
    return shards


def _parse_shard_id(shard_dir: Path) -> int:
    name = shard_dir.name
    prefix = "shard="
    if not name.startswith(prefix):
        raise ValueError(f"Expected shard directory, got: {shard_dir}")
    return int(name[len(prefix) :])


def _assign_shards_to_buckets(
    shard_groups: dict[Path, list[Path]],
) -> dict[int, list[tuple[Path, list[Path]]]]:
    bucket_shards: dict[int, list[tuple[Path, list[Path]]]] = {}
    shard_items = sorted(shard_groups.items(), key=lambda item: _parse_shard_id(item[0]))
    for shard_dir, part_files in shard_items:
        bucket_id = _parse_shard_id(shard_dir) % _BUCKETS_PER_MODEL
        bucket_shards.setdefault(bucket_id, []).append((shard_dir, part_files))
    return bucket_shards


def _flush_pending_rows(
    writer: ModelPartWriter,
    pending: PendingRows,
    stats: ModelStats,
) -> None:
    if not pending.pending_rows:
        return

    table = pa.Table.from_pylist(pending.pending_rows)
    pending.pending_rows.clear()
    pending.pending_pairs = 0

    kept, skipped = writer.append_new_rows(table)
    stats.kept += kept
    stats.skipped += skipped


def _finalize_current_pair(
    writer: ModelPartWriter,
    pending: PendingRows,
    stats: ModelStats,
) -> None:
    if not pending.current_pair:
        return

    pending.pending_rows.extend(pending.current_pair)
    pending.current_pair.clear()
    pending.pending_pairs += 1
    if pending.pending_pairs >= _PAIR_BATCH_SIZE:
        _flush_pending_rows(writer, pending, stats)


def _buffer_record(
    record: dict,
    writer: ModelPartWriter,
    pending: PendingRows,
    stats: ModelStats,
) -> None:
    if record.get("row_type") == ROW_TYPE_SUMMARY:
        _finalize_current_pair(writer, pending, stats)

    pending.current_pair.append(record)


def _process_part_file(
    part_file: Path,
    checkpoint,
    writer: ModelPartWriter,
    pending: PendingRows,
    stats: ModelStats,
) -> None:
    with part_file.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            stats.seen += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.invalid += 1
                logger.warning(
                    "Skipping malformed JSON at %s:%s",
                    part_file,
                    line_no,
                )
                continue

            if extract_stage_commit_key(record) not in checkpoint:
                stats.uncommitted += 1
                continue

            _buffer_record(record, writer, pending, stats)


def _compact_shard(
    model: str,
    shard_dir: Path,
    part_files: list[Path],
    writer: ModelPartWriter,
) -> ModelStats:
    stats = ModelStats(model=model, files=len(part_files))
    checkpoint = load_stage_checkpoint(shard_dir / "checkpoint.jsonl")
    pending = PendingRows()

    for part_file in part_files:
        _process_part_file(part_file, checkpoint, writer, pending, stats)

    _finalize_current_pair(writer, pending, stats)
    _flush_pending_rows(writer, pending, stats)
    return stats


def _compact_bucket(
    model: str,
    bucket_id: int,
    shard_items: list[tuple[Path, list[Path]]],
    output_base: str,
    dataset: str,
    target_part_bytes: int,
    run_id: str,
    bucket_name: str,
) -> ModelStats:
    writer = ModelPartWriter(
        output_base=output_base,
        dataset=dataset,
        model=model,
        target_part_bytes=target_part_bytes,
        run_id=run_id,
        bucket_name=bucket_name,
        bucket_id=bucket_id,
    )

    stats = ModelStats(model=model)
    for shard_dir, part_files in shard_items:
        stats.add(_compact_shard(model, shard_dir, part_files, writer))

    writer.flush_all()
    return stats


def _compact_model(
    model: str,
    dataset_root: Path,
    output_base: str,
    dataset: str,
    target_part_bytes: int,
    run_id: str,
    bucket_name: str,
) -> ModelStats:
    """Read all stage files for one model and write compacted Parquet."""
    stage_files = iter_stage_files_for_model(dataset_root, model)
    if not stage_files:
        return ModelStats(model=model)

    shard_groups = _group_stage_files_by_shard(stage_files)
    bucket_assignments = _assign_shards_to_buckets(shard_groups)
    bucket_workers = min(_BUCKETS_PER_MODEL, len(bucket_assignments))

    stats = ModelStats(model=model)
    with ThreadPoolExecutor(max_workers=bucket_workers) as executor:
        futures = {
            executor.submit(
                _compact_bucket,
                model,
                bucket_id,
                shard_items,
                output_base,
                dataset,
                target_part_bytes,
                run_id,
                bucket_name,
            ): bucket_id
            for bucket_id, shard_items in bucket_assignments.items()
        }
        for future in as_completed(futures):
            stats.add(future.result())

    return stats


def main() -> None:
    args = parse_args()
    if args.target_part_bytes <= 0:
        raise SystemExit("--target-part-bytes must be > 0.")

    dataset_root = Path(args.output_base) / f"dataset={args.dataset}"
    models = discover_models(dataset_root)
    if not models:
        raise SystemExit(f"No model directories found under: {dataset_root}")

    run_id = generate_run_id()
    logger.info("Found %s models: %s", len(models), ", ".join(models))

    num_workers = min(args.workers, len(models))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _compact_model,
                model,
                dataset_root,
                args.output_base,
                args.dataset,
                args.target_part_bytes,
                run_id,
                args.name,
            ): model
            for model in models
        }

        all_stats: list[ModelStats] = []
        for future in as_completed(futures):
            model = futures[future]
            stats = future.result()
            all_stats.append(stats)
            logger.info(
                "Model %-30s  files=%s seen=%s kept=%s "
                "invalid=%s uncommitted=%s skipped=%s",
                model,
                stats.files,
                stats.seen,
                stats.kept,
                stats.invalid,
                stats.uncommitted,
                stats.skipped,
            )

    total_seen = sum(s.seen for s in all_stats)
    total_kept = sum(s.kept for s in all_stats)
    logger.info(
        "Compaction complete: models=%s workers=%s total_seen=%s total_kept=%s",
        len(models),
        num_workers,
        total_seen,
        total_kept,
    )
