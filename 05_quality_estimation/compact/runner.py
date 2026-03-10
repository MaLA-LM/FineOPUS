from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
from compact.writer import ModelPartWriter
from utils.io import ROW_TYPE_SUMMARY
from utils.logger import logger

_ROW_BATCH_SIZE = 10_000


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-pid{os.getpid()}"


@dataclass
class ModelStats:
    model: str
    files: int = 0
    seen: int = 0
    kept: int = 0
    invalid: int = 0
    uncommitted: int = 0
    skipped: int = 0


# ---------- per-model worker (fully independent) ----------


def _compact_model(
    model: str,
    dataset_root: Path,
    output_base: str,
    dataset: str,
    target_part_bytes: int,
    run_id: str,
) -> ModelStats:
    """Read all stage files for one model, write compacted Parquet.
    Runs entirely in its own thread.
    """
    stats = ModelStats(model=model)

    stage_files = iter_stage_files_for_model(dataset_root, model)
    stats.files = len(stage_files)
    if not stage_files:
        return stats

    writer = ModelPartWriter(
        output_base=output_base,
        dataset=dataset,
        model=model,
        target_part_bytes=target_part_bytes,
        run_id=run_id,
    )

    # Group files by shard dir (each shard shares one checkpoint).
    shards: dict[Path, list[Path]] = {}
    for path in stage_files:
        shards.setdefault(path.parent, []).append(path)

    pending: list[dict] = []

    def flush_pending() -> None:
        if not pending:
            return
        table = pa.Table.from_pylist(pending)
        pending.clear()

        # Filter out (direction_key, split) combos already committed.
        if writer.committed_splits:
            dk = table.column("direction_key").to_pylist()
            sp = table.column("split").to_pylist()
            keep = [
                i
                for i in range(table.num_rows)
                if (dk[i], sp[i]) not in writer.committed_splits
            ]
            stats.skipped += table.num_rows - len(keep)
            if not keep:
                return
            table = table.take(pa.array(keep, type=pa.int64()))

        # Collect new (direction_key, split) pairs from summary rows.
        rt = table.column("row_type").to_pylist()
        dk = table.column("direction_key").to_pylist()
        sp = table.column("split").to_pylist()
        new_splits = {
            (dk[i], sp[i]) for i in range(table.num_rows) if rt[i] == ROW_TYPE_SUMMARY
        }

        stats.kept += table.num_rows
        writer.append(table, new_splits=new_splits)

    for shard_dir, part_files in shards.items():
        checkpoint = load_stage_checkpoint(shard_dir / "checkpoint.jsonl")

        for part_file in part_files:
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

                    pending.append(record)
                    if len(pending) >= _ROW_BATCH_SIZE:
                        flush_pending()

    flush_pending()
    writer.flush_all()
    return stats


# ---------- main -----------------------------------------------------------


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
            ): model
            for model in models
        }

        all_stats: list[ModelStats] = []
        for future in as_completed(futures):
            model = futures[future]
            stats = future.result()  # raises if thread failed
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
