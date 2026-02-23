from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compact.buckets import (
    _extract_summary_keys,
    _extract_stage_commit_key,
    _filter_committed,
    _iter_stage_files,
    _load_stage_checkpoint,
    _split_table_by_bucket,
    _stage_checkpoint_path_for_part,
)
from compact.cli import parse_args
from compact.writer import BucketPartWriter
from utils.logger import logger

_ROW_BATCH_SIZE = 10_000


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-pid{os.getpid()}"


def main() -> None:
    args = parse_args()
    if args.num_buckets <= 0:
        raise SystemExit("--num-buckets must be > 0.")
    if args.target_part_bytes <= 0:
        raise SystemExit("--target-part-bytes must be > 0.")

    dataset_root = Path(args.output_base) / f"dataset={args.dataset}"
    stage_files = _iter_stage_files(dataset_root)
    if not stage_files:
        raise SystemExit(f"No stage JSONL files found under: {dataset_root}")

    run_id = generate_run_id()
    writer = BucketPartWriter(
        output_base=args.output_base,
        dataset=args.dataset,
        num_buckets=args.num_buckets,
        target_part_bytes=args.target_part_bytes,
        run_id=run_id,
    )

    if writer.committed_keys:
        logger.info(
            "Resuming: %s direction/model/split combos already committed - will be skipped.",
            len(writer.committed_keys),
        )

    rows_seen = 0
    rows_invalid = 0
    rows_uncommitted = 0
    rows_skipped_compacted = 0
    rows_kept = 0

    stage_checkpoint_cache: dict[Path, set[tuple[str, str, str, int]]] = {}
    pending_records: list[dict[str, Any]] = []

    import pyarrow as pa

    def flush_pending() -> None:
        nonlocal rows_skipped_compacted, rows_kept
        if not pending_records:
            return
        table = pa.Table.from_pylist(pending_records)
        pending_records.clear()

        total_in_batch = table.num_rows
        table = _filter_committed(table, writer.committed_keys)
        rows_skipped_compacted += total_in_batch - table.num_rows
        if table.num_rows == 0:
            return

        rows_kept += table.num_rows
        for bucket_id, bucket_table in _split_table_by_bucket(
            table, args.num_buckets
        ).items():
            summary_keys = _extract_summary_keys(bucket_table)
            writer.append(bucket_id, bucket_table, summary_keys=summary_keys)

    for stage_file in stage_files:
        checkpoint_path = _stage_checkpoint_path_for_part(stage_file)
        stage_committed = stage_checkpoint_cache.get(checkpoint_path)
        if stage_committed is None:
            stage_committed = _load_stage_checkpoint(checkpoint_path)
            stage_checkpoint_cache[checkpoint_path] = stage_committed

        with stage_file.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                rows_seen += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    rows_invalid += 1
                    logger.warning(
                        "Skipping malformed stage JSON at %s:%s",
                        stage_file,
                        line_no,
                    )
                    continue
                stage_key = _extract_stage_commit_key(record)
                if stage_key not in stage_committed:
                    rows_uncommitted += 1
                    continue

                pending_records.append(record)
                if len(pending_records) >= _ROW_BATCH_SIZE:
                    flush_pending()

    flush_pending()

    writer.flush_all()
    logger.info(
        "Compaction complete: stage_files=%s seen=%s kept=%s "
        "invalid=%s uncommitted=%s skipped_already_compacted=%s buckets=%s",
        len(stage_files),
        rows_seen,
        rows_kept,
        rows_invalid,
        rows_uncommitted,
        rows_skipped_compacted,
        args.num_buckets,
    )
