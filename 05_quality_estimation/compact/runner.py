from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from compact.buckets import (
    _extract_summary_keys,
    _filter_committed,
    _iter_stage_files,
    _split_table_by_bucket,
)
from compact.cli import parse_args
from compact.writer import BucketPartWriter


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
        raise SystemExit(f"No stage parquet files found under: {dataset_root}")

    run_id = args.run_id or generate_run_id()
    writer = BucketPartWriter(
        output_base=args.output_base,
        dataset=args.dataset,
        num_buckets=args.num_buckets,
        target_part_bytes=args.target_part_bytes,
        run_id=run_id,
    )

    if writer.committed_keys:
        print(
            f"Resuming: {len(writer.committed_keys)} direction/model/split "
            "combos already committed — will be skipped."
        )

    rows_read = 0
    rows_skipped = 0
    import pyarrow.parquet as pq

    for stage_file in stage_files:
        parquet_file = pq.ParquetFile(stage_file)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            total_in_group = table.num_rows

            table = _filter_committed(table, writer.committed_keys)
            rows_skipped += total_in_group - table.num_rows
            if table.num_rows == 0:
                continue

            rows_read += table.num_rows
            for bucket_id, bucket_table in _split_table_by_bucket(
                table, args.num_buckets
            ).items():
                summary_keys = _extract_summary_keys(bucket_table)
                writer.append(bucket_id, bucket_table, summary_keys=summary_keys)

    writer.flush_all()
    print(
        "Compaction complete: "
        f"stage_files={len(stage_files)} rows={rows_read} "
        f"skipped={rows_skipped} buckets={args.num_buckets}"
    )
