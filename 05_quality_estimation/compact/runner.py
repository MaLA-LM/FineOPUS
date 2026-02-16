from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from compact.buckets import _iter_stage_files, _split_table_by_bucket
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

    stage_root = (
        Path(args.output_base)
        / f"dataset={args.dataset}"
        / f"model={args.model_tag}"
        / f"split={args.split}"
        / "stage"
    )
    stage_files = _iter_stage_files(stage_root)
    if not stage_files:
        raise SystemExit(f"No stage parquet files found under: {stage_root}")

    run_id = args.run_id or generate_run_id()
    writer = BucketPartWriter(
        output_base=args.output_base,
        dataset=args.dataset,
        model_tag=args.model_tag,
        num_buckets=args.num_buckets,
        target_part_bytes=args.target_part_bytes,
        run_id=run_id,
    )

    rows_read = 0
    import pyarrow.parquet as pq

    for stage_file in stage_files:
        parquet_file = pq.ParquetFile(stage_file)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            rows_read += table.num_rows
            for bucket_id, bucket_table in _split_table_by_bucket(
                table, args.num_buckets
            ).items():
                writer.append(bucket_id, bucket_table)

    writer.flush_all()
    print(
        "Compaction complete: "
        f"stage_files={len(stage_files)} rows={rows_read} "
        f"buckets={args.num_buckets}"
    )
