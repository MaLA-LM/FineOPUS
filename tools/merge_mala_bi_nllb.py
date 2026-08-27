#!/usr/bin/env python3
"""Resumably merge two Parquet trees and canonicalize translation text names."""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OLD = ("src_text", "tgt_text")
NEW = ("source_text", "target_text")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mala-root", type=Path, default=Path(
        "/scratch/project_462001069/mala-bilingual-translation-corpus"))
    p.add_argument("--nllb-root", type=Path, default=Path(
        "/scratch/project_462001069/nllb/nllb-conversion"))
    p.add_argument("--output-root", type=Path, default=Path(
        "/scratch/project_462001069/mala-bi-nllb"))
    p.add_argument("--workers", type=int, default=int(
        os.getenv("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--batch-size", type=int, default=250_000)
    p.add_argument("--state-log", type=Path, default=Path(
        "logs/merge_mala_bi_nllb_state.jsonl"))
    p.add_argument("--copy-canonical", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def normalized_schema(schema):
    names = schema.names
    old = all(name in names for name in OLD)
    new = all(name in names for name in NEW)
    if old == new:
        raise ValueError(f"expected exactly one text-column pair, got {names}")
    if new:
        return schema, False
    mapping = dict(zip(OLD, NEW))
    fields = [field.with_name(mapping.get(field.name, field.name)) for field in schema]
    return pa.schema(fields, metadata=schema.metadata), True


def valid(source, destination):
    if not destination.is_file():
        return False, "missing"
    try:
        src = pq.ParquetFile(source)
        dst = pq.ParquetFile(destination)
        expected, _ = normalized_schema(src.schema_arrow)
        actual, old = normalized_schema(dst.schema_arrow)
        if old or expected != actual:
            return False, "schema mismatch"
        if src.metadata.num_rows != dst.metadata.num_rows:
            return False, "row-count mismatch"
        return True, "valid"
    except Exception as exc:
        return False, repr(exc)


def codec_map(parquet_file, names):
    if not parquet_file.metadata.num_row_groups:
        return {name: "snappy" for name in names}
    row_group = parquet_file.metadata.row_group(0)
    return {
        name: row_group.column(index).compression.lower()
        for index, name in enumerate(names)
    }


def rewrite(source, destination, batch_size):
    src = pq.ParquetFile(source)
    schema, needs_rewrite = normalized_schema(src.schema_arrow)
    if not needs_rewrite:
        raise ValueError("rewrite requested for canonical file")
    temp = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temp.unlink(missing_ok=True)
    writer = None
    try:
        writer = pq.ParquetWriter(
            temp,
            schema,
            compression=codec_map(src, schema.names),
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in src.iter_batches(batch_size=batch_size, use_threads=False):
            writer.write_batch(batch.rename_columns(schema.names))
        writer.close()
        writer = None
        check = pq.ParquetFile(temp)
        if check.metadata.num_rows != src.metadata.num_rows:
            raise RuntimeError("temporary output row-count mismatch")
        if check.schema_arrow != schema:
            raise RuntimeError("temporary output schema mismatch")
        os.replace(temp, destination)
    except BaseException:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)
        raise


def link_or_copy(source, destination, force_copy):
    temp = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temp.unlink(missing_ok=True)
    if force_copy:
        shutil.copy2(source, temp)
        status = "copied"
    else:
        try:
            os.link(source, temp)
            status = "linked"
        except OSError as exc:
            if exc.errno not in (
                errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP
            ):
                raise
            shutil.copy2(source, temp)
            status = "copied"
    os.replace(temp, destination)
    return status


def process(task):
    label, source_text, destination_text, batch_size, force_copy = task
    started = time.monotonic()
    source = Path(source_text)
    destination = Path(destination_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, reason = valid(source, destination)
    if ok:
        status = "skipped"
    else:
        if destination.exists():
            raise RuntimeError(f"refusing to replace invalid {destination}: {reason}")
        for stale in destination.parent.glob(f".{destination.name}.tmp-*"):
            stale.unlink(missing_ok=True)
        src = pq.ParquetFile(source)
        _, needs_rewrite = normalized_schema(src.schema_arrow)
        if needs_rewrite:
            rewrite(source, destination, batch_size)
            status = "rewritten"
        else:
            status = link_or_copy(source, destination, force_copy)
        ok, reason = valid(source, destination)
        if not ok:
            raise RuntimeError(f"post-write validation failed: {reason}")
    parquet_file = pq.ParquetFile(source)
    return {
        "source_label": label,
        "source_path": source_text,
        "destination_path": destination_text,
        "status": status,
        "rows": parquet_file.metadata.num_rows,
        "bytes": source.stat().st_size,
        "seconds": time.monotonic() - started,
    }


def discover(args):
    tasks = []
    destinations = {}
    for label, root in (("mala", args.mala_root), ("nllb", args.nllb_root)):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for source in sorted(root.rglob("*.parquet")):
            destination = args.output_root / source.relative_to(root)
            if destination in destinations:
                raise RuntimeError(
                    f"collision: {destinations[destination]} and {source}"
                )
            destinations[destination] = source
            tasks.append((
                label,
                str(source),
                str(destination),
                args.batch_size,
                args.copy_canonical,
            ))
    if not tasks:
        raise RuntimeError("no Parquet files found")
    return tasks


def log_record(handle, record):
    record = dict(record)
    record["time"] = time.strftime("%FT%T%z")
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def main():
    args = arguments()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch-size must be positive")
    tasks = discover(args)
    mala = sum(task[0] == "mala" for task in tasks)
    print(
        f"Discovered {len(tasks):,} files: "
        f"MaLA-BI={mala:,}, NLLB={len(tasks) - mala:,}"
    )
    print(f"Output={args.output_root}; workers={args.workers}", flush=True)
    if args.dry_run:
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.state_log.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state_log.with_suffix(args.state_log.suffix + ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another merge process is already running") from exc

    totals = {
        "linked": 0,
        "copied": 0,
        "rewritten": 0,
        "skipped": 0,
        "failed": 0,
    }
    rows = 0
    started = time.monotonic()
    with args.state_log.open("a", buffering=1) as state:
        log_record(state, {"event": "start", "files": len(tasks)})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                    totals[result["status"]] += 1
                    rows += result["rows"]
                    log_record(state, result)
                    if count % 25 == 0 or result["status"] == "rewritten":
                        print(
                            f"[{count:,}/{len(tasks):,}] {result['status']} "
                            f"{result['destination_path']}",
                            flush=True,
                        )
                except BaseException as exc:
                    totals["failed"] += 1
                    log_record(state, {
                        "event": "failed",
                        "source_path": task[1],
                        "error": repr(exc),
                    })
                    print(f"FAILED {task[1]}: {exc}", flush=True)
        log_record(state, {
            "event": "finish",
            "totals": totals,
            "rows": rows,
            "seconds": time.monotonic() - started,
        })
    print(f"Finished: {totals}; rows={rows:,}", flush=True)
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
