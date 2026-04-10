#!/usr/bin/env python3
"""
Split large parquet files so each shard has at most MAX_ROWS rows.
Shards are renumbered from 0 after splitting.

Usage (in-place, requires write permission on root_dir):
    python split_parquet.py <root_dir> [--max-rows 10000000] [--workers 4]

Usage (write to a separate output dir, source is read-only):
    python split_parquet.py <root_dir> --output-dir <out_dir> [--max-rows 10000000] [--workers 4]

When --output-dir is given:
  - Large files (> max-rows) are split and written as shards under output-dir.
  - Small files are symlinked into output-dir (no data copy, saves space).
  - Source files are never modified.

When --output-dir is not given (in-place mode):
  - Splits are written atomically via temp files, then the original is removed.
  - Requires write permission on the source directory.
"""

import argparse
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import shutil

import pyarrow.parquet as pq


DEFAULT_MAX_ROWS = 10_000_000


def get_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _base_name(stem: str) -> str:
    """Strip trailing _shard_N from a parquet stem."""
    return re.sub(r"_shard_\d+$", "", stem)


def split_file_to(src: Path, out_dir: Path, base: str, start_idx: int, max_rows: int) -> list[Path]:
    """
    Stream-split src into ≤ max_rows-row shards written under out_dir.
    Shards are numbered starting from start_idx.
    Returns list of written paths.
    """
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_paths: list[tuple[Path, int]] = []
    shard_idx = start_idx
    writer = None
    rows_in_current = 0
    tmp_path = None

    def _new_writer():
        nonlocal writer, rows_in_current, tmp_path
        fd_path = out_dir / f".tmp_shard_{shard_idx}.parquet"
        writer = pq.ParquetWriter(str(fd_path), schema)
        rows_in_current = 0
        tmp_path = fd_path

    _new_writer()
    tmp_paths.append((tmp_path, shard_idx))

    for batch in pf.iter_batches(batch_size=500_000):
        offset = 0
        while offset < len(batch):
            space_left = max_rows - rows_in_current
            chunk = batch.slice(offset, space_left)
            writer.write_batch(chunk)
            rows_in_current += len(chunk)
            offset += len(chunk)

            if rows_in_current >= max_rows:
                writer.close()
                shard_idx += 1
                _new_writer()
                tmp_paths.append((tmp_path, shard_idx))

    writer.close()

    final_paths = []
    for t, idx in tmp_paths:
        final = out_dir / f"{base}_shard_{idx}.parquet"
        t.rename(final)
        final_paths.append(final)

    return final_paths


def process_directory(dir_path: Path, max_rows: int, dry_run: bool, output_dir: Path | None) -> dict:
    """
    Process all parquet files in dir_path.

    In-place mode (output_dir is None):
      - Large files are split atomically; original is removed; all shards renumbered from 0.
      - Requires write permission on dir_path.

    output-dir mode:
      - Large files are split into out_subdir.
      - Small files are symlinked into out_subdir.
      - Source is never modified.
    """
    parquet_files = sorted(dir_path.glob("*.parquet"))
    if not parquet_files:
        return {}

    results = {"dir": str(dir_path), "split": [], "skipped": [], "errors": []}

    # ── output-dir mode ──────────────────────────────────────────────────────
    if output_dir is not None:
        # output_dir is already the final subdir (pre-computed by _worker).
        out_subdir = output_dir
        if not dry_run:
            out_subdir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        for pf_path in parquet_files:
            try:
                n = get_row_count(pf_path)
                base = _base_name(pf_path.stem)
                if n <= max_rows:
                    results["skipped"].append((pf_path.name, n))
                    if not dry_run:
                        dst = out_subdir / f"{base}_shard_{shard_idx}.parquet"
                        if not dst.exists():
                            shutil.copy2(pf_path, dst)
                    shard_idx += 1
                else:
                    results["split"].append((pf_path.name, n))
                    if dry_run:
                        n_chunks = (n + max_rows - 1) // max_rows
                        print(f"  [DRY-RUN] {pf_path.name} ({n:,} rows) → {n_chunks} shards", flush=True)
                    else:
                        written = split_file_to(pf_path, out_subdir, base, shard_idx, max_rows)
                        shard_idx += len(written)
            except Exception as exc:
                results["errors"].append((pf_path.name, str(exc)))
        return results

    # ── in-place mode ────────────────────────────────────────────────────────
    needs_split = []
    for pf_path in parquet_files:
        try:
            n = get_row_count(pf_path)
            if n <= max_rows:
                results["skipped"].append((pf_path.name, n))
            else:
                results["split"].append((pf_path.name, n))
                needs_split.append(pf_path)
        except Exception as exc:
            results["errors"].append((pf_path.name, str(exc)))

    if dry_run or not needs_split:
        if dry_run:
            for name, n in results["split"]:
                n_chunks = (n + max_rows - 1) // max_rows
                print(f"  [DRY-RUN] {name} ({n:,} rows) → {n_chunks} shards", flush=True)
        return results

    # Split each large file in-place, writing to temp files in the same dir.
    for pf_path in needs_split:
        try:
            base = _base_name(pf_path.stem)
            pf = pq.ParquetFile(pf_path)
            schema = pf.schema_arrow
            tmp_chunks: list[tuple[Path, int]] = []
            shard_idx = 0
            writer = None
            rows_in_current = 0
            tmp_f = None

            def _new_writer_inplace():
                nonlocal writer, rows_in_current, tmp_f
                tmp_f = tempfile.NamedTemporaryFile(
                    dir=dir_path, suffix=".parquet.tmp", delete=False
                )
                writer = pq.ParquetWriter(tmp_f.name, schema)
                rows_in_current = 0

            _new_writer_inplace()
            tmp_chunks.append((Path(tmp_f.name), shard_idx))

            for batch in pf.iter_batches(batch_size=500_000):
                offset = 0
                while offset < len(batch):
                    space_left = max_rows - rows_in_current
                    chunk = batch.slice(offset, space_left)
                    writer.write_batch(chunk)
                    rows_in_current += len(chunk)
                    offset += len(chunk)
                    if rows_in_current >= max_rows:
                        writer.close()
                        shard_idx += 1
                        _new_writer_inplace()
                        tmp_chunks.append((Path(tmp_f.name), shard_idx))

            writer.close()
            pf_path.unlink()
            for tmp, idx in tmp_chunks:
                final = dir_path / f"{base}_shard_{idx}.parquet"
                tmp.rename(final)
        except Exception as exc:
            results["errors"].append((pf_path.name, str(exc)))

    # Renumber all shards in the directory sequentially from 0.
    current_shards = sorted(
        dir_path.glob("*.parquet"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"_shard_(\d+)\.parquet$", p.name)) else -1,
    )
    if not current_shards:
        return results

    base = _base_name(current_shards[0].stem)
    staged = []
    for shard in current_shards:
        tmp = shard.with_suffix(".parquet.renaming")
        shard.rename(tmp)
        staged.append(tmp)
    for new_idx, tmp in enumerate(staged):
        tmp.rename(dir_path / f"{base}_shard_{new_idx}.parquet")

    return results


def collect_directories(root: Path) -> list[Path]:
    """Return all leaf directories that contain at least one .parquet file."""
    dirs = set()
    for p in root.rglob("*.parquet"):
        dirs.add(p.parent)
    return sorted(dirs)


def _print_result(res: dict, i: int, total: int):
    if not res:
        return
    parts = []
    if res["split"]:
        parts.append("split: " + ", ".join(f"{n}({r:,})" for n, r in res["split"]))
    if res["errors"]:
        parts.append("ERRORS: " + str(res["errors"]))
    if parts:
        print(f"[{i}/{total}] {res['dir']} — {'; '.join(parts)}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root_dir", type=Path, help="Root directory to scan (source)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Write output here instead of modifying source in-place. "
             "Small files are symlinked; large files are split into shards."
    )
    parser.add_argument(
        "--max-rows", type=int, default=DEFAULT_MAX_ROWS,
        help=f"Maximum rows per shard (default: {DEFAULT_MAX_ROWS:,})"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be split; do not modify any files")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--chunk", type=int, default=None,
                        help="SLURM array task: 0-indexed chunk index")
    parser.add_argument("--total-chunks", type=int, default=None,
                        help="SLURM array task: total number of array tasks")
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        sys.exit(f"Error: {args.root_dir} is not a directory")

    if args.output_dir is not None and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = collect_directories(args.root_dir)
    print(f"Found {len(all_dirs)} directories with parquet files under {args.root_dir}")

    if args.chunk is not None and args.total_chunks is not None:
        all_dirs = [d for i, d in enumerate(all_dirs) if i % args.total_chunks == args.chunk]
        print(f"Chunk {args.chunk}/{args.total_chunks}: processing {len(all_dirs)} directories")

    if args.dry_run:
        print("DRY-RUN mode: no files will be modified\n")

    mode = f"→ {args.output_dir}" if args.output_dir else "in-place"
    print(f"Mode: {mode}  |  max-rows: {args.max_rows:,}  |  workers: {args.workers}\n")

    total_split = total_skipped = total_errors = 0

    def _submit(d):
        # Remap output dir to mirror the source subdirectory structure.
        out = args.output_dir / d.relative_to(args.root_dir) if args.output_dir else None
        return process_directory(d, args.max_rows, args.dry_run, out)

    if args.workers > 1:
        # Use a top-level helper so it can be pickled by ProcessPoolExecutor.
        _worker_args = (args.root_dir, args.max_rows, args.dry_run, args.output_dir)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_worker, d, *_worker_args): d
                for d in all_dirs
            }
            for i, fut in enumerate(as_completed(futures), 1):
                d = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"[{i}/{len(all_dirs)}] ERROR {d}: {exc}", flush=True)
                    total_errors += 1
                    continue
                _print_result(res, i, len(all_dirs))
                total_split += len(res.get("split", []))
                total_skipped += len(res.get("skipped", []))
                total_errors += len(res.get("errors", []))
    else:
        for i, d in enumerate(all_dirs, 1):
            res = _submit(d)
            _print_result(res, i, len(all_dirs))
            total_split += len(res.get("split", []))
            total_skipped += len(res.get("skipped", []))
            total_errors += len(res.get("errors", []))

    print(f"\nDone. split={total_split}  skipped={total_skipped}  errors={total_errors}")


def _worker(d: Path, root_dir: Path, max_rows: int, dry_run: bool, output_dir: Path | None) -> dict:
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    out = output_dir / d.relative_to(root_dir) if output_dir else None
    return process_directory(d, max_rows, dry_run, out)


if __name__ == "__main__":
    main()
