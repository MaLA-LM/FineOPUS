#!/usr/bin/env python3
"""
Sample a fixed number of rows from each language-pair subdirectory of a
FineOPUS-style parquet dataset.

Each language pair is expected to live in its own subdirectory, e.g.:
    input_dir/eng_Latn-deu_Latn/eng_Latn-deu_Latn_part_000.parquet

For every pair, the script randomly samples up to N rows (default 5000) and
writes a single parquet file under the mirrored output subdirectory.

Usage:
    python sample_subset.py \\
        --input_dir /scratch/project_462001249/MaLA-LM/FineOPUS-ReLID \\
        --output_dir /scratch/project_462001249/MaLA-LM/FineOPUS-ReLID-sample5k \\
        --sample_size 5000 \\
        --seed 42 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample N rows per language pair from a FineOPUS-style dataset."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Root directory containing one subdirectory per language pair.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Root directory for sampled output (mirrors language-pair subdirs).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="Maximum number of rows to sample per language pair (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; each pair gets seed + hash(pair) for reproducibility.",
    )
    parser.add_argument(
        "--lang_pair",
        type=str,
        help="Process a single language pair subdirectory name.",
    )
    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        help="Process one or more language pair subdirectory names.",
    )
    parser.add_argument(
        "--lang_pairs_file",
        type=Path,
        help="Text file with one language pair subdirectory name per line.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="SLURM array chunk index (0-based).",
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=None,
        help="Total number of SLURM array chunks.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000,
        help="Parquet read batch size while streaming rows (default: 10000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet files.",
    )
    return parser.parse_args()


def _pair_seed(base_seed: int, lang_pair: str) -> int:
    return base_seed + (hash(lang_pair) & 0x7FFFFFFF)


def _list_parquet_files(pair_dir: Path) -> list[Path]:
    return sorted(p for p in pair_dir.iterdir() if p.is_file() and p.suffix == ".parquet")


def _count_total_rows(parquet_files: list[Path]) -> int:
    total = 0
    for path in parquet_files:
        total += pq.read_metadata(path).num_rows
    return total


def sample_pair_to_file(
    pair_dir: Path,
    out_file: Path,
    sample_size: int,
    seed: int,
    batch_size: int,
) -> int:
    """
    Stream sampled rows directly to out_file.

    Avoid accumulating batch.slice() results in memory: each slice keeps a
    reference to its parent batch, so collecting thousands of slices from a
    hundred-million-row file effectively loads the whole file into RAM.
    """
    parquet_files = _list_parquet_files(pair_dir)
    if not parquet_files:
        return 0

    total_rows = _count_total_rows(parquet_files)
    if total_rows == 0:
        return 0

    k = min(sample_size, total_rows)
    chosen = sorted(random.Random(seed).sample(range(total_rows), k))
    chosen_iter = iter(chosen)
    next_chosen = next(chosen_iter, None)

    writer: pq.ParquetWriter | None = None
    sampled_rows = 0
    global_idx = 0

    try:
        for pf_path in parquet_files:
            if next_chosen is None:
                break
            for batch in pq.ParquetFile(pf_path).iter_batches(batch_size=batch_size):
                batch_len = len(batch)
                batch_end = global_idx + batch_len
                while next_chosen is not None and next_chosen < batch_end:
                    row_batch = batch.slice(next_chosen - global_idx, 1)
                    if writer is None:
                        writer = pq.ParquetWriter(out_file, row_batch.schema, compression="snappy")
                    writer.write_batch(row_batch)
                    sampled_rows += 1
                    del row_batch
                    next_chosen = next(chosen_iter, None)
                global_idx = batch_end
                del batch
    finally:
        if writer is not None:
            writer.close()

    gc.collect()
    return sampled_rows


def _collect_requested_pairs(args: argparse.Namespace) -> set[str] | None:
    requested: set[str] = set()
    if args.lang_pair:
        requested.add(args.lang_pair)
    if args.lang_pairs:
        requested.update(args.lang_pairs)
    if args.lang_pairs_file:
        with args.lang_pairs_file.open() as f:
            for line in f:
                pair = line.strip()
                if pair and not pair.startswith("#"):
                    requested.add(pair)
    return requested or None


def _discover_pair_dirs(input_dir: Path, requested: set[str] | None) -> list[Path]:
    pair_dirs: list[Path] = []
    for entry in sorted(input_dir.iterdir()):
        if not entry.is_dir():
            continue
        if requested is not None and entry.name not in requested:
            continue
        pair_dirs.append(entry)
    return pair_dirs


def _apply_chunking(pair_dirs: list[Path], chunk: int | None, total_chunks: int | None) -> list[Path]:
    if chunk is None or total_chunks is None:
        return pair_dirs
    return [d for i, d in enumerate(pair_dirs) if i % total_chunks == chunk]


def _process_pair(task: tuple) -> dict:
    pair_dir, output_dir, sample_size, base_seed, batch_size, overwrite = task
    lang_pair = pair_dir.name
    out_pair_dir = output_dir / lang_pair
    out_file = out_pair_dir / f"{lang_pair}_sample_{sample_size}.parquet"

    result = {
        "lang_pair": lang_pair,
        "total_rows": 0,
        "sampled_rows": 0,
        "output_file": str(out_file),
        "status": "ok",
        "error": "",
    }

    try:
        parquet_files = _list_parquet_files(pair_dir)
        result["total_rows"] = _count_total_rows(parquet_files)

        if result["total_rows"] == 0:
            result["status"] = "skipped"
            result["error"] = "no rows"
            return result

        if out_file.exists() and not overwrite:
            result["status"] = "skipped"
            result["error"] = "output exists"
            result["sampled_rows"] = min(sample_size, result["total_rows"])
            return result

        out_pair_dir.mkdir(parents=True, exist_ok=True)
        sampled_rows = sample_pair_to_file(
            pair_dir=pair_dir,
            out_file=out_file,
            sample_size=sample_size,
            seed=_pair_seed(base_seed, lang_pair),
            batch_size=batch_size,
        )
        if sampled_rows == 0:
            result["status"] = "skipped"
            result["error"] = "empty sample"
            if out_file.exists():
                out_file.unlink()
            return result

        result["sampled_rows"] = sampled_rows
        logger.info(
            "%s: sampled %d / %d -> %s",
            lang_pair,
            result["sampled_rows"],
            result["total_rows"],
            out_file,
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        logger.error("%s: %s", lang_pair, exc)

    return result


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Error: input_dir does not exist: {args.input_dir}")

    requested = _collect_requested_pairs(args)
    pair_dirs = _discover_pair_dirs(args.input_dir, requested)
    pair_dirs = _apply_chunking(pair_dirs, args.chunk, args.total_chunks)

    if not pair_dirs:
        logger.warning("No language-pair directories found under %s", args.input_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk is not None and args.total_chunks is not None:
        logger.info(
            "Chunk %d/%d: processing %d language pairs",
            args.chunk,
            args.total_chunks,
            len(pair_dirs),
        )
    else:
        logger.info("Processing %d language pairs", len(pair_dirs))

    tasks = [
        (
            pair_dir,
            args.output_dir,
            args.sample_size,
            args.seed,
            args.batch_size,
            args.overwrite,
        )
        for pair_dir in pair_dirs
    ]

    results: list[dict] = []
    workers = max(1, args.workers)
    workers = min(workers, len(tasks))

    if workers == 1:
        for task in tasks:
            results.append(_process_pair(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_pair, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    total_sampled = sum(r["sampled_rows"] for r in results if r["status"] == "ok")

    logger.info(
        "Done. ok=%d skipped=%d errors=%d total_sampled_rows=%d",
        ok,
        skipped,
        errors,
        total_sampled,
    )


if __name__ == "__main__":
    main()
