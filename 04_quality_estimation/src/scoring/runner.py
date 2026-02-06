from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from dataset.manifest import ManifestEntry, make_lock_id, read_manifest_entries
from dataset.mediator import DatasetAdapter
from src.scoring.io import is_complete_parquet, write_parquet_atomic
from src.scoring.worker import release_lock, try_acquire_lock

ResolveOutputPath = Callable[[object, str, str, str, str], Path]
ScoreEntry = Callable[[ManifestEntry], object]


def collect_directions(args, dataset: DatasetAdapter) -> list[ManifestEntry]:
    if args.manifest:
        return read_manifest_entries(args.manifest)
    if args.discover_all:
        return [
            ManifestEntry(
                src_lang=src,
                tgt_lang=tgt,
                split=split,
                lock_id=make_lock_id(src, tgt, split),
            )
            for src, tgt, split, _path in dataset.discover_directions(
                args.root, split=args.split
            )
        ]
    return [
        ManifestEntry(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            split=args.split,
            lock_id=make_lock_id(args.src_lang, args.tgt_lang, args.split),
        )
    ]


def run_scoring(
    args,
    dataset: DatasetAdapter,
    directions: list[ManifestEntry],
    model_tag: str,
    resolve_output_path: ResolveOutputPath,
    score_entry: ScoreEntry,
) -> None:
    if args.worker:
        _run_worker(
            args, dataset, directions, model_tag, resolve_output_path, score_entry
        )
    else:
        _run_single(
            args, dataset, directions, model_tag, resolve_output_path, score_entry
        )


def _run_single(
    args,
    dataset: DatasetAdapter,
    directions: list[ManifestEntry],
    model_tag: str,
    resolve_output_path: ResolveOutputPath,
    score_entry: ScoreEntry,
) -> None:
    pending: list[tuple[ManifestEntry, Path]] = []
    for entry in directions:
        output_path = resolve_output_path(
            args, model_tag, entry.src_lang, entry.tgt_lang, entry.split
        )
        expected_rows = dataset.expected_detail_rows(
            args.root, entry.split, entry.src_lang, args.max_rows
        )
        if args.resume and is_complete_parquet(output_path, expected_rows):
            print(f"SKIP (exists): {output_path}")
            continue
        if args.resume and output_path.exists():
            print(f"Recomputing (invalid output): {output_path}")
        pending.append((entry, output_path))

    if not pending:
        return

    for entry, output_path in pending:
        print(f"Scoring {entry.src_lang}->{entry.tgt_lang} split={entry.split}")
        frame = score_entry(entry)
        write_parquet_atomic(frame, output_path)
        print(f"Wrote: {output_path}")


def _run_worker(
    args,
    dataset: DatasetAdapter,
    directions: list[ManifestEntry],
    model_tag: str,
    resolve_output_path: ResolveOutputPath,
    score_entry: ScoreEntry,
) -> None:
    max_files = getattr(args, "worker_max_files", 200)
    if max_files is not None and max_files <= 0:
        max_files = None
    completed = 0
    lock_root = Path(args.output_base) / ".locks" / f"model={model_tag}"
    owner = (
        f"job={os.getenv('SLURM_JOB_ID', '')} "
        f"task={os.getenv('SLURM_ARRAY_TASK_ID', '')} "
        f"pid={os.getpid()}"
    )

    start_index = 0  # Track where to resume searching

    while True:
        if max_files is not None and completed >= max_files:
            print(f"Worker reached target of {completed} files.")
            return

        claimed = False
        pending_locked = False

        # Search from last position, wrap around if needed
        for offset in range(len(directions)):
            idx = (start_index + offset) % len(directions)
            entry = directions[idx]

            if max_files is not None and completed >= max_files:
                print(f"Worker reached target of {completed} files.")
                return

            output_path = resolve_output_path(
                args, model_tag, entry.src_lang, entry.tgt_lang, entry.split
            )
            expected_rows = dataset.expected_detail_rows(
                args.root, entry.split, entry.src_lang, args.max_rows
            )
            if args.resume and is_complete_parquet(output_path, expected_rows):
                print(f"SKIP (exists): {output_path}")
                continue

            lock_path = try_acquire_lock(
                lock_root, entry.lock_id, owner, timeout_seconds=7200
            )
            if lock_path is None:
                pending_locked = True
                continue

            try:
                if args.resume and is_complete_parquet(output_path, expected_rows):
                    print(f"SKIP (exists): {output_path}")
                    continue

                print(f"Scoring {entry.src_lang}->{entry.tgt_lang} split={entry.split}")
                frame = score_entry(entry)
                write_parquet_atomic(frame, output_path)
                print(f"Wrote: {output_path}")
                claimed = True
                completed += 1
                start_index = (idx + 1) % len(directions)  # Resume from next
            finally:
                release_lock(lock_path)

            if claimed:
                break

        if not claimed:
            if pending_locked:
                time.sleep(30)
                continue
            return
