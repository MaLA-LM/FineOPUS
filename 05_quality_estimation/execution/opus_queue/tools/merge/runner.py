from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from execution.opus_queue import db as queue_db
from execution.opus_queue.manifest.reader import manifest_path
from execution.opus_queue.tools.merge.collect import (
    CompletedShard,
    collect_complete_combined_directions,
)
from execution.opus_queue.tools.merge.cleanup import delete_merged_direction_inputs
from execution.opus_queue.tools.merge.convert import merge_direction
from utils.logger import logger

__all__ = ["run"]


@dataclass(frozen=True)
class _MergeTask:
    db_path: Path | None
    output_base: Path
    merged_base: Path
    source_ref: Path | str
    direction_key: str
    model: str
    winners: dict[int, CompletedShard] | None
    force: bool


@dataclass(frozen=True)
class _MergeResult:
    direction_key: str
    model: str
    ok: bool
    shards: int
    rows: int


def _merge_one_direction(task: _MergeTask) -> _MergeResult:
    conn = (
        queue_db.connect(task.db_path)
        if task.db_path is not None and task.winners is None
        else None
    )
    try:
        ok, shards, rows = merge_direction(
            conn,
            task.output_base,
            task.merged_base,
            task.source_ref,
            task.direction_key,
            task.model,
            force=task.force,
            winners=task.winners,
        )
    finally:
        if conn is not None:
            conn.close()
    return _MergeResult(
        direction_key=task.direction_key,
        model=task.model,
        ok=ok,
        shards=shards,
        rows=rows,
    )


def _add_result(
    result: _MergeResult, merged: int, total_shards: int, total_rows: int
) -> tuple[int, int, int]:
    if result.ok:
        merged += 1
        total_shards += result.shards
        total_rows += result.rows
    return merged, total_shards, total_rows


def _has_manifest_source(
    manifest_root: str | None,
    build_tag: str | None,
    trace_root: str | None,
) -> bool:
    return bool(manifest_root or build_tag or trace_root)


def _source_ref(
    db_path: Path | None,
    manifest_root: str | None,
    build_tag: str | None,
) -> str:
    sources: list[str] = []
    if db_path is not None:
        sources.append(str(db_path))
    if manifest_root and build_tag:
        sources.append(str(manifest_path(manifest_root, build_tag)))
    return " + ".join(sources)


def run(args) -> None:
    db_arg = getattr(args, "db", None)
    manifest_root = getattr(args, "manifest_root", None)
    build_tag = getattr(args, "build_tag", None)
    trace_root = getattr(args, "trace_root", None)
    has_manifest_source = _has_manifest_source(manifest_root, build_tag, trace_root)
    db_path = Path(db_arg).expanduser().resolve() if db_arg else None
    output_base = Path(args.output_base).expanduser().resolve()
    merged_base = Path(args.merged_base).expanduser().resolve()
    jobs = int(getattr(args, "jobs", 1) or 1)
    if jobs < 1:
        raise ValueError("--jobs must be >= 1")

    cleanup_mode = bool(getattr(args, "cleanup_merged_inputs", False))
    if cleanup_mode:
        result = delete_merged_direction_inputs(
            output_base,
            merged_base,
            model_filter=args.model,
            dry_run=bool(getattr(args, "dry_run", False)),
        )
        logger.info(
            (
                "Merged-input cleanup complete: deleted=%d missing=%d "
                "skipped_unmerged=%d files=%d bytes=%d dry_run=%s"
            ),
            result.deleted_dirs,
            result.missing_source_dirs,
            result.skipped_unmerged_dirs,
            result.deleted_files,
            result.deleted_bytes,
            result.dry_run,
        )
        return

    if bool(getattr(args, "dry_run", False)):
        raise SystemExit("--dry-run is only valid with --delete-merged-directions.")

    merged_base.mkdir(parents=True, exist_ok=True)

    if has_manifest_source:
        if not manifest_root or not build_tag or not trace_root:
            raise SystemExit(
                "--manifest-root, --build-tag, and --trace-root are required together."
            )
    if db_path is None and not has_manifest_source:
        raise SystemExit(
            "provide --db and/or all of --manifest-root, --build-tag, and --trace-root."
        )

    conn = queue_db.connect(db_path) if db_path is not None else None
    try:
        if conn is not None:
            queue_db.initialize(conn)
        completed = collect_complete_combined_directions(
            conn,
            args.model,
            manifest_root=manifest_root if has_manifest_source else None,
            build_tag=build_tag if has_manifest_source else None,
            trace_root=trace_root if has_manifest_source else None,
        )
        source_ref = _source_ref(db_path, manifest_root, build_tag)
        directions = [
            _MergeTask(
                db_path=None,
                output_base=output_base,
                merged_base=merged_base,
                source_ref=source_ref,
                direction_key=item.direction_key,
                model=item.model,
                winners=item.winners,
                force=args.force,
            )
            for item in completed
        ]

        logger.info(
            "Found %d complete direction(s) to merge (model filter=%s).",
            len(directions),
            args.model,
        )
    finally:
        if conn is not None:
            conn.close()
            conn = None

    merged = 0
    total_rows = 0
    total_shards = 0
    if jobs == 1 or len(directions) <= 1:
        for task in directions:
            ok, shards, rows = merge_direction(
                None,
                output_base,
                merged_base,
                task.source_ref,
                task.direction_key,
                task.model,
                force=task.force,
                winners=task.winners,
            )
            result = _MergeResult(task.direction_key, task.model, ok, shards, rows)
            merged, total_shards, total_rows = _add_result(
                result, merged, total_shards, total_rows
            )
    else:
        max_workers = min(jobs, len(directions))
        logger.info("Merging directions in parallel: jobs=%d", max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_merge_one_direction, task): task
                for task in directions
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception(
                        "Merge failed for %s/%s",
                        task.model,
                        task.direction_key,
                    )
                    raise
                merged, total_shards, total_rows = _add_result(
                    result, merged, total_shards, total_rows
                )
    logger.info(
        "Merge complete: directions=%d winning_shards=%d rows=%d",
        merged,
        total_shards,
        total_rows,
    )
