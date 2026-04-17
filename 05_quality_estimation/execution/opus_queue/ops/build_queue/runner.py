from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from execution.opus_queue.db import (
    connect,
    count_done_jobs,
    delete_pending_for_pair,
    fetch_existing_models,
    initialize,
    log_event,
    reset_pending_for_model,
)
from execution.opus_queue.ops.build_queue.summary import log_summary
from execution.opus_queue.ops.lookup_reader import read_lookup_rows, split_direction
from execution.opus_queue.planning import count_cache, parse_shard_size_overrides, plan
from utils.logger import logger

__all__ = ["run"]


def _handle_reassign(
    conn: sqlite3.Connection,
    direction_key: str,
    new_model: str,
    *,
    force: bool,
) -> bool:
    existing_models = [
        m for m in fetch_existing_models(conn, direction_key) if m != new_model
    ]
    if not existing_models:
        return False

    blocked_models = [
        m
        for m in existing_models
        if count_done_jobs(conn, m, direction_key=direction_key) > 0
    ]
    if blocked_models and not force:
        raise SystemExit(
            f"--reassign would discard 'done' rows for {direction_key} (old={', '.join(blocked_models)}, new={new_model}). Re-run with --force."
        )
    for model in existing_models:
        delete_pending_for_pair(conn, direction_key, model, force=force)
    return True


def run(args) -> None:
    lookup_path = Path(args.lookup).expanduser().resolve()
    opus_root = Path(args.opus_root).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    overrides = parse_shard_size_overrides(args.shard_size_override)

    if not lookup_path.exists():
        raise SystemExit(f"Lookup file not found: {lookup_path}")
    if lookup_path.suffix.lower() != ".csv":
        raise SystemExit(
            f"Lookup file must be a .csv exported for OPUS queueing, got: {lookup_path}"
        )
    if not opus_root.exists():
        raise SystemExit(f"OPUS root not found: {opus_root}")

    logger.info("Reading lookup CSV: %s", lookup_path)
    rows = read_lookup_rows(lookup_path)
    logger.info("Parsed %d direction rows from lookup.", len(rows))

    cache_path = count_cache.resolve_path(args.count_cache, opus_root)
    cache = count_cache.load(cache_path)
    logger.info("Row-count cache: %s (%d entries)", cache_path, len(cache))

    conn: sqlite3.Connection | None = None
    if not args.dry_run:
        conn = connect(db_path)
        initialize(conn)
        if args.reset_pending_for_model:
            logger.warning(
                "Resetting jobs for model=%s (force=%s)",
                args.reset_pending_for_model,
                args.force,
            )
            reset_pending_for_model(
                conn, args.reset_pending_for_model, force=args.force
            )

    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"directions": 0, "shards": 0, "sentences": 0}
    )
    total_directions = 0
    total_shards = 0
    total_sentences = 0
    skipped_missing_dirs = 0
    skipped_empty = 0
    reassigned = 0
    inserted_jobs = 0

    try:
        for row in rows:
            direction_key = row["direction_key"]
            model = row["winner_model"]
            est_hours = row["est_hours"]
            src_lang, tgt_lang = split_direction(direction_key)

            direction_dir = opus_root / direction_key
            if not direction_dir.exists() or not direction_dir.is_dir():
                logger.warning(
                    "OPUS dir missing for %s; skipping (%s)", direction_key, direction_dir
                )
                skipped_missing_dirs += 1
                continue

            cached = cache.get(direction_key)
            if cached is None:
                n_sentences = count_cache.count_direction_rows(direction_dir)
                cache[direction_key] = n_sentences
            else:
                n_sentences = int(cached)

            if n_sentences == 0:
                logger.warning("OPUS dir empty for %s; skipping", direction_key)
                skipped_empty += 1
                continue

            shard_size, shard_ranges = plan(model, n_sentences, overrides)
            total_directions += 1
            total_shards += len(shard_ranges)
            total_sentences += n_sentences
            bucket = summary[model]
            bucket["directions"] += 1
            bucket["shards"] += len(shard_ranges)
            bucket["sentences"] += n_sentences

            if args.dry_run or conn is None:
                continue

            if args.reassign and _handle_reassign(
                conn, direction_key, model, force=args.force
            ):
                reassigned += 1

            conn.execute(
                """
                INSERT OR IGNORE INTO directions
                    (direction_key, model, src_lang, tgt_lang,
                     n_sentences, shard_size, est_hours, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))
                """,
                (
                    direction_key,
                    model,
                    src_lang,
                    tgt_lang,
                    n_sentences,
                    shard_size,
                    est_hours,
                ),
            )

            for shard in shard_ranges:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs
                        (direction_key, model, shard_id,
                         start_idx, end_idx, status, attempts)
                    VALUES (?, ?, ?, ?, ?, 'pending', 0)
                    """,
                    (
                        direction_key,
                        model,
                        shard.shard_id,
                        shard.start_idx,
                        shard.end_idx,
                    ),
                )
                if cur.rowcount:
                    inserted_jobs += 1
    finally:
        if not args.dry_run and cache:
            count_cache.save(cache_path, cache)

    log_summary(
        summary=summary,
        total_directions=total_directions,
        total_shards=total_shards,
        total_sentences=total_sentences,
        inserted_jobs=inserted_jobs,
        skipped_missing_dirs=skipped_missing_dirs,
        skipped_empty=skipped_empty,
        reassigned=reassigned,
        overrides=overrides,
    )

    if conn is not None:
        detail = (
            f"directions={total_directions} shards={total_shards} "
            f"sentences={total_sentences} inserted={inserted_jobs} "
            f"skipped_missing={skipped_missing_dirs} skipped_empty={skipped_empty} "
            f"reassigned={reassigned}"
        )
        log_event(conn, None, "build_queue", detail=detail)
        conn.close()
