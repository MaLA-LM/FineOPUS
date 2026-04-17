from __future__ import annotations

import argparse
import os
import socket
import time
import traceback
from pathlib import Path

from execution.flores_array.manifest import ManifestEntry
from execution.opus_queue import db as queue_db
from execution.opus_queue.planning import (
    expected_shard_seconds,
    parse_shard_size_overrides,
)
from execution.opus_queue.scoring import build_scorer, resolve_backend
from execution.opus_queue.worker.commit import commit_shard
from execution.opus_queue.worker.shard_io import shard_output_path
from execution.opus_queue.worker.shard_loader import build_sharding_opus_adapter
from execution.opus_queue.worker.walltime import remaining_seconds
from utils.logger import logger

__all__ = ["run_loop"]


def _make_worker_id() -> str:
    job = os.environ.get("SLURM_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    host = socket.gethostname()
    pid = os.getpid()
    return f"{job}.{task}.{host}.{pid}"


def _score_shard(run_entry, context, entry, start_idx, end_idx, direction_key, shard_id):
    context.bounds.start = start_idx
    context.bounds.end = end_idx
    context.bounds.active = True

    shard_start = time.time()
    frame = run_entry(entry)
    elapsed = time.time() - shard_start

    frame["direction_key"] = direction_key
    frame["shard_id"] = shard_id
    return frame, elapsed


def run_loop(args: argparse.Namespace) -> int:
    start_ts = time.time()
    worker_id = _make_worker_id()
    queue_model = args.model
    scorer_model = args.scorer_model or queue_model
    backend = args.backend or resolve_backend(scorer_model)
    parse_shard_size_overrides(args.shard_size_override)
    exp_seconds = expected_shard_seconds(queue_model)
    output_base = Path(args.output_base).expanduser().resolve()
    opus_root = args.opus_root

    logger.info(
        "Worker starting: id=%s queue_model=%s scorer_model=%s backend=%s walltime=%s expected_shard=%ds",
        worker_id,
        queue_model,
        scorer_model,
        backend,
        args.walltime_seconds,
        exp_seconds,
    )

    context = build_sharding_opus_adapter()
    dataset = context.adapter

    if opus_root is None:
        opus_root = str(dataset.default_root)
    args.root = opus_root
    args.output_base = str(output_base)
    args.dataset = dataset.id
    args.execution = "opus_queue"

    scorer_args = argparse.Namespace(**vars(args))
    scorer_args.model = scorer_model
    run_entry, _scorer_model_tag = build_scorer(backend, scorer_args, dataset)

    conn = queue_db.connect(args.db)
    queue_db.initialize(conn)
    queue_db.log_event(
        conn,
        worker_id,
        "start",
        model=queue_model,
        detail=f"queue_model={queue_model} scorer_model={scorer_model} backend={backend}",
    )
    reset_count = queue_db.reset_own_stale(conn, worker_id)
    if reset_count:
        logger.warning("Reset %d stale rows owned by %s", reset_count, worker_id)

    completed_shards = 0
    failed_shards = 0

    try:
        while True:
            time_left = remaining_seconds(start_ts, args.walltime_seconds)
            if time_left < 1.5 * exp_seconds:
                logger.info(
                    "Walltime budget low (%.0fs left < 1.5 * %ds); exiting cleanly.",
                    time_left, exp_seconds,
                )
                queue_db.log_event(conn, worker_id, "exit", detail="walltime_budget")
                break

            job = queue_db.claim_next(conn, queue_model, worker_id, retries=args.claim_retries)
            if job is None:
                counts = queue_db.count_by_status(conn, queue_model)
                if counts:
                    logger.info(
                        "Queue drained for model=%s; status_counts=%s; exiting.",
                        queue_model,
                        counts,
                    )
                else:
                    pending_models = conn.execute(
                        """
                        SELECT model, COUNT(*) AS n
                          FROM jobs
                         WHERE status = 'pending'
                         GROUP BY model
                         ORDER BY n DESC, model ASC
                         LIMIT 10
                        """
                    ).fetchall()
                    pending_summary = ", ".join(
                        f"{row['model']}:{row['n']}" for row in pending_models
                    ) or "<none>"
                    logger.warning(
                        "Queue drained for model=%s because the DB has no rows for that exact model key. Pending models: %s",
                        queue_model,
                        pending_summary,
                    )
                queue_db.log_event(
                    conn, worker_id, "exit", model=queue_model, detail="queue_drained"
                )
                break

            direction_key = job["direction_key"]
            shard_id = int(job["shard_id"])
            start_idx = int(job["start_idx"])
            end_idx = int(job["end_idx"])
            attempt = int(job["attempts"])
            logger.info(
                "Claimed shard dir=%s shard=%d range=[%d,%d) attempt=%d",
                direction_key, shard_id, start_idx, end_idx, attempt,
            )
            queue_db.log_event(
                conn,
                worker_id,
                "claim",
                direction_key=direction_key,
                model=queue_model,
                shard_id=shard_id,
                detail=f"attempt={attempt}",
            )

            try:
                src_lang, tgt_lang = direction_key.split("-", 1)
                entry = ManifestEntry(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    split=dataset.split_values[0],
                )
                frame, elapsed = _score_shard(
                    run_entry, context, entry, start_idx, end_idx, direction_key, shard_id
                )
                out_path = shard_output_path(output_base, args.model, direction_key, shard_id)
                if commit_shard(
                    conn, frame, out_path, direction_key, queue_model,
                    shard_id, worker_id, elapsed,
                ):
                    completed_shards += 1
            except Exception:
                failed_shards += 1
                tb = traceback.format_exc()
                logger.error(
                    "Shard failed dir=%s shard=%d:\n%s",
                    direction_key, shard_id, tb,
                )
                finalize_result = queue_db.mark_failed(
                    conn, direction_key, queue_model, shard_id,
                    worker_id, tb, args.max_attempts,
                )
                queue_db.log_event(
                    conn,
                    worker_id,
                    "fail",
                    direction_key=direction_key,
                    model=queue_model,
                    shard_id=shard_id,
                    detail=f"status={finalize_result}",
                )
            finally:
                context.bounds.clear()
    finally:
        queue_db.log_event(
            conn, worker_id, "exit", detail=f"completed={completed_shards} failed={failed_shards}"
        )
        conn.close()

    logger.info(
        "Worker finished: id=%s completed=%d failed=%d",
        worker_id, completed_shards, failed_shards,
    )
    return 0 if failed_shards == 0 else 1
