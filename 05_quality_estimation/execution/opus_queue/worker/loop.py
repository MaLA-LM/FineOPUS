from __future__ import annotations

import argparse
import os
import random
import socket
import sqlite3
import time
import traceback
from pathlib import Path

from execution.flores_array.manifest import ManifestEntry
from execution.opus_queue.manifest import ManifestRow, load_or_create_assignment_copy
from execution.opus_queue import db as queue_db
from execution.opus_queue.planning import (
    expected_shard_seconds,
    parse_shard_size_overrides,
)
from execution.opus_queue.scoring import build_scorer, resolve_backend
from execution.opus_queue.trace import TraceState, TraceWriter, replay_events
from execution.opus_queue.worker.commit import commit_shard
from execution.opus_queue.worker.shard_io import DirectionPartWriter, shard_output_path
from execution.opus_queue.worker.shard_loader import build_sharding_opus_adapter
from execution.opus_queue.worker.walltime import remaining_seconds
from utils.logger import logger

__all__ = ["run_loop"]


_CLAIM_FAILURE_BACKOFF_S = (5.0, 20.0)


def _make_worker_id() -> str:
    job = os.environ.get("SLURM_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    host = socket.gethostname()
    pid = os.getpid()
    return f"{job}.{task}.{host}.{pid}"


def _env_int(*names: str, default: int = 0) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _make_worker_slot_id(model: str) -> str:
    array_task_id = _env_int("OPUS_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_ID", default=0)
    local_id = _env_int("OPUS_LOCAL_ID", "SLURM_LOCALID", default=0)
    return f"{model}-a{array_task_id:05d}-l{local_id}"


def _make_worker_run_id() -> str:
    job = os.environ.get("SLURM_JOB_ID", "local")
    array_task_id = str(
        _env_int("OPUS_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_ID", default=0)
    )
    local_id = str(_env_int("OPUS_LOCAL_ID", "SLURM_LOCALID", default=0))
    host = socket.gethostname()
    pid = os.getpid()
    return f"{job}.{array_task_id}.{local_id}.{host}.{pid}"


def _summarize_exception(exc: BaseException) -> str:
    message = " ".join(str(exc).split()).strip()
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"


def _format_failure_detail(
    exc: BaseException,
    tb: str,
    *,
    direction_key: str,
    shard_id: int,
    start_idx: int,
    end_idx: int,
    queue_model: str,
    scorer_model: str,
    backend: str,
    attempt: int | None = None,
) -> str:
    summary = _summarize_exception(exc)
    shard_context = (
        f"context: queue_model={queue_model} scorer_model={scorer_model} "
        f"backend={backend} direction={direction_key} shard={shard_id} "
        f"range=[{start_idx},{end_idx})"
    )
    if attempt is not None:
        shard_context = f"{shard_context} attempt={attempt}"
    return f"{summary}\n{shard_context}\ntraceback:\n{tb.rstrip()}"


def _score_shard(
    run_entry, context, entry, start_idx, end_idx, direction_key, shard_id
):
    context.bounds.start = start_idx
    context.bounds.end = end_idx
    context.bounds.active = True

    shard_start = time.time()
    frame = run_entry(entry)
    elapsed = time.time() - shard_start

    frame["direction_key"] = direction_key
    frame["shard_id"] = shard_id
    return frame, elapsed


def _run_db_loop(args: argparse.Namespace) -> int:
    start_ts = time.time()
    worker_id = _make_worker_id()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    node_host = socket.gethostname()
    gpu_count = max(1, int(args.gpus))
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
        detail=(
            f"queue_model={queue_model} scorer_model={scorer_model} "
            f"backend={backend} part_writer={args.part_writer}"
        ),
    )
    reset_count = queue_db.reset_own_stale(conn, worker_id)
    if reset_count:
        logger.warning("Reset %d stale rows owned by %s", reset_count, worker_id)

    completed_shards = 0
    failed_shards = 0
    writer = (
        DirectionPartWriter(
            output_base=output_base,
            model=queue_model,
            worker_id=worker_id,
            max_bytes=args.part_max_bytes,
            max_shards_per_part=args.part_max_shards,
        )
        if args.part_writer
        else None
    )

    preferred_direction_key: str | None = None
    try:
        while True:
            time_left = remaining_seconds(start_ts, args.walltime_seconds)
            if time_left < 1.5 * exp_seconds:
                logger.info(
                    "Walltime budget low (%.0fs left < 1.5 * %ds); exiting cleanly.",
                    time_left,
                    exp_seconds,
                )
                queue_db.log_event(conn, worker_id, "exit", detail="walltime_budget")
                break

            try:
                job = queue_db.claim_next(
                    conn,
                    queue_model,
                    worker_id,
                    retries=args.claim_retries,
                    slurm_job_id=slurm_job_id,
                    slurm_array_task_id=slurm_array_task_id,
                    node_host=node_host,
                    gpu_count=gpu_count,
                    preferred_direction_key=preferred_direction_key,
                )
            except sqlite3.OperationalError as exc:
                backoff = random.uniform(*_CLAIM_FAILURE_BACKOFF_S)
                logger.warning(
                    "claim_next failed (db lock?): %s; sleeping %.1fs and retrying.",
                    _summarize_exception(exc),
                    backoff,
                )
                time.sleep(backoff)
                continue
            if job is None:
                counts = queue_db.count_by_status(conn, queue_model)
                if counts:
                    logger.info(
                        "Queue drained for model=%s; status_counts=%s; exiting.",
                        queue_model,
                        counts,
                    )
                else:
                    pending_models = conn.execute("""
                        SELECT model, COUNT(*) AS n
                          FROM jobs
                         WHERE status = 'pending'
                         GROUP BY model
                         ORDER BY n DESC, model ASC
                         LIMIT 10
                        """).fetchall()
                    pending_summary = (
                        ", ".join(
                            f"{row['model']}:{row['n']}" for row in pending_models
                        )
                        or "<none>"
                    )
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
                direction_key,
                shard_id,
                start_idx,
                end_idx,
                attempt,
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
                    run_entry,
                    context,
                    entry,
                    start_idx,
                    end_idx,
                    direction_key,
                    shard_id,
                )
                frame["worker_id"] = worker_id
                out_path = None
                if writer is None:
                    out_path = shard_output_path(
                        output_base, args.model, direction_key, shard_id
                    )
                if commit_shard(
                    conn,
                    frame,
                    direction_key,
                    queue_model,
                    shard_id,
                    worker_id,
                    elapsed,
                    out_path=out_path,
                    writer=writer,
                ):
                    completed_shards += 1
                    preferred_direction_key = direction_key
            except Exception as exc:
                preferred_direction_key = None
                failed_shards += 1
                tb = traceback.format_exc()
                error_detail = _format_failure_detail(
                    exc,
                    tb,
                    direction_key=direction_key,
                    shard_id=shard_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    attempt=attempt,
                    queue_model=queue_model,
                    scorer_model=scorer_model,
                    backend=backend,
                )
                error_summary = _summarize_exception(exc)
                logger.error(
                    "Shard failed dir=%s shard=%d:\n%s",
                    direction_key,
                    shard_id,
                    error_detail,
                )
                try:
                    finalize_result = queue_db.mark_failed(
                        conn,
                        direction_key,
                        queue_model,
                        shard_id,
                        worker_id,
                        error_detail,
                        args.max_attempts,
                    )
                except sqlite3.OperationalError as finalize_exc:
                    logger.error(
                        "mark_failed could not commit (db lock?) dir=%s shard=%d: %s; "
                        "row left 'running' — stale-row reset will requeue it.",
                        direction_key,
                        shard_id,
                        _summarize_exception(finalize_exc),
                    )
                    queue_db.log_event(
                        conn,
                        worker_id,
                        "fail_unrecorded",
                        direction_key=direction_key,
                        model=queue_model,
                        shard_id=shard_id,
                        detail=(f"mark_failed db_lock; original_error={error_summary}"),
                    )
                else:
                    queue_db.log_event(
                        conn,
                        worker_id,
                        "fail",
                        direction_key=direction_key,
                        model=queue_model,
                        shard_id=shard_id,
                        detail=f"status={finalize_result} error={error_summary}",
                    )
            finally:
                context.bounds.clear()
    finally:
        queue_db.log_event(
            conn,
            worker_id,
            "exit",
            detail=f"completed={completed_shards} failed={failed_shards}",
        )
        if writer is not None:
            writer.close_all()
        conn.close()

    logger.info(
        "Worker finished: id=%s completed=%d failed=%d",
        worker_id,
        completed_shards,
        failed_shards,
    )
    return 0 if failed_shards == 0 else 1


def _select_next_assignment(
    assignments: list[ManifestRow],
    state: TraceState,
) -> ManifestRow | None:
    for assignment in assignments:
        if state.can_process(assignment):
            return assignment
    return None


def _run_manifest_loop(args: argparse.Namespace) -> int:
    if not getattr(args, "manifest_root", None):
        raise SystemExit("--manifest-root is required for manifest mode.")
    if not getattr(args, "build_tag", None):
        raise SystemExit("--build-tag is required for manifest mode.")
    if not getattr(args, "trace_root", None):
        raise SystemExit("--trace-root is required for manifest mode.")

    start_ts = time.time()
    queue_model = args.model
    worker_slot_id = _make_worker_slot_id(queue_model)
    worker_run_id = _make_worker_run_id()
    gpu_count = max(1, int(args.gpus))
    scorer_model = args.scorer_model or queue_model
    backend = args.backend or resolve_backend(scorer_model)
    parse_shard_size_overrides(args.shard_size_override)
    exp_seconds = expected_shard_seconds(queue_model)
    output_base = Path(args.output_base).expanduser().resolve()
    opus_root = args.opus_root

    logger.info(
        "Manifest worker starting: slot=%s run=%s queue_model=%s scorer_model=%s backend=%s build_tag=%s walltime=%s expected_shard=%ds",
        worker_slot_id,
        worker_run_id,
        queue_model,
        scorer_model,
        backend,
        args.build_tag,
        args.walltime_seconds,
        exp_seconds,
    )

    trace_writer = TraceWriter(args.trace_root, args.build_tag, worker_slot_id)
    assignments = load_or_create_assignment_copy(
        args.manifest_root,
        args.trace_root,
        args.build_tag,
        worker_slot_id,
    )
    state = replay_events(trace_writer.completed_path)
    trace_writer.write_state(state.snapshot(assignments))
    if not assignments:
        logger.warning(
            "No manifest assignments for worker_slot_id=%s build_tag=%s",
            worker_slot_id,
            args.build_tag,
        )
        return 0

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

    completed_shards = 0
    hard_failure = False
    writer = (
        DirectionPartWriter(
            output_base=output_base,
            model=queue_model,
            worker_id=worker_slot_id,
            max_bytes=args.part_max_bytes,
            max_shards_per_part=args.part_max_shards,
        )
        if args.part_writer
        else None
    )

    try:
        while True:
            time_left = remaining_seconds(start_ts, args.walltime_seconds)
            if time_left < 1.5 * exp_seconds:
                logger.info(
                    "Walltime budget low (%.0fs left < 1.5 * %ds); exiting cleanly.",
                    time_left,
                    exp_seconds,
                )
                break

            assignment = _select_next_assignment(assignments, state)
            if assignment is None:
                logger.info(
                    "Manifest assignments exhausted for slot=%s; state_counts=%s",
                    worker_slot_id,
                    state.counts(assignments),
                )
                break

            direction_key = assignment.direction_key
            shard_id = int(assignment.shard_id)
            start_idx = int(assignment.start_idx)
            end_idx = int(assignment.end_idx)
            logger.info(
                "Starting assigned shard dir=%s shard=%d range=[%d,%d) slot=%s",
                direction_key,
                shard_id,
                start_idx,
                end_idx,
                worker_slot_id,
            )

            try:
                entry = ManifestEntry(
                    src_lang=assignment.src_lang,
                    tgt_lang=assignment.tgt_lang,
                    split=dataset.split_values[0],
                )
                frame, elapsed = _score_shard(
                    run_entry,
                    context,
                    entry,
                    start_idx,
                    end_idx,
                    direction_key,
                    shard_id,
                )
                frame["worker_id"] = worker_slot_id
                frame["worker_run_id"] = worker_run_id
                out_path = None
                if writer is None:
                    out_path = shard_output_path(
                        output_base, queue_model, direction_key, shard_id
                    )
                finished_at = int(time.time())
                done_event = trace_writer.done_event(
                    assignment,
                    worker_run_id=worker_run_id,
                    finished_at=finished_at,
                    gpu_count=gpu_count,
                    gpu_seconds_delta=elapsed * gpu_count,
                    out_path=out_path,
                )
                committed_event = commit_shard(
                    None,
                    frame,
                    direction_key,
                    queue_model,
                    shard_id,
                    worker_slot_id,
                    elapsed,
                    out_path=out_path,
                    writer=writer,
                    trace_writer=trace_writer,
                    trace_event=done_event,
                    return_event=True,
                )
                if isinstance(committed_event, dict):
                    state.apply_event(committed_event)
                else:
                    state.apply_event(done_event)
                trace_writer.write_state(state.snapshot(assignments))
                completed_shards += 1
            except Exception as exc:
                tb = traceback.format_exc()
                error_detail = _format_failure_detail(
                    exc,
                    tb,
                    direction_key=direction_key,
                    shard_id=shard_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    queue_model=queue_model,
                    scorer_model=scorer_model,
                    backend=backend,
                )
                trace_writer.write_state(state.snapshot(assignments))
                logger.error(
                    "Manifest shard failed; stopping worker slot=%s dir=%s shard=%d:\n%s",
                    worker_slot_id,
                    direction_key,
                    shard_id,
                    error_detail,
                )
                hard_failure = True
                break
            finally:
                context.bounds.clear()
    finally:
        if writer is not None:
            writer.close_all()

    logger.info(
        "Manifest worker finished: slot=%s run=%s completed=%d hard_failure=%s state_counts=%s",
        worker_slot_id,
        worker_run_id,
        completed_shards,
        hard_failure,
        state.counts(assignments),
    )
    return 1 if hard_failure else 0


def run_loop(args: argparse.Namespace) -> int:
    mode = getattr(args, "mode", None)
    if mode is None or mode == "auto":
        mode = "manifest" if getattr(args, "manifest_root", None) else "db"
    if mode == "manifest":
        return _run_manifest_loop(args)
    if mode == "db":
        return _run_db_loop(args)
    raise SystemExit(f"Unsupported worker mode: {mode}")
