from __future__ import annotations

import os
from pathlib import Path

from execution.opus_queue import db as queue_db
from execution.opus_queue.trace.writer import TraceWriter
from execution.opus_queue.worker.shard_io import (
    DirectionPartWriter,
    cleanup_temp_file,
    count_detail_rows,
    frame_to_jsonl_bytes,
    write_temp_payload,
)
from utils.logger import logger

__all__ = ["commit_shard"]


def commit_shard(
    conn,
    frame,
    direction_key: str,
    model: str,
    shard_id: int,
    worker_id: str,
    elapsed: float,
    *,
    out_path: Path | None = None,
    writer: DirectionPartWriter | None = None,
    trace_writer: TraceWriter | None = None,
    trace_event: dict | None = None,
    return_event: bool = False,
) -> bool | dict:
    detail_rows = count_detail_rows(frame)
    if writer is not None:
        try:
            out_path = writer.append_shard(frame, direction_key, shard_id)
        except Exception:
            writer.close_direction(direction_key)
            raise
        if trace_writer is not None:
            if trace_event is None:
                raise ValueError("trace_event is required when trace_writer is provided")
            done_event = {**trace_event, "out_path": str(out_path)}
            trace_writer.append_completion(done_event)
            logger.info(
                "Committed shard dir=%s shard=%d rows=%d elapsed=%.1fs path=%s",
                direction_key,
                shard_id,
                detail_rows,
                elapsed,
                out_path,
            )
            return done_event if return_event else True
        result = queue_db.mark_done(
            conn, direction_key, model, shard_id, worker_id, out_path
        )
        if result == "done":
            queue_db.log_event(
                conn,
                worker_id,
                "done",
                direction_key=direction_key,
                model=model,
                shard_id=shard_id,
                detail=f"rows={detail_rows} elapsed={elapsed:.1f}s",
            )
            logger.info(
                "Committed shard dir=%s shard=%d rows=%d elapsed=%.1fs path=%s",
                direction_key,
                shard_id,
                detail_rows,
                elapsed,
                out_path,
            )
            return True
        queue_db.log_event(
            conn,
            worker_id,
            "stale_noop",
            direction_key=direction_key,
            model=model,
            shard_id=shard_id,
            detail="mark_done rejected stale worker ownership",
        )
        logger.warning(
            "Retained stale shard output dir=%s shard=%d at %s; merge will filter it.",
            direction_key,
            shard_id,
            out_path,
        )
        return False

    if out_path is None:
        raise ValueError("legacy shard commits require out_path")

    tmp_path = write_temp_payload(out_path, frame_to_jsonl_bytes(frame), worker_id)

    try:
        if trace_writer is not None:
            if trace_event is None:
                raise ValueError("trace_event is required when trace_writer is provided")
            os.replace(tmp_path, out_path)
            done_event = {**trace_event, "out_path": str(out_path)}
            trace_writer.append_completion(done_event)
            logger.info(
                "Committed shard dir=%s shard=%d rows=%d elapsed=%.1fs path=%s",
                direction_key,
                shard_id,
                detail_rows,
                elapsed,
                out_path,
            )
            return done_event if return_event else True

        result = queue_db.mark_done(conn, direction_key, model, shard_id, worker_id, out_path)
        if result == "done":
            os.replace(tmp_path, out_path)
            queue_db.log_event(
                conn,
                worker_id,
                "done",
                direction_key=direction_key,
                model=model,
                shard_id=shard_id,
                detail=f"rows={detail_rows} elapsed={elapsed:.1f}s",
            )
            logger.info(
                "Committed shard dir=%s shard=%d rows=%d elapsed=%.1fs path=%s",
                direction_key,
                shard_id,
                detail_rows,
                elapsed,
                out_path,
            )
            return True

        cleanup_temp_file(tmp_path)
        queue_db.log_event(
            conn,
            worker_id,
            "stale_noop",
            direction_key=direction_key,
            model=model,
            shard_id=shard_id,
            detail="mark_done rejected stale worker ownership",
        )
        logger.warning(
            "Discarded stale shard output dir=%s shard=%d after ownership was lost.",
            direction_key,
            shard_id,
        )
        return False
    except Exception:
        cleanup_temp_file(tmp_path)
        raise
