from __future__ import annotations

import os

from execution.opus_queue import db as queue_db
from execution.opus_queue.worker.shard_io import (
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
    out_path,
    direction_key: str,
    model: str,
    shard_id: int,
    worker_id: str,
    elapsed: float,
) -> bool:
    tmp_path = write_temp_payload(out_path, frame_to_jsonl_bytes(frame), worker_id)
    detail_rows = count_detail_rows(frame)

    try:
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
