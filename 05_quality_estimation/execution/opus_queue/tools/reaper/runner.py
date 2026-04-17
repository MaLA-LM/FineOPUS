from __future__ import annotations

from execution.opus_queue import db as queue_db
from execution.opus_queue.planning import expected_shard_seconds
from utils.logger import logger

__all__ = ["per_model_cutoffs", "run_once"]


def per_model_cutoffs(
    conn,
    multiplier: float,
    default_seconds: int,
    *,
    reset_failed: bool,
) -> dict[str, int]:
    statuses = ["running"]
    if reset_failed:
        statuses.append("failed")
    placeholders = ", ".join("?" for _ in statuses)
    models = [
        row["model"]
        for row in conn.execute(
            f"SELECT DISTINCT model FROM jobs WHERE status IN ({placeholders})",
            tuple(statuses),
        ).fetchall()
    ]
    cutoffs: dict[str, int] = {}
    for model in models:
        expected = expected_shard_seconds(model) or default_seconds
        cutoffs[model] = int(expected * multiplier)
    return cutoffs


def run_once(args) -> int:
    conn = queue_db.connect(args.db)
    queue_db.initialize(conn)
    try:
        cutoffs = per_model_cutoffs(
            conn,
            args.timeout_multiplier,
            args.default_timeout_seconds,
            reset_failed=args.reset_failed,
        )
        if not cutoffs:
            logger.info("Reaper: no running/failed rows eligible for this sweep; nothing to do.")
            return 0
        logger.info(
            "Reaper sweep cutoffs (seconds): %s reset_failed=%s",
            cutoffs,
            args.reset_failed,
        )
        reclaimed = queue_db.reset_stale_rows(
            conn, cutoffs, reset_failed=args.reset_failed
        )
        for row in reclaimed:
            previous_status = row.get("previous_status")
            if previous_status == "failed":
                event = "revive_failed"
                detail = (
                    f"finished_at={row.get('finished_at')} "
                    f"last_error={row.get('last_error')}"
                )
            else:
                event = "reap"
                detail = (
                    f"worker_id={row.get('worker_id')} "
                    f"started_at={row.get('started_at')}"
                )
            queue_db.log_event(
                conn,
                None,
                event,
                direction_key=row.get("direction_key"),
                model=row.get("model"),
                shard_id=row.get("shard_id"),
                detail=detail,
            )
        logger.info("Reaper sweep reclaimed %d rows.", len(reclaimed))
        return len(reclaimed)
    finally:
        conn.close()
