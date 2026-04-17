from __future__ import annotations

from pathlib import Path

from execution.opus_queue import db as queue_db
from execution.opus_queue.tools.merge.collect import collect_complete_directions
from execution.opus_queue.tools.merge.convert import merge_direction
from utils.logger import logger

__all__ = ["run"]


def run(args) -> None:
    db_path = Path(args.db).expanduser().resolve()
    output_base = Path(args.output_base).expanduser().resolve()
    merged_base = Path(args.merged_base).expanduser().resolve()
    merged_base.mkdir(parents=True, exist_ok=True)

    conn = queue_db.connect(db_path)
    queue_db.initialize(conn)
    try:
        directions = collect_complete_directions(conn, args.model)
        logger.info(
            "Found %d complete direction(s) to merge (model filter=%s).",
            len(directions),
            args.model,
        )
        merged = 0
        total_rows = 0
        total_shards = 0
        for direction_key, model in directions:
            ok, shards, rows = merge_direction(
                output_base,
                merged_base,
                db_path,
                direction_key,
                model,
                force=args.force,
                delete_shards=args.delete_shards,
            )
            if ok:
                merged += 1
                total_rows += rows
                total_shards += shards
        logger.info(
            "Merge complete: directions=%d shards=%d rows=%d",
            merged,
            total_shards,
            total_rows,
        )
    finally:
        conn.close()
