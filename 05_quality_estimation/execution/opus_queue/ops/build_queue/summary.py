from __future__ import annotations

from execution.opus_queue.planning import get_shard_size
from utils.logger import logger

__all__ = ["log_summary"]


def log_summary(
    *,
    summary: dict[str, dict[str, int]],
    total_directions: int,
    total_shards: int,
    total_sentences: int,
    inserted_jobs: int,
    skipped_missing_dirs: int,
    skipped_empty: int,
    reassigned: int,
    overrides: dict[str, int],
) -> None:
    logger.info(
        "Build summary: directions=%d shards=%d sentences=%d inserted_jobs=%d skipped_missing=%d skipped_empty=%d reassigned=%d",
        total_directions,
        total_shards,
        total_sentences,
        inserted_jobs,
        skipped_missing_dirs,
        skipped_empty,
        reassigned,
    )
    logger.info("Per-model breakdown (shard_size defaults in planner):")
    for model, bucket in sorted(summary.items()):
        logger.info(
            "  model=%s directions=%d shards=%d sentences=%d shard_size=%d",
            model,
            bucket["directions"],
            bucket["shards"],
            bucket["sentences"],
            get_shard_size(model, overrides),
        )
