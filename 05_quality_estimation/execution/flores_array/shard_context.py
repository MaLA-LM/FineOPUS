from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ShardContext:
    shard_id: int
    num_shards: int


def _parse_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer in {name}: {raw!r}") from exc


def resolve_shard_context(args) -> ShardContext:
    shard_id = getattr(args, "shard_id", None)
    if shard_id is None:
        shard_id = _parse_int_env("SLURM_ARRAY_TASK_ID")

    num_shards = getattr(args, "num_shards", None)
    if num_shards is None:
        num_shards = _parse_int_env("SLURM_ARRAY_TASK_COUNT")

    if shard_id is None or num_shards is None:
        raise SystemExit(
            "Missing shard context: provide --shard-id and --num-shards, "
            "or run inside a Slurm array with SLURM_ARRAY_TASK_ID/COUNT."
        )

    if num_shards <= 0:
        raise SystemExit("num_shards must be > 0.")
    if shard_id < 0:
        raise SystemExit("shard_id must be >= 0.")
    if shard_id >= num_shards:
        raise SystemExit(
            f"shard_id out of range: {shard_id} not in [0, {num_shards - 1}]."
        )
    return ShardContext(shard_id=shard_id, num_shards=num_shards)
