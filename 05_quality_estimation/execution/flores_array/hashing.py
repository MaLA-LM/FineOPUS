from __future__ import annotations

from utils.hashing import stable_hash_int


def compute_shard_id(direction_key_value: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0.")
    return stable_hash_int(direction_key_value) % num_shards
