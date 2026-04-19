from __future__ import annotations

import math
from dataclasses import dataclass

# Seed shard sizes (sentences per shard) targeting ~30 min of wallclock per
# shard based on the rates observed in the OPUS lookup CSV. These are starting
# points; operators override per-model after calibration via
# --shard-size-override model:int on build_queue / executor CLIs.
SEED_SHARD_SIZES: dict[str, int] = {
    "metricx24": 200_000,
    "wmt23-cometkiwi-da-xl": 50_000,
    "xcomet-xl": 100_000,
    "wmt22-cometkiwi-da": 50_000,
    "remedy": 16_500,
    "shaomutan_remedy-9b-22": 60_000,
    "qwen3-4b-instruct-2507": 10_000,
    "qwen3-4b-instruct-2507-detailed": 10_000,
    "qwen3-4b-instruct-2507-simple": 10_000,
    "qwen3-4b-instruct-2507-batch": 10_000,
    "m-prometheus-7b": 7_500,
    "m-prometheus-7b-detailed": 9_000,
    "bicleaner-ai": 200_000,
    "auto": 125_000,
}

DEFAULT_SHARD_SIZE = 20_000

# Default expected wallclock per shard (used by reaper cutoff and worker
# walltime checks). 30 minutes matches the sizing target; per-model tuning
# can override via EXPECTED_SHARD_SECONDS.
DEFAULT_EXPECTED_SHARD_SECONDS = 30 * 60

EXPECTED_SHARD_SECONDS: dict[str, int] = {
    # Fill in per-model deltas here after calibration. Unset models fall
    # back to DEFAULT_EXPECTED_SHARD_SECONDS.
    "metricx24": 600,
    "wmt23-cometkiwi-da-xl": 1_500,
    "xcomet-xl": 1_500,
    "shaomutan_remedy-9b-22": 1_000,
    "qwen3-4b-instruct-2507": 1_200,
    "m-prometheus-7b": 1_200,
    "m-prometheus-7b-detailed": 9_000,
    "bicleaner-ai": 300,
}


@dataclass(frozen=True)
class ShardRange:
    shard_id: int
    start_idx: int
    end_idx: int


def get_shard_size(model: str, overrides: dict[str, int] | None = None) -> int:
    if overrides and model in overrides:
        return int(overrides[model])
    return SEED_SHARD_SIZES.get(model, DEFAULT_SHARD_SIZE)


def expected_shard_seconds(model: str) -> int:
    return EXPECTED_SHARD_SECONDS.get(model, DEFAULT_EXPECTED_SHARD_SECONDS)


def plan(
    model: str,
    n_sentences: int,
    overrides: dict[str, int] | None = None,
) -> tuple[int, list[ShardRange]]:
    """Partition [0, n_sentences) into contiguous shards.

    Returns (shard_size, shards). Edge cases:
    - n_sentences == 0 -> no shards.
    - n_sentences < shard_size -> one shard covering everything.
    - exact multiple of shard_size -> no trailing ragged shard.
    """
    if n_sentences < 0:
        raise ValueError(f"n_sentences must be >= 0, got {n_sentences}")
    shard_size = get_shard_size(model, overrides)
    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0 for model {model}")
    if n_sentences == 0:
        return shard_size, []
    n_shards = math.ceil(n_sentences / shard_size)
    shards: list[ShardRange] = []
    for idx in range(n_shards):
        start = idx * shard_size
        end = min(start + shard_size, n_sentences)
        shards.append(ShardRange(shard_id=idx, start_idx=start, end_idx=end))
    return shard_size, shards


def parse_shard_size_overrides(entries: list[str] | None) -> dict[str, int]:
    """Parse repeated --shard-size-override model:int entries into a dict."""
    if not entries:
        return {}
    parsed: dict[str, int] = {}
    for raw in entries:
        if ":" not in raw:
            raise SystemExit(f"--shard-size-override expects 'model:int', got {raw!r}")
        name, value = raw.split(":", 1)
        name = name.strip()
        try:
            value_int = int(value.strip())
        except ValueError as exc:
            raise SystemExit(
                f"--shard-size-override value must be int, got {value!r}"
            ) from exc
        if value_int <= 0:
            raise SystemExit(
                f"--shard-size-override value must be > 0, got {value_int}"
            )
        parsed[name] = value_int
    return parsed
