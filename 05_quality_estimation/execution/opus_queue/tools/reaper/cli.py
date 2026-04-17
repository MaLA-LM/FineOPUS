from __future__ import annotations

import argparse
import time

from execution.opus_queue.planning import DEFAULT_EXPECTED_SHARD_SECONDS
from execution.opus_queue.tools.reaper.runner import run_once

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reclaim stuck 'running' OPUS queue rows back to 'pending'."
    )
    parser.add_argument("--db", required=True, help="Shared jobs SQLite file.")
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between sweeps (default: 300). Use 0 for a single pass.",
    )
    parser.add_argument(
        "--timeout-multiplier", type=float, default=2.0,
        help="cutoff = timeout_multiplier * expected_shard_seconds(model).",
    )
    parser.add_argument(
        "--default-timeout-seconds", type=int, default=DEFAULT_EXPECTED_SHARD_SECONDS,
        help="Fallback expected-shard-seconds when a model isn't registered.",
    )
    parser.add_argument(
        "--reset-failed",
        action="store_true",
        help=(
            "Also requeue terminal 'failed' rows after the same model-specific "
            "cooldown used for stale running rows. Off by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        run_once(args)
        return
    while True:
        run_once(args)
        time.sleep(args.interval)
