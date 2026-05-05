from __future__ import annotations

import argparse

from execution.opus_queue.worker.loop import run_loop

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OPUS queue worker: manifest assignment -> score -> trace commit."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "manifest", "db"),
        default="auto",
        help="Worker coordination mode. auto selects manifest when --manifest-root is provided.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Legacy SQLite jobs DB path (only used with --mode db).",
    )
    parser.add_argument(
        "--manifest-root",
        default=None,
        help="Root containing <build_tag>/manifest.jsonl for manifest mode.",
    )
    parser.add_argument(
        "--trace-root",
        default="/scratch/project_462001050/opus_qe/shard_trace",
        help="Root for per-worker trace files in manifest mode.",
    )
    parser.add_argument(
        "--build-tag",
        default=None,
        help="Manifest build tag to read.",
    )
    parser.add_argument("--model", required=True, help="Model key (e.g. metricx24).")
    parser.add_argument(
        "--scorer-model",
        default=None,
        help=(
            "Optional runtime model identifier for the scorer when it differs "
            "from the queue/DB model key."
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Override auto-detected backend (comet|metricx|llm|remedy|bicleaner).",
    )
    parser.add_argument(
        "--output-base",
        required=True,
        help="Shared-storage dir for legacy shard JSONLs or worker-owned part files.",
    )
    parser.add_argument(
        "--part-writer",
        action="store_true",
        help="Append multiple shards into worker-owned part files instead of shard_*.jsonl.",
    )
    parser.add_argument(
        "--part-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Rotate part files before the next shard would push them past this size.",
    )
    parser.add_argument(
        "--part-max-shards",
        type=int,
        default=32,
        help="Rotate part files before appending the next shard once this many shards are present.",
    )
    parser.add_argument("--opus-root", default=None, help="OPUS root; defaults to adapter default.")
    parser.add_argument("--walltime-seconds", type=int, default=None,
                        help="Remaining walltime (seconds). Worker exits cleanly if time runs low.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Legacy DB mode only: attempts before a shard is marked failed.",
    )
    parser.add_argument(
        "--claim-retries",
        type=int,
        default=30,
        help="Legacy DB-mode retries on SQLITE_BUSY.",
    )
    parser.add_argument(
        "--shard-size-override",
        action="append",
        default=[],
        help="Repeatable: model:int (propagates to expected-shard-seconds heuristics).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Debug cap; applied on top of the shard slice.")
    parser.add_argument("--prompt-mode", default="detailed")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--response-format", default=None)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_loop(args))
