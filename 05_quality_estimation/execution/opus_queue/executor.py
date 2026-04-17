from __future__ import annotations

import argparse

from dataset.mediator import DatasetAdapter
from execution.opus_queue.worker import run_loop


class OpusQueueExecutor:
    """ExecutionStrategy wrapper that drives the OPUS SQLite queue worker.

    Unlike the FLORES executor this strategy ignores the passed-in
    score_entry and dataset because shard-aware loading has to happen
    through a sharding OPUS adapter that is constructed inside
    ``worker.run_loop``. The scorer itself is resolved from
    ``execution.opus_queue.scoring.scorer_factory`` using ``args.model``.
    """

    name = "opus_queue"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--db",
            default=None,
            help="Path to the shared SQLite queue database (required for opus_queue).",
        )
        parser.add_argument(
            "--opus-root",
            default=None,
            help="OPUS root dir; defaults to the OPUS adapter default_root.",
        )
        parser.add_argument(
            "--walltime-seconds",
            type=int,
            default=None,
            help="Remaining walltime (seconds). Worker exits cleanly before running out.",
        )
        parser.add_argument(
            "--max-attempts",
            type=int,
            default=3,
            help="Attempts before a shard is marked 'failed' (default: 3).",
        )
        parser.add_argument(
            "--claim-retries",
            type=int,
            default=5,
            help="Retries on SQLITE_BUSY during claim (default: 5).",
        )
        parser.add_argument(
            "--shard-size-override",
            action="append",
            default=[],
            help="Repeatable: --shard-size-override model:int (propagates to worker heuristics).",
        )
        parser.add_argument(
            "--backend",
            default=None,
            help="Optional backend override (comet|metricx|llm|remedy|bicleaner).",
        )

    def run(
        self,
        args: argparse.Namespace,
        dataset: DatasetAdapter,
        model_tag: str,
        score_entry,  # noqa: ARG002 - intentionally unused; see class docstring
    ) -> None:
        self._validate(args, dataset)
        raise SystemExit(run_loop(args))

    @staticmethod
    def _validate(args: argparse.Namespace, dataset: DatasetAdapter) -> None:
        if dataset.id != "opus":
            raise SystemExit(
                f"Execution 'opus_queue' requires --dataset opus, got '{dataset.id}'."
            )
        if not getattr(args, "db", None):
            raise SystemExit("--db is required for execution='opus_queue'.")
        if not getattr(args, "output_base", None):
            raise SystemExit("--output-base is required for execution='opus_queue'.")
        walltime = getattr(args, "walltime_seconds", None)
        if walltime is not None and walltime <= 0:
            raise SystemExit("--walltime-seconds must be > 0 when provided.")
        if getattr(args, "max_attempts", 1) <= 0:
            raise SystemExit("--max-attempts must be > 0.")
        if getattr(args, "claim_retries", 1) <= 0:
            raise SystemExit("--claim-retries must be > 0.")
