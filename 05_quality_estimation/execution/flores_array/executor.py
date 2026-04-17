from __future__ import annotations

import argparse
from collections.abc import Callable

from dataset.mediator import DatasetAdapter
from execution.flores_array.directions import collect_directions, validate_flores_args
from execution.flores_array.manifest import ManifestEntry
from utils.logger import logger

ScoreEntry = Callable[[ManifestEntry], object]

__all__ = ["FloresArrayExecutor"]


class FloresArrayExecutor:
    name = "flores_array"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--manifest",
            default=None,
            help="TSV manifest with columns: src_lang, tgt_lang, split[, shard_id].",
        )
        parser.add_argument(
            "--shard-id",
            type=int,
            default=None,
            help=(
                "Optional shard id for this worker. "
                "Defaults to SLURM_ARRAY_TASK_ID when running under Slurm."
            ),
        )
        parser.add_argument(
            "--num-shards",
            type=int,
            default=None,
            help=(
                "Optional total shard count. "
                "Defaults to SLURM_ARRAY_TASK_COUNT when running under Slurm."
            ),
        )
        parser.add_argument(
            "--max-directions-per-part",
            type=int,
            default=25,
            help="Close and commit a part after this many directions.",
        )
        parser.add_argument(
            "--target-part-bytes",
            type=int,
            default=67_108_864,
            help="Best-effort target bytes before rotating a part file.",
        )

    def run(
        self,
        args: argparse.Namespace,
        dataset: DatasetAdapter,
        model_tag: str,
        score_entry: ScoreEntry,
    ) -> None:
        validate_flores_args(args, dataset)
        directions = collect_directions(args, dataset)
        if not directions:
            logger.info("No directions found.")
            return
        from execution.flores_array.runner import run_scoring

        run_scoring(args, dataset, directions, model_tag, score_entry)
