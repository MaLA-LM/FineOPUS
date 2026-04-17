from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from stand_alone_modules.normalized_scores.normalize import run

DEFAULT_MODELS = [
    "qwen3-4b-instruct-2507-detailed",
    "qwen3-4b-instruct-2507-simple",
]


@dataclass(frozen=True)
class NormalizeConfig:
    src_root: Path
    dst_root: Path
    models: list[str]


__all__ = ["NormalizeConfig", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize FLORES score buckets.")
    parser.add_argument(
        "--src-root",
        default="/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=raw_scores_new",
    )
    parser.add_argument(
        "--dst-root",
        default="/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=normalized_scores",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        NormalizeConfig(
            src_root=Path(args.src_root),
            dst_root=Path(args.dst_root),
            models=list(args.models),
        )
    )
