from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from stand_alone_modules.patch_remedy.patch import run


@dataclass(frozen=True)
class PatchConfig:
    src_root: Path
    dst_root: Path
    model: str


__all__ = ["PatchConfig", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch Remedy seen/unseen flags.")
    parser.add_argument(
        "--src-root",
        default="/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=normalized_scores",
    )
    parser.add_argument(
        "--dst-root",
        default="/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=normalized_scores_patched",
    )
    parser.add_argument(
        "--model",
        default="shaomutan_remedy-9b-22",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        PatchConfig(
            src_root=Path(args.src_root),
            dst_root=Path(args.dst_root),
            model=args.model,
        )
    )
