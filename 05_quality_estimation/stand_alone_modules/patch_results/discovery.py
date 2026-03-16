from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shardwork:

    shard_path: Path  # full path to shard directory: /scratch/project_462001050/QE_flores200_scores/dataset=flores200/model=m-prometheus-7b/split=dev/shard=000
    split: str
    checkpoint_path: Path
    part_files: list[Path]  # list of paths to part files in the shard directory


def _iter_prefixed_dirs(parent: Path, prefix: str) -> list[Path]:
    if not parent.exists() or not parent.is_dir():
        return []
    return sorted(
        (
            child
            for child in parent.iterdir()
            if child.is_dir() and child.name.startswith(prefix)
        ),
        key=lambda path: path.name,
    )


def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix) :]


def get_part_files(shard_dir: Path) -> list[Path]:
    part_files = [
        file
        for file in shard_dir.iterdir()
        if file.is_file() and file.name.startswith("part-")
    ]
    return part_files


def discover_shards_needed(results_path: Path) -> list[Shardwork]:
    shards: list[Shardwork] = []
    for split_dir in _iter_prefixed_dirs(results_path, "split="):
        split_name = _strip_prefix(split_dir.name, "split=")
        for shard_dir in _iter_prefixed_dirs(split_dir, "shard="):
            checkpoint_path = shard_dir / "checkpoint.jsonl"
            files = get_part_files(shard_dir)
            shards.append(
                Shardwork(
                    shard_path=shard_dir,
                    split=split_name,
                    checkpoint_path=checkpoint_path,
                    part_files=files,
                )
            )
    return shards
