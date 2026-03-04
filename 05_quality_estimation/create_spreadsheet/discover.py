from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointFile:
    path: Path
    model_name: str
    split: str


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


def discover_checkpoint_files(results_path: Path) -> list[CheckpointFile]:
    checkpoints: list[CheckpointFile] = []

    for model_dir in _iter_prefixed_dirs(results_path, "model="):
        model_name = _strip_prefix(model_dir.name, "model=")
        for split_dir in _iter_prefixed_dirs(model_dir, "split="):
            split_name = _strip_prefix(split_dir.name, "split=")
            for shard_dir in _iter_prefixed_dirs(split_dir, "shard="):
                checkpoint_path = shard_dir / "checkpoint.jsonl"
                if checkpoint_path.exists() and checkpoint_path.is_file():
                    checkpoints.append(
                        CheckpointFile(
                            path=checkpoint_path,
                            model_name=model_name,
                            split=split_name,
                        )
                    )

    return checkpoints
