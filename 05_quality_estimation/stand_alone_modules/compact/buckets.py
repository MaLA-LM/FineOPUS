from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from utils.io import ROW_TYPE_SUMMARY

StageCommitKey = tuple[str, str, str, int]


def discover_models(dataset_root: Path) -> list[str]:
    """Return sorted list of model names found under dataset_root/model=*."""
    if not dataset_root.exists():
        return []
    return sorted(
        d.name.removeprefix("model=")
        for d in dataset_root.iterdir()
        if d.is_dir() and d.name.startswith("model=")
    )


def iter_stage_files_for_model(dataset_root: Path, model: str) -> list[Path]:
    """Find all part-*.jsonl inside shard=* dirs for a given model."""
    model_dir = dataset_root / f"model={model}"
    if not model_dir.exists():
        return []
    return sorted(
        path
        for path in model_dir.rglob("part-*.jsonl")
        if path.is_file() and any(part.startswith("shard=") for part in path.parts)
    )


def extract_stage_commit_key(record: Mapping[str, object]) -> StageCommitKey:
    return (
        record["direction_key"],
        record["model_name"],
        record["split"],
        int(record["shard_id"]),
    )


def load_stage_checkpoint(checkpoint_path: Path) -> set[StageCommitKey]:
    if not checkpoint_path.exists():
        return set()

    committed: set[StageCommitKey] = set()
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("row_type") == ROW_TYPE_SUMMARY:
                committed.add(extract_stage_commit_key(record))
    return committed
