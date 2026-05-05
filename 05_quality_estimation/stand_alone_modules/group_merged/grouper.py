from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import logger

_PART_RE = re.compile(r"^(.+)\.part-\d{4}\.parquet$")
_META_RE = re.compile(r"^(.+)\.meta\.json$")
_LEGACY_PARQUET_RE = re.compile(r"^(.+)\.parquet$")


@dataclass(frozen=True)
class MovePlan:
    src: Path
    dst: Path


@dataclass
class GroupSummary:
    models_seen: int = 0
    directions_seen: int = 0
    files_moved: int = 0
    files_planned: int = 0
    conflicts: list[Path] = field(default_factory=list)


def _direction_key_for_flat_file(path: Path) -> str | None:
    for pattern in (_PART_RE, _META_RE, _LEGACY_PARQUET_RE):
        match = pattern.match(path.name)
        if match:
            return match.group(1)
    return None


def _model_dirs(merged_base: Path, models: list[str] | None) -> list[Path]:
    if models:
        return [merged_base / model for model in models]
    return sorted(path for path in merged_base.iterdir() if path.is_dir())


def build_move_plan(model_dir: Path) -> dict[str, list[MovePlan]]:
    grouped: dict[str, list[MovePlan]] = {}
    for src in sorted(model_dir.iterdir()):
        if not src.is_file():
            continue
        direction_key = _direction_key_for_flat_file(src)
        if direction_key is None:
            continue
        dst = model_dir / direction_key / src.name
        grouped.setdefault(direction_key, []).append(MovePlan(src=src, dst=dst))
    return grouped


def group_merged_outputs(
    merged_base: Path,
    *,
    models: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> GroupSummary:
    summary = GroupSummary()
    model_plans: list[tuple[Path, dict[str, list[MovePlan]]]] = []
    for model_dir in _model_dirs(merged_base, models):
        if not model_dir.is_dir():
            logger.warning("Skipping missing model directory: %s", model_dir)
            continue

        summary.models_seen += 1
        grouped = build_move_plan(model_dir)
        summary.directions_seen += len(grouped)
        model_plans.append((model_dir, grouped))
        for direction_key, moves in grouped.items():
            for move in moves:
                if move.dst.exists() and not force:
                    summary.conflicts.append(move.dst)

    if summary.conflicts:
        return summary

    for model_dir, grouped in model_plans:
        for direction_key, moves in grouped.items():
            direction_dir = model_dir / direction_key
            if dry_run:
                for move in moves:
                    logger.info("Would move %s -> %s", move.src, move.dst)
                    summary.files_planned += 1
                continue

            direction_dir.mkdir(exist_ok=True)
            for move in moves:
                if move.dst.exists() and force:
                    move.dst.unlink()
                move.src.replace(move.dst)
                summary.files_moved += 1
                logger.info("Moved %s -> %s", move.src, move.dst)

    return summary
