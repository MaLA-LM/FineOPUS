from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from utils.logger import logger

DuplicateKey = tuple[str, str, str]  # (direction_key, model_name, split)


@dataclass(frozen=True)
class ShardApplyResult:
    shard_path: str
    checkpoint_lines_removed: int
    part_files_modified: int
    part_rows_removed: int


@dataclass(frozen=True)
class ApplyResult:
    shards_processed: int
    total_checkpoint_lines_removed: int
    total_part_files_modified: int
    total_part_rows_removed: int
    shard_results: tuple[ShardApplyResult, ...]


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------

_REQUIRED_PLAN_FIELDS = ("dataset_path", "duplicates")
_REQUIRED_ENTRY_FIELDS = (
    "shard_path",
    "direction_key",
    "model_name",
    "split",
    "total_occurrences",
)


def load_plan(plan_path: Path) -> dict:
    if not plan_path.exists():
        raise SystemExit(f"Plan file does not exist: {plan_path}")
    if not plan_path.is_file():
        raise SystemExit(f"Plan path is not a file: {plan_path}")

    with plan_path.open("r", encoding="utf-8") as fh:
        try:
            plan = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in plan file {plan_path}: {exc}") from exc

    if not isinstance(plan, dict):
        raise SystemExit(f"Plan file root must be a JSON object: {plan_path}")

    for field in _REQUIRED_PLAN_FIELDS:
        if field not in plan:
            raise SystemExit(f"Plan file missing required field {field!r}: {plan_path}")

    if not isinstance(plan["duplicates"], list):
        raise SystemExit(f"Plan field 'duplicates' must be a list: {plan_path}")

    for idx, entry in enumerate(plan["duplicates"]):
        for field in _REQUIRED_ENTRY_FIELDS:
            if field not in entry:
                raise SystemExit(
                    f"Plan duplicate entry [{idx}] missing field {field!r}: "
                    f"{plan_path}"
                )

    return plan


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _backup_file(file_path: Path) -> Path:
    """Copy *file_path* to ``<file_path>.bak``, raising if the backup exists."""
    backup_path = file_path.parent / (file_path.name + ".bak")
    if backup_path.exists():
        raise RuntimeError(
            f"Backup already exists (previous dedup not cleaned up?): {backup_path}"
        )
    shutil.copy2(file_path, backup_path)
    return backup_path


def _atomic_write(file_path: Path, lines: list[str]) -> None:
    """Write *lines* to *file_path* atomically via a temporary file."""
    tmp_path = file_path.parent / (file_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.writelines(lines)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, file_path)


def _collect_part_files(shard_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in shard_dir.iterdir()
        if p.is_file() and p.name.startswith("part-") and p.name.endswith(".jsonl")
    )


# ---------------------------------------------------------------------------
# Checkpoint rewrite
# ---------------------------------------------------------------------------


def _rewrite_checkpoint(
    checkpoint_path: Path,
    duplicate_keys: set[DuplicateKey],
) -> int:
    """Rewrite *checkpoint_path* keeping only the first occurrence of each key.

    Returns the number of lines removed.
    """
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    seen_keys: set[DuplicateKey] = set()
    new_lines: list[str] = []
    removed = 0

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            new_lines.append(raw_line)
            continue

        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON at {checkpoint_path}:{line_no}: {exc}"
            ) from exc

        for field_name in ("direction_key", "model_name", "split"):
            if field_name not in record:
                raise RuntimeError(
                    f"Missing required field {field_name!r} at "
                    f"{checkpoint_path}:{line_no}"
                )

        key: DuplicateKey = (
            str(record["direction_key"]),
            str(record["model_name"]),
            str(record["split"]),
        )

        if key in duplicate_keys and key in seen_keys:
            removed += 1
            continue

        seen_keys.add(key)
        new_lines.append(raw_line)

    if removed == 0:
        return 0

    _backup_file(checkpoint_path)
    _atomic_write(checkpoint_path, new_lines)
    return removed


# ---------------------------------------------------------------------------
# Part-file rewrite
# ---------------------------------------------------------------------------


def _rewrite_part_files(
    shard_dir: Path,
    duplicate_keys: set[DuplicateKey],
    expected_occurrences: dict[DuplicateKey, int],
) -> tuple[int, int]:
    """Rewrite ``part-*.jsonl`` files, removing duplicate direction blocks.

    A *block* is a contiguous run of rows sharing the same
    ``(direction_key, model_name, split)``.  The first block for each key
    (across all part files processed in sorted name order) is kept; every
    subsequent block for a duplicate key is removed.

    Returns ``(files_modified, total_rows_removed)``.
    """
    part_files = _collect_part_files(shard_dir)
    # Track how many blocks we have seen per duplicate key across ALL files.
    occurrence_count: dict[DuplicateKey, int] = {}
    files_modified = 0
    total_rows_removed = 0

    for part_file in part_files:
        with part_file.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        new_lines: list[str] = []
        removed_in_file = 0
        prev_key: DuplicateKey | None = None
        skip_current_block = False

        for line_no, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                new_lines.append(raw_line)
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {part_file}:{line_no}: {exc}"
                ) from exc

            for field_name in ("direction_key", "model_name", "split"):
                if field_name not in record:
                    raise RuntimeError(
                        f"Missing required field {field_name!r} at "
                        f"{part_file}:{line_no}"
                    )

            key: DuplicateKey = (
                str(record["direction_key"]),
                str(record["model_name"]),
                str(record["split"]),
            )

            # Detect block transitions.
            if key != prev_key:
                prev_key = key
                if key in duplicate_keys:
                    occurrence_count[key] = occurrence_count.get(key, 0) + 1
                    skip_current_block = occurrence_count[key] > 1
                else:
                    skip_current_block = False

            if skip_current_block:
                removed_in_file += 1
                continue

            new_lines.append(raw_line)

        if removed_in_file > 0:
            _backup_file(part_file)
            _atomic_write(part_file, new_lines)
            files_modified += 1
            total_rows_removed += removed_in_file

    # Validate that the occurrence counts match the plan exactly.
    for key, expected in expected_occurrences.items():
        actual = occurrence_count.get(key, 0)
        if actual != expected:
            raise RuntimeError(
                f"Occurrence count mismatch in {shard_dir} for "
                f"direction_key={key[0]!r}, model_name={key[1]!r}, "
                f"split={key[2]!r}: plan expected {expected} occurrences in "
                f"part files, found {actual}"
            )

    return files_modified, total_rows_removed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_plan(plan_path: Path) -> ApplyResult:
    """Read a dedup plan and apply all deletions, creating ``.bak`` backups."""
    plan = load_plan(plan_path)
    duplicates = plan["duplicates"]

    if not duplicates:
        logger.info("No duplicates in plan — nothing to do.")
        return ApplyResult(
            shards_processed=0,
            total_checkpoint_lines_removed=0,
            total_part_files_modified=0,
            total_part_rows_removed=0,
            shard_results=(),
        )

    # Group duplicate entries by shard path.
    shard_groups: dict[str, list[dict]] = {}
    for entry in duplicates:
        shard_groups.setdefault(entry["shard_path"], []).append(entry)

    shard_results: list[ShardApplyResult] = []

    for shard_path_str in sorted(shard_groups):
        entries = shard_groups[shard_path_str]
        shard_dir = Path(shard_path_str)

        if not shard_dir.exists():
            raise RuntimeError(f"Shard directory from plan does not exist: {shard_dir}")
        if not shard_dir.is_dir():
            raise RuntimeError(f"Shard path from plan is not a directory: {shard_dir}")

        checkpoint_path = shard_dir / "checkpoint.jsonl"
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"checkpoint.jsonl missing in shard directory: {shard_dir}"
            )

        duplicate_keys: set[DuplicateKey] = set()
        expected_occurrences: dict[DuplicateKey, int] = {}
        for entry in entries:
            key: DuplicateKey = (
                entry["direction_key"],
                entry["model_name"],
                entry["split"],
            )
            duplicate_keys.add(key)
            expected_occurrences[key] = entry["total_occurrences"]

        logger.info(
            "Processing shard: %s (%d duplicate keys)",
            shard_path_str,
            len(duplicate_keys),
        )

        checkpoint_removed = _rewrite_checkpoint(checkpoint_path, duplicate_keys)
        part_files_modified, part_rows_removed = _rewrite_part_files(
            shard_dir, duplicate_keys, expected_occurrences
        )

        result = ShardApplyResult(
            shard_path=shard_path_str,
            checkpoint_lines_removed=checkpoint_removed,
            part_files_modified=part_files_modified,
            part_rows_removed=part_rows_removed,
        )
        shard_results.append(result)

        logger.info(
            "Shard %s done: checkpoint_removed=%d "
            "part_files_modified=%d part_rows_removed=%d",
            shard_path_str,
            checkpoint_removed,
            part_files_modified,
            part_rows_removed,
        )

    return ApplyResult(
        shards_processed=len(shard_results),
        total_checkpoint_lines_removed=sum(
            r.checkpoint_lines_removed for r in shard_results
        ),
        total_part_files_modified=sum(r.part_files_modified for r in shard_results),
        total_part_rows_removed=sum(r.part_rows_removed for r in shard_results),
        shard_results=tuple(shard_results),
    )
