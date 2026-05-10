from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from utils.logger import logger

__all__ = ["CleanupResult", "delete_merged_direction_inputs"]


@dataclass(frozen=True)
class CleanupResult:
    deleted_dirs: int
    missing_source_dirs: int
    skipped_unmerged_dirs: int
    deleted_files: int
    deleted_bytes: int
    dry_run: bool


def _contains_merged_parquet(direction_dir: Path, direction_key: str) -> bool:
    if not direction_dir.is_dir():
        return False
    legacy_file = direction_dir / f"{direction_key}.parquet"
    if legacy_file.is_file():
        return True
    return any(direction_dir.glob(f"{direction_key}.part-*.parquet"))


def _directory_stats(path: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        files += 1
        try:
            total_bytes += item.stat().st_size
        except OSError:
            logger.warning("Could not stat source file during cleanup: %s", item)
    return files, total_bytes


def _resolve_existing_parent(path: Path) -> Path:
    return path.parent.resolve() / path.name


def delete_merged_direction_inputs(
    output_base: str | Path,
    merged_base: str | Path,
    *,
    model_filter: str | None = None,
    dry_run: bool = False,
) -> CleanupResult:
    """Delete source JSONL direction directories whose merged output exists.

    The merge completion marker is a direction directory under
    ``merged_base/model/direction_key`` containing at least one parquet file.
    Only the matching source directory under ``output_base`` is removed.
    """

    output_root = Path(output_base).expanduser().resolve()
    merged_root = Path(merged_base).expanduser().resolve()
    if not merged_root.is_dir():
        logger.warning("Merged base does not exist; nothing to clean: %s", merged_root)
        return CleanupResult(0, 0, 0, 0, 0, dry_run)
    if not output_root.exists():
        logger.warning("Output base does not exist; nothing to clean: %s", output_root)
        return CleanupResult(0, 0, 0, 0, 0, dry_run)

    model_dirs = (
        [merged_root / model_filter]
        if model_filter is not None
        else sorted(path for path in merged_root.iterdir() if path.is_dir())
    )
    deleted_dirs = 0
    missing_source_dirs = 0
    skipped_unmerged_dirs = 0
    deleted_files = 0
    deleted_bytes = 0

    for merged_model_dir in model_dirs:
        if not merged_model_dir.is_dir():
            logger.info("No merged outputs for model during cleanup: %s", merged_model_dir)
            continue
        model = merged_model_dir.name
        output_model_dir = output_root / model
        output_model_root = _resolve_existing_parent(output_model_dir)

        for merged_direction_dir in sorted(
            path for path in merged_model_dir.iterdir() if path.is_dir()
        ):
            direction_key = merged_direction_dir.name
            if not _contains_merged_parquet(merged_direction_dir, direction_key):
                skipped_unmerged_dirs += 1
                logger.warning(
                    "Skipping cleanup for merged marker without parquet: %s",
                    merged_direction_dir,
                )
                continue

            source_dir = output_model_dir / direction_key
            if not source_dir.exists():
                missing_source_dirs += 1
                continue
            if not source_dir.is_dir():
                skipped_unmerged_dirs += 1
                logger.warning("Skipping non-directory source path: %s", source_dir)
                continue

            resolved_source_dir = source_dir.resolve()
            try:
                resolved_source_dir.relative_to(output_model_root)
            except ValueError as exc:
                raise ValueError(
                    f"Refusing to delete source outside output base: {resolved_source_dir}"
                ) from exc

            files, total_bytes = _directory_stats(resolved_source_dir)
            deleted_files += files
            deleted_bytes += total_bytes
            deleted_dirs += 1
            if dry_run:
                logger.info(
                    "Would delete merged source direction: %s files=%d bytes=%d",
                    resolved_source_dir,
                    files,
                    total_bytes,
                )
                continue

            shutil.rmtree(resolved_source_dir)
            logger.info(
                "Deleted merged source direction: %s files=%d bytes=%d",
                resolved_source_dir,
                files,
                total_bytes,
            )

    return CleanupResult(
        deleted_dirs=deleted_dirs,
        missing_source_dirs=missing_source_dirs,
        skipped_unmerged_dirs=skipped_unmerged_dirs,
        deleted_files=deleted_files,
        deleted_bytes=deleted_bytes,
        dry_run=dry_run,
    )
