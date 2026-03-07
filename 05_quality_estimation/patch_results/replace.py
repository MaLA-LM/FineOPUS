"""Replace original checkpoint / part files with their patched versions.

For every shard directory found under ``--model-path``:
  - ``checkpoint.jsonl``      is replaced by ``checkpoint-patched.jsonl``
  - ``part-*-patched.jsonl``  replaces the matching ``part-*.jsonl``

The old files are deleted and the ``-patched`` suffix is stripped.
"""

from __future__ import annotations

from pathlib import Path

from patch_results.discovery import discover_shards_needed, Shardwork
from utils.logger import logger


def _replace_file(old: Path, patched: Path) -> None:
    """Delete *old* and rename *patched* to *old*'s name."""
    if not patched.exists():
        logger.warning("Patched file missing, skipping: %s", patched)
        return
    if old.exists():
        old.unlink()
    patched.rename(old)
    logger.info("Replaced %s", old)


def _replace_shard(shard: Shardwork) -> int:
    """Replace originals inside one shard directory. Returns count of files replaced."""
    count = 0

    # checkpoint
    patched_checkpoint = shard.checkpoint_path.parent / "checkpoint-patched.jsonl"
    _replace_file(shard.checkpoint_path, patched_checkpoint)
    if not patched_checkpoint.exists():
        count += 1

    # part files
    for part_file in sorted(shard.part_files):
        # Only consider original part files — skip already-patched ones
        if part_file.stem.endswith("-patched"):
            continue
        patched_part = part_file.parent / f"{part_file.stem}-patched.jsonl"
        _replace_file(part_file, patched_part)
        if not patched_part.exists():
            count += 1

    return count


def replace(model_path: Path) -> None:
    """Discover shards and swap patched files for originals."""
    shards = discover_shards_needed(model_path)
    if not shards:
        logger.warning("No shards found under %s – nothing to replace.", model_path)
        return

    total = 0
    for shard in shards:
        total += _replace_shard(shard)

    logger.info(
        "Replacement complete – %d files swapped across %d shards.", total, len(shards)
    )
