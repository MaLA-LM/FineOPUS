from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from utils.logger import logger

DEFAULT_OPUS_ROOT = "/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3"
DEFAULT_SPLIT = "all"
SPLIT_VALUES = (DEFAULT_SPLIT,)

DirectionSpec = tuple[str, str, str, Path]

# OPUS direction directories look like ``abk_Cyrl-por_Latn``. The ISO part is
# 2-4 lowercase letters; the Script part is exactly four letters whose first
# letter is usually capitalised.
_LANG_CODE = r"[a-z]{2,4}_[A-Za-z]{4}"
_DIR_RE = re.compile(rf"^({_LANG_CODE})-({_LANG_CODE})$")


def resolve_split(split: str | None) -> str:
    """Return OPUS's compatibility split placeholder.

    The shared manifest/executor interfaces still expect a split column, but
    OPUS itself has no split concept. Accept ``None``/``"all"`` and reject any
    real split names.
    """
    if split is None:
        return DEFAULT_SPLIT
    normalized = split.strip()
    if not normalized or normalized == DEFAULT_SPLIT:
        return DEFAULT_SPLIT
    raise ValueError(
        f"OPUS has no named splits; got split={split!r}. "
        f"Use {DEFAULT_SPLIT!r} or omit the split."
    )


def _iter_parquet_paths(direction_dir: Path) -> list[Path]:
    return sorted(
        Path(entry.path)
        for entry in os.scandir(direction_dir)
        if entry.is_file() and entry.name.endswith(".parquet")
    )


def _has_parquet(direction_dir: Path) -> bool:
    return bool(_iter_parquet_paths(direction_dir))


@lru_cache(maxsize=None)
def _count_parquet_rows(direction_dir_str: str) -> int:
    import pyarrow.parquet as pq

    total = 0
    direction_dir = Path(direction_dir_str)
    for path in _iter_parquet_paths(direction_dir):
        parquet_file = pq.ParquetFile(str(path))
        total += int(parquet_file.metadata.num_rows)
    return total


def discover_directions(
    root: str | Path, split: str | None = None
) -> list[DirectionSpec]:
    resolved_split = resolve_split(split)
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"OPUS root not found: {root_path}")

    directions: list[DirectionSpec] = []
    skipped_empty = 0

    for entry in os.scandir(root_path):
        if not entry.is_dir():
            continue
        match = _DIR_RE.match(entry.name)
        if not match:
            continue
        src_lang, tgt_lang = match.group(1), match.group(2)
        if src_lang == tgt_lang:
            continue
        direction_dir = Path(entry.path)
        if not _has_parquet(direction_dir):
            skipped_empty += 1
            logger.warning(
                "OPUS direction has no parquet files, skipping: %s", direction_dir
            )
            continue
        directions.append((src_lang, tgt_lang, resolved_split, direction_dir))

    if skipped_empty:
        logger.warning("OPUS discovery skipped %d empty direction(s).", skipped_empty)
    directions.sort(key=lambda item: (item[0], item[1]))
    return directions


def expected_detail_rows(
    root: str | Path, split: str, src_lang: str, max_rows: int | None
) -> int | None:
    """Return the row count for one OPUS direction when available.

    The shared adapter hook only exposes one string key after ``split``. For
    OPUS, callers can pass the full direction directory name
    (for example ``eng_Latn-fra_Latn``) in that slot.
    """
    direction_dir = Path(root) / src_lang
    if not direction_dir.exists():
        return max_rows

    total = _count_parquet_rows(str(direction_dir))
    if max_rows is None:
        return total
    return min(max_rows, total)
