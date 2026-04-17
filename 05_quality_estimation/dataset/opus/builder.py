from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from dataset.opus.discovery import DEFAULT_OPUS_ROOT, resolve_split
from utils.logger import logger

Example = dict[str, Any]

REQUIRED_COLUMNS = ("source_text", "target_text")


def _iter_parquet_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
    """Stream rows from a list of parquet shards row-group by row-group.

    Using ``ParquetFile.read_row_group`` keeps peak memory bounded to a single
    row group per file, which matters for the most imbalanced OPUS directions
    where a single direction can hold millions of sentence pairs.
    """
    import pyarrow.parquet as pq

    for path in paths:
        try:
            parquet_file = pq.ParquetFile(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open parquet file: {path}") from exc

        schema_names = set(parquet_file.schema_arrow.names)
        missing = [col for col in REQUIRED_COLUMNS if col not in schema_names]
        if missing:
            raise ValueError(
                f"OPUS parquet {path} is missing required columns: {missing}"
            )

        num_row_groups = parquet_file.num_row_groups
        for rg_index in range(num_row_groups):
            table = parquet_file.read_row_group(rg_index)
            for row in table.to_pylist():
                yield row


def load_opus_parallel(
    src_lang: str,
    tgt_lang: str,
    *,
    split: str | None = None,
    root: str | Path = DEFAULT_OPUS_ROOT,
) -> list[Example]:

    root_path = Path(root)
    direction_dir = root_path / f"{src_lang}-{tgt_lang}"
    if not direction_dir.exists():
        raise FileNotFoundError(f"OPUS direction dir not found: {direction_dir}")

    shard_files = sorted(p for p in direction_dir.glob("*.parquet") if p.is_file())
    if not shard_files:
        raise FileNotFoundError(f"No parquet files under: {direction_dir}")

    examples: list[Example] = []
    for row in _iter_parquet_rows(shard_files):
        src = row.get("source_text")
        tgt = row.get("target_text")
        if src is None or tgt is None:
            continue
        # Scorers read ``ex["src"]`` / ``ex["tgt"]``; keep the original
        # ``source_text`` / ``target_text`` keys too so the frame builder can
        # preserve every original parquet column on detail rows.
        row["src"] = src
        row["tgt"] = tgt
        examples.append(row)

    logger.info(
        "OPUS loaded direction %s-%s: shards=%d rows=%d",
        src_lang,
        tgt_lang,
        len(shard_files),
        len(examples),
    )
    return examples


def limit_rows(examples: list[Example], max_rows: int | None) -> list[Example]:
    from dataset.mediator import limit_rows as _shared_limit_rows

    return _shared_limit_rows(examples, max_rows)
