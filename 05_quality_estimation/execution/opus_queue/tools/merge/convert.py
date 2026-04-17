from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from execution.opus_queue.tools.merge.collect import sorted_shard_files
from utils.logger import logger

__all__ = ["merge_direction"]


def _read_shard_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as reader:
        for line_no, raw_line in enumerate(reader, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%s", path, line_no)
    return records


def merge_direction(
    output_base: Path,
    merged_base: Path,
    source_db: Path,
    direction_key: str,
    model: str,
    *,
    force: bool,
    delete_shards: bool,
) -> tuple[bool, int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = output_base / model / direction_key
    if not shard_dir.exists():
        logger.warning("No shard dir for %s/%s at %s", model, direction_key, shard_dir)
        return False, 0, 0

    shards = sorted_shard_files(shard_dir)
    if not shards:
        logger.warning("No shard files in %s", shard_dir)
        return False, 0, 0

    out_file = merged_base / model / f"{direction_key}.parquet"
    meta_file = merged_base / model / f"{direction_key}.meta.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        logger.info("Skipping merged direction (exists): %s", out_file)
        return False, 0, 0

    records: list[dict] = []
    for _shard_id, path in shards:
        records.extend(_read_shard_records(path))

    if not records:
        logger.warning("No records found in %s; skipping merge.", shard_dir)
        return False, len(shards), 0

    table = pa.Table.from_pylist(records)
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    with tmp_file.open("wb") as writer:
        pq.write_table(table, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(tmp_file, out_file)

    total_rows = int(table.num_rows)
    meta = {
        "n_shards": len(shards),
        "n_rows": total_rows,
        "source_db": str(source_db),
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "direction_key": direction_key,
        "model": model,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if delete_shards:
        for _shard_id, path in shards:
            try:
                path.unlink()
            except OSError:
                logger.warning("Failed to delete shard file: %s", path)

    logger.info(
        "Merged %s/%s: shards=%d rows=%d -> %s",
        model,
        direction_key,
        len(shards),
        total_rows,
        out_file,
    )
    return True, len(shards), total_rows
