from __future__ import annotations

import re
from pathlib import Path

_SHARD_RE = re.compile(r"^shard_(\d+)\.jsonl$")

__all__ = ["collect_complete_directions", "sorted_shard_files"]


def collect_complete_directions(
    conn, model_filter: str | None
) -> list[tuple[str, str]]:
    sql = """
        SELECT d.direction_key, d.model
          FROM directions d
         WHERE (? IS NULL OR d.model = ?)
           AND NOT EXISTS (
               SELECT 1 FROM jobs j
                WHERE j.direction_key = d.direction_key
                  AND j.model = d.model
                  AND j.status != 'done'
           )
    """
    rows = conn.execute(sql, (model_filter, model_filter)).fetchall()
    return [(row["direction_key"], row["model"]) for row in rows]


def sorted_shard_files(shard_dir: Path) -> list[tuple[int, Path]]:
    entries: list[tuple[int, Path]] = []
    for path in shard_dir.iterdir():
        if not path.is_file():
            continue
        match = _SHARD_RE.match(path.name)
        if not match:
            continue
        entries.append((int(match.group(1)), path))
    entries.sort(key=lambda item: item[0])
    return entries
