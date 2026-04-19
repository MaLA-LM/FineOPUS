from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_SHARD_RE = re.compile(r"^shard_(\d+)\.jsonl$")
_PART_RE = re.compile(r"^part-(.+)-(\d{4})\.jsonl$")

__all__ = ["PartFileInfo", "collect_complete_directions", "done_jobs_for_direction", "sorted_part_files"]


@dataclass(frozen=True)
class PartFileInfo:
    path: Path
    kind: Literal["legacy", "part"]
    shard_id: int | None = None
    worker_id: str | None = None
    seq: int | None = None


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


def done_jobs_for_direction(conn, direction_key: str, model: str) -> dict[int, str]:
    rows = conn.execute(
        """
        SELECT shard_id, worker_id
          FROM jobs
         WHERE direction_key = ? AND model = ? AND status = 'done'
        """,
        (direction_key, model),
    ).fetchall()
    winners: dict[int, str] = {}
    for row in rows:
        worker_id = row["worker_id"]
        if worker_id is None:
            continue
        winners[int(row["shard_id"])] = str(worker_id)
    return winners


def sorted_part_files(shard_dir: Path) -> list[PartFileInfo]:
    entries: list[PartFileInfo] = []
    for path in shard_dir.iterdir():
        if not path.is_file():
            continue
        legacy_match = _SHARD_RE.match(path.name)
        if legacy_match:
            entries.append(
                PartFileInfo(
                    path=path,
                    kind="legacy",
                    shard_id=int(legacy_match.group(1)),
                )
            )
            continue
        part_match = _PART_RE.match(path.name)
        if part_match:
            entries.append(
                PartFileInfo(
                    path=path,
                    kind="part",
                    worker_id=part_match.group(1),
                    seq=int(part_match.group(2)),
                )
            )
    entries.sort(
        key=lambda item: (
            0 if item.kind == "part" else 1,
            item.worker_id or "",
            item.seq if item.seq is not None else -1,
            item.shard_id if item.shard_id is not None else -1,
            item.path.name,
        )
    )
    return entries
