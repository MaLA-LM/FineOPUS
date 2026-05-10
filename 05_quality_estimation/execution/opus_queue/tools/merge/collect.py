from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from execution.opus_queue.manifest.reader import manifest_path
from execution.opus_queue.trace.reader import read_events

_SHARD_RE = re.compile(r"^shard_(\d+)\.jsonl$")
_PART_RE = re.compile(r"^part-(.+)-(\d{4})\.jsonl$")

__all__ = [
    "CompletedDirection",
    "CompletedShard",
    "PartFileInfo",
    "collect_complete_combined_directions",
    "collect_complete_directions",
    "collect_complete_manifest_directions",
    "done_jobs_for_direction",
    "sorted_part_files",
]


@dataclass(frozen=True)
class CompletedShard:
    worker_id: str
    worker_run_id: str | None = None


@dataclass(frozen=True)
class CompletedDirection:
    direction_key: str
    model: str
    winners: dict[int, CompletedShard]


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


def done_jobs_for_direction(
    conn, direction_key: str, model: str
) -> dict[int, CompletedShard]:
    rows = conn.execute(
        """
        SELECT shard_id, worker_id
          FROM jobs
         WHERE direction_key = ? AND model = ? AND status = 'done'
        """,
        (direction_key, model),
    ).fetchall()
    winners: dict[int, CompletedShard] = {}
    for row in rows:
        worker_id = row["worker_id"]
        if worker_id is None:
            continue
        winners[int(row["shard_id"])] = CompletedShard(worker_id=str(worker_id))
    return winners


def _db_shards_and_winners(
    conn,
    model_filter: str | None,
) -> tuple[
    dict[tuple[str, str], set[int]],
    dict[tuple[str, str], dict[int, CompletedShard]],
]:
    rows = conn.execute(
        """
        SELECT d.direction_key,
               d.model,
               j.shard_id,
               j.status,
               j.worker_id
          FROM directions d
          JOIN jobs j
            ON j.direction_key = d.direction_key
           AND j.model = d.model
         WHERE (? IS NULL OR d.model = ?)
         ORDER BY d.model ASC, d.direction_key ASC, j.shard_id ASC
        """,
        (model_filter, model_filter),
    ).fetchall()
    assigned: dict[tuple[str, str], set[int]] = defaultdict(set)
    winners: dict[tuple[str, str], dict[int, CompletedShard]] = defaultdict(dict)
    for row in rows:
        model = str(row["model"])
        direction_key = str(row["direction_key"])
        shard_id = int(row["shard_id"])
        key = (model, direction_key)
        assigned[key].add(shard_id)
        if row["status"] != "done":
            continue
        worker_id = row["worker_id"]
        if worker_id is None:
            continue
        winners[key][shard_id] = CompletedShard(worker_id=str(worker_id))
    return dict(assigned), {key: dict(value) for key, value in winners.items()}


def _load_manifest_rows(
    manifest_root: str | Path, build_tag: str, model_filter: str | None
) -> dict[tuple[str, str], set[int]]:
    path = manifest_path(manifest_root, build_tag)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    assigned: dict[tuple[str, str], set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid manifest JSON at {path}:{lineno}: {exc}") from exc
            model = str(row["model"])
            if model_filter is not None and model != model_filter:
                continue
            direction_key = str(row["direction_key"])
            assigned[(model, direction_key)].add(int(row["shard_id"]))
    return assigned


def _trace_winners(
    trace_root: str | Path, build_tag: str, model_filter: str | None
) -> dict[tuple[str, str], dict[int, CompletedShard]]:
    root = Path(trace_root).expanduser() / build_tag
    winners: dict[tuple[str, str], dict[int, CompletedShard]] = defaultdict(dict)
    if not root.exists():
        return winners

    trace_paths = sorted(root.glob("*/state.jsonl"))
    for trace_path in trace_paths:
        for event in read_events(trace_path):
            if event.get("event") != "done":
                continue
            model = str(event["model"])
            if model_filter is not None and model != model_filter:
                continue
            direction_key = str(event["direction_key"])
            shard_id = int(event["shard_id"])
            worker_id = str(event.get("worker_slot_id") or event.get("worker_id") or "")
            if not worker_id:
                continue
            worker_run_id = event.get("worker_run_id")
            winners[(model, direction_key)][shard_id] = CompletedShard(
                worker_id=worker_id,
                worker_run_id=str(worker_run_id) if worker_run_id else None,
            )
    return winners


def collect_complete_manifest_directions(
    manifest_root: str | Path,
    build_tag: str,
    trace_root: str | Path,
    model_filter: str | None,
) -> list[CompletedDirection]:
    assigned = _load_manifest_rows(manifest_root, build_tag, model_filter)
    winners_by_direction = _trace_winners(trace_root, build_tag, model_filter)
    complete: list[CompletedDirection] = []
    for (model, direction_key), assigned_shards in sorted(assigned.items()):
        winners = winners_by_direction.get((model, direction_key), {})
        if not assigned_shards or not assigned_shards.issubset(winners):
            continue
        complete.append(
            CompletedDirection(
                direction_key=direction_key,
                model=model,
                winners={shard_id: winners[shard_id] for shard_id in sorted(assigned_shards)},
            )
        )
    return complete


def collect_complete_combined_directions(
    conn,
    model_filter: str | None,
    *,
    manifest_root: str | Path | None = None,
    build_tag: str | None = None,
    trace_root: str | Path | None = None,
) -> list[CompletedDirection]:
    """Collect mergeable directions from DB rows, manifest trace, or both.

    When the DB is available, the DB job table is the full shard inventory for
    each direction. Manifest trace completions can satisfy those DB rows, which
    is the migration case where old completed shards remain in the DB while the
    remaining shards complete through manifest workers. Manifest-only
    directions are still supported when no DB rows exist for that direction.
    """

    db_assigned: dict[tuple[str, str], set[int]] = {}
    db_winners: dict[tuple[str, str], dict[int, CompletedShard]] = {}
    if conn is not None:
        db_assigned, db_winners = _db_shards_and_winners(conn, model_filter)

    manifest_assigned: dict[tuple[str, str], set[int]] = {}
    trace_winners: dict[tuple[str, str], dict[int, CompletedShard]] = {}
    has_manifest_source = bool(manifest_root or build_tag or trace_root)
    if has_manifest_source:
        if manifest_root is None or build_tag is None or trace_root is None:
            raise ValueError(
                "manifest_root, build_tag, and trace_root are required together"
            )
        manifest_assigned = _load_manifest_rows(
            manifest_root,
            build_tag,
            model_filter,
        )
        raw_trace_winners = _trace_winners(trace_root, build_tag, model_filter)
        for key, winners in raw_trace_winners.items():
            assigned_shards = manifest_assigned.get(key)
            if not assigned_shards:
                continue
            trace_winners[key] = {
                shard_id: winner
                for shard_id, winner in winners.items()
                if shard_id in assigned_shards
            }

    complete: list[CompletedDirection] = []
    for model, direction_key in sorted(set(db_assigned) | set(manifest_assigned)):
        key = (model, direction_key)
        expected_shards = db_assigned.get(key) or manifest_assigned.get(key) or set()
        if not expected_shards:
            continue

        winners: dict[int, CompletedShard] = {}
        winners.update(db_winners.get(key, {}))
        # Trace completions are newer than DB rows during migration, so prefer
        # them if a shard appears in both sources.
        winners.update(trace_winners.get(key, {}))

        if not expected_shards.issubset(winners):
            continue

        complete.append(
            CompletedDirection(
                direction_key=direction_key,
                model=model,
                winners={
                    shard_id: winners[shard_id]
                    for shard_id in sorted(expected_shards)
                },
            )
        )
    return complete


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
