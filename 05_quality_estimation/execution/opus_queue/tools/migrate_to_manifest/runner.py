from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from execution.opus_queue.planning import PendingShard, assign_shards
from utils.logger import logger

__all__ = ["run"]


def _default_build_tag() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"DB not found: {path}")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_statuses(raw: str) -> tuple[str, ...]:
    statuses = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not statuses:
        raise SystemExit("--include-status must name at least one status")
    return statuses


def _parse_slots(entries: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    slots: dict[str, int] = {}
    slots_per_task: dict[str, int] = {}
    for raw in entries:
        parts = raw.split(":")
        if len(parts) not in {2, 3}:
            raise SystemExit(
                f"--slots expects model:slots or model:slots:slots_per_array_task, got {raw!r}"
            )
        model = parts[0].strip()
        if not model:
            raise SystemExit(f"--slots has empty model in {raw!r}")
        try:
            slot_count = int(parts[1])
        except ValueError as exc:
            raise SystemExit(f"--slots count must be an int in {raw!r}") from exc
        if slot_count <= 0:
            raise SystemExit(f"--slots count must be > 0 in {raw!r}")
        slots[model] = slot_count
        if len(parts) == 3:
            try:
                per_task = int(parts[2])
            except ValueError as exc:
                raise SystemExit(f"--slots per-task value must be an int in {raw!r}") from exc
            if per_task <= 0:
                raise SystemExit(f"--slots per-task value must be > 0 in {raw!r}")
            slots_per_task[model] = per_task
    return slots, slots_per_task


def _parse_slots_per_task(entries: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw in entries:
        if ":" not in raw:
            raise SystemExit(f"--slots-per-task expects model:int, got {raw!r}")
        model, value = raw.split(":", 1)
        model = model.strip()
        try:
            per_task = int(value.strip())
        except ValueError as exc:
            raise SystemExit(f"--slots-per-task value must be int in {raw!r}") from exc
        if not model or per_task <= 0:
            raise SystemExit(f"--slots-per-task expects non-empty model and value > 0")
        parsed[model] = per_task
    return parsed


def _fetch_pending_shards(conn: sqlite3.Connection, statuses: tuple[str, ...]) -> list[PendingShard]:
    placeholders = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"""
        SELECT j.direction_key,
               j.model,
               j.shard_id,
               j.start_idx,
               j.end_idx,
               d.src_lang,
               d.tgt_lang,
               d.shard_size
          FROM jobs j
          JOIN directions d
            ON d.direction_key = j.direction_key
           AND d.model = j.model
         WHERE j.status IN ({placeholders})
         ORDER BY j.model ASC, j.direction_key ASC, j.shard_id ASC
        """,
        statuses,
    ).fetchall()
    return [
        PendingShard(
            model=str(row["model"]),
            direction_key=str(row["direction_key"]),
            src_lang=str(row["src_lang"]),
            tgt_lang=str(row["tgt_lang"]),
            shard_id=int(row["shard_id"]),
            start_idx=int(row["start_idx"]),
            end_idx=int(row["end_idx"]),
            shard_size=int(row["shard_size"]),
        )
        for row in rows
    ]


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for row in rows
    )
    _write_text_atomic(path, content)


def _readme(build_tag: str) -> str:
    return (
        "OPUS QE manifest build\n"
        f"build_tag={build_tag}\n\n"
        "Workers must be launched with:\n"
        f"  --manifest-root <this root> --build-tag {build_tag} --trace-root <trace root>\n\n"
        "For standard-g manifests, pack with slots_per_array_task=8 so worker slots\n"
        "match SLURM_LOCALID=0..7. For small-g, use slots_per_array_task=1.\n\n"
        "All selected shards are assigned to the declared worker slots. If the\n"
        "summary estimates multiple waves, resubmit the same array range with the\n"
        "same build tag; workers skip shards already completed in state.jsonl.\n"
    )


def run(args) -> int:
    build_tag = args.build_tag or _default_build_tag()
    statuses = _parse_statuses(args.include_status)
    slots_by_model, slots_per_task_from_slots = _parse_slots(args.slots)
    slots_per_task = slots_per_task_from_slots
    slots_per_task.update(_parse_slots_per_task(args.slots_per_task))

    if args.walltime_seconds <= 0:
        raise SystemExit("--walltime-seconds must be > 0")
    if args.safety_factor <= 0:
        raise SystemExit("--safety-factor must be > 0")

    conn = _connect_readonly(args.db)
    try:
        pending = _fetch_pending_shards(conn, statuses)
    finally:
        conn.close()

    grouped: dict[str, list[PendingShard]] = defaultdict(list)
    for shard in pending:
        grouped[shard.model].append(shard)

    missing = sorted(set(grouped) - set(slots_by_model))
    if missing:
        raise SystemExit(
            "--slots is required for every model in the selected statuses; missing: "
            + ", ".join(missing)
        )

    manifest_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, dict[str, Any]] = {}
    for model in sorted(grouped):
        assignments, summary = assign_shards(
            grouped[model],
            slots_declared=slots_by_model[model],
            slots_per_array_task=slots_per_task.get(model, 1),
            walltime_seconds=args.walltime_seconds,
            safety_factor=args.safety_factor,
        )
        manifest_rows.extend(row.to_dict() for row in assignments)
        model_summaries[model] = summary.to_dict()
        if summary.estimated_waves > 1:
            logger.info(
                "Manifest model=%s assigned all %d selected shards across %d slots; "
                "estimated_waves=%d max_load_seconds=%d usable_walltime_seconds=%d",
                model,
                summary.assigned_shards,
                summary.slots_declared,
                summary.estimated_waves,
                summary.max_load_seconds,
                summary.usable_walltime_seconds,
            )

    manifest_rows.sort(
        key=lambda row: (
            row["model"],
            row["array_task_id"],
            row["local_id"],
            row["assignment_seq"],
            row["direction_key"],
            row["shard_id"],
        )
    )
    summary_payload = {
        "build_tag": build_tag,
        "created_at": int(time.time()),
        "source_db": str(Path(args.db).expanduser().resolve()),
        "include_status": list(statuses),
        "walltime_seconds": int(args.walltime_seconds),
        "safety_factor": float(args.safety_factor),
        "models": model_summaries,
    }

    if args.dry_run:
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_dir = Path(args.manifest_root).expanduser() / build_tag
    _write_manifest(output_dir / "manifest.jsonl", manifest_rows)
    _write_text_atomic(
        output_dir / "manifest.summary.json",
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(output_dir / "README.txt", _readme(build_tag))

    logger.info(
        "Wrote manifest build_tag=%s rows=%d dir=%s",
        build_tag,
        len(manifest_rows),
        output_dir,
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
