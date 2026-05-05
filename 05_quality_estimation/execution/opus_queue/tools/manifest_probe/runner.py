from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from execution.opus_queue.manifest.reader import manifest_path
from execution.opus_queue.trace.reader import read_events

__all__ = ["run"]


def _parse_array_spec(raw: str) -> set[int]:
    spec = raw.split("%", 1)[0]
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            values.add(int(part))
            continue
        range_part, _, step_part = part.partition(":")
        start_s, end_s = range_part.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        step = int(step_part) if step_part else 1
        if step <= 0:
            raise ValueError("array step must be > 0")
        if end < start:
            raise ValueError("array range end must be >= start")
        values.update(range(start, end + 1, step))
    if not values:
        raise ValueError("empty array spec")
    return values


def _format_ids(ids: set[int]) -> str:
    if not ids:
        return "<none>"
    ordered = sorted(ids)
    if len(ordered) <= 8:
        return ",".join(str(value) for value in ordered)
    return f"{ordered[0]}-{ordered[-1]} ({len(ordered)} values)"


def _load_model_rows(path: Path, model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid manifest JSON at {path}:{lineno}: {exc}") from exc
            if row.get("model") == model:
                rows.append(row)
    return rows


def _load_model_summary(path: Path, model: str) -> dict[str, Any] | None:
    summary_path = path.with_name("manifest.summary.json")
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid manifest summary JSON at {summary_path}: {exc}") from exc
    models = payload.get("models", {})
    model_summary = models.get(model)
    return model_summary if isinstance(model_summary, dict) else None


def _done_keys(trace_root: Path, build_tag: str, model: str) -> set[tuple[str, str, int]]:
    root = trace_root / build_tag
    if not root.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    trace_paths = list(root.glob("*/events.jsonl")) + list(root.glob("*/state.jsonl"))
    for trace_path in trace_paths:
        for event in read_events(trace_path):
            if event.get("event") != "done" or event.get("model") != model:
                continue
            done.add(
                (
                    str(event["model"]),
                    str(event["direction_key"]),
                    int(event["shard_id"]),
                )
            )
    return done


def run(args) -> int:
    if args.slots_per_task <= 0:
        raise SystemExit("--slots-per-task must be > 0")

    path = manifest_path(args.manifest_root, args.build_tag)
    if not path.is_file():
        raise SystemExit(f"ERROR: manifest not found: {path}")
    if not path.stat().st_size:
        raise SystemExit(f"ERROR: manifest is empty: {path}")

    rows = _load_model_rows(path, args.model)
    if not rows:
        raise SystemExit(
            f"ERROR: manifest has no rows for model={args.model!r} in {path}"
        )

    try:
        requested_array_ids = _parse_array_spec(args.array_spec)
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid --array spec {args.array_spec!r}: {exc}") from exc

    manifest_array_ids = {int(row["array_task_id"]) for row in rows}
    model_summary = _load_model_summary(path, args.model)
    if model_summary is not None:
        slots_declared = int(model_summary["slots_declared"])
        manifest_slots_per_task = int(
            model_summary.get("slots_per_array_task", args.slots_per_task)
        )
        if manifest_slots_per_task != int(args.slots_per_task):
            raise SystemExit(
                "ERROR: manifest was packed with "
                f"slots_per_array_task={manifest_slots_per_task} for model={args.model}, "
                f"but this submitter uses {args.slots_per_task}."
            )
        if slots_declared % int(args.slots_per_task) != 0:
            raise SystemExit(
                f"ERROR: manifest slots_declared={slots_declared} is not divisible "
                f"by slots_per_task={args.slots_per_task} for model={args.model}."
            )
        manifest_expected_array_ids = set(
            range(slots_declared // int(args.slots_per_task))
        )
    else:
        manifest_expected_array_ids = manifest_array_ids

    unknown_array_ids = requested_array_ids - manifest_expected_array_ids
    if unknown_array_ids:
        expected_slots = (max(requested_array_ids) + 1) * int(args.slots_per_task)
        raise SystemExit(
            "ERROR: requested array contains ids not present in the manifest "
            f"for model={args.model}. Unknown ids={_format_ids(unknown_array_ids)}; "
            f"manifest ids={_format_ids(manifest_expected_array_ids)}. "
            f"Re-pack with at least --slots {args.model}:{expected_slots}:{args.slots_per_task} "
            "or submit an array subset covered by the manifest."
        )

    assigned_keys = {
        (str(row["model"]), str(row["direction_key"]), int(row["shard_id"]))
        for row in rows
    }
    selected_rows = [
        row for row in rows if int(row["array_task_id"]) in requested_array_ids
    ]
    selected_keys = {
        (str(row["model"]), str(row["direction_key"]), int(row["shard_id"]))
        for row in selected_rows
    }
    done_keys = _done_keys(Path(args.trace_root).expanduser(), args.build_tag, args.model)
    done_in_manifest = assigned_keys & done_keys
    done_in_selected = selected_keys & done_keys
    remaining = len(assigned_keys) - len(done_in_manifest)
    selected_remaining = len(selected_keys) - len(done_in_selected)
    print(
        "Manifest probe: "
        f"model={args.model} build_tag={args.build_tag} "
        f"assigned_shards={len(assigned_keys)} "
        f"done_in_trace={len(done_in_manifest)} remaining={remaining} "
        f"selected_assigned_shards={len(selected_keys)} "
        f"selected_done_in_trace={len(done_in_selected)} "
        f"selected_remaining={selected_remaining} "
        f"requested_array_task_ids={_format_ids(requested_array_ids)} "
        f"manifest_declared_array_task_ids={_format_ids(manifest_expected_array_ids)} "
        f"manifest_nonempty_array_task_ids={_format_ids(manifest_array_ids)}"
    )
    if requested_array_ids != manifest_expected_array_ids:
        print(
            "WARNING: partial manifest submission; only the requested array task "
            "ids will run in this job."
        )
    if not selected_keys:
        print(
            f"WARNING: requested array task ids have no assigned shards for model={args.model}."
        )
    if remaining == 0:
        print(
            f"WARNING: manifest has no unfinished shards for model={args.model} "
            f"build_tag={args.build_tag}.",
        )
    return 0
