import json
from collections import defaultdict
from pathlib import Path

from stand_alone_modules.opus_trace_summary.constants import DONE
from stand_alone_modules.opus_trace_summary.io import (
    find_worker_dirs,
    iter_jsonl,
    read_json,
)
from stand_alone_modules.opus_trace_summary.keys import parse_state_key, safe_int
from stand_alone_modules.opus_trace_summary.records import (
    assignment_meta,
    done_meta,
    state_meta,
)


def merge_assignment(assignments, duplicate_assignments, key, meta, source):
    previous = assignments.get(key)
    if previous is None:
        meta = dict(meta)
        meta["assignment_source"] = source
        assignments[key] = meta
        return

    previous_worker = previous.get("worker_slot_id")
    worker = meta.get("worker_slot_id")
    if previous_worker and worker and previous_worker != worker:
        duplicate_assignments[key].add(previous_worker)
        duplicate_assignments[key].add(worker)

    if previous.get("assignment_source") == "manifest":
        return
    if source == "manifest":
        meta = dict(meta)
        meta["assignment_source"] = source
        assignments[key] = meta
        return

    for field, value in meta.items():
        if previous.get(field) is None and value is not None:
            previous[field] = value


def merge_done(completed, duplicate_completions, key, meta, source):
    meta = dict(meta)
    meta["source"] = source
    previous = completed.get(key)
    if previous is None:
        completed[key] = meta
        return
    previous_worker = previous.get("worker_slot_id")
    worker = meta.get("worker_slot_id")
    previous_run = previous.get("worker_run_id")
    worker_run = meta.get("worker_run_id")
    if (previous_worker, previous_run) != (worker, worker_run):
        duplicate_completions[key].add("%s:%s" % (previous_worker, previous_run))
        duplicate_completions[key].add("%s:%s" % (worker, worker_run))


def load_manifest(manifest_path, model, assignments, duplicate_assignments, warnings):
    if manifest_path is None:
        return None
    if not manifest_path.is_file():
        warnings.append("Manifest file not found: %s" % manifest_path)
        return None

    selected = 0
    bad_rows = 0
    for row, error in iter_jsonl(manifest_path):
        if error:
            bad_rows += 1
            warnings.append(error)
            continue
        if row.get("model") != model:
            continue
        key, meta = assignment_meta(row, row.get("worker_slot_id"))
        if key is None:
            bad_rows += 1
            continue
        merge_assignment(assignments, duplicate_assignments, key, meta, "manifest")
        selected += 1
    return {
        "path": str(manifest_path),
        "selected_rows": selected,
        "bad_rows": bad_rows,
    }


def load_worker_trace(
    worker_dir,
    model,
    assignments,
    completed,
    duplicate_assignments,
    duplicate_completions,
    warnings,
):
    worker_id = worker_dir.name
    worker_info = {
        "worker_slot_id": worker_id,
        "trace_dir": str(worker_dir),
        "has_assignment_json": False,
        "has_state_json": False,
        "has_state_jsonl": False,
        "has_events_jsonl": False,
        "snapshot_done": None,
        "snapshot_pending": None,
        "bad_jsonl_rows": 0,
    }

    _load_assignment_json(
        worker_dir,
        model,
        worker_id,
        assignments,
        duplicate_assignments,
        warnings,
        worker_info,
    )
    _load_state_json(
        worker_dir,
        model,
        worker_id,
        assignments,
        completed,
        duplicate_assignments,
        duplicate_completions,
        warnings,
        worker_info,
    )
    _load_completion_jsonl(
        worker_dir,
        model,
        worker_id,
        assignments,
        completed,
        duplicate_assignments,
        duplicate_completions,
        warnings,
        worker_info,
    )
    return worker_info


def _load_assignment_json(
    worker_dir,
    model,
    worker_id,
    assignments,
    duplicate_assignments,
    warnings,
    worker_info,
):
    assignment_path = worker_dir / "assignment.json"
    if not assignment_path.is_file():
        return
    worker_info["has_assignment_json"] = True
    try:
        payload = read_json(assignment_path)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append("Could not read %s: %s" % (assignment_path, exc))
        return
    for row in payload.get("assignments", []):
        if row.get("model") != model:
            continue
        key, meta = assignment_meta(row, worker_id)
        if key is not None:
            merge_assignment(
                assignments,
                duplicate_assignments,
                key,
                meta,
                "assignment.json",
            )


def _load_state_json(
    worker_dir,
    model,
    worker_id,
    assignments,
    completed,
    duplicate_assignments,
    duplicate_completions,
    warnings,
    worker_info,
):
    state_path = worker_dir / "state.json"
    if not state_path.is_file():
        return
    worker_info["has_state_json"] = True
    try:
        payload = read_json(state_path)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append("Could not read %s: %s" % (state_path, exc))
        return

    counts = payload.get("counts", {})
    worker_info["snapshot_done"] = safe_int(counts.get("done"))
    worker_info["snapshot_pending"] = safe_int(counts.get("pending"))
    for raw_key, row in payload.get("shards", {}).items():
        key = parse_state_key(raw_key)
        if key is None or key[0] != model:
            continue
        meta = state_meta(key, row, worker_id)
        merge_assignment(
            assignments,
            duplicate_assignments,
            key,
            meta,
            "state.json",
        )
        if row.get("status") == DONE:
            done = {
                "worker_slot_id": meta["worker_slot_id"],
                "worker_run_id": row.get("worker_run_id"),
                "finished_at": row.get("finished_at"),
                "out_path": row.get("out_path"),
                "gpu_count": row.get("gpu_count"),
                "gpu_seconds_delta": row.get("gpu_seconds_total"),
            }
            merge_done(completed, duplicate_completions, key, done, "state.json")


def _load_completion_jsonl(
    worker_dir,
    model,
    worker_id,
    assignments,
    completed,
    duplicate_assignments,
    duplicate_completions,
    warnings,
    worker_info,
):
    for name in ("state.jsonl", "events.jsonl"):
        path = worker_dir / name
        if not path.is_file():
            continue
        if name == "state.jsonl":
            worker_info["has_state_jsonl"] = True
        else:
            worker_info["has_events_jsonl"] = True
        for row, error in iter_jsonl(path):
            if error:
                worker_info["bad_jsonl_rows"] += 1
                warnings.append(error)
                continue
            if row.get("model") != model or row.get("event") != DONE:
                continue
            key, done = done_meta(row, worker_id)
            if key is None:
                continue
            key2, meta = assignment_meta(row, worker_id)
            if key2 is not None:
                merge_assignment(
                    assignments,
                    duplicate_assignments,
                    key2,
                    meta,
                    name,
                )
            merge_done(completed, duplicate_completions, key, done, name)


def load_manifest_summary(summary_path, model, warnings):
    if summary_path is None:
        return None
    if not summary_path.is_file():
        warnings.append("Manifest summary not found: %s" % summary_path)
        return None
    try:
        payload = read_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append("Could not read %s: %s" % (summary_path, exc))
        return None
    model_summary = payload.get("models", {}).get(model)
    if model_summary is None:
        warnings.append("Model %s is not present in %s" % (model, summary_path))
        return {
            "path": str(summary_path),
            "build_tag": payload.get("build_tag"),
            "model": None,
        }
    return {
        "path": str(summary_path),
        "build_tag": payload.get("build_tag"),
        "model": model_summary,
    }


def load_progress(model, trace_root, build_tag, manifest_path, manifest_summary_path):
    warnings = []
    assignments = {}
    completed = {}
    duplicate_assignments = defaultdict(set)
    duplicate_completions = defaultdict(set)

    manifest_info = load_manifest(
        manifest_path,
        model,
        assignments,
        duplicate_assignments,
        warnings,
    )

    worker_dirs = find_worker_dirs(trace_root, model, build_tag)
    worker_infos = []
    for worker_dir in worker_dirs:
        worker_infos.append(
            load_worker_trace(
                worker_dir,
                model,
                assignments,
                completed,
                duplicate_assignments,
                duplicate_completions,
                warnings,
            )
        )

    summary_info = load_manifest_summary(manifest_summary_path, model, warnings)
    return {
        "assignments": assignments,
        "completed": completed,
        "duplicate_assignments": duplicate_assignments,
        "duplicate_completions": duplicate_completions,
        "manifest_info": manifest_info,
        "summary_info": summary_info,
        "worker_infos": worker_infos,
        "warnings": warnings,
    }
