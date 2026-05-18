from collections import defaultdict
from pathlib import Path

from stand_alone_modules.opus_trace_summary.constants import UNKNOWN_WORKER
from stand_alone_modules.opus_trace_summary.keys import key_to_string, pct, safe_int
from stand_alone_modules.opus_trace_summary.loader import load_progress


def summarize(
    model,
    trace_root,
    build_tag=None,
    manifest_path=None,
    manifest_summary_path=None,
):
    progress = load_progress(
        model,
        trace_root,
        build_tag,
        manifest_path,
        manifest_summary_path,
    )
    result = build_summary(
        model,
        trace_root,
        build_tag,
        progress["manifest_info"],
        progress["summary_info"],
        progress["worker_infos"],
        progress["assignments"],
        progress["completed"],
        progress["duplicate_assignments"],
        progress["duplicate_completions"],
    )
    result["warnings"] = progress["warnings"]
    return result


def build_summary(
    model,
    trace_root,
    build_tag,
    manifest_info,
    summary_info,
    worker_infos,
    assignments,
    completed,
    duplicate_assignments,
    duplicate_completions,
):
    assigned_keys = set(assignments)
    completed_keys = set(key for key in completed if key[0] == model)
    done_only_keys = completed_keys - assigned_keys

    _add_done_only_assignments(assignments, completed, done_only_keys)

    assigned_keys = set(assignments)
    completed_keys = set(key for key in completed if key[0] == model)
    completed_assigned_keys = assigned_keys & completed_keys
    remaining_keys = assigned_keys - completed_keys

    direction_stats = _direction_stats(assignments, completed_keys)
    worker_stats = _worker_stats(assignments, completed, completed_keys, worker_infos)
    manifest_totals = _manifest_totals(summary_info)

    totals = {
        "model": model,
        "trace_root": str(Path(trace_root).expanduser()),
        "build_tag": build_tag,
        "assignment_scope": "manifest" if manifest_info is not None else "trace-folders",
        "trace_dirs_found": len(worker_infos),
        "worker_slots_seen": len(worker_stats["all_workers"]),
        "workers_with_done_shards": worker_stats["workers_with_done"],
        "workers_complete": worker_stats["complete_workers"],
        "workers_with_remaining_shards": worker_stats["workers_with_remaining"],
        "assigned_shards_seen": len(assigned_keys),
        "done_shards": len(completed_assigned_keys),
        "done_only_shards": len(done_only_keys),
        "remaining_shards_seen": len(remaining_keys),
        "completion_pct_seen": pct(len(completed_assigned_keys), len(assigned_keys)),
        "directions_seen": len(direction_stats["direction_to_assigned"]),
        "directions_with_done_shards": direction_stats["directions_with_done"],
        "directions_complete": direction_stats["complete_directions"],
        "directions_with_remaining_shards": (
            len(direction_stats["direction_to_assigned"])
            - direction_stats["complete_directions"]
        ),
        "duplicate_assignment_shards": len(duplicate_assignments),
        "duplicate_completion_shards": len(duplicate_completions),
    }
    if manifest_totals["assigned_shards"] is not None:
        totals.update(
            _manifest_completion_totals(
                manifest_totals,
                assigned_keys,
                completed_assigned_keys,
            )
        )

    directions = sorted(
        direction_stats["directions"],
        key=lambda row: (-row["remaining_shards"], row["direction_key"]),
    )
    workers = sorted(
        worker_stats["workers"],
        key=lambda row: (-row["remaining_shards"], row["worker_slot_id"]),
    )
    unfinished_workers = [row for row in workers if row["remaining_shards"]]

    return {
        "totals": totals,
        "manifest": manifest_info,
        "manifest_summary": summary_info,
        "workers": workers,
        "unfinished_workers": unfinished_workers,
        "directions": directions,
        "trace_files": worker_infos,
        "warnings": [],
        "duplicate_assignments": {
            key_to_string(key): sorted(values)
            for key, values in sorted(duplicate_assignments.items())
        },
        "duplicate_completions": {
            key_to_string(key): sorted(values)
            for key, values in sorted(duplicate_completions.items())
        },
    }


def _add_done_only_assignments(assignments, completed, done_only_keys):
    for key in done_only_keys:
        done = completed[key]
        assignments[key] = {
            "model": key[0],
            "direction_key": key[1],
            "shard_id": key[2],
            "worker_slot_id": done.get("worker_slot_id") or UNKNOWN_WORKER,
            "array_task_id": None,
            "local_id": None,
            "assignment_seq": None,
            "expected_seconds": None,
            "start_idx": None,
            "end_idx": None,
            "assignment_source": "done-only",
        }


def _direction_stats(assignments, completed_keys):
    direction_to_assigned = defaultdict(set)
    direction_to_workers = defaultdict(set)
    for key, meta in assignments.items():
        direction = key[1]
        worker = meta.get("worker_slot_id") or UNKNOWN_WORKER
        direction_to_assigned[direction].add(key)
        direction_to_workers[direction].add(worker)

    directions = []
    complete_directions = 0
    directions_with_done = 0
    for direction in sorted(direction_to_assigned):
        assigned = direction_to_assigned[direction]
        done = assigned & completed_keys
        remaining = assigned - completed_keys
        if done:
            directions_with_done += 1
        if assigned and not remaining:
            complete_directions += 1
        directions.append(
            {
                "direction_key": direction,
                "assigned_shards": len(assigned),
                "done_shards": len(done),
                "remaining_shards": len(remaining),
                "workers": len(direction_to_workers[direction]),
            }
        )

    return {
        "direction_to_assigned": direction_to_assigned,
        "directions": directions,
        "complete_directions": complete_directions,
        "directions_with_done": directions_with_done,
    }


def _worker_stats(assignments, completed, completed_keys, worker_infos):
    worker_to_assigned = defaultdict(set)
    worker_to_done = defaultdict(set)
    worker_to_slot = {}
    trace_workers = set(info["worker_slot_id"] for info in worker_infos)

    for key, meta in assignments.items():
        worker = meta.get("worker_slot_id") or UNKNOWN_WORKER
        worker_to_assigned[worker].add(key)
        worker_to_slot.setdefault(
            worker,
            {
                "array_task_id": meta.get("array_task_id"),
                "local_id": meta.get("local_id"),
            },
        )

    for key, done in completed.items():
        worker = done.get("worker_slot_id")
        if not worker and key in assignments:
            worker = assignments[key].get("worker_slot_id")
        worker_to_done[worker or UNKNOWN_WORKER].add(key)

    workers = []
    complete_workers = 0
    workers_with_done = 0
    workers_with_remaining = 0
    all_workers = set(worker_to_assigned) | set(worker_to_done) | trace_workers
    for worker in sorted(all_workers):
        row = _worker_row(
            worker,
            worker_to_assigned,
            completed_keys,
            trace_workers,
            worker_to_slot,
        )
        if row["done_shards"]:
            workers_with_done += 1
        if row["assigned_shards"] and not row["remaining_shards"]:
            complete_workers += 1
        if row["remaining_shards"]:
            workers_with_remaining += 1
        workers.append(row)

    return {
        "all_workers": all_workers,
        "workers": workers,
        "complete_workers": complete_workers,
        "workers_with_done": workers_with_done,
        "workers_with_remaining": workers_with_remaining,
    }


def _worker_row(
    worker,
    worker_to_assigned,
    completed_keys,
    trace_workers,
    worker_to_slot,
):
    assigned = worker_to_assigned.get(worker, set())
    done = assigned & completed_keys
    remaining = assigned - completed_keys
    slot = worker_to_slot.get(worker, {})
    assigned_dirs = defaultdict(set)
    for key in assigned:
        assigned_dirs[key[1]].add(key)
    done_dirs = 0
    for keys in assigned_dirs.values():
        if keys and not (keys - completed_keys):
            done_dirs += 1
    return {
        "worker_slot_id": worker,
        "array_task_id": slot.get("array_task_id"),
        "local_id": slot.get("local_id"),
        "trace_present": worker in trace_workers,
        "assigned_shards": len(assigned),
        "done_shards": len(done),
        "remaining_shards": len(remaining),
        "directions": len(assigned_dirs),
        "done_directions": done_dirs,
        "remaining_directions": len(assigned_dirs) - done_dirs,
    }


def _manifest_totals(summary_info):
    if not summary_info or not summary_info.get("model"):
        return {
            "assigned_shards": None,
            "directions_total": None,
            "slots_declared": None,
        }
    model_summary = summary_info["model"]
    return {
        "assigned_shards": safe_int(model_summary.get("assigned_shards")),
        "directions_total": safe_int(model_summary.get("directions_total")),
        "slots_declared": safe_int(model_summary.get("slots_declared")),
    }


def _manifest_completion_totals(manifest_totals, assigned_keys, completed_assigned_keys):
    manifest_assigned = manifest_totals["assigned_shards"]
    return {
        "manifest_assigned_shards": manifest_assigned,
        "manifest_directions_total": manifest_totals["directions_total"],
        "manifest_slots_declared": manifest_totals["slots_declared"],
        "manifest_remaining_shards": max(
            0,
            manifest_assigned - len(completed_assigned_keys),
        ),
        "completion_pct_manifest": pct(len(completed_assigned_keys), manifest_assigned),
        "unseen_assigned_shards_vs_summary": max(
            0,
            manifest_assigned - len(assigned_keys),
        ),
    }
