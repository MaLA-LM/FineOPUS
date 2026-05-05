"""Summarize OPUS static-manifest shard trace progress.

This module intentionally uses only the Python standard library so it can be
copied or run outside the main project package.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DONE = "done"


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), None
            except json.JSONDecodeError as exc:
                yield None, "%s:%d: %s" % (path, line_no, exc)


def _parse_shard_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _make_key(model, direction_key, shard_id):
    shard_id = _parse_shard_id(shard_id)
    if not model or not direction_key or shard_id is None:
        return None
    return (str(model), str(direction_key), shard_id)


def _parse_state_key(value):
    parts = str(value).split("/")
    if len(parts) != 3:
        return None
    return _make_key(parts[0], parts[1], parts[2])


def _key_to_string(key):
    return "%s/%s/%s" % (key[0], key[1], key[2])


def _pct(part, whole):
    if not whole:
        return 0.0
    return round(100.0 * float(part) / float(whole), 4)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coalesce_worker(row, fallback):
    return str(row.get("worker_slot_id") or fallback or "unknown-worker")


def _assignment_meta(row, fallback_worker):
    key = _make_key(row.get("model"), row.get("direction_key"), row.get("shard_id"))
    if key is None:
        return None, None
    worker = _coalesce_worker(row, fallback_worker)
    return key, {
        "model": key[0],
        "direction_key": key[1],
        "shard_id": key[2],
        "worker_slot_id": worker,
        "array_task_id": _safe_int(row.get("array_task_id")),
        "local_id": _safe_int(row.get("local_id")),
        "assignment_seq": _safe_int(row.get("assignment_seq")),
        "expected_seconds": _safe_int(row.get("expected_seconds")),
        "start_idx": _safe_int(row.get("start_idx")),
        "end_idx": _safe_int(row.get("end_idx")),
    }


def _state_meta(key, row, fallback_worker):
    worker = _coalesce_worker(row, fallback_worker)
    return {
        "model": key[0],
        "direction_key": key[1],
        "shard_id": key[2],
        "worker_slot_id": worker,
        "array_task_id": None,
        "local_id": None,
        "assignment_seq": _safe_int(row.get("assignment_seq")),
        "expected_seconds": None,
        "start_idx": None,
        "end_idx": None,
    }


def _done_meta(row, fallback_worker):
    key = _make_key(row.get("model"), row.get("direction_key"), row.get("shard_id"))
    if key is None:
        return None, None
    worker = _coalesce_worker(row, fallback_worker)
    return key, {
        "worker_slot_id": worker,
        "worker_run_id": row.get("worker_run_id"),
        "finished_at": row.get("finished_at") or row.get("ts"),
        "out_path": row.get("out_path"),
        "gpu_count": row.get("gpu_count"),
        "gpu_seconds_delta": row.get("gpu_seconds_delta"),
        "source": None,
    }


def _scan_worker_dirs_in_bases(bases, prefix):
    found = {}
    for base in bases:
        if base.is_dir() and base.name.startswith(prefix):
            found[str(base.resolve())] = base
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                found[str(child.resolve())] = child
    return found


def _find_worker_dirs(trace_root, model, build_tag):
    root = Path(trace_root).expanduser()
    prefix = "%s-a" % model
    bases = []
    if build_tag:
        tagged = root / build_tag
        if tagged.exists():
            bases.append(tagged)
        else:
            bases.append(root)
    else:
        bases.append(root)

    found = _scan_worker_dirs_in_bases(bases, prefix)

    if not found and build_tag and root.is_dir():
        found.update(_scan_worker_dirs_in_bases([root], prefix))

    if not found and not build_tag and root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith(prefix):
                continue
            for grandchild in child.iterdir():
                if grandchild.is_dir() and grandchild.name.startswith(prefix):
                    found[str(grandchild.resolve())] = grandchild

    return [found[key] for key in sorted(found)]


def _resolve_manifest_path(args):
    if args.manifest:
        return Path(args.manifest).expanduser()
    if args.manifest_root and args.build_tag:
        return Path(args.manifest_root).expanduser() / args.build_tag / "manifest.jsonl"
    return None


def _resolve_summary_path(args, manifest_path):
    if args.manifest_summary:
        return Path(args.manifest_summary).expanduser()
    candidates = []
    if manifest_path is not None:
        candidates.append(manifest_path.parent / "manifest.summary.json")
    if args.manifest_root and args.build_tag:
        candidates.append(
            Path(args.manifest_root).expanduser()
            / args.build_tag
            / "manifest.summary.json"
        )
    if args.build_tag:
        candidates.append(Path(args.build_tag).expanduser() / "manifest.summary.json")
        candidates.append(
            Path(args.trace_root).expanduser()
            / args.build_tag
            / "manifest.summary.json"
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _merge_assignment(assignments, duplicate_assignments, key, meta, source):
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


def _merge_done(completed, duplicate_completions, key, meta, source):
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


def _load_manifest(manifest_path, model, assignments, duplicate_assignments, warnings):
    if manifest_path is None:
        return None
    if not manifest_path.is_file():
        warnings.append("Manifest file not found: %s" % manifest_path)
        return None

    selected = 0
    bad_rows = 0
    for row, error in _iter_jsonl(manifest_path):
        if error:
            bad_rows += 1
            warnings.append(error)
            continue
        if row.get("model") != model:
            continue
        key, meta = _assignment_meta(row, row.get("worker_slot_id"))
        if key is None:
            bad_rows += 1
            continue
        _merge_assignment(assignments, duplicate_assignments, key, meta, "manifest")
        selected += 1
    return {
        "path": str(manifest_path),
        "selected_rows": selected,
        "bad_rows": bad_rows,
    }


def _load_worker_trace(
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

    assignment_path = worker_dir / "assignment.json"
    if assignment_path.is_file():
        worker_info["has_assignment_json"] = True
        try:
            payload = _read_json(assignment_path)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append("Could not read %s: %s" % (assignment_path, exc))
        else:
            for row in payload.get("assignments", []):
                if row.get("model") != model:
                    continue
                key, meta = _assignment_meta(row, worker_id)
                if key is not None:
                    _merge_assignment(
                        assignments,
                        duplicate_assignments,
                        key,
                        meta,
                        "assignment.json",
                    )

    state_path = worker_dir / "state.json"
    if state_path.is_file():
        worker_info["has_state_json"] = True
        try:
            payload = _read_json(state_path)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append("Could not read %s: %s" % (state_path, exc))
        else:
            counts = payload.get("counts", {})
            worker_info["snapshot_done"] = _safe_int(counts.get("done"))
            worker_info["snapshot_pending"] = _safe_int(counts.get("pending"))
            for raw_key, row in payload.get("shards", {}).items():
                key = _parse_state_key(raw_key)
                if key is None or key[0] != model:
                    continue
                meta = _state_meta(key, row, worker_id)
                _merge_assignment(
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
                    _merge_done(
                        completed,
                        duplicate_completions,
                        key,
                        done,
                        "state.json",
                    )

    for name in ("state.jsonl", "events.jsonl"):
        path = worker_dir / name
        if not path.is_file():
            continue
        if name == "state.jsonl":
            worker_info["has_state_jsonl"] = True
        else:
            worker_info["has_events_jsonl"] = True
        for row, error in _iter_jsonl(path):
            if error:
                worker_info["bad_jsonl_rows"] += 1
                warnings.append(error)
                continue
            if row.get("model") != model or row.get("event") != DONE:
                continue
            key, done = _done_meta(row, worker_id)
            if key is None:
                continue
            key2, meta = _assignment_meta(row, worker_id)
            if key2 is not None:
                _merge_assignment(
                    assignments,
                    duplicate_assignments,
                    key2,
                    meta,
                    name,
                )
            _merge_done(completed, duplicate_completions, key, done, name)

    return worker_info


def _load_manifest_summary(summary_path, model, warnings):
    if summary_path is None:
        return None
    if not summary_path.is_file():
        warnings.append("Manifest summary not found: %s" % summary_path)
        return None
    try:
        payload = _read_json(summary_path)
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


def _build_summary(
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

    for key in done_only_keys:
        done = completed[key]
        assignments[key] = {
            "model": key[0],
            "direction_key": key[1],
            "shard_id": key[2],
            "worker_slot_id": done.get("worker_slot_id") or "unknown-worker",
            "array_task_id": None,
            "local_id": None,
            "assignment_seq": None,
            "expected_seconds": None,
            "start_idx": None,
            "end_idx": None,
            "assignment_source": "done-only",
        }

    assigned_keys = set(assignments)
    completed_keys = set(key for key in completed if key[0] == model)
    completed_assigned_keys = assigned_keys & completed_keys
    remaining_keys = assigned_keys - completed_keys

    direction_to_assigned = defaultdict(set)
    direction_to_done = defaultdict(set)
    direction_to_workers = defaultdict(set)
    worker_to_assigned = defaultdict(set)
    worker_to_done = defaultdict(set)
    trace_workers = set(info["worker_slot_id"] for info in worker_infos)

    for key, meta in assignments.items():
        direction = key[1]
        worker = meta.get("worker_slot_id") or "unknown-worker"
        direction_to_assigned[direction].add(key)
        direction_to_workers[direction].add(worker)
        worker_to_assigned[worker].add(key)

    for key, done in completed.items():
        if key[0] != model:
            continue
        direction_to_done[key[1]].add(key)
        worker = done.get("worker_slot_id")
        if not worker and key in assignments:
            worker = assignments[key].get("worker_slot_id")
        worker_to_done[worker or "unknown-worker"].add(key)

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

    worker_rows = []
    complete_workers = 0
    workers_with_done = 0
    workers_with_remaining = 0
    all_workers = set(worker_to_assigned) | set(worker_to_done) | trace_workers
    for worker in sorted(all_workers):
        assigned = worker_to_assigned.get(worker, set())
        done = assigned & completed_keys
        remaining = assigned - completed_keys
        assigned_dirs = defaultdict(set)
        for key in assigned:
            assigned_dirs[key[1]].add(key)
        done_dirs = 0
        for keys in assigned_dirs.values():
            if keys and not (keys - completed_keys):
                done_dirs += 1
        if done:
            workers_with_done += 1
        if assigned and not remaining:
            complete_workers += 1
        if remaining:
            workers_with_remaining += 1
        worker_rows.append(
            {
                "worker_slot_id": worker,
                "trace_present": worker in trace_workers,
                "assigned_shards": len(assigned),
                "done_shards": len(done),
                "remaining_shards": len(remaining),
                "directions": len(assigned_dirs),
                "done_directions": done_dirs,
                "remaining_directions": len(assigned_dirs) - done_dirs,
            }
        )

    manifest_model = None
    if summary_info and summary_info.get("model"):
        manifest_model = summary_info["model"]

    manifest_assigned = None
    manifest_directions = None
    manifest_slots = None
    if manifest_model:
        manifest_assigned = _safe_int(manifest_model.get("assigned_shards"))
        manifest_directions = _safe_int(manifest_model.get("directions_total"))
        manifest_slots = _safe_int(manifest_model.get("slots_declared"))

    selected_manifest_rows = None
    if manifest_info:
        selected_manifest_rows = manifest_info.get("selected_rows")

    completion_base = len(assigned_keys)
    if manifest_assigned is not None and selected_manifest_rows is None:
        completion_base = manifest_assigned
    elif manifest_assigned is not None and selected_manifest_rows is not None:
        completion_base = manifest_assigned

    totals = {
        "model": model,
        "trace_root": str(Path(trace_root).expanduser()),
        "build_tag": build_tag,
        "assignment_scope": (
            "manifest" if selected_manifest_rows is not None else "trace-folders"
        ),
        "trace_dirs_found": len(worker_infos),
        "worker_slots_seen": len(all_workers),
        "workers_with_done_shards": workers_with_done,
        "workers_complete": complete_workers,
        "workers_with_remaining_shards": workers_with_remaining,
        "assigned_shards_seen": len(assigned_keys),
        "done_shards": len(completed_assigned_keys),
        "done_only_shards": len(done_only_keys),
        "remaining_shards_seen": len(remaining_keys),
        "completion_pct_seen": _pct(len(completed_assigned_keys), len(assigned_keys)),
        "directions_seen": len(direction_to_assigned),
        "directions_with_done_shards": directions_with_done,
        "directions_complete": complete_directions,
        "directions_with_remaining_shards": len(direction_to_assigned)
        - complete_directions,
        "duplicate_assignment_shards": len(duplicate_assignments),
        "duplicate_completion_shards": len(duplicate_completions),
    }

    if manifest_assigned is not None:
        totals.update(
            {
                "manifest_assigned_shards": manifest_assigned,
                "manifest_directions_total": manifest_directions,
                "manifest_slots_declared": manifest_slots,
                "manifest_remaining_shards": max(
                    0, manifest_assigned - len(completed_assigned_keys)
                ),
                "completion_pct_manifest": _pct(
                    len(completed_assigned_keys), manifest_assigned
                ),
                "unseen_assigned_shards_vs_summary": max(
                    0, manifest_assigned - len(assigned_keys)
                ),
            }
        )

    directions.sort(
        key=lambda row: (
            -row["remaining_shards"],
            row["direction_key"],
        )
    )
    worker_rows.sort(
        key=lambda row: (
            -row["remaining_shards"],
            row["worker_slot_id"],
        )
    )

    return {
        "totals": totals,
        "manifest": manifest_info,
        "manifest_summary": summary_info,
        "workers": worker_rows,
        "directions": directions,
        "trace_files": worker_infos,
        "warnings": [],
        "duplicate_assignments": {
            _key_to_string(key): sorted(values)
            for key, values in sorted(duplicate_assignments.items())
        },
        "duplicate_completions": {
            _key_to_string(key): sorted(values)
            for key, values in sorted(duplicate_completions.items())
        },
    }


def summarize(
    model, trace_root, build_tag=None, manifest_path=None, manifest_summary_path=None
):
    warnings = []
    assignments = {}
    completed = {}
    duplicate_assignments = defaultdict(set)
    duplicate_completions = defaultdict(set)

    manifest_info = _load_manifest(
        manifest_path,
        model,
        assignments,
        duplicate_assignments,
        warnings,
    )

    worker_dirs = _find_worker_dirs(trace_root, model, build_tag)
    worker_infos = []
    for worker_dir in worker_dirs:
        worker_infos.append(
            _load_worker_trace(
                worker_dir,
                model,
                assignments,
                completed,
                duplicate_assignments,
                duplicate_completions,
                warnings,
            )
        )

    summary_info = _load_manifest_summary(manifest_summary_path, model, warnings)

    result = _build_summary(
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
    )
    result["warnings"] = warnings
    return result


def _format_bool(value):
    return "yes" if value else "no"


def _print_human(result, workers_limit, directions_limit):
    totals = result["totals"]
    manifest = result.get("manifest")
    manifest_summary = result.get("manifest_summary")

    print("OPUS trace summary")
    print("model: %s" % totals["model"])
    print("trace_root: %s" % totals["trace_root"])
    if totals.get("build_tag"):
        print("build_tag: %s" % totals["build_tag"])
    print("assignment_scope: %s" % totals["assignment_scope"])
    if manifest:
        print(
            "manifest: %s (%s rows for model)"
            % (manifest["path"], manifest["selected_rows"])
        )
    if manifest_summary:
        print("manifest_summary: %s" % manifest_summary["path"])

    print("")
    print("workers")
    print("  trace dirs found: %d" % totals["trace_dirs_found"])
    print("  worker slots seen: %d" % totals["worker_slots_seen"])
    print("  workers with done shards: %d" % totals["workers_with_done_shards"])
    print("  workers complete: %d" % totals["workers_complete"])
    print(
        "  workers with remaining shards: %d" % totals["workers_with_remaining_shards"]
    )

    print("")
    print("shards")
    print("  assigned shards seen: %d" % totals["assigned_shards_seen"])
    print("  done shards: %d" % totals["done_shards"])
    print("  remaining shards seen: %d" % totals["remaining_shards_seen"])
    print("  completion seen: %.4f%%" % totals["completion_pct_seen"])
    if "manifest_assigned_shards" in totals:
        print("  manifest assigned shards: %d" % totals["manifest_assigned_shards"])
        print("  manifest remaining shards: %d" % totals["manifest_remaining_shards"])
        print("  completion vs manifest: %.4f%%" % totals["completion_pct_manifest"])
        print(
            "  assigned shards not present in scanned traces: %d"
            % totals["unseen_assigned_shards_vs_summary"]
        )
    if totals["done_only_shards"]:
        print(
            "  done shards without an assignment row: %d" % totals["done_only_shards"]
        )
    if totals["duplicate_assignment_shards"]:
        print(
            "  duplicate assignment shards: %d" % totals["duplicate_assignment_shards"]
        )
    if totals["duplicate_completion_shards"]:
        print(
            "  duplicate completion shards: %d" % totals["duplicate_completion_shards"]
        )

    print("")
    print("directions")
    print("  directions seen: %d" % totals["directions_seen"])
    print("  directions with done shards: %d" % totals["directions_with_done_shards"])
    print("  directions complete: %d" % totals["directions_complete"])
    print(
        "  directions with remaining shards: %d"
        % totals["directions_with_remaining_shards"]
    )
    if (
        "manifest_directions_total" in totals
        and totals["manifest_directions_total"] is not None
    ):
        print("  manifest directions total: %d" % totals["manifest_directions_total"])

    if workers_limit:
        print("")
        print("workers with most remaining shards")
        print("  worker_slot_id assigned done remaining dirs done_dirs trace")
        for row in result["workers"][:workers_limit]:
            print(
                "  {worker_slot_id} {assigned_shards} {done_shards} "
                "{remaining_shards} {directions} {done_directions} {trace_present}".format(
                    worker_slot_id=row["worker_slot_id"],
                    assigned_shards=row["assigned_shards"],
                    done_shards=row["done_shards"],
                    remaining_shards=row["remaining_shards"],
                    directions=row["directions"],
                    done_directions=row["done_directions"],
                    trace_present=_format_bool(row["trace_present"]),
                )
            )

    if directions_limit:
        print("")
        print("directions with most remaining shards")
        print("  direction_key assigned done remaining workers")
        for row in result["directions"][:directions_limit]:
            print(
                "  {direction_key} {assigned_shards} {done_shards} "
                "{remaining_shards} {workers}".format(
                    direction_key=row["direction_key"],
                    assigned_shards=row["assigned_shards"],
                    done_shards=row["done_shards"],
                    remaining_shards=row["remaining_shards"],
                    workers=row["workers"],
                )
            )

    if result["warnings"]:
        print("")
        print("warnings")
        for warning in result["warnings"]:
            print("  %s" % warning)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize OPUS worker trace progress for one static-manifest model."
        )
    )
    parser.add_argument(
        "model_positional",
        nargs="?",
        help="Model key, for example metricx24.",
    )
    parser.add_argument(
        "--model",
        dest="model_option",
        help="Model key. Overrides the positional model when both are passed.",
    )
    parser.add_argument(
        "--trace-root",
        default=".",
        help=(
            "Trace root. This can be the directory containing worker folders, "
            "or the parent directory when --build-tag is passed. Default: ."
        ),
    )
    parser.add_argument(
        "--build-tag",
        default=None,
        help="Optional build tag subdirectory under --trace-root.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional manifest.jsonl path. When passed, remaining shards are "
            "computed against the full manifest for the model."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        default=None,
        help=(
            "Optional root containing <build-tag>/manifest.jsonl. Used when "
            "--manifest is not passed."
        ),
    )
    parser.add_argument(
        "--manifest-summary",
        default=None,
        help=(
            "Optional manifest.summary.json path. Used to compare scanned "
            "trace folders against the manifest totals."
        ),
    )
    parser.add_argument(
        "--workers-limit",
        type=int,
        default=20,
        help="Number of worker rows to print in human output. Use 0 to hide.",
    )
    parser.add_argument(
        "--directions-limit",
        type=int,
        default=20,
        help="Number of direction rows to print in human output. Use 0 to hide.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full JSON summary.",
    )
    args = parser.parse_args(argv)
    args.model = args.model_option or args.model_positional
    if not args.model:
        parser.error("provide a model name either positionally or with --model")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = _resolve_manifest_path(args)
    summary_path = _resolve_summary_path(args, manifest_path)
    result = summarize(
        args.model,
        args.trace_root,
        build_tag=args.build_tag,
        manifest_path=manifest_path,
        manifest_summary_path=summary_path,
    )
    if args.output_json:
        _write_json(Path(args.output_json).expanduser(), result)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_human(result, args.workers_limit, args.directions_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
