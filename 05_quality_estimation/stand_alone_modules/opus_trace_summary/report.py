import json


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def print_human(result, workers_limit, directions_limit):
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

    _print_workers_section(totals)
    _print_shards_section(totals)
    _print_directions_section(totals)
    _print_worker_rows(result, workers_limit)
    _print_direction_rows(result, directions_limit)
    _print_warnings(result)


def _print_workers_section(totals):
    print("")
    print("workers")
    print("  trace dirs found: %d" % totals["trace_dirs_found"])
    print("  worker slots seen: %d" % totals["worker_slots_seen"])
    print("  workers with done shards: %d" % totals["workers_with_done_shards"])
    print("  workers complete: %d" % totals["workers_complete"])
    print("  workers with remaining shards: %d" % totals["workers_with_remaining_shards"])


def _print_shards_section(totals):
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
        print("  done shards without an assignment row: %d" % totals["done_only_shards"])
    if totals["duplicate_assignment_shards"]:
        print("  duplicate assignment shards: %d" % totals["duplicate_assignment_shards"])
    if totals["duplicate_completion_shards"]:
        print("  duplicate completion shards: %d" % totals["duplicate_completion_shards"])


def _print_directions_section(totals):
    print("")
    print("directions")
    print("  directions seen: %d" % totals["directions_seen"])
    print("  directions with done shards: %d" % totals["directions_with_done_shards"])
    print("  directions complete: %d" % totals["directions_complete"])
    print("  directions with remaining shards: %d" % totals["directions_with_remaining_shards"])
    if (
        "manifest_directions_total" in totals
        and totals["manifest_directions_total"] is not None
    ):
        print("  manifest directions total: %d" % totals["manifest_directions_total"])


def _print_worker_rows(result, workers_limit):
    if not workers_limit:
        return
    unfinished = result.get("unfinished_workers")
    if unfinished is None:
        unfinished = [row for row in result["workers"] if row["remaining_shards"]]
    if workers_limit > 0:
        rows = unfinished[:workers_limit]
    else:
        rows = unfinished
    print("")
    if workers_limit > 0 and len(unfinished) > workers_limit:
        print(
            "unfinished workers (top %d of %d by remaining shards)"
            % (workers_limit, len(unfinished))
        )
    else:
        print("unfinished workers")
    if not rows:
        print("  none")
        return
    print("  worker_slot_id array local assigned done remaining dirs done_dirs trace")
    for row in rows:
        print(
            "  {worker_slot_id} {array_task_id} {local_id} "
            "{assigned_shards} {done_shards} "
            "{remaining_shards} {directions} {done_directions} {trace_present}".format(
                worker_slot_id=row["worker_slot_id"],
                array_task_id=_format_optional(row.get("array_task_id")),
                local_id=_format_optional(row.get("local_id")),
                assigned_shards=row["assigned_shards"],
                done_shards=row["done_shards"],
                remaining_shards=row["remaining_shards"],
                directions=row["directions"],
                done_directions=row["done_directions"],
                trace_present=_format_bool(row["trace_present"]),
            )
        )


def _print_direction_rows(result, directions_limit):
    if not directions_limit:
        return
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


def _print_warnings(result):
    if not result["warnings"]:
        return
    print("")
    print("warnings")
    for warning in result["warnings"]:
        print("  %s" % warning)


def _format_bool(value):
    return "yes" if value else "no"


def _format_optional(value):
    return "-" if value is None else str(value)
