from stand_alone_modules.opus_trace_summary.constants import UNKNOWN_WORKER
from stand_alone_modules.opus_trace_summary.keys import make_key, safe_int


def coalesce_worker(row, fallback):
    return str(row.get("worker_slot_id") or fallback or UNKNOWN_WORKER)


def assignment_meta(row, fallback_worker):
    key = make_key(row.get("model"), row.get("direction_key"), row.get("shard_id"))
    if key is None:
        return None, None
    worker = coalesce_worker(row, fallback_worker)
    return key, {
        "model": key[0],
        "direction_key": key[1],
        "shard_id": key[2],
        "worker_slot_id": worker,
        "array_task_id": safe_int(row.get("array_task_id")),
        "local_id": safe_int(row.get("local_id")),
        "assignment_seq": safe_int(row.get("assignment_seq")),
        "expected_seconds": safe_int(row.get("expected_seconds")),
        "start_idx": safe_int(row.get("start_idx")),
        "end_idx": safe_int(row.get("end_idx")),
    }


def state_meta(key, row, fallback_worker):
    worker = coalesce_worker(row, fallback_worker)
    return {
        "model": key[0],
        "direction_key": key[1],
        "shard_id": key[2],
        "worker_slot_id": worker,
        "array_task_id": None,
        "local_id": None,
        "assignment_seq": safe_int(row.get("assignment_seq")),
        "expected_seconds": None,
        "start_idx": None,
        "end_idx": None,
    }


def done_meta(row, fallback_worker):
    key = make_key(row.get("model"), row.get("direction_key"), row.get("shard_id"))
    if key is None:
        return None, None
    worker = coalesce_worker(row, fallback_worker)
    return key, {
        "worker_slot_id": worker,
        "worker_run_id": row.get("worker_run_id"),
        "finished_at": row.get("finished_at") or row.get("ts"),
        "out_path": row.get("out_path"),
        "gpu_count": row.get("gpu_count"),
        "gpu_seconds_delta": row.get("gpu_seconds_delta"),
        "source": None,
    }
