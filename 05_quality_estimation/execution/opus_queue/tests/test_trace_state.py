from __future__ import annotations

from execution.opus_queue.manifest.reader import ManifestRow
from execution.opus_queue.trace.state import TraceState


def _row(direction_key: str, shard_id: int) -> ManifestRow:
    src_lang, tgt_lang = direction_key.split("-", 1)
    return ManifestRow(
        worker_slot_id="metricx24-a00000-l0",
        model="metricx24",
        array_task_id=0,
        local_id=0,
        assignment_seq=shard_id,
        direction_key=direction_key,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        shard_id=shard_id,
        start_idx=shard_id * 100,
        end_idx=(shard_id + 1) * 100,
        shard_size=100,
        expected_seconds=600,
    )


def _event(row: ManifestRow, event: str, delta: float = 0.0) -> dict:
    return {
        "event": event,
        "model": row.model,
        "direction_key": row.direction_key,
        "shard_id": row.shard_id,
        "finished_at": 20 if event in {"done", "failed"} else None,
        "gpu_count": 2,
        "gpu_seconds_delta": delta,
        "out_path": "out.jsonl" if event == "done" else None,
        "worker_run_id": "run",
    }


def test_trace_state_ignores_non_completion_events() -> None:
    row = _row("eng_Latn-fra_Latn", 0)
    state = TraceState()
    state.apply_event(_event(row, "claim"))
    state.apply_event(_event(row, "failed", delta=3.5))

    progress = state.get(row)
    assert progress.status == "pending"
    assert progress.gpu_seconds_total == 0.0
    assert state.can_process(row)

    state.apply_event(_event(row, "done", delta=4.5))
    progress = state.get(row)
    assert progress.status == "done"
    assert progress.gpu_seconds_total == 4.5
    assert progress.out_path == "out.jsonl"
    assert not state.can_process(row)


def test_trace_state_keys_by_direction_and_shard() -> None:
    left = _row("eng_Latn-fra_Latn", 0)
    right = _row("eng_Latn-deu_Latn", 0)
    state = TraceState()
    state.apply_event(_event(left, "done", delta=1.0))
    state.apply_event(_event(right, "done", delta=2.0))

    assert state.get(left).status == "done"
    assert state.get(right).status == "done"
    assert state.get(left).gpu_seconds_total == 1.0
    assert state.get(right).gpu_seconds_total == 2.0


def run_test() -> None:
    test_trace_state_ignores_non_completion_events()
    test_trace_state_keys_by_direction_and_shard()
    print("OK: trace state replays completion-only direction-local keys.")


if __name__ == "__main__":
    run_test()
