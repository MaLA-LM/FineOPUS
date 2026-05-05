from __future__ import annotations

from execution.opus_queue.planning.assigner import PendingShard, assign_shards


def _shard(model: str, direction_key: str, shard_id: int) -> PendingShard:
    return PendingShard(
        model=model,
        direction_key=direction_key,
        src_lang=direction_key.split("-", 1)[0],
        tgt_lang=direction_key.split("-", 1)[1],
        shard_id=shard_id,
        start_idx=shard_id * 100,
        end_idx=(shard_id + 1) * 100,
        shard_size=100,
    )


def test_spills_large_direction_only_at_quota_boundary() -> None:
    rows = [_shard("metricx24", "eng_Latn-fra_Latn", shard_id) for shard_id in range(5)]
    assignments, summary = assign_shards(
        rows,
        slots_declared=3,
        slots_per_array_task=1,
        walltime_seconds=1_200,
        safety_factor=1.0,
    )

    assert summary.quota_shards_per_worker == 2
    assert summary.assigned_shards == 5
    assert summary.shortfall_shards == 0
    assert summary.directions_split_across_slots == 1
    assert [row.worker_slot_id for row in assignments] == [
        "metricx24-a00000-l0",
        "metricx24-a00000-l0",
        "metricx24-a00001-l0",
        "metricx24-a00001-l0",
        "metricx24-a00002-l0",
    ]
    assert [row.assignment_seq for row in assignments] == [0, 1, 0, 1, 0]


def test_standard_g_slot_ids_include_local_id() -> None:
    rows = [
        _shard("metricx24", f"eng_Latn-t{i:03d}_Latn", 0)
        for i in range(8)
    ]
    assignments, summary = assign_shards(
        rows,
        slots_declared=8,
        slots_per_array_task=8,
        walltime_seconds=144_000,
        safety_factor=0.85,
        sparse_round_robin_floor=2.0,
    )

    assert summary.slots_per_array_task == 8
    assert sorted(row.worker_slot_id for row in assignments) == [
        f"metricx24-a00000-l{i}" for i in range(8)
    ]
    assert {row.array_task_id for row in assignments} == {0}
    assert {row.local_id for row in assignments} == set(range(8))


def test_assigns_all_shards_beyond_one_wave_quota() -> None:
    rows = [
        _shard("metricx24", "eng_Latn-fra_Latn", shard_id)
        for shard_id in range(7)
    ]
    assignments, summary = assign_shards(
        rows,
        slots_declared=2,
        slots_per_array_task=1,
        walltime_seconds=1_200,
        safety_factor=1.0,
    )

    assert summary.quota_shards_per_worker == 2
    assert summary.pending_shards == 7
    assert summary.assigned_shards == 7
    assert summary.shortfall_shards == 0
    assert summary.estimated_waves == 2
    assert summary.usable_walltime_seconds == 1_200
    assert [(row.worker_slot_id, row.assignment_seq, row.shard_id) for row in assignments] == [
        ("metricx24-a00000-l0", 0, 0),
        ("metricx24-a00000-l0", 1, 1),
        ("metricx24-a00000-l0", 2, 4),
        ("metricx24-a00000-l0", 3, 6),
        ("metricx24-a00001-l0", 0, 2),
        ("metricx24-a00001-l0", 1, 3),
        ("metricx24-a00001-l0", 2, 5),
    ]


def run_test() -> None:
    test_spills_large_direction_only_at_quota_boundary()
    test_standard_g_slot_ids_include_local_id()
    test_assigns_all_shards_beyond_one_wave_quota()
    print("OK: manifest assigner packs slots deterministically.")


if __name__ == "__main__":
    run_test()
