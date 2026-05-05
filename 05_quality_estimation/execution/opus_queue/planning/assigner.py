from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from execution.opus_queue.planning.shard_planner import expected_shard_seconds

__all__ = [
    "AssignedShard",
    "AssignmentSummary",
    "PendingShard",
    "assign_shards",
    "quota_shards",
]


@dataclass(frozen=True)
class PendingShard:
    model: str
    direction_key: str
    src_lang: str
    tgt_lang: str
    shard_id: int
    start_idx: int
    end_idx: int
    shard_size: int
    expected_seconds: int | None = None

    @property
    def effective_expected_seconds(self) -> int:
        if self.expected_seconds is not None:
            return int(self.expected_seconds)
        return expected_shard_seconds(self.model)


@dataclass(frozen=True)
class AssignedShard:
    worker_slot_id: str
    model: str
    array_task_id: int
    local_id: int
    assignment_seq: int
    direction_key: str
    src_lang: str
    tgt_lang: str
    shard_id: int
    start_idx: int
    end_idx: int
    shard_size: int
    expected_seconds: int

    def to_dict(self) -> dict:
        return {
            "worker_slot_id": self.worker_slot_id,
            "model": self.model,
            "array_task_id": self.array_task_id,
            "local_id": self.local_id,
            "assignment_seq": self.assignment_seq,
            "direction_key": self.direction_key,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "shard_id": self.shard_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "shard_size": self.shard_size,
            "expected_seconds": self.expected_seconds,
        }


@dataclass(frozen=True)
class AssignmentSummary:
    model: str
    slots_declared: int
    slots_per_array_task: int
    quota_shards_per_worker: int
    pending_shards: int
    assigned_shards: int
    shortfall_shards: int
    directions_total: int
    directions_split_across_slots: int
    estimated_waves: int
    usable_walltime_seconds: int
    max_load_seconds: int
    min_load_seconds: int
    avg_load_seconds: float

    def to_dict(self) -> dict:
        return {
            "slots_declared": self.slots_declared,
            "slots_per_array_task": self.slots_per_array_task,
            "quota_shards_per_worker": self.quota_shards_per_worker,
            "pending_shards": self.pending_shards,
            "assigned_shards": self.assigned_shards,
            "shortfall_shards": self.shortfall_shards,
            "directions_total": self.directions_total,
            "directions_split_across_slots": self.directions_split_across_slots,
            "estimated_waves": self.estimated_waves,
            "usable_walltime_seconds": self.usable_walltime_seconds,
            "max_load_seconds": self.max_load_seconds,
            "min_load_seconds": self.min_load_seconds,
            "avg_load_seconds": self.avg_load_seconds,
        }


@dataclass
class _Slot:
    index: int
    model: str
    slots_per_array_task: int
    used_shards: int = 0
    load_seconds: int = 0
    next_seq: int = 0

    @property
    def array_task_id(self) -> int:
        return self.index // self.slots_per_array_task

    @property
    def local_id(self) -> int:
        return self.index % self.slots_per_array_task

    @property
    def worker_slot_id(self) -> str:
        return f"{self.model}-a{self.array_task_id:05d}-l{self.local_id}"


def quota_shards(
    model: str,
    *,
    walltime_seconds: int,
    safety_factor: float,
) -> int:
    expected = expected_shard_seconds(model)
    return max(1, math.floor(int(walltime_seconds) * float(safety_factor) / expected))


def assign_shards(
    shards: Iterable[PendingShard],
    *,
    slots_declared: int,
    slots_per_array_task: int,
    walltime_seconds: int,
    safety_factor: float,
    sparse_round_robin_floor: float = 1.0,
) -> tuple[list[AssignedShard], AssignmentSummary]:
    rows = sorted(
        list(shards),
        key=lambda row: (row.model, row.direction_key, row.shard_id),
    )
    if not rows:
        raise ValueError("assign_shards requires at least one pending shard")
    if slots_declared < 1:
        raise ValueError("slots_declared must be >= 1")
    if slots_per_array_task < 1:
        raise ValueError("slots_per_array_task must be >= 1")

    models = {row.model for row in rows}
    if len(models) != 1:
        raise ValueError("assign_shards expects rows for exactly one model")
    model = rows[0].model
    quota = quota_shards(
        model,
        walltime_seconds=walltime_seconds,
        safety_factor=safety_factor,
    )
    slots = [
        _Slot(index=idx, model=model, slots_per_array_task=slots_per_array_task)
        for idx in range(slots_declared)
    ]

    by_direction: dict[str, list[PendingShard]] = defaultdict(list)
    for row in rows:
        by_direction[row.direction_key].append(row)
    directions = sorted(
        by_direction.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    assignments: list[AssignedShard] = []
    use_sparse_round_robin = (len(rows) / slots_declared) < sparse_round_robin_floor
    current_slot_idx = 0
    for direction_key, direction_rows in directions:
        del direction_key
        if use_sparse_round_robin:
            slot_idx = current_slot_idx
        else:
            slot_idx = current_slot_idx

        for shard in direction_rows:
            if slots[slot_idx].used_shards >= quota and slots[slot_idx].used_shards > 0:
                slot_idx = (slot_idx + 1) % slots_declared

            slot = slots[slot_idx]
            assignments.append(
                AssignedShard(
                    worker_slot_id=slot.worker_slot_id,
                    model=shard.model,
                    array_task_id=slot.array_task_id,
                    local_id=slot.local_id,
                    assignment_seq=slot.next_seq,
                    direction_key=shard.direction_key,
                    src_lang=shard.src_lang,
                    tgt_lang=shard.tgt_lang,
                    shard_id=shard.shard_id,
                    start_idx=shard.start_idx,
                    end_idx=shard.end_idx,
                    shard_size=shard.shard_size,
                    expected_seconds=shard.effective_expected_seconds,
                )
            )
            slot.next_seq += 1
            slot.used_shards += 1
            slot.load_seconds += shard.effective_expected_seconds

        if use_sparse_round_robin:
            current_slot_idx = (slot_idx + 1) % slots_declared
        else:
            current_slot_idx = slot_idx

    direction_slots: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        direction_slots[assignment.direction_key].add(assignment.worker_slot_id)

    loads = [slot.load_seconds for slot in slots]
    assigned = len(assignments)
    usable_walltime = max(1, math.floor(int(walltime_seconds) * float(safety_factor)))
    max_load = max(loads) if loads else 0
    summary = AssignmentSummary(
        model=model,
        slots_declared=slots_declared,
        slots_per_array_task=slots_per_array_task,
        quota_shards_per_worker=quota,
        pending_shards=len(rows),
        assigned_shards=assigned,
        shortfall_shards=0,
        directions_total=len(by_direction),
        directions_split_across_slots=sum(
            1 for slot_ids in direction_slots.values() if len(slot_ids) > 1
        ),
        estimated_waves=max(1, math.ceil(max_load / usable_walltime)) if assigned else 0,
        usable_walltime_seconds=usable_walltime,
        max_load_seconds=max_load,
        min_load_seconds=min(loads) if loads else 0,
        avg_load_seconds=(sum(loads) / len(loads)) if loads else 0.0,
    )
    assignments.sort(
        key=lambda row: (
            row.model,
            row.array_task_id,
            row.local_id,
            row.assignment_seq,
            row.direction_key,
            row.shard_id,
        )
    )
    return assignments, summary
