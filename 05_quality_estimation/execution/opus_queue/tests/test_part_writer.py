"""Unit tests for worker-owned shard part writing.

Run with:
    python -m execution.opus_queue.tests.test_part_writer
"""

from __future__ import annotations

import contextlib
import json
import shutil
import uuid
from pathlib import Path

from execution.opus_queue.worker.shard_io import DirectionPartWriter, sanitize_worker_id


class _Frame:

    def __init__(self, records: list[dict]) -> None:
        self.columns = list(records[0].keys()) if records else []
        self._records = records

    def itertuples(self, index: bool = False, name=None):
        del index, name
        for record in self._records:
            yield tuple(record[column] for column in self.columns)


def _frame(direction_key: str, shard_id: int, worker_id: str, payload: str) -> _Frame:
    return _Frame(
        [
            {
                "source_text": payload,
                "target_text": payload,
                "qe_score": shard_id,
                "direction_key": direction_key,
                "shard_id": shard_id,
                "worker_id": worker_id,
            }
        ]
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@contextlib.contextmanager
def _scratch_dir():
    root = Path(__file__).resolve().parents[3] / ".tmp_test_runs"
    root.mkdir(exist_ok=True)
    path = root / f"part-writer-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_rotation_by_bytes() -> None:
    with _scratch_dir() as tmp:
        output_base = tmp / "shards"
        writer = DirectionPartWriter(
            output_base=output_base,
            model="metricx24",
            worker_id="wid:test/bytes",
            max_bytes=220,
            max_shards_per_part=10,
        )
        direction_key = "eng_Latn-fra_Latn"

        writer.append_shard(_frame(direction_key, 0, "wid_A", "a" * 70), direction_key, 0)
        writer.append_shard(_frame(direction_key, 1, "wid_A", "b" * 70), direction_key, 1)
        writer.close_all()

        part_files = sorted((output_base / "metricx24" / direction_key).glob("part-*.jsonl"))
        assert len(part_files) == 2
        assert sum(len(_read_jsonl(path)) for path in part_files) == 2


def test_rotation_by_shard_count() -> None:
    with _scratch_dir() as tmp:
        output_base = tmp / "shards"
        writer = DirectionPartWriter(
            output_base=output_base,
            model="metricx24",
            worker_id="wid-count",
            max_bytes=10_000,
            max_shards_per_part=2,
        )
        direction_key = "eng_Latn-deu_Latn"

        for shard_id in range(3):
            writer.append_shard(
                _frame(direction_key, shard_id, "wid_A", f"payload-{shard_id}"),
                direction_key,
                shard_id,
            )
        writer.close_all()

        part_files = sorted((output_base / "metricx24" / direction_key).glob("part-*.jsonl"))
        assert len(part_files) == 2
        assert [len(_read_jsonl(path)) for path in part_files] == [2, 1]


def test_direction_handoff_reuses_per_direction_state() -> None:
    with _scratch_dir() as tmp:
        output_base = tmp / "shards"
        writer = DirectionPartWriter(
            output_base=output_base,
            model="metricx24",
            worker_id="wid-handoff",
            max_bytes=10_000,
            max_shards_per_part=10,
        )
        direction_a = "eng_Latn-spa_Latn"
        direction_b = "eng_Latn-ita_Latn"

        writer.append_shard(_frame(direction_a, 0, "wid_A", "a0"), direction_a, 0)
        writer.close_direction(direction_a)
        writer.append_shard(_frame(direction_b, 0, "wid_A", "b0"), direction_b, 0)
        writer.close_direction(direction_b)
        writer.append_shard(_frame(direction_a, 1, "wid_A", "a1"), direction_a, 1)
        writer.close_all()

        dir_a_files = sorted((output_base / "metricx24" / direction_a).glob("part-*.jsonl"))
        dir_b_files = sorted((output_base / "metricx24" / direction_b).glob("part-*.jsonl"))
        assert len(dir_a_files) == 1
        assert len(dir_b_files) == 1
        assert [row["shard_id"] for row in _read_jsonl(dir_a_files[0])] == [0, 1]
        assert [row["shard_id"] for row in _read_jsonl(dir_b_files[0])] == [0]


def test_sanitize_worker_id_truncates_long_names() -> None:
    worker_id = ("host/with:unsafe\\" * 20) + "tail"
    safe = sanitize_worker_id(worker_id)
    assert len(safe) <= 180
    assert "/" not in safe
    assert "\\" not in safe
    assert ":" not in safe


def run_test() -> None:
    test_rotation_by_bytes()
    test_rotation_by_shard_count()
    test_direction_handoff_reuses_per_direction_state()
    test_sanitize_worker_id_truncates_long_names()
    print("OK: worker-owned part writer rotates correctly and handles direction handoff.")


if __name__ == "__main__":
    run_test()
