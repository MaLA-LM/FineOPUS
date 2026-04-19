"""Regression tests for mixed-layout OPUS merge behavior.

Run with:
    python -m execution.opus_queue.tests.test_merge_roundtrip
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import uuid
from pathlib import Path

from execution.opus_queue import db as queue_db
from execution.opus_queue.tools import merge
from execution.opus_queue.tools.merge import convert as merge_convert


def _seed_direction(conn, *, direction_key: str, model: str, n_sentences: int) -> None:
    src_lang, tgt_lang = direction_key.split("-", 1)
    conn.execute(
        """
        INSERT INTO directions
            (direction_key, model, src_lang, tgt_lang,
             n_sentences, shard_size, est_hours, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))
        """,
        (direction_key, model, src_lang, tgt_lang, n_sentences, 1, 1.0),
    )


def _seed_done_job(
    conn,
    *,
    direction_key: str,
    model: str,
    shard_id: int,
    worker_id: str,
    out_path: str,
) -> None:
    conn.execute(
        """
        INSERT INTO jobs
            (direction_key, model, shard_id, start_idx, end_idx,
             status, attempts, worker_id, out_path)
        VALUES (?, ?, ?, ?, ?, 'done', 1, ?, ?)
        """,
        (
            direction_key,
            model,
            shard_id,
            shard_id,
            shard_id + 1,
            worker_id,
            out_path,
        ),
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def _run_merge(db_path: Path, output_base: Path, merged_base: Path, model: str) -> None:
    merge.run(
        argparse.Namespace(
            db=str(db_path),
            output_base=str(output_base),
            merged_base=str(merged_base),
            model=model,
            delete_shards=False,
            force=False,
        )
    )


def _merged_parquet_files(merged_base: Path, model: str, direction_key: str) -> list[Path]:
    merged_model_dir = merged_base / model
    part_files = sorted(merged_model_dir.glob(f"{direction_key}.part-*.parquet"))
    if part_files:
        return [path for path in part_files if path.is_file()]
    legacy_file = merged_model_dir / f"{direction_key}.parquet"
    return [legacy_file] if legacy_file.is_file() else []


def _read_merged_records(merged_base: Path, model: str, direction_key: str) -> list[dict]:
    import pyarrow.parquet as pq

    records: list[dict] = []
    for parquet_file in _merged_parquet_files(merged_base, model, direction_key):
        records.extend(pq.read_table(parquet_file).to_pylist())
    return records


def _assert_merge_only_columns_removed(records: list[dict]) -> None:
    assert all("shard_id" not in record for record in records)
    assert all("worker_id" not in record for record in records)
    assert all("direction_key" not in record for record in records)


@contextlib.contextmanager
def _patched_merge_limits(*, max_bytes: int, batch_rows: int):
    old_max_bytes = merge_convert._MAX_PARQUET_FILE_BYTES
    old_batch_rows = merge_convert._MERGE_BATCH_ROWS
    merge_convert._MAX_PARQUET_FILE_BYTES = max_bytes
    merge_convert._MERGE_BATCH_ROWS = batch_rows
    try:
        yield
    finally:
        merge_convert._MAX_PARQUET_FILE_BYTES = old_max_bytes
        merge_convert._MERGE_BATCH_ROWS = old_batch_rows


@contextlib.contextmanager
def _scratch_dir():
    root = Path(__file__).resolve().parents[3] / ".tmp_test_runs"
    root.mkdir(exist_ok=True)
    path = root / f"merge-roundtrip-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_merge_uses_queue_model_directory() -> None:
    with _scratch_dir() as root:
        db_path = root / "queue.db"
        output_base = root / "shards"
        merged_base = root / "merged"
        model = "qwen3-4b-instruct-2507"
        scorer_tag = "qwen3-4b-instruct-2507-detailed"
        direction_key = "eng_Latn-fra_Latn"

        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_direction(conn, direction_key=direction_key, model=model, n_sentences=2)
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=0,
                worker_id="wid_A",
                out_path="/tmp/eng_fra_0.jsonl",
            )
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=1,
                worker_id="wid_A",
                out_path="/tmp/eng_fra_1.jsonl",
            )
        finally:
            conn.close()

        canonical_dir = output_base / model / direction_key
        _write_jsonl(
            canonical_dir / "shard_00001.jsonl",
            [{
                "source_text": "src-1",
                "target_text": "tgt-1",
                "qe_score": 1,
                "shard_id": 1,
                "direction_key": direction_key,
            }],
        )
        _write_jsonl(
            canonical_dir / "shard_00000.jsonl",
            [{
                "source_text": "src-0",
                "target_text": "tgt-0",
                "qe_score": 0,
                "shard_id": 0,
                "direction_key": direction_key,
            }],
        )
        _write_jsonl(
            output_base / scorer_tag / direction_key / "shard_00000.jsonl",
            [{
                "source_text": "wrong",
                "target_text": "wrong",
                "qe_score": 999,
                "shard_id": 0,
                "direction_key": direction_key,
            }],
        )

        _run_merge(db_path, output_base, merged_base, model)

        merged_files = _merged_parquet_files(merged_base, model, direction_key)
        meta_file = merged_base / model / f"{direction_key}.meta.json"
        assert len(merged_files) == 1
        assert meta_file.exists()

        records = _read_merged_records(merged_base, model, direction_key)
        assert [record["source_text"] for record in records] == ["src-0", "src-1"]
        assert all(record["source_text"] != "wrong" for record in records)
        _assert_merge_only_columns_removed(records)

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["n_shards"] == 2
        assert meta["n_rows"] == 2
        assert meta["source_db"] == str(db_path.resolve())
        assert meta["model"] == model
        assert meta["direction_key"] == direction_key
        assert meta["n_parquet_files"] == 1
        assert meta["parquet_files"] == [path.name for path in merged_files]


def test_merge_filters_orphan_rows() -> None:
    with _scratch_dir() as root:
        db_path = root / "queue.db"
        output_base = root / "shards"
        merged_base = root / "merged"
        model = "metricx24"
        direction_key = "eng_Latn-fra_Latn"

        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_direction(conn, direction_key=direction_key, model=model, n_sentences=2)
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=0,
                worker_id="wid_A",
                out_path="/tmp/part-wid_A-0000.jsonl",
            )
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=1,
                worker_id="wid_A",
                out_path="/tmp/part-wid_A-0000.jsonl",
            )
        finally:
            conn.close()

        shard_dir = output_base / model / direction_key
        _write_jsonl(
            shard_dir / "part-wid_A-0000.jsonl",
            [
                {
                    "source_text": "src-0",
                    "target_text": "tgt-0",
                    "qe_score": 0,
                    "shard_id": 0,
                    "worker_id": "wid_A",
                    "direction_key": direction_key,
                },
                {
                    "source_text": "src-1",
                    "target_text": "tgt-1",
                    "qe_score": 1,
                    "shard_id": 1,
                    "worker_id": "wid_A",
                    "direction_key": direction_key,
                },
            ],
        )
        _write_jsonl(
            shard_dir / "part-wid_B-0000.jsonl",
            [
                {
                    "source_text": "orphan",
                    "target_text": "orphan",
                    "qe_score": 999,
                    "shard_id": 0,
                    "worker_id": "wid_B",
                    "direction_key": direction_key,
                }
            ],
        )

        _run_merge(db_path, output_base, merged_base, model)

        records = _read_merged_records(merged_base, model, direction_key)
        assert [record["source_text"] for record in records] == ["src-0", "src-1"]
        _assert_merge_only_columns_removed(records)


def test_merge_reads_legacy_layout() -> None:
    with _scratch_dir() as root:
        db_path = root / "queue.db"
        output_base = root / "shards"
        merged_base = root / "merged"
        model = "metricx24"
        direction_key = "eng_Latn-deu_Latn"

        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_direction(conn, direction_key=direction_key, model=model, n_sentences=1)
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=0,
                worker_id="wid_legacy",
                out_path="/tmp/shard_00000.jsonl",
            )
        finally:
            conn.close()

        _write_jsonl(
            output_base / model / direction_key / "shard_00000.jsonl",
            [{
                "source_text": "legacy",
                "target_text": "legacy",
                "qe_score": 0,
                "shard_id": 0,
                "direction_key": direction_key,
            }],
        )

        _run_merge(db_path, output_base, merged_base, model)

        records = _read_merged_records(merged_base, model, direction_key)
        assert [record["source_text"] for record in records] == ["legacy"]
        _assert_merge_only_columns_removed(records)


def test_merge_reads_mixed_layouts() -> None:
    with _scratch_dir() as root:
        db_path = root / "queue.db"
        output_base = root / "shards"
        merged_base = root / "merged"
        model = "metricx24"
        direction_key = "eng_Latn-spa_Latn"

        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_direction(conn, direction_key=direction_key, model=model, n_sentences=2)
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=0,
                worker_id="wid_legacy",
                out_path="/tmp/shard_00000.jsonl",
            )
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=1,
                worker_id="wid_new",
                out_path="/tmp/part-wid_new-0000.jsonl",
            )
        finally:
            conn.close()

        shard_dir = output_base / model / direction_key
        _write_jsonl(
            shard_dir / "shard_00000.jsonl",
            [{
                "source_text": "legacy-0",
                "target_text": "legacy-0",
                "qe_score": 0,
                "shard_id": 0,
                "direction_key": direction_key,
            }],
        )
        _write_jsonl(
            shard_dir / "part-wid_new-0000.jsonl",
            [
                {
                    "source_text": "new-1",
                    "target_text": "new-1",
                    "qe_score": 1,
                    "shard_id": 1,
                    "worker_id": "wid_new",
                    "direction_key": direction_key,
                }
            ],
        )

        _run_merge(db_path, output_base, merged_base, model)

        records = _read_merged_records(merged_base, model, direction_key)
        assert sorted(record["source_text"] for record in records) == ["legacy-0", "new-1"]
        _assert_merge_only_columns_removed(records)


def test_merge_prefers_new_layout_over_legacy_duplicate() -> None:
    with _scratch_dir() as root:
        db_path = root / "queue.db"
        output_base = root / "shards"
        merged_base = root / "merged"
        model = "metricx24"
        direction_key = "eng_Latn-ita_Latn"

        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_direction(conn, direction_key=direction_key, model=model, n_sentences=1)
            _seed_done_job(
                conn,
                direction_key=direction_key,
                model=model,
                shard_id=0,
                worker_id="wid_new",
                out_path="/tmp/part-wid_new-0000.jsonl",
            )
        finally:
            conn.close()

        shard_dir = output_base / model / direction_key
        _write_jsonl(
            shard_dir / "shard_00000.jsonl",
            [{
                "source_text": "legacy-old",
                "target_text": "legacy-old",
                "qe_score": 0,
                "shard_id": 0,
                "direction_key": direction_key,
            }],
        )
        _write_jsonl(
            shard_dir / "part-wid_new-0000.jsonl",
            [
                {
                    "source_text": "new-win",
                    "target_text": "new-win",
                    "qe_score": 1,
                    "shard_id": 0,
                    "worker_id": "wid_new",
                    "direction_key": direction_key,
                }
            ],
        )

        _run_merge(db_path, output_base, merged_base, model)

        records = _read_merged_records(merged_base, model, direction_key)
        assert [record["source_text"] for record in records] == ["new-win"]
        _assert_merge_only_columns_removed(records)


def test_merge_splits_large_directions_into_multiple_parquet_files() -> None:
    with _scratch_dir() as root:
        db_path = root / "queue.db"
        output_base = root / "shards"
        merged_base = root / "merged"
        model = "metricx24"
        direction_key = "eng_Latn-por_Latn"

        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_direction(conn, direction_key=direction_key, model=model, n_sentences=3)
            for shard_id in range(3):
                _seed_done_job(
                    conn,
                    direction_key=direction_key,
                    model=model,
                    shard_id=shard_id,
                    worker_id="wid_split",
                    out_path="/tmp/part-wid_split-0000.jsonl",
                )
        finally:
            conn.close()

        _write_jsonl(
            output_base / model / direction_key / "part-wid_split-0000.jsonl",
            [
                {
                    "source_text": f"split-{shard_id}",
                    "target_text": f"split-{shard_id}",
                    "qe_score": shard_id,
                    "shard_id": shard_id,
                    "worker_id": "wid_split",
                    "direction_key": direction_key,
                }
                for shard_id in range(3)
            ],
        )

        with _patched_merge_limits(max_bytes=1, batch_rows=1):
            _run_merge(db_path, output_base, merged_base, model)

        merged_files = _merged_parquet_files(merged_base, model, direction_key)
        meta = json.loads(
            (merged_base / model / f"{direction_key}.meta.json").read_text(encoding="utf-8")
        )
        records = _read_merged_records(merged_base, model, direction_key)
        assert len(merged_files) == 3
        assert [record["source_text"] for record in records] == [
            "split-0",
            "split-1",
            "split-2",
        ]
        _assert_merge_only_columns_removed(records)
        assert meta["n_parquet_files"] == 3
        assert meta["parquet_files"] == [path.name for path in merged_files]


def run_test() -> None:
    test_merge_uses_queue_model_directory()
    test_merge_filters_orphan_rows()
    test_merge_reads_legacy_layout()
    test_merge_reads_mixed_layouts()
    test_merge_prefers_new_layout_over_legacy_duplicate()
    test_merge_splits_large_directions_into_multiple_parquet_files()
    print("OK: merge handles legacy shards, worker-owned part files, orphan filtering, and split parquet output.")


if __name__ == "__main__":
    run_test()
