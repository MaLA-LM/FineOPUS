"""Regression tests for merge path selection and shard ordering.

Run with:
    python -m execution.opus_queue.tests.test_merge_roundtrip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from execution.opus_queue import db as queue_db
from execution.opus_queue.tools import merge


def _seed_done_jobs(conn, *, direction_key: str, model: str) -> None:
    src_lang, tgt_lang = direction_key.split("-", 1)
    conn.execute(
        """
        INSERT INTO directions
            (direction_key, model, src_lang, tgt_lang,
             n_sentences, shard_size, est_hours, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))
        """,
        (direction_key, model, src_lang, tgt_lang, 2, 1, 1.0),
    )
    for shard_id in (0, 1):
        conn.execute(
            """
            INSERT INTO jobs
                (direction_key, model, shard_id, start_idx, end_idx, status, attempts, out_path)
            VALUES (?, ?, ?, ?, ?, 'done', 1, ?)
            """,
            (
                direction_key,
                model,
                shard_id,
                shard_id,
                shard_id + 1,
                f"/tmp/{direction_key}_{shard_id}.jsonl",
            ),
        )


def run_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_path = root / "queue.db"
        output_base = root / "shards"
        merged_base = root / "merged"
        model = "qwen3-4b-instruct-2507"
        scorer_tag = "qwen3-4b-instruct-2507-detailed"
        direction_key = "eng_Latn-fra_Latn"

        conn = queue_db.connect(db_path)
        try:
            queue_db.initialize(conn)
            _seed_done_jobs(conn, direction_key=direction_key, model=model)
        finally:
            conn.close()

        canonical_dir = output_base / model / direction_key
        canonical_dir.mkdir(parents=True, exist_ok=True)
        (canonical_dir / "shard_00001.jsonl").write_text(
            '{"source_text":"src-1","target_text":"tgt-1","qe_score":1}\n',
            encoding="utf-8",
        )
        (canonical_dir / "shard_00000.jsonl").write_text(
            '{"source_text":"src-0","target_text":"tgt-0","qe_score":0}\n',
            encoding="utf-8",
        )

        wrong_dir = output_base / scorer_tag / direction_key
        wrong_dir.mkdir(parents=True, exist_ok=True)
        (wrong_dir / "shard_00000.jsonl").write_text(
            '{"source_text":"wrong","target_text":"wrong","qe_score":999}\n',
            encoding="utf-8",
        )

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

        merged_file = merged_base / model / f"{direction_key}.parquet"
        meta_file = merged_base / model / f"{direction_key}.meta.json"
        assert merged_file.exists()
        assert meta_file.exists()

        import pyarrow.parquet as pq

        records = pq.read_table(merged_file).to_pylist()
        assert [record["source_text"] for record in records] == ["src-0", "src-1"]
        assert all(record["source_text"] != "wrong" for record in records)

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["n_shards"] == 2
        assert meta["n_rows"] == 2
        assert meta["source_db"] == str(db_path.resolve())
        assert meta["model"] == model
        assert meta["direction_key"] == direction_key

    print("OK: merge uses the queue model directory and preserves shard order.")


if __name__ == "__main__":
    run_test()
