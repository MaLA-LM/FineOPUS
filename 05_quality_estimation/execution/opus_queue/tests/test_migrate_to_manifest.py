from __future__ import annotations

import argparse
import json
import sqlite3

from execution.opus_queue.db import initialize
from execution.opus_queue.tests._tmp import workspace_temp_dir
from execution.opus_queue.tools.manifest_probe.runner import run as probe_manifest
from execution.opus_queue.tools.migrate_to_manifest.runner import run as migrate


def test_migration_writes_manifest_and_probe_accepts_array() -> None:
    with workspace_temp_dir("manifest-migration") as tmp:
        db_path = tmp / "jobs.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            initialize(conn)
            conn.execute(
                """
                INSERT INTO directions
                    (direction_key, model, src_lang, tgt_lang, n_sentences,
                     shard_size, est_hours, created_at)
                VALUES
                    ('eng_Latn-fra_Latn', 'metricx24', 'eng_Latn', 'fra_Latn',
                     300, 100, 1.0, 1)
                """
            )
            for shard_id, status in enumerate(["pending", "failed", "pending", "done"]):
                conn.execute(
                    """
                    INSERT INTO jobs
                        (direction_key, model, shard_id, start_idx, end_idx, status)
                    VALUES
                        ('eng_Latn-fra_Latn', 'metricx24', ?, ?, ?, ?)
                    """,
                    (shard_id, shard_id * 100, (shard_id + 1) * 100, status),
                )
            conn.commit()
        finally:
            conn.close()

        manifest_root = tmp / "manifests"
        args = argparse.Namespace(
            db=str(db_path),
            manifest_root=str(manifest_root),
            build_tag="unit",
            walltime_seconds=1_200,
            safety_factor=1.0,
            slots=["metricx24:2:1"],
            slots_per_task=[],
            include_status="pending,failed",
            dry_run=False,
        )
        assert migrate(args) == 0

        manifest_path = manifest_root / "unit" / "manifest.jsonl"
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["shard_id"] for row in rows] == [0, 1, 2]

        probe_args = argparse.Namespace(
            manifest_root=str(manifest_root),
            build_tag="unit",
            model="metricx24",
            array_spec="0-1",
            slots_per_task=1,
            trace_root=str(tmp / "trace"),
        )
        assert probe_manifest(probe_args) == 0

        partial_probe_args = argparse.Namespace(
            manifest_root=str(manifest_root),
            build_tag="unit",
            model="metricx24",
            array_spec="0",
            slots_per_task=1,
            trace_root=str(tmp / "trace"),
        )
        assert probe_manifest(partial_probe_args) == 0

        out_of_range_probe_args = argparse.Namespace(
            manifest_root=str(manifest_root),
            build_tag="unit",
            model="metricx24",
            array_spec="2",
            slots_per_task=1,
            trace_root=str(tmp / "trace"),
        )
        try:
            probe_manifest(out_of_range_probe_args)
        except SystemExit as exc:
            assert "Unknown ids=2" in str(exc)
        else:
            raise AssertionError("out-of-range array id should be rejected")


def run_test() -> None:
    test_migration_writes_manifest_and_probe_accepts_array()
    print("OK: migration writes a manifest accepted by the launcher probe.")


if __name__ == "__main__":
    run_test()
