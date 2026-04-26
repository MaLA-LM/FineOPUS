"""Regression tests for soft direction affinity in queue claims.

Run with:
    python -m execution.opus_queue.tests.test_claim_affinity
"""

from __future__ import annotations

from execution.opus_queue import db as queue_db
from execution.opus_queue.tests._tmp import workspace_temp_dir


def _seed_job(
    conn,
    *,
    direction_key: str,
    model: str,
    shard_id: int,
    start_idx: int,
    end_idx: int,
) -> None:
    src_lang, tgt_lang = direction_key.split("-", 1)
    conn.execute(
        """
        INSERT OR IGNORE INTO directions
            (direction_key, model, src_lang, tgt_lang,
             n_sentences, shard_size, est_hours, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))
        """,
        (direction_key, model, src_lang, tgt_lang, 1_000, 50, 1.0),
    )
    conn.execute(
        """
        INSERT INTO jobs
            (direction_key, model, shard_id, start_idx, end_idx, status, attempts)
        VALUES (?, ?, ?, ?, ?, 'pending', 0)
        """,
        (direction_key, model, shard_id, start_idx, end_idx),
    )


def test_preferred_direction_wins_over_larger_global_shard() -> None:
    with workspace_temp_dir("claim_affinity_preferred") as tmp:
        conn = queue_db.connect(tmp / "test.db")
        try:
            queue_db.initialize(conn)
            model = "metricx24"
            preferred = "eng_Latn-fra_Latn"
            other = "eng_Latn-deu_Latn"
            _seed_job(
                conn,
                direction_key=preferred,
                model=model,
                shard_id=0,
                start_idx=0,
                end_idx=10,
            )
            _seed_job(
                conn,
                direction_key=other,
                model=model,
                shard_id=0,
                start_idx=0,
                end_idx=100,
            )

            claimed = queue_db.claim_next(
                conn,
                model,
                "worker-a",
                preferred_direction_key=preferred,
            )

            assert claimed is not None
            assert claimed["direction_key"] == preferred
        finally:
            conn.close()


def test_claim_falls_back_to_global_policy_when_preferred_drained() -> None:
    with workspace_temp_dir("claim_affinity_fallback") as tmp:
        conn = queue_db.connect(tmp / "test.db")
        try:
            queue_db.initialize(conn)
            model = "metricx24"
            preferred = "eng_Latn-fra_Latn"
            other = "eng_Latn-deu_Latn"
            _seed_job(
                conn,
                direction_key=preferred,
                model=model,
                shard_id=0,
                start_idx=0,
                end_idx=10,
            )
            _seed_job(
                conn,
                direction_key=other,
                model=model,
                shard_id=0,
                start_idx=0,
                end_idx=100,
            )

            first = queue_db.claim_next(
                conn,
                model,
                "worker-a",
                preferred_direction_key=preferred,
            )
            assert first is not None
            assert first["direction_key"] == preferred
            assert (
                queue_db.mark_done(
                    conn,
                    first["direction_key"],
                    first["model"],
                    first["shard_id"],
                    "worker-a",
                    "/tmp/preferred.jsonl",
                )
                == "done"
            )

            second = queue_db.claim_next(
                conn,
                model,
                "worker-a",
                preferred_direction_key=preferred,
            )

            assert second is not None
            assert second["direction_key"] == other
        finally:
            conn.close()


def run_test() -> None:
    test_preferred_direction_wins_over_larger_global_shard()
    test_claim_falls_back_to_global_policy_when_preferred_drained()
    print("OK: soft direction affinity prefers local work and falls back globally.")


if __name__ == "__main__":
    run_test()
