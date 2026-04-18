"""Regression tests for the OPUS adapter contract.

Run with:
    python -m execution.opus_queue.tests.test_opus_adapter_contract
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataset.mediator import get_dataset
from dataset.opus.discovery import DEFAULT_SPLIT, discover_directions
from dataset.opus.frames import build_frames
from dataset.opus.builder import load_opus_parallel
from execution.flores_array.runner import validate_flores_args

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None


def _write_direction(root: Path, direction_key: str) -> Path:
    direction_dir = root / direction_key
    direction_dir.mkdir(parents=True, exist_ok=True)
    shard_path = direction_dir / f"{direction_key}_shard_0.parquet"
    if pa is None or pq is None:
        shard_path.write_bytes(b"")
        return direction_dir
    table = pa.table(
        {
            "source_text": ["hello", "bye"],
            "target_text": ["bonjour", "au revoir"],
            "domain": ["news", "news"],
            "example_id": [1, 2],
        }
    )
    pq.write_table(table, shard_path)
    return direction_dir


def _scratch_dir() -> Path:
    path = Path(__file__).resolve().parent / "_tmp_opus_contract"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_get_dataset_opus_imports_cleanly() -> None:
    dataset = get_dataset("opus")
    assert dataset.id == "opus"
    assert dataset.split_values == (DEFAULT_SPLIT,)


def test_discover_and_load_opus_uses_placeholder_split_only_for_compatibility() -> None:
    root = _scratch_dir()
    try:
        direction_dir = _write_direction(root, "eng_Latn-fra_Latn")

        directions = discover_directions(root, split=None)
        assert directions == [("eng_Latn", "fra_Latn", DEFAULT_SPLIT, direction_dir)]

        if pa is None or pq is None:
            return

        examples_default = load_opus_parallel("eng_Latn", "fra_Latn", root=root)
        examples_all = load_opus_parallel(
            "eng_Latn", "fra_Latn", split=DEFAULT_SPLIT, root=root
        )

        assert len(examples_default) == 2
        assert examples_default == examples_all
        assert examples_default[0]["src"] == "hello"
        assert examples_default[0]["tgt"] == "bonjour"

        try:
            load_opus_parallel("eng_Latn", "fra_Latn", split="dev", root=root)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for a real OPUS split name")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_opus_frames_preserve_input_columns_and_only_add_qe_fields() -> None:
    try:
        frame = build_frames(
            "metricx24",
            "opus",
            None,
            "eng_Latn",
            "fra_Latn",
            [0.25],
            [
                {
                    "source_text": "hello",
                    "target_text": "bonjour",
                    "domain": "news",
                    "example_id": 7,
                    "src": "hello",
                    "tgt": "bonjour",
                }
            ],
            src_lang_seen=True,
            tgt_lang_seen="unknown",
            mean=0.25,
            median=0.25,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pandas":
            return
        raise

    record = frame.to_dict(orient="records")[0]
    assert set(record) == {
        "source_text",
        "target_text",
        "domain",
        "example_id",
        "qe_model",
        "qe_score",
        "src_seen",
        "tgt_seen",
    }
    assert record["qe_model"] == "metricx24"
    assert record["qe_score"] == 0.25
    assert record["src_seen"] is True
    assert record["tgt_seen"] == "unknown"


def test_flores_array_executor_rejects_opus_dataset() -> None:
    dataset = get_dataset("opus")
    args = argparse.Namespace(
        manifest="ignored.tsv",
        shard_id=0,
        num_shards=1,
        max_directions_per_part=1,
        target_part_bytes=1,
    )

    try:
        validate_flores_args(args, dataset)
    except SystemExit as exc:
        assert "opus_queue" in str(exc)
    else:
        raise AssertionError("expected OPUS to be rejected by flores_array")


def run_test() -> None:
    test_get_dataset_opus_imports_cleanly()
    test_discover_and_load_opus_uses_placeholder_split_only_for_compatibility()
    test_opus_frames_preserve_input_columns_and_only_add_qe_fields()
    test_flores_array_executor_rejects_opus_dataset()
    print("OK: OPUS adapter keeps passthrough rows and split compatibility.")


if __name__ == "__main__":
    run_test()
