from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from stand_alone_modules.group_merged.grouper import group_merged_outputs


def _scratch_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / ".tmp_test_runs"
    root.mkdir(exist_ok=True)
    path = root / f"group-merged-{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_groups_flat_direction_files() -> None:
    root = _scratch_dir()
    try:
        model_dir = root / "metricx24"
        _touch(model_dir / "ace_Latn-ceb_Latn.part-0000.parquet")
        _touch(model_dir / "ace_Latn-ceb_Latn.meta.json")
        _touch(model_dir / "eng_Latn-fra_Latn.parquet")
        _touch(model_dir / "notes.txt")

        summary = group_merged_outputs(root)

        assert summary.models_seen == 1
        assert summary.directions_seen == 2
        assert summary.files_moved == 3
        assert (model_dir / "ace_Latn-ceb_Latn" / "ace_Latn-ceb_Latn.part-0000.parquet").is_file()
        assert (model_dir / "ace_Latn-ceb_Latn" / "ace_Latn-ceb_Latn.meta.json").is_file()
        assert (model_dir / "eng_Latn-fra_Latn" / "eng_Latn-fra_Latn.parquet").is_file()
        assert (model_dir / "notes.txt").is_file()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reports_conflicts_before_moving() -> None:
    root = _scratch_dir()
    try:
        model_dir = root / "metricx24"
        flat_file = model_dir / "ace_Latn-ceb_Latn.meta.json"
        grouped_file = model_dir / "ace_Latn-ceb_Latn" / "ace_Latn-ceb_Latn.meta.json"
        _touch(flat_file)
        _touch(grouped_file)

        summary = group_merged_outputs(root)

        assert summary.files_moved == 0
        assert summary.conflicts == [grouped_file]
        assert flat_file.is_file()
        assert grouped_file.is_file()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_test() -> None:
    test_groups_flat_direction_files()
    test_reports_conflicts_before_moving()
    print("OK: group_merged reorganizes flat OPUS merged outputs.")


if __name__ == "__main__":
    run_test()
