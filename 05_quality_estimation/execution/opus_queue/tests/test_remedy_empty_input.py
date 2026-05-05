"""Regression test for empty OPUS slices reaching the ReMedy backend.

Run with:
    python -m execution.opus_queue.tests.test_remedy_empty_input
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from execution.flores_array.manifest import ManifestEntry
from src.backends.remedy.runner import score_entry


class _Dataset:
    id = "opus"
    default_root = Path(".")
    split_values = ("all",)

    def load_parallel(self, src_lang, tgt_lang, *, split="all", root=None):
        return []

    def limit_rows(self, examples, max_rows):
        return examples

    def build_frames(
        self,
        model_name,
        dataset,
        split,
        src_lang,
        tgt_lang,
        scores,
        examples,
        *,
        src_lang_seen,
        tgt_lang_seen,
        mean,
        median,
    ):
        return {
            "model_name": model_name,
            "dataset": dataset,
            "split": split,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "scores": scores,
            "examples": examples,
            "src_lang_seen": src_lang_seen,
            "tgt_lang_seen": tgt_lang_seen,
            "mean": mean,
            "median": median,
        }


class _LanguageSupport:
    def support_status(self, code):
        return True


def test_remedy_skips_empty_input_without_invoking_cli() -> None:
    args = argparse.Namespace(root=Path("."), max_rows=None, output_base=Path("."))
    entry = ManifestEntry(src_lang="aeb_Arab", tgt_lang="ita_Latn", split="all")
    spec = SimpleNamespace(model_id="ShaomuTan/ReMedy-9B-22")

    with patch("src.backends.remedy.runner.run_remedy") as run_remedy:
        frame = score_entry(
            entry,
            args,
            spec,
            _Dataset(),
            _LanguageSupport(),
            Path(".cache"),
        )

    run_remedy.assert_not_called()
    assert frame["scores"] == []
    assert frame["examples"] == []
    assert frame["mean"] is None
    assert frame["median"] is None


def run_test() -> None:
    test_remedy_skips_empty_input_without_invoking_cli()
    print("OK: ReMedy skips empty input before launching remedy-score.")


if __name__ == "__main__":
    run_test()
