"""Regression tests for bicleaner language-flag handling.

Run with:
    python -m execution.opus_queue.tests.test_bicleaner_backend
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import call, patch

from src.backends.bicleaner import backend as bicleaner_backend


def test_run_bicleaner_uses_both_language_flags() -> None:
    with patch.object(bicleaner_backend.subprocess, "run") as run_mock:
        bicleaner_backend.run_bicleaner(
            Path("input.tsv"),
            Path("output.tsv"),
            "model-id",
            "en",
            "fr",
        )

    run_mock.assert_called_once_with(
        [
            "bicleaner-ai-classify",
            "--scol",
            "1",
            "--tcol",
            "2",
            "-s",
            "en",
            "-t",
            "fr",
            "--disable_hardrules",
            "input.tsv",
            "output.tsv",
            "model-id",
        ],
        check=True,
    )


def test_run_bicleaner_uses_source_only_flag() -> None:
    with patch.object(bicleaner_backend.subprocess, "run") as run_mock:
        bicleaner_backend.run_bicleaner(
            Path("input.tsv"),
            Path("output.tsv"),
            "model-id",
            "en",
            None,
        )

    run_mock.assert_called_once_with(
        [
            "bicleaner-ai-classify",
            "--scol",
            "1",
            "--tcol",
            "2",
            "-s",
            "en",
            "--disable_hardrules",
            "input.tsv",
            "output.tsv",
            "model-id",
        ],
        check=True,
    )


def test_run_bicleaner_uses_target_only_flag() -> None:
    with patch.object(bicleaner_backend.subprocess, "run") as run_mock:
        bicleaner_backend.run_bicleaner(
            Path("input.tsv"),
            Path("output.tsv"),
            "model-id",
            None,
            "fr",
        )

    run_mock.assert_called_once_with(
        [
            "bicleaner-ai-classify",
            "--scol",
            "1",
            "--tcol",
            "2",
            "-t",
            "fr",
            "--disable_hardrules",
            "input.tsv",
            "output.tsv",
            "model-id",
        ],
        check=True,
    )


def test_run_bicleaner_omits_language_flags_when_none_present() -> None:
    with patch.object(bicleaner_backend.subprocess, "run") as run_mock:
        bicleaner_backend.run_bicleaner(
            Path("input.tsv"),
            Path("output.tsv"),
            "model-id",
            None,
            None,
        )

    run_mock.assert_called_once_with(
        [
            "bicleaner-ai-classify",
            "--scol",
            "1",
            "--tcol",
            "2",
            "--disable_hardrules",
            "input.tsv",
            "output.tsv",
            "model-id",
        ],
        check=True,
    )


def test_run_bicleaner_retries_without_language_flags_after_failure() -> None:
    with patch.object(
        bicleaner_backend.subprocess,
        "run",
        side_effect=[
            subprocess.CalledProcessError(1, "bicleaner-ai-classify"),
            subprocess.CompletedProcess(["bicleaner-ai-classify"], 0),
        ],
    ) as run_mock:
        bicleaner_backend.run_bicleaner(
            Path("input.tsv"),
            Path("output.tsv"),
            "model-id",
            "en",
            None,
        )

    assert run_mock.call_args_list == [
        call(
            [
                "bicleaner-ai-classify",
                "--scol",
                "1",
                "--tcol",
                "2",
                "-s",
                "en",
                "--disable_hardrules",
                "input.tsv",
                "output.tsv",
                "model-id",
            ],
            check=True,
        ),
        call(
            [
                "bicleaner-ai-classify",
                "--scol",
                "1",
                "--tcol",
                "2",
                "--disable_hardrules",
                "input.tsv",
                "output.tsv",
                "model-id",
            ],
            check=True,
        ),
    ]


def run_test() -> None:
    test_run_bicleaner_uses_both_language_flags()
    test_run_bicleaner_uses_source_only_flag()
    test_run_bicleaner_uses_target_only_flag()
    test_run_bicleaner_omits_language_flags_when_none_present()
    test_run_bicleaner_retries_without_language_flags_after_failure()
    print("OK: bicleaner language flags cover both, partial, none, and fallback modes.")


if __name__ == "__main__":
    run_test()
