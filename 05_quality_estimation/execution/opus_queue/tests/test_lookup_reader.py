"""Regression tests for lookup CSV parsing."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import mkstemp

from execution.opus_queue.ops.lookup_reader import read_lookup_rows


@contextmanager
def lookup_csv(body: str):
    fd, raw_path = mkstemp(
        suffix=".csv", dir=str(Path(__file__).resolve().parent)
    )
    os.close(fd)
    path = Path(raw_path)
    try:
        path.write_text(body, encoding="utf-8", newline="")
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_read_lookup_rows_returns_expected_rows() -> None:
    with lookup_csv(
        "direction_key,winner_model,est_hours\n"
        " eng_Latn-fra_Latn , metricx24 ,1.5\n"
        "eng_Latn-deu_Latn,unsupported,2.0\n"
        "eng_Latn-spa_Latn,,3.0\n"
    ) as lookup_path:
        parsed = read_lookup_rows(lookup_path)

    assert parsed == [
        {
            "direction_key": "eng_Latn-fra_Latn",
            "winner_model": "metricx24",
            "est_hours": 1.5,
        },
        {
            "direction_key": "eng_Latn-deu_Latn",
            "winner_model": "unsupported",
            "est_hours": 2.0,
        },
        {
            "direction_key": "eng_Latn-spa_Latn",
            "winner_model": "",
            "est_hours": 3.0,
        },
    ]


def test_read_lookup_rows_requires_expected_columns() -> None:
    with lookup_csv(
        "direction_key,winner_model\n"
        "eng_Latn-fra_Latn,metricx24\n"
    ) as lookup_path:
        try:
            read_lookup_rows(lookup_path)
        except SystemExit as exc:
            assert "est_hours" in str(exc)
        else:
            raise AssertionError("expected SystemExit when required columns are missing")


def test_read_lookup_rows_maps_missing_est_hours_to_none() -> None:
    with lookup_csv(
        "direction_key,winner_model,est_hours\n"
        "eng_Latn-fra_Latn,metricx24,\n"
    ) as lookup_path:
        parsed = read_lookup_rows(lookup_path)

    assert parsed == [
        {
            "direction_key": "eng_Latn-fra_Latn",
            "winner_model": "metricx24",
            "est_hours": None,
        }
    ]


def test_read_lookup_rows_rejects_invalid_est_hours() -> None:
    with lookup_csv(
        "direction_key,winner_model,est_hours\n"
        "eng_Latn-fra_Latn,metricx24,not-a-number\n"
    ) as lookup_path:
        try:
            read_lookup_rows(lookup_path)
        except SystemExit as exc:
            assert "Invalid est_hours value" in str(exc)
        else:
            raise AssertionError("expected SystemExit for invalid est_hours")


def run_test() -> None:
    test_read_lookup_rows_returns_expected_rows()
    test_read_lookup_rows_requires_expected_columns()
    test_read_lookup_rows_maps_missing_est_hours_to_none()
    test_read_lookup_rows_rejects_invalid_est_hours()
    print("OK: lookup_reader validates OPUS lookup CSV files.")


if __name__ == "__main__":
    run_test()
