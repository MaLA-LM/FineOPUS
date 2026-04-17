"""Read and normalize the OPUS lookup CSV."""

from __future__ import annotations

import csv
from pathlib import Path


def split_direction(direction_key: str) -> tuple[str, str]:
    """Split 'src-tgt' into (src, tgt).

    OPUS lang codes contain underscores (e.g. eng_Latn); the separator
    between the two codes is a single '-'.
    """
    if "-" not in direction_key:
        raise SystemExit(
            f"Unexpected direction_key '{direction_key}' (expected 'src-tgt')"
        )
    src, tgt = direction_key.split("-", 1)
    return src, tgt


def read_lookup_rows(lookup_path: Path) -> list[dict]:
    """Return a list of {direction_key, winner_model, est_hours} dicts."""
    required = ["direction_key", "winner_model", "est_hours"]
    with lookup_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = sorted(set(required) - set(columns))
        if missing:
            raise SystemExit(
                f"lookup CSV missing required columns {missing}; "
                f"present: {columns}"
            )

        rows: list[dict] = []
        for row in reader:
            direction_key = str(row["direction_key"]).strip()
            winner_model = str(row["winner_model"]).strip()
            raw_est_hours = row["est_hours"]

            rows.append(
                {
                    "direction_key": direction_key,
                    "winner_model": winner_model,
                    "est_hours": _parse_est_hours(raw_est_hours),
                }
            )
    return rows


def _parse_est_hours(raw_est_hours: str | None) -> float | None:
    if raw_est_hours is None:
        return None

    est_hours = str(raw_est_hours).strip()
    if not est_hours:
        return None

    try:
        return float(est_hours)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid est_hours value {raw_est_hours!r}; expected a number or blank."
        ) from exc
