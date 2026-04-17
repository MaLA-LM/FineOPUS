"""I/O helpers for lookup table generation."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor

REQUIRED_FLORES_COLUMNS = {
    "direction_key",
    "winner",
    "winner_mean",
    "metricx24_mean",
    "qwen3_4b_instruct_2507_mean",
}


def _require_column_set(fieldnames, csv_path):
    """Validate the FLORES lookup schema exactly enough for this script."""
    if fieldnames is None:
        raise ValueError(f"{csv_path} is empty or missing a header row")

    missing = sorted(REQUIRED_FLORES_COLUMNS - set(fieldnames))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"{csv_path} is missing required column(s): {missing_str}"
        )


def _require_value(row, column_name, row_number):
    """Return a required non-empty CSV value or raise a descriptive error."""
    value = row.get(column_name)
    if value in (None, ""):
        direction_key = row.get("direction_key", f"row {row_number}")
        raise ValueError(
            f"Missing value in column '{column_name}' for {direction_key}"
        )
    return value


def load_flores_lookup(csv_path):
    """Load lookup_flores.csv into a dict keyed by FLORES direction."""
    lookup = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        _require_column_set(reader.fieldnames, csv_path)

        for row_number, row in enumerate(reader, start=2):
            direction_key = _require_value(row, "direction_key", row_number)
            lookup[direction_key] = {
                "winner": _require_value(row, "winner", row_number),
                "winner_avg_score": _require_value(row, "winner_mean", row_number),
                "metricx24_mean": _require_value(row, "metricx24_mean", row_number),
                "qwen3_4b_instruct_2507_mean": _require_value(
                    row,
                    "qwen3_4b_instruct_2507_mean",
                    row_number,
                ),
            }
    return lookup


def load_model_runtime(csv_path):
    """Load model_runtime.csv into a dict {model_name: pairs_per_hour}."""
    runtime = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            runtime[row["model"]] = int(row["pairs/hour"])
    return runtime


def scan_opus_directions(base_path):
    """Return a sorted list of immediate sub-directory names under *base_path*."""
    return sorted(entry.name for entry in os.scandir(base_path) if entry.is_dir())


def count_parquet_sentences(dir_path):
    """Count total rows across all parquet files in *dir_path*."""
    import pyarrow.parquet as pq

    total = 0
    for fname in os.listdir(dir_path):
        if fname.endswith(".parquet"):
            total += pq.read_metadata(os.path.join(dir_path, fname)).num_rows
    return total


def count_all_directions(base_path, dir_names, max_workers=32):
    """Count parquet rows for all directions in parallel."""

    def _count(name):
        return name, count_parquet_sentences(os.path.join(base_path, name))

    counts = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for name, num_sentences in pool.map(_count, dir_names):
            counts[name] = num_sentences
    return counts
