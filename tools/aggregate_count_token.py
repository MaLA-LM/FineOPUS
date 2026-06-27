import argparse
import csv
from collections import defaultdict
from pathlib import Path


DIRECTION_HEADER = [
    "lang_pair",
    "src_lang",
    "tgt_lang",
    "n_lines",
    "n_src_tokens_space",
    "n_tgt_tokens_space",
    "n_src_tokens_deepseekv4",
    "n_tgt_tokens_deepseekv4",
]

PARQUET_HEADER = [
    "lang_pair",
    "src_lang",
    "tgt_lang",
    "parquet_file",
    "n_lines",
    "n_src_tokens_space",
    "n_tgt_tokens_space",
    "n_src_tokens_deepseekv4",
    "n_tgt_tokens_deepseekv4",
]

COUNT_COLUMNS = [
    "n_lines",
    "n_src_tokens_space",
    "n_tgt_tokens_space",
    "n_src_tokens_deepseekv4",
    "n_tgt_tokens_deepseekv4",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate parquet-level token-count worker CSVs into the main "
            "direction-level CSV."
        )
    )
    parser.add_argument("--data_dir", required=True, help="Root data directory.")
    parser.add_argument(
        "--task_root_dir",
        required=True,
        help="Directory containing token_count_tasks run_* folders.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Main direction-level CSV to append completed directions to.",
    )
    parser.add_argument(
        "--report_dir",
        required=True,
        help="Directory where aggregation reports will be written.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write reports but do not append rows to the main CSV.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen3_5",
        help="Suffix for the tokenizer token count columns (e.g., 'Qwen3_5').",
    )
    return parser.parse_args()


def read_header(path):
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)
    except OSError:
        return None


def clean(value):
    return (value or "").strip()


def split_lang_pair(lang_pair):
    parts = lang_pair.split("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def normalize_parquet_id(data_dir, lang_pair, parquet_file):
    parquet_path = Path(parquet_file)
    if parquet_path.is_absolute():
        try:
            return parquet_path.relative_to(data_dir).as_posix()
        except ValueError:
            return parquet_path.as_posix()

    if parquet_path.parts and parquet_path.parts[0] == lang_pair:
        return parquet_path.as_posix()

    return (Path(lang_pair) / parquet_path).as_posix()


def list_worker_csvs(task_root_dir):
    if not task_root_dir.is_dir():
        return []
    return sorted(
        path
        for path in task_root_dir.rglob("*.csv")
        if path.parent.name == "worker_outputs"
    )


def load_main_completed(output_file):
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return set()

    completed = set()
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return completed

        for row in reader:
            lang_pair = clean(row.get("lang_pair"))
            if not lang_pair:
                src_lang = clean(row.get("src_lang"))
                tgt_lang = clean(row.get("tgt_lang"))
                if src_lang and tgt_lang:
                    lang_pair = f"{src_lang}-{tgt_lang}"
            if lang_pair:
                completed.add(lang_pair)

    return completed


def parse_count(value, path, row_number, column, bad_rows):
    value = clean(value)
    try:
        return int(value)
    except ValueError:
        bad_rows.append(
            {
                "source_csv": str(path),
                "row_number": row_number,
                "reason": f"invalid integer in {column}",
                "lang_pair": "",
                "parquet_file": "",
            }
        )
        return None


def load_parquet_rows(data_dir, worker_csvs):
    rows_by_parquet = {}
    parquet_sources = defaultdict(list)
    duplicate_conflicts = []
    duplicate_identical = []
    bad_rows = []
    parquet_worker_csvs = []
    skipped_worker_csvs = []

    for path in worker_csvs:
        header = read_header(path)
        if header != PARQUET_HEADER:
            skipped_worker_csvs.append(str(path))
            continue

        parquet_worker_csvs.append(str(path))
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_number, row in enumerate(reader, start=2):
                lang_pair = clean(row.get("lang_pair"))
                parquet_file = clean(row.get("parquet_file"))
                if not lang_pair or not parquet_file:
                    bad_rows.append(
                        {
                            "source_csv": str(path),
                            "row_number": row_number,
                            "reason": "missing lang_pair or parquet_file",
                            "lang_pair": lang_pair,
                            "parquet_file": parquet_file,
                        }
                    )
                    continue

                parsed = split_lang_pair(lang_pair)
                if parsed is None:
                    bad_rows.append(
                        {
                            "source_csv": str(path),
                            "row_number": row_number,
                            "reason": "invalid lang_pair",
                            "lang_pair": lang_pair,
                            "parquet_file": parquet_file,
                        }
                    )
                    continue

                counts = {}
                valid_counts = True
                for column in COUNT_COLUMNS:
                    count_value = parse_count(row.get(column), path, row_number, column, bad_rows)
                    if count_value is None:
                        valid_counts = False
                        break
                    counts[column] = count_value

                if not valid_counts:
                    continue

                normalized_parquet = normalize_parquet_id(
                    data_dir, lang_pair, parquet_file
                )
                normalized_row = {
                    "lang_pair": lang_pair,
                    "src_lang": clean(row.get("src_lang")) or parsed[0],
                    "tgt_lang": clean(row.get("tgt_lang")) or parsed[1],
                    "parquet_file": normalized_parquet,
                    **counts,
                    "source_csv": str(path),
                    "row_number": row_number,
                }

                existing = rows_by_parquet.get(normalized_parquet)
                if existing is None:
                    rows_by_parquet[normalized_parquet] = normalized_row
                    parquet_sources[normalized_parquet].append(str(path))
                    continue

                parquet_sources[normalized_parquet].append(str(path))
                comparable_columns = ["lang_pair", *COUNT_COLUMNS]
                if all(existing[column] == normalized_row[column] for column in comparable_columns):
                    duplicate_identical.append(
                        {
                            "parquet_file": normalized_parquet,
                            "kept_source_csv": existing["source_csv"],
                            "duplicate_source_csv": str(path),
                        }
                    )
                else:
                    duplicate_conflicts.append(
                        {
                            "parquet_file": normalized_parquet,
                            "first_source_csv": existing["source_csv"],
                            "conflicting_source_csv": str(path),
                            "first_lang_pair": existing["lang_pair"],
                            "conflicting_lang_pair": normalized_row["lang_pair"],
                            "first_counts": "|".join(
                                str(existing[column]) for column in COUNT_COLUMNS
                            ),
                            "conflicting_counts": "|".join(
                                str(normalized_row[column]) for column in COUNT_COLUMNS
                            ),
                        }
                    )

    return {
        "rows_by_parquet": rows_by_parquet,
        "parquet_sources": parquet_sources,
        "duplicate_conflicts": duplicate_conflicts,
        "duplicate_identical": duplicate_identical,
        "bad_rows": bad_rows,
        "parquet_worker_csvs": parquet_worker_csvs,
        "skipped_worker_csvs": skipped_worker_csvs,
    }


def collect_expected_parquets(data_dir):
    expected = {}
    no_parquet_dirs = []

    for lang_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        lang_pair = lang_dir.name
        parquet_files = sorted(
            f"{lang_pair}/{path.name}"
            for path in lang_dir.iterdir()
            if path.is_file() and path.suffix == ".parquet"
        )
        if parquet_files:
            expected[lang_pair] = set(parquet_files)
        else:
            no_parquet_dirs.append(lang_pair)

    return expected, no_parquet_dirs


def aggregate_complete_directions(expected_parquets, completed_main, rows_by_parquet, conflict_files):
    rows_to_append = []
    added_report = []
    missing_report = []
    missing_detail = []
    conflict_direction_report = []
    already_completed_report = []

    for lang_pair, expected_files in sorted(expected_parquets.items()):
        if lang_pair in completed_main:
            already_completed_report.append(
                {
                    "lang_pair": lang_pair,
                    "expected_parquet_files": len(expected_files),
                    "reason": "already present in main CSV",
                }
            )
            continue

        direction_conflicts = sorted(expected_files & conflict_files)
        if direction_conflicts:
            conflict_direction_report.append(
                {
                    "lang_pair": lang_pair,
                    "conflicting_parquet_files": len(direction_conflicts),
                    "examples": "|".join(direction_conflicts[:20]),
                }
            )
            continue

        available_files = expected_files & rows_by_parquet.keys()
        missing_files = sorted(expected_files - available_files)
        if missing_files:
            missing_report.append(
                {
                    "lang_pair": lang_pair,
                    "expected_parquet_files": len(expected_files),
                    "completed_parquet_files": len(available_files),
                    "missing_parquet_files": len(missing_files),
                    "examples": "|".join(missing_files[:20]),
                }
            )
            for parquet_file in missing_files:
                missing_detail.append(
                    {
                        "lang_pair": lang_pair,
                        "missing_parquet_file": parquet_file,
                    }
                )
            continue

        parsed = split_lang_pair(lang_pair)
        if parsed is None:
            missing_report.append(
                {
                    "lang_pair": lang_pair,
                    "expected_parquet_files": len(expected_files),
                    "completed_parquet_files": len(available_files),
                    "missing_parquet_files": 0,
                    "examples": "invalid lang_pair",
                }
            )
            continue

        totals = {column: 0 for column in COUNT_COLUMNS}
        for parquet_file in sorted(expected_files):
            row = rows_by_parquet[parquet_file]
            for column in COUNT_COLUMNS:
                totals[column] += row[column]

        output_row = {
            "lang_pair": lang_pair,
            "src_lang": parsed[0],
            "tgt_lang": parsed[1],
            **totals,
        }
        rows_to_append.append(output_row)
        added_report.append(
            {
                **output_row,
                "parquet_files": len(expected_files),
            }
        )

    return {
        "rows_to_append": rows_to_append,
        "added_report": added_report,
        "missing_report": missing_report,
        "missing_detail": missing_detail,
        "conflict_direction_report": conflict_direction_report,
        "already_completed_report": already_completed_report,
    }


def collect_orphan_rows(expected_parquets, rows_by_parquet):
    expected_files = set()
    for parquet_files in expected_parquets.values():
        expected_files.update(parquet_files)

    orphan_rows = []
    for parquet_file, row in sorted(rows_by_parquet.items()):
        if parquet_file in expected_files:
            continue
        orphan_rows.append(
            {
                "lang_pair": row["lang_pair"],
                "parquet_file": parquet_file,
                "source_csv": row["source_csv"],
                "reason": "parquet file not found in current DATA_DIR scan",
            }
        )

    return orphan_rows


def append_direction_rows(output_file, rows):
    if not rows:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_file.is_file() or output_file.stat().st_size == 0

    with output_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIRECTION_HEADER)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in DIRECTION_HEADER})


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main():
    args = parse_args()
    
    global DIRECTION_HEADER, PARQUET_HEADER, COUNT_COLUMNS
    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
        DIRECTION_HEADER = [
            "lang_pair",
            "src_lang",
            "tgt_lang",
            "n_lines",
            "n_src_tokens_space",
            "n_tgt_tokens_space",
            f"n_src_tokens_{tokenizer_name}",
            f"n_tgt_tokens_{tokenizer_name}",
        ]
        PARQUET_HEADER = [
            "lang_pair",
            "src_lang",
            "tgt_lang",
            "parquet_file",
            "n_lines",
            "n_src_tokens_space",
            "n_tgt_tokens_space",
            f"n_src_tokens_{tokenizer_name}",
            f"n_tgt_tokens_{tokenizer_name}",
        ]
        COUNT_COLUMNS = [
            "n_lines",
            "n_src_tokens_space",
            "n_tgt_tokens_space",
            f"n_src_tokens_{tokenizer_name}",
            f"n_tgt_tokens_{tokenizer_name}",
        ]

    data_dir = Path(args.data_dir)
    task_root_dir = Path(args.task_root_dir)
    output_file = Path(args.output_file)
    report_dir = Path(args.report_dir)

    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    worker_csvs = list_worker_csvs(task_root_dir)
    completed_main = load_main_completed(output_file)
    loaded = load_parquet_rows(data_dir, worker_csvs)
    expected_parquets, no_parquet_dirs = collect_expected_parquets(data_dir)
    orphan_rows = collect_orphan_rows(expected_parquets, loaded["rows_by_parquet"])

    conflict_files = {row["parquet_file"] for row in loaded["duplicate_conflicts"]}
    aggregation = aggregate_complete_directions(
        expected_parquets=expected_parquets,
        completed_main=completed_main,
        rows_by_parquet=loaded["rows_by_parquet"],
        conflict_files=conflict_files,
    )

    if not args.dry_run:
        append_direction_rows(output_file, aggregation["rows_to_append"])

    write_csv(
        report_dir / "complete_directions_added.csv",
        [*DIRECTION_HEADER, "parquet_files"],
        aggregation["added_report"],
    )
    write_csv(
        report_dir / "missing_directions.csv",
        [
            "lang_pair",
            "expected_parquet_files",
            "completed_parquet_files",
            "missing_parquet_files",
            "examples",
        ],
        aggregation["missing_report"],
    )
    write_csv(
        report_dir / "missing_parquet_files.csv",
        ["lang_pair", "missing_parquet_file"],
        aggregation["missing_detail"],
    )
    write_csv(
        report_dir / "conflicting_duplicate_parquet_rows.csv",
        [
            "parquet_file",
            "first_source_csv",
            "conflicting_source_csv",
            "first_lang_pair",
            "conflicting_lang_pair",
            "first_counts",
            "conflicting_counts",
        ],
        loaded["duplicate_conflicts"],
    )
    write_csv(
        report_dir / "directions_blocked_by_conflicts.csv",
        ["lang_pair", "conflicting_parquet_files", "examples"],
        aggregation["conflict_direction_report"],
    )
    write_csv(
        report_dir / "bad_worker_rows.csv",
        ["source_csv", "row_number", "reason", "lang_pair", "parquet_file"],
        loaded["bad_rows"],
    )
    write_csv(
        report_dir / "orphan_parquet_rows.csv",
        ["lang_pair", "parquet_file", "source_csv", "reason"],
        orphan_rows,
    )
    write_csv(
        report_dir / "already_in_main.csv",
        ["lang_pair", "expected_parquet_files", "reason"],
        aggregation["already_completed_report"],
    )
    write_csv(
        report_dir / "directions_without_parquet_files.csv",
        ["lang_pair"],
        [{"lang_pair": lang_pair} for lang_pair in no_parquet_dirs],
    )
    write_csv(
        report_dir / "identical_duplicate_parquet_rows.csv",
        ["parquet_file", "kept_source_csv", "duplicate_source_csv"],
        loaded["duplicate_identical"],
    )
    write_text(report_dir / "parquet_worker_csvs_used.txt", loaded["parquet_worker_csvs"])
    write_text(report_dir / "worker_csvs_skipped_non_parquet.txt", loaded["skipped_worker_csvs"])

    summary_lines = [
        f"dry_run: {args.dry_run}",
        f"worker_csvs_found: {len(worker_csvs)}",
        f"parquet_worker_csvs_used: {len(loaded['parquet_worker_csvs'])}",
        f"non_parquet_worker_csvs_skipped: {len(loaded['skipped_worker_csvs'])}",
        f"deduplicated_parquet_rows: {len(loaded['rows_by_parquet'])}",
        f"identical_duplicate_parquet_rows: {len(loaded['duplicate_identical'])}",
        f"conflicting_duplicate_parquet_rows: {len(loaded['duplicate_conflicts'])}",
        f"bad_worker_rows: {len(loaded['bad_rows'])}",
        f"orphan_parquet_rows: {len(orphan_rows)}",
        f"directions_already_in_main: {len(aggregation['already_completed_report'])}",
        f"directions_added_to_main: {len(aggregation['added_report']) if not args.dry_run else 0}",
        f"directions_complete_but_not_appended_due_to_dry_run: {len(aggregation['added_report']) if args.dry_run else 0}",
        f"directions_missing_parquet_rows: {len(aggregation['missing_report'])}",
        f"directions_blocked_by_conflicts: {len(aggregation['conflict_direction_report'])}",
        f"directions_without_parquet_files: {len(no_parquet_dirs)}",
        f"main_csv: {output_file}",
        f"report_dir: {report_dir}",
    ]
    write_text(report_dir / "summary.txt", summary_lines)

    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
