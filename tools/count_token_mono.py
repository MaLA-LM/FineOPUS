#!/usr/bin/env python3
"""Count and aggregate token statistics for monolingual Parquet/JSONL files.

Subcommands
-----------
count      Count lines and tokens.  Two modes:
           • language mode  – one row per language folder
           • manifest mode – one row per data file (for chunked SLURM workers)
aggregate  Merge file-level worker CSVs into the main language-level CSV.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


# ---------------------------------------------------------------------------
# Headers & column helpers
# ---------------------------------------------------------------------------

DEFAULT_TEXT_COLUMN = "text"
SUPPORTED_FILE_FORMATS = ("parquet", "jsonl")


def _file_column(file_format: str) -> str:
    return f"{file_format}_file"


def _file_suffix(file_format: str) -> str:
    return f".{file_format}"


def _file_label(file_format: str) -> str:
    return "Parquet" if file_format == "parquet" else "JSONL"


def _lang_header(tokenizer_name: str) -> list[str]:
    return [
        "lang",
        "n_lines",
        "n_tokens_space",
        f"n_tokens_{tokenizer_name}",
    ]


def _file_header(tokenizer_name: str, file_format: str) -> list[str]:
    return [
        "lang",
        _file_column(file_format),
        "n_lines",
        "n_tokens_space",
        f"n_tokens_{tokenizer_name}",
    ]


def _count_columns(tokenizer_name: str) -> list[str]:
    return [
        "n_lines",
        "n_tokens_space",
        f"n_tokens_{tokenizer_name}",
    ]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def clean(value: str | None) -> str:
    return (value or "").strip()


def read_csv_header(path: Path):
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)
    except OSError:
        return None


def read_csv_rows(output_file: Path, expected_header: list[str]) -> list[list[str]]:
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return []
    rows = []
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != expected_header:
            return rows
        rows.extend(row for row in reader if row)
    return rows


def append_rows(output_file: Path, rows: list, header: list[str]):
    if not rows:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_file.is_file() and output_file.stat().st_size > 0
    with output_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


# ---------------------------------------------------------------------------
# count subcommand
# ---------------------------------------------------------------------------

def _parse_count_args(subparsers):
    p = subparsers.add_parser(
        "count",
        help="Count lines and tokens in monolingual data files.",
    )
    p.add_argument("--data_dir", required=True, type=str,
                    help="Root data directory.")
    p.add_argument("--lang", type=str,
                    help="Single language folder (e.g., 'eng_Latn').")
    p.add_argument("--langs", nargs="+",
                    help="One or more language folders.")
    p.add_argument("--langs_file", type=str,
                    help="Text file with one language folder per line.")
    p.add_argument("--parquet_manifest_file", type=str,
                    help="TSV manifest with worker_id, lang, parquet_file.")
    p.add_argument("--jsonl_manifest_file", type=str,
                    help="TSV manifest with worker_id, lang, jsonl_file.")
    p.add_argument("--worker_id", type=str,
                    help="Worker id to select from a manifest file.")
    p.add_argument("--output_file", required=True, type=str,
                    help="Path to the output CSV file.")
    p.add_argument("--processed_file", type=str,
                    help="CSV used to skip already processed languages.")
    p.add_argument("--tokenizer_batch_size", type=int, default=1024,
                    help="Texts per tokenizer batch.")
    p.add_argument("--parquet_batch_size", type=int, default=10_000,
                    help="Parquet rows streamed per batch.")
    p.add_argument("--jsonl_batch_size", type=int, default=10_000,
                    help="JSONL lines streamed per batch.")
    p.add_argument("--file_format", choices=SUPPORTED_FILE_FORMATS,
                    default="parquet",
                    help="Input file format to count (default: parquet).")
    p.add_argument("--tokenizer", type=str,
                    default="Qwen/Qwen3.5-9B",
                    help="Tokenizer name or path.")
    p.add_argument("--tokenizer_name", type=str, default="Qwen3_5",
                    help="Column suffix for tokenizer counts.")
    p.add_argument("--text_column", type=str, default=DEFAULT_TEXT_COLUMN,
                    help="Text column/key name (default: text).")


def collect_langs(args):
    langs = []
    if args.lang:
        langs.append(args.lang)
    if args.langs:
        langs.extend(args.langs)
    if args.langs_file:
        langs_file = Path(args.langs_file)
        if not langs_file.is_file():
            print(f"Error: Language list not found: {langs_file}", file=sys.stderr)
            sys.exit(1)
        with langs_file.open("r", encoding="utf-8") as f:
            langs.extend(line.strip() for line in f if line.strip())

    deduped, seen = [], set()
    for lang in langs:
        if lang not in seen:
            deduped.append(lang)
            seen.add(lang)

    if not deduped:
        print("Error: Provide --lang, --langs, or --langs_file.",
              file=sys.stderr)
        sys.exit(1)
    return deduped


def collect_file_tasks(manifest_file, worker_id, file_format):
    manifest_file = Path(manifest_file)
    if not manifest_file.is_file():
        print(f"Error: {_file_label(file_format)} manifest not found: {manifest_file}",
              file=sys.stderr)
        sys.exit(1)
    if worker_id is None or str(worker_id).strip() == "":
        print("Error: --worker_id is required with a manifest file.",
              file=sys.stderr)
        sys.exit(1)

    tasks, seen = [], set()
    file_col = _file_column(file_format)
    with manifest_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"worker_id", "lang", file_col}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            print("Error: Manifest must be TSV with columns: "
                  f"worker_id, lang, {file_col}.", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            if row.get("worker_id") != str(worker_id):
                continue
            lang = (row.get("lang") or "").strip()
            data_file = (row.get(file_col) or "").strip()
            key = (lang, data_file)
            if not lang or not data_file or key in seen:
                continue
            tasks.append(key)
            seen.add(key)
    return tasks


def load_processed_langs(processed_file: Path) -> set[str]:
    if not processed_file.is_file() or processed_file.stat().st_size == 0:
        print(f"No processed CSV found at {processed_file}.")
        return set()
    print(f"Loading processed languages from {processed_file}...")
    try:
        existing_df = pd.read_csv(processed_file)
    except (pd.errors.EmptyDataError, Exception) as e:
        if not isinstance(e, pd.errors.EmptyDataError):
            print(f"Warning: Could not read {processed_file}: {e}. "
                  "Continuing without skip list.", file=sys.stderr)
        return set()

    if "lang" in existing_df.columns:
        return set(existing_df["lang"].dropna().astype(str))
    print(f"Warning: {processed_file} has no lang column.", file=sys.stderr)
    return set()


def load_tokenizer(tokenizer_path_or_name: str):
    print(f"Loading tokenizer '{tokenizer_path_or_name}'...")
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_path_or_name, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)


def count_tokenizer_tokens(tokenizer, texts: list[str], batch_size: int) -> int:
    total = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(batch, add_special_tokens=False)["input_ids"]
        total += sum(len(t) for t in tokenized)
    return total


def count_parquet_file(file_path, tokenizer, tokenizer_batch_size, parquet_batch_size,
                       text_column=None):
    if text_column is None:
        text_column = DEFAULT_TEXT_COLUMN

    if pq is None:
        print("Error: pyarrow is required. Install it.", file=sys.stderr)
        sys.exit(1)

    total_lines = 0
    total_space = 0
    total_model = 0

    try:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=parquet_batch_size,
                                     columns=[text_column]):
            col_idx = batch.schema.get_field_index(text_column)
            if col_idx < 0:
                raise ValueError(f"Missing required column: {text_column}")

            texts = ["" if t is None else str(t)
                     for t in batch.column(col_idx).to_pylist()]

            n = len(texts)
            if n == 0:
                continue
            total_lines += n
            total_space += sum(len(t.split()) for t in texts)
            total_model += count_tokenizer_tokens(tokenizer, texts,
                                                  tokenizer_batch_size)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

    return [total_lines, total_space, total_model]


def count_jsonl_file(file_path, tokenizer, tokenizer_batch_size, jsonl_batch_size,
                     text_column=None):
    if text_column is None:
        text_column = DEFAULT_TEXT_COLUMN

    total_lines = 0
    total_space = 0
    total_model = 0
    texts = []

    def flush_batch():
        nonlocal total_lines, total_space, total_model, texts
        if not texts:
            return
        total_lines += len(texts)
        total_space += sum(len(t.split()) for t in texts)
        total_model += count_tokenizer_tokens(tokenizer, texts,
                                              tokenizer_batch_size)
        texts = []

    try:
        with Path(file_path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc

                if isinstance(record, dict):
                    value = record.get(text_column)
                else:
                    value = record
                texts.append("" if value is None else str(value))

                if len(texts) >= jsonl_batch_size:
                    flush_batch()
        flush_batch()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

    return [total_lines, total_space, total_model]


def count_data_file(file_path, file_format, tokenizer, tokenizer_batch_size,
                    parquet_batch_size, jsonl_batch_size, text_column=None):
    if file_format == "parquet":
        return count_parquet_file(file_path, tokenizer, tokenizer_batch_size,
                                  parquet_batch_size, text_column)
    if file_format == "jsonl":
        return count_jsonl_file(file_path, tokenizer, tokenizer_batch_size,
                                jsonl_batch_size, text_column)
    raise ValueError(f"Unsupported file format: {file_format}")


def normalize_task_path(data_dir, lang, data_file):
    data_path = Path(data_file)
    if data_path.is_absolute():
        return data_path, data_path.as_posix()
    root_path = data_dir / data_path
    if root_path.is_file():
        return root_path, data_path.as_posix()
    if data_path.parts and data_path.parts[0] == lang:
        return data_dir / data_path, data_path.as_posix()
    rel_path = Path(lang) / data_path
    return data_dir / rel_path, rel_path.as_posix()


def process_lang(data_dir, lang, file_format, tokenizer, tok_bs, pq_bs,
                 jsonl_bs, text_column):
    lang_dir = data_dir / lang
    flat_file = data_dir / f"{lang}{_file_suffix(file_format)}"
    if not lang_dir.is_dir() and not flat_file.is_file():
        print(f"Warning: Directory not found: {lang_dir}. Skipping.", file=sys.stderr)
        return None
    data_files = []
    if lang_dir.is_dir():
        data_files.extend(sorted(lang_dir.glob(f"*{_file_suffix(file_format)}")))
    if flat_file.is_file():
        data_files.append(flat_file)
    if not data_files:
        print(f"Warning: No {_file_suffix(file_format)} files in {lang_dir}. "
              "Skipping.", file=sys.stderr)
        return None

    print(f"Processing {lang}: {len(data_files)} {file_format} files")
    totals = [0, 0, 0]
    for fp in tqdm(data_files, desc=lang, leave=False):
        counts = count_data_file(fp, file_format, tokenizer, tok_bs, pq_bs,
                                 jsonl_bs, text_column)
        if counts is None:
            continue
        for i in range(3):
            totals[i] += counts[i]
    print(f"Finished {lang}: {totals[0]} lines")
    return [lang, *totals]


def process_file_task(data_dir, lang, data_file, file_format, tokenizer, tok_bs,
                      pq_bs, jsonl_bs, text_column):
    fp, output_file = normalize_task_path(data_dir, lang, data_file)
    if not fp.is_file():
        print(f"Warning: {_file_label(file_format)} file not found: {fp}. "
              "Skipping.", file=sys.stderr)
        return None
    print(f"Processing {output_file}")
    counts = count_data_file(fp, file_format, tokenizer, tok_bs, pq_bs,
                             jsonl_bs, text_column)
    if counts is None:
        return None
    return [lang, output_file, *counts]


def load_processed_files(output_file: Path, header: list[str],
                         file_format: str) -> set[str]:
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return set()
    processed = set()
    file_col = _file_column(file_format)
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != header:
            return processed
        for row in reader:
            data_file = (row.get(file_col) or "").strip()
            if data_file:
                processed.add(data_file)
    return processed


def run_manifest_mode(args, data_dir, output_file, file_header):
    manifest_file = (args.jsonl_manifest_file if args.file_format == "jsonl"
                     else args.parquet_manifest_file)
    tasks = collect_file_tasks(manifest_file, args.worker_id, args.file_format)
    processed = load_processed_files(output_file, file_header, args.file_format)
    pending = [
        (lang, data_file) for lang, data_file in tasks
        if normalize_task_path(data_dir, lang, data_file)[1] not in processed
    ]
    print(f"{_file_label(args.file_format)} tasks for worker {args.worker_id}: "
          f"{len(tasks)}")
    print(f"Already checkpointed: {len(tasks) - len(pending)}")
    print(f"Pending: {len(pending)}")
    if not pending:
        print(f"No pending {args.file_format} files.")
        return

    tokenizer = load_tokenizer(args.tokenizer)
    print("Tokenizer loaded successfully.")
    written = 0
    for lang, data_file in pending:
        row = process_file_task(data_dir, lang, data_file, args.file_format,
                                tokenizer, args.tokenizer_batch_size,
                                args.parquet_batch_size, args.jsonl_batch_size,
                                args.text_column)
        if row is not None:
            append_rows(output_file, [row], file_header)
            processed.add(row[1])
            written += 1
            print(f"Checkpointed {row[1]} -> {output_file}")
    print(f"Wrote {written} {args.file_format} rows to {output_file}")


def run_language_mode(args, data_dir, output_file, lang_header):
    processed_file = Path(args.processed_file) if args.processed_file else output_file
    requested = collect_langs(args)
    processed = load_processed_langs(processed_file)
    pending = [lang for lang in requested if lang not in processed]

    print(f"Requested: {len(requested)}")
    print(f"Already processed: {len(requested) - len(pending)}")
    print(f"Pending: {len(pending)}")
    if not pending:
        print("No pending languages.")
        return

    tokenizer = load_tokenizer(args.tokenizer)
    print("Tokenizer loaded successfully.")
    worker_rows = read_csv_rows(output_file, lang_header)
    worker_langs = {row[0] for row in worker_rows if row}
    written = 0
    for lang in pending:
        if lang in worker_langs:
            print(f"Worker CSV already has {lang}. Skipping.")
            continue
        row = process_lang(data_dir, lang, args.file_format, tokenizer,
                           args.tokenizer_batch_size,
                           args.parquet_batch_size, args.jsonl_batch_size,
                           args.text_column)
        if row is not None:
            append_rows(output_file, [row], lang_header)
            worker_langs.add(row[0])
            written += 1
            print(f"Checkpointed {row[0]} → {output_file}")
    print(f"Wrote {written} rows to {output_file}")


def cmd_count(args):
    tok_name = args.tokenizer_name
    lang_header = _lang_header(tok_name)
    file_header = _file_header(tok_name, args.file_format)

    data_dir = Path(args.data_dir)
    output_file = Path(args.output_file)

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    if args.tokenizer_batch_size <= 0:
        print("Error: --tokenizer_batch_size must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.parquet_batch_size <= 0:
        print("Error: --parquet_batch_size must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.jsonl_batch_size <= 0:
        print("Error: --jsonl_batch_size must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.parquet_manifest_file and args.jsonl_manifest_file:
        print("Error: Use only one of --parquet_manifest_file or "
              "--jsonl_manifest_file.", file=sys.stderr)
        sys.exit(1)
    if args.file_format == "jsonl" and args.parquet_manifest_file:
        print("Error: Use --jsonl_manifest_file with --file_format jsonl.",
              file=sys.stderr)
        sys.exit(1)
    if args.file_format == "parquet" and args.jsonl_manifest_file:
        print("Error: Use --parquet_manifest_file with --file_format parquet.",
              file=sys.stderr)
        sys.exit(1)

    if args.parquet_manifest_file or args.jsonl_manifest_file:
        run_manifest_mode(args, data_dir, output_file, file_header)
    else:
        run_language_mode(args, data_dir, output_file, lang_header)


# ---------------------------------------------------------------------------
# aggregate subcommand
# ---------------------------------------------------------------------------

def _parse_aggregate_args(subparsers):
    p = subparsers.add_parser(
        "aggregate",
        help="Merge file-level worker CSVs into language-level CSV.",
    )
    p.add_argument("--data_dir", required=True, help="Root data directory.")
    p.add_argument("--task_root_dir", required=True,
                    help="Directory containing token_count_tasks run_* folders.")
    p.add_argument("--output_file", required=True,
                    help="Main language-level CSV.")
    p.add_argument("--report_dir", required=True,
                    help="Directory for aggregation reports.")
    p.add_argument("--dry_run", action="store_true",
                    help="Write reports but don't append to main CSV.")
    p.add_argument("--file_format", choices=SUPPORTED_FILE_FORMATS,
                    default="parquet",
                    help="Worker file format to aggregate (default: parquet).")
    p.add_argument("--tokenizer_name", type=str, default="Qwen3_5",
                    help="Column suffix for tokenizer counts.")


def _list_worker_csvs(task_root_dir: Path) -> list[Path]:
    if not task_root_dir.is_dir():
        return []
    return sorted(
        p for p in task_root_dir.rglob("*.csv")
        if p.parent.name == "worker_outputs"
    )


def _load_main_completed(output_file: Path) -> set[str]:
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return set()
    completed = set()
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return completed
        for row in reader:
            lang = clean(row.get("lang"))
            if lang:
                completed.add(lang)
    return completed


def _parse_count_value(value, path, row_num, column, bad_rows, file_col):
    value = clean(value)
    try:
        return int(value)
    except ValueError:
        bad_rows.append({
            "source_csv": str(path), "row_number": row_num,
            "reason": f"invalid integer in {column}",
            "lang": "", file_col: "",
        })
        return None


def _normalize_file_id(data_dir, lang, data_file):
    fp = Path(data_file)
    if fp.is_absolute():
        try:
            return fp.relative_to(data_dir).as_posix()
        except ValueError:
            return fp.as_posix()
    if (data_dir / fp).is_file():
        return fp.as_posix()
    if fp.parts and fp.parts[0] == lang:
        return fp.as_posix()
    return (Path(lang) / fp).as_posix()


def _load_file_rows(data_dir, worker_csvs, file_header, count_cols, file_format):
    file_col = _file_column(file_format)
    rows_by_file = {}
    file_sources = defaultdict(list)
    dup_conflicts = []
    dup_identical = []
    bad_rows = []
    file_csvs = []
    skipped_csvs = []

    for path in worker_csvs:
        header = read_csv_header(path)
        if header != file_header:
            skipped_csvs.append(str(path))
            continue
        file_csvs.append(str(path))
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rn, row in enumerate(reader, start=2):
                lang = clean(row.get("lang"))
                data_file = clean(row.get(file_col))
                if not lang or not data_file:
                    bad_rows.append({
                        "source_csv": str(path), "row_number": rn,
                        "reason": f"missing lang or {file_col}",
                        "lang": lang, file_col: data_file,
                    })
                    continue

                counts = {}
                valid = True
                for col in count_cols:
                    v = _parse_count_value(row.get(col), path, rn, col,
                                           bad_rows, file_col)
                    if v is None:
                        valid = False
                        break
                    counts[col] = v
                if not valid:
                    continue

                norm_file = _normalize_file_id(data_dir, lang, data_file)
                norm_row = {
                    "lang": lang,
                    file_col: norm_file,
                    **counts,
                    "source_csv": str(path),
                    "row_number": rn,
                }

                existing = rows_by_file.get(norm_file)
                if existing is None:
                    rows_by_file[norm_file] = norm_row
                    file_sources[norm_file].append(str(path))
                    continue

                file_sources[norm_file].append(str(path))
                cmp_cols = ["lang", *count_cols]
                if all(existing[c] == norm_row[c] for c in cmp_cols):
                    dup_identical.append({
                        file_col: norm_file,
                        "kept_source_csv": existing["source_csv"],
                        "duplicate_source_csv": str(path),
                    })
                else:
                    dup_conflicts.append({
                        file_col: norm_file,
                        "first_source_csv": existing["source_csv"],
                        "conflicting_source_csv": str(path),
                        "first_lang": existing["lang"],
                        "conflicting_lang": norm_row["lang"],
                        "first_counts": "|".join(
                            str(existing[c]) for c in count_cols),
                        "conflicting_counts": "|".join(
                            str(norm_row[c]) for c in count_cols),
                    })

    return {
        "rows_by_file": rows_by_file,
        "file_sources": file_sources,
        "duplicate_conflicts": dup_conflicts,
        "duplicate_identical": dup_identical,
        "bad_rows": bad_rows,
        "file_worker_csvs": file_csvs,
        "skipped_worker_csvs": skipped_csvs,
    }


def _collect_expected_files(data_dir, file_format):
    expected = {}
    no_file_dirs = []
    suffix = _file_suffix(file_format)
    for path in sorted(p for p in data_dir.iterdir()
                       if p.is_file() and p.suffix == suffix):
        expected.setdefault(path.stem, set()).add(path.name)
    for lang_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        lang = lang_dir.name
        files = sorted(
            f"{lang}/{p.name}" for p in lang_dir.iterdir()
            if p.is_file() and p.suffix == suffix
        )
        if files:
            expected[lang] = set(files)
        else:
            no_file_dirs.append(lang)
    return expected, no_file_dirs


def _aggregate_complete(expected_files, completed_main, rows_by_file,
                        conflict_files, count_cols, file_format):
    rows_to_append = []
    added = []
    missing_report = []
    missing_detail = []
    conflict_report = []
    already_done = []
    file_col = _file_column(file_format)

    for lang, exp_files in sorted(expected_files.items()):
        if lang in completed_main:
            already_done.append({
                "lang": lang,
                f"expected_{file_format}_files": len(exp_files),
                "reason": "already present in main CSV",
            })
            continue

        lang_conflicts = sorted(exp_files & conflict_files)
        if lang_conflicts:
            conflict_report.append({
                "lang": lang,
                f"conflicting_{file_format}_files": len(lang_conflicts),
                "examples": "|".join(lang_conflicts[:20]),
            })
            continue

        available = exp_files & rows_by_file.keys()
        missing = sorted(exp_files - available)
        if missing:
            missing_report.append({
                "lang": lang,
                f"expected_{file_format}_files": len(exp_files),
                f"completed_{file_format}_files": len(available),
                f"missing_{file_format}_files": len(missing),
                "examples": "|".join(missing[:20]),
            })
            for data_file in missing:
                missing_detail.append({"lang": lang, f"missing_{file_col}": data_file})
            continue

        totals = {c: 0 for c in count_cols}
        for data_file in sorted(exp_files):
            row = rows_by_file[data_file]
            for c in count_cols:
                totals[c] += row[c]

        out_row = {"lang": lang, **totals}
        rows_to_append.append(out_row)
        added.append({**out_row, f"{file_format}_files": len(exp_files)})

    return {
        "rows_to_append": rows_to_append,
        "added_report": added,
        "missing_report": missing_report,
        "missing_detail": missing_detail,
        "conflict_lang_report": conflict_report,
        "already_completed_report": already_done,
    }


def _collect_orphan_rows(expected_files, rows_by_file, file_format):
    all_expected = set()
    file_col = _file_column(file_format)
    for files in expected_files.values():
        all_expected.update(files)
    orphans = []
    for data_file, row in sorted(rows_by_file.items()):
        if data_file not in all_expected:
            orphans.append({
                "lang": row["lang"], file_col: data_file,
                "source_csv": row["source_csv"],
                "reason": f"{file_format} file not found in current DATA_DIR scan",
            })
    return orphans


def _append_lang_rows(output_file, rows, lang_header):
    if not rows:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_file.is_file() or output_file.stat().st_size == 0
    with output_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=lang_header)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in lang_header})


def cmd_aggregate(args):
    tok_name = args.tokenizer_name
    lang_header = _lang_header(tok_name)
    file_header = _file_header(tok_name, args.file_format)
    count_cols = _count_columns(tok_name)
    file_col = _file_column(args.file_format)

    data_dir = Path(args.data_dir)
    task_root_dir = Path(args.task_root_dir)
    output_file = Path(args.output_file)
    report_dir = Path(args.report_dir)

    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    worker_csvs = _list_worker_csvs(task_root_dir)
    completed_main = _load_main_completed(output_file)
    loaded = _load_file_rows(data_dir, worker_csvs, file_header, count_cols,
                             args.file_format)
    expected_files, no_file_dirs = _collect_expected_files(data_dir,
                                                           args.file_format)
    orphans = _collect_orphan_rows(expected_files, loaded["rows_by_file"],
                                   args.file_format)

    conflict_files = {r[file_col] for r in loaded["duplicate_conflicts"]}
    agg = _aggregate_complete(expected_files, completed_main,
                              loaded["rows_by_file"],
                              conflict_files, count_cols, args.file_format)

    if not args.dry_run:
        _append_lang_rows(output_file, agg["rows_to_append"], lang_header)

    write_csv(report_dir / "complete_languages_added.csv",
              [*lang_header, f"{args.file_format}_files"], agg["added_report"])
    write_csv(report_dir / "missing_languages.csv",
              ["lang", f"expected_{args.file_format}_files",
               f"completed_{args.file_format}_files",
               f"missing_{args.file_format}_files", "examples"],
              agg["missing_report"])
    write_csv(report_dir / f"missing_{args.file_format}_files.csv",
              ["lang", f"missing_{file_col}"], agg["missing_detail"])
    write_csv(report_dir / f"conflicting_duplicate_{args.file_format}_rows.csv",
              [file_col, "first_source_csv", "conflicting_source_csv",
               "first_lang", "conflicting_lang",
               "first_counts", "conflicting_counts"],
              loaded["duplicate_conflicts"])
    write_csv(report_dir / "languages_blocked_by_conflicts.csv",
              ["lang", f"conflicting_{args.file_format}_files", "examples"],
              agg["conflict_lang_report"])
    write_csv(report_dir / "bad_worker_rows.csv",
              ["source_csv", "row_number", "reason", "lang", file_col],
              loaded["bad_rows"])
    write_csv(report_dir / f"orphan_{args.file_format}_rows.csv",
              ["lang", file_col, "source_csv", "reason"], orphans)
    write_csv(report_dir / "already_in_main.csv",
              ["lang", f"expected_{args.file_format}_files", "reason"],
              agg["already_completed_report"])
    write_csv(report_dir / f"languages_without_{args.file_format}_files.csv",
              ["lang"],
              [{"lang": lang} for lang in no_file_dirs])
    write_csv(report_dir / f"identical_duplicate_{args.file_format}_rows.csv",
              [file_col, "kept_source_csv", "duplicate_source_csv"],
              loaded["duplicate_identical"])
    write_text(report_dir / f"{args.file_format}_worker_csvs_used.txt",
               loaded["file_worker_csvs"])
    write_text(report_dir / f"worker_csvs_skipped_non_{args.file_format}.txt",
               loaded["skipped_worker_csvs"])

    summary = [
        f"dry_run: {args.dry_run}",
        f"file_format: {args.file_format}",
        f"worker_csvs_found: {len(worker_csvs)}",
        f"{args.file_format}_worker_csvs_used: {len(loaded['file_worker_csvs'])}",
        f"non_{args.file_format}_worker_csvs_skipped: "
        f"{len(loaded['skipped_worker_csvs'])}",
        f"deduplicated_{args.file_format}_rows: {len(loaded['rows_by_file'])}",
        f"identical_duplicate_{args.file_format}_rows: "
        f"{len(loaded['duplicate_identical'])}",
        f"conflicting_duplicate_{args.file_format}_rows: "
        f"{len(loaded['duplicate_conflicts'])}",
        f"bad_worker_rows: {len(loaded['bad_rows'])}",
        f"orphan_{args.file_format}_rows: {len(orphans)}",
        f"languages_already_in_main: {len(agg['already_completed_report'])}",
        f"languages_added_to_main: {len(agg['added_report']) if not args.dry_run else 0}",
        f"languages_complete_but_not_appended_due_to_dry_run: "
        f"{len(agg['added_report']) if args.dry_run else 0}",
        f"languages_missing_{args.file_format}_rows: {len(agg['missing_report'])}",
        f"languages_blocked_by_conflicts: {len(agg['conflict_lang_report'])}",
        f"languages_without_{args.file_format}_files: {len(no_file_dirs)}",
        f"main_csv: {output_file}",
        f"report_dir: {report_dir}",
    ]
    write_text(report_dir / "summary.txt", summary)
    for line in summary:
        print(line)


# ---------------------------------------------------------------------------
# build-manifest subcommand
# ---------------------------------------------------------------------------

def _parse_build_manifest_args(subparsers):
    p = subparsers.add_parser(
        "build-manifest",
        help="Build a size-balanced file manifest for SLURM workers.",
    )
    p.add_argument("--data_dir", required=True,
                    help="Root data directory.")
    p.add_argument("--main_csv", required=True,
                    help="Main language-level CSV (used to skip done languages).")
    p.add_argument("--task_root_dir", required=True,
                    help="Root dir containing previous run_* worker output folders.")
    p.add_argument("--num_jobs", type=int, required=True,
                    help="Number of SLURM array jobs to distribute work across.")
    p.add_argument("--file_format", choices=SUPPORTED_FILE_FORMATS,
                    default="parquet",
                    help="Input file format to schedule (default: parquet).")
    p.add_argument("--tokenizer_name", type=str, default="Qwen3_5",
                    help="Column suffix for tokenizer counts.")
    p.add_argument("--output_dir", required=True,
                    help="Directory where manifest and reports are written.")


def _scan_files_with_sizes(data_dir: Path, langs: list[str], file_format: str):
    """Return list of (lang, relative_path, size_bytes) for all data files."""
    results = []
    no_file_dirs = []
    suffix = _file_suffix(file_format)
    for lang in langs:
        lang_dir = data_dir / lang
        found = False
        flat_file = data_dir / f"{lang}{suffix}"
        if flat_file.is_file():
            results.append((lang, flat_file.name, flat_file.stat().st_size))
            found = True
        if lang_dir.is_dir():
            for f in sorted(lang_dir.iterdir()):
                if f.is_file() and f.suffix == suffix:
                    rel = f"{lang}/{f.name}"
                    results.append((lang, rel, f.stat().st_size))
                    found = True
        if not found:
            no_file_dirs.append(lang)
    return results, no_file_dirs


def _collect_language_ids(data_dir: Path, file_format: str) -> list[str]:
    suffix = _file_suffix(file_format)
    langs = {d.name for d in data_dir.iterdir() if d.is_dir()}
    langs.update(p.stem for p in data_dir.iterdir()
                 if p.is_file() and p.suffix == suffix)
    return sorted(langs)


def _greedy_bin_pack(items: list[tuple[str, str, int]], num_bins: int):
    import heapq

    if not items:
        return []

    num_bins = min(num_bins, len(items))
    sorted_items = sorted(items, key=lambda x: x[2], reverse=True)

    heap = [(0, i + 1) for i in range(num_bins)]
    heapq.heapify(heap)

    assignments = []
    for lang, pf, size in sorted_items:
        total_size, wid = heapq.heappop(heap)
        assignments.append((wid, lang, pf, size))
        heapq.heappush(heap, (total_size + size, wid))

    assignments.sort(key=lambda x: (x[0], x[2]))
    return assignments


def cmd_build_manifest(args):
    data_dir = Path(args.data_dir)
    main_csv = Path(args.main_csv)
    task_root_dir = Path(args.task_root_dir)
    output_dir = Path(args.output_dir)
    num_jobs = args.num_jobs
    tok_name = args.tokenizer_name
    lang_header = _lang_header(tok_name)
    file_header = _file_header(tok_name, args.file_format)
    file_col = _file_column(args.file_format)

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    if num_jobs <= 0:
        print("Error: --num_jobs must be positive.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    worker_output_dir = output_dir / "worker_outputs"
    worker_output_dir.mkdir(exist_ok=True)

    if not main_csv.is_file() or main_csv.stat().st_size == 0:
        main_csv.parent.mkdir(parents=True, exist_ok=True)
        with main_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(lang_header)

    worker_csvs = []
    if task_root_dir.is_dir():
        worker_csvs = sorted(
            p for p in task_root_dir.rglob("*.csv")
            if p.parent.name == "worker_outputs"
        )
    print(f"Worker CSV files found: {len(worker_csvs)}")

    completed_langs = set()
    with main_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = clean(row.get("lang"))
            if lang:
                completed_langs.add(lang)

    merged = 0
    for wcsv in worker_csvs:
        header = read_csv_header(wcsv)
        if header != lang_header:
            continue
        rows_to_add = []
        with wcsv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = clean(row.get("lang"))
                if lang and lang not in completed_langs:
                    rows_to_add.append([row.get(c, "") for c in lang_header])
                    completed_langs.add(lang)
        if rows_to_add:
            with main_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows_to_add)
            merged += len(rows_to_add)

    print(f"Language-level worker rows merged: {merged}")
    print(f"Completed languages: {len(completed_langs)}")

    completed_parquets = set()
    for wcsv in worker_csvs:
        header = read_csv_header(wcsv)
        if header != file_header:
            continue
        with wcsv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_file = clean(row.get(file_col))
                if data_file:
                    completed_parquets.add(data_file)
    print(f"Completed {args.file_format} files from workers: "
          f"{len(completed_parquets)}")

    all_languages = _collect_language_ids(data_dir, args.file_format)
    incomplete = [lang for lang in all_languages if lang not in completed_langs]
    print(f"Total languages/filesets: {len(all_languages)}")
    print(f"Incomplete languages: {len(incomplete)}")

    print(f"Scanning {args.file_format} files with sizes...")
    all_parquets, no_parquet_dirs = _scan_files_with_sizes(
        data_dir, incomplete, args.file_format
    )
    print(f"Total {args.file_format} files in incomplete languages: "
          f"{len(all_parquets)}")
    print(f"Languages without {args.file_format} files: {len(no_parquet_dirs)}")

    missing = [(lang, pf, sz) for lang, pf, sz in all_parquets
               if pf not in completed_parquets]

    langs_with_parquets = {lang for lang, _, _ in all_parquets}
    langs_with_missing = {lang for lang, _, _ in missing}
    ready_for_agg = langs_with_parquets - langs_with_missing
    print(f"Languages ready for aggregation: {len(ready_for_agg)}")
    print(f"Languages with missing {args.file_format} work: "
          f"{len(langs_with_missing)}")
    print(f"Missing {args.file_format} files to process: {len(missing)}")

    manifest_file = output_dir / f"{args.file_format}_manifest.tsv"
    if not missing:
        with manifest_file.open("w", encoding="utf-8") as f:
            f.write(f"worker_id\tlang\t{file_col}\n")
        print(f"No {args.file_format} files need processing. Nothing to submit.")
        (output_dir / "num_jobs.txt").write_text("0\n")
        return

    actual_jobs = min(num_jobs, len(missing))
    assignments = _greedy_bin_pack(missing, actual_jobs)

    total_size = sum(sz for _, _, sz in missing)
    with manifest_file.open("w", encoding="utf-8") as f:
        f.write(f"worker_id\tlang\t{file_col}\n")
        for wid, lang, pf, _ in assignments:
            f.write(f"{wid}\t{lang}\t{pf}\n")

    worker_sizes = defaultdict(int)
    worker_counts = defaultdict(int)
    for wid, _, _, sz in assignments:
        worker_sizes[wid] += sz
        worker_counts[wid] += 1

    sizes = list(worker_sizes.values())
    avg_size = total_size / actual_jobs if actual_jobs else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0

    print(f"\nManifest: {manifest_file}")
    print(f"Workers: {actual_jobs}")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Avg per worker: {avg_size / 1e9:.2f} GB")
    print(f"Min worker: {min_size / 1e9:.2f} GB ({min(worker_counts.values())} files)")
    print(f"Max worker: {max_size / 1e9:.2f} GB ({max(worker_counts.values())} files)")

    (output_dir / "num_jobs.txt").write_text(f"{actual_jobs}\n")

    write_text(output_dir / "languages_ready_for_aggregation.txt",
               sorted(ready_for_agg))
    write_text(output_dir / f"languages_without_{args.file_format}s.txt",
               sorted(no_parquet_dirs))
    write_text(output_dir / f"languages_with_missing_{args.file_format}s.txt",
               sorted(langs_with_missing))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Count and aggregate token statistics for monolingual "
                    "Parquet/JSONL files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _parse_count_args(subparsers)
    _parse_aggregate_args(subparsers)
    _parse_build_manifest_args(subparsers)

    args = parser.parse_args()
    if args.command == "count":
        cmd_count(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)
    elif args.command == "build-manifest":
        cmd_build_manifest(args)


if __name__ == "__main__":
    main()
