#!/usr/bin/env python3
"""Count and aggregate token statistics for monolingual Parquet files.

Subcommands
-----------
count      Count lines and tokens.  Two modes:
           • language mode  – one row per language folder
           • parquet-manifest mode – one row per parquet file (for chunked
             SLURM workers)
aggregate  Merge parquet-level worker CSVs into the main language-level CSV.
"""

import argparse
import csv
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


def _lang_header(tokenizer_name: str) -> list[str]:
    return [
        "lang",
        "n_lines",
        "n_tokens_space",
        f"n_tokens_{tokenizer_name}",
    ]


def _parquet_header(tokenizer_name: str) -> list[str]:
    return [
        "lang",
        "parquet_file",
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
        help="Count lines and tokens in monolingual data (Parquet files).",
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
    p.add_argument("--worker_id", type=str,
                    help="Worker id to select from --parquet_manifest_file.")
    p.add_argument("--output_file", required=True, type=str,
                    help="Path to the output CSV file.")
    p.add_argument("--processed_file", type=str,
                    help="CSV used to skip already processed languages.")
    p.add_argument("--tokenizer_batch_size", type=int, default=1024,
                    help="Texts per tokenizer batch.")
    p.add_argument("--parquet_batch_size", type=int, default=10_000,
                    help="Parquet rows streamed per batch.")
    p.add_argument("--tokenizer", type=str,
                    default="Qwen/Qwen3.5-9B",
                    help="Tokenizer name or path.")
    p.add_argument("--tokenizer_name", type=str, default="Qwen3_5",
                    help="Column suffix for tokenizer counts.")
    p.add_argument("--text_column", type=str, default=DEFAULT_TEXT_COLUMN,
                    help="Parquet column name for text (default: text).")


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


def collect_parquet_tasks(manifest_file, worker_id):
    manifest_file = Path(manifest_file)
    if not manifest_file.is_file():
        print(f"Error: Parquet manifest not found: {manifest_file}", file=sys.stderr)
        sys.exit(1)
    if worker_id is None or str(worker_id).strip() == "":
        print("Error: --worker_id is required with --parquet_manifest_file.",
              file=sys.stderr)
        sys.exit(1)

    tasks, seen = [], set()
    with manifest_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"worker_id", "lang", "parquet_file"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            print("Error: Manifest must be TSV with columns: "
                  "worker_id, lang, parquet_file.", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            if row.get("worker_id") != str(worker_id):
                continue
            lang = (row.get("lang") or "").strip()
            pf = (row.get("parquet_file") or "").strip()
            key = (lang, pf)
            if not lang or not pf or key in seen:
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


def normalize_parquet_task_path(data_dir, lang, parquet_file):
    parquet_path = Path(parquet_file)
    if parquet_path.is_absolute():
        return parquet_path, parquet_path.as_posix()
    if parquet_path.parts and parquet_path.parts[0] == lang:
        return data_dir / parquet_path, parquet_path.as_posix()
    rel_path = Path(lang) / parquet_path
    return data_dir / rel_path, rel_path.as_posix()


def process_lang(data_dir, lang, tokenizer, tok_bs, pq_bs, text_column):
    lang_dir = data_dir / lang
    if not lang_dir.is_dir():
        print(f"Warning: Directory not found: {lang_dir}. Skipping.", file=sys.stderr)
        return None
    parquet_files = sorted(lang_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Warning: No .parquet files in {lang_dir}. Skipping.", file=sys.stderr)
        return None

    print(f"Processing {lang}: {len(parquet_files)} parquet files")
    totals = [0, 0, 0]
    for fp in tqdm(parquet_files, desc=lang, leave=False):
        counts = count_parquet_file(fp, tokenizer, tok_bs, pq_bs, text_column)
        if counts is None:
            continue
        for i in range(3):
            totals[i] += counts[i]
    print(f"Finished {lang}: {totals[0]} lines")
    return [lang, *totals]


def process_parquet_task(data_dir, lang, parquet_file, tokenizer, tok_bs, pq_bs,
                         text_column):
    fp, output_pf = normalize_parquet_task_path(data_dir, lang, parquet_file)
    if not fp.is_file():
        print(f"Warning: Parquet file not found: {fp}. Skipping.", file=sys.stderr)
        return None
    print(f"Processing {output_pf}")
    counts = count_parquet_file(fp, tokenizer, tok_bs, pq_bs, text_column)
    if counts is None:
        return None
    return [lang, output_pf, *counts]


def load_processed_parquet_files(output_file: Path, header: list[str]) -> set[str]:
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return set()
    processed = set()
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != header:
            return processed
        for row in reader:
            pf = (row.get("parquet_file") or "").strip()
            if pf:
                processed.add(pf)
    return processed


def run_parquet_manifest_mode(args, data_dir, output_file, pq_header):
    tasks = collect_parquet_tasks(args.parquet_manifest_file, args.worker_id)
    processed = load_processed_parquet_files(output_file, pq_header)
    pending = [
        (lang, pf) for lang, pf in tasks
        if normalize_parquet_task_path(data_dir, lang, pf)[1] not in processed
    ]
    print(f"Parquet tasks for worker {args.worker_id}: {len(tasks)}")
    print(f"Already checkpointed: {len(tasks) - len(pending)}")
    print(f"Pending: {len(pending)}")
    if not pending:
        print("No pending parquet files.")
        return

    tokenizer = load_tokenizer(args.tokenizer)
    print("Tokenizer loaded successfully.")
    written = 0
    for lang, pf in pending:
        row = process_parquet_task(data_dir, lang, pf, tokenizer,
                                   args.tokenizer_batch_size,
                                   args.parquet_batch_size,
                                   args.text_column)
        if row is not None:
            append_rows(output_file, [row], pq_header)
            processed.add(row[1])
            written += 1
            print(f"Checkpointed {row[1]} → {output_file}")
    print(f"Wrote {written} parquet rows to {output_file}")


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
        row = process_lang(data_dir, lang, tokenizer,
                           args.tokenizer_batch_size,
                           args.parquet_batch_size,
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
    pq_header = _parquet_header(tok_name)

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

    if args.parquet_manifest_file:
        run_parquet_manifest_mode(args, data_dir, output_file, pq_header)
    else:
        run_language_mode(args, data_dir, output_file, lang_header)


# ---------------------------------------------------------------------------
# aggregate subcommand
# ---------------------------------------------------------------------------

def _parse_aggregate_args(subparsers):
    p = subparsers.add_parser(
        "aggregate",
        help="Merge parquet-level worker CSVs into language-level CSV.",
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


def _parse_count_value(value, path, row_num, column, bad_rows):
    value = clean(value)
    try:
        return int(value)
    except ValueError:
        bad_rows.append({
            "source_csv": str(path), "row_number": row_num,
            "reason": f"invalid integer in {column}",
            "lang": "", "parquet_file": "",
        })
        return None


def _normalize_parquet_id(data_dir, lang, parquet_file):
    pp = Path(parquet_file)
    if pp.is_absolute():
        try:
            return pp.relative_to(data_dir).as_posix()
        except ValueError:
            return pp.as_posix()
    if pp.parts and pp.parts[0] == lang:
        return pp.as_posix()
    return (Path(lang) / pp).as_posix()


def _load_parquet_rows(data_dir, worker_csvs, pq_header, count_cols):
    rows_by_parquet = {}
    parquet_sources = defaultdict(list)
    dup_conflicts = []
    dup_identical = []
    bad_rows = []
    pq_csvs = []
    skipped_csvs = []

    for path in worker_csvs:
        header = read_csv_header(path)
        if header != pq_header:
            skipped_csvs.append(str(path))
            continue
        pq_csvs.append(str(path))
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rn, row in enumerate(reader, start=2):
                lang = clean(row.get("lang"))
                pf = clean(row.get("parquet_file"))
                if not lang or not pf:
                    bad_rows.append({
                        "source_csv": str(path), "row_number": rn,
                        "reason": "missing lang or parquet_file",
                        "lang": lang, "parquet_file": pf,
                    })
                    continue

                counts = {}
                valid = True
                for col in count_cols:
                    v = _parse_count_value(row.get(col), path, rn, col, bad_rows)
                    if v is None:
                        valid = False
                        break
                    counts[col] = v
                if not valid:
                    continue

                norm_pf = _normalize_parquet_id(data_dir, lang, pf)
                norm_row = {
                    "lang": lang,
                    "parquet_file": norm_pf,
                    **counts,
                    "source_csv": str(path),
                    "row_number": rn,
                }

                existing = rows_by_parquet.get(norm_pf)
                if existing is None:
                    rows_by_parquet[norm_pf] = norm_row
                    parquet_sources[norm_pf].append(str(path))
                    continue

                parquet_sources[norm_pf].append(str(path))
                cmp_cols = ["lang", *count_cols]
                if all(existing[c] == norm_row[c] for c in cmp_cols):
                    dup_identical.append({
                        "parquet_file": norm_pf,
                        "kept_source_csv": existing["source_csv"],
                        "duplicate_source_csv": str(path),
                    })
                else:
                    dup_conflicts.append({
                        "parquet_file": norm_pf,
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
        "rows_by_parquet": rows_by_parquet,
        "parquet_sources": parquet_sources,
        "duplicate_conflicts": dup_conflicts,
        "duplicate_identical": dup_identical,
        "bad_rows": bad_rows,
        "parquet_worker_csvs": pq_csvs,
        "skipped_worker_csvs": skipped_csvs,
    }


def _collect_expected_parquets(data_dir):
    expected = {}
    no_pq_dirs = []
    for lang_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        lang = lang_dir.name
        files = sorted(
            f"{lang}/{p.name}" for p in lang_dir.iterdir()
            if p.is_file() and p.suffix == ".parquet"
        )
        if files:
            expected[lang] = set(files)
        else:
            no_pq_dirs.append(lang)
    return expected, no_pq_dirs


def _aggregate_complete(expected_pq, completed_main, rows_by_pq,
                        conflict_files, count_cols):
    rows_to_append = []
    added = []
    missing_report = []
    missing_detail = []
    conflict_report = []
    already_done = []

    for lang, exp_files in sorted(expected_pq.items()):
        if lang in completed_main:
            already_done.append({
                "lang": lang,
                "expected_parquet_files": len(exp_files),
                "reason": "already present in main CSV",
            })
            continue

        lang_conflicts = sorted(exp_files & conflict_files)
        if lang_conflicts:
            conflict_report.append({
                "lang": lang,
                "conflicting_parquet_files": len(lang_conflicts),
                "examples": "|".join(lang_conflicts[:20]),
            })
            continue

        available = exp_files & rows_by_pq.keys()
        missing = sorted(exp_files - available)
        if missing:
            missing_report.append({
                "lang": lang,
                "expected_parquet_files": len(exp_files),
                "completed_parquet_files": len(available),
                "missing_parquet_files": len(missing),
                "examples": "|".join(missing[:20]),
            })
            for pf in missing:
                missing_detail.append({"lang": lang, "missing_parquet_file": pf})
            continue

        totals = {c: 0 for c in count_cols}
        for pf in sorted(exp_files):
            row = rows_by_pq[pf]
            for c in count_cols:
                totals[c] += row[c]

        out_row = {"lang": lang, **totals}
        rows_to_append.append(out_row)
        added.append({**out_row, "parquet_files": len(exp_files)})

    return {
        "rows_to_append": rows_to_append,
        "added_report": added,
        "missing_report": missing_report,
        "missing_detail": missing_detail,
        "conflict_lang_report": conflict_report,
        "already_completed_report": already_done,
    }


def _collect_orphan_rows(expected_pq, rows_by_pq):
    all_expected = set()
    for files in expected_pq.values():
        all_expected.update(files)
    orphans = []
    for pf, row in sorted(rows_by_pq.items()):
        if pf not in all_expected:
            orphans.append({
                "lang": row["lang"], "parquet_file": pf,
                "source_csv": row["source_csv"],
                "reason": "parquet file not found in current DATA_DIR scan",
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
    pq_header = _parquet_header(tok_name)
    count_cols = _count_columns(tok_name)

    data_dir = Path(args.data_dir)
    task_root_dir = Path(args.task_root_dir)
    output_file = Path(args.output_file)
    report_dir = Path(args.report_dir)

    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    worker_csvs = _list_worker_csvs(task_root_dir)
    completed_main = _load_main_completed(output_file)
    loaded = _load_parquet_rows(data_dir, worker_csvs, pq_header, count_cols)
    expected_pq, no_pq_dirs = _collect_expected_parquets(data_dir)
    orphans = _collect_orphan_rows(expected_pq, loaded["rows_by_parquet"])

    conflict_files = {r["parquet_file"] for r in loaded["duplicate_conflicts"]}
    agg = _aggregate_complete(expected_pq, completed_main,
                              loaded["rows_by_parquet"],
                              conflict_files, count_cols)

    if not args.dry_run:
        _append_lang_rows(output_file, agg["rows_to_append"], lang_header)

    write_csv(report_dir / "complete_languages_added.csv",
              [*lang_header, "parquet_files"], agg["added_report"])
    write_csv(report_dir / "missing_languages.csv",
              ["lang", "expected_parquet_files", "completed_parquet_files",
               "missing_parquet_files", "examples"], agg["missing_report"])
    write_csv(report_dir / "missing_parquet_files.csv",
              ["lang", "missing_parquet_file"], agg["missing_detail"])
    write_csv(report_dir / "conflicting_duplicate_parquet_rows.csv",
              ["parquet_file", "first_source_csv", "conflicting_source_csv",
               "first_lang", "conflicting_lang",
               "first_counts", "conflicting_counts"],
              loaded["duplicate_conflicts"])
    write_csv(report_dir / "languages_blocked_by_conflicts.csv",
              ["lang", "conflicting_parquet_files", "examples"],
              agg["conflict_lang_report"])
    write_csv(report_dir / "bad_worker_rows.csv",
              ["source_csv", "row_number", "reason", "lang", "parquet_file"],
              loaded["bad_rows"])
    write_csv(report_dir / "orphan_parquet_rows.csv",
              ["lang", "parquet_file", "source_csv", "reason"], orphans)
    write_csv(report_dir / "already_in_main.csv",
              ["lang", "expected_parquet_files", "reason"],
              agg["already_completed_report"])
    write_csv(report_dir / "languages_without_parquet_files.csv",
              ["lang"],
              [{"lang": lang} for lang in no_pq_dirs])
    write_csv(report_dir / "identical_duplicate_parquet_rows.csv",
              ["parquet_file", "kept_source_csv", "duplicate_source_csv"],
              loaded["duplicate_identical"])
    write_text(report_dir / "parquet_worker_csvs_used.txt",
               loaded["parquet_worker_csvs"])
    write_text(report_dir / "worker_csvs_skipped_non_parquet.txt",
               loaded["skipped_worker_csvs"])

    summary = [
        f"dry_run: {args.dry_run}",
        f"worker_csvs_found: {len(worker_csvs)}",
        f"parquet_worker_csvs_used: {len(loaded['parquet_worker_csvs'])}",
        f"non_parquet_worker_csvs_skipped: {len(loaded['skipped_worker_csvs'])}",
        f"deduplicated_parquet_rows: {len(loaded['rows_by_parquet'])}",
        f"identical_duplicate_parquet_rows: {len(loaded['duplicate_identical'])}",
        f"conflicting_duplicate_parquet_rows: {len(loaded['duplicate_conflicts'])}",
        f"bad_worker_rows: {len(loaded['bad_rows'])}",
        f"orphan_parquet_rows: {len(orphans)}",
        f"languages_already_in_main: {len(agg['already_completed_report'])}",
        f"languages_added_to_main: {len(agg['added_report']) if not args.dry_run else 0}",
        f"languages_complete_but_not_appended_due_to_dry_run: "
        f"{len(agg['added_report']) if args.dry_run else 0}",
        f"languages_missing_parquet_rows: {len(agg['missing_report'])}",
        f"languages_blocked_by_conflicts: {len(agg['conflict_lang_report'])}",
        f"languages_without_parquet_files: {len(no_pq_dirs)}",
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
        help="Build a size-balanced parquet manifest for SLURM workers.",
    )
    p.add_argument("--data_dir", required=True,
                    help="Root data directory.")
    p.add_argument("--main_csv", required=True,
                    help="Main language-level CSV (used to skip done languages).")
    p.add_argument("--task_root_dir", required=True,
                    help="Root dir containing previous run_* worker output folders.")
    p.add_argument("--num_jobs", type=int, required=True,
                    help="Number of SLURM array jobs to distribute work across.")
    p.add_argument("--tokenizer_name", type=str, default="Qwen3_5",
                    help="Column suffix for tokenizer counts.")
    p.add_argument("--output_dir", required=True,
                    help="Directory where manifest and reports are written.")


def _scan_parquet_files_with_sizes(data_dir: Path, langs: list[str]):
    """Return list of (lang, relative_path, size_bytes) for all parquets."""
    results = []
    no_parquet_dirs = []
    for lang in langs:
        lang_dir = data_dir / lang
        if not lang_dir.is_dir():
            continue
        found = False
        for f in sorted(lang_dir.iterdir()):
            if f.is_file() and f.suffix == ".parquet":
                rel = f"{lang}/{f.name}"
                results.append((lang, rel, f.stat().st_size))
                found = True
        if not found:
            no_parquet_dirs.append(lang)
    return results, no_parquet_dirs


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
    pq_header = _parquet_header(tok_name)

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
        if header != pq_header:
            continue
        with wcsv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pf = clean(row.get("parquet_file"))
                if pf:
                    completed_parquets.add(pf)
    print(f"Completed parquet files from workers: {len(completed_parquets)}")

    all_folders = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir()
    )
    incomplete = [lang for lang in all_folders if lang not in completed_langs]
    print(f"Total folders: {len(all_folders)}")
    print(f"Incomplete languages: {len(incomplete)}")

    print("Scanning parquet files with sizes...")
    all_parquets, no_parquet_dirs = _scan_parquet_files_with_sizes(
        data_dir, incomplete
    )
    print(f"Total parquet files in incomplete languages: {len(all_parquets)}")
    print(f"Languages without parquet files: {len(no_parquet_dirs)}")

    missing = [(lang, pf, sz) for lang, pf, sz in all_parquets
               if pf not in completed_parquets]

    langs_with_parquets = {lang for lang, _, _ in all_parquets}
    langs_with_missing = {lang for lang, _, _ in missing}
    ready_for_agg = langs_with_parquets - langs_with_missing
    print(f"Languages ready for aggregation: {len(ready_for_agg)}")
    print(f"Languages with missing parquet work: {len(langs_with_missing)}")
    print(f"Missing parquet files to process: {len(missing)}")

    manifest_file = output_dir / "parquet_manifest.tsv"
    if not missing:
        with manifest_file.open("w", encoding="utf-8") as f:
            f.write("worker_id\tlang\tparquet_file\n")
        print("No parquet files need processing. Nothing to submit.")
        (output_dir / "num_jobs.txt").write_text("0\n")
        return

    actual_jobs = min(num_jobs, len(missing))
    assignments = _greedy_bin_pack(missing, actual_jobs)

    total_size = sum(sz for _, _, sz in missing)
    with manifest_file.open("w", encoding="utf-8") as f:
        f.write("worker_id\tlang\tparquet_file\n")
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
    write_text(output_dir / "languages_without_parquets.txt",
               sorted(no_parquet_dirs))
    write_text(output_dir / "languages_with_missing_parquets.txt",
               sorted(langs_with_missing))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Count and aggregate token statistics for monolingual Parquet files.",
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
