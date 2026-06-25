import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


TEXT_COLUMNS = ["source_text", "target_text"]


DIRECTION_HEADER = [
    "lang_pair",
    "src_lang",
    "tgt_lang",
    "n_lines",
    "n_src_tokens_space",
    "n_tgt_tokens_space",
    # "n_src_tokens_gemma3",
    # "n_tgt_tokens_gemma3",
    # "n_src_tokens_o200kbase",
    # "n_tgt_tokens_o200kbase",
    # "n_src_tokens_gpt_oss",
    # "n_tgt_tokens_gpt_oss",
    # "n_src_tokens_llama3",
    # "n_tgt_tokens_llama3",
    # "n_src_tokens_qwen3",
    # "n_tgt_tokens_qwen3",
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

# Backward-compatible name for older scripts/imports.
HEADER = DIRECTION_HEADER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count lines and tokens in parallel data (Parquet files)."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Root data directory (e.g., './data_dir').",
    )
    parser.add_argument(
        "--lang_pair",
        type=str,
        help="Single language pair to process (e.g., 'en-de').",
    )
    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        help="One or more language pairs to process (e.g., 'en-de fi-en').",
    )
    parser.add_argument(
        "--lang_pairs_file",
        type=str,
        help="Text file containing one language pair per line.",
    )
    parser.add_argument(
        "--parquet_manifest_file",
        type=str,
        help=(
            "TSV manifest with worker_id, lang_pair, and parquet_file columns. "
            "When provided, this script writes one row per parquet file."
        ),
    )
    parser.add_argument(
        "--worker_id",
        type=str,
        help="Worker id to select from --parquet_manifest_file.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--processed_file",
        type=str,
        help=(
            "CSV file used to skip already processed pairs. Defaults to "
            "--output_file. Useful when each worker writes a temporary CSV."
        ),
    )
    parser.add_argument(
        "--tokenizer_batch_size",
        type=int,
        default=1024,
        help="Number of texts to send to the tokenizer per batch.",
    )
    parser.add_argument(
        "--parquet_batch_size",
        type=int,
        default=10_000,
        help="Number of parquet rows to stream into memory at a time.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/DeepSeek-V4-Flash",
        help="Tokenizer name or path (e.g., 'deepseek-ai/DeepSeek-V4-Flash').",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="deepseekv4",
        help="Suffix for the tokenizer token count columns (e.g., 'deepseekv4').",
    )
    return parser.parse_args()


def collect_lang_pairs(args):
    lang_pairs = []

    if args.lang_pair:
        lang_pairs.append(args.lang_pair)

    if args.lang_pairs:
        lang_pairs.extend(args.lang_pairs)

    if args.lang_pairs_file:
        lang_pairs_file = Path(args.lang_pairs_file)
        if not lang_pairs_file.is_file():
            print(
                f"Error: Language-pair list not found: {lang_pairs_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        with lang_pairs_file.open("r", encoding="utf-8") as f:
            lang_pairs.extend(line.strip() for line in f if line.strip())

    deduped = []
    seen = set()
    for lang_pair in lang_pairs:
        if lang_pair not in seen:
            deduped.append(lang_pair)
            seen.add(lang_pair)

    if not deduped:
        print(
            "Error: Provide --lang_pair, --lang_pairs, or --lang_pairs_file.",
            file=sys.stderr,
        )
        sys.exit(1)

    return deduped


def collect_parquet_tasks(manifest_file, worker_id):
    manifest_file = Path(manifest_file)
    if not manifest_file.is_file():
        print(f"Error: Parquet manifest not found: {manifest_file}", file=sys.stderr)
        sys.exit(1)

    if worker_id is None or str(worker_id).strip() == "":
        print("Error: --worker_id is required with --parquet_manifest_file.", file=sys.stderr)
        sys.exit(1)

    tasks = []
    seen = set()
    with manifest_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required_columns = {"worker_id", "lang_pair", "parquet_file"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            print(
                "Error: Parquet manifest must be a TSV with columns: "
                "worker_id, lang_pair, parquet_file.",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            if row.get("worker_id") != str(worker_id):
                continue

            lang_pair = (row.get("lang_pair") or "").strip()
            parquet_file = (row.get("parquet_file") or "").strip()
            task_key = (lang_pair, parquet_file)

            if not lang_pair or not parquet_file or task_key in seen:
                continue

            tasks.append(task_key)
            seen.add(task_key)

    return tasks


def split_lang_pair(lang_pair):
    try:
        src_lang, tgt_lang = lang_pair.split("-", 1)
    except ValueError:
        print(
            f"Warning: Invalid lang_pair format '{lang_pair}'. "
            "Expected 'src-tgt' (e.g., 'en-de'). Skipping.",
            file=sys.stderr,
        )
        return None

    return src_lang, tgt_lang


def load_processed_pairs(processed_file):
    if not processed_file.is_file() or processed_file.stat().st_size == 0:
        print(f"No processed CSV found at {processed_file}.")
        return set()

    print(f"Loading processed language pairs from {processed_file}...")
    try:
        existing_df = pd.read_csv(processed_file)
    except pd.errors.EmptyDataError:
        return set()
    except Exception as e:
        print(
            f"Warning: Could not read processed CSV {processed_file}: {e}. "
            "Continuing without skip list.",
            file=sys.stderr,
        )
        return set()

    if "lang_pair" in existing_df.columns:
        return set(existing_df["lang_pair"].dropna().astype(str))

    if {"src_lang", "tgt_lang"}.issubset(existing_df.columns):
        return set(
            existing_df["src_lang"].astype(str) + "-" + existing_df["tgt_lang"].astype(str)
        )

    print(
        f"Warning: {processed_file} has no lang_pair/src_lang/tgt_lang columns. "
        "Continuing without skip list.",
        file=sys.stderr,
    )
    return set()


def load_tokenizer(tokenizer_path_or_name):
    print(f"Loading tokenizer '{tokenizer_path_or_name}' once for this worker...")
    try:
        # Disabled tokenizers retained for reference.
        # gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
        # o200k_tokenizer = tiktoken.get_encoding("o200k_base")
        # gpt_oss_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
        # llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        # qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return AutoTokenizer.from_pretrained(
            tokenizer_path_or_name, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        print(
            "Please ensure 'transformers', 'sentencepiece', 'torch', 'tiktoken', "
            "'protobuf', and 'accelerate' are installed (see requirements.txt).",
            file=sys.stderr,
        )
        print(
            "Note: Some models may require authentication via "
            "`huggingface-cli login` and/or setting `trust_remote_code=True`.",
            file=sys.stderr,
        )
        sys.exit(1)


def count_tokenizer_tokens(tokenizer, texts, batch_size):
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(batch, add_special_tokens=False)["input_ids"]
        total_tokens += sum(len(tokens) for tokens in tokenized)
    return total_tokens


def count_parquet_file(file_path, tokenizer, tokenizer_batch_size, parquet_batch_size):
    if pq is None:
        print(
            "Error: pyarrow is required for chunked parquet reading. "
            "Install pyarrow in the active environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_lines = 0
    total_src_tokens_space = 0
    total_tgt_tokens_space = 0
    total_src_tokens_model = 0
    total_tgt_tokens_model = 0

    try:
        parquet_file = pq.ParquetFile(file_path)
        batches = parquet_file.iter_batches(
            batch_size=parquet_batch_size,
            columns=TEXT_COLUMNS,
        )

        for batch in batches:
            src_index = batch.schema.get_field_index("source_text")
            tgt_index = batch.schema.get_field_index("target_text")
            if src_index < 0 or tgt_index < 0:
                raise ValueError(
                    "Parquet file is missing required columns: source_text, target_text"
                )

            src_texts = [
                "" if text is None else str(text)
                for text in batch.column(src_index).to_pylist()
            ]
            tgt_texts = [
                "" if text is None else str(text)
                for text in batch.column(tgt_index).to_pylist()
            ]

            n_lines = len(src_texts)
            if n_lines == 0:
                continue

            total_lines += n_lines
            total_src_tokens_space += sum(len(text.split()) for text in src_texts)
            total_tgt_tokens_space += sum(len(text.split()) for text in tgt_texts)

            total_src_tokens_model += count_tokenizer_tokens(
                tokenizer, src_texts, tokenizer_batch_size
            )
            total_tgt_tokens_model += count_tokenizer_tokens(
                tokenizer, tgt_texts, tokenizer_batch_size
            )
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        return None

    return [
        total_lines,
        total_src_tokens_space,
        total_tgt_tokens_space,
        total_src_tokens_model,
        total_tgt_tokens_model,
    ]


def process_lang_pair(
    data_dir,
    lang_pair,
    tokenizer,
    tokenizer_batch_size,
    parquet_batch_size,
):
    parsed = split_lang_pair(lang_pair)
    if parsed is None:
        return None

    src_lang, tgt_lang = parsed
    lang_pair_dir = data_dir / lang_pair

    if not lang_pair_dir.is_dir():
        print(f"Warning: Directory not found: {lang_pair_dir}. Skipping.", file=sys.stderr)
        return None

    parquet_files = sorted(lang_pair_dir.glob("*.parquet"))
    if not parquet_files:
        print(
            f"Warning: No .parquet files found in {lang_pair_dir}. Skipping.",
            file=sys.stderr,
        )
        return None

    print(f"Processing {lang_pair}: {len(parquet_files)} parquet files")

    total_lines = 0
    total_src_tokens_space = 0
    total_tgt_tokens_space = 0
    total_src_tokens_model = 0
    total_tgt_tokens_model = 0

    for file_path in tqdm(parquet_files, desc=f"{lang_pair}", leave=False):
        counts = count_parquet_file(
            file_path,
            tokenizer,
            tokenizer_batch_size,
            parquet_batch_size,
        )
        if counts is None:
            continue

        (
            num_lines_in_file,
            n_src_tokens_space,
            n_tgt_tokens_space,
            n_src_tokens_model,
            n_tgt_tokens_model,
        ) = counts

        total_lines += num_lines_in_file
        total_src_tokens_space += n_src_tokens_space
        total_tgt_tokens_space += n_tgt_tokens_space
        total_src_tokens_model += n_src_tokens_model
        total_tgt_tokens_model += n_tgt_tokens_model

    print(f"Finished {lang_pair}: {total_lines} lines")

    return [
        f"{src_lang}-{tgt_lang}",
        src_lang,
        tgt_lang,
        total_lines,
        total_src_tokens_space,
        total_tgt_tokens_space,
        total_src_tokens_model,
        total_tgt_tokens_model,
    ]


def normalize_parquet_task_path(data_dir, lang_pair, parquet_file):
    parquet_path = Path(parquet_file)
    if parquet_path.is_absolute():
        return parquet_path, parquet_path.as_posix()

    if parquet_path.parts and parquet_path.parts[0] == lang_pair:
        return data_dir / parquet_path, parquet_path.as_posix()

    rel_path = Path(lang_pair) / parquet_path
    return data_dir / rel_path, rel_path.as_posix()


def process_parquet_task(
    data_dir,
    lang_pair,
    parquet_file,
    tokenizer,
    tokenizer_batch_size,
    parquet_batch_size,
):
    parsed = split_lang_pair(lang_pair)
    if parsed is None:
        return None

    src_lang, tgt_lang = parsed
    file_path, output_parquet_file = normalize_parquet_task_path(
        data_dir, lang_pair, parquet_file
    )

    if not file_path.is_file():
        print(f"Warning: Parquet file not found: {file_path}. Skipping.", file=sys.stderr)
        return None

    print(f"Processing {output_parquet_file}")
    counts = count_parquet_file(
        file_path,
        tokenizer,
        tokenizer_batch_size,
        parquet_batch_size,
    )
    if counts is None:
        return None

    return [
        f"{src_lang}-{tgt_lang}",
        src_lang,
        tgt_lang,
        output_parquet_file,
        *counts,
    ]


def read_csv_rows(output_file, expected_header):
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


def load_processed_parquet_files(output_file):
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return set()

    processed = set()
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PARQUET_HEADER:
            return processed

        for row in reader:
            parquet_file = (row.get("parquet_file") or "").strip()
            if parquet_file:
                processed.add(parquet_file)

    return processed


def append_rows(output_file, rows, header):
    if not rows:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists_and_has_content = output_file.is_file() and output_file.stat().st_size > 0

    with output_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists_and_has_content:
            writer.writerow(header)
        writer.writerows(rows)


def run_parquet_manifest_mode(args, data_dir, output_file):
    tasks = collect_parquet_tasks(args.parquet_manifest_file, args.worker_id)
    processed_parquet_files = load_processed_parquet_files(output_file)
    pending_tasks = [
        (lang_pair, parquet_file)
        for lang_pair, parquet_file in tasks
        if normalize_parquet_task_path(data_dir, lang_pair, parquet_file)[1]
        not in processed_parquet_files
    ]

    print(f"Parquet tasks assigned to worker {args.worker_id}: {len(tasks)}")
    print(f"Already checkpointed parquet files skipped: {len(tasks) - len(pending_tasks)}")
    print(f"Pending parquet files for this worker: {len(pending_tasks)}")

    if not pending_tasks:
        print("No pending parquet files for this worker.")
        return

    tokenizer = load_tokenizer(args.tokenizer)
    print("Tokenizer loaded successfully.")

    written_rows = 0
    for lang_pair, parquet_file in pending_tasks:
        row = process_parquet_task(
            data_dir=data_dir,
            lang_pair=lang_pair,
            parquet_file=parquet_file,
            tokenizer=tokenizer,
            tokenizer_batch_size=args.tokenizer_batch_size,
            parquet_batch_size=args.parquet_batch_size,
        )
        if row is not None:
            append_rows(output_file, [row], PARQUET_HEADER)
            processed_parquet_files.add(row[3])
            written_rows += 1
            print(f"Checkpointed {row[3]} to worker CSV {output_file}")

    print(f"Wrote {written_rows} parquet result rows to {output_file}")


def run_direction_mode(args, data_dir, output_file):
    processed_file = Path(args.processed_file) if args.processed_file else output_file
    requested_pairs = collect_lang_pairs(args)
    processed_pairs = load_processed_pairs(processed_file)
    pending_pairs = [pair for pair in requested_pairs if pair not in processed_pairs]

    print(f"Requested language pairs: {len(requested_pairs)}")
    print(f"Already processed language pairs skipped: {len(requested_pairs) - len(pending_pairs)}")
    print(f"Pending language pairs for this worker: {len(pending_pairs)}")

    if not pending_pairs:
        print("No pending language pairs for this worker.")
        return

    tokenizer = load_tokenizer(args.tokenizer)
    print("Tokenizer loaded successfully.")

    worker_rows = read_csv_rows(output_file, DIRECTION_HEADER)
    worker_pairs = {row[0] for row in worker_rows if row}
    written_rows = 0

    for lang_pair in pending_pairs:
        if lang_pair in worker_pairs:
            print(f"Worker CSV already has {lang_pair}. Skipping.")
            continue

        row = process_lang_pair(
            data_dir=data_dir,
            lang_pair=lang_pair,
            tokenizer=tokenizer,
            tokenizer_batch_size=args.tokenizer_batch_size,
            parquet_batch_size=args.parquet_batch_size,
        )
        if row is not None:
            append_rows(output_file, [row], DIRECTION_HEADER)
            worker_pairs.add(row[0])
            written_rows += 1
            print(f"Checkpointed {row[0]} to worker CSV {output_file}")

    print(f"Wrote {written_rows} result rows to {output_file}")


def main():
    args = parse_args()
    
    global DIRECTION_HEADER, PARQUET_HEADER, HEADER
    if args.tokenizer_name:
        DIRECTION_HEADER = [
            "lang_pair",
            "src_lang",
            "tgt_lang",
            "n_lines",
            "n_src_tokens_space",
            "n_tgt_tokens_space",
            f"n_src_tokens_{args.tokenizer_name}",
            f"n_tgt_tokens_{args.tokenizer_name}",
        ]
        PARQUET_HEADER = [
            "lang_pair",
            "src_lang",
            "tgt_lang",
            "parquet_file",
            "n_lines",
            "n_src_tokens_space",
            "n_tgt_tokens_space",
            f"n_src_tokens_{args.tokenizer_name}",
            f"n_tgt_tokens_{args.tokenizer_name}",
        ]
        HEADER = DIRECTION_HEADER

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
        run_parquet_manifest_mode(args, data_dir, output_file)
    else:
        run_direction_mode(args, data_dir, output_file)


if __name__ == "__main__":
    main()
