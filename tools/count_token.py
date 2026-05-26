import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


HEADER = [
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


def split_lang_pair(lang_pair):
    try:
        src_lang, tgt_lang = lang_pair.split("-")
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


def load_tokenizer():
    print("Loading tokenizer once for this worker...")
    try:
        # Disabled tokenizers retained for reference.
        # gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
        # o200k_tokenizer = tiktoken.get_encoding("o200k_base")
        # gpt_oss_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
        # llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        # qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V4-Flash", trust_remote_code=True
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


def process_lang_pair(data_dir, lang_pair, tokenizer, tokenizer_batch_size):
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
    total_src_tokens_deepseek = 0
    total_tgt_tokens_deepseek = 0

    for file_path in tqdm(parquet_files, desc=f"{lang_pair}", leave=False):
        try:
            df = pd.read_parquet(file_path, columns=["source_text", "target_text"])
        except Exception as e:
            print(f"Error reading file {file_path}: {e}", file=sys.stderr)
            continue

        src_texts = df["source_text"].fillna("").astype(str).tolist()
        tgt_texts = df["target_text"].fillna("").astype(str).tolist()

        num_lines_in_file = len(src_texts)
        total_lines += num_lines_in_file
        if num_lines_in_file == 0:
            continue

        total_src_tokens_space += sum(len(text.split()) for text in src_texts)
        total_tgt_tokens_space += sum(len(text.split()) for text in tgt_texts)

        total_src_tokens_deepseek += count_tokenizer_tokens(
            tokenizer, src_texts, tokenizer_batch_size
        )
        total_tgt_tokens_deepseek += count_tokenizer_tokens(
            tokenizer, tgt_texts, tokenizer_batch_size
        )

    print(f"Finished {lang_pair}: {total_lines} lines")

    return [
        f"{src_lang}-{tgt_lang}",
        src_lang,
        tgt_lang,
        total_lines,
        total_src_tokens_space,
        total_tgt_tokens_space,
        total_src_tokens_deepseek,
        total_tgt_tokens_deepseek,
    ]


def read_csv_rows(output_file):
    if not output_file.is_file() or output_file.stat().st_size == 0:
        return []

    rows = []
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != HEADER:
            return rows
        rows.extend(row for row in reader if row)
    return rows


def append_rows(output_file, rows):
    if not rows:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists_and_has_content = output_file.is_file() and output_file.stat().st_size > 0

    with output_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists_and_has_content:
            writer.writerow(HEADER)
        writer.writerows(rows)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_file = Path(args.output_file)
    processed_file = Path(args.processed_file) if args.processed_file else output_file

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.tokenizer_batch_size <= 0:
        print("Error: --tokenizer_batch_size must be positive.", file=sys.stderr)
        sys.exit(1)

    requested_pairs = collect_lang_pairs(args)
    processed_pairs = load_processed_pairs(processed_file)
    pending_pairs = [pair for pair in requested_pairs if pair not in processed_pairs]

    print(f"Requested language pairs: {len(requested_pairs)}")
    print(f"Already processed language pairs skipped: {len(requested_pairs) - len(pending_pairs)}")
    print(f"Pending language pairs for this worker: {len(pending_pairs)}")

    if not pending_pairs:
        print("No pending language pairs for this worker.")
        return

    tokenizer = load_tokenizer()
    print("Tokenizer loaded successfully.")

    worker_rows = read_csv_rows(output_file)
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
        )
        if row is not None:
            append_rows(output_file, [row])
            worker_pairs.add(row[0])
            written_rows += 1
            print(f"Checkpointed {row[0]} to worker CSV {output_file}")

    print(f"Wrote {written_rows} result rows to {output_file}")


if __name__ == "__main__":
    main()
