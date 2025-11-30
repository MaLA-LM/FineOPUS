import argparse
import csv
import sys
from pathlib import Path
import pandas as pd
import tiktoken
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    """
    Main function to parse arguments and run the token counting process.
    """
    parser = argparse.ArgumentParser(
        description="Count lines and tokens in parallel data (Parquet files)."
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        help="Root data directory (e.g., './data_dir')"
    )
    parser.add_argument(
        "--lang_pair", 
        type=str, 
        help="Language pair to process, (e.g., 'en-de')"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        help="Path to the output CSV file (e.g., './counts.csv')"
    )
    args = parser.parse_args()

    # --- 1. Setup Paths and Language Info ---
    data_dir = Path(args.data_dir)
    lang_pair_dir = data_dir / args.lang_pair
    output_file = Path(args.output_file)

    if not lang_pair_dir.is_dir():
        print(f"Error: Directory not found: {lang_pair_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        # Assumes column names in Parquet are the language codes
        src_lang, tgt_lang = args.lang_pair.split('-')
    except ValueError:
        print(
            f"Error: Invalid lang_pair format '{args.lang_pair}'. "
            "Expected 'src-tgt' (e.g., 'en-de').", 
            file=sys.stderr
        )
        sys.exit(1)

    # --- Check if lang_pair is already processed ---
    if output_file.is_file():
        print(f"Output file found: {output_file}. Checking for existing data...")
        try:
            # Read the existing CSV
            existing_df = pd.read_csv(output_file)
            
            # Check if the required columns exist
            if "src_lang" in existing_df.columns and "tgt_lang" in existing_df.columns:
                # Check for the specific lang_pair
                is_processed = (
                    (existing_df["src_lang"] == src_lang) &
                    (existing_df["tgt_lang"] == tgt_lang)
                ).any()
                
                if is_processed:
                    print(
                        f"Lang pair {args.lang_pair} (src={src_lang}, tgt={tgt_lang}) "
                        f"already exists in {output_file}."
                    )
                    print("Skipping processing.")
                    sys.exit(0) # Successful exit, no work to do
                else:
                    print(f"Lang pair {args.lang_pair} not found. Proceeding with processing.")
            else:
                print(f"Warning: Output file {output_file} missing 'src_lang' or 'tgt_lang' columns. Will re-process and append.")
        
        except pd.errors.EmptyDataError:
            print(f"Output file {output_file} is empty. Proceeding with processing.")
        except Exception as e:
            print(f"Error reading {output_file}: {e}. Proceeding with processing.")
    else:
        print(f"Output file not found. Will create it.")

    parquet_files = list(lang_pair_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Error: No .parquet files found in {lang_pair_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files in {lang_pair_dir}")
    print(f"Source Language: {src_lang}, Target Language: {tgt_lang}")
    print("Counting tokens for 'source_text' and 'target_text' columns.")

    # --- 2. Initialize Tokenizers ---
    print("Loading tokenizers...")
    try:
        # 1. Whitespace (will be handled by a lambda function)
        
        # 2. Gemma
        gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        # 3. Tiktoken o200k_base
        o200k_tokenizer = tiktoken.get_encoding("o200k_base")

        # 4. openai/gpt-oss-120b
        gpt_oss_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

        # 5. Llama 3
        llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

        # 6. Qwen 3
        qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        # 7. DeepSeek
        deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)

    except Exception as e:
        print(f"Error loading a tokenizer: {e}", file=sys.stderr)
        print(
            "Please ensure 'transformers', 'sentencepiece', 'torch', 'tiktoken', "
            "'protobuf', and 'accelerate' are installed (see requirements.txt).",
            file=sys.stderr
        )
        print(
            "Note: Some models (Llama 3, Qwen 3, DeepSeek) may require "
            "authentication via `huggingface-cli login` and/or "
            "setting `trust_remote_code=True`.",
            file=sys.stderr
        )
        sys.exit(1)
    print("Tokenizers loaded successfully.")

    # --- 3. Initialize Counters ---
    total_lines = 0
    total_src_tokens_space = 0
    total_tgt_tokens_space = 0
    total_src_tokens_gemma = 0
    total_tgt_tokens_gemma = 0
    total_src_tokens_o200k = 0
    total_tgt_tokens_o200k = 0
    total_src_tokens_gpt_oss = 0
    total_tgt_tokens_gpt_oss = 0
    total_src_tokens_llama3 = 0
    total_tgt_tokens_llama3 = 0
    total_src_tokens_qwen3 = 0
    total_tgt_tokens_qwen3 = 0
    total_src_tokens_deepseek = 0
    total_tgt_tokens_deepseek = 0

    # --- 4. Process Files ---
    for file_path in tqdm(parquet_files, desc="Processing files"):
        try:
            df = pd.read_parquet(file_path)

            # Check for expected columns
            if "source_text" not in df.columns or "target_text" not in df.columns:
                print(
                    f"Warning: Skipping {file_path}. "
                    f"Missing columns 'source_text' or 'target_text'.", 
                    file=sys.stderr
                )
                continue
            
            # Ensure text data is string and replace nulls with empty strings
            df["source_text"] = df["source_text"].astype(str).fillna('')
            df["target_text"] = df["target_text"].astype(str).fillna('')

            # Get all texts for batch processing
            src_texts = df["source_text"].tolist()
            tgt_texts = df["target_text"].tolist()
            
            num_lines_in_file = len(src_texts)
            total_lines += num_lines_in_file

            if num_lines_in_file == 0:
                continue # Skip empty files

            # --- Count Tokens ---
            
            # 1. Whitespace
            total_src_tokens_space += df["source_text"].apply(lambda x: len(x.split())).sum()
            total_tgt_tokens_space += df["target_text"].apply(lambda x: len(x.split())).sum()

            # 2. Gemma
            src_gemma_tokens = gemma_tokenizer(
                src_texts, add_special_tokens=False
            )['input_ids']
            tgt_gemma_tokens = gemma_tokenizer(
                tgt_texts, add_special_tokens=False
            )['input_ids']
            total_src_tokens_gemma += sum(len(t) for t in src_gemma_tokens)
            total_tgt_tokens_gemma += sum(len(t) for t in tgt_gemma_tokens)

            # 3. Tiktoken
            src_o200k_tokens = o200k_tokenizer.encode_batch(src_texts)
            tgt_o200k_tokens = o200k_tokenizer.encode_batch(tgt_texts)
            total_src_tokens_o200k += sum(len(t) for t in src_o200k_tokens)
            total_tgt_tokens_o200k += sum(len(t) for t in tgt_o200k_tokens)

            # 4. GPT-OSS
            src_gpt_oss_tokens = gpt_oss_tokenizer(
                src_texts, add_special_tokens=False
            )['input_ids']
            tgt_gpt_oss_tokens = gpt_oss_tokenizer(
                tgt_texts, add_special_tokens=False
            )['input_ids']
            total_src_tokens_gpt_oss += sum(len(t) for t in src_gpt_oss_tokens)
            total_tgt_tokens_gpt_oss += sum(len(t) for t in tgt_gpt_oss_tokens)

            # 5. Llama 3
            src_llama3_tokens = llama3_tokenizer(
                src_texts, add_special_tokens=False
            )['input_ids']
            tgt_llama3_tokens = llama3_tokenizer(
                tgt_texts, add_special_tokens=False
            )['input_ids']
            total_src_tokens_llama3 += sum(len(t) for t in src_llama3_tokens)
            total_tgt_tokens_llama3 += sum(len(t) for t in tgt_llama3_tokens)

            # 6. Qwen 3
            src_qwen3_tokens = qwen3_tokenizer(
                src_texts, add_special_tokens=False
            )['input_ids']
            tgt_qwen3_tokens = qwen3_tokenizer(
                tgt_texts, add_special_tokens=False
            )['input_ids']
            total_src_tokens_qwen3 += sum(len(t) for t in src_qwen3_tokens)
            total_tgt_tokens_qwen3 += sum(len(t) for t in tgt_qwen3_tokens)
            
            # 7. DeepSeek
            src_deepseek_tokens = deepseek_tokenizer(
                src_texts, add_special_tokens=False
            )['input_ids']
            tgt_deepseek_tokens = deepseek_tokenizer(
                tgt_texts, add_special_tokens=False
            )['input_ids']
            total_src_tokens_deepseek += sum(len(t) for t in src_deepseek_tokens)
            total_tgt_tokens_deepseek += sum(len(t) for t in tgt_deepseek_tokens)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)
            continue

    print("Processing complete.")
    print(f"Total lines processed: {total_lines}")

    # --- 5. Write Output ---
    header = [
        "lang_pair", "src_lang", "tgt_lang", "n_lines",
        "n_src_tokens_space", "n_tgt_tokens_space",
        "n_src_tokens_gemma3", "n_tgt_tokens_gemma3",
        "n_src_tokens_o200kbase", "n_tgt_tokens_o200kbase",
        "n_src_tokens_gpt_oss", "n_tgt_tokens_gpt_oss",
        "n_src_tokens_llama3", "n_tgt_tokens_llama3",
        "n_src_tokens_qwen3", "n_tgt_tokens_qwen3",
        "n_src_tokens_deepseekv3", "n_tgt_tokens_deepseekv3"
    ]
    
    data_row = [
        f"{src_lang}-{tgt_lang}", src_lang, tgt_lang, total_lines,
        total_src_tokens_space, total_tgt_tokens_space,
        total_src_tokens_gemma, total_tgt_tokens_gemma,
        total_src_tokens_o200k, total_tgt_tokens_o200k,
        total_src_tokens_gpt_oss, total_tgt_tokens_gpt_oss,
        total_src_tokens_llama3, total_tgt_tokens_llama3,
        total_src_tokens_qwen3, total_tgt_tokens_qwen3,
        total_src_tokens_deepseek, total_tgt_tokens_deepseek
    ]

    try:
        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # --- MODIFIED: Append mode and header check ---
        # Check if file exists and is not empty to decide on writing header
        file_exists_and_has_content = (
            output_file.is_file() and output_file.stat().st_size > 0
        )
        
        with open(output_file, 'a', newline='', encoding='utf-8') as f: # 'a' = append
            writer = csv.writer(f)
            if not file_exists_and_has_content:
                writer.writerow(header)  # Write header only if new or empty file
            writer.writerow(data_row)
            
        print(f"Successfully appended counts to {output_file}")
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()