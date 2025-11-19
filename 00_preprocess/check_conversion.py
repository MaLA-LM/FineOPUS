import os
import argparse
import re
import sys
import logging
from pathlib import Path
import io
import shutil
import pyarrow.parquet as pq
from GlotScript import sp

# Regex to validate language code: 3 lowercase letters, underscore, 1 uppercase, 3 lowercase
LANG_CODE_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
DEFAULT_LANG_CODE = "und_Zyyy"
MAX_LINES = 100

# Regex to find 3-letter lowercase codes
THREE_LETTER_RE = re.compile(r"^[a-z]{3}$")

# Map for specific, non-standard codes
CODE_MAP = {
    "eng-simple": "eng",
    "eng-simple_Latn": "eng_Latn",
    "cn": "zho_Hani",
    "da": "dan_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "fr": "fra_Latn",
    "jp": "jpn_Jpan",
    "nl": "nld_Latn",
    "pt": "por_Latn",
    "zh_Tw": "zho_Hant",
}

def detect_script_from_parquets(
    data_dir: Path, 
    lang_code: str, 
    column_name: str
) -> str:
    """
    Detects the script from up to MAX_LINES of a specific column
    across parquet files. Uses GlotScript for detection.
    """

    logging.info(f"Running GlotScript detection for '{lang_code}' in {data_dir.name} "
                 f"using column '{column_name}'")
    print(f"  > Running 'script detection' for '{lang_code}' in {data_dir.name} "
          f"(reading column '{column_name}')...")

    all_lines = []
    try:
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"  > Warning: No .parquet files found in {data_dir.name}.")
            return "Zyyy"

        for pf_name in parquet_files:
            pf = pq.ParquetFile(pf_name)

            # Skip empty files
            if pf.metadata.num_row_groups == 0:
                print(f"  > Skipping empty file: {pf_name.name}")
                continue
            
            # Read the entire column from the first row group
            column_data_table = pf.read_row_group(0, columns=[column_name])
            
            print(f"  > Reading {column_data_table.num_rows} lines from {pf_name.name}...")

            # Get the actual data array from the table
            rows_data_array = column_data_table[0]
        
            # Convert to Python list and filter out None or empty values
            text_lines = [str(line) for line in rows_data_array.to_pylist() if line]
            
            all_lines.extend(text_lines)
            
            if len(all_lines) >= MAX_LINES:
                print(f"  > Reached {len(all_lines)} total lines (limit is {MAX_LINES}). Stopping read.")
                break  # Stop if we've gathered enough lines

    except Exception as e:
        print(f"  > Error reading parquet file in {data_dir.name}: {e}", file=sys.stderr)
        return "Zyyy"

    if all_lines:
        try:
            # Join only the lines we need (up to MAX_LINES) with a space
            text_to_check = " ".join(all_lines[:MAX_LINES])
            
            script_code = sp(text_to_check)[0]
            print(f"  > GlotScript detected script: {script_code}")
            return script_code
        except Exception as e:
            print(f"  > Error during GlotScript detection: {e}", file=sys.stderr)
            return "Zyyy"


def normalize_and_validate_code(
    code: str, 
    lang_pair_dir: Path,
    column_name_to_read: str
) -> str:
    """
    Normalizes and validates a language code based on specific rules.
    """
    if LANG_CODE_RE.match(code):
        return code
        
    original_code = code

    if code in CODE_MAP:
        code = CODE_MAP[code]
        if LANG_CODE_RE.match(code):
            return code

    if THREE_LETTER_RE.match(code):
        script = detect_script_from_parquets(lang_pair_dir, code, column_name_to_read)
        normalized_code = f"{code}_{script}"
        
        if LANG_CODE_RE.match(normalized_code):
            return normalized_code
        else:
            logging.warning(f"Script detection for '{original_code}' in {lang_pair_dir.name} "
                            f"produced invalid code: '{normalized_code}'.")
            return DEFAULT_LANG_CODE

    return DEFAULT_LANG_CODE


def process_lang_pair(
    lang_pair_dir: Path, 
    src_lang: str, 
    tgt_lang: str, 
    mismatch_logger: logging.Logger
):
    """
    Processes a single language pair directory.
    Normalizes codes, logs discrepancies, and moves files if needed.
    """
    # 1. Normalize and validate language codes
    valid_src_lang = normalize_and_validate_code(src_lang, lang_pair_dir, column_name_to_read="source_text")
    valid_tgt_lang = normalize_and_validate_code(tgt_lang, lang_pair_dir, column_name_to_read="target_text")
    
    original_pair_name = lang_pair_dir.name
    valid_pair_name = f"{valid_src_lang}-{valid_tgt_lang}"

    # 2. Check for and log mismatches
    if original_pair_name != valid_pair_name:
        message = (
            f"Name mismatch found in: {lang_pair_dir}\n"
            f"  Original: {original_pair_name}\n"
            f"  Validated: {valid_pair_name}\n"
        )
        print(f"  Warning: {message}", file=sys.stderr)
        mismatch_logger.warning(message)
        
        # Move .parquet files and delete old directory
        new_dir_path = lang_pair_dir.parent / valid_pair_name
        
        try:
            # 1. Create the new (valid) directory
            new_dir_path.mkdir(parents=True, exist_ok=True)
            
            # 2. Find all parquet files in the old (original) directory
            parquet_files_to_move = list(lang_pair_dir.glob("*.parquet"))
            
            if not parquet_files_to_move:
                print(f"  > Warning: No .parquet files found in {lang_pair_dir.name}.")
            
            files_moved = 0
            for file_path in parquet_files_to_move:
                
                # Conflict handling logic
                target_path = new_dir_path / file_path.name
                counter = 1
                
                # Keep generating new names until we find one that doesn't exist
                while target_path.exists():
                    original_stem = file_path.stem 
                    original_suffix = file_path.suffix
                    
                    # Create a new name like 'train_conflict_1.parquet'
                    new_name = f"{original_stem}_conflict_{counter}{original_suffix}"
                    target_path = new_dir_path / new_name
                    counter += 1
                
                # Move the file to the (now guaranteed unique) target path
                shutil.move(str(file_path), str(target_path))
                
                if counter > 1: # This means a rename occurred
                    warn_msg = (f"  > Warning: File {file_path.name} already existed in destination. "
                                f"Renamed to {target_path.name}.")
                    print(warn_msg)
                    mismatch_logger.warning(f"Conflict: Renamed {file_path.name} to {target_path.name} in {new_dir_path.name}")
                else:
                    print(f"  > Moved {file_path.name} to {new_dir_path.name}")
                    
                files_moved += 1
            
            print(f"  > Moved {files_moved} file(s).")
            
            # 3. Remove the old (original) directory and any remaining files
            print(f"  > Removing original directory {lang_pair_dir.name}...")
            shutil.rmtree(lang_pair_dir)
            print(f"  > Remove complete.")

        except Exception as e:
            err_msg = (f"  > Error: Failed to move files/remove directory for "
                       f"{lang_pair_dir.name} due to: {e}. Manual cleanup may be required.")
            print(err_msg, file=sys.stderr)
            mismatch_logger.error(err_msg)

    else:
        print(f"  {original_pair_name} (OK)")


def process_data_dir(
    data_dir: Path, 
    skip_logger: logging.Logger, 
    mismatch_logger: logging.Logger
):
    """
    Finds all language pair directories in data_dir and processes them.
    """
    # Get the list of directories *before* starting
    lang_pair_dirs = [d for d in data_dir.glob("*-*") if d.is_dir()]
    if not lang_pair_dirs:
        print(f"No language pair directories (e.g., 'eng_Latn-fra_Latn') found in {data_dir}", file=sys.stderr)
        return

    total_pairs = len(lang_pair_dirs)
    print(f"Found {total_pairs} language pair directories to check...")
    
    for i, lang_pair_dir in enumerate(lang_pair_dirs):
        progress = (i + 1) / total_pairs * 100
        
        # Check if directory still exists. It might have been deleted
        if not lang_pair_dir.exists():
            print(f"Skipping {lang_pair_dir.name} (directory no longer exists, "
                  "likely already processed and moved)")
            continue
            
        print(f"Processing {lang_pair_dir.name} ({i+1}/{total_pairs}, {progress:.2f}%)")
        
        pair_name = lang_pair_dir.name
        try:
            src_lang, tgt_lang = pair_name.split('-', 1)
            if "eng-simple_Latn" in pair_name:
                print(f"  > Special handling for 'eng-simple_Latn' in {pair_name}")
                # If "eng-simple_Latn" is src, split differently
                if src_lang == "eng" and 'simple_Latn' in tgt_lang:
                    src_lang, tgt_lang = pair_name.rsplit('-', 1)
                # If "eng-simple_Latn" is tgt, split differently
                elif tgt_lang == "eng":
                    tgt_lang = "eng-simple_Latn"
                print(f"  > Detected source language: {src_lang}, target language: {tgt_lang}")
        except ValueError:
            skip_logger.warning(f"Skipping invalid directory name (does not match 'src-tgt' format): {pair_name}")
            continue
        
        process_lang_pair(lang_pair_dir, src_lang, tgt_lang, mismatch_logger)
        
    print("\nProcessing complete.")

def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Helper function to set up a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check Parquet dataset directories for valid language codes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        help="Root directory containing the language pair subdirectories (e.g., /path/to/data)"
    )
    parser.add_argument(
        "--skip_log",
        type=str,
        default="./check_skipdirs.log",
        help="Path for the log file for skipped directories."
    )
    parser.add_argument(
        "--mismatch_log",
        type=str,
        default="./check_file_mismatch.log",
        help="Path for the log file for directories with non-standard language codes."
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.is_dir():
        print(f"Error: data_dir '{data_dir}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)
        
    skip_log_path = Path(args.skip_log).resolve()
    mismatch_log_path = Path(args.mismatch_log).resolve()
    
    skip_log_path.parent.mkdir(parents=True, exist_ok=True)
    mismatch_log_path.parent.mkdir(parents=True, exist_ok=True)

    skip_logger = setup_logger('skip_logger', skip_log_path)
    mismatch_logger = setup_logger('mismatch_logger', mismatch_log_path)
    
    print(f"Starting processing from: {data_dir}")
    print(f"Logging skipped dirs to: {skip_log_path}")
    print(f"Logging mismatched names to: {mismatch_log_path}")
    print("-" * 30)
    
    process_data_dir(data_dir, skip_logger, mismatch_logger)

    print(f"Scanning '{data_dir}' for empty folders...")
    
    # We walk 'bottom-up' (topdown=False). This is crucial.
    # It ensures we visit child directories before their parents.
    for dirpath, dirnames, filenames in os.walk(data_dir, topdown=False):
        
        # Don't try to remove the root directory itself,
        # only folders *under* it.
        if dirpath == data_dir:
            continue

        # We use a try-except block, which is the most reliable way.
        # We attempt to remove the directory. If it's not empty,
        # os.rmdir() will raise an OSError, which we catch and ignore.
        try:
            # os.rmdir() only removes empty directories.
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")
            
        except OSError as e:
            # If the directory is not empty, an OSError is raised.
            # We can safely ignore this error and continue.
            # We'll print any other unexpected errors.
            if "Directory not empty" not in str(e) and "not empty" not in str(e):
                print(f"Error removing {dirpath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred at {dirpath}: {e}")