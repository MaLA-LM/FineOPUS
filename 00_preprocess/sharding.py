import argparse
import csv
import re
import sys
import logging
import os
import multiprocessing
from functools import partial
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Regex to validate language code: 3 lowercase letters, underscore, 1 uppercase, 3 lowercase
LANG_CODE_PATTERN = r"[a-z]{3}_[A-Z][a-z]{3}"
# Regex to validate the entire pair, e.g., eng_Latn-ara_Arab
LANG_PAIR_RE = re.compile(f"^{LANG_CODE_PATTERN}-{LANG_CODE_PATTERN}$")
MAX_LINES_PER_SHARD = 100_000_000

def write_shard(output_dir: Path, shard_index: int, batches: list, schema: pa.Schema):
    """
    Writes a list of pyarrow batches to a new Parquet shard file.
    """
    if not batches:
        print(f"Skipping empty shard {shard_index} for {output_dir.name}")
        return
    
    output_file = output_dir / f"{output_dir.name}_shard_{shard_index:03d}.parquet"
    try:
        # Combine batches into a single table
        table = pa.Table.from_batches(batches, schema=schema)
        # Write the table to a Parquet file
        pq.write_table(table, output_file, compression='snappy')
    except Exception as e:
        print(f"Error writing shard {output_file}: {e}", file=sys.stderr)

def process_lang_pair(lang_pair_dir: Path, base_out_dir: Path, pq_logger: logging.Logger) -> int:
    """
    Processes all Parquet files for a single language pair,
    sharding them into a new output directory.
    
    This function writes batches directly to disk using ParquetWriter
    to maintain low memory usage.
    
    Returns:
        The total number of lines processed for this pair.
    """
    # 1. Get pair name from the directory
    pair_name = lang_pair_dir.name

    # 2. Create output directory based on the original directory name
    output_shard_dir = base_out_dir / pair_name
    output_shard_dir.mkdir(parents=True, exist_ok=True)

    # 3. Delete any existing shards to start fresh
    deleted_count = 0
    try:
        for shard_file in output_shard_dir.glob("shard_*.parquet"):
            shard_file.unlink()  # Delete the file
            deleted_count += 1
    except Exception as e:
        print(f"Warning: Could not delete existing shards in {output_shard_dir}: {e}", file=sys.stderr)
    
    if deleted_count > 0:
        print(f"  Deleted {deleted_count} existing shards from {output_shard_dir.name} to start fresh.")

    # 4. Find all input parquet files
    parquet_files = sorted(list(lang_pair_dir.glob("*.parquet")))
    if not parquet_files:
        print(f"No .parquet files found in {lang_pair_dir}", file=sys.stderr)
        return 0

    # 5. Initialize sharding variables
    total_lines_for_pair = 0
    current_shard_lines = 0
    schema = None
    
    # Start sharding from index 1
    shard_index = 1
    current_shard_writer = None
    
    try:
        # Process all parquet files for this pair
        for pq_file in parquet_files:
            try:
                parquet_reader = pq.ParquetFile(pq_file)
                
                # Get schema from first file
                if schema is None:
                    # Use .schema_arrow to get a pyarrow.Schema
                    schema = parquet_reader.schema_arrow
                
                # Check for schema consistency
                # Compare against .schema_arrow
                elif schema != parquet_reader.schema_arrow:
                    pq_logger.warning(
                        f"Schema mismatch in {pq_file}. Skipping file.\n"
                        f"  Expected: {schema}\n"
                        f"  Got:      {parquet_reader.schema_arrow}"
                    )
                    continue

                # 6. Iterate over batches in the file
                for batch in parquet_reader.iter_batches():
                    total_lines_for_pair += batch.num_rows
                    batch_to_process = batch
                    
                    # This loop handles splitting a single batch if it crosses a shard boundary
                    while batch_to_process.num_rows > 0:
                        
                        # If no writer is open, create a new one for the new shard
                        if current_shard_writer is None:
                            output_file = output_shard_dir / f"{pair_name}_shard_{shard_index:03d}.parquet"
                            current_shard_writer = pq.ParquetWriter(output_file, schema, compression='snappy')
                        
                        rows_needed = MAX_LINES_PER_SHARD - current_shard_lines
                        
                        if batch_to_process.num_rows <= rows_needed:
                            # Batch fits entirely in the current shard
                            current_shard_writer.write_batch(batch_to_process)
                            current_shard_lines += batch_to_process.num_rows
                            batch_to_process = None # Exit inner loop
                        
                        else:
                            # Batch must be split
                            slice_to_add = batch_to_process.slice(0, rows_needed)
                            
                            # Write the slice that fills the current shard
                            current_shard_writer.write_batch(slice_to_add)
                            
                            # Close this now-full shard
                            current_shard_writer.close()
                            current_shard_writer = None
                            
                            # Reset for next shard
                            shard_index += 1
                            current_shard_lines = 0
                            
                            # The remainder of the batch will be processed in the next loop iteration
                            batch_to_process = batch_to_process.slice(rows_needed)

                        if batch_to_process is None:
                            break # Go to next batch from file

            except Exception as e:
                print(f"Error reading {pq_file}: {e}", file=sys.stderr)
                continue
    
    finally:
        # 7. Write any remaining data in the last shard
        if current_shard_writer is not None:
            # If the loop finishes and the writer is still open, close it.
            current_shard_writer.close()

    return total_lines_for_pair

def load_processed_pairs(csv_file_path: Path) -> set[str]:
    """
    Reads the output CSV file to find which pairs have already been processed.
    Returns a set of processed pair names (e.g., 'eng_Latn-ara_Arab').
    """
    if not csv_file_path.exists():
        return set()
    
    processed = set()
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as f:
            # Use DictReader to read by column name, more robust to column order
            reader = csv.DictReader(f)
            for row in reader:
                # The 'src_lang-tgt_lang' column holds the pair name
                if 'src_lang-tgt_lang' in row:
                    processed.add(row['src_lang-tgt_lang'])
    except csv.Error as e:
        print(f"Warning: Could not read or parse CSV {csv_file_path}. Will process all folders. Error: {e}", file=sys.stderr)
        return set() # Return empty set on error, forcing processing
    except Exception as e:
        print(f"Warning: Unexpected error reading {csv_file_path}: {e}", file=sys.stderr)
        return set()
    
    return processed

def append_line_count(output_file: Path, src_lang: str, tgt_lang: str, pair_name: str, line_count: int):
    """
    Appends a single processed folder's line count to the CSV file.
    Creates the file and writes the header if it doesn't exist.
    """
    header = ['src_lang', 'tgt_lang', 'src_lang-tgt_lang', 'lines']
    # Check if file exists to determine if we need to write a header
    file_exists = output_file.is_file()
    
    try:
        # Use a lock to ensure only one process writes at a time
        # Note: This is now called from the main thread, but a lock
        # is good practice if it were ever parallelized.
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header only if the file is new
            if not file_exists:
                writer.writerow(header)
            # Write the data row
            writer.writerow([src_lang, tgt_lang, pair_name, line_count])
    except Exception as e:
        print(f"Error appending line count to {output_file}: {e}", file=sys.stderr)

def setup_worker_logger(logger_name: str, log_file_path: Path) -> logging.Logger:
    """
    Helper function to set up a logger within a worker process.
    This is process-safe and avoids duplicate handlers.
    """
    logger = logging.getLogger(logger_name)
    # Avoids adding duplicate handlers if pool reuses process
    if not logger.hasHandlers():
        logger.setLevel(logging.WARNING)
        # Use mode 'a' for append, which is generally process-safe
        handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
    return logger

def process_pair_worker(lang_pair_dir: Path, base_out_dir: Path, log_file_path: Path, pq_mismatch_log_path: Path):
    """
    Worker function for a single process in the pool.
    
    Sets up its own loggers and processes one lang pair.
    Returns:
        A tuple (src, tgt, pair_name, line_count) on success,
        or None on failure/skip.
    """
    # 1. Setup loggers for this specific worker process
    skip_logger = setup_worker_logger('skip_logger', log_file_path)
    pq_logger = setup_worker_logger('pq_logger', pq_mismatch_log_path)
    
    pair_name = lang_pair_dir.name

    # 2. Language pair name validation
    if not LANG_PAIR_RE.match(pair_name):
        print(f"[Worker] Skipping {pair_name} - Invalid name format")
        skip_logger.warning(f"Skipping directory, invalid name format (expected 'xxx_Yyyy-xxx_Yyyy'): {pair_name}")
        return None

    # 3. We can now safely split because the regex passed
    try:
        src_lang, tgt_lang = pair_name.split('-', 1)
    except ValueError:
        # This should be unreachable due to the regex, but good to have
        print(f"[Worker] Skipping {pair_name} - Internal split error")
        skip_logger.warning(f"Skipping directory, internal split error despite passing regex: {pair_name}")
        return None
    
    print(f"[Worker] Processing {pair_name}")
    
    # 4. Process this pair
    total_lines = process_lang_pair(lang_pair_dir, base_out_dir, pq_logger)
    
    # 5. Return results for the main thread to write to CSV
    if total_lines > 0:
        print(f"[Worker] Finished {pair_name}. Processed {total_lines} lines.")
        return (src_lang, tgt_lang, pair_name, total_lines)
    else:
        print(f"[Worker] Finished {pair_name}. No lines processed.")
        return None

def process_data_dir(data_dir: Path, out_dir: Path, csv_report_file: Path, log_file_path: Path, pq_mismatch_log_path: Path, num_workers: int):
    """
    Finds all language pair directories in data_dir and processes them
    in parallel using a process pool.
    Writes line counts to the CSV file *after* all processing is complete.
    """
    # 1. Load already processed pairs
    processed_pairs = load_processed_pairs(csv_report_file)
    if processed_pairs:
        print(f"Loaded {len(processed_pairs)} already processed pairs from {csv_report_file.name}")

    # 2. Find all candidate directories
    all_lang_pair_dirs = [d for d in data_dir.glob("*-*") if d.is_dir()]
    if not all_lang_pair_dirs:
        print(f"No language pair directories (e.g., 'eng_Latn-fra_Latn') found in {data_dir}", file=sys.stderr)
        return

    total_pairs = len(all_lang_pair_dirs)
    print(f"Found {total_pairs} total language pair directories.")

    # 3. Filter out already processed pairs *before* starting the pool
    dirs_to_process = []
    for lang_pair_dir in all_lang_pair_dirs:
        if lang_pair_dir.name not in processed_pairs:
            dirs_to_process.append(lang_pair_dir)
        else:
            print(f"Skipping {lang_pair_dir.name} - Already in CSV report")
    
    num_to_process = len(dirs_to_process)
    if num_to_process == 0:
        print("No new language pairs to process.")
        return

    print(f"Starting parallel processing for {num_to_process} new pairs using {num_workers} workers...")

    # 4. Create the partial function for the worker
    # We use `partial` to "bake in" the static arguments that
    # won't change for any of the workers.
    worker_func = partial(
        process_pair_worker,
        base_out_dir=out_dir,
        log_file_path=log_file_path,
        pq_mismatch_log_path=pq_mismatch_log_path
    )

    # 5. Run the pool
    # The 'results' will be a list of tuples or Nones
    results = []
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # pool.map applies `worker_func` to each item in `dirs_to_process`
            # and returns a list of the results in order.
            results = pool.map(worker_func, dirs_to_process)
    except Exception as e:
        print(f"A critical error occurred during parallel processing: {e}", file=sys.stderr)
        return

    # 6. Process results in the main thread (THIS IS CSV-SAFE)
    print("Parallel processing complete. Writing results to CSV...")
    newly_processed_count = 0
    for result in results:
        # result will be None if skipped, or (src, tgt, name, lines)
        if result:
            src_lang, tgt_lang, pair_name, total_lines = result
            append_line_count(csv_report_file, src_lang, tgt_lang, pair_name, total_lines)
            newly_processed_count += 1
    
    print(f"Finished. Appended {newly_processed_count} new pair results to {csv_report_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shard Parquet datasets based on line count and validate lang pair codes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data_dir", 
        type=str,
        required=True,
        help="Root directory containing the language pair subdirectories (e.g., /path/to/data)"
    )
    parser.add_argument(
        "--out_dir", 
        type=str,
        required=True,
        help="Output directory to save sharded Parquet files and line_counts.csv"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Path for the output CSV file. (default: {out_dir}/sharding_line_counts.csv)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path for the log file for skipped directories. (default: {out_dir}/sharding_skipped_dirs.log)"
    )
    parser.add_argument(
        "--pq_mismatch",
        type=str,
        default=None,
        help="Path for the log file for parquet schema mismatches. (default: {out_dir}/sharding_pq_mismatch.log)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help=f"Number of parallel processes to use. (default: {os.cpu_count()})"
    )
    
    args = parser.parse_args()
        
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    
    if not data_dir.is_dir():
        print(f"Error: data_dir '{data_dir}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve file paths, implementing the described defaults
    output_file_path = Path(args.output_file) if args.output_file else out_dir / "sharding_line_counts.csv"
    log_file_path = Path(args.log_file) if args.log_file else out_dir / "sharding_skipped_dirs.log"
    pq_mismatch_log_path = Path(args.pq_mismatch) if args.pq_mismatch else out_dir / "sharding_pq_mismatch.log"

    # --- Logger setup is removed from main ---
    # Loggers will be initialized in their own worker processes
    # to avoid file handle conflicts.
    
    print(f"Starting processing from: {data_dir}")
    print(f"Writing outputs to:    {out_dir}")
    print(f"Writing line counts to: {output_file_path}")
    print(f"Logging skipped dirs to: {log_file_path}")
    print(f"Logging PQ mismatches to: {pq_mismatch_log_path}")
    print(f"Using {args.num_workers} parallel workers.")
    
    # Process the data directory
    process_data_dir(
        data_dir,
        out_dir,
        output_file_path,
        log_file_path,
        pq_mismatch_log_path,
        args.num_workers
    )
    
    print(f"\nProcessing complete. Results are saved in {output_file_path}")