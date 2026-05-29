import argparse
import os
import pyarrow.parquet as pq
import logging
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def _count_rows_for_subdir(task):
    subdir, subdir_path = task
    logging.info(f"Processing {subdir_path}")
    total_rows = 0
    try:
        with os.scandir(subdir_path) as it:
            parquet_files = [entry.path for entry in it if entry.is_file() and entry.name.endswith('.parquet')]
        parquet_files.sort()
        for file in parquet_files:
            try:
                meta = pq.read_metadata(file)
                total_rows += meta.num_rows
            except Exception as e:
                logging.warning(f"Failed to read metadata for {file}: {e}")
    except Exception as e:
        logging.warning(f"Failed to process directory {subdir_path}: {e}")
    return subdir, total_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="Root directory")
    parser.add_argument("--output_file", required=True, help="Output file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads (defaults to CPU count)")
    args = parser.parse_args()

    root_dir = args.root_dir

    # Dictionary to store the total rows for each language pair
    language_pair_counts = {}

    # Collect language pair subdirectories
    subdir_tasks = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            subdir_tasks.append((subdir, subdir_path))

    if not subdir_tasks:
        logging.warning("No language pair subdirectories found.")
    
    # Determine workers
    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    workers = min(workers, max(len(subdir_tasks), 1))
    logging.info(f"Found {len(subdir_tasks)} language pairs. Using {workers} workers.")

    # Parallel processing of subdirectories
    if subdir_tasks:
        if workers <= 1:
            for task in subdir_tasks:
                subdir, total_rows = _count_rows_for_subdir(task)
                language_pair_counts[subdir] = total_rows
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_subdir = {executor.submit(_count_rows_for_subdir, task): task[0] for task in subdir_tasks}
                for future in as_completed(future_to_subdir):
                    subdir, total_rows = future.result()
                    language_pair_counts[subdir] = total_rows

    # Write the results to an Excel file
    output_file = args.output_file
    df = pd.DataFrame(
        sorted(language_pair_counts.items()),
        columns=["language_pair", "total_rows"]
    )
    df.to_excel(output_file, index=False)
    for pair, count in sorted(language_pair_counts.items()):
        logging.info(f"{pair}: {count} samples")
    logging.info(f"Results written to {output_file}")

if __name__ == "__main__":
    main()