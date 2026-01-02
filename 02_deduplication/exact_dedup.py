import duckdb
import os
import glob
import argparse
import sys
import csv
import time
import shutil

def deduplicate_with_duckdb(input_files, output_dir, max_lines_per_shard, src_col, trg_col, stats_file_path, compression='SNAPPY', memory_limit="10GB", temp_dir=None):
    """
    Deduplicates Parquet files using DuckDB with disk-based offloading to prevent OOM.
    """
    print(f"--- Starting DuckDB Deduplication ---")
    print(f"Processing {len(input_files)} input files.")
    print(f"Deduplicating based on columns: '{src_col}' and '{trg_col}'")
    print(f"Output Directory: {output_dir}")
    print(f"Max Lines Per Shard: {max_lines_per_shard:_}")
    print(f"Memory Limit: {memory_limit}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a path for the temporary DuckDB database file.
    # Using a physical file instead of :memory: forces DuckDB 
    # to page to disk immediately, respecting the memory limit strictly.
    db_file_path = ":memory:"
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        # We create a specific file for the DB state
        db_file_path = os.path.join(temp_dir, "dedup_process.duckdb")
        print(f"Using Disk-Based Database at: {db_file_path}")
    else:
        print("WARNING: No temp_dir provided. Running in-memory (High OOM Risk).")

    con = None
    try:
        # 1. Connect to the database (Disk-based if temp_dir is provided)
        con = duckdb.connect(database=db_file_path)

        # 2. Configure Memory & Performance Limits
        if temp_dir:
            # Also set the temp_directory for intermediate query spilling (sorting/grouping buffers)
            safe_temp_dir = temp_dir.replace(os.sep, '/')
            con.execute(f"PRAGMA temp_directory='{safe_temp_dir}';")
            
        con.execute(f"PRAGMA memory_limit='{memory_limit}';")
        # Disabling order preservation speeds up processing significantly
        con.execute("PRAGMA preserve_insertion_order=FALSE;")
        # Allow multi-threading
        con.execute("PRAGMA threads=4;") 
        
        # Escape paths for SQL to ensure they work on Windows and inside the query string
        files_sql = ", ".join([f"'{f.replace(os.sep, '/')}'" for f in input_files])
        
        # --- STEP A: Count Lines BEFORE Deduplication ---
        print("Counting input lines...")
        try:
            # Using count(*) on parquet metadata is usually fast
            count_query_before = f"SELECT COUNT(*) FROM read_parquet([{files_sql}])"
            # This forces a scan of a specific column to ensure data integrity
            # SELECT COUNT(1) FROM read_parquet([{files_sql}]) WHERE {src_col} IS NOT NULL AND {trg_col} IS NOT NULL
            input_count = con.execute(count_query_before).fetchone()[0]
            print(f"Total Input Lines: {input_count:_}")
        except Exception as e:
            print(f"Warning: Could not count input lines: {e}")
            input_count = 0

        # 3. Construct the Query
        # Note: We calculate shard_id dynamically.
        # The inner SELECT DISTINCT handles the deduplication.
        # The middle SELECT adds a Row Number.
        # The outer COPY writes to Parquet partitions based on that Row Number.
        query = f"""
            COPY (
                SELECT 
                    * EXCLUDE (rn),
                    (rn - 1) // {max_lines_per_shard} AS shard_id
                FROM (
                    SELECT 
                        *, 
                        ROW_NUMBER() OVER () AS rn
                    FROM (
                        SELECT DISTINCT {src_col}, {trg_col} 
                        FROM read_parquet([{files_sql}])
                        WHERE {src_col} IS NOT NULL AND {trg_col} IS NOT NULL
                    )
                )
            ) TO '{output_dir.replace(os.sep, '/')}' 
            (FORMAT PARQUET, PARTITION_BY (shard_id), COMPRESSION {compression.upper()}, OVERWRITE_OR_IGNORE);
        """
        
        print("Executing Deduplication Query (this may take time, relying on disk spill)...")
        con.execute(query)
        print(f"Success! Sharded output saved to: {output_dir}")

        # --- STEP B: Count Lines AFTER Deduplication ---
        print("Counting output lines...")
        # We query the output directory specifically. 
        output_glob = os.path.join(output_dir, "**", "*.parquet").replace(os.sep, '/')
        count_query_after = f"SELECT COUNT(*) FROM read_parquet('{output_glob}')"
        output_count = con.execute(count_query_after).fetchone()[0]
        print(f"Total Output Lines: {output_count:_}")
        
        # --- Post-Processing: Flatten Hive Partitions ---
        print("Reorganizing output files (flattening Hive partitions)...")
        
        partition_folders = glob.glob(os.path.join(output_dir, "shard_id=*"))
        base_dir_name = os.path.basename(os.path.normpath(output_dir))
        
        for folder in partition_folders:
            try:
                # Extract shard number (e.g., 'shard_id=5' -> '5')
                folder_name = os.path.basename(folder)
                shard_num = folder_name.split('=')[1]
                
                # Find the parquet file inside (usually data.parquet)
                files_in_shard = glob.glob(os.path.join(folder, "*.parquet"))
                
                if files_in_shard:
                    src_file = files_in_shard[0]
                    # New Naming: dirname_shard_N.parquet
                    dst_file = os.path.join(output_dir, f"{base_dir_name}_shard_{shard_num}.parquet")
                    
                    # Move and rename
                    os.rename(src_file, dst_file)
                
                # Remove the empty directory
                os.rmdir(folder)
            except Exception as e:
                print(f"Warning: Could not process partition folder {folder}: {e}")

        # --- STEP C: Write Stats to CSV ---
        if stats_file_path:
            print(f"Writing statistics to {stats_file_path}...")
            removed_count = input_count - output_count
            
            # Check if file exists to write header
            file_exists = os.path.isfile(stats_file_path)
            
            # Ensure stats dir exists
            stats_dir = os.path.dirname(stats_file_path)
            if stats_dir:
                os.makedirs(stats_dir, exist_ok=True)

            with open(stats_file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow(['lang_pair', 'input_rows', 'output_rows', 'rows_removed', 'timestamp'])
                
                writer.writerow([
                    base_dir_name,
                    input_count,
                    output_count,
                    removed_count,
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])
                
        print(f"Processing Complete. Deduplicated files are in: {output_dir}")

    except Exception as e:
        print(f"Error during DuckDB processing: {e}")
        # Re-raise to ensure non-zero exit code on failure
        sys.exit(1)
    finally:
        # Close connection explicitly
        if con:
            con.close()
        
        # CLEANUP: Remove the temporary database file
        # This is important because the DB file can grow very large during processing
        if temp_dir and db_file_path != ":memory:" and os.path.exists(db_file_path):
            print(f"Cleaning up temporary database file: {db_file_path}")
            try:
                os.remove(db_file_path)
                # Optional: Remove WAL (Write Ahead Log) files if they exist
                wal_path = db_file_path + ".wal"
                if os.path.exists(wal_path):
                    os.remove(wal_path)
            except OSError as e:
                print(f"Warning: Could not remove temp DB file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate and shard Parquet datasets (Disk-Backed).")
    
    parser.add_argument("--data_dir", required=True, help="Root directory containing input parquet files.")
    parser.add_argument("--out_dir", required=True, help="Root directory for output.")
    parser.add_argument("--stats_file", required=True, help="Path to the CSV file where stats will be saved.")
    parser.add_argument("--src_col", default="source_text", help="Name of source column.")
    parser.add_argument("--trg_col", default="target_text", help="Name of target column.")
    parser.add_argument("--max_lines", type=int, default=100_000_000, help="Maximum lines per output shard.")
    parser.add_argument("--temp_dir", required=True, help="Directory for DuckDB temp files (REQUIRED for large data).")
    parser.add_argument("--compression", default="SNAPPY", help="Compression codec (SNAPPY, ZSTD).")
    parser.add_argument("--memory_limit", default="10GB", help="RAM limit for DuckDB (e.g., 10GB).")
    
    args = parser.parse_args()

    # 1. Find Files
    search_pattern = os.path.join(args.data_dir, "**", "*.parquet")
    input_files = glob.glob(search_pattern, recursive=True)
    
    if not input_files:
        print(f"Error: No .parquet files found in {args.data_dir}")
        sys.exit(1)

    # 2. Determine Output Path
    base_name = os.path.basename(os.path.normpath(args.data_dir))
    final_output_path = os.path.join(args.out_dir, base_name)

    # 3. Execution
    deduplicate_with_duckdb(
        input_files, 
        final_output_path, 
        max_lines_per_shard=args.max_lines,
        src_col=args.src_col,
        trg_col=args.trg_col,
        stats_file_path=args.stats_file,
        compression=args.compression,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir
    )