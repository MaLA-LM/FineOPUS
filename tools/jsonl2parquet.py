import os
import argparse
import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import jsonlines

# --- Configuration ---
# Set up basic logging to capture progress and errors.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conversion.log"),
        logging.StreamHandler()
    ]
)

# The name of the file to store the names of completed directories.
COMPLETED_LOG_FILE = "completed_folders.txt"

def load_completed_folders():
    """
    Reads the log file to get a set of folders that have already been converted.

    Returns:
        set: A set of folder names that have been successfully processed.
    """
    if not os.path.exists(COMPLETED_LOG_FILE):
        return set()
    try:
        with open(COMPLETED_LOG_FILE, 'r') as f:
            completed = {line.strip() for line in f if line.strip()}
            logging.info(f"Loaded {len(completed)} completed folder(s) from '{COMPLETED_LOG_FILE}'.")
            return completed
    except IOError as e:
        logging.error(f"Could not read log file at '{COMPLETED_LOG_FILE}': {e}")
        return set()

def mark_folder_as_completed(folder_name):
    """
    Appends a folder name to the log file to mark it as completed.

    Args:
        folder_name (str): The name of the folder to log as completed.
    """
    try:
        with open(COMPLETED_LOG_FILE, 'a') as f:
            f.write(f"{folder_name}\n")
        logging.info(f"Successfully marked folder '{folder_name}' as completed in '{COMPLETED_LOG_FILE}'.")
    except IOError as e:
        logging.error(f"Could not write to log file for folder '{folder_name}': {e}")

def convert_files_in_folder(folder_path, chunksize=1000000):
    """
    Finds all .jsonl files in a given folder, converts them to .parquet
    using jsonlines and pyarrow, and deletes the original .jsonl file
    upon successful conversion.

    Args:
        folder_path (str): The full path to the directory containing .jsonl files.
        chunksize (int): The number of lines to read into memory before
                         writing to the parquet file.

    Returns:
        bool: True if all files were converted successfully, False otherwise.
    """
    all_successful = True
    try:
        # Using pathlib for cleaner path operations
        p = Path(folder_path)
        files_to_process = list(p.glob('*.jsonl'))
        
        if not files_to_process:
            logging.warning(f"No .jsonl files found in '{folder_path}'. Skipping.")
            return True  # No files to process is considered a success for this folder.

        logging.info(f"Found {len(files_to_process)} .jsonl file(s) in '{p.name}'.")

        for jsonl_path in files_to_process:
            # Create the new parquet path by changing the file extension
            parquet_path = jsonl_path.with_suffix('.parquet')
            writer = None
            
            try:
                logging.info(f"Starting conversion of '{jsonl_path}' to '{parquet_path}'...")
                
                chunk_data = []
                chunk_index = 0

                # Use jsonlines.open to read the file line by line
                with jsonlines.open(jsonl_path) as reader:
                    for line in reader:
                        chunk_data.append(line)
                        
                        # When the chunk is full, write it to the parquet file
                        if len(chunk_data) == chunksize:
                            chunk_index += 1
                            logging.info(f"Writing chunk {chunk_index}...")
                            
                            # Convert the list of dicts directly to an Arrow Table
                            table = pa.Table.from_pylist(chunk_data)
                            
                            if writer is None:
                                # For the first chunk, create the writer and schema
                                writer = pq.ParquetWriter(parquet_path, table.schema)
                                logging.info(f"Created '{parquet_path}' and writing the first chunk...")
                            
                            writer.write_table(table)
                            chunk_data = [] # Reset the chunk

                # After the loop, write any remaining data in the last chunk
                if chunk_data:
                    logging.info("Writing final chunk...")
                    table = pa.Table.from_pylist(chunk_data)
                    
                    if writer is None: # Handles files smaller than chunksize
                        writer = pq.ParquetWriter(parquet_path, table.schema)
                        logging.info(f"Created '{parquet_path}' and writing the only chunk...")
                        
                    writer.write_table(table)

                if writer:
                    writer.close()
                    logging.info(f"Successfully created '{parquet_path}'.")

                    # Delete the original .jsonl file only after a successful conversion.
                    os.remove(jsonl_path)
                    logging.info(f"Deleted original file '{jsonl_path}'.")
                else:
                    logging.warning(f"Input file '{jsonl_path}' appears to be empty. No output file was created.")

            except FileNotFoundError:
                logging.error(f"Error: The input file was not found at '{jsonl_path}'.")
                all_successful = False
            except jsonlines.InvalidLineError as e:
                logging.error(f"Invalid JSON line in '{jsonl_path}': {e}")
                all_successful = False
                # If an error happens, clean up any partially created Parquet file.
                if writer:
                    writer.close()
                if os.path.exists(parquet_path):
                    os.remove(parquet_path)
                    logging.info(f"Cleaned up partially created file '{parquet_path}'.")
            except Exception as e:
                logging.error(f"An unexpected error occurred converting '{jsonl_path}': {e}")
                all_successful = False
                # Clean up
                if writer:
                    writer.close()
                if os.path.exists(parquet_path):
                    os.remove(parquet_path)
                    logging.info(f"Cleaned up partially created file '{parquet_path}'.")

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing folder '{folder_path}': {e}")
        all_successful = False

    return all_successful

def main(input_dir):
    """
    Main function to orchestrate the conversion process.
    It processes subdirectories if they exist, otherwise, it processes the input directory itself.
    """
    if not os.path.isdir(input_dir):
        logging.error(f"Error: Input directory '{input_dir}' not found.")
        return

    logging.info(f"Starting conversion process for directory: '{input_dir}'")
    
    # Get all subdirectories in the input directory. Using os.scandir is more efficient.
    try:
        subdirectories = [d.name for d in os.scandir(input_dir) if d.is_dir()]
    except OSError as e:
        logging.error(f"Could not list directories in '{input_dir}': {e}")
        return
    
    # If subdirectories exist, process them.
    if subdirectories:
        logging.info(f"Found {len(subdirectories)} subdirectories to process.")
        completed_folders = load_completed_folders()

        for folder_name in subdirectories:
            if folder_name in completed_folders:
                logging.info(f"Skipping already processed folder: '{folder_name}'")
                continue

            logging.info(f"--- Processing folder: '{folder_name}' ---")
            folder_path = os.path.join(input_dir, folder_name)
            
            if convert_files_in_folder(folder_path):
                mark_folder_as_completed(folder_name)
            else:
                logging.error(f"--- Errors occurred in '{folder_name}'. It will be re-attempted on next run. ---")
    
    # If no subdirectories exist, process the input directory itself.
    else:
        logging.info("No subdirectories found. Processing the input directory directly.")
        convert_files_in_folder(input_dir)

    logging.info("Conversion process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON Lines (.jsonl) files to Parquet format.",
        epilog="Example: python your_script_name.py --input_dir /path/to/your/data"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True, # The script requires this argument to run.
        help="The root directory containing .jsonl files or subfolders with .jsonl files."
    )

    args = parser.parse_args()
    if args.input_dir:
        main(args.input_dir)