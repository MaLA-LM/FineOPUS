import os
import json
import requests
import zipfile
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import langcodes
import argparse
from typing import List, Dict, Set, Tuple
from pathlib import Path

def setup_directories(out_dir: str, tmp_dir: str):
    """Creates the output and temporary directories if they don't exist."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Directories '{out_dir}' and '{tmp_dir}' are ready.")

def load_progress(progress_file: str) -> Set[str]:
    """
    Reads the progress file and returns a set of processed URLs for fast lookup.
    """
    if not os.path.exists(progress_file):
        return set()
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except IOError as e:
        print(f"Warning: Could not read progress file {progress_file}. Starting fresh. Error: {e}")
        return set()

def save_progress(progress_file: str, url: str):
    """
    Appends a successfully processed URL to the progress file.
    """
    try:
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{url}\n")
    except IOError as e:
        print(f"Error: Could not write to progress file {progress_file}. Error: {e}")

def download_file(url: str, dest_path: str):
    """
    Downloads a file from a URL to a destination path using streaming.
    """
    print(f"Downloading {url} to {dest_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def unzip_file(zip_path: str, extract_to_dir: str):
    """
    Unzips a file to a specified directory.
    """
    print(f"Unzipping {zip_path} to {extract_to_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to_dir)
    print("Unzip complete.")

def find_lang_files(extract_to_dir: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
    """
    Walks the extraction directory to find the source and target language files.
    """
    src_file_path, tgt_file_path = None, None
    for root, _, files in os.walk(extract_to_dir):
        for file in files:
            # Check for the file extension. Assumes format like 'filename.en', 'filename.ru'
            if file.endswith(f'.{src_lang}'):
                src_file_path = os.path.join(root, file)
            elif file.endswith(f'.{tgt_lang}'):
                tgt_file_path = os.path.join(root, file)
        
        # If we found both in the same directory, we can stop
        if src_file_path and tgt_file_path:
            break
            
    if not src_file_path or not tgt_file_path:
        raise FileNotFoundError(
            f"Could not find both source (.__{src_lang}__) and target (.__{tgt_lang}__) "
            f"files in {extract_to_dir}"
        )
        
    print(f"Found source file: {src_file_path}")
    print(f"Found target file: {tgt_file_path}")
    return src_file_path, tgt_file_path


def process_parallel_files(
    src_path: str,
    tgt_path: str,
    metadata: Dict,
    conv_src_code: str,
    conv_tgt_code: str,
    out_dir: str,
    shard_size: int = 10_000_000
) -> List[str]:
    """
    Reads two parallel text files line by line, combines them with metadata,
    and saves them in sharded Parquet files.

    Args:
        src_path: Path to the source language file.
        tgt_path: Path to the target language file.
        metadata: Dictionary containing metadata for the dataset.
        conv_src_code: Converted source language code (e.g., 'eng').
        conv_tgt_code: Converted target language code (e.g., 'rus').
        out_dir: The root output directory (e.g., 'output/').
        shard_size: The maximum number of rows per Parquet file.

    Returns:
        A list of file paths for all created shards.
    """
    print(f"Processing parallel files in shards of size {shard_size}...")
    collected_rows = []
    shard_paths = []
    shard_index = 0

    # Create the language-pair directory, e.g., 'output/eng-rus'
    lang_pair_dir = os.path.join(out_dir, f"{conv_src_code}-{conv_tgt_code}")
    os.makedirs(lang_pair_dir, exist_ok=True)

    # Prepare a clean metadata template *before* the loop for efficiency
    row_template = metadata.copy()
    if 'details' in row_template:
        del row_template['details']
    if 'file_size' in row_template:
        del row_template['file_size']
    row_template['orig_src_lang'] = metadata.get('src_lang')
    row_template['orig_tgt_lang'] = metadata.get('tgt_lang')
    row_template['conv_src_lang'] = conv_src_code
    row_template['conv_tgt_lang'] = conv_tgt_code

    with open(src_path, 'r', encoding='utf-8') as f_src, \
         open(tgt_path, 'r', encoding='utf-8') as f_tgt:

        # zip() stops at the shortest file, which is correct
        for src_line, tgt_line in zip(f_src, f_tgt):
            # Create row data from the template
            row_data = row_template.copy()
            row_data['source_text'] = src_line.strip()
            row_data['target_text'] = tgt_line.strip()
            collected_rows.append(row_data)

            # Check if the shard is full and save it
            if len(collected_rows) >= shard_size:
                # Define shard path
                shard_filename = f"{metadata['dataset']}_shard_{shard_index:03d}.parquet"
                output_parquet_path = os.path.join(lang_pair_dir, shard_filename)

                # Save the shard (assumes save_to_parquet exists)
                print(f"Saving shard {shard_index} with {len(collected_rows)} rows...")
                save_to_parquet(collected_rows, output_parquet_path)

                # Record path and reset for next shard
                shard_paths.append(output_parquet_path)
                collected_rows = []
                shard_index += 1

    # Save any remaining rows in the final shard
    if collected_rows:
        print(f"Saving final shard {shard_index} with {len(collected_rows)} rows...")
        shard_filename = f"{metadata['dataset']}_shard_{shard_index:03d}.parquet"
        output_parquet_path = os.path.join(lang_pair_dir, shard_filename)

        save_to_parquet(collected_rows, output_parquet_path)
        shard_paths.append(output_parquet_path)

    print(f"Completed saving {len(shard_paths)} shards to {lang_pair_dir}.")
    return shard_paths


def save_to_parquet(data: List[Dict], output_path: str):
    """
    Saves a list of dictionaries to a Parquet file using pyarrow.
    """
    if not data:
        print(f"No data to save to {output_path}.")
        return
    
    try:
        # Convert list of dicts to a PyArrow Table
        table = pa.Table.from_pylist(data)
        
        # Write the table to a Parquet file
        # Use Snappy compression for a good balance of speed and size
        pq.write_table(table, output_path, compression='snappy')
        print(f"Successfully saved {len(data)} rows to {output_path}")
    except Exception as e:
        print(f"Error: Could not save Parquet file {output_path}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, process, and convert OPUS data to Parquet.")
    parser.add_argument(
        '-i', '--input_json',
        type=str,
        default='sample_data.json',
        help='Path to the input JSON-Lines file.'
    )
    parser.add_argument(
        '-m', '--mapping_file_dir',
        type=str,
        default='corpus_mapping',
        help='Directory containing language mapping files.'
    )
    parser.add_argument(
        '-o', '--out_dir',
        type=str,
        default='output',
        help='Directory to store the final Parquet files and progress file.'
    )
    parser.add_argument(
        '-t', '--tmp_dir',
        type=str,
        default='temp_processing',
        help='Temporary directory for downloading and unzipping files.'
    )
    parser.add_argument(
        '-p', '--progress_file',
        type=str,
        default='processed_urls.txt',
        help='File to track processed URLs.'
    )
    parser.add_argument(
        '-e', '--error_log',
        type=str,
        default='error_log.txt'
    )
    args = parser.parse_args()

    setup_directories(args.out_dir, args.tmp_dir)

    processed_urls = load_progress(args.progress_file)
    print(f"Loaded {len(processed_urls)} processed URLs from progress file.")

    # Load the input JSON file
    with open(args.input_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    if not json_data:
        print("No data found in input file. Exiting.")
        exit
    print(f"Found {len(json_data)} total items to process.")
    
    for idx, metadata in enumerate(json_data):
        print(f"Processing dataset: {idx}/{len(json_data)} {metadata.get('dataset')} {metadata.get('version')}")
        url = metadata.get('url')
        if not url:
            print(f"Skipping item with no URL: {metadata}")
            continue

        if url in processed_urls:
            print(f"Skipping already processed URL: {url}")
            continue

        # Define paths for this specific file
        zip_filename = url.split('/')[-1]
        zip_path = os.path.join(args.tmp_dir, zip_filename)
        # Create a unique extraction dir for each zip to avoid conflicts
        extract_dir = os.path.join(args.tmp_dir, zip_filename.replace('.zip', ''))

        try:
            # 1. Download
            download_file(url, zip_path)
            
            # 2. Unzip
            os.makedirs(extract_dir, exist_ok=True)
            unzip_file(zip_path, extract_dir)
            
            # 3. Find language files and convert codes
            src_lang = metadata['src_lang']
            tgt_lang = metadata['tgt_lang']
            
            try:
                mapping_file = Path(args.mapping_file_dir) / metadata['dataset'] / metadata['version'] / "language_mappings.tsv"
                mapping_dict = {}
                if mapping_file.is_file():
                    print(f"Using mapping file: {str(mapping_file)}")
                    with open(mapping_file, 'r', encoding='utf-8') as mf:
                        for line in mf:
                            parts = line.strip().split('\t')
                            if len(parts) == 1: 
                                continue # skip invalid lines
                            # print(parts)
                            if len(parts[1].split('_')) == 3:
                                lang_code, script_code, region_code = parts[1].split('_')
                                mapping_dict[parts[0]] = lang_code + '_' + script_code
                            else:
                                mapping_dict[parts[0]] = parts[1]
                    if src_lang in mapping_dict.keys() and src_lang in mapping_dict.keys():
                        src_code_converted = mapping_dict.get(src_lang)
                        tgt_code_converted = mapping_dict.get(tgt_lang)
                        print(f"Mapped {src_lang} to {src_code_converted} and {tgt_lang} to {tgt_code_converted} using mapping file.")
                    else:
                        print(f"Warning: Cannot find language code in mapping file: {line.strip()}. Using langcodes conversion.")
                        src_code_converted = langcodes.get(src_lang).to_alpha3()
                        tgt_code_converted = langcodes.get(tgt_lang).to_alpha3()
                        print(f"Converted {src_lang} to {src_code_converted} and {tgt_lang} to {tgt_code_converted} using langcodes.")                   
                else:
                    print(f"Mapping file not found: {str(mapping_file)}. Using langcodes conversion.")
                    src_code_converted = langcodes.get(src_lang).to_alpha3()
                    tgt_code_converted = langcodes.get(tgt_lang).to_alpha3()
                    print(f"Converted {src_lang} to {src_code_converted} and {tgt_lang} to {tgt_code_converted} using langcodes.")
            except Exception as e:
                print(f"Warning: Could not get 3-letter code for {str(src_lang)}-{str(tgt_lang)}. Using originals. Error: {str(e)}")
                src_code_converted = src_lang
                tgt_code_converted = tgt_lang
                print(f"Using original codes: {src_lang}, {tgt_lang}.")
                with open(args.error_log, 'a') as f:
                    lang_code_error = metadata.copy()
                    lang_code_error['error'] = f"Langcode conversion failed for {src_lang}-{tgt_lang}: {e}"
                    f.write(json.dumps(lang_code_error) + "\n")

            src_file, tgt_file = find_lang_files(extract_dir, src_lang, tgt_lang)
            
            # 4. Collect rows and save to Parquet
            process_parallel_files(src_file, tgt_file, metadata, src_code_converted, tgt_code_converted, args.out_dir)
            
            # 5. Record progress *only on success*
            save_progress(args.progress_file, url)

        except Exception as e:
            print(f"--- ERROR processing {url}: {e} ---")
            print("Skipping this file and continuing...")

        finally:
            # 7. Clean up temp files for this item
            print("Cleaning up temp files...")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print(f"Dataset complete {metadata['dataset']} {metadata['version']}.")
            print("-" * 89)

    print("Script finished.")

