import os
import json
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
from datetime import datetime, date

def json_serializer(obj):
    """
    Custom JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    # Add more type handlers if needed
    raise TypeError(f"Type {type(obj)} not serializable")

def convert_row_to_serializable(row):
    """
    Convert a pandas row to a serializable dictionary
    """
    row_dict = {}
    for column, value in row.items():
        if pd.isna(value):
            row_dict[column] = None
        elif isinstance(value, (pd.Timestamp, datetime, date)):
            row_dict[column] = value.isoformat()
        elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            row_dict[column] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            row_dict[column] = float(value)
        elif isinstance(value, np.bool_):
            row_dict[column] = bool(value)
        elif isinstance(value, np.ndarray):
            row_dict[column] = value.tolist()
        else:
            row_dict[column] = value
    
    return row_dict

def process_parquet_to_jsonl(data_dir, output_dir, max_lines_per_file=20000000):
    """
    Process Parquet files to JSONL files with partitioning.
    
    Args:
        data_dir (str): Directory containing Parquet files with language subfolders
        output_dir (str): Directory to save JSONL files
        max_lines_per_file (int): Maximum number of lines per JSONL file
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all language subdirectories
    language_dirs = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
    
    for lang_dir in language_dirs:
        lang_path = os.path.join(data_dir, lang_dir)
        print(f"Processing language: {lang_dir}")
        
        # Find all Parquet files in the language directory
        parquet_files = [f for f in os.listdir(lang_path) 
                        if f.endswith('.parquet') or f.endswith('.parq')]
        
        if not parquet_files:
            print(f"No Parquet files found in {lang_path}")
            continue
        
        file_counter = 1
        line_counter = 0
        current_file = None
        
        for parquet_file in parquet_files:
            parquet_path = os.path.join(lang_path, parquet_file)
            print(f"  Reading: {parquet_file}")
            
            try:
                # Read Parquet file
                table = pq.read_table(parquet_path)
                
                # Check if 'text' field exists
                if 'text' not in table.column_names:
                    print(f"  Warning: 'text' field not found in {parquet_file}, skipping")
                    continue
                
                # Convert to pandas DataFrame for easier iteration
                df = table.to_pandas()
                
                # Convert datetime columns to strings to avoid serialization issues
                datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
                for col in datetime_cols:
                    df[col] = df[col].astype(str)
                
                for _, row in df.iterrows():
                    # Create new file if needed
                    if current_file is None or line_counter >= max_lines_per_file:
                        if current_file:
                            current_file.close()
                        
                        output_filename = f"{lang_dir}_{file_counter:03d}.jsonl"
                        output_path = os.path.join(output_dir, output_filename)
                        current_file = open(output_path, 'w', encoding='utf-8')
                        file_counter += 1
                        line_counter = 0
                        print(f"    Creating new file: {output_filename}")
                    
                    # Convert row to serializable dictionary and write as JSON line
                    row_dict = convert_row_to_serializable(row)
                    try:
                        json_line = json.dumps(row_dict, ensure_ascii=False, default=json_serializer)
                        current_file.write(json_line + '\n')
                        line_counter += 1
                    except TypeError as e:
                        print(f"    JSON serialization error: {e}")
                        # Fallback: convert all values to string
                        fallback_dict = {k: str(v) if not pd.isna(v) else None for k, v in row_dict.items()}
                        json_line = json.dumps(fallback_dict, ensure_ascii=False)
                        current_file.write(json_line + '\n')
                        line_counter += 1
                    
            except Exception as e:
                print(f"  Error processing {parquet_file}: {e}")
                continue
        
        # Close the last file
        if current_file:
            current_file.close()
    
    print("Processing completed!")


def get_language_directories(data_dir, lang_code):
    """
    Find all language subdirectories that match the given language code pattern.
    
    Args:
        data_dir (str): Directory containing language subfolders
        lang_code (str): Language code pattern to match (e.g., "zho_Hans")
        
    Returns:
        list: List of language directory names that match the pattern
    """
    # Escape special characters in the lang_code and create a regex pattern
    # that matches directories starting with the lang_code followed by optional suffix
    escaped_lang = re.escape(lang_code)
    lang_pattern = re.compile(rf'^{escaped_lang}(?:_\d+)?$')
    
    language_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and lang_pattern.match(item):
            language_dirs.append(item)
    
    return language_dirs


def process_parquet_to_jsonl_memory_efficient(data_dir, output_dir, lang_code, max_lines_per_file=20000000):
    """
    Memory-efficient version that processes Parquet files in batches.
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all language subdirectories
    language_dirs = get_language_directories(data_dir, lang_code)
    print(f"Found {len(language_dirs)} language directories matching pattern '{lang_code}'")
    
    for lang_dir in language_dirs:
        lang_path = os.path.join(data_dir, lang_dir)
        print(f"Processing language: {lang_dir}")
        
        # Find all Parquet files in the language directory
        parquet_files = [f for f in os.listdir(lang_path) 
                        if f.endswith('.parquet') or f.endswith('.parq') or f.endswith('.pq')]
        
        if not parquet_files:
            print(f"No Parquet files found in {lang_path}")
            continue
        
        file_counter = 1
        line_counter = 0
        current_file = None
        
        for parquet_file in parquet_files:
            parquet_path = os.path.join(lang_path, parquet_file)
            print(f"  Reading: {parquet_file}")
            
            try:
                # Open Parquet file for streaming
                parquet_file_obj = pq.ParquetFile(parquet_path)
                
                # Check if 'text' field exists
                schema = parquet_file_obj.schema
                if 'text' not in schema.names:
                    print(f"  Warning: 'text' field not found in {parquet_file}, skipping")
                    continue
                
                # Process in batches
                for batch in parquet_file_obj.iter_batches():
                    df = batch.to_pandas()
                    
                    # Convert datetime columns to strings to avoid serialization issues
                    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
                    for col in datetime_cols:
                        df[col] = df[col].astype(str)
                    
                    for _, row in df.iterrows():
                        # Create new file if needed
                        if current_file is None or line_counter >= max_lines_per_file:
                            if current_file:
                                current_file.close()
                            
                            output_filename = f"{lang_dir}_{file_counter:03d}.jsonl"
                            output_path = os.path.join(output_dir, output_filename)
                            current_file = open(output_path, 'w', encoding='utf-8')
                            file_counter += 1
                            line_counter = 0
                            print(f"    Creating new file: {output_filename}")
                        
                        # Convert row to serializable dictionary and write as JSON line
                        row_dict = convert_row_to_serializable(row)
                        try:
                            json_line = json.dumps(row_dict, ensure_ascii=False, default=json_serializer)
                            current_file.write(json_line + '\n')
                            line_counter += 1
                        except TypeError as e:
                            print(f"    JSON serialization error: {e}")
                            # Fallback: convert all values to string
                            fallback_dict = {k: str(v) if not pd.isna(v) else None for k, v in row_dict.items()}
                            json_line = json.dumps(fallback_dict, ensure_ascii=False)
                            current_file.write(json_line + '\n')
                            line_counter += 1
                
            except Exception as e:
                print(f"  Error processing {parquet_file}: {e}")
                continue
        
        # Close the last file
        if current_file:
            current_file.close()
    
    print("Processing completed!")

# Alternative approach using Arrow tables directly (avoids pandas conversion issues)
def process_with_arrow_direct(data_dir, output_dir, lang_code, max_lines_per_file=20000000):
    """
    Process using Arrow tables directly to avoid pandas conversion issues.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    language_dirs = get_language_directories(data_dir, lang_code)
    
    if not language_dirs:
        print(f"No language directories found matching pattern '{lang_code}' in {data_dir}")
        return
    
    for lang_dir in language_dirs:
        lang_path = os.path.join(data_dir, lang_dir)
        print(f"Processing language directory: {lang_dir}")
        
        parquet_files = [f for f in os.listdir(lang_path) 
                        if f.endswith(('.parquet', '.parq', '.pq'))]
        
        if not parquet_files:
            print(f"  No Parquet files found in {lang_path}")
            continue
        
        file_counter = 1
        line_counter = 0
        current_file = None
        
        for parquet_file in parquet_files:
            parquet_path = os.path.join(lang_path, parquet_file)
            print(f"  Reading: {parquet_file}")
            
            try:
                # Read the entire table (or use iter_batches for memory efficiency)
                table = pq.read_table(parquet_path)
                
                if 'text' not in table.column_names:
                    print(f"  Warning: 'text' field not found in {parquet_file}, skipping")
                    continue
                
                # Convert each row to a dictionary
                for i in range(table.num_rows):
                    row_dict = {}
                    for col_name in table.column_names:
                        value = table[col_name][i].as_py()
                        # Handle timestamp types
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[col_name] = value
                    
                    # Create new file if needed
                    if current_file is None or line_counter >= max_lines_per_file:
                        if current_file:
                            current_file.close()
                        
                        output_filename = f"{lang_dir}_{file_counter:03d}.jsonl"
                        output_path = os.path.join(output_dir, output_filename)
                        current_file = open(output_path, 'w', encoding='utf-8')
                        file_counter += 1
                        line_counter = 0
                        print(f"    Creating new file: {output_filename}")
                    
                    json_line = json.dumps(row_dict, ensure_ascii=False)
                    current_file.write(json_line + '\n')
                    line_counter += 1
                    
            except Exception as e:
                print(f"  Error processing {parquet_file}: {e}")
                continue
        
        if current_file:
            current_file.close()
    
    print("Processing completed!")

# Example usage
if __name__ == "__main__":
    """Main function to process all JSONL files for a specific language."""
    parser = argparse.ArgumentParser(description='Process and combine JSONL files for a specific language')
    parser.add_argument('--base-dir', default='/scratch/project_462000964/source_data/monolingual/HPLT2.0_cleaned', help='Base directory to search for JSONL files')
    parser.add_argument('--lang-code', default="zho_Hans", required=True, help='Language code to process')
    parser.add_argument('--output-dir', default='/scratch/project_462000964/FineOPUS/ablation_data/HPLT2.0_cleaned', help='Output base directory')
    parser.add_argument('--max-lines', type=int, default=5000, help='Maximum lines per JSONL file')
    parser.add_argument('--method', choices=['pandas', 'arrow'], default='arrow', help='Processing method (pandas or arrow)')
    
    args = parser.parse_args()
    
    print(f"Base directory: {args.base_dir}")
    print(f"Language code: {args.lang_code}")
    print(f"Output directory: {args.output_dir}")
    print(f"Maximum lines per file: {args.max_lines}")
    print(f"Processing method: {args.method}")
    print("-" * 50)
    
    if args.method == 'arrow':
        # Use Arrow direct method which handles timestamps better
        process_with_arrow_direct(
            data_dir=args.base_dir,
            output_dir=args.output_dir,
            lang_code=args.lang_code,
            max_lines_per_file=args.max_lines
        )
    else:
        # Use the pandas-based method with enhanced serialization
        process_parquet_to_jsonl_memory_efficient(
            data_dir=args.base_dir,
            output_dir=args.output_dir,
            lang_code=args.lang_code,
            max_lines_per_file=args.max_lines
        )