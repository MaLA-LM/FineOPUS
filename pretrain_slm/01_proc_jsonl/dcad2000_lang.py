#!/usr/bin/env python3
"""
Script to extract JSONL files with pattern *_keep.jsonl from ./{path_name}/{lang_code} 
directories and combine them into JSONL files with max 5000 lines each.
Processes one language at a time and saves all output to {output_dir}/{lang_code}.
"""

import os
import glob
from pathlib import Path
import jsonlines
from typing import List, Dict, Any, Tuple, Iterator
import argparse


def find_jsonl_files(base_dir: str, lang_code: str) -> List[str]:
    """
    Find all JSONL files matching pattern *_keep.jsonl across all subdirectories for the specified language.
    
    Args:
        base_dir: Base directory to search from
        lang_code: Language code to search for
        
    Returns:
        List of file paths matching the pattern
    """
    # Search pattern: base_dir/lang_code/*_keep.jsonl
    pattern = os.path.join(base_dir, lang_code, "*_keep.jsonl")
    jsonl_files = glob.glob(pattern, recursive=True)
    print(f"Searching pattern: {pattern}")
    print(f"Found {len(jsonl_files)} JSONL files matching pattern")
    
    # Print found files for verification
    for file_path in jsonl_files:
        print(f"  Found: {file_path}")
    
    return jsonl_files


def count_lines_in_file(file_path: str) -> int:
    """
    Count the number of valid JSON lines in a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Number of valid JSON lines
    """
    try:
        with jsonlines.open(file_path, mode='r') as reader:
            count = sum(1 for _ in reader)
        return count
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}")
        return 0


def read_jsonl_file(file_path: str) -> Iterator[Dict[Any, Any]]:
    """
    Read a JSONL file and yield dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Yields:
        Dictionary objects from the JSONL file
    """
    try:
        with jsonlines.open(file_path, mode='r') as reader:
            for obj in reader:
                yield obj
        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def create_file_batches(jsonl_files: List[str], max_lines: int) -> List[List[Tuple[str, int, str]]]:
    """
    Create batches of files that should be combined, respecting max_lines limit.
    
    Args:
        jsonl_files: List of JSONL file paths
        max_lines: Maximum lines per batch
        
    Returns:
        List of batches, where each batch is a list of (file_path, line_count, source_name) tuples
    """
    # First, collect file info with line counts and source names
    file_info_list = []
    for file_path in jsonl_files:
        line_count = count_lines_in_file(file_path)
        # Extract source name from file path for better naming
        source_name = Path(file_path).stem.replace('_keep', '')
        # Also include the parent directory name for uniqueness
        parent_dir_name = Path(file_path).parent.name
        unique_source_name = f"{parent_dir_name}_{source_name}"
        
        file_info_list.append((file_path, line_count, unique_source_name))
        print(f"File {file_path}: {line_count} lines, source: {unique_source_name}")
    
    batches = []
    current_batch = []
    current_batch_lines = 0
    
    # Sort files by line count (process smaller files first for better batching)
    sorted_files = sorted(file_info_list, key=lambda x: x[1])
    
    for file_path, line_count, source_name in sorted_files:
        # If this single file exceeds max_lines, it goes in its own batch
        if line_count > max_lines:
            # Finish current batch if it has files
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_lines = 0
            
            # Add the large file as its own batch
            batches.append([(file_path, line_count, source_name)])
        
        # If adding this file would exceed max_lines, start a new batch
        elif current_batch and current_batch_lines + line_count > max_lines:
            batches.append(current_batch)
            current_batch = [(file_path, line_count, source_name)]
            current_batch_lines = line_count
        else:
            current_batch.append((file_path, line_count, source_name))
            current_batch_lines += line_count
    
    # Add the last batch if it has files
    if current_batch:
        batches.append(current_batch)
    
    return batches


def generate_output_filename(file_batch: List[Tuple[str, int, str]], batch_idx: int, chunk_idx: int = None) -> str:
    """
    Generate appropriate output filename for a batch.
    
    Args:
        file_batch: List of (file_path, line_count, source_name) tuples
        batch_idx: Batch index
        chunk_idx: Chunk index for split files (if applicable)
        
    Returns:
        Output filename
    """
    if len(file_batch) == 1:
        # Single file
        _, _, source_name = file_batch[0]
        if chunk_idx is not None:
            # Large file being split
            return f"{source_name}_part_{chunk_idx:03d}.jsonl"
        else:
            # Small file being copied
            return f"{source_name}.jsonl"
    else:
        # Multiple files combined
        if chunk_idx is not None:
            return f"combined_batch_{batch_idx:03d}_part_{chunk_idx:03d}.jsonl"
        else:
            return f"combined_batch_{batch_idx:03d}.jsonl"


def save_batch_to_jsonl(file_batch: List[Tuple[str, int, str]], output_dir: str, batch_idx: int, max_lines: int):
    """
    Read multiple JSONL files, concatenate them, and save to JSONL file(s).
    
    Args:
        file_batch: List of (file_path, line_count, source_name) tuples
        output_dir: Output directory
        batch_idx: Batch index for naming
        max_lines: Maximum lines per JSONL file
    """
    print(f"\nProcessing batch {batch_idx} with {len(file_batch)} files:")
    for file_path, line_count, source_name in file_batch:
        print(f"  - {file_path} ({line_count} lines, source: {source_name})")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If single file and it's small enough, just copy with better naming
    if len(file_batch) == 1 and file_batch[0][1] <= max_lines:
        file_path, line_count, source_name = file_batch[0]
        output_filename = generate_output_filename(file_batch, batch_idx)
        output_file = os.path.join(output_dir, output_filename)
        
        try:
            with jsonlines.open(file_path, mode='r') as reader, \
                 jsonlines.open(output_file, mode='w') as writer:
                
                for obj in reader:
                    writer.write(obj)
            
            print(f"Copied {line_count} records to {output_file}")
            return
        except Exception as e:
            print(f"Error copying {file_path}: {e}")
            return
    
    # For multiple files or large single files, process with streaming and splitting
    current_chunk = 1
    current_lines = 0
    writer = None
    current_output_file = None
    
    try:
        for file_path, _, _ in file_batch:
            for obj in read_jsonl_file(file_path):
                # Start new file if needed
                if writer is None or current_lines >= max_lines:
                    if writer is not None:
                        writer.close()
                        print(f"Saved {current_lines} records to {current_output_file}")
                    
                    # Generate output filename
                    output_filename = generate_output_filename(file_batch, batch_idx, current_chunk)
                    current_output_file = os.path.join(output_dir, output_filename)
                    
                    writer = jsonlines.open(current_output_file, mode='w')
                    current_lines = 0
                    current_chunk += 1
                
                # Write the object
                writer.write(obj)
                current_lines += 1
        
        # Close the last file
        if writer is not None:
            writer.close()
            print(f"Saved {current_lines} records to {current_output_file}")
    
    except Exception as e:
        print(f"Error processing batch: {e}")
        if writer is not None:
            writer.close()


def main():
    """Main function to process all JSONL files for a specific language."""
    parser = argparse.ArgumentParser(description='Process and combine JSONL files for a specific language')
    parser.add_argument('--base-dir', default='.', help='Base directory to search for JSONL files')
    parser.add_argument('--lang-code', required=True, help='Language code to process')
    parser.add_argument('--output-dir', default='output', help='Output base directory')
    parser.add_argument('--max-lines', type=int, default=5000, help='Maximum lines per JSONL file')
    
    args = parser.parse_args()
    
    print(f"Base directory: {args.base_dir}")
    print(f"Language code: {args.lang_code}")
    print(f"Searching pattern: {args.base_dir}/**/{args.lang_code}/*_keep.jsonl")
    print(f"Output directory: {args.output_dir}")
    print(f"Maximum lines per file: {args.max_lines}")
    print("-" * 50)
    
    # Find all matching JSONL files for this language
    jsonl_files = find_jsonl_files(args.base_dir, args.lang_code)
    
    if not jsonl_files:
        print(f"No JSONL files found matching the pattern *_keep.jsonl for language {args.lang_code}")
        return
    
    # Create batches of files to process together
    batches = create_file_batches(jsonl_files, args.max_lines)
    
    total_lines = sum(line_count for batch in batches for _, line_count, _ in batch)
    print(f"\nTotal files to process: {len(jsonl_files)}")
    print(f"Total lines across all files: {total_lines}")
    print(f"Created {len(batches)} batches for processing")
    
    # Process each batch
    for batch_idx, file_batch in enumerate(batches, 1):
        save_batch_to_jsonl(file_batch, args.output_dir, batch_idx, args.max_lines)
    
    print(f"\n{'='*50}")
    print(f"Processing complete for language '{args.lang_code}'!")
    print(f"All JSONL files saved in: {args.output_dir}")
    print(f"Successfully processed {len(jsonl_files)} input files into batches.")


if __name__ == "__main__":
    # Check for required dependencies
    try:
        import jsonlines
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install jsonlines")
        exit(1)
    
    main()