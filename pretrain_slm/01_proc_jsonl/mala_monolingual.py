import os
import jsonlines
from pathlib import Path
from typing import Generator, Dict, Any

def read_jsonl_files(data_dir: str, output_dir: str, partition: str, max_lines_per_file: int = 5000000):
    """
    Read JSONL files from nested directory structure and split into chunks.
    
    Args:
        data_dir: Root directory containing language code subdirectories
        output_dir: Directory to save the processed JSONL files
        partition: Which partition to process ('train' or 'validation')
        max_lines_per_file: Maximum number of lines per output file (default: 5,000,000)
    """
    # Validate partition parameter
    if partition not in ['train', 'validation']:
        raise ValueError("partition must be either 'train' or 'validation'")
    
    target_filename = f"{partition}.jsonl"
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Counter for output file naming
    output_file_counter = 1
    current_line_count = 0
    current_writer = None
    current_output_file = None
    
    def get_next_output_file():
        """Create next output file and return writer."""
        nonlocal output_file_counter, current_writer, current_output_file
        
        if current_writer:
            current_writer.close()
        
        output_filename = f"{partition}_{output_file_counter:04d}.jsonl"
        current_output_file = os.path.join(output_dir, output_filename)
        current_writer = jsonlines.open(current_output_file, mode='w')
        output_file_counter += 1
        print(f"Created new output file: {output_filename}")
        return current_writer
    
    try:
        # Initialize first output file
        writer = get_next_output_file()
        
        # Walk through the data directory
        for root, dirs, files in os.walk(data_dir):
            # Skip the root directory itself
            if root == data_dir:
                continue
                
            # Extract language code from directory path
            lang_code = os.path.basename(root)
            
            # Process only the specified partition file
            target_file_path = os.path.join(root, target_filename)
            
            # Check if the target file exists
            if not os.path.exists(target_file_path):
                print(f"  Skipping {lang_code}: {target_filename} not found")
                continue
            
            print(f"Processing language: {lang_code} - {partition} partition")
            
            try:
                with jsonlines.open(target_file_path, mode='r') as reader:
                    line_count_for_lang = 0
                    for line_data in reader:
                        # Add metadata about source
                        enhanced_data = {
                            'source_lang': lang_code,
                            'source_file': target_filename,
                            'partition': partition,
                            **line_data
                        }
                        
                        # Write to current output file
                        writer.write(enhanced_data)
                        current_line_count += 1
                        line_count_for_lang += 1
                        
                        # Check if we need to create a new output file
                        if current_line_count >= max_lines_per_file:
                            writer = get_next_output_file()
                            current_line_count = 0
                    
                    print(f"  Processed {line_count_for_lang} lines from {lang_code}")
                            
            except Exception as e:
                print(f"Error processing {target_file_path}: {str(e)}")
                continue
        
        print(f"\nProcessing complete!")
        print(f"Total output files created: {output_file_counter - 1}")
        print(f"Lines in final file: {current_line_count}")
        
    finally:
        # Clean up - close the final writer
        if current_writer:
            current_writer.close()

def get_file_statistics(data_dir: str, partition: str) -> Dict[str, Any]:
    """
    Get statistics about the input files before processing.
    
    Args:
        data_dir: Root directory containing language code subdirectories
        partition: Which partition to analyze ('train' or 'validation')
        
    Returns:
        Dictionary containing file statistics
    """
    # Validate partition parameter
    if partition not in ['train', 'validation']:
        raise ValueError("partition must be either 'train' or 'validation'")
    
    target_filename = f"{partition}.jsonl"
    
    stats = {
        'partition': partition,
        'total_files': 0,
        'languages': {},
        'missing_files': []
    }
    
    for root, dirs, files in os.walk(data_dir):
        if root == data_dir:
            continue
            
        lang_code = os.path.basename(root)
        target_file_path = os.path.join(root, target_filename)
        
        if os.path.exists(target_file_path):
            stats['languages'][lang_code] = {
                'file_exists': True,
                'file_path': target_file_path
            }
            stats['total_files'] += 1
        else:
            stats['missing_files'].append(lang_code)
    
    return stats

# Example usage
if __name__ == "__main__":
    # Set your directories here
    DATA_DIR = "/scratch/project_462000964/MaLA-LM/mala-monolingual-split/mala-monolingual-split"
    OUTPUT_DIR = "/scratch/project_462000964/FineOPUS/ablation_data/mala-mono"
    
    # Choose partition to process: 'train' or 'validation'
    PARTITION = "train"  # Change this to "validation" to process validation files
    
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, PARTITION)

    # Optional: Get statistics first
    print(f"Scanning input directory for {PARTITION} partition...")
    stats = get_file_statistics(DATA_DIR, PARTITION)
    print(f"Found {stats['total_files']} {PARTITION}.jsonl files across {len(stats['languages'])} languages")
    
    if stats['missing_files']:
        print(f"Missing {PARTITION}.jsonl files in: {', '.join(stats['missing_files'])}")
    
    for lang in stats['languages']:
        print(f"  {lang}: {PARTITION}.jsonl found")
    
    print(f"\nStarting processing of {PARTITION} partition...")
    
    # Process the files
    read_jsonl_files(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        partition=PARTITION,
        max_lines_per_file=20000000
    )