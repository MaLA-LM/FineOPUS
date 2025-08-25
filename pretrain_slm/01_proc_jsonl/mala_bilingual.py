import os
import json
import jsonlines
import random
import langcodes
from pathlib import Path
from typing import Dict, List, Any

def get_language_display_name(lang_code: str) -> str:
    """Convert language code to display name using langcodes package."""
    try:
        lang = langcodes.Language.get(lang_code.split('_')[0])
        return lang.display_name()
    except:
        # Fallback to language code if langcodes fails
        return lang_code

def load_templates(templates_file: str) -> List[Dict[str, Any]]:
    """Load templates from templates.jsonl file."""
    templates = []
    with jsonlines.open(templates_file, 'r') as reader:
        for template in reader:
            templates.append(template)
    return templates

def apply_template(src_text: str, trg_text: str, src_name: str, trg_name: str, template: str) -> str:
    """Apply a template to create the combined text field."""
    return template.format(
        src_line=src_text,
        trg_line=trg_text,
        src_name=src_name,
        trg_name=trg_name
    )

def process_jsonl_files(data_dir: str, output_dir: str, templates_file: str = "templates.jsonl", max_lines_per_file: int = 50000000):
    """
    Process JSONL files from the specified directory structure.
    
    Args:
        data_dir: Input directory containing language pair subdirectories
        output_dir: Output directory for processed files
        templates_file: Path to templates.jsonl file
        max_lines_per_file: Maximum lines per output file
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load templates
    templates = load_templates(templates_file)
    print(f"Loaded {len(templates)} templates")
    
    # Initialize variables for output file management
    current_output_file = None
    current_writer = None
    current_file_index = 0
    current_line_count = 0
    
    def get_new_output_file():
        nonlocal current_output_file, current_writer, current_file_index, current_line_count
        
        if current_writer:
            current_writer.close()
        
        current_file_index += 1
        output_filename = f"processed_data_{current_file_index:04d}.jsonl"
        current_output_file = os.path.join(output_dir, output_filename)
        current_writer = jsonlines.open(current_output_file, 'w')
        current_line_count = 0
        print(f"Created new output file: {output_filename}")
        return current_writer
    
    # Get the first output file
    writer = get_new_output_file()
    
    total_processed = 0
    
    # Process each language pair directory
    for lang_pair_dir in os.listdir(data_dir):
        lang_pair_path = os.path.join(data_dir, lang_pair_dir)
        
        if not os.path.isdir(lang_pair_path):
            continue
            
        print(f"Processing language pair directory: {lang_pair_dir}")
        
        # Extract source and target language codes
        if '-' in lang_pair_dir:
            src_lang_code, tgt_lang_code = lang_pair_dir.split('-', 1)
            src_lang_name = get_language_display_name(src_lang_code)
            tgt_lang_name = get_language_display_name(tgt_lang_code)
        else:
            print(f"Warning: Could not parse language pair from directory name: {lang_pair_dir}")
            continue
        
        # Process each JSONL file in the language pair directory
        for filename in os.listdir(lang_pair_path):
            if not filename.endswith('.jsonl'):
                continue
                
            file_path = os.path.join(lang_pair_path, filename)
            print(f"  Processing file: {filename}")
            
            try:
                with jsonlines.open(file_path, 'r') as reader:
                    for line_data in reader:
                        # Check if we need a new output file
                        if current_line_count >= max_lines_per_file:
                            writer = get_new_output_file()
                        
                        # Extract required fields
                        src_text = line_data.get('src_text', '')
                        tgt_text = line_data.get('tgt_text', '')
                        
                        # Skip if essential fields are missing
                        if not src_text or not tgt_text:
                            continue
                        
                        # Randomly select a template
                        template_data = random.choice(templates)
                        template = template_data['template']
                        
                        # Apply template to create combined text
                        combined_text = apply_template(
                            src_text, tgt_text, src_lang_name, tgt_lang_name, template
                        )
                        
                        # Create output record
                        output_record = {
                            'text': combined_text,
                            'src_lang': src_lang_code,
                            'src_text': src_text,
                            'tgt_lang': tgt_lang_code,
                            'tgt_text': tgt_text,
                            'src_lang_name': src_lang_name,
                            'tgt_lang_name': tgt_lang_name,
                            'template_used': template,
                            'token_count': template_data.get('token_count', 0),
                            'language_id': template_data.get('language_id', False)
                        }
                        
                        # Write to output file
                        writer.write(output_record)
                        current_line_count += 1
                        total_processed += 1
                        
                        if total_processed % 10000 == 0:
                            print(f"    Processed {total_processed} lines...")
                            
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
    
    # Close the final output file
    if current_writer:
        current_writer.close()
    
    print(f"\nProcessing complete!")
    print(f"Total lines processed: {total_processed}")
    print(f"Output files created: {current_file_index}")
    print(f"Files saved to: {output_dir}")

# Example usage
if __name__ == "__main__":
    # Set your directories
    DATA_DIR = "/scratch/project_462000964/MaLA-LM/mala-bilingual"
    OUTPUT_DIR = "/scratch/project_462000964/FineOPUS/ablation_data/mala-bi"
    templates_file_path = "../../assests/templates.jsonl"  # Path to templates file
    
    # Process the files
    process_jsonl_files(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        templates_file=templates_file_path,
        max_lines_per_file=100000000
    )