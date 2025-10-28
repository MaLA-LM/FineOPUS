import re
import json # Using json just for pretty printing the output

def parse_detail_string(detail_str):
    """
    Parses the detail part of the log line to extract structured data.
    e.g., "4122 alignment pairs, 78058 source tokens, 58772 target tokens (id 17)"
    """
    details = {
        'alignment_pairs': None,
        'source_tokens': None,
        'target_tokens': None,
        'id': None,
        'raw_text': detail_str.strip()
    }
    
    # Regex to find alignment pairs
    pairs_match = re.search(r'([\d,]+)\s+alignment\s+pairs', detail_str)
    if pairs_match:
        details['alignment_pairs'] = int(pairs_match.group(1).replace(',', ''))

    # Regex to find source tokens
    source_match = re.search(r'([\d,]+)\s+source\s+tokens', detail_str)
    if source_match:
        details['source_tokens'] = int(source_match.group(1).replace(',', ''))

    # Regex to find target tokens
    target_match = re.search(r'([\d,]+)\s+target\s+tokens', detail_str)
    if target_match:
        details['target_tokens'] = int(target_match.group(1).replace(',', ''))

    # Regex to find id
    id_match = re.search(r'\(id\s+([\d,]+)\)', detail_str)
    if id_match:
        details['id'] = int(id_match.group(1).replace(',', ''))
        
    return details


def parse_url_details(url):
    """
    Parses the URL to extract dataset, version, and language information.
    e.g., "https://.../OPUS-ada83/v1/moses/en-ru.txt.zip"
    """
    url_details = {
        'dataset': None,
        'version': None,
        'src_lang': None,
        'tgt_lang': None
    }
    
    # Regex to capture the required parts from the URL
    # https://.../OPUS-{dataset}/{version}/moses/{src_lang}-{tgt_lang}.txt.zip
    url_pattern = re.compile(
        r'https://.*\/OPUS-([^/]+)\/([^/]+)\/moses\/([^-]+)-([^.]+)\.txt\.zip'
    )
    
    match = url_pattern.search(url)
    
    if match:
        url_details['dataset'] = match.group(1)
        url_details['version'] = match.group(2)
        url_details['src_lang'] = match.group(3)
        url_details['tgt_lang'] = match.group(4)
        
    return url_details


def parse_log_data(log_string):
    """
    Parses the full log string and returns a list of dictionaries.
    """
    parsed_data = []
    
    # This regex captures the main components of each line
    # Group 1: File size (e.g., "271 KB", "3 MB")
    # Group 2: URL (https://...)
    # Group 3: The detail string after the "|"
    log_pattern = re.compile(
        r'^\s*([\d\s]+[KMGT]?B)\s+(https:\/\/[\S]+)\s*\|\s*(.+)$', 
        re.MULTILINE
    )
    
    matches = log_pattern.finditer(log_string)
    
    for match in matches:
        size, url, detail_raw = match.groups()
        
        url_info = parse_url_details(url)

        # Parse the extracted detail string
        structured_details = parse_detail_string(detail_raw)
        
        entry = {
            'file_size': size.strip(),
            'url': url.strip(),
            'details': structured_details
        }
        entry.update(url_info)

        parsed_data.append(entry)
        
    return parsed_data


if __name__ == "__main__":

    filename = "all_corpora_moses.txt"


    print(f"--- Parsing from File ({filename}) ---")

    # Read from the file and parse
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parsed_corpora_info = parse_log_data(content)
        
        print(f"Successfully parsed {len(parsed_corpora_info)} entries from file.")
        print("--- Parsed Data (from file) ---")
        # Pretty print the result
        # print(json.dumps(parsed_data, indent=2))
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("--- Parsed Data line 0 ---")
    print(json.dumps(parsed_corpora_info[0], indent=2))

    # --- Calculate total alignment pairs ---
    total_pairs = 0
    for entry in parsed_corpora_info:
        # Use .get() for safe access
        pairs = entry.get('details', {}).get('alignment_pairs')
        if pairs: # This checks if pairs is not None and not 0
            total_pairs += pairs
    print(f"\nTotal alignment pairs across all corpora: {total_pairs}")

    try:
        with open("OPUS_corpus_collection.json", 'w', encoding='utf-8') as f:
            json.dump(parsed_corpora_info, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved data to {filename}")
    except Exception as e:
        print(f"An unexpected error occurred while saving to JSON: {e}")