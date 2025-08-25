import orjson
import gzip
import numpy as np
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

def aggregate_confidence_stats(input_dir, output_file):
    """
    Aggregate confidence statistics from all JSON.gz files in the directory.
    
    Args:
        input_dir: Directory containing conf_stats_*.json.gz files
        output_file: Output file to save aggregated statistics
    """
    # Dictionary to store all confidence scores for each language
    lang_conf_all = defaultdict(list)
    
    # Get all JSON.gz files
    json_files = [f for f in os.listdir(input_dir) if f.startswith('conf_stats_') and f.endswith('.json.gz')]
    json_files.sort()  # Sort for consistent processing
    
    print(f"Found {len(json_files)} JSON.gz files to process")
    
    # Read and aggregate data from all files
    for filename in tqdm(json_files, desc="Processing files"):
        filepath = os.path.join(input_dir, filename)
        
        try:
            with gzip.open(filepath, 'rb') as f:
                data = orjson.loads(f.read())
                
            # Aggregate confidence scores for each language
            for lang_code, conf_scores in data.items():
                if isinstance(conf_scores, list) and len(conf_scores) > 0:
                    lang_conf_all[lang_code].extend(conf_scores)
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Aggregated data for {len(lang_conf_all)} languages")
    
    # Calculate statistics for each language
    stats = {}
    for lang, values in tqdm(lang_conf_all.items(), desc="Computing statistics"):
        arr = np.array(values)
        count = int(len(arr))
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if count > 1 else 0.0
        variance = float(np.var(arr, ddof=1)) if count > 1 else 0.0
        thr = max(0.3, min(0.9, median - std))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        
        # Additional statistics
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        
        stats[lang] = {
            "count": count,
            "mean": mean,
            "median": median,
            "std": std,
            "variance": variance,
            "thr": thr,
            "min": min_val,
            "max": max_val,
            "q25": q25,
            "q75": q75,
        }
    
    # Sort by count (descending) for easier analysis
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True))
    
    # Save results
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(sorted_stats, option=orjson.OPT_INDENT_2))
    
    print(f"Statistics saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total languages: {len(sorted_stats)}")
    total_samples = sum(stats['count'] for stats in sorted_stats.values())
    print(f"Total confidence scores: {total_samples:,}")
    
    # Top 10 languages by count
    print("\nTop 10 languages by sample count:")
    for i, (lang, stats) in enumerate(list(sorted_stats.items())[:10]):
        print(f"{i+1:2d}. {lang:15s}: {stats['count']:>10,} samples (mean: {stats['mean']:.4f})")
    
    return sorted_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate confidence statistics from JSON.gz files")
    parser.add_argument("--input_dir", required=True, help="Directory containing conf_stats_*.json.gz files")
    parser.add_argument("--output_file", required=True, help="Output JSON file to save aggregated statistics")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory")
        exit(1)
    
    # Run aggregation
    results = aggregate_confidence_stats(args.input_dir, args.output_file)
    
    print(f"\nAggregation complete! Results saved to {args.output_file}")