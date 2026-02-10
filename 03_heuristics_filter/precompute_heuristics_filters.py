import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import os
import glob
import argparse
import time
import math
import unicodedata
import regex 
import numpy as np
import logging

try:
    import rapidfuzz
except ImportError:
    print("Warning: rapidfuzz not installed. Falling back to slow difflib.")
    import difflib as rapidfuzz

try:
    import Levenshtein
except ImportError:
    Levenshtein = None 

logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# 0. Multilingual Regex Patterns
# ==========================================

# 1. HTML Tags (Global)
RE_HTML = regex.compile(r'(?><[a-zA-Z][^>]*>)') 

# 2. Numerals (Unicode Aware)
RE_NUMERALS = regex.compile(r'\p{Nd}+')

# 3. Terminal Punctuation (Unicode Aware)
RE_PUNCT_TERM = regex.compile(r'[\p{STerm}.?!;…¿¡\u3002\u06D4\u0964\u0965]+')

# ==========================================
# 1. Helper: Digit Normalization
# ==========================================

# def normalize_digits_to_ascii(text):
#     if not text: return ""
#     return "".join([str(unicodedata.digit(c)) for c in text if c.isdigit()])

def normalize_digits_to_ascii(text):
    if not text: return ""
    digits = []
    for c in text:
        try:
            # Using digit() specifically, but catching errors
            digits.append(str(unicodedata.digit(c)))
        except ValueError:
            continue
    return "".join(digits)

# ==========================================
# Helper: Adaptive Word Statistics
# ==========================================

def compute_adaptive_word_stats(text_col, lang_col):
    """
    Computes word stats using Numpy for high-performance reduction of jagged arrays.
    """
    # 1. Character Length (Fast, Vectorized)
    char_len = pc.utf8_length(text_col)
    
    # 2. Tokenize by whitespace
    tokens = pc.utf8_split_whitespace(text_col)
    
    # 3. Space-based Word Count (Vectorized)
    word_len_space = pc.list_value_length(tokens)
    
    # 4. Space-based Max/Sum using Numpy reduceat (High Performance)
    token_lens = pc.utf8_length(tokens.values)
    
    # --- HELPER: Constant Array Creation (Corrected) ---
    def create_zeros(length):
        if length == 0: return pa.array([], type=pa.int64())
        return pa.repeat(0, length)

    def create_ones_float(length):
        if length == 0: return pa.array([], type=pa.float64())
        return pa.repeat(1.0, length)
    
    def create_false(length):
        if length == 0: return pa.array([], type=pa.bool_())
        return pa.repeat(False, length)
    # ---------------------------------------------------

    if len(tokens) > 0 and len(token_lens) > 0:
        # Extract buffers to Numpy (Zero-copy where possible)
        offsets = tokens.offsets.to_numpy()
        flat_values = token_lens.to_numpy()
        
        # SUM: reduceat is extremely fast for jagged array summation
        sum_result = np.add.reduceat(flat_values, offsets[:-1])
        
        # MAX: reduceat calculates max per slice
        max_result = np.maximum.reduceat(flat_values, offsets[:-1])
        
        # Mask empty rows (fix reduceat behavior on empty slices)
        counts = word_len_space.to_numpy()
        mask_empty = (counts == 0)
        
        sum_result[mask_empty] = 0
        max_result[mask_empty] = 0
        
        max_word_len = pa.array(max_result)
        sum_word_len = pa.array(sum_result)
        
    else:
        # Fallback for completely empty batch
        max_word_len = create_zeros(len(text_col))
        sum_word_len = create_zeros(len(text_col))
        
    # 5. Determine Script Type
    if lang_col is not None:
        cjk_pattern = r"_(Hans|Hant|Jpan|Kore|Hang|Thai|Laoo|Khmr|Mymr)$"
        is_no_space = pc.match_substring_regex(lang_col, cjk_pattern)
    else:
        is_no_space = create_false(len(text_col))

    # 6. Final Logic

    final_word_len = pc.if_else(is_no_space, char_len, word_len_space)

    # --- FIX: Safe Denominator Calculation ---
    # We must cast the replacement value (1) to match word_len_space type (usually int32)
    # or cast the array to int64 first. Safest is to generate a scalar of the correct type.
    
    # Get the type of the word_len_space array (likely int32)
    target_type = word_len_space.type
    
    # Create a scalar '1' of that exact type
    one_scalar = pa.scalar(1, type=target_type)
    
    safe_denom = pc.replace_with_mask(word_len_space, pc.equal(word_len_space, 0), one_scalar)

    avg_word_len_space = pc.divide(sum_word_len, safe_denom)
    
    final_avg_word_len = pc.if_else(is_no_space, 
                                    create_ones_float(len(text_col)),
                                    avg_word_len_space)

    return final_word_len, max_word_len, final_avg_word_len

# ==========================================
# 2. Complex Filter Logic
# ==========================================

def compute_html_tag(text_list):
    return [bool(RE_HTML.search(t)) if t else False for t in text_list]

def compute_terminal_punctuation(src_list, trg_list):
    scores = []
    for src, trg in zip(src_list, trg_list):
        if not src or not trg:
            scores.append(-20.0)
            continue
        s_matches = RE_PUNCT_TERM.findall(src)
        t_matches = RE_PUNCT_TERM.findall(trg)
        spun = sum(len(m) for m in s_matches)
        tpun = sum(len(m) for m in t_matches)
        score = abs(spun - tpun)
        if spun > 1: score += (spun - 1)
        if tpun > 1: score += (tpun - 1)
        scores.append(-math.log(score + 1))
    return scores

def compute_non_zero_numerals(src_list, trg_list):
    scores = []
    for src, trg in zip(src_list, trg_list):
        if not src or not trg:
            scores.append(0.0)
            continue
        
        src_raw = "".join(RE_NUMERALS.findall(src))
        trg_raw = "".join(RE_NUMERALS.findall(trg))
        
        def get_safe_digits(raw_str):
            res = []
            for c in raw_str:
                try:
                    d = unicodedata.digit(c)
                    if d != 0:
                        res.append(str(d))
                except ValueError:
                    # Skip characters that claim to be Nd but lack a digit value
                    continue
            return "".join(res)

        src_norm = get_safe_digits(src_raw)
        trg_norm = get_safe_digits(trg_raw)
        
        if not src_norm and not trg_norm:
            scores.append(1.0)
        elif not src_norm or not trg_norm:
            scores.append(0.0)
        else:
            if hasattr(rapidfuzz, 'fuzz'):
                scores.append(rapidfuzz.fuzz.ratio(src_norm, trg_norm) / 100.0)
            else:
                matcher = rapidfuzz.SequenceMatcher(None, src_norm, trg_norm)
                scores.append(matcher.ratio())
    return scores

def compute_lcs_ratio(src_list, trg_list):
    scores = []
    for src, trg in zip(src_list, trg_list):
        if not src or not trg:
            scores.append(0.0)
            continue
        min_len = min(len(src), len(trg))
        if min_len == 0:
            scores.append(0.0)
            continue
        if hasattr(rapidfuzz.distance, 'LCSseq'):
             lcs_len = rapidfuzz.distance.LCSseq.similarity(src, trg)
        elif hasattr(rapidfuzz, 'SequenceMatcher'):
             matcher = rapidfuzz.SequenceMatcher(None, src, trg)
             match = matcher.find_longest_match(0, len(src), 0, len(trg))
             lcs_len = match.size
        else:
             scores.append(0.0)
             continue
        scores.append(lcs_len / min_len)
    return scores

def compute_levenshtein_similarity(src_list, trg_list):
    scores = []
    for src, trg in zip(src_list, trg_list):
        src = src or ""
        trg = trg or ""
        max_len = max(len(src), len(trg))
        if max_len == 0:
            scores.append(1.0)
            continue
        if Levenshtein:
            dist = Levenshtein.distance(src, trg)
        elif hasattr(rapidfuzz.distance, 'Levenshtein'):
            dist = rapidfuzz.distance.Levenshtein.distance(src, trg)
        else:
            dist = max_len
        scores.append(1.0 - (dist / max_len))
    return scores

def compute_repetition(text_list, threshold=2, min_len=3, max_len=100):
    scores = []
    rstring = (f'(?>(\\S.{{{min_len-1},{max_len}}}?))'
               f'(?: *\\1){{{threshold},}}')
    pattern = regex.compile(rstring)
    for text in text_list:
        if not text:
            scores.append(0)
            continue
        try:
            match = pattern.search(text, timeout=0.1)
            if match:
                full_match = match.group(0)
                repeated_segment = match.group(1)
                scores.append(full_match.count(repeated_segment) - 1)
            else:
                scores.append(0)
        except (TimeoutError, Exception):
            scores.append(0)
    return scores

def compute_regexp_match(text_list, compiled_pattern):
    if not compiled_pattern:
        return [False] * len(text_list)
    return [bool(compiled_pattern.search(t)) if t else False for t in text_list]

# ==========================================
# 3. Main Processing Loop
# ==========================================

def process_dataset(input_dir, output_dir, src_col, trg_col, regex_pattern_str=None, stat_file_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- STAT FILE LOGIC: Load existing ---
    completed_files = set()
    if stat_file_path and os.path.exists(stat_file_path):
        print(f"Loading processing history from {stat_file_path}...")
        with open(stat_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                completed_files.add(line.strip())
        print(f"Resuming: {len(completed_files)} files already completed.")

    user_regex = None
    if regex_pattern_str:
        try:
            user_regex = regex.compile(regex_pattern_str)
        except Exception as e:
            logging.error(f"Error compiling user regex: {e}")
            return

    files = glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True)
    
    print(f"--- Found {len(files)} files in input directory ---")
    total_start = time.time()

    for i, file_path in enumerate(files):
        file_name = os.path.basename(file_path)
        
        # --- STAT FILE LOGIC: Check skip ---
        if file_name in completed_files:
            print(f"[{i+1}/{len(files)}] Skipping {file_name} (Already in stat file)")
            continue

        out_path = os.path.join(output_dir, file_name)
        print(f"[{i+1}/{len(files)}] Processing {file_name}...")
        
        try:
            pq_file = pq.ParquetFile(file_path)
            writer = None
            
            for batch in pq_file.iter_batches():
                if src_col not in batch.column_names or trg_col not in batch.column_names:
                    continue

                src = batch[src_col]
                trg = batch[trg_col]
                src_lang = batch['src_lang'] if 'src_lang' in batch.column_names else None
                trg_lang = batch['trg_lang'] if 'trg_lang' in batch.column_names else None

                # --- PHASE A: Adaptive Vectorized Stats ---
                src_char_len = pc.utf8_length(src)
                trg_char_len = pc.utf8_length(trg)

                src_word_len, src_max_word_len, src_avg_word_len = compute_adaptive_word_stats(src, src_lang)
                trg_word_len, trg_max_word_len, trg_avg_word_len = compute_adaptive_word_stats(trg, trg_lang)

                def calc_ratio(c1, c2):
                    mx = pc.max_element_wise(c1, c2)
                    mn = pc.min_element_wise(c1, c2)
                    # --- FIX: Type-Safe Mask Replacement for Ratio ---
                    target_type = mn.type
                    one_scalar = pa.scalar(1, type=target_type)
                    safe_mn = pc.replace_with_mask(mn, pc.equal(mn, 0), one_scalar)
                    return pc.divide(mx, safe_mn)

                char_ratio = calc_ratio(src_char_len, trg_char_len)
                word_ratio = calc_ratio(src_word_len, trg_word_len)

                # --- PHASE B: Complex Stats (Python UDFs) ---
                src_pylist = src.to_pylist()
                trg_pylist = trg.to_pylist()

                new_columns = {
                    'src_char_len': src_char_len,
                    'trg_char_len': trg_char_len,
                    'src_word_len': src_word_len,
                    'trg_word_len': trg_word_len,
                    'src_max_word_len': src_max_word_len,
                    'trg_max_word_len': trg_max_word_len,
                    'src_avg_word_len': src_avg_word_len,
                    'trg_avg_word_len': trg_avg_word_len,
                    'char_len_ratio': char_ratio,
                    'word_len_ratio': word_ratio,
                    'filter_html_src': pa.array(compute_html_tag(src_pylist)),
                    'filter_html_trg': pa.array(compute_html_tag(trg_pylist)),
                    'score_term_punct': pa.array(compute_terminal_punctuation(src_pylist, trg_pylist)),
                    'score_numerals': pa.array(compute_non_zero_numerals(src_pylist, trg_pylist)),
                    'score_lcs_ratio': pa.array(compute_lcs_ratio(src_pylist, trg_pylist)),
                    'score_levenshtein': pa.array(compute_levenshtein_similarity(src_pylist, trg_pylist)),
                    'score_repeat_src': pa.array(compute_repetition(src_pylist)),
                    'score_repeat_trg': pa.array(compute_repetition(trg_pylist)),
                    'filter_regex_src': pa.array(compute_regexp_match(src_pylist, user_regex))
                }

                # --- PHASE C: Write ---
                for name, col_data in new_columns.items():
                    batch = batch.append_column(name, col_data)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, batch.schema, compression='SNAPPY')
                
                writer.write_batch(batch)
            
            if writer:
                writer.close()
                
        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")
            if os.path.exists(out_path):
                try: os.remove(out_path)
                except: pass
    # --- STAT FILE LOGIC: Update record ---
    # Only record if processing finished successfully
    if stat_file_path:
        with open(stat_file_path, 'a', encoding='utf-8') as f:
            f.write(os.path.basename(os.path.normpath(input_dir)) + "\n")

    print(f"--- Done. Total time: {time.time() - total_start:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute multilingual quality filters.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--src_col", default="source_text")
    parser.add_argument("--trg_col", default="target_text")
    parser.add_argument("--regex_pattern", default=None)
    parser.add_argument("--stats_file", default="processed_files.csv", 
                        help="Path to file recording completed basenames to allow resuming.")
    
    args = parser.parse_args()

    base_name = os.path.basename(os.path.normpath(args.data_dir))
    final_output_path = os.path.join(args.out_dir, base_name)

    process_dataset(
        args.data_dir, 
        final_output_path, 
        src_col=args.src_col,
        trg_col=args.trg_col,
        regex_pattern_str=args.regex_pattern,
        stat_file_path=args.stats_file
    )