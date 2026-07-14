import json
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, Optional, Tuple

ENG_CODE = "eng_Latn"

LANG_CODE_TO_NAME = {
    "afr_Latn": "Afrikaans",
    "amh_Ethi": "Amharic",
    "ara_Arab": "Arabic",
    "aze_Latn": "Azerbaijani",
    "bak_Cyrl": "Bashkir",
    "bel_Cyrl": "Belarusian",
    "bem_Latn": "Bemba",
    "ben_Beng": "Bengali",
    "bod_Tibt": "Tibetan",
    "bos_Latn": "Bosnian",
    "bul_Cyrl": "Bulgarian",
    "cat_Latn": "Catalan",
    "ces_Latn": "Czech",
    "cym_Latn": "Welsh",
    "dan_Latn": "Danish",
    "deu_Latn": "German",
    "dzo_Tibt": "Dzongkha",
    "ell_Grek": "Greek",
    "eng_Latn": "English",
    "est_Latn": "Estonian",
    "ewe_Latn": "Ewe",
    "fao_Latn": "Faroese",
    "fas_Arab": "Persian",
    "fij_Latn": "Fijian",
    "fil_Latn": "Filipino",
    "fin_Latn": "Finnish",
    "fra_Latn": "French",
    "gle_Latn": "Irish",
    "glg_Latn": "Galician",
    "guj_Gujr": "Gujarati",
    "hau_Latn": "Hausa",
    "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi",
    "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian",
    "hye_Armn": "Armenian",
    "ibo_Latn": "Igbo",
    "ind_Latn": "Indonesian",
    "isl_Latn": "Icelandic",
    "ita_Latn": "Italian",
    "kan_Knda": "Kannada",
    "kat_Geor": "Georgian",
    "kaz_Cyrl": "Kazakh",
    "khm_Khmr": "Khmer",
    "kin_Latn": "Kinyarwanda",
    "kir_Cyrl": "Kyrgyz",
    "kor_Hang": "Korean",
    "lao_Laoo": "Lao",
    "lav_Latn": "Latvian",
    "lit_Latn": "Lithuanian",
    "ltz_Latn": "Luxembourgish",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mkd_Cyrl": "Macedonian",
    "mlg_Latn": "Malagasy",
    "mlt_Latn": "Maltese",
    "mri_Latn": "Maori",
    "msa_Latn": "Malay",
    "mya_Mymr": "Burmese",
    "nep_Deva": "Nepali",
    "nld_Latn": "Dutch",
    "nso_Latn": "Northern Sotho",
    "nya_Latn": "Nyanja",
    "pan_Guru": "Punjabi",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "pus_Arab": "Pashto",
    "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian",
    "sin_Sinh": "Sinhala",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "smo_Latn": "Samoan",
    "sna_Latn": "Shona",
    "snd_Arab": "Sindhi",
    "som_Latn": "Somali",
    "spa_Latn": "Spanish",
    "sqi_Latn": "Albanian",
    "srp_Latn": "Serbian",
    "ssw_Latn": "Swati",
    "swa_Latn": "Swahili",
    "swe_Latn": "Swedish",
    "tam_Taml": "Tamil",
    "tat_Cyrl": "Tatar",
    "tel_Telu": "Telugu",
    "tgk_Cyrl": "Tajik",
    "tir_Ethi": "Tigrinya",
    "tsn_Latn": "Tswana",
    "tur_Latn": "Turkish",
    "uig_Arab": "Uyghur",
    "ukr_Cyrl": "Ukrainian",
    "urd_Arab": "Urdu",
    "uzb_Latn": "Uzbek",
    "vie_Latn": "Vietnamese",
    "wol_Latn": "Wolof",
    "xho_Latn": "Xhosa",
    "yor_Latn": "Yoruba",
    "zho_Hans": "Chinese",
    "zul_Latn": "Zulu",
}


def parse_lang_pair(input_dir: Path, explicit_lang_pair: Optional[str]) -> Optional[Tuple[str, str]]:
    """Infer language codes from a folder such as eng_Latn-fra_Latn."""
    candidates = []
    if explicit_lang_pair:
        candidates.append(explicit_lang_pair)
    candidates.extend([input_dir.name, input_dir.parent.name])

    for name in candidates:
        if name.count("-") == 1:
            src_code, tgt_code = name.split("-", 1)
            if src_code and tgt_code:
                return src_code, tgt_code
    return None


def get_lang_name(lang_code: str) -> str:
    return LANG_CODE_TO_NAME.get(lang_code, lang_code)


def chunk_ranges(start: int, stop: int, group_size: int) -> Iterable[Tuple[int, int]]:
    for i in range(start, stop, group_size):
        yield i, min(i + group_size, stop)


def get_directions(
    src_code: str,
    tgt_code: str,
    total_rows: int,
    group_size: int,
    direction_mode: str,
) -> Iterable[Tuple[int, int, str, str]]:
    if direction_mode == "bidirectional_english" and ENG_CODE in {src_code, tgt_code}:
        other_code = tgt_code if src_code == ENG_CODE else src_code
        split_at = (total_rows + 1) // 2

        for start, stop in chunk_ranges(0, split_at, group_size):
            yield start, stop, ENG_CODE, other_code
        for start, stop in chunk_ranges(split_at, total_rows, group_size):
            yield start, stop, other_code, ENG_CODE
        return

    for start, stop in chunk_ranges(0, total_rows, group_size):
        yield start, stop, src_code, tgt_code

def process_single_parquet(file_info: Dict):
    """Worker function to process one parquet file."""
    input_path = file_info['input_path']
    output_path = file_info['output_path']
    src_code = file_info['src_code']
    tgt_code = file_info['tgt_code']
    lang_names = file_info['lang_names']
    n_concat = file_info['n_concat']
    source_text_col = file_info['source_text_col']
    target_text_col = file_info['target_text_col']
    direction_mode = file_info['direction_mode']
    
    try:
        df = pd.read_parquet(input_path)
        total_rows = len(df)
        processed_entries = 0
        
        # If n_concat is 0, keep each direction split as one entry.
        group_size = n_concat if n_concat > 0 else max(total_rows, 1)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for start, stop, first_code, second_code in get_directions(
                src_code,
                tgt_code,
                total_rows,
                group_size,
                direction_mode,
            ):
                chunk = df.iloc[start:stop]
                
                # Format each pair in the chunk
                formatted_pairs = []
                for _, row in chunk.iterrows():
                    text_by_code = {
                        src_code: row[source_text_col],
                        tgt_code: row[target_text_col],
                    }
                    pair = (
                        f"{lang_names[first_code]}: {text_by_code[first_code]}\n"
                        f"{lang_names[second_code]}: {text_by_code[second_code]}"
                    )
                    formatted_pairs.append(pair)
                
                # Join pairs with double newline to distinguish them.
                final_text = "\n\n".join(formatted_pairs)
                
                json.dump({"text": final_text}, f, ensure_ascii=False)
                f.write('\n')
                processed_entries += 1

        return f"SUCCESS: {input_path.name} | Rows: {total_rows} | JSONL Lines: {processed_entries}"
    except Exception as e:
        return f"FAILED: {input_path.name} | Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Parallel MaLa-LM to Megatron-LM JSONL converter")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--lang_pair", type=str, default=None, help="Language pair code, e.g. eng_Latn-fra_Latn")
    parser.add_argument("--source_lang", type=str, default=None, help="Full name override: e.g. English")
    parser.add_argument("--target_lang", type=str, default=None, help="Full name override: e.g. German")
    parser.add_argument("--concat_n_lines", type=int, default=1, help="0 for all lines in one block")
    parser.add_argument(
        "--direction_mode",
        choices=["original", "bidirectional_english"],
        default="bidirectional_english",
        help="bidirectional_english writes half English-X and half X-English for English pairs.",
    )
    parser.add_argument(
        "--source_text_col",
        type=str,
        default="source_text",
        help="Parquet column name for source text (e.g. source_text, src_text)",
    )
    parser.add_argument(
        "--target_text_col",
        type=str,
        default="target_text",
        help="Parquet column name for target text (e.g. target_text, tgt_text)",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes")
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_pair = parse_lang_pair(input_dir, args.lang_pair)
    if lang_pair is None:
        if args.source_lang is None or args.target_lang is None:
            raise ValueError(
                "Could not infer language pair from input folder. Pass --lang_pair or both "
                "--source_lang and --target_lang."
            )
        src_code, tgt_code = "source", "target"
    else:
        src_code, tgt_code = lang_pair

    lang_names = {
        src_code: args.source_lang or get_lang_name(src_code),
        tgt_code: args.target_lang or get_lang_name(tgt_code),
    }

    parquet_files = sorted(input_dir.glob("*.parquet"))
    
    # Prepare task list
    tasks = []
    for p in parquet_files:
        tasks.append({
            'input_path': p,
            'output_path': output_dir / p.with_suffix('.jsonl').name,
            'src_code': src_code,
            'tgt_code': tgt_code,
            'lang_names': lang_names,
            'n_concat': args.concat_n_lines,
            'source_text_col': args.source_text_col,
            'target_text_col': args.target_text_col,
            'direction_mode': args.direction_mode,
        })

    print(
        f"Language pair: {src_code} ({lang_names[src_code]}) -> "
        f"{tgt_code} ({lang_names[tgt_code]})"
    )
    print(f"Direction mode: {args.direction_mode}")
    print(f"Starting processing of {len(tasks)} files with {args.workers or 'max'} workers...")

    # Parallel Execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_single_parquet, tasks))

    # Print statistics as requested
    print("\n--- Processing Statistics ---")
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
