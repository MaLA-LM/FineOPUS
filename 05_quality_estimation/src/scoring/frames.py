from __future__ import annotations

from src.scoring.io import OUTPUT_COLUMNS, ROW_TYPE_DETAIL, ROW_TYPE_SUMMARY
import pandas as pd


def build_frames(
    model_name: str,
    dataset: str,
    split: str,
    src_lang: str,
    tgt_lang: str,
    scores: list[float],
    examples: list[dict[str, str]],
    *,
    src_lang_seen: str | bool,
    tgt_lang_seen: str | bool,
    mean: float | None,
    median: float | None,
):

    summary_frame = pd.DataFrame(
        [
            {
                "row_type": ROW_TYPE_SUMMARY,
                "model_name": model_name,
                "dataset": dataset,
                "split": split,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_lang_seen": src_lang_seen,
                "tgt_lang_seen": tgt_lang_seen,
                "mean": mean,
                "median": median,
            }
        ]
    )
    detail_frame = pd.DataFrame(
        {
            "row_type": [ROW_TYPE_DETAIL] * len(scores),
            "model_name": [model_name] * len(scores),
            "dataset": [dataset] * len(scores),
            "split": [split] * len(scores),
            "src_lang": [src_lang] * len(scores),
            "tgt_lang": [tgt_lang] * len(scores),
            "src_txt": [ex["src"] for ex in examples],
            "tgt_txt": [ex["tgt"] for ex in examples],
            "score": scores,
        }
    )
    return pd.concat(
        [summary_frame, detail_frame], ignore_index=True, sort=False
    ).reindex(columns=OUTPUT_COLUMNS)
