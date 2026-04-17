from __future__ import annotations

from utils.frames import sanitize_scores
from utils.io import ROW_TYPE_DETAIL, ROW_TYPE_SUMMARY

# Column order for FLORES-200 output JSONL.  The stage writer does not
# enforce this — it writes whatever the DataFrame contains — but
# ``reindex(columns=...)`` keeps part files human-readable and lets
# downstream consumers rely on a stable column order.
OUTPUT_COLUMNS = [
    "row_type",
    "model_name",
    "dataset",
    "split",
    "src_lang",
    "tgt_lang",
    "src_lang_seen",
    "tgt_lang_seen",
    "mean",
    "median",
    "score",
    "src_txt",
    "tgt_txt",
]


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
    import pandas as pd

    sanitized = sanitize_scores(scores)

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
            "row_type": [ROW_TYPE_DETAIL] * len(sanitized),
            "model_name": [model_name] * len(sanitized),
            "dataset": [dataset] * len(sanitized),
            "split": [split] * len(sanitized),
            "src_lang": [src_lang] * len(sanitized),
            "tgt_lang": [tgt_lang] * len(sanitized),
            "src_txt": [ex["src"] for ex in examples],
            "tgt_txt": [ex["tgt"] for ex in examples],
            "score": sanitized,
        }
    )
    return pd.concat(
        [summary_frame, detail_frame], ignore_index=True, sort=False
    ).reindex(columns=OUTPUT_COLUMNS)
