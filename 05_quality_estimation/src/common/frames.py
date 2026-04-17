from __future__ import annotations

from src.common.scoring_stats import summarize_scores


def build_scored_frames(
    dataset,
    entry,
    model_name: str,
    scores: list[float],
    examples: list,
    language_support,
):
    src_lang_seen = language_support.support_status(entry.src_lang)
    tgt_lang_seen = language_support.support_status(entry.tgt_lang)
    mean_score, median_score = summarize_scores(scores)

    return dataset.build_frames(
        model_name,
        dataset.id,
        entry.split,
        entry.src_lang,
        entry.tgt_lang,
        scores,
        examples,
        src_lang_seen=src_lang_seen,
        tgt_lang_seen=tgt_lang_seen,
        mean=mean_score,
        median=median_score,
    )
