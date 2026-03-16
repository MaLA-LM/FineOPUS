"""
patching algorithm:
1. run discover_shards_needed on model directory,
2. for each shard, create a thread to patch the checkpoint and part files in the shard directory
3. wait for all threads to finish.
4. each thread, should read each part file in its directory, and do:
   - Read the file line by line
- When you see a `summary` row:
    - if you were already collecting a previous group, finalize that previous group first
    - start a new group buffer
    - store this summary row separately
- For each following `detail` row in that group:
    - fix `score` if needed (`null -> 0.0`)
    - fix seen/unseen flags if needed (using the language support module)
    - collect the corrected score into a list
    - store the corrected detail row in memory for that group
- When the next `summary` appears, or EOF is reached:
    - compute the new `mean` and `median` from the corrected detail scores
    - update the stored summary row
    - write:
        - corrected summary row
        - corrected detail rows
- Continue for the next group
"""

import json
import math
from pathlib import Path
from utils.logger import logger
from patch_results.discovery import Shardwork
from utils.io import ROW_TYPE_SUMMARY, ROW_TYPE_DETAIL
from models.language_support import PrometheusLanguages
from dataset.mediator import get_dataset
from src.common import summarize_scores
from concurrent.futures import ThreadPoolExecutor, as_completed

language_support = PrometheusLanguages(dataset=get_dataset(None))


def _is_language_seen(lang_code: str) -> bool:
    return language_support.support_status(lang_code)


def _finalize_block(block_rows: list[dict]) -> list[dict]:
    if not block_rows:
        raise RuntimeError("Cannot finalize empty block")

    summary_row = block_rows[0]
    detail_rows = block_rows[1:]

    if summary_row.get("row_type") != ROW_TYPE_SUMMARY:
        raise RuntimeError("Block does not start with a summary row")

    if not detail_rows:
        raise RuntimeError("No detail rows in block to summarize")

    for row in detail_rows:
        if row.get("row_type") != ROW_TYPE_DETAIL:
            raise RuntimeError("Non-detail row found inside block")

    scores = [
        (
            0.0
            if (
                row.get("score") is None
                or (isinstance(row.get("score"), float) and math.isnan(row["score"]))
            )
            else row["score"]
        )
        for row in detail_rows
    ]
    mean_score, median_score = summarize_scores(scores)

    summary_row["mean"] = mean_score
    summary_row["median"] = median_score

    src_lang = summary_row.get("src_lang")
    tgt_lang = summary_row.get("tgt_lang")
    if not src_lang or not tgt_lang:
        raise RuntimeError("Missing src_lang or tgt_lang in summary row")

    summary_row["src_lang_seen"] = _is_language_seen(src_lang)
    summary_row["tgt_lang_seen"] = _is_language_seen(tgt_lang)

    return [summary_row, *detail_rows]


def _write_rows(rows: list[dict], fh) -> None:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _process_one_part_file(part_file: Path) -> list[dict]:
    out_path = part_file.parent / f"{part_file.stem}-patched.jsonl"
    current_block: list[dict] = []
    summary_rows: list[dict] = []
    with part_file.open("r", encoding="utf-8") as in_f, out_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line_no, raw_line in enumerate(in_f, start=1):
            row = json.loads(raw_line)
            row_type = row.get("row_type")

            if row_type == ROW_TYPE_SUMMARY:
                if current_block:
                    processed_block = _finalize_block(current_block)
                    summary_rows.append(processed_block[0])
                    _write_rows(processed_block, out_f)
                current_block = [row]

            elif row_type == ROW_TYPE_DETAIL:
                if not current_block:
                    raise RuntimeError(
                        f"Detail row without preceding summary in {part_file}:{line_no}"
                    )

                score = row.get("score")
                if score is None or (isinstance(score, float) and math.isnan(score)):
                    row["score"] = 0.0

                current_block.append(row)

            else:
                raise RuntimeError(
                    f"Unknown row_type={row_type!r} in {part_file}:{line_no}"
                )

        if current_block:
            processed_block = _finalize_block(current_block)
            summary_rows.append(processed_block[0])
            _write_rows(processed_block, out_f)

    return summary_rows


def _process_shard(
    shard: Shardwork,
) -> tuple[int, int]:

    logger.info("Processing shard: %s, split: %s", shard.shard_path, shard.split)
    total_summary_rows = 0
    checkpoint_new_path = shard.checkpoint_path.parent / "checkpoint-patched.jsonl"
    with checkpoint_new_path.open("w", encoding="utf-8") as out_f:
        for part_file in sorted(shard.part_files):
            summary_rows = _process_one_part_file(part_file)
            total_summary_rows += len(summary_rows)
            _write_rows(summary_rows, out_f)

    return total_summary_rows, len(shard.part_files)


def patch(shards: list[Shardwork], max_workers: int | None = None) -> None:
    """Patch the checkpoint and part files for all shards using threads."""
    logger.info("Starting patch for %d shards", len(shards))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_shard, shard): shard for shard in shards}

        for future in as_completed(futures):
            shard = futures[future]
            try:
                summary_count, part_count = future.result()
                logger.info(
                    "Shard %s done – %d summary rows from %d parts",
                    shard.shard_path.name,
                    summary_count,
                    part_count,
                )
            except Exception:
                logger.exception("Shard %s failed", shard.shard_path)
                raise
