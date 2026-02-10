from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import dspy
from pydantic import ValidationError

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.manifest import ManifestEntry
from dataset.mediator import Example, get_dataset
from models.gemma_qe import QEResult
from prompts.gemma_prompt import render_prompt
from src.scoring.args import add_common_scoring_args
from src.scoring.cli import resolve_output_path, validate_args
from src.scoring.frames import build_frames
from src.scoring.output_path import sanitize_model_tag
from src.scoring.runner import collect_directions, run_scoring

GEMMA_MODELS = {"gemma-3-12b-it": "gemma-3-12b-it"}

ALIASES = {
    "openai/gemma-3-12b-it": "gemma-3-12b-it",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with Gemma-3 12B via vLLM + DSPy."
    )
    add_common_scoring_args(parser)
    parser.add_argument(
        "--model",
        default="gemma-3-12b-it",
        help=(
            "Served model name. "
            f"Supported: {', '.join(sorted(GEMMA_MODELS.keys()))}."
        ),
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the OpenAI-compatible endpoint (optional).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per response.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per segment if the output fails validation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue scoring when model outputs fail validation; "
            "record NaN scores and report failures."
        ),
    )
    return parser.parse_args()


def resolve_model(name: str) -> tuple[str, str]:
    key = ALIASES.get(name, name)
    if key in GEMMA_MODELS:
        return GEMMA_MODELS[key], name
    supported = ", ".join(sorted(GEMMA_MODELS.keys()))
    raise ValueError(f"Unknown Gemma model '{name}'. Supported: {supported}.")


def overall_to_unit(score_0_to_100: int) -> float:
    if not 0 <= score_0_to_100 <= 100:
        raise ValueError(f"overall_0to100 out of range: {score_0_to_100}")
    return score_0_to_100 / 100.0


def build_lm(
    api_base: str, model: str, api_key: str | None, temperature: float, max_tokens: int
):
    api_key = api_key or "EMPTY"
    model_name = model if model.startswith("openai/") else f"openai/{model}"
    return dspy.LM(
        model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )


def score_gemma(
    examples: list[Example],
    lm,
    max_retries: int,
    src_lang: str,
    tgt_lang: str,
    continue_on_error: bool = False,
) -> list[float]:
    scores: list[float] = []
    invalid_rows: list[int] = []
    for idx, ex in enumerate(examples):
        prompt = render_prompt(src_lang, ex["src"], tgt_lang, ex["tgt"])
        last_error: Exception | None = None
        for _attempt in range(max_retries + 1):
            try:
                raw = lm(messages=[{"role": "user", "content": prompt}])
                # Extract text content from DSPy response
                if isinstance(raw, str):
                    text = raw
                elif isinstance(raw, (list, tuple)) and raw:  # List of responses
                    text = (
                        str(raw[0])
                        if not isinstance(raw[0], dict)
                        else raw[0].get("content", str(raw[0]))
                    )
                elif isinstance(raw, dict):  # Dict with content field
                    text = raw.get("content", json.dumps(raw))
                else:  # Fallback
                    text = str(raw)
                # Parse and validate JSON
                payload = json.loads(text.strip())
                parsed = QEResult.model_validate(payload)
                scores.append(overall_to_unit(int(parsed.overall_0to100)))
                last_error = None
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                continue
        if last_error is not None:
            if continue_on_error:
                scores.append(float("nan"))
                invalid_rows.append(idx)
                last_error = None
                continue
            raise RuntimeError(
                f"Row {idx} failed validation: {last_error}"
            ) from last_error
    if continue_on_error and invalid_rows:
        sample = ", ".join(str(idx) for idx in invalid_rows[:5])
        suffix = f" Example indices: {sample}." if sample else ""
        LOGGER.info(
            f"Invalid JSON rows: {len(invalid_rows)}.{suffix}",
            file=sys.stderr,
        )
    return scores


def summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    return float(statistics.fmean(scores)), float(statistics.median(scores))


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    lm,
    model_id: str,
    dataset,
):
    examples = dataset.load_parallel(
        entry.src_lang, entry.tgt_lang, split=entry.split, root=args.root
    )
    examples = dataset.limit_rows(examples, args.max_rows)
    scores = score_gemma(
        examples,
        lm,
        args.max_retries,
        entry.src_lang,
        entry.tgt_lang,
        continue_on_error=args.continue_on_error,
    )
    mean_score, median_score = summarize_scores(scores)

    return build_frames(
        model_id,
        dataset.id,
        entry.split,
        entry.src_lang,
        entry.tgt_lang,
        scores,
        examples,
        src_lang_seen="unknown",
        tgt_lang_seen="unknown",
        mean=mean_score,
        median=median_score,
    )


def main() -> None:
    args = parse_args()
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    try:
        model_id, request_model = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    if args.root is None:
        args.root = dataset.default_root
    if args.split not in dataset.split_values:
        supported = ", ".join(dataset.split_values)
        raise SystemExit(
            f"Unsupported split '{args.split}' for dataset '{dataset.id}'. "
            f"Supported: {supported}."
        )
    validate_args(args)

    directions = collect_directions(args, dataset)
    if not directions:
        print("No directions found.")
        return

    lm = build_lm(
        api_base=args.api_base,
        model=request_model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    model_tag = sanitize_model_tag(model_id)

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        resolve_output_path,
        lambda entry: score_entry(entry, args, lm, model_id, dataset),
    )


if __name__ == "__main__":
    main()
LOGGER = logging.getLogger(__name__)
