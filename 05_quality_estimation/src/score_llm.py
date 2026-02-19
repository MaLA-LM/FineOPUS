from __future__ import annotations

import argparse

from dataset.manifest import ManifestEntry
from dataset.mediator import get_dataset
from models.language_support import QwenLanguages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.common import (
    ensure_dataset_ready,
    load_examples,
    sanitize_model_tag,
    summarize_scores,
)
from src.llm_backend import build_lm, score_llm
from utils.args import add_common_scoring_args
from utils.cli import validate_args
from utils.frames import build_frames
from utils.runner import collect_directions, run_scoring

DEFAULT_LLM_MODEL = "Qwen/Qwen3-14B"
LLM_LANGUAGE_SUPPORT = QwenLanguages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with an LLM via vLLM + DSPy."
    )
    add_common_scoring_args(
        parser,
        batch_size_default=8,
        gpus_default=1,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=(
            "Served model name. "
            f"Registered aliases: {', '.join(supported_model_keys('llm'))}."
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


def resolve_model(name: str) -> tuple[str, str, str]:
    normalized = name.strip()
    if not normalized:
        raise ValueError("--model cannot be empty.")

    try:
        spec, model_key = resolve_model_spec(normalized, "llm")
        canonical = spec.model_id
    except ValueError:
        # Keep support for arbitrary served model names.
        canonical = normalized
        model_key = canonical

    request_name = (
        canonical[len("openai/") :] if canonical.startswith("openai/") else canonical
    )
    return canonical, request_name, model_key


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    lm,
    model_id: str,
    language_support: QwenLanguages,
    dataset,
):
    examples = load_examples(entry, args, dataset)
    scores = score_llm(
        examples,
        lm,
        args.max_retries,
        entry.src_lang,
        entry.tgt_lang,
        continue_on_error=args.continue_on_error,
    )
    src_lang_seen = language_support.support_status(entry.src_lang)
    tgt_lang_seen = language_support.support_status(entry.tgt_lang)
    mean_score, median_score = summarize_scores(scores)

    return build_frames(
        model_id,
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


def main() -> None:
    args = parse_args()
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    try:
        model_id, request_model, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    ensure_dataset_ready(args, dataset)
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

    model_tag = sanitize_model_tag(model_key)
    language_support = LLM_LANGUAGE_SUPPORT(dataset=dataset)
    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        lambda entry: score_entry(entry, args, lm, model_id, language_support, dataset),
    )


if __name__ == "__main__":
    main()
