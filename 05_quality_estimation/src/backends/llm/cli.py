from __future__ import annotations

import argparse

from dataset.mediator import get_dataset
from execution import get_executor
from execution.flores_array.manifest import ManifestEntry
from models.model_registry import resolve_model_spec, supported_model_keys
from src.backends.llm.backend import (
    PROMPT_MODE_DETAILED,
    PROMPT_MODES,
    RESPONSE_FORMATS,
    build_engine,
)
from src.backends.llm.language_support import (
    auto_response_format,
    select_language_support,
)
from src.backends.llm.runner import score_entry
from src.common import ensure_dataset_ready, sanitize_model_tag
from utils.args import add_common_scoring_args, add_selected_executor_args
from utils.cli import validate_args
from utils.logger import logger

DEFAULT_LLM_MODEL = "Qwen/Qwen3-14B"

__all__ = ["DEFAULT_LLM_MODEL", "parse_args", "resolve_model", "main"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with an LLM via vLLM (offline)"
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
            "HuggingFace model repo or local path. "
            f"Registered aliases: {', '.join(supported_model_keys('llm'))}."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default=PROMPT_MODE_DETAILED,
        help=(
            "Prompt variant: 'detailed' (7 dimensions + overall), "
            "'simple' (overall score only), or 'batch' (batched "
            "7-dimension scoring, groups --batch-size segments per prompt). "
            "Default: detailed."
        ),
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
        help="Max tokens to generate per segment.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for segments that fail JSON validation.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype (bfloat16, float16, auto).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM KV cache.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable torch.compile / CUDA graphs (required on ROCm/MI250X).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help="Max tokens vLLM processes in one scheduler step (old server: 8192).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Max concurrent sequences in vLLM scheduler (old server: 32).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Cap the model's max context length. Without this, vLLM uses the "
            "model's full advertised length (e.g. 262144 for Qwen3) which "
            "wastes GPU memory on KV cache. 8192-16384 is plenty for QE."
        ),
    )
    parser.add_argument(
        "--response-format",
        choices=RESPONSE_FORMATS,
        default=None,
        help=(
            "Guided decoding mode: 'none' (free generation), "
            "'json_object' (force valid JSON, no schema), "
            "'json_schema' (enforce exact Pydantic schema per-token). "
            "Default: auto (json_schema for all models)."
        ),
    )
    add_selected_executor_args(parser)
    return parser.parse_args()


def resolve_model(name: str) -> tuple[str, str]:
    normalized = name.strip()
    if not normalized:
        raise ValueError("--model cannot be empty.")

    try:
        spec, model_key = resolve_model_spec(normalized, "llm")
        return spec.model_id, model_key
    except ValueError:
        return normalized, normalized


def main() -> None:
    args = parse_args()
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    try:
        model_id, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    ensure_dataset_ready(args, dataset)
    validate_args(args)

    executor = get_executor(args.execution)
    model_tag = sanitize_model_tag(f"{model_key}-{args.prompt_mode}")
    model_name = f"{model_key}-{args.prompt_mode}"
    language_support = select_language_support(model_key, dataset)

    if args.response_format is not None:
        logger.info(
            "response_format explicitly set to '%s' via CLI", args.response_format
        )
    else:
        args.response_format = auto_response_format(model_key)

    engine_holder: dict[str, object] = {}

    def run_entry(entry: ManifestEntry):
        engine = engine_holder.get("engine")
        if engine is None:
            engine = build_engine(
                model=model_id,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
            )
            engine_holder["engine"] = engine
            logger.info(
                "LLM scoring: model=%s prompt_mode=%s model_tag=%s max_tokens=%d response_format=%s",
                model_id,
                args.prompt_mode,
                model_tag,
                args.max_tokens,
                args.response_format,
            )
        return score_entry(
            entry,
            args,
            engine,
            model_name,
            language_support,
            dataset,
        )

    executor.run(
        args,
        dataset,
        model_tag,
        run_entry,
    )
