#!/usr/bin/env python3
"""Three-shot eng<->x machine-translation evaluation with offline vLLM."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dev_splits: tuple[str, ...]
    test_split: str = "test"


DATASETS = {
    "FLORES-200": DatasetSpec("FLORES-200", ("validation", "dev")),
    "NTREX-128": DatasetSpec("NTREX-128", ()),
    "BOUQuET_Sentence": DatasetSpec("BOUQuET_Sentence", ("dev", "validation")),
}

# Dataset naming is not fully consistent with the training manifest. Keep the
# mapping dataset-specific so a code is never silently redirected in a dataset
# where its exact code exists. A tuple may intentionally expand one training
# language into multiple separately reported benchmark varieties.
DATASET_LANGUAGE_MAP = {
    "FLORES-200": {
        "ara_Arab": ("arb_Arab",),
        "aze_Latn": ("azj_Latn",),
        "fas_Arab": ("pes_Arab",),
        "fil_Latn": ("tgl_Latn",),
        "lav_Latn": ("lvs_Latn",),
        "mlg_Latn": ("plt_Latn",),
        "msa_Latn": ("zsm_Latn",),
        "nep_Deva": ("npi_Deva",),
        "pus_Arab": ("pbt_Arab",),
        "sqi_Latn": ("als_Latn",),
        "swa_Latn": ("swh_Latn",),
        "uzb_Latn": ("uzn_Latn",),
    },
    "NTREX-128": {
        "ara_Arab": ("arb_Arab",),
    },
    "BOUQuET_Sentence": {
        "ara_Arab": ("apc_Arab", "arz_Arab"),
        "aze_Latn": ("azj_Latn",),
        "est_Latn": ("ekk_Latn",),
        "fas_Arab": ("pes_Arab",),
        "fil_Latn": ("tgl_Latn",),
        "kor_Hang": ("kor_Kore",),
        "lav_Latn": ("lvs_Latn",),
        "mlg_Latn": ("plt_Latn",),
        "msa_Latn": ("zsm_Latn",),
        "nep_Deva": ("npi_Deva",),
        "por_Latn": ("por_Latn_braz1246",),
        "pus_Arab": ("pbt_Arab",),
        "sqi_Latn": ("als_Latn",),
        "swa_Latn": ("swh_Latn",),
        "uzb_Latn": ("uzn_Latn",),
        "zho_Hans": ("cmn_Hans",),
    },
}

LANGUAGE_NAME_OVERRIDES = {
    "arb": "Arabic",
    "ara": "Arabic",
    "cmn": "Chinese",
    "eng": "English",
    "fas": "Persian",
    "swa": "Swahili",
    "swh": "Swahili",
    "zho": "Chinese",
}

SUMMARY_FIELDS = [
    "status", "model", "checkpoint", "dataset", "source", "target", "language",
    "num_examples", "bleu", "chrf", "comet", "seconds", "comet_seconds",
    "few_shot", "limit", "bleu_tokenizer", "bleu_signature", "chrf_word_order",
    "comet_model", "comet_num_examples", "reason", "result_dir",
]

DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-name", default="unknown")
    parser.add_argument("--checkpoint-name", default="unknown")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", required=True, help="Comma-separated target language codes")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset names")
    parser.add_argument("--few-shot", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, help="Score at most this many rows per direction")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--comet-model", default=DEFAULT_COMET_MODEL)
    parser.add_argument("--comet-batch-size", type=int, default=8)
    parser.add_argument("--comet-gpus", type=int, default=1)
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.epilog = (
        "Saved predictions can be rescored with the 'rescore' and "
        "'submit-rescore' commands."
    )
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(42).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def language_candidates(dataset_name: str, code: str) -> tuple[str, ...]:
    return DATASET_LANGUAGE_MAP.get(dataset_name, {}).get(code, (code,))


def resolve_language_dirs(dataset_dir: Path, code: str, test_split: str) -> list[tuple[str, Path]]:
    resolved = []
    for candidate in language_candidates(dataset_dir.name, code):
        lang_dir = dataset_dir / candidate
        if (lang_dir / f"{test_split}.parquet").is_file():
            resolved.append((candidate, lang_dir))
    return resolved


def selected_specs(names: Iterable[str]) -> list[DatasetSpec]:
    specs = []
    for name in names:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
        specs.append(DATASETS[name])
    return specs


def discover_files(
    data_root: Path, specs: list[DatasetSpec], languages: list[str]
) -> tuple[list[tuple[DatasetSpec, str, str, Path, Path]], list[dict[str, Any]]]:
    cases = []
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        dataset_dir = data_root / spec.name
        english_dirs = resolve_language_dirs(dataset_dir, "eng_Latn", spec.test_split)
        if not english_dirs:
            raise FileNotFoundError(f"{spec.name}: eng_Latn/{spec.test_split}.parquet is missing")
        _, eng_dir = english_dirs[0]
        for model_code in languages:
            targets = resolve_language_dirs(dataset_dir, model_code, spec.test_split)
            if not targets:
                skipped.append(
                    {"status": "skipped", "dataset": spec.name, "language": model_code, "reason": "language_not_available"}
                )
                continue
            for data_code, lang_dir in targets:
                cases.append((spec, model_code, data_code, eng_dir, lang_dir))
    return cases, skipped


def runtime_preflight(use_comet: bool) -> None:
    missing = []
    modules = ["pandas", "pyarrow", "sacrebleu", "transformers", "vllm"]
    if use_comet:
        modules.append("comet")
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # imports may fail due to incompatible versions
            missing.append(f"{module}: {exc}")
    if missing:
        details = "\n  ".join(missing)
        raise RuntimeError(
            "Evaluation environment is not usable:\n  "
            + details
            + "\nRun: python -m pip install -r evaluation/requirements-mt.txt"
        )


def data_preflight(
    cases: list[tuple[DatasetSpec, str, str, Path, Path]], few_shot: int
) -> None:
    pointers: set[Path] = set()
    missing_dev: list[str] = []
    for spec, model_code, _, eng_dir, lang_dir in cases:
        paths = [eng_dir / f"{spec.test_split}.parquet", lang_dir / f"{spec.test_split}.parquet"]
        if spec.dev_splits:
            found_dev = False
            for split in spec.dev_splits:
                source_dev = eng_dir / f"{split}.parquet"
                target_dev = lang_dir / f"{split}.parquet"
                if source_dev.is_file() and target_dev.is_file():
                    paths.extend((source_dev, target_dev))
                    found_dev = True
                    break
            if not found_dev and few_shot:
                missing_dev.append(f"{spec.name}/{model_code}")
        pointers.update(path for path in paths if is_lfs_pointer(path))
    if missing_dev:
        raise FileNotFoundError("Few-shot split missing for: " + ", ".join(missing_dev[:20]))
    if pointers:
        examples = "\n  ".join(str(path) for path in sorted(pointers)[:8])
        raise RuntimeError(
            f"Found {len(pointers)} Git LFS pointer files instead of parquet data, for example:\n  {examples}\n"
            "Fetch the dataset LFS objects on a login node before submitting GPU jobs."
        )


def read_text_column(path: Path, language_code: str) -> list[str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if is_lfs_pointer(path):
        raise RuntimeError(f"Git LFS pointer is not parquet data: {path}")
    table = pq.read_table(path)
    preferred = (
        language_code,
        f"sentence_{language_code}",
        "sentence",
        "text",
        "translation",
        "target",
        "source",
    )
    for name in preferred:
        if name in table.column_names:
            column = table[name]
            if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
                return ["" if value is None else str(value) for value in column.to_pylist()]
    string_columns = [
        name
        for name in table.column_names
        if pa.types.is_string(table[name].type) or pa.types.is_large_string(table[name].type)
    ]
    if len(string_columns) == 1:
        return ["" if value is None else str(value) for value in table[string_columns[0]].to_pylist()]
    raise ValueError(f"Cannot identify a unique text column in {path}; schema: {table.schema}")


def choose_dev_paths(spec: DatasetSpec, source_dir: Path, target_dir: Path) -> tuple[Path, Path] | None:
    for split in spec.dev_splits:
        source = source_dir / f"{split}.parquet"
        target = target_dir / f"{split}.parquet"
        if source.is_file() and target.is_file():
            return source, target
    return None


def parallel_rows(source: list[str], target: list[str], label: str) -> list[tuple[int, str, str]]:
    if len(source) != len(target):
        raise ValueError(f"Unaligned {label}: source has {len(source)} rows, target has {len(target)}")
    return [(index, src.strip(), tgt.strip()) for index, (src, tgt) in enumerate(zip(source, target)) if src and tgt]


def language_name(code: str) -> str:
    alpha3 = code.split("_", 1)[0].split("-", 1)[0]
    if alpha3 in LANGUAGE_NAME_OVERRIDES:
        return LANGUAGE_NAME_OVERRIDES[alpha3]
    try:
        import pycountry

        language = pycountry.languages.get(alpha_3=alpha3)
        if language is not None:
            return str(language.name)
    except Exception:
        pass
    return code


def make_prompt(source_code: str, target_code: str, shots: list[tuple[int, str, str]], text: str) -> str:
    source_name = language_name(source_code)
    target_name = language_name(target_code)
    lines = [f"Translate from {source_name} to {target_name}. Output only the translation.", ""]
    for _, source, target in shots:
        lines.extend((f"{source_name}: {source}", f"{target_name}: {target}", ""))
    lines.extend((f"{source_name}: {text}", f"{target_name}:"))
    return "\n".join(lines)


def clean_prediction(text: str, target_code: str) -> str:
    prediction = text.strip().splitlines()[0].strip() if text.strip() else ""
    labels = (language_name(target_code), target_code)
    for label in labels:
        prediction = re.sub(rf"^\s*{re.escape(label)}\s*:\s*", "", prediction, flags=re.IGNORECASE)
    return prediction.strip()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_json(output_dir / "summary.json", rows)
    temporary = output_dir / "summary.tsv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_dir / "summary.tsv")


def load_direction_data(
    spec: DatasetSpec,
    source_code: str,
    target_code: str,
    source_dir: Path,
    target_dir: Path,
    few_shot: int,
    limit: int | None,
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    source_test = read_text_column(source_dir / f"{spec.test_split}.parquet", source_code)
    target_test = read_text_column(target_dir / f"{spec.test_split}.parquet", target_code)
    test_rows = parallel_rows(source_test, target_test, f"{spec.name} test")
    dev_paths = choose_dev_paths(spec, source_dir, target_dir)
    if dev_paths:
        source_dev = read_text_column(dev_paths[0], source_code)
        target_dev = read_text_column(dev_paths[1], target_code)
        shots = parallel_rows(source_dev, target_dev, f"{spec.name} dev")[:few_shot]
    else:
        shots = test_rows[:few_shot]
        test_rows = test_rows[few_shot:]
    if len(shots) != few_shot:
        raise ValueError(f"{spec.name} {source_code}->{target_code}: requested {few_shot} shots, found {len(shots)}")
    if limit is not None:
        test_rows = test_rows[:limit]
    if not test_rows:
        raise ValueError(f"{spec.name} {source_code}->{target_code}: no test rows remain")
    return shots, test_rows


def evaluate(args: argparse.Namespace) -> int:
    if args.comet_batch_size < 1:
        raise ValueError("--comet-batch-size must be positive")
    if args.comet_gpus < 0:
        raise ValueError("--comet-gpus cannot be negative")
    languages = split_csv(args.languages)
    specs = selected_specs(split_csv(args.datasets))
    if not languages:
        raise ValueError("--languages is empty")
    cases, skipped = discover_files(args.data_root, specs, languages)
    if not cases:
        raise RuntimeError("No requested language is available in the selected datasets")
    data_preflight(cases, args.few_shot)
    if args.check_runtime:
        runtime_preflight(not args.no_comet)
    if args.preflight_only:
        print(f"Preflight passed: {len(cases)} dataset/language cases, {len(skipped)} unavailable combinations")
        return 0
    if args.model is None or not args.model.is_dir():
        raise FileNotFoundError(f"Converted model directory does not exist: {args.model}")

    from sacrebleu.metrics import BLEU, CHRF
    from vllm import LLM, SamplingParams

    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["\n", "<|im_end|>", "<|endoftext|>"],
    )
    bleu = BLEU(tokenize="flores200", effective_order=True)
    chrf = CHRF(word_order=2)
    summary: list[dict[str, Any]] = skipped

    for spec, model_code, data_code, eng_dir, lang_dir in cases:
        for source_code, target_code, source_dir, target_dir in (
            ("eng_Latn", data_code, eng_dir, lang_dir),
            (data_code, "eng_Latn", lang_dir, eng_dir),
        ):
            direction = f"{source_code}__to__{target_code}"
            result_dir = args.output_dir / spec.name / direction
            metrics_path = result_dir / "metrics.json"
            if metrics_path.is_file() and not args.overwrite:
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                if existing.get("few_shot") == args.few_shot and existing.get("limit") == args.limit:
                    summary.append(existing)
                    print(f"Skipping matching completed result: {spec.name}/{direction}")
                    continue
            shots, test_rows = load_direction_data(
                spec, source_code, target_code, source_dir, target_dir, args.few_shot, args.limit
            )
            prompts = [make_prompt(source_code, target_code, shots, source) for _, source, _ in test_rows]
            started = time.monotonic()
            outputs = llm.generate(prompts, sampling, use_tqdm=True)
            elapsed = time.monotonic() - started
            predictions = [clean_prediction(output.outputs[0].text, target_code) for output in outputs]
            references = [target for _, _, target in test_rows]
            prediction_rows = [
                {
                    "id": row_id,
                    "dataset": spec.name,
                    "source_language": source_code,
                    "target_language": target_code,
                    "source": source,
                    "reference": reference,
                    "prediction": prediction,
                    "prompt": prompt,
                }
                for (row_id, source, reference), prediction, prompt in zip(test_rows, predictions, prompts)
            ]
            atomic_jsonl(result_dir / "predictions.jsonl", prediction_rows)
            metrics = {
                "status": "complete" if args.limit is None else "smoke",
                "model": args.model_name,
                "checkpoint": args.checkpoint_name,
                "dataset": spec.name,
                "source": source_code,
                "target": target_code,
                "language": model_code,
                "num_examples": len(predictions),
                "bleu": round(bleu.corpus_score(predictions, [references]).score, 4),
                "chrf": round(chrf.corpus_score(predictions, [references]).score, 4),
                "seconds": round(elapsed, 3),
                "reason": "",
                "result_dir": str(result_dir),
                "few_shot": args.few_shot,
                "limit": args.limit,
                "bleu_tokenizer": "flores200",
                "bleu_signature": str(bleu.get_signature()),
                "chrf_word_order": 2,
            }
            atomic_json(metrics_path, metrics)
            summary.append(metrics)
            write_summary(args.output_dir, summary)
            print(
                f"Completed {spec.name}/{direction}: n={len(predictions)} "
                f"BLEU={metrics['bleu']:.4f} chrF++={metrics['chrf']:.4f}"
            )

    expected_status = "complete" if args.limit is None else "smoke"
    completed = sum(row.get("status") == expected_status for row in summary)
    if completed == 0:
        raise RuntimeError("No translation direction was evaluated")
    write_summary(args.output_dir, summary)
    marker = "_SUCCESS" if args.limit is None else f"_SMOKE_SUCCESS_limit_{args.limit}"
    if args.no_comet:
        (args.output_dir / marker).write_text(
            f"completed_directions={completed}\n", encoding="utf-8"
        )
        print(f"Evaluation complete: {completed} directions; summary: {args.output_dir / 'summary.tsv'}")
        return 0

    (args.output_dir / marker).unlink(missing_ok=True)
    print(f"Inference complete: {completed} directions; restarting process for COMET scoring")
    sys.stdout.flush()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "score-comet",
        "--checkpoint-dir",
        str(args.output_dir.resolve()),
        "--model",
        args.comet_model,
        "--batch-size",
        str(args.comet_batch_size),
        "--gpus",
        str(args.comet_gpus),
        "--success-marker",
        marker,
    ]
    if args.overwrite:
        command.append("--overwrite")
    os.execv(sys.executable, command)
    raise AssertionError("os.execv returned unexpectedly")


class _NonMistralModelInfo:
    config = {"model_type": "xlm-roberta"}
    tags: list[str] = []
    siblings: list[object] = []

    def __getattr__(self, name: str) -> None:
        return None


@contextmanager
def skip_xlmr_mistral_hub_probe() -> Iterable[None]:
    """Avoid a Transformers Hub metadata probe for the offline XLM-R encoder."""
    patched: list[tuple[Any, str, Any]] = []
    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = None
    if huggingface_hub is not None and hasattr(huggingface_hub, "model_info"):
        patched.append((huggingface_hub, "model_info", huggingface_hub.model_info))

    try:
        import transformers.tokenization_utils_base as tokenization_utils_base
    except ImportError:
        tokenization_utils_base = None
    if tokenization_utils_base is not None and hasattr(tokenization_utils_base, "model_info"):
        patched.append(
            (tokenization_utils_base, "model_info", tokenization_utils_base.model_info)
        )

    def make_model_info(original: Any) -> Any:
        def model_info(model_id: str, *args: Any, **kwargs: Any) -> Any:
            normalized = str(model_id).lower()
            if "xlm-roberta" in normalized or "xlmr" in normalized:
                return _NonMistralModelInfo()
            return original(model_id, *args, **kwargs)

        return model_info

    for module, name, original in patched:
        setattr(module, name, make_model_info(original))
    try:
        yield
    finally:
        for module, name, original in patched:
            setattr(module, name, original)


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: prediction row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def read_predictions(path: Path) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []
    for line_number, row in enumerate(read_prediction_rows(path), 1):
        try:
            prediction = row["prediction"]
            reference = row["reference"]
        except KeyError as exc:
            raise ValueError(f"{path}:{line_number}: missing field {exc}") from exc
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError(f"{path}:{line_number}: prediction/reference must be strings")
        predictions.append(prediction)
        references.append(reference)
    return predictions, references


def load_comet_model(model_name: str) -> Any:
    from comet import download_model, load_from_checkpoint

    offline_values = {"1", "on", "true", "yes"}
    offline = any(
        os.environ.get(name, "").lower() in offline_values
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    model_path = download_model(model_name, local_files_only=offline)
    with skip_xlmr_mistral_hub_probe():
        return load_from_checkpoint(model_path, local_files_only=offline)


def comet_output_scores(output: Any) -> tuple[list[float], float]:
    raw_scores = getattr(output, "scores", None)
    if raw_scores is None and isinstance(output, dict):
        raw_scores = output.get("scores")
    if raw_scores is None:
        raise TypeError("COMET prediction output does not contain sentence scores")
    scores = [float(score) for score in raw_scores]
    if not scores:
        raise ValueError("COMET returned no sentence scores")

    raw_system_score = getattr(output, "system_score", None)
    if raw_system_score is None and isinstance(output, dict):
        raw_system_score = output.get("system_score")
    system_score = (
        float(raw_system_score)
        if raw_system_score is not None
        else sum(scores) / len(scores)
    )
    return scores, system_score


def score_comet_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    batch_size: int,
    gpus: int,
    *,
    overwrite: bool = False,
    success_marker: str | None = None,
) -> int:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    if batch_size < 1:
        raise ValueError("COMET batch size must be positive")
    if gpus < 0:
        raise ValueError("COMET GPU count cannot be negative")
    if success_marker is not None and Path(success_marker).name != success_marker:
        raise ValueError("--success-marker must be a filename, not a path")

    metric_paths = sorted(checkpoint_dir.glob("*/*/metrics.json"))
    if not metric_paths:
        raise RuntimeError(f"No metrics.json files under {checkpoint_dir}")

    model: Any | None = None
    completed = 0
    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") not in {"complete", "smoke"}:
            continue
        if (
            not overwrite
            and metrics.get("comet") is not None
            and metrics.get("comet_model") == model_name
        ):
            completed += 1
            print(f"Skipping matching COMET result: {metrics_path.parent}")
            continue

        predictions_path = metrics_path.with_name("predictions.jsonl")
        rows = read_prediction_rows(predictions_path)
        expected = metrics.get("num_examples")
        if expected is not None and int(expected) != len(rows):
            raise ValueError(
                f"{predictions_path}: found {len(rows)} rows, metrics expect {expected}"
            )

        samples: list[dict[str, str]] = []
        for line_number, row in enumerate(rows, 1):
            try:
                sample = {
                    "src": row["source"],
                    "mt": row["prediction"],
                    "ref": row["reference"],
                }
            except KeyError as exc:
                raise ValueError(f"{predictions_path}:{line_number}: missing field {exc}") from exc
            if not all(isinstance(value, str) for value in sample.values()):
                raise TypeError(
                    f"{predictions_path}:{line_number}: source/prediction/reference must be strings"
                )
            samples.append(sample)

        if model is None:
            print(f"Loading COMET model: {model_name}")
            model = load_comet_model(model_name)
        started = time.monotonic()
        output = model.predict(samples, batch_size=batch_size, gpus=gpus)
        elapsed = time.monotonic() - started
        sentence_scores, system_score = comet_output_scores(output)
        if len(sentence_scores) != len(rows):
            raise ValueError(
                f"COMET returned {len(sentence_scores)} scores for {len(rows)} predictions"
            )

        for row, score in zip(rows, sentence_scores):
            row["comet_score"] = round(score, 6)
        atomic_jsonl(predictions_path, rows)
        metrics["comet"] = round(system_score, 6)
        metrics["comet_model"] = model_name
        metrics["comet_num_examples"] = len(sentence_scores)
        metrics["comet_seconds"] = round(elapsed, 3)
        atomic_json(metrics_path, metrics)
        completed += 1
        print(
            f"Completed COMET {metrics.get('dataset')}/"
            f"{metrics.get('source')}__to__{metrics.get('target')}: "
            f"n={len(sentence_scores)} score={metrics['comet']:.6f}"
        )

    if completed == 0:
        raise RuntimeError(f"No completed metrics to score under {checkpoint_dir}")
    rebuild_checkpoint_summary(checkpoint_dir)
    if success_marker is not None:
        (checkpoint_dir / success_marker).write_text(
            f"completed_directions={completed}\ncomet_model={model_name}\n",
            encoding="utf-8",
        )
    print(
        f"COMET scoring complete: {completed} directions; "
        f"summary: {checkpoint_dir / 'summary.tsv'}"
    )
    return completed


def parse_score_comet_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score saved MT predictions with reference-based COMET."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_COMET_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--success-marker", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_score_comet(argv: list[str]) -> int:
    args = parse_score_comet_args(argv)
    score_comet_checkpoint(
        args.checkpoint_dir,
        args.model,
        args.batch_size,
        args.gpus,
        overwrite=args.overwrite,
        success_marker=args.success_marker,
    )
    return 0


def rebuild_checkpoint_summary(checkpoint_dir: Path) -> None:
    skipped: list[dict[str, Any]] = []
    old_summary = checkpoint_dir / "summary.json"
    if old_summary.is_file():
        try:
            skipped = [
                row
                for row in json.loads(old_summary.read_text(encoding="utf-8"))
                if row.get("status") == "skipped"
            ]
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            skipped = []

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in checkpoint_dir.glob("*/*/metrics.json")
    ]
    rows.sort(key=lambda row: tuple(str(row.get(key, "")) for key in ("dataset", "source", "target")))
    write_summary(checkpoint_dir, rows + skipped)


def rescore_checkpoint(
    checkpoint_dir: Path,
    tokenizer: str,
    *,
    allow_incomplete: bool = False,
    dry_run: bool = False,
) -> int:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    if not allow_incomplete and not (checkpoint_dir / "_SUCCESS").is_file():
        raise RuntimeError(f"Refusing incomplete checkpoint without _SUCCESS: {checkpoint_dir}")

    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize=tokenizer, effective_order=True)
    metric_paths = sorted(checkpoint_dir.glob("*/*/metrics.json"))
    if not metric_paths:
        raise RuntimeError(f"No metrics.json files under {checkpoint_dir}")

    rescored = 0
    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") not in {"complete", "smoke"}:
            continue
        predictions_path = metrics_path.with_name("predictions.jsonl")
        if not predictions_path.is_file():
            raise FileNotFoundError(predictions_path)
        predictions, references = read_predictions(predictions_path)
        expected = metrics.get("num_examples")
        if expected is not None and int(expected) != len(predictions):
            raise ValueError(
                f"{predictions_path}: found {len(predictions)} rows, metrics expect {expected}"
            )

        metrics.pop("bleu_13a", None)
        metrics.pop("bleu_13a_tokenizer", None)
        score = bleu.corpus_score(predictions, [references])
        metrics["bleu"] = round(score.score, 4)
        metrics["bleu_tokenizer"] = tokenizer
        metrics["bleu_signature"] = str(bleu.get_signature())
        if not dry_run:
            atomic_json(metrics_path, metrics)
        rescored += 1

    if rescored == 0:
        raise RuntimeError(f"No completed metrics to rescore under {checkpoint_dir}")
    if not dry_run:
        rebuild_checkpoint_summary(checkpoint_dir)
        (checkpoint_dir / f"_RESCORED_{tokenizer}").write_text(
            f"rescored_directions={rescored}\n", encoding="utf-8"
        )
    print(f"Rescored {rescored} directions with {tokenizer}: {checkpoint_dir}")
    return rescored


def parse_rescore_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute BLEU from saved MT predictions.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="flores200")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_rescore(argv: list[str]) -> int:
    args = parse_rescore_args(argv)
    rescore_checkpoint(
        args.checkpoint_dir,
        args.tokenizer,
        allow_incomplete=args.allow_incomplete,
        dry_run=args.dry_run,
    )
    return 0


def parse_rescore_worker_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore checkpoint directories assigned to a Slurm worker.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--tokenizer", default="flores200")
    return parser.parse_args(argv)


def run_rescore_worker(argv: list[str]) -> int:
    args = parse_rescore_worker_args(argv)
    task_id = args.task_id
    task_count = args.task_count
    if task_id is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--task-id or SLURM_ARRAY_TASK_ID is required")
        task_id = int(value)
    if task_count is None:
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    if task_count < 1 or not 0 <= task_id < task_count:
        raise ValueError(f"Invalid worker assignment: task_id={task_id}, task_count={task_count}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest missing: {args.manifest}")

    checkpoints = [Path(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    processed = 0
    for index in range(task_id, len(checkpoints), task_count):
        rescore_checkpoint(checkpoints[index], args.tokenizer)
        processed += 1
    print(f"Worker {task_id}/{task_count} processed {processed} checkpoints")
    return 0


def parse_submit_rescore_args(argv: list[str]) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Submit CPU-only BLEU rescoring and a dependent aggregate job."
    )
    parser.add_argument("--results-root", type=Path, default=script_dir / "results")
    parser.add_argument("--model-glob", default="0.4B_*")
    parser.add_argument("--tokenizer", default="flores200")
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output-prefix", default="all_results_0.4B_flores200")
    parser.add_argument("--account", default="project_465002530")
    parser.add_argument("--partition", default="small")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def submit_job(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip().split(";", 1)[0]


def run_submit_rescore(argv: list[str]) -> int:
    args = parse_submit_rescore_args(argv)
    if args.workers < 1 or args.max_concurrent < 1:
        raise ValueError("--workers and --max-concurrent must be positive")

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    results_root = args.results_root.resolve()
    manifest_dir = results_root / "task_manifests"
    log_dir = script_dir / "logs" / "rescore"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        marker.parent.resolve()
        for marker in sorted(results_root.glob("*/*/_SUCCESS"))
        if marker.is_file() and fnmatch.fnmatchcase(marker.parent.parent.name, args.model_glob)
    ]
    if not checkpoints:
        raise RuntimeError(f"No completed checkpoints matched {args.model_glob}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = manifest_dir / f"rescore_{args.tokenizer}_{stamp}.txt"
    manifest.write_text("".join(f"{path}\n" for path in checkpoints), encoding="utf-8")
    print(f"Wrote {len(checkpoints)} completed checkpoints to {manifest}")
    if args.dry_run:
        for checkpoint in checkpoints[:10]:
            print(checkpoint)
        print(
            f"Would submit one {args.workers}-worker array with at most "
            f"{args.max_concurrent} running"
        )
        return 0

    worker_parts = [
        "module purge",
        "module use /appl/local/csc/modulefiles/",
        "module load pytorch/2.7",
        f"source {shlex.quote(str(script_dir / 'eval_env' / 'bin' / 'activate'))}",
        "export PYTHONNOUSERSITE=1",
        " ".join(
            shlex.quote(part)
            for part in (
                "python",
                str(script_path),
                "rescore-worker",
                "--manifest",
                str(manifest),
                "--tokenizer",
                args.tokenizer,
            )
        ),
    ]
    worker_command = f"bash -lc {shlex.quote(' && '.join(worker_parts))}"
    array_job = submit_job(
        [
            "sbatch",
            "--parsable",
            f"--array=0-{args.workers - 1}%{args.max_concurrent}",
            "--job-name=rescore-mt",
            "--cpus-per-task=2",
            "--ntasks=1",
            "--mem=8G",
            f"--partition={args.partition}",
            "--time=0-06:00:00",
            f"--account={args.account}",
            "--output=logs/rescore/%x_%A_%a.out",
            "--error=logs/rescore/%x_%A_%a.err",
            "--wrap",
            worker_command,
        ],
        cwd=script_dir,
    )

    aggregate_job = submit_job(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{array_job}",
            str(script_dir / "aggregate_mt_results.sh"),
            "--results-root",
            str(results_root),
            "--output-prefix",
            args.output_prefix,
            "--model-glob",
            args.model_glob,
            "--bleu-tokenizer",
            args.tokenizer,
        ],
        cwd=script_dir,
    )
    print(
        f"Submitted rescoring array: {array_job} "
        f"({args.workers} workers for {len(checkpoints)} checkpoints)"
    )
    print(f"Submitted dependent aggregate: {aggregate_job}")
    print(
        f"Aggregate outputs: {results_root / args.output_prefix}.tsv "
        f"and {results_root / args.output_prefix}.json"
    )
    return 0


def main() -> int:
    commands = {
        "rescore": run_rescore,
        "rescore-worker": run_rescore_worker,
        "score-comet": run_score_comet,
        "submit-rescore": run_submit_rescore,
    }
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        return commands[sys.argv[1]](sys.argv[2:])
    return evaluate(parse_eval_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
