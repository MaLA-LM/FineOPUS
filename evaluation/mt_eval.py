#!/usr/bin/env python3
"""Three-shot eng<->x machine-translation evaluation with offline vLLM."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass
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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    return parser.parse_args()


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


def runtime_preflight() -> None:
    missing = []
    for module in ("pandas", "pyarrow", "sacrebleu", "transformers", "vllm"):
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
    fields = [
        "status", "model", "checkpoint", "dataset", "source", "target", "language",
        "num_examples", "bleu", "chrf", "seconds", "few_shot", "limit",
        "bleu_tokenizer", "chrf_word_order", "reason", "result_dir",
    ]
    temporary = output_dir / "summary.tsv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
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


def main() -> int:
    args = parse_args()
    languages = split_csv(args.languages)
    specs = selected_specs(split_csv(args.datasets))
    if not languages:
        raise ValueError("--languages is empty")
    cases, skipped = discover_files(args.data_root, specs, languages)
    if not cases:
        raise RuntimeError("No requested language is available in the selected datasets")
    data_preflight(cases, args.few_shot)
    if args.check_runtime:
        runtime_preflight()
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
    bleu = BLEU(tokenize="13a", effective_order=True)
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
                "bleu_tokenizer": "13a",
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
    (args.output_dir / marker).write_text(f"completed_directions={completed}\n", encoding="utf-8")
    print(f"Evaluation complete: {completed} directions; summary: {args.output_dir / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
