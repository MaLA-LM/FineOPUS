from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dataset.mediator import Example

DEFAULT_REMEDY_COMMAND = "remedy-score"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


def _log(message: str) -> None:
    print(f"[remedy-score] {message}", file=sys.stderr)


def _model_output_dir_name(model_id: str) -> str:
    cleaned = model_id.strip().rstrip("/")
    if not cleaned:
        return "model"
    return cleaned.split("/")[-1]


def write_parallel_files(
    examples: list[Example],
    output_dir: Path,
    src_lang: str,
    tgt_lang: str,
) -> tuple[Path, Path]:
    src_file = output_dir / f"{src_lang}.src"
    mt_file = output_dir / f"{src_lang}-{tgt_lang}.hyp"
    _write_lines(src_file, [row["src"] for row in examples])
    _write_lines(mt_file, [row["tgt"] for row in examples])
    return src_file, mt_file


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if lines:
            handle.write("\n".join(lines))
            handle.write("\n")


def build_remedy_command(
    command: str,
    model_id: str,
    src_file: Path,
    mt_file: Path,
    src_lang: str,
    tgt_lang: str,
    cache_dir: Path,
    save_dir: Path,
    num_gpus: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        command,
        "--model",
        model_id,
        "--src_file",
        str(src_file),
        "--mt_file",
        str(mt_file),
        "--no_ref",
        "--src_lang",
        src_lang,
        "--tgt_lang",
        tgt_lang,
        "--cache_dir",
        str(cache_dir),
        "--save_dir",
        str(save_dir),
        "--num_gpus",
        str(num_gpus),
        "--calibrate",
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
    ]


def run_remedy(
    *,
    command: str,
    model_id: str,
    src_file: Path,
    mt_file: Path,
    src_lang: str,
    tgt_lang: str,
    cache_dir: Path,
    save_dir: Path,
    num_gpus: int,
    gpu_memory_utilization: float,
) -> None:
    if num_gpus <= 0:
        raise ValueError("--gpus/--num_gpus must be >= 1 for remedy-score.")
    args = build_remedy_command(
        command=command,
        model_id=model_id,
        src_file=src_file,
        mt_file=mt_file,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        cache_dir=cache_dir,
        save_dir=save_dir,
        num_gpus=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    _log(
        f"Running '{command}' model='{model_id}' src_lang={src_lang} tgt_lang={tgt_lang} "
        f"num_gpus={num_gpus}"
    )
    subprocess.run(args, check=True)


def resolve_calibration_scores_path(
    *,
    save_dir: Path,
    model_id: str,
    src_lang: str,
    tgt_lang: str,
) -> Path:
    model_output_dir = save_dir / _model_output_dir_name(model_id)
    if not model_output_dir.exists():
        raise RuntimeError(
            "remedy-score output directory not found: "
            f"{model_output_dir} (expected from --save_dir {save_dir})."
        )

    prefix = f"{src_lang}-{tgt_lang}"
    candidates = [
        model_output_dir / f"{prefix}_calibration_scores.txt",
        model_output_dir / f"{prefix}_callibration_scores.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    available = sorted(path.name for path in model_output_dir.iterdir())
    raise RuntimeError(
        "Calibration score file not found in remedy output directory. "
        f"Checked: {[path.name for path in candidates]}. "
        f"Available: {available}."
    )


def read_calibration_scores(path: Path) -> list[float]:
    scores: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                scores.append(float(value))
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid calibration score in {path} line {lineno}: {value!r}"
                ) from exc
    return scores
