from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dataset.mediator import Example
from utils.logger import logger

__all__ = [
    "DEFAULT_REMEDY_COMMAND",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
    "PORT_RANGE_START",
    "PORT_RANGE_SIZE",
    "build_remedy_command",
    "run_remedy",
    "resolve_calibration_scores_path",
    "read_calibration_scores",
    "write_parallel_files",
]

DEFAULT_REMEDY_COMMAND = "remedy-score"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
PORT_RANGE_START = 20000
PORT_RANGE_SIZE = 20000


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


def _parse_int(raw: str | None, default: int = 0) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_master_port(env: dict[str, str], iteration: int) -> int:
    master_port = _parse_int(env.get("MASTER_PORT"), default=0)
    if (
        master_port < PORT_RANGE_START
        or master_port >= PORT_RANGE_START + PORT_RANGE_SIZE
    ):
        seed_job = _parse_int(
            env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID"),
            default=0,
        )
        seed_task = _parse_int(env.get("SLURM_ARRAY_TASK_ID"), default=0)
        master_port = PORT_RANGE_START + ((seed_job + seed_task) % PORT_RANGE_SIZE)

    slot = (master_port - PORT_RANGE_START + max(iteration, 0)) % PORT_RANGE_SIZE
    return PORT_RANGE_START + slot


def build_remedy_command(
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
        DEFAULT_REMEDY_COMMAND,
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
    model_id: str,
    src_file: Path,
    mt_file: Path,
    src_lang: str,
    tgt_lang: str,
    cache_dir: Path,
    save_dir: Path,
    num_gpus: int,
    gpu_memory_utilization: float,
    iteration: int = 0,
) -> None:
    if num_gpus <= 0:
        raise ValueError("--gpus/--num_gpus must be >= 1 for remedy-score.")
    args = build_remedy_command(
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
    logger.warning(
        "[remedy-score] Running '%s' model='%s' src_lang=%s tgt_lang=%s num_gpus=%s",
        DEFAULT_REMEDY_COMMAND,
        model_id,
        src_lang,
        tgt_lang,
        num_gpus,
    )

    env = os.environ.copy()
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(_resolve_master_port(env, iteration))
    logger.info(
        "[remedy-score] master=%s:%s iteration=%s",
        env["MASTER_ADDR"],
        env["MASTER_PORT"],
        iteration,
    )

    subprocess.run(args, check=True, env=env)


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

    scores_path = model_output_dir / f"{src_lang}-{tgt_lang}_calibration_scores.txt"
    if scores_path.exists():
        return scores_path

    available = sorted(path.name for path in model_output_dir.iterdir())
    raise RuntimeError(
        "Calibration score file not found in remedy output directory. "
        f"Checked: {scores_path.name}. "
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
