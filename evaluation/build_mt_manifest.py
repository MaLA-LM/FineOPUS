#!/usr/bin/env python3
"""Discover trained models and write one Slurm task per model/checkpoint."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import re
from pathlib import Path


BILINGUAL_RE = re.compile(
    r"^0\.4B_Pretrain_eng_Latn-(?P<lang>.+?)_"
    r"(?:FineOPUS-Stage[1-4]|MaLA_Bi|NLLB)$"
)
HF_TEMPLATES = {
    "0.4B": "openeurollm/Qwen3-0.4B-ne",
    "0.9B": "openeurollm/Qwen3-0.9B-ne",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--language-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="latest (default), all, an integer, or an iter_NNNNNNN name",
    )
    parser.add_argument(
        "--model-glob",
        action="append",
        default=[],
        help="Include matching model basenames; repeat to use multiple globs",
    )
    parser.add_argument("--force", action="store_true", help="Include tasks with _SUCCESS already present")
    return parser.parse_args()


def multilingual_languages(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        languages = sorted({row["lang"].strip() for row in rows if row["lang"].strip() != "eng_Latn"})
    if not languages:
        raise ValueError(f"No non-English languages found in {path}")
    return languages


def model_languages(name: str, multi: list[str]) -> list[str] | None:
    match = BILINGUAL_RE.match(name)
    if match:
        return [match.group("lang")]
    if re.match(r"^(?:0\.4B|0\.9B)_Pretrain_multilingual_", name):
        return multi
    return None


def checkpoint_dirs(model_dir: Path, selection: str) -> list[Path]:
    root = model_dir / "checkpoints"
    candidates = sorted(
        (p for p in root.glob("iter_*") if p.is_dir()),
        key=lambda p: int(p.name.removeprefix("iter_")),
    )
    if not candidates:
        return []
    if selection == "all":
        return candidates
    if selection == "latest":
        marker = root / "latest_checkpointed_iteration.txt"
        if marker.is_file():
            iteration = int(marker.read_text(encoding="utf-8").strip())
            selected = root / f"iter_{iteration:07d}"
            if not selected.is_dir():
                raise FileNotFoundError(f"Latest marker points to missing checkpoint: {selected}")
            return [selected]
        return [candidates[-1]]
    if selection.startswith("iter_"):
        name = selection
    else:
        name = f"iter_{int(selection):07d}"
    selected = root / name
    if not selected.is_dir():
        raise FileNotFoundError(f"Requested checkpoint does not exist: {selected}")
    return [selected]


def main() -> int:
    args = parse_args()
    multi = multilingual_languages(args.language_manifest)
    patterns = args.model_glob or ["*"]
    tasks: list[dict[str, str]] = []
    skipped_complete = 0

    for model_dir in sorted(p for p in args.models_root.iterdir() if p.is_dir()):
        name = model_dir.name
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            continue
        languages = model_languages(name, multi)
        if languages is None:
            continue
        size = name.split("_", 1)[0]
        hf_template = HF_TEMPLATES.get(size)
        if hf_template is None:
            raise ValueError(f"No conversion template configured for {name}")
        for checkpoint in checkpoint_dirs(model_dir, args.checkpoint):
            task_output = args.output_root / name / checkpoint.name
            if not args.force and (task_output / "_SUCCESS").is_file():
                skipped_complete += 1
                continue
            tasks.append(
                {
                    "task_id": str(len(tasks)),
                    "model_name": name,
                    "model_dir": str(model_dir.resolve()),
                    "checkpoint_name": checkpoint.name,
                    "checkpoint_dir": str(checkpoint.resolve()),
                    "hf_template": hf_template,
                    "languages": ",".join(languages),
                    "output_dir": str(task_output.resolve()),
                }
            )

    if not tasks:
        reason = "all matching tasks are already complete" if skipped_complete else "no matching model/checkpoint tasks"
        raise SystemExit(f"No tasks generated: {reason}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(tasks[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)

    bilingual = sum(len(task["languages"].split(",")) == 1 for task in tasks)
    multilingual = len(tasks) - bilingual
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    print(f"  bilingual={bilingual}, multilingual={multilingual}, already_complete={skipped_complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
