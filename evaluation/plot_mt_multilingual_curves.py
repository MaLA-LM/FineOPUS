#!/usr/bin/env python3
"""Plot learning curves for the four 0.4B multilingual MT models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_mt_common import (
    CURRENT_PLOTS_DIR,
    DATASET_ORDER,
    LANGUAGE_NAMES,
    METRIC_COLUMNS,
    METRIC_SPECS,
    RESULTS_DIR,
    add_common_plot_arguments,
    iteration_from_checkpoint,
    parse_formats,
    save_figure,
)


MODEL_VARIANTS = {
    "0.4B_Pretrain_multilingual_FineOPUS": "FineOPUS",
    "0.4B_Pretrain_multilingual_MaLA_Bi": "MaLA Bi",
    "0.4B_Pretrain_multilingual_MaLA_Bi_NLLB": "MaLA Bi + NLLB",
    "0.4B_Pretrain_multilingual_NLLB": "NLLB",
}
VARIANT_ORDER = ["FineOPUS", "MaLA Bi", "MaLA Bi + NLLB", "NLLB"]
VARIANT_STYLE = {
    "FineOPUS": ("#0072B2", "o", "-"),
    "MaLA Bi": ("#D55E00", "s", "--"),
    "MaLA Bi + NLLB": ("#CC79A7", "P", "--"),
    "NLLB": ("#009E73", "^", ":"),
}

DEFAULT_INPUT = RESULTS_DIR / "all_results_0.4B_multilingual_flores200.tsv"
DEFAULT_OUTPUT_DIR = CURRENT_PLOTS_DIR / "multilingual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_plot_arguments(parser, DEFAULT_INPUT, DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def prepare_data(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "status", "model", "checkpoint", "dataset", "source", "target",
        "language", "bleu", "chrf", "comet", "bleu_tokenizer",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=columns)
    frame = frame[
        (frame["status"] == "complete")
        & (frame["bleu_tokenizer"] == "flores200")
        & frame["model"].isin(MODEL_VARIANTS)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No completed 0.4B multilingual rows in {path}")
    missing = frame[list(METRIC_COLUMNS)].isna().sum()
    if missing.any():
        raise RuntimeError(f"Missing metric values: {missing[missing > 0].to_dict()}")
    frame["variant"] = frame["model"].map(MODEL_VARIANTS)
    frame["iteration"] = frame["checkpoint"].map(iteration_from_checkpoint)
    frame["direction"] = frame["source"].eq("eng_Latn").map(
        {True: "eng_to_x", False: "x_to_eng"}
    )

    # Collapse dataset-specific aliases before any macro averaging.
    cells = (
        frame.groupby(
            [
                "language", "model", "variant", "checkpoint", "iteration",
                "dataset", "direction",
            ],
            as_index=False,
            observed=True,
        )[list(METRIC_COLUMNS)]
        .mean()
    )
    language_macro = (
        cells.groupby(
            ["language", "model", "variant", "checkpoint", "iteration"],
            as_index=False,
            observed=True,
        )
        .agg(
            bleu=("bleu", "mean"),
            chrf=("chrf", "mean"),
            comet=("comet", "mean"),
            num_conditions=("dataset", "size"),
        )
    )
    dataset_curves = (
        cells.groupby(
            [
                "language", "model", "variant", "checkpoint", "iteration",
                "dataset",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            bleu=("bleu", "mean"),
            chrf=("chrf", "mean"),
            comet=("comet", "mean"),
            num_directions=("direction", "size"),
        )
    )
    # Give every language equal weight, even when benchmark coverage differs.
    global_macro = (
        language_macro.groupby(
            ["model", "variant", "checkpoint", "iteration"],
            as_index=False,
            observed=True,
        )
        .agg(
            bleu=("bleu", "mean"),
            chrf=("chrf", "mean"),
            comet=("comet", "mean"),
            num_languages=("language", "size"),
        )
    )
    for table in (language_macro, dataset_curves, global_macro):
        table["variant"] = pd.Categorical(
            table["variant"], categories=VARIANT_ORDER, ordered=True
        )
    language_macro = language_macro.sort_values(
        ["language", "variant", "iteration"]
    )
    dataset_curves = dataset_curves.sort_values(
        ["language", "dataset", "variant", "iteration"]
    )
    global_macro = global_macro.sort_values(["variant", "iteration"])
    validate_data(language_macro, global_macro)
    return cells, language_macro, dataset_curves, global_macro


def validate_data(language_macro: pd.DataFrame, global_macro: pd.DataFrame) -> None:
    problems = []
    for language, group in language_macro.groupby("language", observed=True):
        variants = set(group["variant"].astype(str))
        if variants != set(VARIANT_ORDER):
            problems.append(f"{language}: variants={sorted(variants)}")
        counts = group.groupby("variant", observed=True)["checkpoint"].nunique()
        if counts.nunique() != 1:
            problems.append(f"{language}: unequal checkpoints {counts.to_dict()}")
    if global_macro["num_languages"].nunique() != 1:
        problems.append("global curves do not contain a stable language count")
    if problems:
        raise RuntimeError("Incomplete plotting data:\n  " + "\n  ".join(problems))


def draw_lines(axes: object, values: pd.DataFrame) -> None:
    for variant in VARIANT_ORDER:
        data = values[values["variant"] == variant].sort_values("iteration")
        color, marker, line_style = VARIANT_STYLE[variant]
        for axis, (metric, _) in zip(axes, METRIC_SPECS):
            axis.plot(
                data["iteration"] / 1000,
                data[metric],
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.8,
                markersize=3.8,
                markevery=5,
                label=variant,
            )


def style_axes(axes: object, ylabel: str) -> None:
    axes[0].set_ylabel(ylabel)
    for axis, (_, title) in zip(axes, METRIC_SPECS):
        axis.set_title(title)
    for axis in axes:
        axis.set_xlabel("Checkpoint iteration (thousands)")
        axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)


def add_legend(fig: plt.Figure, axis: plt.Axes, y: float) -> None:
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, y),
    )


def save_metric_panels(
    values: pd.DataFrame,
    title: str,
    note: str,
    stem: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 4.8), sharex=True)
    draw_lines(axes, values)
    style_axes(axes, "Macro-average score")
    fig.suptitle(title, fontsize=14, y=1.02)
    add_legend(fig, axes[0], -0.04)
    fig.text(0.5, -0.085, note, ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout()
    save_figure(fig, stem, formats, dpi)
    plt.close(fig)


def plot_global_and_languages(
    global_macro: pd.DataFrame,
    language_macro: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    language_count = int(language_macro["language"].nunique())
    save_metric_panels(
        global_macro,
        f"0.4B multilingual models — macro average over {language_count} languages",
        "Each language has equal weight after averaging its available datasets and directions.",
        output_dir / "global_learning_curves",
        formats,
        dpi,
    )
    curve_dir = output_dir / "per_language"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for language, values in language_macro.groupby(
        "language", sort=True, observed=True
    ):
        name = LANGUAGE_NAMES.get(str(language), str(language))
        save_metric_panels(
            values,
            f"English ↔ {name} ({language}) — 0.4B multilingual models",
            "Equal-weight average over available datasets and directions; "
            "dataset-specific language varieties are averaged first.",
            curve_dir / f"eng_Latn-{language}_multilingual_learning_curves",
            formats,
            dpi,
        )


def plot_by_dataset(
    dataset_curves: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    curve_dir = output_dir / "per_language_by_dataset"
    curve_dir.mkdir(parents=True, exist_ok=True)
    available = set(dataset_curves["dataset"].astype(str))
    dataset_order = [name for name in DATASET_ORDER if name in available]
    dataset_order.extend(sorted(available - set(dataset_order)))
    for language, language_data in dataset_curves.groupby(
        "language", sort=True, observed=True
    ):
        present = set(language_data["dataset"].astype(str))
        datasets = [name for name in dataset_order if name in present]
        fig, axes = plt.subplots(
            len(datasets), 3, figsize=(18.4, 3.65 * len(datasets)),
            sharex=True, squeeze=False,
        )
        for row, dataset in enumerate(datasets):
            draw_lines(axes[row], language_data[language_data["dataset"] == dataset])
            for column, (_, title) in enumerate(METRIC_SPECS):
                axes[row, column].set_title(f"{dataset} — {title}")
            axes[row, 0].set_ylabel("Direction-average score")
            for axis in axes[row]:
                axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
                axis.spines[["top", "right"]].set_visible(False)
        for axis in axes[-1]:
            axis.set_xlabel("Checkpoint iteration (thousands)")
        name = LANGUAGE_NAMES.get(str(language), str(language))
        fig.suptitle(
            f"English ↔ {name} ({language}) — multilingual scores by dataset",
            fontsize=14,
            y=1.01,
        )
        add_legend(fig, axes[0, 0], -0.015)
        fig.text(
            0.5, -0.045,
            "Datasets are separate; each point averages only English→X and X→English.",
            ha="center", fontsize=8.5, color="#555555",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        save_figure(
            fig,
            curve_dir
            / f"eng_Latn-{language}_multilingual_learning_curves_by_dataset",
            formats,
            dpi,
        )
        plt.close(fig)


def main() -> int:
    args = parse_args()
    formats = parse_formats(args.formats)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells, language_macro, dataset_curves, global_macro = prepare_data(args.input)
    tables = {
        "multilingual_dataset_direction_cells.tsv": cells,
        "multilingual_macro_by_checkpoint.tsv": language_macro,
        "multilingual_by_dataset_by_checkpoint.tsv": dataset_curves,
        "multilingual_global_macro_by_checkpoint.tsv": global_macro,
    }
    for filename, table in tables.items():
        table.to_csv(args.output_dir / filename, sep="\t", index=False)
    plot_global_and_languages(
        global_macro, language_macro, args.output_dir, formats, args.dpi
    )
    plot_by_dataset(dataset_curves, args.output_dir, formats, args.dpi)
    checkpoints = global_macro.groupby(
        "variant", observed=True
    )["checkpoint"].nunique().to_dict()
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Languages plotted: {language_macro['language'].nunique()}")
    print(f"Checkpoints per model: {checkpoints}")
    print(f"Global overview: {args.output_dir / 'global_learning_curves.png'}")
    print(f"Per-language figures: {args.output_dir / 'per_language'}")
    print(f"Per-language, per-dataset figures: {args.output_dir / 'per_language_by_dataset'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
