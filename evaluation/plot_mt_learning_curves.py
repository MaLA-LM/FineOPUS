#!/usr/bin/env python3
"""Plot 0.4B bilingual MT learning curves from the aggregate result table."""

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


VARIANT_ORDER = [
    "FineOPUS Stage 1",
    "FineOPUS Stage 2",
    "FineOPUS Stage 3",
    "FineOPUS Stage 4",
    "FineOPUS Stage 4 xHigh",
    "MaLA Bi",
    "NLLB",
]

VARIANT_STYLE = {
    "FineOPUS Stage 1": ("#0072B2", "o", "-"),
    "FineOPUS Stage 2": ("#E69F00", "s", "-"),
    "FineOPUS Stage 3": ("#009E73", "^", "-"),
    "FineOPUS Stage 4": ("#D55E00", "D", "-"),
    "FineOPUS Stage 4 xHigh": ("#882255", "*", "-"),
    "MaLA Bi": ("#CC79A7", "P", "--"),
    "NLLB": ("#6B4C3B", "X", ":"),
}

DEFAULT_INPUT = RESULTS_DIR / "all_results_0.4B_bilingual_flores200.tsv"
DEFAULT_OUTPUT_DIR = CURRENT_PLOTS_DIR / "bilingual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_plot_arguments(parser, DEFAULT_INPUT, DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def variant_from_model(model: str) -> str:
    suffixes = {
        "FineOPUS-Stage4-xHigh": "FineOPUS Stage 4 xHigh",
        "FineOPUS-Stage4-High": "FineOPUS Stage 4 High",
        "MaLA_Bi_NLLB": "MaLA Bi + NLLB",
        "FineOPUS-Stage1": "FineOPUS Stage 1",
        "FineOPUS-Stage2": "FineOPUS Stage 2",
        "FineOPUS-Stage3": "FineOPUS Stage 3",
        "FineOPUS-Stage4": "FineOPUS Stage 4",
        "MaLA_Bi": "MaLA Bi",
        "NLLB": "NLLB",
    }
    for suffix, label in suffixes.items():
        if model.endswith("_" + suffix):
            return label
    raise ValueError(f"Unrecognized bilingual model variant: {model}")


def prepare_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "status", "model", "checkpoint", "dataset", "source", "target",
        "language", "bleu", "chrf", "comet", "bleu_tokenizer",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=columns)
    frame = frame[
        (frame["status"] == "complete")
        & (frame["bleu_tokenizer"] == "flores200")
        & frame["model"].str.startswith("0.4B_Pretrain_eng_Latn-", na=False)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No completed 0.4B bilingual flores200 rows in {path}")
    missing = frame[list(METRIC_COLUMNS)].isna().sum()
    if missing.any():
        raise RuntimeError(f"Missing metric values: {missing[missing > 0].to_dict()}")

    frame["variant"] = frame["model"].map(variant_from_model)
    frame = frame[frame["variant"].isin(VARIANT_ORDER)].copy()
    frame["iteration"] = frame["checkpoint"].map(iteration_from_checkpoint)
    frame["direction"] = frame["source"].eq("eng_Latn").map(
        {True: "eng_to_x", False: "x_to_eng"}
    )

    # BOUQuET may map one training language to multiple benchmark varieties
    # (e.g. Arabic -> apc_Arab and arz_Arab). Average those varieties first so
    # each dataset/direction receives one vote in the final macro average.
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
    macro = (
        cells.groupby(
            ["language", "model", "variant", "checkpoint", "iteration"],
            as_index=False,
            observed=True,
        )
        .agg(
            bleu=("bleu", "mean"),
            bleu_sd=("bleu", "std"),
            chrf=("chrf", "mean"),
            chrf_sd=("chrf", "std"),
            comet=("comet", "mean"),
            comet_sd=("comet", "std"),
            num_conditions=("dataset", "size"),
        )
    )
    macro["variant"] = pd.Categorical(
        macro["variant"], categories=VARIANT_ORDER, ordered=True
    )
    macro = macro.sort_values(["language", "variant", "iteration"])
    validate_data(macro)
    return cells, macro


def validate_data(macro: pd.DataFrame) -> None:
    problems = []
    for language, group in macro.groupby("language", observed=True):
        variants = set(group["variant"].astype(str))
        if variants != set(VARIANT_ORDER):
            problems.append(f"{language}: variants={sorted(variants)}")
        checkpoint_counts = group.groupby("variant", observed=True)["checkpoint"].nunique()
        if checkpoint_counts.nunique() != 1:
            problems.append(f"{language}: unequal checkpoint counts {checkpoint_counts.to_dict()}")
    if problems:
        raise RuntimeError("Incomplete plotting data:\n  " + "\n  ".join(problems))


def plot_language_curves(
    macro: pd.DataFrame, output_dir: Path, formats: list[str], dpi: int
) -> None:
    curve_dir = output_dir / "per_language"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for language, language_data in macro.groupby("language", sort=True, observed=True):
        fig, axes = plt.subplots(1, 3, figsize=(18.2, 4.8), sharex=True)
        for variant in VARIANT_ORDER:
            values = language_data[language_data["variant"] == variant].sort_values("iteration")
            color, marker, line_style = VARIANT_STYLE[variant]
            for axis, (metric, _) in zip(axes, METRIC_SPECS):
                axis.plot(
                    values["iteration"] / 1000,
                    values[metric],
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    linewidth=1.8,
                    markersize=4.2,
                    markevery=2,
                    label=variant,
                )

        axes[0].set_ylabel("Macro-average score")
        for axis, (_, title) in zip(axes, METRIC_SPECS):
            axis.set_title(title)
        for axis in axes:
            axis.set_xlabel("Checkpoint iteration (thousands)")
            axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
            axis.spines[["top", "right"]].set_visible(False)

        language_name = LANGUAGE_NAMES.get(str(language), str(language))
        fig.suptitle(
            f"English ↔ {language_name} ({language}) — 0.4B bilingual models",
            fontsize=14,
            y=1.02,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=5, frameon=False,
            bbox_to_anchor=(0.5, -0.07),
        )
        fig.text(
            0.5,
            -0.12,
            "Equal-weight macro average over available datasets and translation directions; "
            "BOUQuET language varieties are averaged first.",
            ha="center",
            fontsize=8.5,
            color="#555555",
        )
        fig.tight_layout()
        save_figure(
            fig,
            curve_dir / f"eng_Latn-{language}_learning_curves",
            formats,
            dpi,
        )
        plt.close(fig)


def prepare_dataset_curves(cells: pd.DataFrame) -> pd.DataFrame:
    """Average only the two translation directions, retaining each dataset."""
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
            bleu_sd=("bleu", "std"),
            chrf=("chrf", "mean"),
            chrf_sd=("chrf", "std"),
            comet=("comet", "mean"),
            comet_sd=("comet", "std"),
            num_directions=("direction", "size"),
        )
    )
    dataset_curves["variant"] = pd.Categorical(
        dataset_curves["variant"], categories=VARIANT_ORDER, ordered=True
    )
    return dataset_curves.sort_values(
        ["language", "dataset", "variant", "iteration"]
    )


def plot_language_dataset_curves(
    dataset_curves: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Plot one row per dataset and one column per metric for each language."""
    curve_dir = output_dir / "per_language_by_dataset"
    curve_dir.mkdir(parents=True, exist_ok=True)

    available_datasets = set(dataset_curves["dataset"].astype(str))
    datasets = [name for name in DATASET_ORDER if name in available_datasets]
    datasets.extend(sorted(available_datasets - set(datasets)))

    for language, language_data in dataset_curves.groupby(
        "language", sort=True, observed=True
    ):
        language_datasets = [
            name for name in datasets if name in set(language_data["dataset"].astype(str))
        ]
        fig, axes = plt.subplots(
            len(language_datasets),
            3,
            figsize=(18.4, 3.65 * len(language_datasets)),
            sharex=True,
            squeeze=False,
        )
        for row, dataset in enumerate(language_datasets):
            dataset_data = language_data[language_data["dataset"] == dataset]
            for variant in VARIANT_ORDER:
                values = dataset_data[
                    dataset_data["variant"] == variant
                ].sort_values("iteration")
                color, marker, line_style = VARIANT_STYLE[variant]
                for column, (metric, _) in enumerate(METRIC_SPECS):
                    axes[row, column].plot(
                        values["iteration"] / 1000,
                        values[metric],
                        color=color,
                        marker=marker,
                        linestyle=line_style,
                        linewidth=1.7,
                        markersize=4.0,
                        markevery=2,
                        label=variant,
                    )

            for column, (_, title) in enumerate(METRIC_SPECS):
                axes[row, column].set_title(f"{dataset} — {title}")
            axes[row, 0].set_ylabel("Direction-average score")
            for axis in axes[row]:
                axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
                axis.spines[["top", "right"]].set_visible(False)

        for axis in axes[-1]:
            axis.set_xlabel("Checkpoint iteration (thousands)")

        language_name = LANGUAGE_NAMES.get(str(language), str(language))
        fig.suptitle(
            f"English ↔ {language_name} ({language}) — scores by dataset",
            fontsize=14,
            y=1.01,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.015),
        )
        fig.text(
            0.5,
            -0.045,
            "Datasets are shown separately; each point averages only English→X and X→English. "
            "BOUQuET language varieties are averaged first.",
            ha="center",
            fontsize=8.5,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        save_figure(
            fig,
            curve_dir / f"eng_Latn-{language}_learning_curves_by_dataset",
            formats,
            dpi,
        )
        plt.close(fig)


def plot_final_heatmap(
    macro: pd.DataFrame, output_dir: Path, formats: list[str], dpi: int
) -> pd.DataFrame:
    final_rows = (
        macro.sort_values("iteration")
        .groupby(["language", "variant"], as_index=False, observed=True)
        .tail(1)
        .copy()
    )
    final_rows["language_name"] = final_rows["language"].map(
        lambda code: f"{LANGUAGE_NAMES.get(str(code), str(code))}\n({code})"
    )

    fig, axes = plt.subplots(1, 3, figsize=(24.0, 7.4))
    for axis, metric, title, color_map, value_format in (
        (axes[0], "bleu", "Final checkpoint: spBLEU-200", "YlGnBu", ".1f"),
        (axes[1], "chrf", "Final checkpoint: chrF++", "YlOrRd", ".1f"),
        (axes[2], "comet", "Final checkpoint: COMET", "YlGn", ".3f"),
    ):
        matrix = final_rows.pivot(index="language_name", columns="variant", values=metric)
        matrix = matrix.reindex(columns=VARIANT_ORDER)
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=color_map,
            annot=True,
            fmt=value_format,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.78},
            annot_kws={"fontsize": 7.5},
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=38, labelsize=8.5)
        axis.tick_params(axis="y", rotation=0, labelsize=8.5)

    fig.suptitle(
        "0.4B bilingual model comparison at the final checkpoint",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "final_checkpoint_heatmaps", formats, dpi)
    plt.close(fig)
    return final_rows


def plot_final_heatmaps_by_dataset(
    dataset_curves: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> pd.DataFrame:
    """Plot final-checkpoint heatmaps separately for every dataset."""
    final_rows = (
        dataset_curves.sort_values("iteration")
        .groupby(
            ["language", "variant", "dataset"],
            as_index=False,
            observed=True,
        )
        .tail(1)
        .copy()
    )
    final_rows["language_name"] = final_rows["language"].map(
        lambda code: f"{LANGUAGE_NAMES.get(str(code), str(code))}\n({code})"
    )

    heatmap_dir = output_dir / "final_checkpoint_heatmaps_by_dataset"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    available = set(final_rows["dataset"].astype(str))
    datasets = [name for name in DATASET_ORDER if name in available]
    datasets.extend(sorted(available - set(datasets)))

    for dataset in datasets:
        dataset_rows = final_rows[final_rows["dataset"] == dataset]
        fig, axes = plt.subplots(1, 3, figsize=(24.0, 7.4))
        for axis, metric, title, color_map, value_format in (
            (axes[0], "bleu", "spBLEU-200", "YlGnBu", ".1f"),
            (axes[1], "chrf", "chrF++", "YlOrRd", ".1f"),
            (axes[2], "comet", "COMET", "YlGn", ".3f"),
        ):
            matrix = dataset_rows.pivot(
                index="language_name", columns="variant", values=metric
            ).reindex(columns=VARIANT_ORDER)
            sns.heatmap(
                matrix,
                ax=axis,
                cmap=color_map,
                annot=True,
                fmt=value_format,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"shrink": 0.78},
                annot_kws={"fontsize": 7.5},
            )
            axis.set_title(title)
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.tick_params(axis="x", rotation=38, labelsize=8.5)
            axis.tick_params(axis="y", rotation=0, labelsize=8.5)

        fig.suptitle(
            f"{dataset}: 0.4B bilingual models at the final checkpoint",
            fontsize=15,
            y=1.01,
        )
        fig.tight_layout()
        save_figure(
            fig,
            heatmap_dir / f"{dataset}_final_checkpoint_heatmaps",
            formats,
            dpi,
        )
        plt.close(fig)
    return final_rows


def main() -> int:
    args = parse_args()
    formats = parse_formats(args.formats)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells, macro = prepare_data(args.input)
    cells.to_csv(args.output_dir / "dataset_direction_cells.tsv", sep="\t", index=False)
    macro.to_csv(args.output_dir / "bilingual_macro_by_checkpoint.tsv", sep="\t", index=False)
    plot_language_curves(macro, args.output_dir, formats, args.dpi)
    dataset_curves = prepare_dataset_curves(cells)
    dataset_curves.to_csv(
        args.output_dir / "bilingual_by_dataset_by_checkpoint.tsv",
        sep="\t",
        index=False,
    )
    plot_language_dataset_curves(dataset_curves, args.output_dir, formats, args.dpi)
    final_rows = plot_final_heatmap(macro, args.output_dir, formats, args.dpi)
    final_rows.to_csv(args.output_dir / "final_checkpoint_scores.tsv", sep="\t", index=False)
    final_dataset_rows = plot_final_heatmaps_by_dataset(
        dataset_curves, args.output_dir, formats, args.dpi
    )
    final_dataset_rows.to_csv(
        args.output_dir / "final_checkpoint_scores_by_dataset.tsv",
        sep="\t",
        index=False,
    )

    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Languages plotted: {macro['language'].nunique()}")
    print(f"Per-language figures: {args.output_dir / 'per_language'}")
    print(
        "Per-language, per-dataset figures: "
        f"{args.output_dir / 'per_language_by_dataset'}"
    )
    print(f"Final-checkpoint heatmap: {args.output_dir / 'final_checkpoint_heatmaps.png'}")
    print(
        "Final-checkpoint heatmaps by dataset: "
        f"{args.output_dir / 'final_checkpoint_heatmaps_by_dataset'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
