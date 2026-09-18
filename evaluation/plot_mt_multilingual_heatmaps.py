#!/usr/bin/env python3
"Plot final-checkpoint heatmaps for selected 0.4B multilingual models."

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
    RESULTS_DIR,
    add_common_plot_arguments,
    iteration_from_checkpoint,
    parse_formats,
    save_figure,
)


MODEL_VARIANTS = {
    "0.4B_Pretrain_multilingual_FineOPUS": "FineOPUS",
    "0.4B_Pretrain_multilingual_FineOPUS-High": "FineOPUS High",
    "0.4B_Pretrain_multilingual_FineOPUS-xHigh": "FineOPUS xHigh",
    "0.4B_Pretrain_multilingual_MaLA_Bi": "MaLA Bi",
    "0.4B_Pretrain_multilingual_NLLB": "NLLB",
}
VARIANT_ORDER = ["FineOPUS", "FineOPUS High", "FineOPUS xHigh", "MaLA Bi", "NLLB"]
HEATMAP_SPECS = (
    ("bleu", "spBLEU-200", "YlGnBu", ".1f"),
    ("chrf", "chrF++", "YlOrRd", ".1f"),
    ("comet", "COMET", "YlGn", ".3f"),
)

DEFAULT_INPUT = RESULTS_DIR / "all_results_0.4B_multilingual_flores200.tsv"
DEFAULT_OUTPUT_DIR = CURRENT_PLOTS_DIR / "multilingual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_plot_arguments(parser, DEFAULT_INPUT, DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--languages",
        default="all",
        help="Comma-separated language codes, or all (default: all evaluated languages)",
    )
    return parser.parse_args()


def parse_languages(value: str) -> list[str] | None:
    if value.strip().lower() == "all":
        return None
    languages = list(
        dict.fromkeys(code.strip() for code in value.split(",") if code.strip())
    )
    if not languages:
        raise ValueError("At least one language is required")
    return languages


def prepare_data(
    path: Path, languages: list[str] | None
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "status", "model", "checkpoint", "dataset", "source", "target",
        "language", "bleu", "chrf", "comet", "bleu_tokenizer",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=columns)
    frame = frame[
        frame["status"].eq("complete")
        & frame["bleu_tokenizer"].eq("flores200")
        & frame["model"].isin(MODEL_VARIANTS)
    ].copy()
    if languages is not None:
        frame = frame[frame["language"].isin(languages)].copy()
    if frame.empty:
        raise RuntimeError(f"No selected multilingual results in {path}")

    missing_metrics = frame[list(METRIC_COLUMNS)].isna().sum()
    if missing_metrics.any():
        raise RuntimeError(
            f"Missing metric values: {missing_metrics[missing_metrics > 0].to_dict()}"
        )
    frame["variant"] = frame["model"].map(MODEL_VARIANTS)
    frame["iteration"] = frame["checkpoint"].map(iteration_from_checkpoint)

    checkpoint_sets = [
        set(frame.loc[frame["variant"].eq(variant), "checkpoint"])
        for variant in VARIANT_ORDER
    ]
    common_checkpoints = set.intersection(*checkpoint_sets)
    if not common_checkpoints:
        raise RuntimeError("The selected models have no common completed checkpoint")
    checkpoint = max(common_checkpoints, key=iteration_from_checkpoint)
    frame = frame[frame["checkpoint"].eq(checkpoint)].copy()
    frame["direction"] = frame["source"].eq("eng_Latn").map(
        {True: "eng_to_x", False: "x_to_eng"}
    )

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
    dataset_scores = (
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
    macro_scores = (
        dataset_scores.groupby(
            ["language", "model", "variant", "checkpoint", "iteration"],
            as_index=False,
            observed=True,
        )
        .agg(
            bleu=("bleu", "mean"),
            chrf=("chrf", "mean"),
            comet=("comet", "mean"),
            num_datasets=("dataset", "size"),
        )
    )

    selected_languages = languages or sorted(macro_scores["language"].unique())
    validate_data(cells, dataset_scores, macro_scores, selected_languages)
    for table in (cells, dataset_scores, macro_scores):
        table["variant"] = pd.Categorical(
            table["variant"], categories=VARIANT_ORDER, ordered=True
        )
    return checkpoint, cells, dataset_scores, macro_scores


def validate_data(
    cells: pd.DataFrame,
    dataset_scores: pd.DataFrame,
    macro_scores: pd.DataFrame,
    languages: list[str],
) -> None:
    problems = []
    if set(macro_scores["language"].astype(str)) != set(languages):
        problems.append("selected language coverage is incomplete")
    for language, group in macro_scores.groupby("language", observed=True):
        if set(group["variant"].astype(str)) != set(VARIANT_ORDER):
            problems.append(f"{language}: model coverage is incomplete")
    if not dataset_scores["num_directions"].eq(2).all():
        problems.append("some dataset scores do not contain both directions")
    for (language, dataset), group in cells.groupby(
        ["language", "dataset"], observed=True
    ):
        if set(group["variant"].astype(str)) != set(VARIANT_ORDER):
            problems.append(f"{language}/{dataset}: model coverage is incomplete")
    if problems:
        raise RuntimeError("Incomplete heatmap data:\n  " + "\n  ".join(problems))


def plot_heatmaps(
    rows: pd.DataFrame,
    languages: list[str],
    title: str,
    stem: Path,
    formats: list[str],
    dpi: int,
) -> None:
    rows = rows.copy()
    rows["language_name"] = rows["language"].map(
        lambda code: f"{LANGUAGE_NAMES.get(str(code), str(code))}\n({code})"
    )
    language_order = [
        f"{LANGUAGE_NAMES.get(code, code)}\n({code})" for code in languages
    ]
    height = max(7.4, 0.30 * len(languages) + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, height))
    for axis, (metric, metric_title, color_map, value_format) in zip(
        axes, HEATMAP_SPECS
    ):
        matrix = rows.pivot(
            index="language_name", columns="variant", values=metric
        ).reindex(index=language_order, columns=VARIANT_ORDER)
        if matrix.isna().any().any():
            raise RuntimeError(f"Incomplete {metric_title} matrix for {title}")
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=color_map,
            annot=True,
            fmt=value_format,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.78},
            annot_kws={"fontsize": 7.5 if len(languages) <= 20 else 4.5},
        )
        axis.set_title(metric_title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=32, labelsize=8.5)
        y_label_size = 8.5 if len(languages) <= 20 else 5.5
        axis.tick_params(axis="y", rotation=0, labelsize=y_label_size)
    fig.suptitle(title, fontsize=15, y=1.01)
    fig.tight_layout()
    save_figure(fig, stem, formats, dpi)
    plt.close(fig)



def rank_language_averages(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank models after averaging language scores with equal weights."""
    ranking = (
        scores.groupby("variant", as_index=False, observed=True)
        .agg(
            bleu=("bleu", "mean"),
            chrf=("chrf", "mean"),
            comet=("comet", "mean"),
            num_languages=("language", "nunique"),
        )
    )
    rank_columns = []
    for metric in METRIC_COLUMNS:
        column = f"{metric}_rank"
        ranking[column] = ranking[metric].rank(
            ascending=False, method="min"
        ).astype(int)
        rank_columns.append(column)
    ranking["mean_metric_rank"] = ranking[rank_columns].mean(axis=1)
    ranking["overall_rank"] = ranking["mean_metric_rank"].rank(
        ascending=True, method="min"
    ).astype(int)
    return ranking.sort_values(["overall_rank", "variant"])


def main() -> int:
    args = parse_args()
    formats = parse_formats(args.formats)
    requested_languages = parse_languages(args.languages)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, cells, dataset_scores, macro_scores = prepare_data(
        args.input, requested_languages
    )
    languages = requested_languages or sorted(
        macro_scores["language"].astype(str).unique()
    )
    cells.to_csv(
        args.output_dir / "multilingual_final_dataset_direction_cells.tsv",
        sep="\t",
        index=False,
    )
    dataset_scores.to_csv(
        args.output_dir / "multilingual_final_scores_by_dataset.tsv",
        sep="\t",
        index=False,
    )
    macro_scores.to_csv(
        args.output_dir / "multilingual_final_scores.tsv",
        sep="\t",
        index=False,
    )

    ranking = rank_language_averages(macro_scores)
    ranking.to_csv(
        args.output_dir / "multilingual_language_average_ranking.tsv",
        sep="\t",
        index=False,
    )
    dataset_rankings = []
    for dataset, rows in dataset_scores.groupby("dataset", observed=True):
        dataset_ranking = rank_language_averages(rows)
        dataset_ranking.insert(0, "dataset", dataset)
        dataset_rankings.append(dataset_ranking)
    pd.concat(dataset_rankings, ignore_index=True).to_csv(
        args.output_dir / "multilingual_language_average_ranking_by_dataset.tsv",
        sep="\t",
        index=False,
    )

    plot_heatmaps(
        macro_scores,
        languages,
        f"0.4B multilingual models at {checkpoint}: "
        "available-dataset macro average",
        args.output_dir / "final_checkpoint_heatmaps",
        formats,
        args.dpi,
    )
    heatmap_dir = args.output_dir / "final_checkpoint_heatmaps_by_dataset"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_ORDER:
        dataset_rows = dataset_scores[dataset_scores["dataset"].eq(dataset)]
        available_languages = set(dataset_rows["language"].astype(str))
        dataset_languages = [
            language for language in languages if language in available_languages
        ]
        plot_heatmaps(
            dataset_rows,
            dataset_languages,
            f"{dataset}: 0.4B multilingual models at {checkpoint}",
            heatmap_dir / f"{dataset}_final_checkpoint_heatmaps",
            formats,
            args.dpi,
        )

    print(f"Input: {args.input}")
    print(f"Checkpoint: {checkpoint}")
    print("Models: " + ", ".join(VARIANT_ORDER))
    print(f"Languages: {len(languages)}")
    print(f"Output: {args.output_dir}")
    print(f"Heatmaps by dataset: {heatmap_dir}")
    print("Language-average ranking:")
    print(ranking.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
