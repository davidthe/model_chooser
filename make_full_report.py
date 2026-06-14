#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parent
RUN_ID = "full_run_20260613"
REPORT_DIR = REPO_ROOT / "analysis_results" / RUN_ID
TABLES_DIR = REPORT_DIR / "tables"
FIGURES_DIR = REPORT_DIR / "figures"
PDF_PATH = REPORT_DIR / "report.pdf"
MD_PATH = REPORT_DIR / "report.md"

METRIC_COLS = ["pll_score", "wer", "cer", "levenshtein"]
PLL_VALID_THRESHOLD = 90000.0

DATASET_LABELS = {
    "bnf150amitaitanchuma2": "BNF 150 Amitai Tanchuma 2",
    "bnf150test2": "BNF 150 test 2",
    "bodleianoppenheimad": "Bodleian Oppenheim Add. 4-128",
    "constantinople1520": "Constantinople 1520",
    "huntingtonamitaitanchuma2": "Huntington Amitai Tanchuma 2",
    "matteozzmsoxfordhuntington2": "Matteo Zz MS Oxford Huntington 2",
    "midrashhagadoldvarim": "Midrash Hagadol Dvarim",
    "mshuntington115": "Bodleian MS Huntington 115",
    "munich229amitaitanchuma2": "Munich 229 Amitai Tanchuma 2",
    "munichstaatsbibliothektest2": "Munich Staatsbibliothek test 2",
    "neubauer147amitaitanchuma2": "Neubauer 147 Amitai Tanchuma 2",
    "oppaddfol3test2": "Oppenheim Add. Fol. 3 test 2",
    "parma3122test2": "Parma 3122 test 2",
    "printeditionmantoue1544": "Print edition Mantoue 1544",
    "seferhaikarimbnf740experimen": "Sefer Haikarim BNF 740 experiment",
    "vat44": "Vat 44",
    "vatebr34": "Vat Ebr 34",
    "vatican34midrashtanchu": "Vatican 34 Midrash Tanchuma",
    "vatican44amitaimidrashtanchu": "Vatican 44 Amitai Midrash Tanchuma",
    "vaybertaytshexperiment": "Vayber Taytsh experiment",
}


def pretty_dataset_label(name: str) -> str:
    return DATASET_LABELS.get(name, name.replace("_", " ").title())


def pretty_model_label(name: str) -> str:
    return name.replace("_", " ")


def wrapped_label(name: str, width: int = 18) -> str:
    return textwrap.fill(name, width=width, break_long_words=False, break_on_hyphens=False)


def ensure_inputs() -> None:
    required = [
        TABLES_DIR / "per_image_metrics.csv",
        TABLES_DIR / "dataset_model_summary.csv",
        TABLES_DIR / "model_summary.csv",
        TABLES_DIR / "dataset_best_models.csv",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The full run results are incomplete. Missing:\n" + "\n".join(f"- {item}" for item in missing)
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_image = pd.read_csv(TABLES_DIR / "per_image_metrics.csv")
    dataset_model = pd.read_csv(TABLES_DIR / "dataset_model_summary.csv")
    model_summary = pd.read_csv(TABLES_DIR / "model_summary.csv")
    best_by_dataset = pd.read_csv(TABLES_DIR / "dataset_best_models.csv")
    return per_image, dataset_model, model_summary, best_by_dataset


def split_valid_rows(per_image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_mask = per_image[METRIC_COLS].isna().any(axis=1)
    penalty_mask = per_image["pll_score"].notna() & (per_image["pll_score"] >= PLL_VALID_THRESHOLD)
    valid_mask = ~(missing_mask | penalty_mask)

    valid = per_image[valid_mask].copy()
    invalid = per_image[~valid_mask].copy()
    invalid["issue"] = "fallback_penalty"
    invalid.loc[missing_mask, "issue"] = "missing_score"
    penalty = per_image[penalty_mask].copy()
    penalty["issue"] = "fallback_penalty"
    missing = per_image[missing_mask].copy()
    missing["issue"] = "missing_score"
    return valid, penalty, missing


def compute_dataset_correlations(per_image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets: list[str] = []
    matrices: list[np.ndarray] = []
    detailed_rows: list[dict[str, object]] = []

    for dataset, group in per_image.groupby("dataset", sort=True):
        valid = group[METRIC_COLS].dropna()
        if len(valid) < 2:
            continue
        corr = valid.corr(method="pearson").reindex(index=METRIC_COLS, columns=METRIC_COLS)
        datasets.append(dataset)
        matrices.append(corr.to_numpy(dtype=float))
        for row_metric in METRIC_COLS:
            for col_metric in METRIC_COLS:
                detailed_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": pretty_dataset_label(dataset),
                        "row_metric": row_metric,
                        "col_metric": col_metric,
                        "correlation": float(corr.loc[row_metric, col_metric]),
                    }
                )

    if not matrices:
        raise ValueError("Could not compute any dataset correlations.")

    stack = np.stack(matrices, axis=0)
    mean_matrix = np.nanmean(stack, axis=0)
    std_matrix = np.nanstd(stack, axis=0)
    count_matrix = np.sum(~np.isnan(stack), axis=0)

    mean_corr = pd.DataFrame(mean_matrix, index=METRIC_COLS, columns=METRIC_COLS)
    std_corr = pd.DataFrame(std_matrix, index=METRIC_COLS, columns=METRIC_COLS)
    count_corr = pd.DataFrame(count_matrix, index=METRIC_COLS, columns=METRIC_COLS)

    mean_corr.to_csv(TABLES_DIR / "correlation_matrix_mean_by_dataset.csv")
    std_corr.to_csv(TABLES_DIR / "correlation_matrix_std_by_dataset.csv")
    count_corr.to_csv(TABLES_DIR / "correlation_matrix_dataset_counts.csv")

    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(TABLES_DIR / "per_dataset_correlations.csv", index=False)
    return mean_corr, std_corr, detailed


def pooled_correlation(per_image: pd.DataFrame) -> pd.DataFrame:
    pooled = per_image[METRIC_COLS].corr(method="pearson")
    pooled.to_csv(TABLES_DIR / "correlation_matrix_pooled.csv")
    return pooled


def save_table_image(df: pd.DataFrame, path: Path, title: str, *, figsize: tuple[float, float], font_size: float = 10) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.suptitle(title, fontsize=18, weight="bold", y=0.97)

    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.45)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_size_bars(per_image: pd.DataFrame) -> None:
    counts = (
        per_image.groupby(["dataset", "dataset_label"], as_index=False)
        .agg(images=("image", "nunique"))
        .sort_values("images", ascending=False)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=counts, x="images", y="dataset_label", color="#3b82f6")
    plt.xlabel("Images")
    plt.ylabel("Handwriting style")
    plt.title("Dataset size by handwriting style")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dataset_sizes.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_model_summary(model_summary: pd.DataFrame) -> None:
    metrics = [
        ("mean_pll", "Mean PLL", "PLL"),
        ("mean_wer", "Mean WER", "WER"),
        ("mean_cer", "Mean CER", "CER"),
        ("mean_levenshtein", "Mean Levenshtein", "Levenshtein"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()
    order = model_summary.sort_values("mean_pll")["model_label"]
    for ax, (col, title, ylabel) in zip(axes, metrics):
        sns.barplot(data=model_summary, x="model_label", y=col, order=order, ax=ax, color="#2563eb")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Model comparison across all handwriting styles", fontsize=15, y=1.02)
    fig.savefig(FIGURES_DIR / "model_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_heatmap(dataset_model: pd.DataFrame) -> None:
    pivot = dataset_model.pivot(index="dataset_label", columns="model_label", values="mean_pll")
    pivot = pivot.sort_index(axis=0)
    order = (
        dataset_model.groupby("model_label", as_index=False)["mean_pll"]
        .mean()
        .sort_values("mean_pll")["model_label"]
        .tolist()
    )
    pivot = pivot[order]
    plt.figure(figsize=(max(14, len(order) * 1.2), max(10, len(pivot.index) * 0.45)))
    sns.heatmap(pivot, cmap="mako_r", linewidths=0.2, linecolor="white")
    plt.title("Mean PLL score by handwriting style and model")
    plt.xlabel("Model")
    plt.ylabel("Handwriting style")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dataset_model_pll_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_winner_counts(best_by_dataset: pd.DataFrame) -> None:
    winner_counts = best_by_dataset["best_pll_model_label"].value_counts().reset_index()
    winner_counts.columns = ["model_label", "wins"]
    winner_counts = winner_counts.sort_values("wins", ascending=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=winner_counts, x="wins", y="model_label", color="#10b981")
    plt.xlabel("Handwriting sets won on mean PLL")
    plt.ylabel("Model")
    plt.title("Who wins each handwriting style?")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pll_winner_counts.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmaps(mean_corr: pd.DataFrame, pooled_corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(mean_corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".2f", ax=axes[0])
    axes[0].set_title("Mean of per-handwriting Pearson matrices")
    sns.heatmap(pooled_corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".2f", ax=axes[1])
    axes[1].set_title("Pooled Pearson matrix across all rows")
    fig.suptitle("PLL versus GT-based score correlations", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_distributions(corr_details: pd.DataFrame) -> None:
    pair_map = {
        "PLL vs WER": ("pll_score", "wer"),
        "PLL vs CER": ("pll_score", "cer"),
        "PLL vs Levenshtein": ("pll_score", "levenshtein"),
    }
    rows = []
    corr_lookup = corr_details.pivot_table(
        index=["dataset", "dataset_label"],
        columns=["row_metric", "col_metric"],
        values="correlation",
    )
    for label, (row_metric, col_metric) in pair_map.items():
        if (row_metric, col_metric) not in corr_lookup.columns:
            continue
        for dataset, dataset_label in corr_lookup.index:
            rows.append(
                {
                    "pair": label,
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "correlation": corr_lookup.loc[(dataset, dataset_label), (row_metric, col_metric)],
                }
            )
    pair_df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pair_df, x="pair", y="correlation", color="#93c5fd")
    sns.stripplot(data=pair_df, x="pair", y="correlation", color="#1e3a8a", alpha=0.35, jitter=0.12, size=4)
    plt.ylim(-0.1, 1.05)
    plt.ylabel("Dataset-wise Pearson correlation")
    plt.xlabel("")
    plt.title("How stable the PLL signal is across handwriting styles")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_distribution.png", dpi=220, bbox_inches="tight")
    plt.close()


def top_winners_table(best_by_dataset: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_label",
        "best_pll_model_label",
        "best_wer_model_label",
        "best_cer_model_label",
        "best_lev_model_label",
    ]
    table = best_by_dataset[cols].copy()
    table.columns = ["Handwriting style", "Best PLL", "Best WER", "Best CER", "Best Levenshtein"]
    table["Handwriting style"] = table["Handwriting style"].map(lambda s: wrapped_label(str(s), 24))
    table["Best PLL"] = table["Best PLL"].map(lambda s: wrapped_label(str(s), 16))
    table["Best WER"] = table["Best WER"].map(lambda s: wrapped_label(str(s), 16))
    table["Best CER"] = table["Best CER"].map(lambda s: wrapped_label(str(s), 16))
    table["Best Levenshtein"] = table["Best Levenshtein"].map(lambda s: wrapped_label(str(s), 16))
    return table


def narrative(
    per_image: pd.DataFrame,
    valid_per_image: pd.DataFrame,
    model_summary: pd.DataFrame,
    best_by_dataset: pd.DataFrame,
    mean_corr: pd.DataFrame,
    pooled_corr: pd.DataFrame,
    penalty_rows: int,
    missing_rows: int,
) -> str:
    best = model_summary.sort_values("mean_pll").iloc[0]
    n_datasets = int(per_image["dataset"].nunique())
    n_images = int(per_image[["dataset", "image"]].drop_duplicates().shape[0])
    n_models = int(per_image["model"].nunique())
    n_rows = int(len(per_image))
    n_valid_rows = int(len(valid_per_image))
    n_valid_styles = int(valid_per_image["dataset"].nunique())
    pll_wer = mean_corr.loc["pll_score", "wer"]
    pll_cer = mean_corr.loc["pll_score", "cer"]
    pll_lev = mean_corr.loc["pll_score", "levenshtein"]
    pooled_wer = pooled_corr.loc["pll_score", "wer"]
    pooled_cer = pooled_corr.loc["pll_score", "cer"]
    pooled_lev = pooled_corr.loc["pll_score", "levenshtein"]
    wins = best_by_dataset["best_pll_model_label"].value_counts().to_dict()
    win_text = ", ".join(f"{model}: {count}" for model, count in wins.items())
    text = f"""
    This report summarizes the full handwriting corpus in `datasets/` using {n_datasets} handwriting styles,
    {n_images} images, {n_models} OCR models, and {n_rows} model-image comparisons. After removing {penalty_rows}
    fallback-penalty rows and {missing_rows} missing-score rows, {n_valid_rows} comparisons remain usable across
    {n_valid_styles} handwriting styles. The strongest average
    result on PLL is `{best['model_label']}`, with the lowest mean PLL {best['mean_pll']:.3f}, mean WER {best['mean_wer']:.3f},
    mean CER {best['mean_cer']:.3f}, and mean Levenshtein {best['mean_levenshtein']:.3f}.

    The main correlation matrix in this report is not the raw pooled correlation. Instead, each handwriting style
    gets its own Pearson matrix first, and the final matrix is the elementwise mean of those style-level matrices.
    That makes every handwriting style count equally, which is the right choice when the point is to compare styles
    rather than let the largest set dominate the statistics.

    On that equal-weight mean matrix, PLL correlates {pll_wer:.3f} with WER, {pll_cer:.3f} with CER, and {pll_lev:.3f}
    with Levenshtein. For comparison, the pooled all-rows correlations are {pooled_wer:.3f}, {pooled_cer:.3f}, and
    {pooled_lev:.3f}. The model wins by handwriting style are: {win_text}.
    """
    return " ".join(textwrap.fill(" ".join(text.split()), width=95).splitlines())


def save_summary_markdown(
    per_image: pd.DataFrame,
    valid_per_image: pd.DataFrame,
    model_summary: pd.DataFrame,
    best_by_dataset: pd.DataFrame,
    mean_corr: pd.DataFrame,
    pooled_corr: pd.DataFrame,
    quality_summary: pd.DataFrame,
    penalty_rows: int,
    missing_rows: int,
) -> None:
    best = model_summary.sort_values("mean_pll").iloc[0]
    lines = []
    lines.append("# Full handwriting report")
    lines.append("")
    lines.append("This report summarizes the full `datasets/` corpus and compares all OCR models on every handwriting style.")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Handwriting styles: {per_image['dataset'].nunique()}")
    lines.append(f"- Images: {per_image[['dataset', 'image']].drop_duplicates().shape[0]}")
    lines.append(f"- OCR models: {per_image['model'].nunique()}")
    lines.append(f"- Model-image comparisons: {len(per_image)}")
    lines.append(f"- Valid comparisons used for statistics: {len(valid_per_image)}")
    lines.append(f"- Fallback-penalty rows excluded: {penalty_rows}")
    lines.append(f"- Missing-score rows excluded: {missing_rows}")
    lines.append("")
    lines.append("## Method")
    lines.append(
        "For each dataset, I ran the OCR models on every image, computed PLL from the language model, and computed "
        "WER, CER, and Levenshtein distance against the ground truth text. Lower PLL is better in this pipeline. Rows with the fallback penalty or missing scores were excluded from the statistical summaries. The main correlation matrix is the "
        "equal-weight mean of the per-dataset Pearson matrices, so larger datasets do not dominate the result."
    )
    lines.append("")
    lines.append("## Data quality")
    for _, row in quality_summary.iterrows():
        lines.append(
            f"- {row['dataset_label']}: valid={int(row['valid_rows'])}, penalty={int(row['penalty_rows'])}, missing={int(row['missing_rows'])}"
        )
    lines.append("")
    lines.append("## Overall ranking by mean PLL")
    for _, row in model_summary.sort_values("mean_pll").iterrows():
        lines.append(
            f"- {row['model_label']}: PLL={row['mean_pll']:.3f}, WER={row['mean_wer']:.3f}, "
            f"CER={row['mean_cer']:.3f}, Lev={row['mean_levenshtein']:.3f}"
        )
    lines.append("")
    lines.append("## Best model by handwriting style")
    for _, row in best_by_dataset.iterrows():
        lines.append(
            f"- {row['dataset_label']}: PLL={row['best_pll_model_label']}, WER={row['best_wer_model_label']}, "
            f"CER={row['best_cer_model_label']}, Lev={row['best_lev_model_label']}"
        )
    lines.append("")
    lines.append("## Mean correlation matrix across handwriting styles")
    lines.append(mean_corr.round(3).to_string())
    lines.append("")
    lines.append("## Pooled correlation matrix across all rows")
    lines.append(pooled_corr.round(3).to_string())
    lines.append("")
    lines.append(
        "The strongest average model is `"
        + str(best["model_label"])
        + "`. The correlation results show that PLL stays positively aligned with the GT-based scores, "
        "and the equal-weight mean matrix is the fairest summary when the goal is to compare handwriting styles."
    )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_pdf(
    per_image: pd.DataFrame,
    valid_per_image: pd.DataFrame,
    dataset_model: pd.DataFrame,
    model_summary: pd.DataFrame,
    best_by_dataset: pd.DataFrame,
    mean_corr: pd.DataFrame,
    pooled_corr: pd.DataFrame,
    corr_details: pd.DataFrame,
    quality_summary: pd.DataFrame,
    penalty_rows: int,
    missing_rows: int,
) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(8.27, 11.69))
    plt.close(fig)

    with PdfPages(PDF_PATH) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.text(0.08, 0.93, "Full OCR Report Across All Handwriting Styles", fontsize=23, weight="bold")
        fig.text(0.08, 0.89, f"Run: {RUN_ID}", fontsize=14)
        fig.text(0.08, 0.84, "Executive summary", fontsize=16, weight="bold")
        fig.text(
            0.08,
            0.79,
            textwrap.fill(
                narrative(
                    per_image,
                    valid_per_image,
                    model_summary,
                    best_by_dataset,
                    mean_corr,
                    pooled_corr,
                    penalty_rows,
                    missing_rows,
                ),
                width=88,
            ),
            fontsize=11,
            va="top",
        )
        key_rows = [
            ["Handwriting styles", int(per_image["dataset"].nunique())],
            ["Images", int(per_image[["dataset", "image"]].drop_duplicates().shape[0])],
            ["Models", int(per_image["model"].nunique())],
            ["Comparisons", int(len(per_image))],
            ["Valid comparisons", int(len(valid_per_image))],
            ["Fallback penalties", int(penalty_rows)],
            ["Missing scores", int(missing_rows)],
            ["Best average model", str(model_summary.sort_values("mean_pll").iloc[0]["model_label"])],
            ["Best mean PLL", f"{model_summary.sort_values('mean_pll').iloc[0]['mean_pll']:.3f}"],
        ]
        table_ax = fig.add_axes([0.08, 0.28, 0.84, 0.34])
        table_ax.axis("off")
        table = table_ax.table(cellText=key_rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)
        fig.text(0.08, 0.17, "Interpretation", fontsize=16, weight="bold")
        fig.text(
            0.08,
            0.12,
            textwrap.fill(
                "Every handwriting style contributes equally to the headline correlation matrix. Lower PLL is the better side of the scale here. That makes the report "
                "useful for comparing styles, not just counting lines. The model summary and winner counts show which "
                "recognizer is most stable across the corpus.",
                width=90,
            ),
            fontsize=11,
            va="top",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Overview page
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
        fig.suptitle("Corpus overview and model ranking", fontsize=18, weight="bold", y=0.97)
        size_counts = (
            per_image.groupby(["dataset_label"], as_index=False)
            .agg(images=("image", "nunique"))
            .sort_values("images", ascending=False)
        )
        sns.barplot(data=size_counts, x="images", y="dataset_label", ax=axes[0], color="#60a5fa")
        axes[0].set_xlabel("Images")
        axes[0].set_ylabel("Handwriting style")
        axes[0].set_title("Dataset size by handwriting style")
        sns.barplot(
            data=model_summary.sort_values("mean_pll"),
            x="model_label",
            y="mean_pll",
            ax=axes[1],
            color="#2563eb",
        )
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("Mean PLL")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_title("Average PLL across the whole corpus")
        fig.tight_layout(rect=[0, 0.01, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Heatmaps page
        fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
        fig.suptitle("Handwriting-level model behavior", fontsize=18, weight="bold", y=0.97)
        pll_pivot = dataset_model.pivot(index="dataset_label", columns="model_label", values="mean_pll")
        pll_pivot = pll_pivot.sort_index(axis=0)
        model_order = model_summary.sort_values("mean_pll")["model_label"].tolist()
        pll_pivot = pll_pivot[model_order]
        sns.heatmap(pll_pivot, cmap="mako_r", linewidths=0.2, linecolor="white", ax=axes[0])
        axes[0].set_title("Mean PLL by handwriting style and model")
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("Handwriting style")
        axes[0].tick_params(axis="x", rotation=30)

        winner_counts = best_by_dataset["best_pll_model_label"].value_counts().reset_index()
        winner_counts.columns = ["model_label", "wins"]
        winner_counts = winner_counts.sort_values("wins", ascending=True)
        sns.barplot(data=winner_counts, x="wins", y="model_label", ax=axes[1], color="#10b981")
        axes[1].set_xlabel("Handwriting styles won on mean PLL")
        axes[1].set_ylabel("Model")
        axes[1].set_title("Which model wins each handwriting style?")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Correlation page
        fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
        fig.suptitle("Correlation analysis", fontsize=18, weight="bold", y=0.97)
        sns.heatmap(mean_corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".2f", ax=axes[0])
        axes[0].set_title("Equal-weight mean Pearson matrix")
        sns.heatmap(pooled_corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".2f", ax=axes[1])
        axes[1].set_title("Pooled Pearson matrix")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Correlation distribution and summary table page
        pair_map = {
            "PLL vs WER": ("pll_score", "wer"),
            "PLL vs CER": ("pll_score", "cer"),
            "PLL vs Levenshtein": ("pll_score", "levenshtein"),
        }
        corr_lookup = corr_details.pivot_table(
            index=["dataset", "dataset_label"],
            columns=["row_metric", "col_metric"],
            values="correlation",
        )
        rows = []
        for label, (row_metric, col_metric) in pair_map.items():
            if (row_metric, col_metric) not in corr_lookup.columns:
                continue
            for dataset, dataset_label in corr_lookup.index:
                rows.append(
                    {
                        "pair": label,
                        "dataset": dataset,
                        "dataset_label": dataset_label,
                        "correlation": corr_lookup.loc[(dataset, dataset_label), (row_metric, col_metric)],
                    }
                )
        pair_df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
        fig.suptitle("PLL alignment across handwriting styles", fontsize=18, weight="bold", y=0.97)
        sns.boxplot(data=pair_df, x="pair", y="correlation", ax=axes[0], color="#93c5fd")
        sns.stripplot(data=pair_df, x="pair", y="correlation", ax=axes[0], color="#1e3a8a", alpha=0.35, jitter=0.12, size=4)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Dataset-wise Pearson correlation")
        axes[0].set_ylim(-0.1, 1.05)
        axes[0].tick_params(axis="x", rotation=15)
        pair_summary = []
        for pair_name, sub in pair_df.groupby("pair"):
            pair_summary.append(
                {
                    "Pair": pair_name,
                    "Mean": sub["correlation"].mean(),
                    "Std": sub["correlation"].std(ddof=0),
                    "Datasets": sub.shape[0],
                }
            )
        pair_summary_df = pd.DataFrame(pair_summary)
        pair_summary_df["Mean"] = pair_summary_df["Mean"].map(lambda x: f"{x:.3f}")
        pair_summary_df["Std"] = pair_summary_df["Std"].map(lambda x: f"{x:.3f}")
        pair_summary_df.columns = ["Pair", "Mean", "Std", "Datasets"]
        axes[1].axis("off")
        table = axes[1].table(cellText=pair_summary_df.values, colLabels=pair_summary_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 1.7)
        axes[1].set_title("Summary of dataset-wise correlations", pad=15)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Appendix page
        winners_table = top_winners_table(best_by_dataset)
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.suptitle("Best model by handwriting style", fontsize=18, weight="bold", y=0.97)
        table_ax = fig.add_axes([0.04, 0.08, 0.92, 0.82])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=winners_table.values,
            colLabels=winners_table.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        pdf.savefig(fig)
        plt.close(fig)

        # Data quality page
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.suptitle("Rows excluded from the statistical summary", fontsize=18, weight="bold", y=0.97)
        quality_display = quality_summary.copy()
        quality_display["dataset_label"] = quality_display["dataset_label"].map(lambda s: wrapped_label(str(s), 24))
        quality_display = quality_display[["dataset_label", "valid_rows", "penalty_rows", "missing_rows"]].copy()
        quality_display.columns = ["Handwriting style", "Valid", "Penalty", "Missing"]
        quality_display = quality_display.sort_values(["Penalty", "Missing"], ascending=False)
        quality_ax = fig.add_axes([0.04, 0.10, 0.92, 0.78])
        quality_ax.axis("off")
        table = quality_ax.table(
            cellText=quality_display.values,
            colLabels=quality_display.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.55)
        fig.text(
            0.05,
            0.04,
            textwrap.fill(
                "These rows were still produced by the evaluation run, but they are excluded from the corpus-level means and correlations because the PLL scorer returned a fallback penalty or no score at all.",
                width=110,
            ),
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ensure_inputs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    per_image, dataset_model, model_summary, best_by_dataset = load_inputs()
    valid_per_image, penalty_rows, missing_rows = split_valid_rows(per_image)

    quality_summary = (
        per_image.assign(
            valid=lambda df: ((df["pll_score"] < PLL_VALID_THRESHOLD) & df["pll_score"].notna() & df[METRIC_COLS].notna().all(axis=1)),
            penalty=lambda df: df["pll_score"].notna() & (df["pll_score"] >= PLL_VALID_THRESHOLD),
            missing=lambda df: df[METRIC_COLS].isna().any(axis=1),
        )
        .groupby(["dataset", "dataset_label"], as_index=False)
        .agg(
            valid_rows=("valid", "sum"),
            penalty_rows=("penalty", "sum"),
            missing_rows=("missing", "sum"),
        )
    )
    quality_summary.to_csv(TABLES_DIR / "quality_summary.csv", index=False)
    penalty_rows.to_csv(TABLES_DIR / "penalty_rows.csv", index=False)
    missing_rows.to_csv(TABLES_DIR / "missing_rows.csv", index=False)

    dataset_model_valid = (
        valid_per_image
        .groupby(["dataset", "dataset_label", "model", "model_label"], as_index=False)
        .agg(
            mean_pll=("pll_score", "mean"),
            median_pll=("pll_score", "median"),
            std_pll=("pll_score", "std"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_levenshtein=("levenshtein", "mean"),
            images=("image", "nunique"),
        )
    )
    model_summary_valid = (
        valid_per_image
        .groupby(["model", "model_label"], as_index=False)
        .agg(
            mean_pll=("pll_score", "mean"),
            median_pll=("pll_score", "median"),
            std_pll=("pll_score", "std"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_levenshtein=("levenshtein", "mean"),
            images=("image", "nunique"),
            datasets=("dataset", "nunique"),
        )
        .sort_values("mean_pll", ascending=True)
    )
    best_by_dataset_valid = []
    for dataset, group in dataset_model_valid.groupby("dataset", sort=True):
        row_pll = group.loc[group["mean_pll"].idxmin()]
        row_wer = group.loc[group["mean_wer"].idxmin()]
        row_cer = group.loc[group["mean_cer"].idxmin()]
        row_lev = group.loc[group["mean_levenshtein"].idxmin()]
        best_by_dataset_valid.append({
            "dataset": dataset,
            "dataset_label": group["dataset_label"].iloc[0],
            "best_pll_model": row_pll["model"],
            "best_pll_model_label": row_pll["model_label"],
            "best_wer_model": row_wer["model"],
            "best_wer_model_label": row_wer["model_label"],
            "best_cer_model": row_cer["model"],
            "best_cer_model_label": row_cer["model_label"],
            "best_lev_model": row_lev["model"],
            "best_lev_model_label": row_lev["model_label"],
        })
    best_by_dataset_valid = pd.DataFrame(best_by_dataset_valid)

    mean_corr, std_corr, corr_details = compute_dataset_correlations(valid_per_image)
    pooled_corr = pooled_correlation(valid_per_image)

    plot_dataset_size_bars(per_image)
    plot_model_summary(model_summary_valid)
    plot_dataset_heatmap(dataset_model_valid)
    plot_winner_counts(best_by_dataset_valid)
    plot_correlation_heatmaps(mean_corr, pooled_corr)
    plot_correlation_distributions(corr_details)

    save_table_image(
        (
            model_summary_valid.assign(model_label=model_summary_valid["model_label"].map(lambda s: wrapped_label(str(s), 18)))
            [["model_label", "mean_pll", "mean_wer", "mean_cer", "mean_levenshtein", "images", "datasets"]]
            .rename(
                columns={
                    "model_label": "Model",
                    "mean_pll": "Mean PLL",
                    "mean_wer": "Mean WER",
                    "mean_cer": "Mean CER",
                    "mean_levenshtein": "Mean Lev",
                    "images": "Images",
                    "datasets": "Styles",
                }
            )
            .assign(
                **{
                    "Mean PLL": lambda df: df["Mean PLL"].map(lambda v: f"{v:.3f}"),
                    "Mean WER": lambda df: df["Mean WER"].map(lambda v: f"{v:.3f}"),
                    "Mean CER": lambda df: df["Mean CER"].map(lambda v: f"{v:.3f}"),
                    "Mean Lev": lambda df: df["Mean Lev"].map(lambda v: f"{v:.3f}"),
                    "Images": lambda df: df["Images"].astype(int).astype(str),
                    "Styles": lambda df: df["Styles"].astype(int).astype(str),
                }
            )
            .set_index("Model")
        ),
        FIGURES_DIR / "model_summary_table.png",
        "Model summary table",
        figsize=(12, 4.8),
        font_size=10,
    )
    save_table_image(
        mean_corr.round(3),
        FIGURES_DIR / "correlation_mean_table.png",
        "Mean correlation table",
        figsize=(8, 4.8),
        font_size=11,
    )

    save_summary_markdown(
        per_image,
        valid_per_image,
        model_summary_valid,
        best_by_dataset_valid,
        mean_corr,
        pooled_corr,
        quality_summary,
        len(penalty_rows),
        len(missing_rows),
    )
    build_pdf(
        per_image,
        valid_per_image,
        dataset_model_valid,
        model_summary_valid,
        best_by_dataset_valid,
        mean_corr,
        pooled_corr,
        corr_details,
        quality_summary,
        len(penalty_rows),
        len(missing_rows),
    )
    print(PDF_PATH)


if __name__ == "__main__":
    main()
