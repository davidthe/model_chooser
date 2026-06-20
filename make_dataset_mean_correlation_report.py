#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parent
RUN_ID = os.environ.get("RUN_ID", "full_run_20260617")
RUN_DIR = REPO_ROOT / "analysis_results" / RUN_ID
TABLES_DIR = RUN_DIR / "tables"
FIGURES_DIR = RUN_DIR / "figures"

METRIC_COLS = ["pll_score", "wer", "cer", "levenshtein"]
DISPLAY_NAMES = {
    "pll_score": "PLL",
    "wer": "WER",
    "cer": "CER",
    "levenshtein": "Levenshtein",
}

MD_PATH = RUN_DIR / "dataset_mean_correlation_report.md"
PDF_PATH = RUN_DIR / "dataset_mean_correlation_report.pdf"
LONG_CSV_PATH = TABLES_DIR / "dataset_correlation_matrices_long.csv"
SUMMARY_CSV_PATH = TABLES_DIR / "dataset_mean_correlation_summary.csv"


def display_metric(metric: str) -> str:
    return DISPLAY_NAMES.get(metric, metric)


def format_value(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{value:.3f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(clean(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def matrix_to_markdown(matrix: pd.DataFrame) -> str:
    display = matrix.copy()
    display.index = [display_metric(idx) for idx in display.index]
    display.columns = [display_metric(col) for col in display.columns]
    display = display.apply(lambda col: col.map(format_value))
    headers = [""] + list(display.columns)
    rows = [[idx] + list(display.loc[idx].values) for idx in display.index]
    return markdown_table(headers, rows)


def count_matrix_to_markdown(matrix: pd.DataFrame) -> str:
    display = matrix.copy()
    display.index = [display_metric(idx) for idx in display.index]
    display.columns = [display_metric(col) for col in display.columns]
    display = display.apply(lambda col: col.map(lambda value: "" if pd.isna(value) else str(int(value))))
    headers = [""] + list(display.columns)
    rows = [[idx] + list(display.loc[idx].values) for idx in display.index]
    return markdown_table(headers, rows)


def load_per_page() -> pd.DataFrame:
    path = TABLES_DIR / "per_page_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path)


def variant_frames(per_page: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    variants = [
        (
            "base_valid_all_pages",
            "Base-valid rows before removing diagnostic pages",
            per_page[per_page["base_valid"]].copy(),
        )
    ]
    if "keep_after_worst_page_trim" in per_page.columns:
        variants.append(
            (
                "base_valid_after_worst20_trim",
                "Base-valid rows after removing the worst 20% diagnostic pages",
                per_page[per_page["base_valid"] & per_page["keep_after_worst_page_trim"]].copy(),
            )
        )
    if "page_high_confidence" in per_page.columns and "keep_after_worst_page_trim" in per_page.columns:
        variants.append(
            (
                "high_confidence_after_worst20_trim",
                "High-confidence rows after removing the worst 20% diagnostic pages",
                per_page[per_page["page_high_confidence"] & per_page["keep_after_worst_page_trim"]].copy(),
            )
        )
    return variants


def correlation_matrix(group: pd.DataFrame) -> pd.DataFrame:
    valid = group[METRIC_COLS].dropna()
    if len(valid) < 3:
        return pd.DataFrame(index=METRIC_COLS, columns=METRIC_COLS, dtype=float)
    return valid.corr(method="pearson").reindex(index=METRIC_COLS, columns=METRIC_COLS)


def build_mean_tables(per_page: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    mean_matrices: dict[str, pd.DataFrame] = {}
    count_matrices: dict[str, pd.DataFrame] = {}
    std_matrices: dict[str, pd.DataFrame] = {}

    for variant, description, frame in variant_frames(per_page):
        dataset_matrices: list[pd.DataFrame] = []
        for dataset, group in frame.groupby("dataset", sort=True):
            dataset_label = group["dataset_label"].iloc[0]
            matrix = correlation_matrix(group)
            if matrix.isna().all().all():
                continue
            dataset_matrices.append(matrix)
            for row_metric in METRIC_COLS:
                for col_metric in METRIC_COLS:
                    long_rows.append(
                        {
                            "variant": variant,
                            "variant_description": description,
                            "dataset": dataset,
                            "dataset_label": dataset_label,
                            "pages": int(group["image"].nunique()),
                            "rows": int(len(group)),
                            "row_metric": row_metric,
                            "col_metric": col_metric,
                            "pearson_r": matrix.loc[row_metric, col_metric],
                        }
                    )

        if not dataset_matrices:
            mean = pd.DataFrame(index=METRIC_COLS, columns=METRIC_COLS, dtype=float)
            counts = pd.DataFrame(index=METRIC_COLS, columns=METRIC_COLS, dtype=float)
            std = pd.DataFrame(index=METRIC_COLS, columns=METRIC_COLS, dtype=float)
        else:
            stacked = pd.concat(
                {idx: matrix for idx, matrix in enumerate(dataset_matrices)},
                names=["dataset_index", "row_metric"],
            )
            mean = stacked.groupby(level="row_metric").mean(numeric_only=True).reindex(index=METRIC_COLS, columns=METRIC_COLS)
            counts = stacked.groupby(level="row_metric").count().reindex(index=METRIC_COLS, columns=METRIC_COLS)
            std = stacked.groupby(level="row_metric").std(ddof=0, numeric_only=True).reindex(index=METRIC_COLS, columns=METRIC_COLS)

        mean_matrices[variant] = mean
        count_matrices[variant] = counts
        std_matrices[variant] = std
        mean.to_csv(TABLES_DIR / f"{variant}_mean_dataset_correlation_matrix.csv")
        counts.to_csv(TABLES_DIR / f"{variant}_mean_dataset_correlation_counts.csv")
        std.to_csv(TABLES_DIR / f"{variant}_mean_dataset_correlation_std.csv")

        summary_rows.append(
            {
                "variant": variant,
                "variant_description": description,
                "datasets_used_pll_wer": counts.loc["pll_score", "wer"] if "pll_score" in counts.index else 0,
                "mean_pll_vs_wer": mean.loc["pll_score", "wer"] if "pll_score" in mean.index else float("nan"),
                "mean_pll_vs_cer": mean.loc["pll_score", "cer"] if "pll_score" in mean.index else float("nan"),
                "mean_pll_vs_levenshtein": mean.loc["pll_score", "levenshtein"] if "pll_score" in mean.index else float("nan"),
                "mean_wer_vs_cer": mean.loc["wer", "cer"] if "wer" in mean.index else float("nan"),
                "mean_cer_vs_levenshtein": mean.loc["cer", "levenshtein"] if "cer" in mean.index else float("nan"),
            }
        )

    long_df = pd.DataFrame(long_rows)
    summary = pd.DataFrame(summary_rows)
    long_df.to_csv(LONG_CSV_PATH, index=False)
    summary.to_csv(SUMMARY_CSV_PATH, index=False)
    return summary, mean_matrices, count_matrices, std_matrices


def plot_matrix(matrix: pd.DataFrame, path: Path, title: str) -> None:
    display = matrix.copy()
    display.index = [display_metric(idx) for idx in display.index]
    display.columns = [display_metric(col) for col in display.columns]
    plt.figure(figsize=(7.2, 5.4))
    sns.heatmap(display, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def write_figures(mean_matrices: dict[str, pd.DataFrame]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for variant, matrix in mean_matrices.items():
        plot_matrix(
            matrix,
            FIGURES_DIR / f"{variant}_mean_dataset_correlation_heatmap.png",
            f"{variant}: mean of per-dataset correlations",
        )


def write_markdown(summary: pd.DataFrame, mean_matrices: dict[str, pd.DataFrame], count_matrices: dict[str, pd.DataFrame]) -> None:
    lines: list[str] = []
    lines.append("# Mean Dataset Correlation Report")
    lines.append("")
    lines.append(f"Run: `{RUN_ID}`")
    lines.append("")
    lines.append(
        "This report computes a correlation matrix separately for each handwriting dataset, then averages those "
        "matrices element by element. Each dataset therefore contributes equally, unlike a pooled correlation table "
        "where datasets with more pages or model rows can dominate."
    )
    lines.append("")
    lines.append("## Summary")
    display = summary.copy()
    display = display[
        [
            "variant",
            "datasets_used_pll_wer",
            "mean_pll_vs_wer",
            "mean_pll_vs_cer",
            "mean_pll_vs_levenshtein",
            "mean_wer_vs_cer",
            "mean_cer_vs_levenshtein",
        ]
    ]
    display.columns = ["Variant", "Datasets", "PLL-WER", "PLL-CER", "PLL-Lev", "WER-CER", "CER-Lev"]
    for col in ["PLL-WER", "PLL-CER", "PLL-Lev", "WER-CER", "CER-Lev"]:
        display[col] = display[col].map(format_value)
    display["Datasets"] = display["Datasets"].map(lambda value: int(value) if math.isfinite(float(value)) else "")
    lines.append(markdown_table(list(display.columns), display.values.tolist()))
    lines.append("")

    preferred = "base_valid_after_worst20_trim" if "base_valid_after_worst20_trim" in mean_matrices else list(mean_matrices)[0]
    lines.append("## Recommended Mean Correlation Table")
    lines.append(f"Variant: `{preferred}`")
    lines.append("")
    lines.append(matrix_to_markdown(mean_matrices[preferred]))
    lines.append("")
    lines.append("Dataset counts contributing to each cell:")
    lines.append("")
    lines.append(count_matrix_to_markdown(count_matrices[preferred]))

    lines.append("")
    lines.append("## All Variants")
    for variant, matrix in mean_matrices.items():
        if variant == preferred:
            continue
        lines.append("")
        lines.append(f"### {variant}")
        lines.append(matrix_to_markdown(matrix))

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def add_pdf_title(pdf: PdfPages, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.93, "Mean Dataset Correlation Report", fontsize=22, weight="bold")
    fig.text(0.08, 0.895, f"Run: {RUN_ID}", fontsize=12)
    text = (
        "Each handwriting dataset gets its own Pearson correlation matrix across PLL, WER, CER, and Levenshtein. "
        "The final table is the elementwise mean of those dataset-level matrices, so every handwriting style has "
        "equal weight."
    )
    fig.text(0.08, 0.84, textwrap.fill(text, width=88), fontsize=11, va="top")

    display = summary.copy()
    display = display[["variant", "datasets_used_pll_wer", "mean_pll_vs_wer", "mean_pll_vs_cer", "mean_pll_vs_levenshtein"]]
    display.columns = ["Variant", "Datasets", "PLL-WER", "PLL-CER", "PLL-Lev"]
    for col in ["PLL-WER", "PLL-CER", "PLL-Lev"]:
        display[col] = display[col].map(format_value)
    display["Variant"] = display["Variant"].map(lambda value: textwrap.fill(str(value), width=26, break_long_words=False))
    display["Datasets"] = display["Datasets"].astype(int).astype(str)
    table_ax = fig.add_axes([0.06, 0.36, 0.88, 0.34])
    table_ax.axis("off")
    table = table_ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.45)
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_matrix_page(pdf: PdfPages, variant: str, matrix: pd.DataFrame, counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.suptitle(f"{variant}: mean of per-dataset correlations", fontsize=15, weight="bold", y=0.97)

    for ax, title, data in [(axes[0], "Mean correlation matrix", matrix), (axes[1], "Dataset counts per cell", counts)]:
        ax.axis("off")
        display = data.copy()
        display.index = [display_metric(idx) for idx in display.index]
        display.columns = [display_metric(col) for col in display.columns]
        if title.startswith("Mean"):
            display = display.apply(lambda col: col.map(format_value))
        else:
            display = display.apply(lambda col: col.map(lambda value: "" if pd.isna(value) else str(int(value))))
        table = ax.table(
            cellText=display.values,
            rowLabels=display.index,
            colLabels=display.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.55)
        ax.set_title(title, pad=12)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(summary: pd.DataFrame, mean_matrices: dict[str, pd.DataFrame], count_matrices: dict[str, pd.DataFrame]) -> None:
    with PdfPages(PDF_PATH) as pdf:
        add_pdf_title(pdf, summary)
        preferred = "base_valid_after_worst20_trim" if "base_valid_after_worst20_trim" in mean_matrices else list(mean_matrices)[0]
        add_pdf_matrix_page(pdf, preferred, mean_matrices[preferred], count_matrices[preferred])
        for variant, matrix in mean_matrices.items():
            if variant == preferred:
                continue
            add_pdf_matrix_page(pdf, variant, matrix, count_matrices[variant])


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    per_page = load_per_page()
    summary, mean_matrices, count_matrices, _ = build_mean_tables(per_page)
    write_figures(mean_matrices)
    write_markdown(summary, mean_matrices, count_matrices)
    write_pdf(summary, mean_matrices, count_matrices)
    print(MD_PATH)
    print(PDF_PATH)
    print(SUMMARY_CSV_PATH)
    print(LONG_CSV_PATH)


if __name__ == "__main__":
    main()
