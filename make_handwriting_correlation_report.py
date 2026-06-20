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
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parent
RUN_ID = os.environ.get("RUN_ID", "full_run_20260617")
RUN_DIR = REPO_ROOT / "analysis_results" / RUN_ID
TABLES_DIR = RUN_DIR / "tables"

METRIC_COLS = ["pll_score", "wer", "cer", "levenshtein"]
DISPLAY_NAMES = {
    "pll_score": "PLL",
    "wer": "WER",
    "cer": "CER",
    "levenshtein": "Levenshtein",
}

MD_PATH = RUN_DIR / "handwriting_correlation_report.md"
PDF_PATH = RUN_DIR / "handwriting_correlation_report.pdf"
LONG_CSV_PATH = TABLES_DIR / "per_handwriting_correlation_long.csv"
PLL_SUMMARY_CSV_PATH = TABLES_DIR / "per_handwriting_pll_summary.csv"


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


def load_per_page() -> pd.DataFrame:
    path = TABLES_DIR / "per_page_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    return pd.read_csv(path)


def filter_variants(per_page: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
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
    return (
        group[METRIC_COLS]
        .dropna()
        .corr(method="pearson")
        .reindex(index=METRIC_COLS, columns=METRIC_COLS)
    )


def build_tables(per_page: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    long_rows: list[dict[str, object]] = []
    pll_rows: list[dict[str, object]] = []
    matrices: dict[tuple[str, str], pd.DataFrame] = {}

    for variant, description, frame in filter_variants(per_page):
        for dataset, group in frame.groupby("dataset", sort=True):
            dataset_label = group["dataset_label"].iloc[0]
            pages = int(group["image"].nunique())
            rows = int(len(group))
            valid = group[METRIC_COLS].dropna()
            enough_rows = len(valid) >= 3
            matrix = correlation_matrix(group) if enough_rows else pd.DataFrame(index=METRIC_COLS, columns=METRIC_COLS)
            matrices[(variant, dataset)] = matrix

            for row_metric in METRIC_COLS:
                for col_metric in METRIC_COLS:
                    value = matrix.loc[row_metric, col_metric] if row_metric in matrix.index and col_metric in matrix.columns else float("nan")
                    long_rows.append(
                        {
                            "variant": variant,
                            "variant_description": description,
                            "dataset": dataset,
                            "dataset_label": dataset_label,
                            "pages": pages,
                            "rows": rows,
                            "row_metric": row_metric,
                            "col_metric": col_metric,
                            "pearson_r": value,
                        }
                    )

            pll_rows.append(
                {
                    "variant": variant,
                    "variant_description": description,
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "pages": pages,
                    "rows": rows,
                    "pll_vs_wer": matrix.loc["pll_score", "wer"] if enough_rows else float("nan"),
                    "pll_vs_cer": matrix.loc["pll_score", "cer"] if enough_rows else float("nan"),
                    "pll_vs_levenshtein": matrix.loc["pll_score", "levenshtein"] if enough_rows else float("nan"),
                    "mean_pll_correlation": matrix.loc["pll_score", ["wer", "cer", "levenshtein"]].mean() if enough_rows else float("nan"),
                }
            )

    long_df = pd.DataFrame(long_rows)
    pll_summary = pd.DataFrame(pll_rows).sort_values(["variant", "mean_pll_correlation"], ascending=[True, False])
    long_df.to_csv(LONG_CSV_PATH, index=False)
    pll_summary.to_csv(PLL_SUMMARY_CSV_PATH, index=False)
    return long_df, pll_summary, matrices


def matrix_to_markdown(matrix: pd.DataFrame) -> str:
    display = matrix.copy()
    display.index = [display_metric(idx) for idx in display.index]
    display.columns = [display_metric(col) for col in display.columns]
    display = display.apply(lambda col: col.map(format_value))
    headers = [""] + list(display.columns)
    rows = [[idx] + list(display.loc[idx].values) for idx in display.index]
    return markdown_table(headers, rows)


def write_markdown(per_page: pd.DataFrame, pll_summary: pd.DataFrame, matrices: dict[tuple[str, str], pd.DataFrame]) -> None:
    lines: list[str] = []
    lines.append(f"# Per-Handwriting Correlation Report")
    lines.append("")
    lines.append(f"Run: `{RUN_ID}`")
    lines.append("")
    lines.append(
        "This report computes Pearson correlation matrices between the four scoring methods for each handwriting "
        "set: PLL, WER, CER, and Levenshtein. Rows are model-image comparisons. The main section uses the same "
        "worst-20% page trim from the current run, and the appendix keeps the all-page base-valid view for comparison."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Handwriting sets in source table: {per_page['dataset'].nunique()}")
    lines.append(f"- Model-image rows in source table: {len(per_page)}")
    lines.append(f"- Base-valid rows: {int(per_page['base_valid'].sum())}")
    if "keep_after_worst_page_trim" in per_page.columns:
        removed_pages = int(
            per_page[["dataset", "image", "removed_by_worst_page_trim"]]
            .drop_duplicates()["removed_by_worst_page_trim"]
            .sum()
        )
        lines.append(f"- Pages removed by worst-20% diagnostic trim: {removed_pages}")
    lines.append("")

    main_variant = "base_valid_after_worst20_trim" if "base_valid_after_worst20_trim" in set(pll_summary["variant"]) else "base_valid_all_pages"
    main = pll_summary[pll_summary["variant"] == main_variant].copy()
    lines.append("## PLL Correlation Summary")
    display = main[["dataset_label", "pages", "rows", "pll_vs_wer", "pll_vs_cer", "pll_vs_levenshtein", "mean_pll_correlation"]].copy()
    display.columns = ["Handwriting", "Pages", "Rows", "PLL-WER", "PLL-CER", "PLL-Lev", "Mean"]
    for col in ["PLL-WER", "PLL-CER", "PLL-Lev", "Mean"]:
        display[col] = display[col].map(format_value)
    lines.append(markdown_table(list(display.columns), display.values.tolist()))
    lines.append("")

    lines.append("## Per-Handwriting Matrices After Trim")
    for _, row in main.sort_values("dataset_label").iterrows():
        dataset = row["dataset"]
        matrix = matrices.get((main_variant, dataset))
        lines.append("")
        lines.append(f"### {row['dataset_label']}")
        lines.append(f"Pages: {int(row['pages'])}; rows: {int(row['rows'])}")
        if matrix is None or matrix.isna().all().all():
            lines.append("")
            lines.append("Not enough valid rows to compute a matrix.")
            continue
        lines.append("")
        lines.append(matrix_to_markdown(matrix))

    appendix_variant = "base_valid_all_pages"
    appendix = pll_summary[pll_summary["variant"] == appendix_variant].copy()
    if not appendix.empty and appendix_variant != main_variant:
        lines.append("")
        lines.append("## Appendix: All Base-Valid Pages")
        for _, row in appendix.sort_values("dataset_label").iterrows():
            dataset = row["dataset"]
            matrix = matrices.get((appendix_variant, dataset))
            lines.append("")
            lines.append(f"### {row['dataset_label']}")
            lines.append(f"Pages: {int(row['pages'])}; rows: {int(row['rows'])}")
            if matrix is None or matrix.isna().all().all():
                lines.append("")
                lines.append("Not enough valid rows to compute a matrix.")
                continue
            lines.append("")
            lines.append(matrix_to_markdown(matrix))

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def add_pdf_title(pdf: PdfPages, per_page: pd.DataFrame, pll_summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.93, "Per-Handwriting Correlation Report", fontsize=22, weight="bold")
    fig.text(0.08, 0.895, f"Run: {RUN_ID}", fontsize=12)
    text = (
        "This report shows Pearson correlation matrices between PLL, WER, CER, and Levenshtein for each handwriting "
        "set. The main tables use the current worst-20% page trim where available, so pages suspected of broken "
        "ground truth do not drive the per-handwriting summary."
    )
    fig.text(0.08, 0.84, textwrap.fill(text, width=88), fontsize=11, va="top")
    key_rows = [
        ["Handwriting sets", int(per_page["dataset"].nunique())],
        ["Source rows", int(len(per_page))],
        ["Base-valid rows", int(per_page["base_valid"].sum())],
        ["Report variants", int(pll_summary["variant"].nunique())],
    ]
    table_ax = fig.add_axes([0.12, 0.48, 0.76, 0.24])
    table_ax.axis("off")
    table = table_ax.table(cellText=key_rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_summary(pdf: PdfPages, pll_summary: pd.DataFrame) -> None:
    main_variant = "base_valid_after_worst20_trim" if "base_valid_after_worst20_trim" in set(pll_summary["variant"]) else "base_valid_all_pages"
    main = pll_summary[pll_summary["variant"] == main_variant].copy()
    display = main[["dataset_label", "pages", "rows", "pll_vs_wer", "pll_vs_cer", "pll_vs_levenshtein"]].copy()
    display = display.sort_values("pll_vs_wer", ascending=False)
    display.columns = ["Handwriting", "Pages", "Rows", "PLL-WER", "PLL-CER", "PLL-Lev"]
    for col in ["PLL-WER", "PLL-CER", "PLL-Lev"]:
        display[col] = display[col].map(format_value)
    display["Handwriting"] = display["Handwriting"].map(lambda value: textwrap.fill(str(value), width=28, break_long_words=False))

    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.suptitle("PLL Correlation by Handwriting", fontsize=18, weight="bold", y=0.96)
    table_ax = fig.add_axes([0.04, 0.08, 0.92, 0.82])
    table_ax.axis("off")
    table = table_ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.35)
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_matrix_pages(pdf: PdfPages, pll_summary: pd.DataFrame, matrices: dict[tuple[str, str], pd.DataFrame]) -> None:
    main_variant = "base_valid_after_worst20_trim" if "base_valid_after_worst20_trim" in set(pll_summary["variant"]) else "base_valid_all_pages"
    main = pll_summary[pll_summary["variant"] == main_variant].sort_values("dataset_label").copy()
    rows = list(main.itertuples(index=False))
    for start in range(0, len(rows), 2):
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig.suptitle("Per-Handwriting 4-Method Correlation Matrices", fontsize=16, weight="bold", y=0.97)
        for ax, row in zip(axes, rows[start:start + 2]):
            ax.axis("off")
            matrix = matrices.get((main_variant, row.dataset))
            ax.set_title(f"{row.dataset_label} | pages={int(row.pages)}, rows={int(row.rows)}", fontsize=11, pad=10)
            if matrix is None or matrix.isna().all().all():
                ax.text(0.5, 0.5, "Not enough valid rows", ha="center", va="center")
                continue
            display = matrix.copy()
            display.index = [display_metric(idx) for idx in display.index]
            display.columns = [display_metric(col) for col in display.columns]
            display = display.apply(lambda col: col.map(format_value))
            table = ax.table(
                cellText=display.values,
                rowLabels=display.index,
                colLabels=display.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9.5)
            table.scale(1, 1.55)
        for ax in axes[len(rows[start:start + 2]):]:
            ax.axis("off")
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)


def write_pdf(per_page: pd.DataFrame, pll_summary: pd.DataFrame, matrices: dict[tuple[str, str], pd.DataFrame]) -> None:
    with PdfPages(PDF_PATH) as pdf:
        add_pdf_title(pdf, per_page, pll_summary)
        add_pdf_summary(pdf, pll_summary)
        add_pdf_matrix_pages(pdf, pll_summary, matrices)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_page = load_per_page()
    _, pll_summary, matrices = build_tables(per_page)
    write_markdown(per_page, pll_summary, matrices)
    write_pdf(per_page, pll_summary, matrices)
    print(MD_PATH)
    print(PDF_PATH)
    print(LONG_CSV_PATH)
    print(PLL_SUMMARY_CSV_PATH)


if __name__ == "__main__":
    main()
