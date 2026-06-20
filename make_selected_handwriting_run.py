#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import json
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
SOURCE_RUN_ID = os.environ.get("SOURCE_RUN_ID", "full_run_20260617")
RUN_ID = os.environ.get("RUN_ID", "full_run_20260617_selected_handwritings")
SOURCE_DIR = REPO_ROOT / "analysis_results" / SOURCE_RUN_ID
RUN_DIR = REPO_ROOT / "analysis_results" / RUN_ID
TABLES_DIR = RUN_DIR / "tables"
FIGURES_DIR = RUN_DIR / "figures"

SELECTED_DATASETS = [
    "vatican44amitaimidrashtanchu",
    "bnf150test2",
    "bnf150amitaitanchuma2",
    "seferhaikarimbnf740experimen",
    "huntingtonamitaitanchuma2",
    "vatican34midrashtanchu",
    "munichstaatsbibliothektest2",
]

METRIC_COLS = ["pll_score", "wer", "cer", "levenshtein"]
DISPLAY_NAMES = {
    "pll_score": "PLL",
    "wer": "WER",
    "cer": "CER",
    "levenshtein": "Levenshtein",
}

REPORT_MD = RUN_DIR / "report.md"
REPORT_PDF = RUN_DIR / "report.pdf"
SUMMARY_MD = RUN_DIR / "summary.md"


def setup_dirs() -> None:
    for path in [RUN_DIR, TABLES_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def load_source_tables() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    per_page_path = SOURCE_DIR / "tables" / "per_page_metrics.csv"
    if not per_page_path.exists():
        raise FileNotFoundError(f"Missing source table: {per_page_path}")
    per_page = pd.read_csv(per_page_path)

    line_path = SOURCE_DIR / "tables" / "line_level_metrics.csv"
    line_df = pd.read_csv(line_path) if line_path.exists() else None
    return per_page, line_df


def filter_selected(per_page: pd.DataFrame, line_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    selected = per_page[per_page["dataset"].isin(SELECTED_DATASETS)].copy()
    selected["selected_run"] = RUN_ID
    selected.to_csv(TABLES_DIR / "per_page_metrics.csv", index=False)

    selected_lines = None
    if line_df is not None and "dataset" in line_df.columns:
        selected_lines = line_df[line_df["dataset"].isin(SELECTED_DATASETS)].copy()
        selected_lines["selected_run"] = RUN_ID
        selected_lines.to_csv(TABLES_DIR / "line_level_metrics.csv", index=False)
    return selected, selected_lines


def corr_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[METRIC_COLS].dropna()
    if len(valid) < 3:
        return pd.DataFrame(index=METRIC_COLS, columns=METRIC_COLS)
    return valid.corr(method="pearson").reindex(index=METRIC_COLS, columns=METRIC_COLS)


def matrix_display(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.copy()
    display.index = [display_metric(idx) for idx in display.index]
    display.columns = [display_metric(col) for col in display.columns]
    return display


def matrix_to_markdown(matrix: pd.DataFrame) -> str:
    display = matrix_display(matrix).apply(lambda col: col.map(format_value))
    headers = [""] + list(display.columns)
    rows = [[idx] + list(display.loc[idx].values) for idx in display.index]
    return markdown_table(headers, rows)


def build_correlations(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    base = selected[selected["base_valid"]].copy()
    kept = base[base["keep_after_worst_page_trim"]] if "keep_after_worst_page_trim" in base.columns else base

    overall = corr_matrix(kept)
    overall.to_csv(TABLES_DIR / "selected_overall_correlation_matrix.csv")

    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    matrices: dict[str, pd.DataFrame] = {}

    for dataset, group in kept.groupby("dataset", sort=True):
        label = group["dataset_label"].iloc[0]
        matrix = corr_matrix(group)
        matrices[dataset] = matrix
        matrix.to_csv(TABLES_DIR / f"{dataset}_correlation_matrix.csv")

        pages = int(group["image"].nunique())
        rows = int(len(group))
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_label": label,
                "pages": pages,
                "rows": rows,
                "pll_vs_wer": matrix.loc["pll_score", "wer"],
                "pll_vs_cer": matrix.loc["pll_score", "cer"],
                "pll_vs_levenshtein": matrix.loc["pll_score", "levenshtein"],
                "mean_pll_correlation": matrix.loc["pll_score", ["wer", "cer", "levenshtein"]].mean(),
            }
        )
        for row_metric in METRIC_COLS:
            for col_metric in METRIC_COLS:
                long_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": label,
                        "pages": pages,
                        "rows": rows,
                        "row_metric": row_metric,
                        "col_metric": col_metric,
                        "pearson_r": matrix.loc[row_metric, col_metric],
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values("mean_pll_correlation", ascending=False)
    long_df = pd.DataFrame(long_rows)
    summary.to_csv(TABLES_DIR / "selected_handwriting_pll_summary.csv", index=False)
    long_df.to_csv(TABLES_DIR / "selected_handwriting_correlation_long.csv", index=False)
    return overall, summary, long_df, matrices


def selected_model_summary(selected: pd.DataFrame) -> pd.DataFrame:
    base = selected[selected["base_valid"]].copy()
    kept = base[base["keep_after_worst_page_trim"]] if "keep_after_worst_page_trim" in base.columns else base
    summary = (
        kept.groupby(["model", "model_label"], as_index=False)
        .agg(
            mean_pll=("pll_score", "mean"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_levenshtein=("levenshtein", "mean"),
            rows=("model", "size"),
            pages=("image", "nunique"),
            datasets=("dataset", "nunique"),
        )
        .sort_values("mean_pll")
    )
    summary.to_csv(TABLES_DIR / "selected_model_summary.csv", index=False)
    return summary


def plot_overall_heatmap(overall: pd.DataFrame) -> None:
    plt.figure(figsize=(7.5, 5.5))
    sns.heatmap(matrix_display(overall), annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".3f")
    plt.title("Selected handwriting sets: 4-method correlation matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "selected_overall_correlation_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_pll_summary(summary: pd.DataFrame) -> None:
    plot = summary.sort_values("pll_vs_wer", ascending=True).copy()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot, x="pll_vs_wer", y="dataset_label", color="#2563eb")
    plt.axvline(0.70, color="#111827", linestyle="--", linewidth=1)
    plt.xlim(-0.1, 1)
    plt.xlabel("Pearson r: PLL vs WER")
    plt.ylabel("")
    plt.title("PLL-WER correlation by selected handwriting set")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "selected_pll_wer_by_handwriting.png", dpi=220, bbox_inches="tight")
    plt.close()


def write_markdown(
    selected: pd.DataFrame,
    selected_lines: pd.DataFrame | None,
    overall: pd.DataFrame,
    handwriting_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
) -> None:
    base = selected[selected["base_valid"]].copy()
    kept = base[base["keep_after_worst_page_trim"]] if "keep_after_worst_page_trim" in base.columns else base

    lines: list[str] = []
    lines.append("# Selected Handwriting Run Report")
    lines.append("")
    lines.append(f"Run: `{RUN_ID}`")
    lines.append(f"Source run: `{SOURCE_RUN_ID}`")
    lines.append("")
    lines.append("This report focuses on the requested handwriting sets and reuses the completed OCR predictions.")
    lines.append("")
    lines.append("## Selected Handwriting Sets")
    for _, row in selected[["dataset", "dataset_label"]].drop_duplicates().sort_values("dataset_label").iterrows():
        lines.append(f"- {row['dataset_label']} (`{row['dataset']}`)")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Selected handwriting sets: {selected['dataset'].nunique()}")
    lines.append(f"- Source model-image rows: {len(selected)}")
    lines.append(f"- Base-valid rows: {len(base)}")
    lines.append(f"- Rows used after current page-trim flag: {len(kept)}")
    lines.append(f"- Images used after trim: {kept[['dataset', 'image']].drop_duplicates().shape[0]}")
    if selected_lines is not None:
        lines.append(f"- Prepared line rows for these sets: {len(selected_lines)}")
    lines.append("")

    lines.append("## Overall 4-Method Correlation Matrix")
    lines.append(matrix_to_markdown(overall))
    lines.append("")

    lines.append("## PLL Correlation by Handwriting")
    display = handwriting_summary[["dataset_label", "pages", "rows", "pll_vs_wer", "pll_vs_cer", "pll_vs_levenshtein", "mean_pll_correlation"]].copy()
    display.columns = ["Handwriting", "Pages", "Rows", "PLL-WER", "PLL-CER", "PLL-Lev", "Mean"]
    for col in ["PLL-WER", "PLL-CER", "PLL-Lev", "Mean"]:
        display[col] = display[col].map(format_value)
    lines.append(markdown_table(list(display.columns), display.values.tolist()))
    lines.append("")

    lines.append("## Model Ranking on Selected Sets")
    model_display = model_summary[["model_label", "mean_pll", "mean_wer", "mean_cer", "mean_levenshtein", "pages", "datasets"]].copy()
    model_display.columns = ["Model", "Mean PLL", "Mean WER", "Mean CER", "Mean Lev", "Pages", "Sets"]
    for col in ["Mean PLL", "Mean WER", "Mean CER", "Mean Lev"]:
        model_display[col] = model_display[col].map(format_value)
    lines.append(markdown_table(list(model_display.columns), model_display.values.tolist()))
    lines.append("")

    lines.append("## Per-Handwriting 4-Method Matrices")
    for _, row in handwriting_summary.sort_values("dataset_label").iterrows():
        dataset = row["dataset"]
        matrix = matrices[dataset]
        lines.append("")
        lines.append(f"### {row['dataset_label']}")
        lines.append(f"Pages: {int(row['pages'])}; rows: {int(row['rows'])}")
        lines.append("")
        lines.append(matrix_to_markdown(matrix))

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def add_pdf_title(pdf: PdfPages, selected: pd.DataFrame, selected_lines: pd.DataFrame | None) -> None:
    base = selected[selected["base_valid"]].copy()
    kept = base[base["keep_after_worst_page_trim"]] if "keep_after_worst_page_trim" in base.columns else base
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.93, "Selected Handwriting Run Report", fontsize=22, weight="bold")
    fig.text(0.08, 0.895, f"Run: {RUN_ID}", fontsize=12)
    fig.text(0.08, 0.86, "Requested handwriting sets", fontsize=15, weight="bold")
    names = selected[["dataset_label"]].drop_duplicates().sort_values("dataset_label")["dataset_label"].tolist()
    fig.text(0.08, 0.82, textwrap.fill(", ".join(names), width=88), fontsize=10.5, va="top")

    key_rows = [
        ["Selected sets", selected["dataset"].nunique()],
        ["Source rows", len(selected)],
        ["Base-valid rows", len(base)],
        ["Rows used after trim", len(kept)],
        ["Images used after trim", kept[["dataset", "image"]].drop_duplicates().shape[0]],
        ["Prepared line rows", len(selected_lines) if selected_lines is not None else 0],
    ]
    table_ax = fig.add_axes([0.10, 0.38, 0.80, 0.28])
    table_ax.axis("off")
    table = table_ax.table(cellText=key_rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.55)
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_overview(pdf: PdfPages, overall: pd.DataFrame, handwriting_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
    fig.suptitle("Selected Set Correlations", fontsize=18, weight="bold", y=0.97)
    sns.heatmap(matrix_display(overall), annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".3f", ax=axes[0])
    axes[0].set_title("Overall 4-method matrix")
    plot = handwriting_summary.sort_values("pll_vs_wer", ascending=True).copy()
    sns.barplot(data=plot, x="pll_vs_wer", y="dataset_label", ax=axes[1], color="#2563eb")
    axes[1].axvline(0.70, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_xlim(-0.1, 1)
    axes[1].set_xlabel("Pearson r: PLL vs WER")
    axes[1].set_ylabel("")
    axes[1].set_title("PLL-WER by handwriting")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_summary_table(pdf: PdfPages, handwriting_summary: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Selected Run Tables", fontsize=18, weight="bold", y=0.97)
    hand = handwriting_summary[["dataset_label", "pages", "rows", "pll_vs_wer", "pll_vs_cer", "pll_vs_levenshtein"]].copy()
    hand.columns = ["Handwriting", "Pages", "Rows", "PLL-WER", "PLL-CER", "PLL-Lev"]
    for col in ["PLL-WER", "PLL-CER", "PLL-Lev"]:
        hand[col] = hand[col].map(format_value)
    hand["Handwriting"] = hand["Handwriting"].map(lambda value: textwrap.fill(str(value), width=28, break_long_words=False))
    axes[0].axis("off")
    table = axes[0].table(cellText=hand.values, colLabels=hand.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.25)
    axes[0].set_title("PLL correlations by handwriting", pad=10)

    models = model_summary[["model_label", "mean_pll", "mean_wer", "mean_cer", "mean_levenshtein"]].copy()
    models.columns = ["Model", "Mean PLL", "Mean WER", "Mean CER", "Mean Lev"]
    for col in ["Mean PLL", "Mean WER", "Mean CER", "Mean Lev"]:
        models[col] = models[col].map(format_value)
    axes[1].axis("off")
    table = axes[1].table(cellText=models.values, colLabels=models.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.35)
    axes[1].set_title("Model ranking on selected sets", pad=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_matrix_pages(pdf: PdfPages, handwriting_summary: pd.DataFrame, matrices: dict[str, pd.DataFrame]) -> None:
    rows = list(handwriting_summary.sort_values("dataset_label").itertuples(index=False))
    for start in range(0, len(rows), 2):
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig.suptitle("Per-Handwriting 4-Method Matrices", fontsize=16, weight="bold", y=0.97)
        current = rows[start:start + 2]
        for ax, row in zip(axes, current):
            ax.axis("off")
            matrix = matrix_display(matrices[row.dataset]).apply(lambda col: col.map(format_value))
            ax.set_title(f"{row.dataset_label} | pages={int(row.pages)}, rows={int(row.rows)}", fontsize=11, pad=10)
            table = ax.table(
                cellText=matrix.values,
                rowLabels=matrix.index,
                colLabels=matrix.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9.5)
            table.scale(1, 1.5)
        for ax in axes[len(current):]:
            ax.axis("off")
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)


def write_pdf(
    selected: pd.DataFrame,
    selected_lines: pd.DataFrame | None,
    overall: pd.DataFrame,
    handwriting_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
) -> None:
    with PdfPages(REPORT_PDF) as pdf:
        add_pdf_title(pdf, selected, selected_lines)
        add_pdf_overview(pdf, overall, handwriting_summary)
        add_pdf_summary_table(pdf, handwriting_summary, model_summary)
        add_pdf_matrix_pages(pdf, handwriting_summary, matrices)


def write_manifest(selected: pd.DataFrame, selected_lines: pd.DataFrame | None) -> None:
    manifest = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "selected_datasets": SELECTED_DATASETS,
        "selected_dataset_labels": selected[["dataset", "dataset_label"]].drop_duplicates().sort_values("dataset")["dataset_label"].tolist(),
        "page_rows": int(len(selected)),
        "base_valid_rows": int(selected["base_valid"].sum()),
        "line_rows": int(len(selected_lines)) if selected_lines is not None else 0,
        "results_root": str(RUN_DIR.relative_to(REPO_ROOT)),
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    setup_dirs()
    per_page, line_df = load_source_tables()
    selected, selected_lines = filter_selected(per_page, line_df)
    overall, handwriting_summary, _, matrices = build_correlations(selected)
    model_summary = selected_model_summary(selected)
    plot_overall_heatmap(overall)
    plot_pll_summary(handwriting_summary)
    write_markdown(selected, selected_lines, overall, handwriting_summary, model_summary, matrices)
    write_pdf(selected, selected_lines, overall, handwriting_summary, model_summary, matrices)
    write_manifest(selected, selected_lines)
    print(REPORT_MD)
    print(REPORT_PDF)


if __name__ == "__main__":
    main()
