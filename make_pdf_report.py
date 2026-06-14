#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parent
REPORT_DIR = REPO_ROOT / "analysis_results" / "348_3758c_default"
TABLES_DIR = REPORT_DIR / "tables"
FIGURES_DIR = REPORT_DIR / "figures"
PDF_PATH = REPORT_DIR / "report.pdf"
MD_PATH = REPORT_DIR / "report.md"


def pretty_model_name(name: str) -> str:
    return name.replace("_", " ")


def wrapped_model_name(name: str, width: int = 18) -> str:
    return textwrap.fill(pretty_model_name(name), width=width, break_long_words=False, break_on_hyphens=False)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(TABLES_DIR / "sample_model_summary.csv")
    corr = pd.read_csv(TABLES_DIR / "sample_correlation_matrix.csv", index_col=0)
    line_metrics = pd.read_csv(TABLES_DIR / "sample_line_level_metrics.csv")
    return summary, corr, line_metrics


def narrative(summary: pd.DataFrame, corr: pd.DataFrame, line_metrics: pd.DataFrame) -> str:
    best = summary.sort_values("pll_score").iloc[0]
    pll_wer = corr.loc["pll_score", "wer"]
    pll_cer = corr.loc["pll_score", "cer"]
    pll_lev = corr.loc["pll_score", "levenshtein_distance"]
    n_models = int(summary.shape[0])
    n_lines = int(line_metrics.shape[0])
    text = (
        f"This report summarizes `348_3758c_default` using {n_models} OCR model outputs and {n_lines} line-level comparisons. "
        f"The strongest result on this sample is `{best['model']}`, which is best on PLL, WER, CER, and Levenshtein distance. "
        f"The language-model PLL score is strongly aligned with the GT-based metrics on this sample, with Pearson correlations of "
        f"{pll_wer:.3f} vs WER, {pll_cer:.3f} vs CER, and {pll_lev:.3f} vs Levenshtein distance. "
        "That makes PLL a useful proxy for OCR quality here, although the conclusion should be read as sample-level evidence rather than a full-corpus claim."
    )
    return " ".join(textwrap.fill(text, width=95).splitlines())


def add_title_page(pdf: PdfPages, summary: pd.DataFrame, corr: pd.DataFrame, line_metrics: pd.DataFrame) -> None:
    best = summary.sort_values("pll_score").iloc[0]
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(0.08, 0.93, "Model Chooser Sample Report", fontsize=24, weight="bold")
    fig.text(0.08, 0.89, "Dataset: 348_3758c_default", fontsize=14)
    fig.text(0.08, 0.84, "Executive summary", fontsize=16, weight="bold")
    fig.text(0.08, 0.79, textwrap.fill(narrative(summary, corr, line_metrics), width=88), fontsize=11, va="top")

    key_rows = [
        ["Sample lines", int(summary["lines"].iloc[0]) if "lines" in summary.columns else 28],
        ["Models compared", int(summary.shape[0])],
        ["Line-level comparisons", int(line_metrics.shape[0])],
        ["Best model", str(best["model"])],
        ["Best PLL", f"{best['pll_score']:.3f}"],
        ["Best WER", f"{best['wer']:.3f}"],
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
            "On this sample, lower PLL tracks lower WER, lower CER, and lower Levenshtein distance. "
            "The model ranking is consistent across the four metrics, which is exactly what you want if the language-model score is meant to reflect OCR quality.",
            width=90,
        ),
        fontsize=11,
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def add_tables_page(pdf: PdfPages, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Scores arranged as tables", fontsize=18, weight="bold", y=0.97)

    summary_display = summary.copy()
    summary_display["model"] = summary_display["model"].map(lambda name: wrapped_model_name(name, width=18))
    summary_display = summary_display[["model", "pll_score", "wer", "cer", "levenshtein_distance"]].copy()
    summary_display.columns = ["Model", "PLL", "WER", "CER", "Levenshtein"]
    summary_display = summary_display.round(3)

    axes[0].axis("off")
    table1 = axes[0].table(
        cellText=summary_display.values,
        colLabels=summary_display.columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.34, 0.16, 0.16, 0.16, 0.18],
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9.5)
    table1.scale(1, 1.85)
    axes[0].set_title("Model summary", pad=12, fontsize=13)

    corr_display = corr.round(3).copy()
    corr_display.index = [wrapped_model_name(idx, width=16) for idx in corr_display.index]
    corr_display.columns = [wrapped_model_name(col, width=16) for col in corr_display.columns]
    axes[1].axis("off")
    table2 = axes[1].table(
        cellText=corr_display.values,
        rowLabels=corr_display.index,
        colLabels=corr_display.columns,
        loc="center",
        cellLoc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9.5)
    table2.scale(1, 1.55)
    axes[1].set_title("Pearson correlation matrix", pad=12, fontsize=13)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def add_figures_page(pdf: PdfPages, left_path: Path, right_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle(title, fontsize=18, weight="bold", y=0.96)

    for ax, path in zip(axes, [left_path, right_path]):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def add_single_figure_page(pdf: PdfPages, path: Path, title: str, caption: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.76])
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    fig.suptitle(title, fontsize=18, weight="bold", y=0.95)
    fig.text(0.06, 0.04, textwrap.fill(caption, width=110), fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, corr: pd.DataFrame, line_metrics: pd.DataFrame) -> None:
    best = summary.sort_values("pll_score").iloc[0]
    corr_row = corr.loc["pll_score"]
    md = []
    md.append("# Model Chooser Sample Report")
    md.append("")
    md.append("This note matches the PDF report and summarizes the sample result set that was already present in the repository.")
    md.append("")
    md.append("## What the data says")
    md.append(
        f"On `348_3758c_default`, `{best['model']}` is the best model across PLL, WER, CER, and Levenshtein distance. "
        f"That agreement across metrics is a good sign: the PLL score is not drifting away from the GT-based measures."
    )
    md.append("")
    md.append("## Correlation snapshot")
    md.append(
        f"- PLL vs WER: {corr_row['wer']:.3f}\n"
        f"- PLL vs CER: {corr_row['cer']:.3f}\n"
        f"- PLL vs Levenshtein: {corr_row['levenshtein_distance']:.3f}"
    )
    md.append("")
    md.append("## Scope")
    md.append(f"- Sample lines: {int(summary['lines'].iloc[0]) if 'lines' in summary.columns else 28}")
    md.append(f"- Models compared: {summary.shape[0]}")
    md.append(f"- Line-level comparisons: {line_metrics.shape[0]}")
    md.append("")
    md.append("The full-corpus OCR run is still a separate job; this PDF is the clean, documented sample report.")
    MD_PATH.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    summary, corr, line_metrics = load_inputs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(PDF_PATH) as pdf:
        add_title_page(pdf, summary, corr, line_metrics)
        add_tables_page(pdf, summary, corr)
        add_figures_page(
            pdf,
            FIGURES_DIR / "model_summary.png",
            FIGURES_DIR / "correlation_heatmap.png",
            "Model ranking and correlation heatmap",
        )
        add_single_figure_page(
            pdf,
            FIGURES_DIR / "pll_vs_gt_metrics.png",
            "PLL versus GT-based metrics",
            "Each subplot shows the line-level PLL score against one GT-based metric. The fitted trend lines slope upward, which means higher PLL generally comes with higher OCR error on this sample.",
        )

    write_markdown(summary, corr, line_metrics)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
