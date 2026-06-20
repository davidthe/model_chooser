#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parent
FULL_RUN_ID = "full_run_20260617"
SELECTED_RUN_ID = "full_run_20260617_selected_handwritings"
OUT_ID = "final_run_20260617_pll_proxy"

FULL_DIR = REPO_ROOT / "analysis_results" / FULL_RUN_ID
SELECTED_DIR = REPO_ROOT / "analysis_results" / SELECTED_RUN_ID
OUT_DIR = REPO_ROOT / "analysis_results" / OUT_ID
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
REPORT_MD = OUT_DIR / "report.md"
REPORT_PDF = OUT_DIR / "report.pdf"
MANIFEST_PATH = OUT_DIR / "manifest.json"

METRICS = ["wer", "cer", "levenshtein"]
DISPLAY = {
    "pll_score": "Raw PLL",
    "pll_per_pred_char": "PLL / predicted char",
    "wer": "WER",
    "cer": "CER",
    "levenshtein": "Levenshtein",
}

GOOD_EXAMPLE = (
    "vatican44amitaimidrashtanchu",
    "217_39817_default.jpg",
    "Vatican 44 page where PLL ranking and GT ranking almost perfectly agree",
)
SECOND_EXAMPLE = (
    "bnf150amitaitanchuma2",
    "Midrach_rabba_Nombres_(hebreu)__btv1b105399567_Page_375_Image_0001.tif",
    "BNF 150 Amitai page where PLL / predicted char tracks WER across models",
)
FALLBACK_SECOND_IMAGE = "Midrach_rabba_Nombres_(hébreu)__btv1b105399567_Page_375_Image_0001.tif"
CAVEAT_EXAMPLE = (
    "munichstaatsbibliothektest2",
    "146_899d5_default.jpg",
    "Munich page showing why PLL is a proxy, not a perfect replacement",
)


def setup() -> None:
    for path in [OUT_DIR, TABLES_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fmt(value: object, digits: int = 3) -> str:
    try:
        value_float = float(value)
    except Exception:
        return ""
    if not math.isfinite(value_float):
        return ""
    return f"{value_float:.{digits}f}"


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


def load_tables() -> dict[str, pd.DataFrame]:
    full_page = pd.read_csv(FULL_DIR / "tables" / "per_page_metrics.csv")
    selected_page = pd.read_csv(SELECTED_DIR / "tables" / "per_page_metrics.csv")
    corr_summary = pd.read_csv(FULL_DIR / "tables" / "correlation_summary.csv")
    selected_mean = pd.read_csv(
        SELECTED_DIR / "tables" / "base_valid_after_worst20_trim_mean_dataset_correlation_matrix.csv",
        index_col=0,
    )
    full_mean = pd.read_csv(
        FULL_DIR / "tables" / "base_valid_after_worst20_trim_mean_dataset_correlation_matrix.csv",
        index_col=0,
    )
    return {
        "full_page": full_page,
        "selected_page": selected_page,
        "corr_summary": corr_summary,
        "selected_mean": selected_mean,
        "full_mean": full_mean,
    }


def validation_frame(df: pd.DataFrame, high_confidence: bool = True) -> pd.DataFrame:
    frame = df[df["base_valid"] & df["keep_after_worst_page_trim"]].copy()
    if high_confidence:
        frame = frame[frame["page_high_confidence"]].copy()
    return frame


def get_corr(corr_summary: pd.DataFrame, table: str, score: str, metric: str, col: str = "pearson_r") -> float:
    rows = corr_summary[
        (corr_summary["table"] == table)
        & (corr_summary["score"] == score)
        & (corr_summary["metric"] == metric)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][col])


def corr_pair(df: pd.DataFrame, x: str, y: str, method: str = "pearson") -> float:
    valid = df[[x, y]].dropna()
    if len(valid) < 3:
        return float("nan")
    return float(valid[x].corr(valid[y], method=method))


def model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["model", "model_label"], sort=True)
        .agg(
            pages=("image", "nunique"),
            rows=("image", "size"),
            pll_score=("pll_score", "mean"),
            pll_per_pred_char=("pll_per_pred_char", "mean"),
            wer=("wer", "mean"),
            cer=("cer", "mean"),
            levenshtein=("levenshtein", "mean"),
        )
        .reset_index()
    )
    for col in ["pll_score", "pll_per_pred_char", "wer", "cer", "levenshtein"]:
        summary[f"{col}_rank"] = summary[col].rank(method="min", ascending=True).astype(int)
    return summary.sort_values("pll_per_pred_char")


def selected_handwriting_correlations(selected_page: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = validation_frame(selected_page, high_confidence=False)
    for dataset, group in frame.groupby("dataset", sort=True):
        if len(group) < 3:
            continue
        row = {
            "dataset": dataset,
            "dataset_label": group["dataset_label"].iloc[0],
            "pages": int(group["image"].nunique()),
            "rows": int(len(group)),
        }
        for metric in METRICS:
            row[f"pll_{metric}_r"] = corr_pair(group, "pll_score", metric)
        row["mean_pll_gt_r"] = float(np.nanmean([row[f"pll_{metric}_r"] for metric in METRICS]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_pll_gt_r", ascending=False)


def evidence_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    corr_summary = tables["corr_summary"]
    full_clean = validation_frame(tables["full_page"], high_confidence=True)
    selected_clean = validation_frame(tables["selected_page"], high_confidence=False)
    selected_models = model_summary(selected_clean)
    full_models = model_summary(validation_frame(tables["full_page"], high_confidence=False))
    selected_mean = tables["selected_mean"]
    full_mean = tables["full_mean"]

    rows = [
        {
            "evidence": "No-GT normalized page score",
            "comparison": "PLL / predicted char vs WER",
            "scope": "Full run, high-confidence rows, worst 20% diagnostic pages removed",
            "value": get_corr(corr_summary, "page_worst_20pct_high_confidence", "pll_per_pred_char", "wer"),
            "stat": "Pearson r",
            "n": int(len(full_clean.dropna(subset=["pll_per_pred_char", "wer"]))),
            "interpretation": "Operational no-GT score crosses the strong-correlation threshold.",
        },
        {
            "evidence": "No-GT absolute page score",
            "comparison": "Raw PLL vs Levenshtein",
            "scope": "Full run, high-confidence rows, worst 20% diagnostic pages removed",
            "value": get_corr(corr_summary, "page_worst_20pct_high_confidence", "pll_score", "levenshtein"),
            "stat": "Pearson r",
            "n": int(len(full_clean.dropna(subset=["pll_score", "levenshtein"]))),
            "interpretation": "Raw PLL tracks absolute edit burden.",
        },
        {
            "evidence": "No-GT absolute validation ceiling",
            "comparison": "Raw PLL vs Levenshtein",
            "scope": "Full base-valid run before diagnostic page trim",
            "value": get_corr(corr_summary, "page_base", "pll_score", "levenshtein"),
            "stat": "Pearson r",
            "n": int(tables["full_page"][tables["full_page"]["base_valid"]].dropna(subset=["pll_score", "levenshtein"]).shape[0]),
            "interpretation": "The strongest corpus-level result: PLL strongly predicts absolute GT edit distance.",
        },
        {
            "evidence": "Selected handwriting mean",
            "comparison": "Mean dataset PLL vs WER",
            "scope": "Mean of 7 selected handwriting-level correlations",
            "value": float(selected_mean.loc["pll_score", "wer"]),
            "stat": "Mean Pearson r",
            "n": 7,
            "interpretation": "Every selected dataset gets equal weight; PLL-WER remains strong.",
        },
        {
            "evidence": "Selected handwriting mean",
            "comparison": "Mean dataset PLL vs CER",
            "scope": "Mean of 7 selected handwriting-level correlations",
            "value": float(selected_mean.loc["pll_score", "cer"]),
            "stat": "Mean Pearson r",
            "n": 7,
            "interpretation": "The same equal-weight dataset view also supports PLL-CER.",
        },
        {
            "evidence": "Selected handwriting mean",
            "comparison": "Mean dataset PLL vs Levenshtein",
            "scope": "Mean of 7 selected handwriting-level correlations",
            "value": float(selected_mean.loc["pll_score", "levenshtein"]),
            "stat": "Mean Pearson r",
            "n": 7,
            "interpretation": "PLL is consistently aligned with absolute edit distance in the selected sets.",
        },
        {
            "evidence": "Selected model choice",
            "comparison": "PLL model rank vs WER model rank",
            "scope": "7 selected handwriting sets aggregated by model",
            "value": corr_pair(selected_models, "pll_per_pred_char", "wer", method="spearman"),
            "stat": "Spearman rho",
            "n": int(len(selected_models)),
            "interpretation": "Choosing models by PLL gives almost the same ordering as choosing by WER.",
        },
        {
            "evidence": "Full model choice",
            "comparison": "PLL model rank vs WER model rank",
            "scope": "Full run after diagnostic page trim, aggregated by model",
            "value": corr_pair(full_models, "pll_per_pred_char", "wer", method="spearman"),
            "stat": "Spearman rho",
            "n": int(len(full_models)),
            "interpretation": "Model-level ranking remains strong on the broader corpus.",
        },
        {
            "evidence": "Validation-only upper bound",
            "comparison": "PLL / mean GT-pred length vs WER",
            "scope": "Full run, high-confidence rows, worst 20% diagnostic pages removed",
            "value": get_corr(corr_summary, "page_worst_20pct_high_confidence", "pll_per_char_mean", "wer"),
            "stat": "Pearson r",
            "n": int(len(full_clean.dropna(subset=["pll_per_char_mean", "wer"]))),
            "interpretation": "This is the best normalized validation score, but it uses GT length and is not no-GT.",
        },
        {
            "evidence": "Full dataset equal-weight mean",
            "comparison": "Mean dataset PLL vs Levenshtein",
            "scope": "Mean of all available handwriting-level correlations after diagnostic trim",
            "value": float(full_mean.loc["pll_score", "levenshtein"]),
            "stat": "Mean Pearson r",
            "n": 11,
            "interpretation": "The full set is moderate-to-strong and just below the 0.70 threshold.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "evidence_summary.csv", index=False)
    selected_models.to_csv(TABLES_DIR / "selected_model_proxy_ranking.csv", index=False)
    full_models.to_csv(TABLES_DIR / "full_model_proxy_ranking.csv", index=False)
    return df


def example_rows(selected_page: pd.DataFrame, example: tuple[str, str, str], filename: str) -> pd.DataFrame:
    dataset, image, _ = example
    frame = selected_page[(selected_page["dataset"] == dataset) & (selected_page["image"] == image) & selected_page["base_valid"]].copy()
    if frame.empty and image == SECOND_EXAMPLE[1]:
        frame = selected_page[
            (selected_page["dataset"] == dataset)
            & (selected_page["image"] == FALLBACK_SECOND_IMAGE)
            & selected_page["base_valid"]
        ].copy()
    keep = [
        "dataset_label",
        "image",
        "model_label",
        "pll_score",
        "pll_per_pred_char",
        "wer",
        "cer",
        "levenshtein",
        "page_weighted_wc",
    ]
    frame = frame[keep].sort_values("pll_per_pred_char")
    frame.to_csv(TABLES_DIR / filename, index=False)
    return frame


def write_outputs(tables: dict[str, pd.DataFrame], evidence: pd.DataFrame) -> dict[str, pd.DataFrame]:
    selected_page = tables["selected_page"]
    handwriting = selected_handwriting_correlations(selected_page)
    handwriting.to_csv(TABLES_DIR / "selected_handwriting_proxy_correlations.csv", index=False)
    good = example_rows(selected_page, GOOD_EXAMPLE, "example_vatican44_proxy_matches_gt.csv")
    second = example_rows(selected_page, SECOND_EXAMPLE, "example_bnf150_proxy_matches_gt.csv")
    caveat = example_rows(selected_page, CAVEAT_EXAMPLE, "example_munich_proxy_caveat.csv")
    return {"handwriting": handwriting, "good": good, "second": second, "caveat": caveat}


def add_trendline(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return
    xs = valid["x"].to_numpy(dtype=float)
    ys = valid["y"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    x_line = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=2)


def plot_workflow() -> Path:
    path = FIGURES_DIR / "workflow_gt_vs_pll.png"
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.6,
            edgecolor="#1f2937",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, textwrap.fill(text, 18), ha="center", va="center", fontsize=12, weight="bold")

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color="#374151",
            )
        )

    ax.text(0.5, 0.94, "Two ways to score OCR quality", ha="center", va="center", fontsize=20, weight="bold")
    ax.text(0.25, 0.86, "Ground-truth metrics", ha="center", va="center", fontsize=15, weight="bold", color="#7f1d1d")
    ax.text(0.75, 0.86, "PLL-only proxy", ha="center", va="center", fontsize=15, weight="bold", color="#064e3b")

    box(0.08, 0.66, 0.22, 0.10, "OCR output", "#fee2e2")
    box(0.08, 0.46, 0.22, 0.10, "Human ground truth transcription", "#fecaca")
    box(0.08, 0.26, 0.22, 0.10, "WER / CER / Levenshtein", "#fca5a5")
    box(0.08, 0.08, 0.22, 0.10, "Quality decision", "#fee2e2")
    arrow(0.19, 0.66, 0.19, 0.56)
    arrow(0.19, 0.46, 0.19, 0.36)
    arrow(0.19, 0.26, 0.19, 0.18)
    ax.text(0.35, 0.51, "Requires a manually prepared reference text", fontsize=12, va="center", color="#7f1d1d")

    box(0.58, 0.66, 0.22, 0.10, "OCR output", "#d1fae5")
    box(0.58, 0.46, 0.22, 0.10, "Language model scores text plausibility", "#a7f3d0")
    box(0.58, 0.26, 0.22, 0.10, "PLL score or PLL / predicted char", "#6ee7b7")
    box(0.58, 0.08, 0.22, 0.10, "Quality estimate and model ranking", "#d1fae5")
    arrow(0.69, 0.66, 0.69, 0.56)
    arrow(0.69, 0.46, 0.69, 0.36)
    arrow(0.69, 0.26, 0.69, 0.18)
    ax.text(0.82, 0.51, "Needs no reference transcription", fontsize=12, va="center", color="#064e3b")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter(frame: pd.DataFrame, x: str, y: str, path: Path, title: str, subtitle: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.scatterplot(data=frame, x=x, y=y, hue="model_label", ax=ax, alpha=0.72, s=42, edgecolor="white", linewidth=0.4)
    add_trendline(ax, frame[x], frame[y])
    r = corr_pair(frame, x, y)
    rho = corr_pair(frame, x, y, method="spearman")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.text(0.01, 0.99, f"{subtitle}\nPearson r={fmt(r)}, Spearman rho={fmt(rho)}, n={len(frame.dropna(subset=[x, y]))}", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlabel(DISPLAY.get(x, x))
    ax.set_ylabel(DISPLAY.get(y, y))
    ax.legend(title="Model", fontsize=8, title_fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selected_mean_heatmap(selected_mean: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "selected_mean_dataset_correlation_heatmap.png"
    display = selected_mean.copy()
    display.index = [DISPLAY.get(idx, idx) for idx in display.index]
    display.columns = [DISPLAY.get(col, col) for col in display.columns]
    fig, ax = plt.subplots(figsize=(7.6, 6))
    sns.heatmap(display, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Selected sets: mean of per-dataset correlations", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_handwriting_bars(handwriting: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "selected_handwriting_pll_gt_correlation_bars.png"
    display = handwriting.copy()
    display["label_short"] = display["dataset_label"].str.wrap(24)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(display["label_short"], display["mean_pll_gt_r"], color="#2563eb")
    ax.axvline(0.7, color="#dc2626", linestyle="--", linewidth=1.5, label="r = 0.70")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Mean correlation of raw PLL with WER, CER, and Levenshtein")
    ax.set_title("Selected handwritings: PLL agreement with GT metrics", fontsize=14, weight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_rank_agreement(selected_models: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "selected_model_rank_agreement.png"
    rank_cols = ["pll_per_pred_char_rank", "wer_rank", "cer_rank", "levenshtein_rank"]
    display = selected_models.sort_values("pll_per_pred_char_rank").set_index("model_label")[rank_cols]
    display.columns = ["PLL / pred char", "WER", "CER", "Levenshtein"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.heatmap(display, annot=True, fmt="d", cmap="YlGnBu_r", cbar_kws={"label": "Rank; 1 is best"}, ax=ax)
    ax.set_title("Model ranking: PLL-only score vs GT-needed metrics", fontsize=14, weight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_example(frame: pd.DataFrame, path: Path, title: str) -> Path:
    display = frame.sort_values("pll_per_pred_char").copy()
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(display))
    pll_values = display["pll_per_pred_char"].to_numpy(dtype=float)
    wer_values = display["wer"].to_numpy(dtype=float)
    ax1.bar(x - 0.18, pll_values, width=0.36, color="#0f766e", label="PLL / predicted char")
    ax1.set_ylabel("PLL / predicted char; lower is better", color="#0f766e")
    ax1.tick_params(axis="y", labelcolor="#0f766e")
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, wer_values, marker="o", color="#b91c1c", linewidth=2, label="WER")
    ax2.set_ylabel("WER; lower is better", color="#b91c1c")
    ax2.tick_params(axis="y", labelcolor="#b91c1c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display["model_label"], rotation=30, ha="right")
    ax1.set_title(title, fontsize=14, weight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(tables: dict[str, pd.DataFrame], extra: dict[str, pd.DataFrame]) -> dict[str, Path]:
    full_clean = validation_frame(tables["full_page"], high_confidence=True)
    selected_clean = validation_frame(tables["selected_page"], high_confidence=False)
    selected_models = model_summary(selected_clean)
    figures = {
        "workflow": plot_workflow(),
        "scatter_pred_wer": plot_scatter(
            full_clean,
            "pll_per_pred_char",
            "wer",
            FIGURES_DIR / "scatter_pll_per_pred_char_vs_wer.png",
            "No-GT normalized PLL predicts WER",
            "Full run, high-confidence rows, worst diagnostic pages removed",
        ),
        "scatter_raw_lev": plot_scatter(
            full_clean,
            "pll_score",
            "levenshtein",
            FIGURES_DIR / "scatter_raw_pll_vs_levenshtein.png",
            "Raw PLL predicts absolute edit distance",
            "Full run, high-confidence rows, worst diagnostic pages removed",
        ),
        "selected_heatmap": plot_selected_mean_heatmap(tables["selected_mean"]),
        "handwriting_bars": plot_handwriting_bars(extra["handwriting"]),
        "rank_agreement": plot_model_rank_agreement(selected_models),
        "example_good": plot_example(
            extra["good"],
            FIGURES_DIR / "example_vatican44_proxy_matches_gt.png",
            "Example: Vatican 44 page, PLL and WER choose the same good models",
        ),
        "example_second": plot_example(
            extra["second"],
            FIGURES_DIR / "example_bnf150_proxy_matches_gt.png",
            "Example: BNF 150 Amitai page, PLL follows the GT metrics",
        ),
        "example_caveat": plot_example(
            extra["caveat"],
            FIGURES_DIR / "example_munich_proxy_caveat.png",
            "Caveat example: one page where normalized PLL is not enough by itself",
        ),
    }
    return figures


def evidence_rows_for_md(evidence: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in evidence.itertuples(index=False):
        rows.append([
            row.evidence,
            row.comparison,
            row.stat,
            fmt(row.value),
            str(int(row.n)),
            row.interpretation,
        ])
    return rows


def example_rows_for_md(frame: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in frame.itertuples(index=False):
        rows.append([
            row.model_label,
            fmt(row.pll_score),
            fmt(row.pll_per_pred_char),
            fmt(row.wer),
            fmt(row.cer),
            fmt(row.levenshtein),
        ])
    return rows


def write_markdown(evidence: pd.DataFrame, extra: dict[str, pd.DataFrame], figures: dict[str, Path]) -> None:
    selected_mean = pd.read_csv(
        SELECTED_DIR / "tables" / "base_valid_after_worst20_trim_mean_dataset_correlation_matrix.csv",
        index_col=0,
    )
    lines: list[str] = []
    lines.append("# Final PLL Proxy Report")
    lines.append("")
    lines.append(f"Run: `{OUT_ID}`")
    lines.append(f"Evidence source runs: `{FULL_RUN_ID}` and `{SELECTED_RUN_ID}`")
    lines.append("")
    lines.append("## Main Claim")
    lines.append("")
    lines.append(
        "PLL scores can replace ground-truth-needed metrics for the practical task of screening OCR outputs "
        "and choosing a model when no transcription is available. They should not be described as a perfect "
        "replacement for a final benchmark, because WER, CER, and Levenshtein still measure direct agreement "
        "with human ground truth. The defensible claim is: PLL gives a strong no-GT proxy for model ranking "
        "and page-level quality estimation."
    )
    lines.append("")
    lines.append("## Why PLL Can Work Without Ground Truth")
    lines.append("")
    lines.append(
        "WER, CER, and Levenshtein need a reference transcription. PLL does not. PLL asks how plausible the OCR "
        "text is under a language model. OCR mistakes usually create unlikely letter sequences, broken words, "
        "or impossible phrases. Those outputs receive a worse PLL score. Therefore, when several OCR models read "
        "the same manuscript page, the model with lower PLL is often also the model that would have lower WER/CER "
        "if ground truth existed."
    )
    lines.append("")
    lines.append("Operationally, the no-GT scores are:")
    lines.append("")
    lines.append("- `pll_score`: raw page PLL, best for comparing candidate models on the same page and for absolute edit burden.")
    lines.append("- `pll_per_pred_char`: normalized by OCR output length, best for no-GT comparison across pages or datasets.")
    lines.append("")
    lines.append(
        "`pll_per_gt_char` and `pll_per_char_mean` are useful validation scores, but they use GT length, so they are "
        "not no-GT deployment methods."
    )
    lines.append("")
    lines.append(f"![Workflow diagram]({figures['workflow'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append("## Evidence Summary")
    lines.append("")
    lines.append(markdown_table(["Evidence", "Comparison", "Stat", "Value", "N", "Interpretation"], evidence_rows_for_md(evidence)))
    lines.append("")
    lines.append("## Key Drawings")
    lines.append("")
    lines.append(f"![PLL per predicted char vs WER]({figures['scatter_pred_wer'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append(f"![Raw PLL vs Levenshtein]({figures['scatter_raw_lev'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append(f"![Selected mean correlation heatmap]({figures['selected_heatmap'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append(f"![Per-handwriting PLL agreement]({figures['handwriting_bars'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append(f"![Selected model rank agreement]({figures['rank_agreement'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append("## Selected Handwriting Mean Correlation Table")
    lines.append("")
    display = selected_mean.copy()
    display.index = [DISPLAY.get(idx, idx) for idx in display.index]
    display.columns = [DISPLAY.get(col, col) for col in display.columns]
    display = display.apply(lambda col: col.map(fmt))
    lines.append(markdown_table([""] + list(display.columns), [[idx] + list(display.loc[idx]) for idx in display.index]))
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    lines.append("### Example 1: Vatican 44")
    lines.append("")
    lines.append(
        "On this page, the model with the best PLL / predicted character is also the model with by far the best "
        "WER, CER, and Levenshtein. This is the cleanest picture of how PLL can replace GT metrics for model "
        "choice when GT is unavailable."
    )
    lines.append("")
    lines.append(markdown_table(["Model", "Raw PLL", "PLL/pred char", "WER", "CER", "Lev"], example_rows_for_md(extra["good"])))
    lines.append("")
    lines.append(f"![Vatican example]({figures['example_good'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append("### Example 2: BNF 150 Amitai")
    lines.append("")
    lines.append(
        "This page shows the same pattern: as PLL / predicted character gets worse, WER and CER also get worse. "
        "The no-GT score recovers the same model ordering that a GT evaluation would have shown."
    )
    lines.append("")
    lines.append(markdown_table(["Model", "Raw PLL", "PLL/pred char", "WER", "CER", "Lev"], example_rows_for_md(extra["second"])))
    lines.append("")
    lines.append(f"![BNF example]({figures['example_second'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append("### Caveat Example")
    lines.append("")
    lines.append(
        "The Munich page shows why this should be called a proxy, not a perfect replacement. On some difficult "
        "or structurally unusual pages, normalized PLL can disagree with WER. In practice, this means PLL should "
        "be used with confidence filters, page sanity checks, and occasional GT calibration."
    )
    lines.append("")
    lines.append(markdown_table(["Model", "Raw PLL", "PLL/pred char", "WER", "CER", "Lev"], example_rows_for_md(extra["caveat"])))
    lines.append("")
    lines.append(f"![Caveat example]({figures['example_caveat'].relative_to(OUT_DIR)})")
    lines.append("")
    lines.append("## Recommended Use")
    lines.append("")
    lines.append("1. For untranscribed pages, score each OCR output with `pll_per_pred_char` and choose the lowest score.")
    lines.append("2. If comparing models on the exact same page, also inspect raw `pll_score`; it tracks absolute edit distance strongly.")
    lines.append("3. Use OCR confidence and text-length sanity checks before trusting the score.")
    lines.append("4. Validate once on a small GT sample; after calibration, use PLL for large-scale no-GT ranking.")
    lines.append("5. Report the method as a no-GT proxy for ranking and screening, not as a total replacement for final GT benchmarks.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `tables/evidence_summary.csv`")
    lines.append("- `tables/selected_model_proxy_ranking.csv`")
    lines.append("- `tables/full_model_proxy_ranking.csv`")
    lines.append("- `tables/selected_handwriting_proxy_correlations.csv`")
    lines.append("- `figures/` with workflow, scatter plots, heatmaps, model ranking, and examples")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str], table: pd.DataFrame | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.07, 0.94, title, fontsize=21, weight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        fig.text(0.07, y, textwrap.fill(paragraph, width=92), fontsize=10.5, va="top")
        y -= 0.065 + 0.012 * max(0, len(paragraph) // 92)
    if table is not None and not table.empty:
        table_ax = fig.add_axes([0.05, 0.08, 0.90, min(0.52, max(0.18, 0.065 * (len(table) + 1)))])
        table_ax.axis("off")
        rendered = table.copy()
        for col in rendered.columns:
            if pd.api.types.is_float_dtype(rendered[col]):
                rendered[col] = rendered[col].map(fmt)
        mpl_table = table_ax.table(
            cellText=rendered.values,
            colLabels=rendered.columns,
            loc="center",
            cellLoc="center",
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(7.7)
        mpl_table.scale(1, 1.35)
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, path: Path, title: str, caption: str = "") -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=17, weight="bold", va="top")
    if caption:
        fig.text(0.07, 0.91, textwrap.fill(caption, width=95), fontsize=10, va="top")
    image = plt.imread(path)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.78])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(evidence: pd.DataFrame, figures: dict[str, Path]) -> None:
    evidence_table = evidence[["evidence", "comparison", "stat", "value", "n"]].copy()
    evidence_table.columns = ["Evidence", "Comparison", "Stat", "Value", "N"]
    with PdfPages(REPORT_PDF) as pdf:
        add_text_page(
            pdf,
            "Final PLL Proxy Report",
            [
                f"Evidence source runs: {FULL_RUN_ID} and {SELECTED_RUN_ID}.",
                "Main claim: PLL can replace ground-truth-needed metrics for screening OCR outputs and choosing models when no transcription is available. It is a proxy for ranking, not a perfect substitute for a final WER/CER benchmark.",
                "The operational no-GT scores are raw PLL and PLL per predicted character. Scores using GT length are kept only as validation checks.",
            ],
            evidence_table,
        )
        add_image_page(pdf, figures["workflow"], "Scoring Workflow", "GT metrics require reference transcription; PLL only requires OCR text and a language model.")
        add_image_page(pdf, figures["scatter_pred_wer"], "No-GT Normalized PLL vs WER")
        add_image_page(pdf, figures["scatter_raw_lev"], "Raw PLL vs Levenshtein")
        add_image_page(pdf, figures["selected_heatmap"], "Selected Handwriting Mean Correlations")
        add_image_page(pdf, figures["handwriting_bars"], "Per-Handwriting Agreement")
        add_image_page(pdf, figures["rank_agreement"], "Model Ranking Agreement")
        add_image_page(pdf, figures["example_good"], "Clean Example")
        add_image_page(pdf, figures["example_second"], "Second Clean Example")
        add_image_page(pdf, figures["example_caveat"], "Caveat Example")


def write_manifest(evidence: pd.DataFrame, figures: dict[str, Path]) -> None:
    manifest = {
        "run_id": OUT_ID,
        "source_runs": [FULL_RUN_ID, SELECTED_RUN_ID],
        "claim": "PLL-only scores can act as no-GT proxies for OCR model ranking and page quality screening.",
        "report_md": str(REPORT_MD.relative_to(REPO_ROOT)),
        "report_pdf": str(REPORT_PDF.relative_to(REPO_ROOT)),
        "figures": {key: str(path.relative_to(REPO_ROOT)) for key, path in figures.items()},
        "evidence_rows": int(len(evidence)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    setup()
    tables = load_tables()
    evidence = evidence_summary(tables)
    extra = write_outputs(tables, evidence)
    figures = make_figures(tables, extra)
    write_markdown(evidence, extra, figures)
    write_pdf(evidence, figures)
    write_manifest(evidence, figures)
    print(REPORT_MD)
    print(REPORT_PDF)
    print(TABLES_DIR / "evidence_summary.csv")
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
