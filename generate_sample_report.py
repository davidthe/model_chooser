#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import math
import re
import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jiwer
from kraken.lib.xml import parse_alto

import app


REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_ID = "348_3758c_default"
RESULTS_ROOT = REPO_ROOT / "analysis_results" / SAMPLE_ID
FIGURES_ROOT = RESULTS_ROOT / "figures"
TABLES_ROOT = RESULTS_ROOT / "tables"
GT_CANDIDATES = [
    REPO_ROOT / "pictures_examples" / "gt_for_pics" / f"{SAMPLE_ID}.txt",
    REPO_ROOT / "tests" / "results" / SAMPLE_ID / "gt_for_pics" / f"{SAMPLE_ID}.txt",
]
XML_ROOT = REPO_ROOT / "xml_output"
PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in string.punctuation + "׳״“”‘’—–…"})

WER_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({"ﭏ": "אל"}),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])

CER_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({"ﭏ": "אל"}),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def setup_dirs() -> None:
    for path in [RESULTS_ROOT, FIGURES_ROOT, TABLES_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def normalize(text: str) -> str:
    text = text.replace("ﭏ", "אל")
    text = text.translate(PUNCT_TRANSLATION)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def parse_gt_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def parse_pred_lines(path: Path) -> list[str]:
    val = parse_alto(str(path))
    return [line.get("text", "").strip() for line in val.get("lines", []) if line.get("text", "").strip()]


def score_pll(line: str) -> float:
    if not line:
        return float("nan")
    return float(app.get_score_from_text([line]))


def compute_metrics(gt: str, pred: str) -> dict[str, float]:
    gt_n = normalize(gt)
    pred_n = normalize(pred)
    return {
        "wer": jiwer.wer(gt, pred, reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM),
        "cer": jiwer.cer(gt, pred, reference_transform=CER_TRANSFORM, hypothesis_transform=CER_TRANSFORM),
        "levenshtein_distance": float(levenshtein_distance(gt_n, pred_n)),
    }


def extract_model_name(xml_path: Path) -> str:
    stem = xml_path.stem
    return stem[: -len(SAMPLE_ID)] if stem.endswith(SAMPLE_ID) else stem


def load_rows() -> pd.DataFrame:
    gt_path = next((path for path in GT_CANDIDATES if path.exists()), None)
    if gt_path is None:
        raise FileNotFoundError("Could not find the sample GT text in either pictures_examples or tests/results.")
    gt_lines = parse_gt_lines(gt_path)
    rows = []
    for xml_path in sorted(XML_ROOT.glob(f"*{SAMPLE_ID}.xml")):
        model = extract_model_name(xml_path)
        pred_lines = parse_pred_lines(xml_path)
        count = min(len(gt_lines), len(pred_lines))
        for idx in range(count):
            gt = gt_lines[idx]
            pred = pred_lines[idx]
            metrics = compute_metrics(gt, pred)
            rows.append({
                "model": model,
                "line_number": idx + 1,
                "gt_text": gt,
                "ocr_text": pred,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "levenshtein_distance": metrics["levenshtein_distance"],
                "pll_score": score_pll(pred),
            })
    return pd.DataFrame(rows)


def save_table_image(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.45 * len(df) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(title, pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation between PLL and GT-based scores")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_model_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()
    metrics = [
        ("pll_score", "Mean PLL"),
        ("wer", "Mean WER"),
        ("cer", "Mean CER"),
        ("levenshtein_distance", "Mean Levenshtein"),
    ]
    order = summary.sort_values("pll_score")["model"]
    for ax, (col, title) in zip(axes, metrics):
        sns.barplot(data=summary, x="model", y=col, order=order, ax=ax, color="#2563eb")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Sample handwriting comparison by model", y=1.02, fontsize=14)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    pairs = [
        ("pll_score", "wer", "PLL vs WER"),
        ("pll_score", "cer", "PLL vs CER"),
        ("pll_score", "levenshtein_distance", "PLL vs Levenshtein"),
    ]
    for ax, (x, y, title) in zip(axes, pairs):
        sns.regplot(data=df, x=x, y=y, scatter_kws={"s": 26, "alpha": 0.7}, line_kws={"color": "black"}, ax=ax)
        ax.set_title(title)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    best_pll = summary.sort_values("pll_score").iloc[0]
    best_wer = summary.sort_values("wer").iloc[0]
    best_cer = summary.sort_values("cer").iloc[0]
    best_lev = summary.sort_values("levenshtein_distance").iloc[0]

    lines = []
    lines.append("# Sample report for `348_3758c_default`")
    lines.append("")
    lines.append("This report is based on the result artifacts already present in the repository.")
    lines.append("It summarizes the four model outputs that were available for the sample image.")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Ground truth lines: {df['line_number'].nunique()}")
    lines.append(f"- Models evaluated: {df['model'].nunique()}")
    lines.append(f"- Line-level comparisons: {len(df)}")
    lines.append("")
    lines.append("## Model summary")
    lines.append("```text")
    lines.append(summary.round(3).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Best scores on this sample")
    lines.append(f"- Best PLL: `{best_pll['model']}`")
    lines.append(f"- Best WER: `{best_wer['model']}`")
    lines.append(f"- Best CER: `{best_cer['model']}`")
    lines.append(f"- Best Levenshtein: `{best_lev['model']}`")
    lines.append("")
    lines.append("## Correlation matrix")
    lines.append("```text")
    lines.append(corr.round(3).to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Note")
    lines.append("The repo is now prepared for a full corpus evaluation run, but the runtime visible in this session does not expose a GPU, so the complete dataset sweep has not been finished here.")
    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_dirs()
    df = load_rows()
    if df.empty:
        raise RuntimeError("No sample rows were collected.")

    df.to_csv(TABLES_ROOT / "sample_line_level_metrics.csv", index=False)

    summary = (
        df.groupby("model", as_index=False)
        .agg(
            pll_score=("pll_score", "mean"),
            wer=("wer", "mean"),
            cer=("cer", "mean"),
            levenshtein_distance=("levenshtein_distance", "mean"),
            lines=("line_number", "nunique"),
        )
        .sort_values("pll_score", ascending=True)
    )
    summary.to_csv(TABLES_ROOT / "sample_model_summary.csv", index=False)

    corr = df[["pll_score", "wer", "cer", "levenshtein_distance"]].corr(method="pearson")
    corr.to_csv(TABLES_ROOT / "sample_correlation_matrix.csv")

    corr_table = corr.round(3).reset_index().rename(columns={"index": "metric"})
    save_table_image(corr_table, FIGURES_ROOT / "correlation_table.png", "Pearson correlation matrix")
    plot_correlation(corr, FIGURES_ROOT / "correlation_heatmap.png")
    plot_model_summary(summary, FIGURES_ROOT / "model_summary.png")
    plot_scatter(df, FIGURES_ROOT / "pll_vs_gt_metrics.png")
    write_summary(df, summary, corr)

    appendix = df.copy()
    appendix["wer"] = appendix["wer"].round(4)
    appendix["cer"] = appendix["cer"].round(4)
    appendix["levenshtein_distance"] = appendix["levenshtein_distance"].round(2)
    appendix["pll_score"] = appendix["pll_score"].round(3)
    (RESULTS_ROOT / "appendix.md").write_text("```text\n" + appendix.to_string(index=False) + "\n```", encoding="utf-8")


if __name__ == "__main__":
    main()
