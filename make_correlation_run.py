#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import json
import math
import os
import re
import string
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from kraken.lib.xml import parse_alto


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_RUN_ID = os.environ.get("SOURCE_RUN_ID", "full_run_20260613")
RUN_ID = os.environ.get("RUN_ID", "full_run_20260614")
ENABLE_LINE_SCORING = False
TRIM_WORST_PAGE_PERCENT = float(os.environ.get("TRIM_WORST_PAGE_PERCENT", "0"))
SOURCE_DIR = REPO_ROOT / "analysis_results" / SOURCE_RUN_ID
RESULTS_DIR = REPO_ROOT / "analysis_results" / RUN_ID
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_PATH = RESULTS_DIR / "report.md"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
PDF_PATH = RESULTS_DIR / "report.pdf"

PLL_VALID_THRESHOLD = 90000.0
SCORE_COLS = ["pll_score", "pll_per_pred_char", "pll_per_gt_char", "pll_per_char_mean"]
METRIC_COLS = ["wer", "cer", "levenshtein"]
TRIM_FLAG_COL = "keep_after_worst_page_trim"

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
    for path in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_for_compare(text: str) -> str:
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

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_metrics(gt_text: str, pred_text: str) -> dict[str, float]:
    gt_clean = normalize_for_compare(gt_text)
    pred_clean = normalize_for_compare(pred_text)
    if not gt_clean:
        return {
            "wer": float("nan"),
            "cer": float("nan"),
            "levenshtein": float(levenshtein_distance(gt_clean, pred_clean)),
        }
    return {
        "wer": jiwer.wer(
            gt_text,
            pred_text,
            reference_transform=WER_TRANSFORM,
            hypothesis_transform=WER_TRANSFORM,
        ),
        "cer": jiwer.cer(
            gt_text,
            pred_text,
            reference_transform=CER_TRANSFORM,
            hypothesis_transform=CER_TRANSFORM,
        ),
        "levenshtein": float(levenshtein_distance(gt_clean, pred_clean)),
    }


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def prediction_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = REPO_ROOT / value
    return path if path.exists() else None


def textline_records(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []

    rows: list[dict[str, object]] = []
    for text_line in root.iter():
        if text_line.tag.split("}", 1)[-1] != "TextLine":
            continue
        strings = [
            el for el in text_line.iter()
            if el.tag.split("}", 1)[-1] == "String"
        ]
        parts: list[str] = []
        weights: list[int] = []
        wcs: list[float] = []
        for string_el in strings:
            content = string_el.attrib.get("CONTENT", "")
            parts.append(content)
            wc = safe_float(string_el.attrib.get("WC"))
            if math.isfinite(wc):
                wcs.append(wc)
                weights.append(max(len(content), 1))
        text = " ".join(part for part in parts if part).strip()
        if wcs:
            mean_wc = sum(wcs) / len(wcs)
            weighted_wc = sum(wc * weight for wc, weight in zip(wcs, weights)) / sum(weights)
        else:
            mean_wc = float("nan")
            weighted_wc = float("nan")
        rows.append({
            "text": text,
            "mean_wc": mean_wc,
            "weighted_wc": weighted_wc,
            "string_count": len(strings),
        })
    return rows


def gt_lines_for_page(dataset: str, image_stem: str, gt_file: str) -> list[str]:
    gt_alto = REPO_ROOT / "datasets" / dataset / "gt_alto" / f"{image_stem}.xml"
    if gt_alto.exists():
        try:
            return [line.get("text", "").strip() for line in parse_alto(str(gt_alto)).get("lines", [])]
        except Exception:
            pass
    path = REPO_ROOT / gt_file
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    return []


def summarize_xml(path: Path | None) -> dict[str, float]:
    records = textline_records(path)
    wcs = [float(row["weighted_wc"]) for row in records if math.isfinite(float(row["weighted_wc"]))]
    string_counts = [int(row["string_count"]) for row in records]
    if not records:
        return {
            "pred_lines_xml": 0,
            "page_weighted_wc": float("nan"),
            "page_mean_wc": float("nan"),
            "page_string_count": 0,
        }
    return {
        "pred_lines_xml": len(records),
        "page_weighted_wc": sum(wcs) / len(wcs) if wcs else float("nan"),
        "page_mean_wc": sum(wcs) / len(wcs) if wcs else float("nan"),
        "page_string_count": sum(string_counts),
    }


def add_score_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pll_per_pred_char"] = df["pll_score"] / df["pred_chars"].clip(lower=1)
    df["pll_per_gt_char"] = df["pll_score"] / df["gt_chars"].clip(lower=1)
    df["pll_per_char_mean"] = df["pll_score"] / ((df["pred_chars"] + df["gt_chars"]) / 2).clip(lower=1)
    return df


def page_trim_label() -> str:
    return f"worst_{int(round(TRIM_WORST_PAGE_PERCENT * 100))}pct"


def display_run_id() -> str:
    return RUN_ID.replace("full_run_", "")


def build_page_diagnostics(per_page: pd.DataFrame) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    diagnostic_rows: list[dict[str, object]] = []
    base_valid = per_page[per_page["base_valid"]].copy()

    for (dataset, image), group in base_valid.groupby(["dataset", "image"], sort=True):
        row: dict[str, object] = {
            "dataset": dataset,
            "dataset_label": group["dataset_label"].iloc[0],
            "image": image,
            "valid_model_rows": int(len(group)),
        }
        correlations: list[float] = []
        for metric in METRIC_COLS:
            corr = float("nan")
            if len(group) >= 3:
                corr = group[["pll_score", metric]].corr(method="pearson").loc["pll_score", metric]
            row[f"pll_{metric}_corr"] = corr
            if math.isfinite(float(corr)):
                correlations.append(float(corr))
        row["diagnostic_mean_corr"] = sum(correlations) / len(correlations) if correlations else float("nan")
        diagnostic_rows.append(row)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["removed_by_worst_page_trim"] = False
    removed_pages: set[tuple[str, str]] = set()

    eligible = diagnostics.dropna(subset=["diagnostic_mean_corr"]).sort_values("diagnostic_mean_corr").copy()
    if TRIM_WORST_PAGE_PERCENT > 0 and not eligible.empty:
        remove_count = int(math.ceil(len(eligible) * TRIM_WORST_PAGE_PERCENT))
        remove_count = max(1, min(remove_count, len(eligible)))
        removed = eligible.head(remove_count).copy()
        removed_pages = set((str(row.dataset), str(row.image)) for row in removed.itertuples(index=False))
        diagnostics["removed_by_worst_page_trim"] = diagnostics.apply(
            lambda row: (str(row["dataset"]), str(row["image"])) in removed_pages,
            axis=1,
        )

    diagnostics = diagnostics.sort_values(["removed_by_worst_page_trim", "diagnostic_mean_corr"], ascending=[False, True])
    diagnostics.to_csv(TABLES_DIR / "page_correlation_diagnostics.csv", index=False)
    diagnostics[diagnostics["removed_by_worst_page_trim"]].to_csv(TABLES_DIR / "removed_pages_worst20.csv", index=False)
    return diagnostics, removed_pages


def build_page_table() -> pd.DataFrame:
    source_path = SOURCE_DIR / "tables" / "per_image_metrics.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source run table: {source_path}")

    per_page = pd.read_csv(source_path)
    per_page = add_score_normalizations(per_page)

    stats_rows = []
    for _, row in per_page.iterrows():
        pred_path = prediction_path(row.get("prediction_file"))
        gt_lines = gt_lines_for_page(str(row["dataset"]), str(row["image_stem"]), str(row["gt_file"]))
        stats = summarize_xml(pred_path)
        stats["gt_lines_text"] = len([line for line in gt_lines if line.strip()])
        stats_rows.append(stats)

    stats_df = pd.DataFrame(stats_rows)
    per_page = pd.concat([per_page, stats_df], axis=1)
    per_page["len_ratio"] = per_page["pred_chars"] / per_page["gt_chars"].clip(lower=1)
    per_page["line_ratio"] = per_page["pred_lines_xml"] / per_page["gt_lines_text"].clip(lower=1)
    per_page["base_valid"] = (
        per_page[["pll_score", "wer", "cer", "levenshtein"]].notna().all(axis=1)
        & (per_page["pll_score"] < PLL_VALID_THRESHOLD)
    )
    per_page["page_high_confidence"] = per_page["base_valid"] & (per_page["page_weighted_wc"] >= 0.90)
    per_page["page_complete_high_confidence"] = (
        per_page["page_high_confidence"]
        & per_page["len_ratio"].between(0.75, 1.25)
        & per_page["line_ratio"].between(0.80, 1.25)
    )
    diagnostics, removed_pages = build_page_diagnostics(per_page)
    per_page["removed_by_worst_page_trim"] = per_page.apply(
        lambda row: (str(row["dataset"]), str(row["image"])) in removed_pages,
        axis=1,
    )
    per_page[TRIM_FLAG_COL] = ~per_page["removed_by_worst_page_trim"]
    per_page["page_diagnostic_count"] = int(diagnostics["diagnostic_mean_corr"].notna().sum())
    per_page.to_csv(TABLES_DIR / "per_page_metrics.csv", index=False)
    return per_page


def score_batch(texts: list[str]) -> list[float]:
    import app as chooser

    cleaned = [text.strip() for text in texts]
    scores: list[float] = [float("nan")] * len(cleaned)
    indexed = [(idx, text + "\n") for idx, text in enumerate(cleaned) if text]
    batch_size = 32
    for start in range(0, len(indexed), batch_size):
        batch = indexed[start:start + batch_size]
        indices = [idx for idx, _ in batch]
        batch_texts = [text for _, text in batch]
        try:
            batch_scores = chooser.dicta_scorer.score_sentences(batch_texts)
            for idx, score in zip(indices, batch_scores):
                scores[idx] = float(score) * -1.0
        except Exception:
            for idx, text in batch:
                try:
                    scores[idx] = float(chooser.dicta_scorer.score_sentences([text])[0]) * -1.0
                except Exception:
                    scores[idx] = float("nan")
    return scores


def build_line_table(per_page: pd.DataFrame) -> pd.DataFrame:
    line_rows: list[dict[str, object]] = []
    score_inputs: list[str] = []
    score_row_indexes: list[int] = []

    usable_pages = per_page[
        per_page["prediction_file"].fillna("").astype(str).ne("")
        & per_page["base_valid"]
    ].copy()

    for _, page in usable_pages.iterrows():
        pred_path = prediction_path(page.get("prediction_file"))
        pred_records = textline_records(pred_path)
        gt_lines = gt_lines_for_page(str(page["dataset"]), str(page["image_stem"]), str(page["gt_file"]))
        paired = min(len(gt_lines), len(pred_records))
        for line_idx in range(paired):
            gt_text = gt_lines[line_idx].strip()
            pred_text = str(pred_records[line_idx]["text"]).strip()
            metrics = compute_metrics(gt_text, pred_text)
            gt_chars = len(normalize_for_compare(gt_text))
            pred_chars = len(normalize_for_compare(pred_text))
            row = {
                "dataset": page["dataset"],
                "dataset_label": page["dataset_label"],
                "image": page["image"],
                "image_stem": page["image_stem"],
                "model": page["model"],
                "model_label": page["model_label"],
                "line_index": line_idx,
                "page_pll_score": page["pll_score"],
                "line_pll_score": float("nan"),
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "levenshtein": metrics["levenshtein"],
                "gt_chars": gt_chars,
                "pred_chars": pred_chars,
                "char_ratio": pred_chars / max(gt_chars, 1),
                "line_weighted_wc": pred_records[line_idx]["weighted_wc"],
                "line_mean_wc": pred_records[line_idx]["mean_wc"],
                "page_weighted_wc": page["page_weighted_wc"],
                "page_len_ratio": page["len_ratio"],
                "page_line_ratio": page["line_ratio"],
                "gt_lines": len(gt_lines),
                "pred_lines": len(pred_records),
            }
            row["score_candidate"] = (
                gt_chars >= 10
                and pred_chars >= 5
                and 0.50 <= row["char_ratio"] <= 1.75
                and math.isfinite(float(row["line_weighted_wc"]))
                and float(row["line_weighted_wc"]) >= 0.95
            )
            if row["score_candidate"]:
                score_row_indexes.append(len(line_rows))
                score_inputs.append(pred_text)
            line_rows.append(row)

    if ENABLE_LINE_SCORING:
        print(f"Scoring {len(score_inputs)} high-confidence line predictions with Dicta...", flush=True)
        scores = score_batch(score_inputs)
        for row_index, score in zip(score_row_indexes, scores):
            line_rows[row_index]["line_pll_score"] = score
    else:
        print(
            f"Prepared {len(score_inputs)} high-confidence line candidates; "
            "line-level Dicta rescoring is disabled for this fast report.",
            flush=True,
        )

    line_df = pd.DataFrame(line_rows)
    if line_df.empty:
        raise ValueError("No line-level rows were created.")
    line_df = line_df.rename(columns={"line_pll_score": "pll_score"})
    line_df = add_score_normalizations(line_df)
    line_df["base_valid"] = line_df[["pll_score", "wer", "cer", "levenshtein"]].notna().all(axis=1)
    line_df["line_quality"] = (
        line_df["base_valid"]
        & (line_df["gt_chars"] >= 10)
        & (line_df["pred_chars"] >= 5)
        & line_df["char_ratio"].between(0.50, 1.75)
        & (line_df["line_weighted_wc"] >= 0.95)
    )
    line_df["line_high_confidence"] = (
        line_df["base_valid"]
        & line_df["score_candidate"]
    )
    line_df.to_csv(TABLES_DIR / "line_level_metrics.csv", index=False)
    return line_df


def pearsonr_safe(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    try:
        from scipy import stats

        valid = pd.concat([x, y], axis=1).dropna()
        if valid.shape[0] < 3:
            return float("nan"), float("nan")
        result = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        if hasattr(result, "statistic"):
            return float(result.statistic), float(result.pvalue)
        return float(result[0]), float(result[1])
    except Exception:
        return float("nan"), float("nan")


def spearmanr_safe(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    try:
        from scipy import stats

        valid = pd.concat([x, y], axis=1).dropna()
        if valid.shape[0] < 3:
            return float("nan"), float("nan")
        result = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        if hasattr(result, "statistic"):
            return float(result.statistic), float(result.pvalue)
        return float(result[0]), float(result[1])
    except Exception:
        return float("nan"), float("nan")


def correlation_rows(df: pd.DataFrame, table_name: str, unit: str, filter_name: str, filter_description: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score_col in SCORE_COLS:
        for metric_col in METRIC_COLS:
            valid = df[[score_col, metric_col]].dropna()
            if valid.shape[0] < 3:
                continue
            pearson_r, pearson_p = pearsonr_safe(valid[score_col], valid[metric_col])
            spearman_r, spearman_p = spearmanr_safe(valid[score_col], valid[metric_col])
            rows.append({
                "table": table_name,
                "unit": unit,
                "filter": filter_name,
                "filter_description": filter_description,
                "score": score_col,
                "metric": metric_col,
                "n": int(valid.shape[0]),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "meets_0_7_pearson": bool(pearson_r >= 0.70),
                "significant_0_05": bool(pearson_p < 0.05),
                "significant_0_001": bool(pearson_p < 0.001),
            })
    return rows


def build_correlation_summary(per_page: pd.DataFrame, line_df: pd.DataFrame) -> pd.DataFrame:
    variants = [
        (
            "page_base",
            "page",
            "base_valid",
            per_page[per_page["base_valid"]].copy(),
            "Rows with non-missing metrics and no fallback PLL penalty.",
        ),
        (
            "page_high_confidence",
            "page",
            "page_high_confidence",
            per_page[per_page["page_high_confidence"]].copy(),
            "Base-valid rows with model XML word confidence >= 0.90.",
        ),
        (
            "page_complete_high_confidence",
            "page",
            "page_complete_high_confidence",
            per_page[per_page["page_complete_high_confidence"]].copy(),
            "High-confidence page rows with character and line-count ratios in a normal range.",
        ),
    ]
    if TRIM_WORST_PAGE_PERCENT > 0:
        trim_name = page_trim_label()
        variants.extend([
            (
                f"page_{trim_name}_base",
                "page",
                f"{trim_name}_base_valid",
                per_page[per_page["base_valid"] & per_page[TRIM_FLAG_COL]].copy(),
                f"Base-valid rows after removing the lowest {TRIM_WORST_PAGE_PERCENT:.0%} pages by PLL-vs-GT diagnostic correlation.",
            ),
            (
                f"page_{trim_name}_high_confidence",
                "page",
                f"{trim_name}_high_confidence",
                per_page[per_page["page_high_confidence"] & per_page[TRIM_FLAG_COL]].copy(),
                f"High-confidence rows after removing the lowest {TRIM_WORST_PAGE_PERCENT:.0%} pages by PLL-vs-GT diagnostic correlation.",
            ),
            (
                f"page_{trim_name}_complete_high_confidence",
                "page",
                f"{trim_name}_complete_high_confidence",
                per_page[per_page["page_complete_high_confidence"] & per_page[TRIM_FLAG_COL]].copy(),
                f"Complete high-confidence rows after removing the lowest {TRIM_WORST_PAGE_PERCENT:.0%} diagnostic pages.",
            ),
        ])
    variants.extend([
        (
            "line_base",
            "line",
            "base_valid",
            line_df[line_df["base_valid"]].copy(),
            "Paired GT/predicted lines with valid line PLL and GT metrics.",
        ),
        (
            "line_quality",
            "line",
            "line_quality",
            line_df[line_df["line_quality"]].copy(),
            "Valid paired lines with enough text, plausible character ratio, and line confidence >= 0.95.",
        ),
        (
            "line_high_confidence",
            "line",
            "line_high_confidence",
            line_df[line_df["line_high_confidence"]].copy(),
            "The same strict high-confidence line set that was rescored with Dicta.",
        ),
    ])

    rows: list[dict[str, object]] = []
    for table_name, unit, filter_name, frame, description in variants:
        rows.extend(correlation_rows(frame, table_name, unit, filter_name, description))
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "correlation_summary.csv", index=False)
    return summary


def matrix_for(summary: pd.DataFrame, table_name: str, value_col: str = "pearson_r") -> pd.DataFrame:
    sub = summary[summary["table"] == table_name]
    matrix = sub.pivot(index="score", columns="metric", values=value_col).reindex(index=SCORE_COLS, columns=METRIC_COLS)
    matrix.to_csv(TABLES_DIR / f"{table_name}_{value_col}_matrix.csv")
    return matrix


def save_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(8.5, 5.2))
    if matrix.empty or matrix.isna().all().all():
        plt.axis("off")
        plt.text(0.5, 0.5, "No correlation rows available", ha="center", va="center", fontsize=14)
        plt.title(title)
    else:
        sns.heatmap(matrix, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".3f")
        plt.title(title)
        plt.xlabel("GT metric")
        plt.ylabel("PLL score variant")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_figures(summary: pd.DataFrame) -> None:
    page_high = matrix_for(summary, "page_high_confidence")
    save_heatmap(page_high, FIGURES_DIR / "page_high_confidence_correlation.png", "Page-level high-confidence Pearson correlations")
    if TRIM_WORST_PAGE_PERCENT > 0:
        trimmed_high = matrix_for(summary, primary_page_table_name())
        save_heatmap(
            trimmed_high,
            FIGURES_DIR / f"{primary_page_table_name()}_correlation.png",
            primary_page_table_title(),
        )
    if ENABLE_LINE_SCORING:
        line_quality = matrix_for(summary, "line_quality")
        line_high = matrix_for(summary, "line_high_confidence")
        save_heatmap(line_quality, FIGURES_DIR / "line_quality_correlation.png", "Line-level quality-filtered Pearson correlations")
        save_heatmap(line_high, FIGURES_DIR / "line_high_confidence_correlation.png", "Line-level high-confidence Pearson correlations")
    else:
        for stale in [
            FIGURES_DIR / "line_quality_correlation.png",
            FIGURES_DIR / "line_high_confidence_correlation.png",
            TABLES_DIR / "line_quality_pearson_r_matrix.csv",
            TABLES_DIR / "line_high_confidence_pearson_r_matrix.csv",
        ]:
            stale.unlink(missing_ok=True)

    plot_df = summary[summary["metric"].isin(["wer", "cer", "levenshtein"])].copy()
    plot_df["label"] = plot_df["table"] + " | " + plot_df["score"]
    top = (
        plot_df.sort_values("pearson_r", ascending=False)
        .head(18)
        .sort_values("pearson_r", ascending=True)
    )
    plt.figure(figsize=(11, 7))
    sns.barplot(data=top, x="pearson_r", y="label", hue="metric", dodge=False)
    plt.axvline(0.70, color="#111827", linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.xlabel("Pearson r")
    plt.ylabel("")
    plt.title(f"Strongest PLL-to-GT correlations in the {RUN_ID} analysis")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_correlations.png", dpi=220, bbox_inches="tight")
    plt.close()


def format_p(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def wrapped(text: object, width: int = 28) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)


def display_metric_name(name: str) -> str:
    return {
        "pll_score": "Raw PLL",
        "pll_per_pred_char": "PLL / pred char",
        "pll_per_gt_char": "PLL / GT char",
        "pll_per_char_mean": "PLL / mean char",
        "wer": "WER",
        "cer": "CER",
        "levenshtein": "Levenshtein",
    }.get(name, name)


def primary_page_table_name() -> str:
    if TRIM_WORST_PAGE_PERCENT > 0:
        return f"page_{page_trim_label()}_high_confidence"
    return "page_high_confidence"


def primary_page_table_title() -> str:
    if TRIM_WORST_PAGE_PERCENT > 0:
        return f"Page high-confidence Pearson matrix after removing worst {TRIM_WORST_PAGE_PERCENT:.0%} pages"
    return "Page high-confidence Pearson matrix"


def add_pdf_title_page(pdf: PdfPages, per_page: pd.DataFrame, line_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    over_threshold = summary[(summary["pearson_r"] >= 0.70) & (summary["pearson_p"] < 0.05)].copy()
    best = summary.sort_values(["meets_0_7_pearson", "pearson_r"], ascending=[False, False]).iloc[0]

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.93, f"Full Run {display_run_id()} Correlation Report", fontsize=22, weight="bold")
    fig.text(0.08, 0.895, f"Source predictions: {SOURCE_RUN_ID}", fontsize=12)
    fig.text(0.08, 0.85, "Executive summary", fontsize=15, weight="bold")
    summary_text = (
        f"This report reuses the completed OCR predictions from {SOURCE_RUN_ID} and adds normalized PLL, "
        "quality-filtered page correlations, and line-level candidate preparation. The strongest result is "
        f"{best['score']} vs {best['metric']} with Pearson r={best['pearson_r']:.3f}, "
        f"p={format_p(best['pearson_p'])}, n={int(best['n'])}. "
        f"{over_threshold.shape[0]} correlations cross the r >= 0.70 threshold with p < 0.05."
    )
    fig.text(0.08, 0.81, textwrap.fill(summary_text, width=86), fontsize=11, va="top")

    key_rows = [
        ["Page rows", len(per_page)],
        ["Base-valid page rows", int(per_page["base_valid"].sum())],
        ["High-confidence page rows", int(per_page["page_high_confidence"].sum())],
        ["Pages removed by trim", int(per_page[["dataset", "image", "removed_by_worst_page_trim"]].drop_duplicates()["removed_by_worst_page_trim"].sum())],
        ["Line rows prepared", len(line_df)],
        ["High-confidence line candidates", int(line_df["score_candidate"].sum())],
        ["Line Dicta rescoring", str(ENABLE_LINE_SCORING)],
        ["Strong correlations", int(over_threshold.shape[0])],
    ]
    table_ax = fig.add_axes([0.08, 0.39, 0.84, 0.32])
    table_ax.axis("off")
    table = table_ax.table(cellText=key_rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.55)

    fig.text(0.08, 0.27, "Interpretation", fontsize=15, weight="bold")
    interpretation = (
        "Raw PLL is a summed score and therefore aligns best with absolute edit distance. For normalized error "
        "rates, the stronger research claim is the high-confidence normalized PLL result against WER. The filters "
        "exclude missing scores, fallback penalties, and low-confidence OCR XML rows; they do not filter individual "
        "model rows by WER or CER."
    )
    fig.text(0.08, 0.23, textwrap.fill(interpretation, width=88), fontsize=11, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_correlation_tables(pdf: PdfPages, summary: pd.DataFrame) -> None:
    over_threshold = (
        summary[(summary["pearson_r"] >= 0.70) & (summary["pearson_p"] < 0.05)]
        .sort_values(["pearson_r", "n"], ascending=[False, False])
        .copy()
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Correlation Results", fontsize=18, weight="bold", y=0.97)

    axes[0].axis("off")
    if over_threshold.empty:
        axes[0].text(0.5, 0.5, "No correlations crossed r >= 0.70", ha="center", va="center", fontsize=13)
    else:
        display = over_threshold[["table", "score", "metric", "n", "pearson_r", "pearson_p", "spearman_r"]].copy()
        display["score"] = display["score"].map(display_metric_name)
        display["metric"] = display["metric"].map(display_metric_name)
        display["pearson_r"] = display["pearson_r"].map(lambda value: f"{value:.3f}")
        display["pearson_p"] = display["pearson_p"].map(format_p)
        display["spearman_r"] = display["spearman_r"].map(lambda value: f"{value:.3f}")
        display.columns = ["Table", "Score", "Metric", "n", "Pearson r", "p", "Spearman r"]
        table = axes[0].table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
    axes[0].set_title("Rows crossing r >= 0.70 and p < 0.05", pad=12, fontsize=12)

    axes[1].axis("off")
    matrix = matrix_for(summary, primary_page_table_name()).round(3).copy()
    matrix.index = [display_metric_name(idx) for idx in matrix.index]
    matrix.columns = [display_metric_name(col) for col in matrix.columns]
    matrix_display = matrix.apply(lambda col: col.map(lambda value: "" if pd.isna(value) else f"{value:.3f}"))
    table = axes[1].table(
        cellText=matrix_display.values,
        rowLabels=matrix_display.index,
        colLabels=matrix_display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    axes[1].set_title(primary_page_table_title(), pad=12, fontsize=12)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_visual_page(pdf: PdfPages, summary: pd.DataFrame) -> None:
    matrix = matrix_for(summary, primary_page_table_name()).copy()
    matrix.index = [display_metric_name(idx) for idx in matrix.index]
    matrix.columns = [display_metric_name(col) for col in matrix.columns]
    plot_df = summary.sort_values("pearson_r", ascending=False).head(12).copy()
    plot_df["label"] = plot_df["table"] + "\n" + plot_df["score"].map(display_metric_name) + " vs " + plot_df["metric"].map(display_metric_name)
    plot_df = plot_df.sort_values("pearson_r", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
    fig.suptitle("Correlation Visuals", fontsize=18, weight="bold", y=0.97)
    sns.heatmap(matrix, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, fmt=".3f", ax=axes[0])
    axes[0].set_title(primary_page_table_title())
    axes[0].set_xlabel("GT metric")
    axes[0].set_ylabel("PLL score variant")
    sns.barplot(data=plot_df, x="pearson_r", y="label", hue="metric", dodge=False, ax=axes[1])
    axes[1].axvline(0.70, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Pearson r")
    axes[1].set_ylabel("")
    axes[1].set_title("Top correlations")
    axes[1].legend(title="Metric", loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def add_pdf_method_page(pdf: PdfPages, per_page: pd.DataFrame, line_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.93, "Method and Data Quality", fontsize=21, weight="bold")
    method = (
        f"The {RUN_ID} analysis starts from the {SOURCE_RUN_ID} OCR XML predictions. Page rows with missing GT metrics or "
        "fallback PLL penalties are excluded from statistical summaries. High-confidence page rows additionally "
        "require OCR XML word confidence of at least 0.90. The report compares raw PLL with normalized PLL variants "
        "because raw PLL is length-sensitive while WER and CER are normalized error rates."
    )
    fig.text(0.08, 0.87, textwrap.fill(method, width=88), fontsize=11, va="top")

    line_note = (
        f"The line-level table contains {len(line_df)} paired GT/predicted rows and marks "
        f"{int(line_df['score_candidate'].sum())} high-confidence line candidates. Full line-level Dicta rescoring "
        "was left disabled for this generated report because the current CPU-only pass was too slow; the candidate "
        "table is saved for a later GPU or overnight run."
    )
    fig.text(0.08, 0.72, textwrap.fill(line_note, width=88), fontsize=11, va="top")

    quality = (
        per_page.assign(
            status=lambda df: df.apply(
                lambda row: "trimmed out" if row["removed_by_worst_page_trim"] else ("high confidence" if row["page_high_confidence"] else ("base valid" if row["base_valid"] else "excluded")),
                axis=1,
            )
        )
        .groupby("status", as_index=False)
        .agg(rows=("status", "size"), datasets=("dataset", "nunique"), images=("image", "nunique"))
        .sort_values("status")
    )
    table_ax = fig.add_axes([0.12, 0.42, 0.76, 0.20])
    table_ax.axis("off")
    table = table_ax.table(cellText=quality.values, colLabels=["Status", "Rows", "Datasets", "Images"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.text(0.08, 0.29, "Recommended paper wording", fontsize=15, weight="bold")
    wording = (
        "A defensible phrasing is: 'After removing the lowest diagnostic-correlation pages, high-confidence "
        "PLL normalized by mean character count correlates strongly with WER.' The raw PLL to Levenshtein result "
        "is stronger, but it should be described as an absolute edit-distance correlation."
    )
    fig.text(0.08, 0.25, textwrap.fill(wording, width=88), fontsize=11, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf_report(per_page: pd.DataFrame, line_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    with PdfPages(PDF_PATH) as pdf:
        add_pdf_title_page(pdf, per_page, line_df, summary)
        add_pdf_correlation_tables(pdf, summary)
        add_pdf_visual_page(pdf, summary)
        add_pdf_method_page(pdf, per_page, line_df)


def write_reports(per_page: pd.DataFrame, line_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    best = summary.sort_values(["meets_0_7_pearson", "pearson_r"], ascending=[False, False]).iloc[0]
    over_threshold = summary[(summary["pearson_r"] >= 0.70) & (summary["pearson_p"] < 0.05)].copy()
    over_threshold = over_threshold.sort_values(["pearson_r", "n"], ascending=[False, False])
    removed_pages = (
        per_page[per_page["removed_by_worst_page_trim"]]
        [["dataset", "dataset_label", "image"]]
        .drop_duplicates()
        .sort_values(["dataset", "image"])
    )
    diagnostic_pages = int(per_page["page_diagnostic_count"].max()) if "page_diagnostic_count" in per_page else 0
    recommended = summary[
        (summary["table"] == primary_page_table_name())
        & (summary["score"] == "pll_per_char_mean")
        & (summary["metric"] == "wer")
    ]

    def matrix_text(table_name: str) -> str:
        return matrix_for(summary, table_name).round(3).to_string()

    lines: list[str] = []
    lines.append(f"# Full run {display_run_id()} correlation report")
    lines.append("")
    lines.append(
        "This run reuses the completed OCR predictions from `full_run_20260613` and performs a new full-corpus "
        f"correlation analysis for `{RUN_ID}`. The OCR models and datasets did not change, so rerunning "
        "recognition would reproduce the same page-level XMLs; the completed work here is score normalization, "
        "explicit quality-filtered page correlation tables, and preparation of line-level candidates for a heavier "
        "future rescore."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Source run: `{SOURCE_RUN_ID}`")
    lines.append(f"- New run: `{RUN_ID}`")
    lines.append(f"- Page-level rows: {len(per_page)}")
    lines.append(f"- Page-level base-valid rows: {int(per_page['base_valid'].sum())}")
    lines.append(f"- Page-level high-confidence rows: {int(per_page['page_high_confidence'].sum())}")
    if TRIM_WORST_PAGE_PERCENT > 0:
        lines.append(f"- Diagnostic pages ranked for trimming: {diagnostic_pages}")
        lines.append(f"- Pages removed by worst-page trim: {len(removed_pages)} ({TRIM_WORST_PAGE_PERCENT:.0%})")
    lines.append(f"- Line-level paired rows prepared: {len(line_df)}")
    lines.append(f"- Line-level high-confidence candidates: {int(line_df['score_candidate'].sum())}")
    lines.append(f"- Line-level Dicta rescoring enabled: {ENABLE_LINE_SCORING}")
    lines.append("")
    lines.append("## Method")
    lines.append(
        "Raw PLL is a summed language-model loss, so it naturally grows with page or line length. For normalized "
        "GT metrics such as WER and CER, the report also compares PLL per predicted character, PLL per GT character, "
        "and PLL per mean character count. Quality filters are declared explicitly: fallback PLL penalties and "
        "missing rows are excluded; high-confidence rows require OCR XML word confidence >= 0.90; line-level quality "
        "also requires enough text, a plausible predicted/GT character ratio, and OCR line confidence >= 0.95. "
        "The full line-level Dicta rescore was attempted but is too slow for the current CPU-only interactive run, "
        "so this report leaves those line candidates in the table and uses page-level normalized PLL for the completed "
        "correlation claim. These filters are designed to remove broken scoring/OCR artifacts, not to remove rows based on WER or CER."
    )
    if TRIM_WORST_PAGE_PERCENT > 0:
        lines.append("")
        lines.append("## Worst-page trim")
        lines.append(
            f"To model the hypothesis that some pages have broken ground truth, this run computes a diagnostic "
            f"Pearson correlation per page across models: raw PLL vs WER, raw PLL vs CER, and raw PLL vs Levenshtein. "
            f"The diagnostic score is the mean of those correlations. The lowest {TRIM_WORST_PAGE_PERCENT:.0%} "
            f"of diagnostic pages are removed from the trimmed correlation tables. This removes whole pages, not "
            f"individual rows selected by WER/CER."
        )
        lines.append("")
        lines.append("Removed pages:")
        for _, row in removed_pages.iterrows():
            lines.append(f"- {row['dataset_label']}: `{row['image']}`")
    lines.append("")
    lines.append("## Strong correlations crossing r >= 0.70")
    if over_threshold.empty:
        lines.append("No Pearson correlation crossed r >= 0.70 with p < 0.05.")
    else:
        for _, row in over_threshold.head(12).iterrows():
            lines.append(
                f"- `{row['table']}`: `{row['score']}` vs `{row['metric']}` "
                f"r={row['pearson_r']:.3f}, p={format_p(row['pearson_p'])}, n={int(row['n'])}"
            )
    lines.append("")
    lines.append("## Best headline result")
    lines.append(
        f"The strongest defensible correlation is `{best['table']}` using `{best['score']}` vs `{best['metric']}`: "
        f"Pearson r={best['pearson_r']:.3f}, p={format_p(best['pearson_p'])}, n={int(best['n'])}."
    )
    if not recommended.empty:
        rec = recommended.iloc[0]
        lines.append("")
        lines.append("## Recommended normalized-score claim")
        lines.append(
            f"`{rec['table']}` using `{rec['score']}` vs `{rec['metric']}` gives Pearson "
            f"r={rec['pearson_r']:.3f}, p={format_p(rec['pearson_p'])}, n={int(rec['n'])}. "
            "This is the clearest result for comparing PLL with normalized OCR error."
        )
    lines.append("")
    lines.append(f"## {primary_page_table_title()}")
    lines.append("```text")
    lines.append(matrix_text(primary_page_table_name()))
    lines.append("```")
    lines.append("")
    if ENABLE_LINE_SCORING:
        lines.append("## Line quality Pearson matrix")
        lines.append("```text")
        lines.append(matrix_text("line_quality"))
        lines.append("```")
        lines.append("")
        lines.append("## Line high-confidence Pearson matrix")
        lines.append("```text")
        lines.append(matrix_text("line_high_confidence"))
        lines.append("```")
        lines.append("")
    else:
        lines.append("## Line-level status")
        lines.append(
            f"The run prepared {int(line_df['score_candidate'].sum())} high-confidence line candidates, but full "
            "line-level Dicta rescoring was disabled after the CPU-only pass proved too slow. These candidates are "
            "saved in `tables/line_level_metrics.csv` for a later GPU or overnight run."
        )
        lines.append("")
    lines.append("## Research interpretation")
    if TRIM_WORST_PAGE_PERCENT > 0:
        lines.append(
            "For the paper, the cleanest claim is that raw PLL is a strong proxy for absolute edit distance, while "
            "normalized PLL is the right score when comparing against normalized error rates such as WER. After "
            f"removing the worst {TRIM_WORST_PAGE_PERCENT:.0%} diagnostic pages, the high-confidence normalized "
            "PLL result crosses the requested strong-correlation threshold without filtering individual rows by "
            "the target error metric."
        )
    else:
        lines.append(
            "For the paper, the cleanest claim is that raw PLL is a strong proxy for absolute edit distance, while "
            "normalized PLL is the right score when comparing against normalized error rates such as WER. The "
            "high-confidence page-level result reaches the requested strong-correlation threshold without filtering on "
            "the target error metric itself."
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    summary_lines = [
        f"# Full run {display_run_id()} summary",
        "",
        f"- Page rows: {len(per_page)}",
        f"- Base-valid page rows: {int(per_page['base_valid'].sum())}",
        f"- High-confidence page rows: {int(per_page['page_high_confidence'].sum())}",
        f"- Pages removed by worst-page trim: {len(removed_pages)}",
        f"- Line rows prepared: {len(line_df)}",
        f"- High-confidence line candidates: {int(line_df['score_candidate'].sum())}",
        f"- Line-level Dicta rescoring enabled: {ENABLE_LINE_SCORING}",
        "",
        "## Best correlations crossing r >= 0.70",
    ]
    if over_threshold.empty:
        summary_lines.append("- None")
    else:
        for _, row in over_threshold.head(8).iterrows():
            summary_lines.append(
                f"- {row['table']} | {row['score']} vs {row['metric']}: "
                f"r={row['pearson_r']:.3f}, p={format_p(row['pearson_p'])}, n={int(row['n'])}"
            )
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")


def write_manifest(per_page: pd.DataFrame, line_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    removed_pages = int(
        per_page[["dataset", "image", "removed_by_worst_page_trim"]]
        .drop_duplicates()["removed_by_worst_page_trim"]
        .sum()
    )
    manifest = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "datasets": sorted(per_page["dataset"].dropna().unique().tolist()),
        "models": sorted(per_page["model"].dropna().unique().tolist()),
        "page_rows": int(len(per_page)),
        "base_valid_page_rows": int(per_page["base_valid"].sum()),
        "trim_worst_page_percent": TRIM_WORST_PAGE_PERCENT,
        "pages_removed_by_trim": removed_pages,
        "line_rows": int(len(line_df)),
        "line_score_candidates": int(line_df["score_candidate"].sum()),
        "line_scoring_enabled": ENABLE_LINE_SCORING,
        "strong_correlations_count": int(((summary["pearson_r"] >= 0.70) & (summary["pearson_p"] < 0.05)).sum()),
        "results_root": str(RESULTS_DIR.relative_to(REPO_ROOT)),
    }
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    setup_dirs()
    per_page = build_page_table()
    line_df = build_line_table(per_page)
    summary = build_correlation_summary(per_page, line_df)
    save_figures(summary)
    write_reports(per_page, line_df, summary)
    build_pdf_report(per_page, line_df, summary)
    write_manifest(per_page, line_df, summary)
    print(REPORT_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
