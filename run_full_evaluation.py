#!/home/userm/Programs/anaconda3/bin/python
from __future__ import annotations

import json
import os
import re
import string
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jiwer
from kraken.lib.xml import parse_alto

import app as chooser


REPO_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = REPO_ROOT / "datasets"
RUN_ID = "full_run_20260613"
RESULTS_ROOT = REPO_ROOT / "analysis_results" / RUN_ID
PREDICTIONS_ROOT = RESULTS_ROOT / "predictions"
FIGURES_ROOT = RESULTS_ROOT / "figures"
TABLES_ROOT = RESULTS_ROOT / "tables"
LOG_PATH = RESULTS_ROOT / "run.log"

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
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


def allowed_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES


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


def lines_to_text(lines) -> str:
    return "\n".join(line.strip() for line in lines if line and line.strip())


def alto_text(path: Path) -> str:
    val = parse_alto(str(path))
    lines = [line.get("text", "") for line in val.get("lines", [])]
    return lines_to_text(lines)


def load_or_create_gt_text(dataset_dir: Path, image_stem: str) -> tuple[str, Path, str]:
    gt_dir = dataset_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    gt_txt = gt_dir / f"{image_stem}.txt"
    if gt_txt.exists():
        return gt_txt.read_text(encoding="utf-8", errors="ignore"), gt_txt, "gt"

    gt_xml = dataset_dir / "gt_alto" / f"{image_stem}.xml"
    if gt_xml.exists():
        text = alto_text(gt_xml)
        gt_txt.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")
        return text, gt_txt, "gt_alto"

    raise FileNotFoundError(f"No GT text or ALTO XML found for {dataset_dir.name}/{image_stem}")


def image_files(dataset_dir: Path) -> list[Path]:
    images_dir = dataset_dir / "images"
    return sorted([p for p in images_dir.iterdir() if p.is_file() and allowed_image(p)])


def compute_metrics(gt_text: str, pred_text: str) -> dict[str, float]:
    gt_clean = normalize_for_compare(gt_text)
    pred_clean = normalize_for_compare(pred_text)
    return {
        "wer": jiwer.wer(gt_text, pred_text, reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM),
        "cer": jiwer.cer(gt_text, pred_text, reference_transform=CER_TRANSFORM, hypothesis_transform=CER_TRANSFORM),
        "levenshtein": float(levenshtein_distance(gt_clean, pred_clean)),
    }


def setup_results_dirs() -> None:
    for path in [RESULTS_ROOT, PREDICTIONS_ROOT, FIGURES_ROOT, TABLES_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def load_models() -> dict[str, object]:
    models = chooser.discover_recognition_models()
    chooser.models_load_dict = {}
    for name, path in models.items():
        chooser.models_load_dict[name] = chooser.models.load_any(path, device=chooser.TORCH_DEVICE)
    return models


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def run_dataset(dataset_dir: Path, models: dict[str, str], log_handle) -> list[dict]:
    images = image_files(dataset_dir)
    if not images:
        return []

    dataset_pred_dir = PREDICTIONS_ROOT / dataset_dir.name
    dataset_pred_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    chooser.models_scores = {}
    chooser.segmentations_dict = {}
    chooser.xml_dict = {}

    print(f"Dataset: {pretty_dataset_label(dataset_dir.name)} | images: {len(images)}", file=log_handle, flush=True)

    for image_path in images:
        image_name = image_path.name
        image_stem = image_path.stem
        gt_text, gt_path, gt_source = load_or_create_gt_text(dataset_dir, image_stem)

        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            chooser.read_and_segment_image(str(dataset_dir / "images"), image_name, None, chooser.TORCH_DEVICE)

        segmentation = chooser.segmentations_dict[image_name]
        bw_im = segmentation["bw_im"]
        baseline_seg = segmentation["baseline_seg"]

        for model_name in models:
            try:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    pll_score = chooser.read_txt_and_score(baseline_seg, bw_im, model_name, image_name)
                xml_key = f"{model_name}__{image_stem}"
                pred_xml = chooser.xml_dict.get(xml_key, "")
                pred_path = dataset_pred_dir / f"{xml_key}.xml"
                pred_path.write_text(pred_xml, encoding="utf-8")
                pred_text = alto_text(pred_path)
                metrics = compute_metrics(gt_text, pred_text)
                rows.append({
                    "dataset": dataset_dir.name,
                    "dataset_label": pretty_dataset_label(dataset_dir.name),
                    "image": image_name,
                    "image_stem": image_stem,
                    "model": model_name,
                    "model_label": pretty_model_label(model_name),
                    "pll_score": float(pll_score),
                    "wer": metrics["wer"],
                    "cer": metrics["cer"],
                    "levenshtein": metrics["levenshtein"],
                    "gt_source": gt_source,
                    "gt_file": str(gt_path.relative_to(REPO_ROOT)),
                    "prediction_file": str(pred_path.relative_to(REPO_ROOT)),
                    "gt_chars": len(normalize_for_compare(gt_text)),
                    "pred_chars": len(normalize_for_compare(pred_text)),
                })
                chooser.xml_dict.pop(xml_key, None)
            except Exception as exc:
                print(f"  model {model_name} failed on {image_name}: {exc}", file=log_handle, flush=True)
                rows.append({
                    "dataset": dataset_dir.name,
                    "dataset_label": pretty_dataset_label(dataset_dir.name),
                    "image": image_name,
                    "image_stem": image_stem,
                    "model": model_name,
                    "model_label": pretty_model_label(model_name),
                    "pll_score": float("nan"),
                    "wer": float("nan"),
                    "cer": float("nan"),
                    "levenshtein": float("nan"),
                    "gt_source": gt_source,
                    "gt_file": str(gt_path.relative_to(REPO_ROOT)),
                    "prediction_file": "",
                    "gt_chars": len(normalize_for_compare(gt_text)),
                    "pred_chars": 0,
                })

        chooser.segmentations_dict.pop(image_name, None)

    print(f"  finished {dataset_dir.name}", file=log_handle, flush=True)
    return rows


def write_tables(per_image: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_image_path = TABLES_ROOT / "per_image_metrics.csv"
    per_image.to_csv(per_image_path, index=False)

    dataset_model = (
        per_image
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
    dataset_model.to_csv(TABLES_ROOT / "dataset_model_summary.csv", index=False)

    model_summary = (
        per_image
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
    model_summary.to_csv(TABLES_ROOT / "model_summary.csv", index=False)

    best_by_dataset = []
    for dataset, group in dataset_model.groupby("dataset", sort=True):
        row_pll = group.loc[group["mean_pll"].idxmin()]
        row_wer = group.loc[group["mean_wer"].idxmin()]
        row_cer = group.loc[group["mean_cer"].idxmin()]
        row_lev = group.loc[group["mean_levenshtein"].idxmin()]
        best_by_dataset.append({
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
    best_by_dataset_df = pd.DataFrame(best_by_dataset)
    best_by_dataset_df.to_csv(TABLES_ROOT / "dataset_best_models.csv", index=False)

    return {
        "per_image": per_image,
        "dataset_model": dataset_model,
        "model_summary": model_summary,
        "best_by_dataset": best_by_dataset_df,
    }


def plot_correlation(per_image: pd.DataFrame) -> pd.DataFrame:
    corr_df = per_image[["pll_score", "wer", "cer", "levenshtein"]].corr(method="pearson")
    corr_df.to_csv(TABLES_ROOT / "correlation_matrix_pearson.csv")

    spearman_df = per_image[["pll_score", "wer", "cer", "levenshtein"]].corr(method="spearman")
    spearman_df.to_csv(TABLES_ROOT / "correlation_matrix_spearman.csv")

    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_df, annot=True, cmap="viridis", vmin=-1, vmax=1, center=0, fmt=".2f")
    plt.title("Pearson correlation between PLL and GT-based metrics")
    save_figure(FIGURES_ROOT / "correlation_heatmap_pearson.png")

    plt.figure(figsize=(7, 5))
    sns.heatmap(spearman_df, annot=True, cmap="viridis", vmin=-1, vmax=1, center=0, fmt=".2f")
    plt.title("Spearman correlation between PLL and GT-based metrics")
    save_figure(FIGURES_ROOT / "correlation_heatmap_spearman.png")

    return corr_df


def plot_model_summary(model_summary: pd.DataFrame) -> None:
    metrics = [
        ("mean_pll", "Mean PLL"),
        ("mean_wer", "Mean WER"),
        ("mean_cer", "Mean CER"),
        ("mean_levenshtein", "Mean Levenshtein"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()
    order = model_summary.sort_values("mean_pll")["model_label"]
    for ax, (col, title) in zip(axes, metrics):
        sns.barplot(data=model_summary, x="model_label", y=col, order=order, ax=ax, color="#3b82f6")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Model comparison across all datasets", y=1.02, fontsize=14)
    save_figure(FIGURES_ROOT / "model_summary_by_metric.png")


def plot_dataset_heatmap(dataset_model: pd.DataFrame) -> None:
    pivot = dataset_model.pivot(index="dataset_label", columns="model_label", values="mean_pll")
    pivot = pivot.sort_index(axis=0)
    model_order = (
        dataset_model.groupby("model_label", as_index=False)["mean_pll"]
        .mean()
        .sort_values("mean_pll")["model_label"]
        .tolist()
    )
    pivot = pivot[model_order]

    plt.figure(figsize=(max(12, len(model_order) * 1.2), max(10, len(pivot.index) * 0.45)))
    sns.heatmap(pivot, cmap="mako_r", linewidths=0.2, linecolor="white")
    plt.title("Mean PLL score by handwriting and model")
    plt.xlabel("Model")
    plt.ylabel("Handwriting")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    save_figure(FIGURES_ROOT / "dataset_model_pll_heatmap.png")


def plot_winner_counts(best_by_dataset: pd.DataFrame) -> None:
    winner_counts = best_by_dataset["best_pll_model_label"].value_counts().reset_index()
    winner_counts.columns = ["model_label", "wins"]
    winner_counts = winner_counts.sort_values("wins", ascending=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=winner_counts, x="wins", y="model_label", color="#10b981")
    plt.title("Number of handwriting sets won by each model")
    plt.xlabel("Dataset wins on mean PLL")
    plt.ylabel("Model")
    save_figure(FIGURES_ROOT / "pll_winner_counts.png")


def write_summary(per_image: pd.DataFrame, tables: dict[str, pd.DataFrame], corr_df: pd.DataFrame) -> None:
    model_summary = tables["model_summary"]
    best_by_dataset = tables["best_by_dataset"]

    summary_path = RESULTS_ROOT / "summary.md"
    lines = []
    lines.append("# Full evaluation summary")
    lines.append("")
    lines.append(f"- Datasets evaluated: {per_image['dataset'].nunique()}")
    lines.append(f"- Images evaluated: {per_image[['dataset', 'image']].drop_duplicates().shape[0]}")
    lines.append(f"- Recognition models evaluated: {per_image['model'].nunique()}")
    lines.append(f"- Per-sample comparisons: {len(per_image)}")
    lines.append("")
    lines.append("## Model ranking by mean PLL")
    for _, row in model_summary.sort_values("mean_pll").iterrows():
        lines.append(
            f"- {row['model_label']}: PLL={row['mean_pll']:.3f}, "
            f"WER={row['mean_wer']:.3f}, CER={row['mean_cer']:.3f}, Lev={row['mean_levenshtein']:.3f}"
        )
    lines.append("")
    lines.append("## Best model by handwriting")
    for _, row in best_by_dataset.iterrows():
        lines.append(
            f"- {row['dataset_label']}: PLL={row['best_pll_model_label']}, "
            f"WER={row['best_wer_model_label']}, CER={row['best_cer_model_label']}, "
            f"Lev={row['best_lev_model_label']}"
        )
    lines.append("")
    lines.append("## Pearson correlations")
    lines.append(corr_df.round(3).to_string())
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_results_dirs()

    with open(LOG_PATH, "w", encoding="utf-8") as log_handle:
        print("Loading models...", file=log_handle, flush=True)
        models = load_models()
        print(f"Loaded recognition models: {', '.join(models.keys())}", file=log_handle, flush=True)

        all_rows = []
        for dataset_dir in sorted([p for p in DATASETS_ROOT.iterdir() if p.is_dir() and (p / "images").exists()]):
            dataset_rows = run_dataset(dataset_dir, models, log_handle)
            all_rows.extend(dataset_rows)
            pd.DataFrame(all_rows).to_csv(TABLES_ROOT / "per_image_metrics.csv", index=False)

    per_image = pd.DataFrame(all_rows)
    tables = write_tables(per_image)
    corr_df = plot_correlation(per_image)
    plot_model_summary(tables["model_summary"])
    plot_dataset_heatmap(tables["dataset_model"])
    plot_winner_counts(tables["best_by_dataset"])
    write_summary(per_image, tables, corr_df)

    manifest = {
        "run_id": RUN_ID,
        "datasets": sorted(per_image["dataset"].unique().tolist()),
        "models": sorted(per_image["model"].unique().tolist()),
        "rows": int(len(per_image)),
        "results_root": str(RESULTS_ROOT.relative_to(REPO_ROOT)),
    }
    (RESULTS_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
