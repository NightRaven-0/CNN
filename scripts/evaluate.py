"""Per-class AUROC and AUPRC on the official test split, with bootstrap intervals.

AUROC alone flatters a rare finding: Hernia is 0.2% of images, so a model can
score well on it while being useless in practice. AUPRC is reported beside it,
and every number carries a confidence interval, because the difference between
two papers on this dataset is often smaller than the interval on either.

The published comparison numbers below are all on the same official split. The
widely quoted CheXNet figure is deliberately absent: it was measured on its own
random split and does not belong in this table.

Usage:
    python scripts/evaluate.py --run runs/densenet121_512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from arc.data import IMAGE_COL, LABELS, load_metadata

#: Mean AUROC on the official ChestX-ray14 test split, from the literature.
PUBLISHED = {
    "Wang et al. 2017 (ResNet-50)": 0.745,
    "Yao et al. 2017": 0.761,
    "Baltruschat et al. 2019 (ResNet-38 + metadata)": 0.806,
    "arXiv:2404.18933 (DenseNet-121)": 0.812,
    "arXiv:2404.18933 (DenseNet-121 + their method)": 0.824,
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def bootstrap_interval(
    y: np.ndarray, p: np.ndarray, metric, *, resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile interval for a metric, resampling images with replacement."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            scores.append(metric(y[idx], p[idx]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the classifier on the official split.")
    parser.add_argument("--run", type=Path, required=True, help="a training output directory")
    parser.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    predictions = pd.read_csv(args.run / "test_predictions.csv")
    meta = load_metadata(args.data)
    truth = meta.loc[meta["split"] == "test", [IMAGE_COL, *LABELS]]

    merged = predictions.merge(truth, on=IMAGE_COL, suffixes=("_pred", "_true"))
    if len(merged) != len(predictions):
        log(f"{len(predictions) - len(merged)} predictions had no matching label row")
        return 1
    log(f"scoring {len(merged)} test images")

    rows = []
    for label in LABELS:
        y = merged[f"{label}_true"].to_numpy(dtype=np.int8)
        p = merged[f"{label}_pred"].to_numpy(dtype=np.float64)
        if not 0 < y.sum() < len(y):
            rows.append({"finding": label, "positives": int(y.sum())})
            continue
        auroc_lo, auroc_hi = bootstrap_interval(
            y, p, roc_auc_score, resamples=args.resamples, seed=args.seed
        )
        auprc_lo, auprc_hi = bootstrap_interval(
            y, p, average_precision_score, resamples=args.resamples, seed=args.seed
        )
        rows.append(
            {
                "finding": label,
                "positives": int(y.sum()),
                "prevalence": round(float(y.mean()), 4),
                "auroc": round(float(roc_auc_score(y, p)), 4),
                "auroc_lo": round(auroc_lo, 4),
                "auroc_hi": round(auroc_hi, 4),
                "auprc": round(float(average_precision_score(y, p)), 4),
                "auprc_lo": round(auprc_lo, 4),
                "auprc_hi": round(auprc_hi, 4),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.run / "test_metrics_per_class.csv", index=False)

    mean_auroc = float(table["auroc"].mean(skipna=True))
    summary = {
        "mean_auroc": round(mean_auroc, 4),
        "mean_auprc": round(float(table["auprc"].mean(skipna=True)), 4),
        "n_test_images": len(merged),
        "published_official_split": PUBLISHED,
    }
    (args.run / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    print(table.to_string(index=False))
    log(f"mean AUROC {mean_auroc:.4f}")
    # Print this run in its place among the published numbers, so the ranking
    # reads at a glance instead of marking every row we happen to beat.
    ranked = [(value, f"published   {name}") for name, value in PUBLISHED.items()]
    ranked.append((mean_auroc, ">>> THIS RUN (DenseNet-121, official split)"))
    for value, label in sorted(ranked):
        log(f"  {value:.4f}  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
