"""Calibrate the probabilities and choose a decision threshold for each finding.

The model ranks images well, which is what AUROC measures, but its scores are
not probabilities. Training weighted positive cases up, so every finding's
scores are pushed upwards by a different amount, and a score of 0.5 means
nothing in particular. This script repairs that after training, using only
validation data:

1. it runs the saved model on the validation images, never on the test set;
2. it fits Platt scaling for each finding on validation. The map is monotone,
   so AUROC does not change;
3. it picks two cut-offs for each finding on validation: the one with the best
   F1, and the highest one that still catches 90% of cases;
4. it applies both, unchanged, to the test predictions and reports precision,
   recall, specificity, F1 and accuracy, next to the accuracy of always saying
   "no", which is what makes accuracy misleading for rare findings.

Writes to the run directory: val_predictions.csv, calibration.json,
test_threshold_metrics.csv and calibration_metrics.csv.

Usage:
    python scripts/calibrate.py --run runs/densenet121_512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from arc.calibration import (
    apply_platt,
    best_f1_threshold,
    binary_metrics,
    expected_calibration_error,
    fit_platt,
    invert_platt,
    threshold_for_recall,
)
from arc.data import IMAGE_COL, LABELS, ChestXray14, build_transforms, load_metadata
from arc.model import DenseNet121Classifier


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


@torch.no_grad()
def predict(state: dict, frame: pd.DataFrame, images: Path, size: int, batch_size: int,
            workers: int) -> pd.DataFrame:
    """Scores for every image in ``frame``, computed exactly as training scored the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    model = DenseNet121Classifier(len(LABELS), pretrained=False)
    model.load_state_dict(state["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()

    ds = ChestXray14(frame, images, build_transforms(size, train=False))
    # A one-off pass, so no persistent workers: they outlive the loop and were
    # what exhausted memory after the first training run.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=False)
    probs = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu())
    out = pd.DataFrame(torch.cat(probs).numpy(), columns=LABELS)
    out.insert(0, IMAGE_COL, ds.frame[IMAGE_COL].to_numpy())
    return out.set_index(IMAGE_COL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate scores and choose a threshold per finding.")
    parser.add_argument("--run", type=Path, default=Path("runs/densenet121_512"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--images", type=Path, default=Path("data/processed/nih512"))
    parser.add_argument("--recall-target", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fresh", action="store_true", help="recompute the validation predictions")
    args = parser.parse_args()

    state = torch.load(args.run / "best.pt", map_location="cpu")
    seed, size = int(state["args"]["seed"]), int(state["args"]["size"])
    meta = load_metadata(args.data, seed=seed).set_index(IMAGE_COL)

    # Validation must be the exact set this run was tuned on: the saved split,
    # which the seed has to reproduce.
    splits = pd.read_csv(args.run / "splits.csv")
    val_ids = splits.loc[splits["split"] == "val", IMAGE_COL]
    if set(val_ids) != set(meta.index[meta["split"] == "val"]):
        log("the saved validation split does not match the one this seed produces; stopping")
        return 1

    val_file = args.run / "val_predictions.csv"
    if val_file.exists() and not args.fresh:
        val_pred = pd.read_csv(val_file).set_index(IMAGE_COL)
        log(f"reusing {val_file}")
    else:
        started = time.perf_counter()
        val_pred = predict(state, meta.loc[val_ids].reset_index(), args.images, size,
                           args.batch_size, args.workers)
        val_pred.to_csv(val_file)
        log(f"scored {len(val_pred)} validation images in {time.perf_counter() - started:.0f}s")

    test_pred = pd.read_csv(args.run / "test_predictions.csv").set_index(IMAGE_COL)
    y_val, y_test = meta.loc[val_pred.index, LABELS], meta.loc[test_pred.index, LABELS]

    recall_name = f"catch {args.recall_target:.0%} of cases"
    params, decision_rows, calibration_rows = {}, [], []
    for finding in LABELS:
        yv, sv = y_val[finding].to_numpy(), val_pred[finding].to_numpy()
        yt, st = y_test[finding].to_numpy(), test_pred[finding].to_numpy()
        a, b = fit_platt(sv, yv)
        cv, ct = apply_platt(sv, a, b), apply_platt(st, a, b)
        cut_f1 = best_f1_threshold(yv, cv)
        cut_recall = threshold_for_recall(yv, cv, args.recall_target)
        params[finding] = {
            "platt_a": a, "platt_b": b, "validation_positives": int(yv.sum()),
            "threshold_best_f1": cut_f1, "raw_threshold_best_f1": invert_platt(cut_f1, a, b),
            "threshold_recall": cut_recall, "raw_threshold_recall": invert_platt(cut_recall, a, b),
        }
        for name, cut in (("best F1", cut_f1), (recall_name, cut_recall)):
            decision_rows.append({
                "finding": finding, "operating_point": name, "threshold": cut,
                **binary_metrics(yt, ct, cut),
                "always_no_accuracy": 1 - yt.mean(), "validation_positives": int(yv.sum()),
            })
        calibration_rows.append({
            "finding": finding, "prevalence": yt.mean(),
            "mean_score_raw": st.mean(), "mean_score_calibrated": ct.mean(),
            "ece_raw": expected_calibration_error(yt, st), "ece_calibrated": expected_calibration_error(yt, ct),
            "brier_raw": float(np.mean((st - yt) ** 2)), "brier_calibrated": float(np.mean((ct - yt) ** 2)),
            "auroc_raw": roc_auc_score(yt, st), "auroc_calibrated": roc_auc_score(yt, ct),
        })

    (args.run / "calibration.json").write_text(json.dumps(
        {"seed": seed, "recall_target": args.recall_target, "findings": params}, indent=2))
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(args.run / "test_threshold_metrics.csv", index=False)
    calibration = pd.DataFrame(calibration_rows).set_index("finding")
    calibration.to_csv(args.run / "calibration_metrics.csv")

    pd.set_option("display.width", 200)
    shown = ["threshold", "precision", "recall", "specificity", "f1", "accuracy", "always_no_accuracy"]
    for name in ("best F1", recall_name):
        table = decisions[decisions["operating_point"] == name].set_index("finding")[shown]
        log(f"test set, cut-offs chosen on validation for '{name}':\n{table.round(3).to_string()}")
        log(f"mean of 14: {table.mean().round(3).to_dict()}")
    log(f"calibration on the test set:\n{calibration.round(4).to_string()}")
    log(f"mean ECE {calibration['ece_raw'].mean():.4f} before, {calibration['ece_calibrated'].mean():.4f} after; "
        f"largest AUROC change {np.abs(calibration['auroc_raw'] - calibration['auroc_calibrated']).max():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
