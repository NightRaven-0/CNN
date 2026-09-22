"""Turning ranking scores into usable probabilities and yes-or-no decisions.

AUROC only asks whether the model ranks cases above non-cases, so a model can
score well on it while its outputs are not probabilities at all. Ours are not:
the loss weighted positive cases up, which pushes every finding's scores
upwards by a different amount. The functions here repair that after training,
on validation data, and turn the result into decisions that can be scored with
precision and recall.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit ``calibrated = sigmoid(a * logit(score) + b)`` on held-out data.

    Platt scaling with a slope and an offset per finding. The offset undoes the
    shift the positive weighting introduced and the slope fixes over- or
    under-confidence. With ``a > 0`` the map is monotone, so AUROC is unchanged.
    """
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logit(scores).reshape(-1, 1), np.asarray(labels).astype(int))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * logit(scores) + b)))


def invert_platt(calibrated: float, a: float, b: float) -> float:
    """The raw score that maps to a given calibrated probability."""
    return float(1.0 / (1.0 + np.exp(-((logit(np.array([calibrated]))[0] - b) / a))))


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """The cut-off with the highest F1 on the given data (a case is flagged at score >= cut-off)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # The last precision and recall pair has no threshold: it is the empty prediction.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    """The highest cut-off that still catches at least ``target`` of the positive cases."""
    positives = np.sort(np.asarray(scores)[np.asarray(labels).astype(bool)])[::-1]
    if len(positives) == 0:
        raise ValueError("no positive cases to set a recall threshold on")
    k = int(np.ceil(target * len(positives)))
    return float(positives[k - 1])


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Counts and rates for the decision ``score >= threshold``."""
    y = np.asarray(labels).astype(bool)
    flagged = np.asarray(scores) >= threshold
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * precision * recall / (precision + recall) if tp else 0.0,
        "accuracy": (tp + tn) / len(y),
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 15) -> float:
    """Average gap between predicted probability and observed frequency, over equal-width bins."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    which = np.clip(np.digitize(p, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            total += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(total)
