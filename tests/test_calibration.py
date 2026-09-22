"""Calibration and threshold helpers."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from arc.calibration import (
    apply_platt,
    best_f1_threshold,
    binary_metrics,
    expected_calibration_error,
    fit_platt,
    invert_platt,
    threshold_for_recall,
)


def test_binary_metrics_counts_a_known_case():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["accuracy"] == pytest.approx(0.6)


def test_best_f1_threshold_separates_a_clean_split():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.7
    assert binary_metrics(y, p, t)["f1"] == 1.0


def test_threshold_for_recall_reaches_the_target():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.1
    p = np.where(y, rng.beta(4, 2, 2000), rng.beta(2, 4, 2000))
    t = threshold_for_recall(y, p, 0.9)
    assert binary_metrics(y, p, t)["recall"] >= 0.9
    # and it is the strictest such cut-off: nudging it up loses a case
    assert binary_metrics(y, p, t + 1e-9)["recall"] < 0.9


def test_platt_scaling_repairs_shifted_scores_and_keeps_the_ranking():
    rng = np.random.default_rng(1)
    true_p = rng.beta(1, 12, 20000)  # a rare finding
    y = rng.random(20000) < true_p
    z = np.log(true_p / (1 - true_p))
    shifted = 1 / (1 + np.exp(-(1.6 * z + 2.0)))  # overconfident and pushed upwards
    a, b = fit_platt(shifted, y)
    fixed = apply_platt(shifted, a, b)
    assert expected_calibration_error(y, fixed) < expected_calibration_error(y, shifted) / 3
    assert roc_auc_score(y, fixed) == pytest.approx(roc_auc_score(y, shifted))
    assert invert_platt(float(fixed[0]), a, b) == pytest.approx(float(shifted[0]), rel=1e-4)
