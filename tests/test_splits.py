"""Tests for the splitter.

These guard the two failure modes that silently invalidate a whole study:
a group crossing splits, and a rare finding landing entirely in one fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arc.splits import assert_no_leakage, make_splits, summarise

LABELS = [f"finding_{i}" for i in range(8)]


def synthetic(n_groups: int = 900, rows_per_group: int = 2, seed: int = 0) -> pd.DataFrame:
    """Multi-label frame with a deliberately steep prevalence gradient.

    Prevalences run from 40% down to ~0.7%, which is roughly the shape of
    VinDr-CXR and the regime where naive stratification breaks.
    """
    rng = np.random.default_rng(seed)
    prevalences = np.geomspace(0.4, 0.007, len(LABELS))
    group_labels = (rng.random((n_groups, len(LABELS))) < prevalences).astype(int)

    rows = []
    for g in range(n_groups):
        for r in range(rows_per_group):
            rows.append(
                {"patient_id": f"p{g:04d}", "image_id": f"p{g:04d}_{r}"}
                | dict(zip(LABELS, group_labels[g], strict=True))
            )
    return pd.DataFrame(rows)


def test_no_group_spans_two_splits():
    df = make_splits(synthetic(), LABELS, group_col="patient_id")
    assert_no_leakage(df, group_col="patient_id")


def test_leakage_assertion_actually_fires():
    df = make_splits(synthetic(), LABELS, group_col="patient_id")
    # Force one patient's second image into a different split.
    victim = df["patient_id"].iloc[0]
    idx = df.index[df["patient_id"] == victim][-1]
    df.loc[idx, "split"] = "test" if df.loc[idx, "split"] != "test" else "train"

    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(df, group_col="patient_id")


def test_every_label_has_positives_in_every_split():
    df = make_splits(synthetic(), LABELS, group_col="patient_id")
    counts = summarise(df, LABELS)
    for label in LABELS:
        zero = counts.index[counts[f"{label}__n"] == 0].tolist()
        assert not zero, f"{label} has no positives in split(s) {zero} — AUPRC undefined"


def test_fractions_are_approximately_respected():
    fractions = {"train": 0.7, "val": 0.1, "test": 0.2}
    df = make_splits(synthetic(), LABELS, group_col="patient_id", fractions=fractions)
    observed = df["split"].value_counts(normalize=True)
    for name, target in fractions.items():
        assert abs(observed[name] - target) < 0.03, f"{name}: {observed[name]:.3f} vs {target}"


def test_prevalence_is_preserved_across_splits():
    df = make_splits(synthetic(), LABELS, group_col="patient_id")
    overall = df[LABELS].mean()
    for _, part in df.groupby("split"):
        drift = (part[LABELS].mean() - overall).abs()
        # Rare labels drift more in absolute terms is fine; what matters is that
        # no split is wildly off on a common finding.
        assert drift.max() < 0.06, f"prevalence drift {drift.max():.3f}"


def test_deterministic_for_a_given_seed():
    a = make_splits(synthetic(), LABELS, group_col="patient_id", seed=7)
    b = make_splits(synthetic(), LABELS, group_col="patient_id", seed=7)
    pd.testing.assert_series_equal(a["split"], b["split"])


def test_warns_when_grouping_on_a_unique_column():
    df = synthetic(n_groups=200, rows_per_group=1)
    with pytest.warns(UserWarning, match="unique per row"):
        make_splits(df, LABELS)  # defaults to image_id


def test_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        make_splits(synthetic(), LABELS, group_col="patient_id", fractions={"a": 0.5, "b": 0.4})
