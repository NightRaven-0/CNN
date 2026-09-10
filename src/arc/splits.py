"""Group-aware, label-stratified splitting with explicit leakage assertions.

The single most common way a chest-radiograph study is quietly invalidated is
splitting on rows instead of on patients: the same patient appears in train and
test through a second study, the test metric inherits memorised anatomy, and the
reported AUROC is meaningless. Every split in this project goes through
:func:`make_splits`, and every training run calls :func:`assert_no_leakage`
before the first batch.

Stratification uses greedy iterative stratification (Sechidis, Tsoumakas &
Vlahavas, 2011), which is the standard treatment for multi-label data — plain
stratification on a single column leaves rare findings absent from a fold
entirely, and a fold with zero positives makes AUPRC undefined rather than bad.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Column used to group rows when the dataset carries no patient identifier.
#: VinDr-CXR ships one study per image with no cross-study patient linkage, so
#: image_id is the correct atomic unit *for that dataset only*. Any dataset with
#: multiple studies per patient MUST pass its own patient column.
DEFAULT_GROUP_COL = "image_id"


def make_splits(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    *,
    group_col: str = DEFAULT_GROUP_COL,
    fractions: dict[str, float] | None = None,
    min_positives: int = 1,
    seed: int = 1337,
) -> pd.DataFrame:
    """Assign every row a ``split`` label, grouping by ``group_col``.

    Args:
        df: One row per image. Must contain ``group_col`` and every label column.
        label_cols: Binary (0/1) multi-label columns to stratify on.
        group_col: Atomic unit that must not cross splits — patient id where one
            exists, image/study id otherwise.
        fractions: Split name to target proportion; must sum to 1.
        min_positives: Floor on positives per (split, label). Proportional
            allocation alone starves the rarest findings: with six positives and
            a 10% validation fold the expected count is 0.6, so validation gets
            none and that finding's AUPRC is undefined rather than merely bad.
            Seeding the floor first costs the common findings a handful of
            samples and keeps every finding measurable.
        seed: Controls tie-breaking order only; the algorithm is otherwise
            deterministic.

    Returns:
        A copy of ``df`` with an added ``split`` column.
    """
    fractions = fractions or {"train": 0.7, "val": 0.1, "test": 0.2}
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    missing = [c for c in [group_col, *label_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    if group_col == DEFAULT_GROUP_COL and df[group_col].is_unique:
        warnings.warn(
            f"Grouping on '{group_col}', which is unique per row — this assumes "
            "one study per patient with no cross-study linkage (true for "
            "VinDr-CXR). If this dataset has a patient identifier, pass it as "
            "group_col or the split will leak.",
            stacklevel=2,
        )

    # Collapse to one label vector per group: a group is positive for a finding
    # if any of its images is.
    grouped = df.groupby(group_col)[list(label_cols)].max()
    group_ids = grouped.index.to_numpy()
    y = grouped.to_numpy(dtype=np.int64)

    assignment = _iterative_stratification(
        y, fractions, seed=seed, min_positives=min_positives, label_names=list(label_cols)
    )

    lookup = dict(zip(group_ids, assignment, strict=True))
    out = df.copy()
    out["split"] = out[group_col].map(lookup)

    if out["split"].isna().any():
        raise RuntimeError("internal error: some rows were left unassigned")
    return out


def _iterative_stratification(
    y: np.ndarray,
    fractions: dict[str, float],
    *,
    seed: int,
    min_positives: int,
    label_names: Sequence[str],
) -> list[str]:
    """Greedy iterative stratification over a binary label matrix.

    Runs in two passes. The first seeds a floor of ``min_positives`` per
    (split, label), rarest label first, so no finding can be squeezed out of a
    small fold. The second is standard greedy iterative stratification: take the
    rarest label still in play and give each of its samples to whichever split
    is furthest below its quota for that label.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_labels = y.shape
    names = list(fractions)
    n_splits = len(names)
    ratios = np.array([fractions[k] for k in names], dtype=np.float64)

    # Remaining capacity, overall and per label.
    desired = ratios * n_samples
    desired_per_label = np.outer(ratios, y.sum(axis=0))  # (n_splits, n_labels)
    placed_per_label = np.zeros((n_splits, n_labels), dtype=np.int64)

    assignment: list[str | None] = [None] * n_samples
    unassigned = set(range(n_samples))

    def place(i: int, s: int) -> None:
        assignment[i] = names[s]
        desired[s] -= 1
        desired_per_label[s] -= y[i]
        placed_per_label[s] += y[i]
        unassigned.discard(i)

    # --- pass 1: guarantee the floor -------------------------------------
    totals = y.sum(axis=0)
    starved: list[str] = []
    for label in np.argsort(totals):
        if totals[label] == 0:
            continue
        for s in range(n_splits):
            while placed_per_label[s, label] < min_positives:
                pool = [i for i in unassigned if y[i, label] == 1]
                if not pool:
                    starved.append(f"{label_names[label]}->{names[s]}")
                    break
                # Spend the sample carrying the fewest other positives, so a
                # multi-label case is not burned to satisfy one quota.
                i = min(pool, key=lambda j: (int(y[j].sum()), rng.random()))
                place(i, s)

    if starved:
        warnings.warn(
            "Not enough positives to guarantee "
            f"min_positives={min_positives} for: {sorted(set(starved))}. "
            "Those findings cannot be scored in the affected split and must be "
            "excluded from the headline metric rather than averaged in.",
            stacklevel=3,
        )

    # --- pass 2: greedy iterative stratification -------------------------
    # Samples with no positive labels carry no stratification signal; hold them
    # back and distribute them at the end so they act as filler.
    negatives = {i for i in unassigned if y[i].sum() == 0}
    remaining = unassigned - negatives
    label_counts = y[sorted(remaining)].sum(axis=0) if remaining else np.zeros(n_labels)

    while remaining:
        candidates = np.where(label_counts > 0)[0]
        if candidates.size == 0:
            break
        label = candidates[np.argmin(label_counts[candidates])]

        members = [i for i in remaining if y[i, label] == 1]
        rng.shuffle(members)

        for i in members:
            # Prefer the split most short of this label; break ties on overall
            # remaining capacity, then at random.
            need = desired_per_label[:, label]
            best = np.flatnonzero(need == need.max())
            if best.size > 1:
                cap = desired[best]
                best = best[np.flatnonzero(cap == cap.max())]
            choice = int(best[0]) if best.size == 1 else int(rng.choice(best))

            place(i, choice)
            label_counts -= y[i]
            remaining.discard(i)

        label_counts[label] = 0

    for i in sorted(remaining | negatives):
        place(i, int(np.argmax(desired)))

    if any(a is None for a in assignment):
        raise RuntimeError("internal error: stratification left samples unassigned")
    return [a for a in assignment if a is not None]


def assert_no_leakage(df: pd.DataFrame, *, group_col: str = DEFAULT_GROUP_COL) -> None:
    """Raise if any group appears in more than one split.

    Call this at the top of every training run. It is cheap and it is the only
    thing standing between the project and an unpublishable result.
    """
    if "split" not in df.columns:
        raise KeyError("df has no 'split' column — call make_splits first")

    per_group = df.groupby(group_col)["split"].nunique()
    offenders = per_group[per_group > 1]
    if len(offenders):
        sample = offenders.index[:10].tolist()
        raise AssertionError(
            f"LEAKAGE: {len(offenders)} group(s) span multiple splits, e.g. {sample}"
        )


def summarise(df: pd.DataFrame, label_cols: Sequence[str]) -> pd.DataFrame:
    """Per-split positive counts and prevalence, for the thesis data table.

    A finding with zero positives in a split makes that split's AUPRC undefined;
    this table is how you catch it before a training run rather than after.
    """
    rows = []
    for split, part in df.groupby("split"):
        row: dict[str, object] = {"split": split, "n_images": len(part)}
        for col in label_cols:
            row[f"{col}__n"] = int(part[col].sum())
            row[f"{col}__prev"] = round(float(part[col].mean()), 5)
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")
