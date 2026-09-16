"""Compare training runs that differ only by their random seed.

One run gives one number and no idea how much of it is luck. These runs re-draw
the weight initialisation, the shuffling, the augmentation and the validation
split, so the spread between them covers repeating the whole procedure. The test
set is fixed by NIH's official list, so the comparison is fair.

The figure is the answer to "why these weights": each run keeps the weights from
its own best validation epoch, marked on its curve, and the test score is
computed once with those weights and never used to choose anything.

Writes runs/seed_comparison.csv and figures/9_seed_comparison.png.

Usage:
    python scripts/compare_runs.py runs/densenet121_512 runs/densenet121_512_seed2024 runs/densenet121_512_seed7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from arc.figstyle import AXIS, INK, INK_2, MUTED, SURFACE, apply_style

#: Categorical slots 1 to 3, the three that stay distinct under every colour
#: vision check even when all pairs are on screen together.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

#: For context on the test panel: the closest published DenseNet-121 on this split.
REFERENCE = ("arXiv:2404.18933, DenseNet-121", 0.812)

DEFAULT_RUNS = [
    Path("runs/densenet121_512"),
    Path("runs/densenet121_512_seed2024"),
    Path("runs/densenet121_512_seed7"),
]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_run(path: Path) -> dict | None:
    """Collect one run's history and scores, or None if it has not finished."""
    history_file, checkpoint = path / "history.csv", path / "best.pt"
    if not (history_file.exists() and checkpoint.exists()):
        log(f"{path}: no finished run here yet, skipping")
        return None

    history = pd.read_csv(history_file)
    saved = torch.load(checkpoint, map_location="cpu")
    seed = saved.get("args", {}).get("seed", "unknown")

    test_mean = None
    if (path / "test_metrics.json").exists():
        test_mean = json.loads((path / "test_metrics.json").read_text()).get("mean_auroc")
    elif (path / "metrics.json").exists():
        test_mean = json.loads((path / "metrics.json").read_text()).get("test_mean_auroc")

    best = history.loc[history["val_mean_auroc"].idxmax()]
    return {
        "run": path.name,
        "seed": seed,
        "epochs_trained": int(history["epoch"].max()),
        "best_epoch": int(best["epoch"]),
        "best_val_auroc": round(float(best["val_mean_auroc"]), 4),
        "test_mean_auroc": None if test_mean is None else round(float(test_mean), 4),
        "minutes": round(float(history["minutes"].sum()), 1),
        "history": history,
    }


def figure(runs: list[dict], out: Path) -> None:
    fig, (curves, scores) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained",
                                         width_ratios=[1.35, 1])

    for run, colour in zip(runs, SERIES, strict=False):
        history = run["history"]
        best = history.loc[history["val_mean_auroc"].idxmax()]
        # The best epoch goes in the legend as well as on the curve: two runs can
        # peak at nearly the same score, and then one marker hides the other.
        curves.plot(history["epoch"], history["val_mean_auroc"], color=colour, lw=2,
                    solid_capstyle="round",
                    label=f"seed {run['seed']}, best epoch {int(best['epoch'])}")
        curves.scatter(best["epoch"], best["val_mean_auroc"], s=70, color=colour,
                       edgecolor=SURFACE, linewidth=2, zorder=3)
    curves.set_xlabel("epoch")
    curves.set_ylabel("validation mean AUROC")
    curves.set_title("Each run keeps its best validation epoch")
    curves.set_xticks(sorted({int(e) for r in runs for e in r["history"]["epoch"]}))
    curves.grid(axis="y")
    curves.set_axisbelow(True)
    curves.legend(loc="lower right", fontsize=9)

    finished = [r for r in runs if r["test_mean_auroc"] is not None]
    ys = np.arange(len(finished))
    scores.axvline(REFERENCE[1], color=AXIS, lw=1, zorder=1)
    scores.annotate(REFERENCE[0], (REFERENCE[1], len(finished) - 0.35), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", color=MUTED, fontsize=8)
    for y, (run, colour) in enumerate(zip(finished, SERIES, strict=False)):
        scores.scatter(run["test_mean_auroc"], y, s=80, color=colour, edgecolor=SURFACE,
                       linewidth=2, zorder=3)
        scores.annotate(f"{run['test_mean_auroc']:.4f}", (run["test_mean_auroc"], y), xytext=(10, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
    scores.set_yticks(ys, [f"seed {r['seed']}" for r in finished])
    scores.tick_params(axis="y", length=0)
    scores.spines["left"].set_visible(False)
    scores.set_ylim(-0.6, len(finished) - 0.2)
    scores.grid(axis="x")
    scores.set_axisbelow(True)
    spread_values = [r["test_mean_auroc"] for r in finished]
    # Wide enough to take in the neighbouring published figures. On a tight axis a
    # spread of a few thousandths looks like a real difference, which it is not.
    scores.set_xlim(min(0.805, min(spread_values) - 0.002), max(0.828, max(spread_values) + 0.002))
    scores.set_xlabel("test mean AUROC, official split")
    scores.set_title("Test score, computed once per run")

    values = [r["test_mean_auroc"] for r in finished]
    spread = (f"The three runs land within {max(values) - min(values):.4f} of each other, "
              f"mean {np.mean(values):.4f}. " if len(values) > 1 else "")
    fig.text(0.01, -0.02,
             "Seeds change the weight initialisation, the shuffling, the augmentation and which 10% of the "
             "training data is held out for validation. The test set is fixed by NIH's official list and is "
             f"never used to pick an epoch. {spread}A difference smaller than this spread is not a result.",
             ha="left", va="top", color=INK_2, fontsize=8.5, wrap=True)
    fig.savefig(out / "9_seed_comparison.png")
    plt.close(fig)
    log(f"wrote {out / '9_seed_comparison.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare runs that differ only by seed.")
    parser.add_argument("runs", nargs="*", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--table", type=Path, default=Path("runs/seed_comparison.csv"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    runs = [run for run in (read_run(path) for path in args.runs) if run is not None]
    if not runs:
        log("no finished runs found")
        return 1

    table = pd.DataFrame([{k: v for k, v in run.items() if k != "history"} for run in runs])
    table.to_csv(args.table, index=False)
    print(table.to_string(index=False))

    scored = table["test_mean_auroc"].dropna()
    if len(scored) > 1:
        log(f"test mean AUROC across {len(scored)} runs: mean {scored.mean():.4f}, "
            f"range {scored.min():.4f} to {scored.max():.4f}, spread {scored.max() - scored.min():.4f}")
    if len(runs) > 1:
        figure(runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
