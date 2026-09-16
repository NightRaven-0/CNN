"""Check how far box-overlap scores can measure nodule localisation at all.

A nodule's box is about one cell of the 16 by 16 explanation grid, so the box
drawn from any map is many times larger than the nodule, and IoU and T(IoBB)
are capped however well the map is placed. Scoring nodules by overlap therefore
mixes two things: whether the explanation is wrong, and whether the measure can
see it being right.

Writes two tables to runs/xai/:

* box_sizes.csv, radiologist box sizes by finding against one grid cell
* nodule_resolution.csv, for each method on every nodule case: how much larger
  the derived box is than the nodule, how many cases could reach T(IoBB) 0.5 by
  any placement, and how often the derived box at least contains the nodule's
  centre. The random map is the baseline for that last number, since a box of
  the same size dropped at random also contains the centre some of the time.

Usage:
    python scripts/check_nodule_resolution.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from arc.data import IMAGE_COL, LABELS, build_transforms, load_boxes
from arc.model import DenseNet121Classifier
from arc.xai import builtin_cam, cam_explain
from arc.xai_metrics import iobb, map_to_box

#: Boxes are in 1024 px space; DenseNet-121 at 512 px input gives a 16 by 16 grid.
GRID_CELL = 1024 // 16
METHODS = ("cam", "gradcam", "gradcam++", "eigencam", "random")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the box-overlap ceiling for nodules.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/densenet121_512/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--images", type=Path, default=Path("data/processed/nih512"))
    parser.add_argument("--out", type=Path, default=Path("runs/xai"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    boxes = load_boxes(args.data)
    sized = boxes.assign(
        area=boxes["w"] * boxes["h"],
        under_one_cell=(boxes["w"] < GRID_CELL) & (boxes["h"] < GRID_CELL),
    )
    sizes = (
        sized.groupby("label")
        .agg(n=("area", "size"), median_w=("w", "median"), median_h=("h", "median"),
             median_area=("area", "median"), under_one_cell=("under_one_cell", "mean"))
        .rename_axis("finding")
        .sort_values("median_area")
        .round(3)
    )
    sizes.to_csv(args.out / "box_sizes.csv")
    log(f"box sizes by finding, one grid cell is {GRID_CELL} by {GRID_CELL} px:\n{sizes.to_string()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Classifier(len(LABELS), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model = model.to(device).eval()
    transform = build_transforms(512, train=False)
    nodule = LABELS.index("Nodule")
    cases = boxes[boxes["label"] == "Nodule"]

    rows = []
    for method in METHODS:
        # RandomCAM draws its weights from numpy's global generator, so seed it
        # for every method to keep the baseline the same from run to run.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        stats = []
        for _, case in cases.iterrows():
            image = Image.open(args.images / case[IMAGE_COL]).convert("L")
            x = transform(image).unsqueeze(0).to(device)
            if method == "cam":
                raw = builtin_cam(model, x, nodule)[0]
            else:
                raw = cam_explain(model, x, nodule, method)[0]
            truth = (float(case["x"]), float(case["y"]), float(case["w"]), float(case["h"]))
            derived = map_to_box(raw)
            if derived is None:
                stats.append({"ratio": np.nan, "ceiling": 0.0, "centre_inside": False, "iobb": 0.0})
                continue
            area, derived_area = truth[2] * truth[3], derived[2] * derived[3]
            cx, cy = truth[0] + truth[2] / 2, truth[1] + truth[3] / 2
            stats.append({
                "ratio": derived_area / area,
                "ceiling": min(1.0, area / derived_area),
                "centre_inside": (derived[0] <= cx <= derived[0] + derived[2])
                and (derived[1] <= cy <= derived[1] + derived[3]),
                "iobb": iobb(derived, truth),
            })
        df = pd.DataFrame(stats)
        rows.append({
            "method": method,
            "n": len(df),
            "median_area_ratio": round(float(df["ratio"].median()), 2),
            "reachable_cases": int((df["ceiling"] >= 0.5).sum()),
            "centre_inside": round(float(df["centre_inside"].mean()), 4),
            "iobb_at_0_5_cases": int((df["iobb"] >= 0.5).sum()),
        })
        log(f"{method}: {len(df)} nodule cases in {time.perf_counter() - started:.0f}s")

    table = pd.DataFrame(rows).set_index("method")
    table.to_csv(args.out / "nodule_resolution.csv")
    log(f"nodule measurement ceiling:\n{table.to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
