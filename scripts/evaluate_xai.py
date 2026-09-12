"""Score every explanation method against the radiologist boxes.

Runs on the 880 boxed images, all of which are in the official test split, so the
boxes have never been seen during training and this is an honest test of the
explanations rather than of a localisation model.

Three numbers per method, per finding:

* pointing game, does the hottest point land inside a box
* IoU of the derived box with the radiologist's
* T(IoBB) accuracy at 0.1, 0.25 and 0.5, the measure Wang et al. (2017) report,
  so our numbers have something published to sit beside

``random`` is included deliberately. It is a control, and any metric that rates
it well is too easy to pass and should not be reported as evidence.

Usage:
    python scripts/evaluate_xai.py --checkpoint runs/densenet121_512/best.pt
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

from arc.data import IMAGE_COL, LABELS, build_transforms, load_boxes, load_metadata
from arc.model import DenseNet121Classifier
from arc.xai import builtin_cam, cam_explain
from arc.xai_metrics import iobb, iou, map_to_box, pointing_game

#: ScoreCAM needs one forward pass per channel, and the last block has 1024, so
#: it is opt-in on a subsample rather than part of the default sweep.
DEFAULT_METHODS = ("cam", "gradcam", "gradcam++", "eigencam", "random")
IOBB_THRESHOLDS = (0.1, 0.25, 0.5)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def explain_batch(model, images: torch.Tensor, class_index: int, method: str) -> np.ndarray:
    if method == "cam":
        return builtin_cam(model, images, class_index)
    return cam_explain(model, images, class_index, method)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score explanations against boxes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--images", type=Path, default=Path("data/processed/nih512"))
    parser.add_argument("--out", type=Path, default=Path("runs/xai"))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="cap boxed images (smoke tests)")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="score only the boxed images already preprocessed (smoke tests)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    boxes = load_boxes(args.data)
    meta = load_metadata(args.data)
    test_images = set(meta.loc[meta["split"] == "test", IMAGE_COL])
    boxes = boxes[boxes[IMAGE_COL].isin(test_images)]
    if boxes.empty:
        log("no boxed images in the test split, which should be impossible")
        return 1

    present = {p.name for p in args.images.glob("*.png")}
    missing = sorted(set(boxes[IMAGE_COL]) - present)
    if missing:
        # The real run must cover all 880, or the headline numbers are computed
        # on whichever images happened to be on disk.
        if not args.allow_missing:
            log(f"{len(missing)} boxed images are not preprocessed, e.g. {missing[:3]}")
            return 1
        log(f"smoke test: skipping {len(missing)} boxed images that are not preprocessed")
        boxes = boxes[boxes[IMAGE_COL].isin(present)]
        if boxes.empty:
            log("none of the boxed images are preprocessed yet")
            return 1

    # One task per (image, finding): a few images carry boxes for two findings.
    tasks = sorted({(row[IMAGE_COL], row["label"]) for _, row in boxes.iterrows()})
    if args.limit:
        tasks = tasks[: args.limit]
    log(f"{len(tasks)} (image, finding) pairs over {boxes[IMAGE_COL].nunique()} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Classifier(len(LABELS), pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    transform = build_transforms(args.size, train=False)
    truth = {
        (image, label): [
            (r.x, r.y, r.w, r.h) for r in boxes[
                (boxes[IMAGE_COL] == image) & (boxes["label"] == label)
            ].itertuples()
        ]
        for image, label in tasks
    }

    records: list[dict[str, object]] = []
    for method in args.methods:
        started = time.perf_counter()
        for start in range(0, len(tasks), args.batch_size):
            chunk = tasks[start : start + args.batch_size]
            # One class index per batch, so group by finding within the chunk.
            for label in sorted({lbl for _, lbl in chunk}):
                names = [n for n, lbl in chunk if lbl == label]
                images = torch.stack(
                    [transform(Image.open(args.images / n).convert("L")) for n in names]
                ).to(device)
                maps = explain_batch(model, images, LABELS.index(label), method)

                for name, heatmap in zip(names, maps, strict=True):
                    gt = truth[(name, label)]
                    predicted = map_to_box(heatmap)
                    best_iou = max((iou(predicted, g) for g in gt), default=0.0) if predicted else 0.0
                    best_iobb = max((iobb(predicted, g) for g in gt), default=0.0) if predicted else 0.0
                    records.append(
                        {
                            "method": method,
                            "finding": label,
                            IMAGE_COL: name,
                            "pointing_hit": pointing_game(heatmap, gt),
                            "iou": best_iou,
                            "iobb": best_iobb,
                            "found_box": predicted is not None,
                        }
                    )
        log(f"{method}: {len(tasks)} pairs in {time.perf_counter() - started:.0f}s")

    detail = pd.DataFrame(records)
    detail.to_csv(args.out / "xai_per_image.csv", index=False)

    for threshold in IOBB_THRESHOLDS:
        detail[f"iobb@{threshold}"] = detail["iobb"] >= threshold

    columns = ["pointing_hit", "iou", "found_box"] + [f"iobb@{t}" for t in IOBB_THRESHOLDS]
    by_finding = detail.groupby(["method", "finding"])[columns].mean().round(4)
    overall = detail.groupby("method")[columns].mean().round(4)
    by_finding.to_csv(args.out / "xai_by_finding.csv")
    overall.to_csv(args.out / "xai_overall.csv")

    log("overall, averaged over all boxed pairs:")
    print(overall.to_string())
    log("a method that cannot beat 'random' here has not explained anything")
    return 0


if __name__ == "__main__":
    sys.exit(main())
