"""Test whether explanations are faithful to the model, and whether they depend on it.

Two questions that localisation scores cannot answer:

* Faithfulness. Blank out the pixels a map calls important and the score should
  fall quickly; add them back to an empty image and it should climb quickly.
  A map that ranks pixels no better than chance moves the score no faster than
  the random control does (Petsiuk et al., 2018).
* Dependence on the model. Re-randomise the network from the output back and the
  map should change. One that stays put is describing the image rather than
  explaining the prediction, and its localisation score is worthless however
  good it looks (Adebayo et al., 2018).

Both run on a subsample: deletion and insertion cost about 40 forward passes per
image per method, and the randomisation trace re-explains the image once per
parameterised layer, of which DenseNet-121 has 242.

Usage:
    python scripts/evaluate_faithfulness.py --checkpoint runs/densenet121_512/best.pt
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
from arc.xai import builtin_cam, cam_explain, cascading_randomisation
from arc.xai_metrics import curve_auc, deletion_curve, insertion_curve

DEFAULT_METHODS = ("cam", "gradcam", "gradcam++", "eigencam", "random")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def explain_batch(model, images: torch.Tensor, class_index: int, method: str) -> np.ndarray:
    if method == "cam":
        return builtin_cam(model, images, class_index)
    return cam_explain(model, images, class_index, method)


def main() -> int:
    parser = argparse.ArgumentParser(description="Faithfulness and sanity checks.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--images", type=Path, default=Path("data/processed/nih512"))
    parser.add_argument("--out", type=Path, default=Path("runs/xai"))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=50, help="images for deletion/insertion")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--randomisation-samples", type=int, default=4)
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(args.data)
    boxes = load_boxes(args.data)
    test_images = set(meta.loc[meta["split"] == "test", IMAGE_COL])
    present = {p.name for p in args.images.glob("*.png")}
    boxes = boxes[boxes[IMAGE_COL].isin(test_images) & boxes[IMAGE_COL].isin(present)]
    if boxes.empty:
        log("no boxed test images are preprocessed yet")
        return 1

    pairs = sorted({(r[IMAGE_COL], r["label"]) for _, r in boxes.iterrows()})
    rng = np.random.default_rng(args.seed)
    chosen = [pairs[i] for i in rng.permutation(len(pairs))[: args.samples]]
    log(f"{len(chosen)} (image, finding) pairs for deletion and insertion")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Classifier(len(LABELS), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model = model.to(device).eval()
    transform = build_transforms(args.size, train=False)

    rows: list[dict[str, object]] = []
    for method in args.methods:
        started = time.perf_counter()
        for name, label in chosen:
            class_index = LABELS.index(label)
            image = transform(Image.open(args.images / name).convert("L")).to(device)
            heatmap = explain_batch(model, image.unsqueeze(0), class_index, method)[0]

            @torch.no_grad()
            def predict(x: torch.Tensor, _c: int = class_index) -> float:
                return float(torch.sigmoid(model(x.unsqueeze(0))[0, _c]))

            # Zeros here means the ImageNet mean after normalisation, a neutral
            # grey rather than a black rectangle the model has never seen.
            deletion = deletion_curve(predict, image, heatmap, steps=args.steps)
            insertion = insertion_curve(predict, image, heatmap, steps=args.steps)
            rows.append(
                {
                    "method": method,
                    "finding": label,
                    IMAGE_COL: name,
                    "deletion_auc": curve_auc(deletion),
                    "insertion_auc": curve_auc(insertion),
                }
            )
        log(f"{method}: {len(chosen)} pairs in {time.perf_counter() - started:.0f}s")

    detail = pd.DataFrame(rows)
    detail.to_csv(args.out / "faithfulness_per_image.csv", index=False)
    summary = detail.groupby("method")[["deletion_auc", "insertion_auc"]].mean().round(4)
    # Insertion minus deletion in one column, so methods sort by a single number.
    summary["insertion_minus_deletion"] = (
        summary["insertion_auc"] - summary["deletion_auc"]
    ).round(4)
    summary.to_csv(args.out / "faithfulness_overall.csv")

    log("faithfulness (low deletion is good, high insertion is good):")
    print(summary.to_string())

    if args.skip_randomisation:
        return 0

    sample = chosen[: args.randomisation_samples]
    batch = torch.stack(
        [transform(Image.open(args.images / n).convert("L")) for n, _ in sample]
    ).to(device)
    sanity: list[dict[str, object]] = []
    for method in args.methods:
        started = time.perf_counter()
        trace = cascading_randomisation(
            model, batch, LABELS.index(sample[0][1]), method=method, seed=args.seed
        )
        tail = trace[-20:]
        rhos = [rho for _, rho, _ in tail if rho == rho]
        sanity.append(
            {
                "method": method,
                "layers": len(trace),
                # The correlation and the flat fraction below cover the same last
                # 20 layers, so they read together: a NaN correlation beside a
                # flat fraction of 1.0 means the method returned an empty map
                # every time, which is a finding rather than a missing value.
                "mean_abs_rho_last20": round(float(np.mean(np.abs(rhos))), 4) if rhos else float("nan"),
                "flat_fraction_last20": round(float(np.mean([f for _, _, f in tail])), 4),
                "flat_fraction_all": round(float(np.mean([f for _, _, f in trace])), 4),
            }
        )
        log(f"{method}: randomisation trace in {time.perf_counter() - started:.0f}s")

    table = pd.DataFrame(sanity)
    table.to_csv(args.out / "randomisation_check.csv", index=False)
    log("sanity check (correlation near zero once randomised is what we want):")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
