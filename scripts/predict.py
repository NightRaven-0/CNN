"""Run the trained model on one chest X-ray and show what it looked at.

Prints all 14 findings ranked by probability, then saves a picture of the X-ray
with an explanation heatmap for the top finding, or for one you name, laid over
it. When the image comes from the NIH release and a radiologist drew a box for
that finding, the box is drawn as well, so the two can be compared directly.

This is a research prototype trained on one public dataset from one hospital.
It is not a diagnostic tool, and nothing it prints is medical advice.

Usage:
    python scripts/predict.py data/processed/nih512/00019426_000.png
    python scripts/predict.py some_xray.png --finding Effusion --method gradcam++
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

from arc.data import IMAGE_COL, LABELS, build_transforms, load_boxes, load_metadata
from arc.figstyle import HEAT, INK, INK_2, SLOT_1, apply_style
from arc.model import DenseNet121Classifier
from arc.xai import builtin_cam, cam_explain
from arc.xai_metrics import normalise, resize_map
from arc.zones import LungZone

SIZE = 512
#: Radiologist boxes are recorded in the 1024 px space of the original release.
BOX_SCALE = SIZE / 1024


def load_xray(path: Path) -> Image.Image:
    """Greyscale image from PNG, JPEG or DICOM."""
    if path.suffix.lower() == ".dcm":
        import pydicom

        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels  # MONOCHROME1 stores the grey scale inverted
        span = max(float(pixels.max() - pixels.min()), 1e-6)
        return Image.fromarray((255 * (pixels - pixels.min()) / span).astype(np.uint8))
    return Image.open(path).convert("L")


def approximate_zone(heatmap: np.ndarray) -> str:
    """Name the region around the map's hottest point, from image geometry alone.

    On a standard frontal film the patient's right is on the left of the image.
    The image is split into halves for side and thirds for height, with a central
    strip for the heart and mediastinum. There is no lung segmentation behind
    this, so it is a rough readout for demonstrations and has not been checked
    against anatomy.
    """
    height, width = heatmap.shape
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    third = ("UPPER", "MIDDLE", "LOWER")[min(int(3 * y / height), 2)]
    if 0.42 * width <= x <= 0.58 * width:
        return f"central strip, {third.lower()} third"
    side = "RIGHT" if x < width / 2 else "LEFT"
    return LungZone[f"{side}_{third}"].value.replace("_", " ") + " zone"


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict findings on one chest X-ray and explain the top one.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/densenet121_512/best.pt"))
    parser.add_argument("--finding", choices=LABELS, help="explain this finding instead of the top one")
    parser.add_argument("--method", choices=["cam", "gradcam", "gradcam++", "eigencam"], default="cam")
    parser.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--out", type=Path, default=Path("runs/predictions"))
    args = parser.parse_args()

    if not args.image.exists():
        print(f"no such file: {args.image}")
        return 1

    name = args.image.with_suffix(".png").name
    meta_path = args.data / "Data_Entry_2017_v2020.csv"
    split = None
    if meta_path.exists():
        meta = load_metadata(args.data)
        match = meta.loc[meta[IMAGE_COL] == name, "split"]
        split = match.iloc[0] if len(match) else None
        if split in {"train", "val"}:
            print(f"warning: {name} was in the {split} split, so the model has seen it.")
            print("         Use an image from the test split for an honest demonstration.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Classifier(len(LABELS), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model = model.to(device).eval()

    pil = load_xray(args.image).resize((SIZE, SIZE), Image.BILINEAR)
    x = build_transforms(SIZE, train=False)(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].float().cpu().numpy()

    ranked = sorted(zip(LABELS, probs, strict=True), key=lambda kv: -kv[1])
    print(f"{args.image.name}" + (f"  (NIH {split} split)" if split else ""))
    print("-" * 44)
    for label, p in ranked:
        bar = "#" * int(round(p * 20))
        print(f"  {label.replace('_', ' '):20s} {p:6.3f}  {bar}")

    finding = args.finding or ranked[0][0]
    class_index = LABELS.index(finding)
    if args.method == "cam":
        raw = np.maximum(builtin_cam(model, x, class_index)[0], 0.0)
    else:
        raw = cam_explain(model, x, class_index, args.method)[0]
    heat = normalise(resize_map(raw, SIZE))
    zone = approximate_zone(heat)

    truth = []
    if (args.data / "BBox_List_2017.csv").exists():
        boxes = load_boxes(args.data)
        hits = boxes[(boxes[IMAGE_COL] == name) & (boxes["label"] == finding)]
        truth = [(r.x, r.y, r.w, r.h) for r in hits.itertuples()]

    apply_style()
    image = np.asarray(pil)
    fig, (plain, overlay) = plt.subplots(1, 2, figsize=(8.4, 4.9), layout="constrained")
    plain.imshow(image, cmap="gray", vmin=0, vmax=255)
    plain.set_title("X-ray", loc="center", fontsize=10)
    overlay.imshow(image, cmap="gray", vmin=0, vmax=255)
    overlay.imshow(heat, cmap=HEAT, vmin=0, vmax=1)
    overlay.set_title(f"{args.method} explanation for {finding.replace('_', ' ')}", loc="center", fontsize=10)
    for bx, by, bw, bh in truth:
        overlay.add_patch(Rectangle((bx * BOX_SCALE, by * BOX_SCALE), bw * BOX_SCALE, bh * BOX_SCALE,
                                    fill=False, edgecolor=SLOT_1, linewidth=2))
    for ax in (plain, overlay):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=HEAT), ax=overlay, orientation="horizontal",
                 fraction=0.04, pad=0.02, label="explanation strength")
    fig.suptitle(f"{finding.replace('_', ' ')}: p = {probs[class_index]:.2f}, strongest near the {zone}",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    note = "Blue box: radiologist. " if truth else ""
    fig.text(0.01, -0.02, note + "Research prototype trained on one public dataset. Not for diagnosis. "
             "Zone names are approximate, from image geometry.", ha="left", va="top", color=INK_2, fontsize=8)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.image.stem}_{finding}_{args.method}.png"
    fig.savefig(path)
    plt.close(fig)

    print("-" * 44)
    print(f"explained: {finding} (p = {probs[class_index]:.3f}) with {args.method}")
    print(f"strongest evidence near the {zone} (approximate)")
    if truth:
        print(f"radiologist box drawn: {len(truth)}")
    print(f"saved: {path}")
    print("\nResearch prototype. Not a diagnostic tool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
