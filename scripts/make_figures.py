"""Figures for the first review: what the model learned and what its explanations show.

Writes static PNGs to figures/. Every number in them already exists as a table
under runs/, which stays the exact and accessible version; the figures are for
seeing the pattern at a glance.

The heatmap galleries show the median case for each finding rather than hand
picked examples, and say so on the figure, because the first question anyone
asks of a gallery is whether the examples were chosen to look good.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from arc.data import IMAGE_COL, LABELS, build_transforms, load_boxes, load_metadata
from arc.figstyle import AXIS, CONTEXT, HEAT, INK, INK_2, MUTED, SLOT_1, SLOT_2, SURFACE, apply_style
from arc.model import DenseNet121Classifier
from arc.xai import builtin_cam, cam_explain
from arc.xai_metrics import normalise, resize_map

METHOD_NAMES = {
    "cam": "CAM",
    "gradcam": "Grad-CAM",
    "gradcam++": "Grad-CAM++",
    "eigencam": "EigenCAM",
    "random": "random map (control)",
}
METHOD_ORDER = ["cam", "gradcam", "gradcam++", "eigencam", "random"]

#: Images are drawn from the 512 px copies; the radiologist boxes are in 1024 px space.
BOX_SCALE = 512 / 1024


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def caption(fig, text: str) -> None:
    width = int(fig.get_figwidth() * 15)
    fig.text(0.01, -0.02, textwrap.fill(text, width), ha="left", va="top", color=INK_2, fontsize=8.5)


def save(fig, out: Path, name: str) -> None:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    log(f"wrote {path}")


def quiet_y(ax) -> None:
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def training_curves(run: Path, out: Path) -> None:
    h = pd.read_csv(run / "history.csv")
    best = h.loc[h["val_mean_auroc"].idxmax()]
    best_epoch, last_epoch = int(best["epoch"]), int(h["epoch"].max())
    line = {"color": SLOT_1, "lw": 2, "solid_capstyle": "round", "solid_joinstyle": "round"}

    fig, (loss_ax, auc_ax) = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    loss_ax.plot(h["epoch"], h["train_loss"], **line)
    loss_ax.set_title("Training loss")

    auc_ax.plot(h["epoch"], h["val_mean_auroc"], **line)
    auc_ax.scatter(best["epoch"], best["val_mean_auroc"], s=70, color=SLOT_1,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
    auc_ax.annotate(f"best: epoch {best_epoch}, {best['val_mean_auroc']:.3f}",
                    (best["epoch"], best["val_mean_auroc"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    low, high = h["val_mean_auroc"].min(), h["val_mean_auroc"].max()
    auc_ax.set_ylim(low - 0.01, high + 0.014)
    auc_ax.set_title("Validation mean AUROC")

    for ax in (loss_ax, auc_ax):
        ax.set_xlabel("epoch")
        ax.set_xticks(h["epoch"])
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    caption(fig, f"Training stopped after epoch {last_epoch}, {last_epoch - best_epoch} epochs without a "
                 f"better validation score. The weights from epoch {best_epoch} are the ones evaluated. "
                 "Training loss kept falling after that point while validation did not, which is the start "
                 "of the overfitting early stopping exists to catch.")
    save(fig, out, "1_training_curves.png")


def published_comparison(run: Path, out: Path) -> None:
    metrics = json.loads((run / "test_metrics.json").read_text())
    rows = [(name, value, False) for name, value in metrics["published_official_split"].items()]
    rows.append(("This project (DenseNet-121, 512 px)", metrics["mean_auroc"], True))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(8.2, 3.4), layout="constrained")
    ys = np.arange(len(rows))
    for y, (_, value, ours) in zip(ys, rows, strict=True):
        ax.scatter(value, y, s=90 if ours else 64, color=SLOT_1 if ours else MUTED,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{value:.3f}", (value, y), xytext=(9, 0), textcoords="offset points", va="center",
                    color=INK if ours else INK_2, fontsize=9, fontweight="bold" if ours else "normal")
    ax.set_yticks(ys, [r[0] for r in rows])
    for tick, (_, _, ours) in zip(ax.get_yticklabels(), rows, strict=True):
        if ours:
            tick.set_color(INK)
            tick.set_fontweight("bold")
    quiet_y(ax)
    ax.set_xlim(0.72, 0.85)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel(f"mean AUROC on the official NIH test split ({metrics['n_test_images']:,} images)")
    ax.set_title("Where this run sits among published results")
    caption(fig, "All published figures used the same official split. CheXNet's often quoted 0.841 is left "
                 "out on purpose: it was measured on its own random split and is not comparable.")
    save(fig, out, "2_published_comparison.png")


def auroc_vs_auprc(run: Path, out: Path) -> None:
    t = pd.read_csv(run / "test_metrics_per_class.csv").sort_values("prevalence").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.2), layout="constrained")
    ys = np.arange(len(t))
    ax.hlines(ys, t["auprc"], t["auroc"], color=AXIS, lw=1.5, zorder=1)
    ax.scatter(t["prevalence"], ys, marker="|", s=170, color=MUTED, linewidths=2, zorder=2,
               label="AUPRC of a random guess (equals prevalence)")
    ax.scatter(t["auprc"], ys, s=64, color=SLOT_2, edgecolor=SURFACE, linewidth=2, zorder=3, label="AUPRC")
    ax.scatter(t["auroc"], ys, s=64, color=SLOT_1, edgecolor=SURFACE, linewidth=2, zorder=3, label="AUROC")
    labels = [f"{f.replace('_', ' ')}  ({p:.1%})" for f, p in zip(t["finding"], t["prevalence"], strict=True)]
    ax.set_yticks(ys, labels)
    quiet_y(ax)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("score on the official test split")
    i = int(t.index[t["finding"] == "Pneumonia"][0])
    ax.annotate(f"AUROC {t.at[i, 'auroc']:.3f}, AUPRC {t.at[i, 'auprc']:.3f}", (t.at[i, "auroc"], i),
                xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_title("AUROC flatters rare findings", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5,
              handletextpad=0.3, columnspacing=1.4, borderaxespad=0.2)
    caption(fig, "Findings ordered by how common they are in the test set, rarest at the bottom, with prevalence "
                 "in brackets. AUPRC measures how often a positive call is right; a random guess scores the "
                 "prevalence, marked by the grey tick.")
    save(fig, out, "3_auroc_vs_auprc.png")


def method_disagreement(xai: Path, out: Path) -> None:
    loc = pd.read_csv(xai / "xai_overall.csv", index_col="method")["pointing_hit"]
    faith = pd.read_csv(xai / "faithfulness_overall.csv", index_col="method")["insertion_minus_deletion"]
    sane = pd.read_csv(xai / "randomisation_check.csv", index_col="method")["mean_abs_rho_last20"]
    panels = [
        ("Points at the radiologist's box", "pointing game, higher is better", loc, False),
        ("Faithful to the model", "insertion minus deletion, higher is better", faith, False),
        ("Moves when the model is randomised", "correlation left over, lower is better", sane, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True, layout="constrained")
    ys = np.arange(len(METHOD_ORDER))[::-1]
    for ax, (title, measure, values, skip_control) in zip(axes, panels, strict=True):
        top = max(float(values[m]) for m in METHOD_ORDER if not (skip_control and m == "random"))
        for y, method in zip(ys, METHOD_ORDER, strict=True):
            if skip_control and method == "random":
                ax.text(0, y, "  not meaningful for a random map", va="center", color=MUTED, fontsize=8)
                continue
            value = float(values[method])
            ax.barh(y, value, height=0.55, color=SLOT_1 if method == "gradcam++" else CONTEXT, zorder=2)
            ax.annotate(f"{value:.3f}", (value, y), xytext=(4, 0), textcoords="offset points",
                        va="center", color=INK_2, fontsize=8.5)
        ax.set_xlim(0, top * 1.3)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(measure)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        quiet_y(ax)
    axes[0].set_yticks(ys, [METHOD_NAMES[m] for m in METHOD_ORDER])
    fig.suptitle("The three measures rank the methods differently", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    caption(fig, "Grad-CAM++, in blue, matches the radiologists' boxes best, yet keeps the most correlation "
                 "once the model's weights are randomised. Grad-CAM and CAM are the most faithful. No method "
                 "leads on all three, which is why all three are reported.")
    save(fig, out, "4_method_disagreement.png")


def localisation_by_finding(xai: Path, out: Path) -> None:
    by = pd.read_csv(xai / "xai_by_finding.csv")
    detail = pd.read_csv(xai / "xai_per_image.csv")
    best = by[by["method"] == "gradcam++"].set_index("finding")
    control = by[by["method"] == "random"].set_index("finding")
    counts = detail[detail["method"] == "gradcam++"].groupby("finding").size()
    order = best["pointing_hit"].sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
    ys = np.arange(len(order))
    ax.barh(ys, best.loc[order, "pointing_hit"], height=0.55, color=SLOT_1, zorder=2, label="Grad-CAM++")
    ax.scatter(control.loc[order, "pointing_hit"], ys, marker="|", s=280, color=INK, linewidths=2,
               zorder=3, label="random map (control)")
    for y, finding in zip(ys, order, strict=True):
        value = best.at[finding, "pointing_hit"]
        ax.annotate(f"{value:.2f}", (value, y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_yticks(ys, [f"{f.replace('_', ' ')}  (n={counts[f]})" for f in order])
    quiet_y(ax)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("pointing game: share of cases where the hottest point lands inside the box")
    ax.set_title("How well the best localiser does, finding by finding", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, borderaxespad=0.2)
    nodule = best.loc["Nodule"]
    resolution = pd.read_csv(xai / "nodule_resolution.csv", index_col="method")
    sizes = pd.read_csv(xai / "box_sizes.csv", index_col="finding")
    weakest = best["pointing_hit"].drop("Nodule").idxmin()
    cases = int(resolution.at["gradcam++", "n"])
    reachable = int(resolution.at["gradcam++", "reachable_cases"])
    reach = (f"could not be reached in any of the {cases} cases" if reachable == 0
             else f"was within reach in only {reachable} of {cases} cases")
    caption(fig, f"{weakest} is the clearest failure: its maps usually point somewhere else, for a pointing "
                 f"score of {best.at[weakest, 'pointing_hit']:.2f}. Nodules score almost as low, "
                 f"{nodule['pointing_hit']:.2f} against {control.at['Nodule', 'pointing_hit']:.2f} for a random "
                 "map, but part of that is measurement. The median nodule box is "
                 f"{sizes.at['Nodule', 'median_w']:.0f} by {sizes.at['Nodule', 'median_h']:.0f} px, about one cell "
                 "of the 16 by 16 explanation grid, so the box drawn from a map is many times larger and T(IoBB) "
                 f"at 0.5 {reach}, however well the map was placed. Counting instead whether that box contains "
                 f"the nodule's centre, Grad-CAM++ manages {resolution.at['gradcam++', 'centre_inside']:.0%} "
                 f"against {resolution.at['random', 'centre_inside']:.0%} for a random map.")
    save(fig, out, "5_localisation_by_finding.png")


def roc_grid(run: Path, data: Path, out: Path) -> None:
    predictions = pd.read_csv(run / "test_predictions.csv")
    meta = load_metadata(data)
    truth = meta.loc[meta["split"] == "test", [IMAGE_COL, *LABELS]]
    merged = predictions.merge(truth, on=IMAGE_COL, suffixes=("_pred", "_true"))
    scores = {lbl: roc_auc_score(merged[f"{lbl}_true"], merged[f"{lbl}_pred"]) for lbl in LABELS}
    order = sorted(LABELS, key=lambda lbl: -scores[lbl])

    cols = 4
    rows = math.ceil(len(order) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10.4), sharex=True, sharey=True, layout="constrained")
    flat = list(axes.flat)
    for ax, label in zip(flat, order, strict=False):
        fpr, tpr, _ = roc_curve(merged[f"{label}_true"], merged[f"{label}_pred"])
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        ax.plot(fpr, tpr, color=SLOT_1, lw=2, solid_capstyle="round")
        ax.set_title(f"{label.replace('_', ' ')}   {scores[label]:.3f}", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    for index in range(len(order), len(flat)):
        flat[index].set_visible(False)
        # Shared x axes only label the bottom row, so a panel sitting above an
        # empty slot would otherwise have no scale under it at all.
        above = index - cols
        if 0 <= above < len(order):
            flat[above].tick_params(labelbottom=True)
    fig.supxlabel("false positive rate", color=INK_2)
    fig.supylabel("true positive rate", color=INK_2)
    fig.suptitle(f"ROC curves on the official test split ({len(merged):,} images), ordered by AUROC",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    caption(fig, "The grey diagonal is a random guess. Infiltration and Pneumonia are the hardest classes here, "
                 "as they are in every published table on this dataset.")
    save(fig, out, "6_roc_curves.png")


def load_model(run: Path, device: torch.device) -> DenseNet121Classifier:
    model = DenseNet121Classifier(len(LABELS), pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    return model.to(device).eval()


def explanation(model, x: torch.Tensor, class_index: int, method: str) -> np.ndarray:
    """A display-ready map at 512 px scaled 0 to 1. CAM is signed, so negative evidence is clipped."""
    if method == "cam":
        raw = np.maximum(builtin_cam(model, x, class_index)[0], 0.0)
    else:
        raw = cam_explain(model, x, class_index, method)[0]
    return normalise(resize_map(raw, 512))


def draw_case(ax, image: np.ndarray, heat: np.ndarray | None, boxes: list[tuple]) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    if heat is not None:
        ax.imshow(heat, cmap=HEAT, vmin=0, vmax=1)
    for x, y, w, h in boxes:
        ax.add_patch(Rectangle((x * BOX_SCALE, y * BOX_SCALE), w * BOX_SCALE, h * BOX_SCALE,
                               fill=False, edgecolor=SLOT_1, linewidth=2))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def boxes_for(boxes: pd.DataFrame, name: str, finding: str) -> list[tuple]:
    hits = boxes[(boxes[IMAGE_COL] == name) & (boxes["label"] == finding)]
    return [(r.x, r.y, r.w, r.h) for r in hits.itertuples()]


def score_label(scores: pd.DataFrame, name: str, finding: str) -> str:
    """The evaluated probability, plus the share of test images scoring at least as high.

    The scores are not calibrated, so a raw probability misleads on its own: a case
    at 0.05 can still rank near the top for a finding whose negatives mostly score
    below that.
    """
    p = float(scores.at[name, finding])
    top = float((scores[finding] >= p).mean())
    return f"p = {p:.2f}\ntop {top:.0%} of test"


def heat_scale(fig, axes) -> None:
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=HEAT), ax=axes, orientation="horizontal",
                 fraction=0.025, pad=0.02, aspect=50,
                 label="explanation strength, scaled within each map")


def gallery_typical(model, scores: pd.DataFrame, xai: Path, data: Path, images: Path, out: Path, device) -> None:
    detail = pd.read_csv(xai / "xai_per_image.csv")
    boxes = load_boxes(data)
    transform = build_transforms(512, train=False)
    chooser = detail[detail["method"] == "gradcam++"]
    medians = chooser.groupby("finding")["iou"].median().sort_values(ascending=False)
    cases = []
    for finding in medians.index:
        rows = chooser[chooser["finding"] == finding].sort_values(["iou", IMAGE_COL]).reset_index(drop=True)
        cases.append((finding, rows.at[len(rows) // 2, IMAGE_COL]))

    columns = ["cam", "gradcam", "gradcam++"]
    pages = math.ceil(len(cases) / 4)
    for page in range(pages):
        chunk = cases[page * 4 : page * 4 + 4]
        ranks: list[tuple[str, float]] = []
        fig, axes = plt.subplots(len(chunk), 4, figsize=(9.4, 2.5 * len(chunk) + 0.9),
                                 squeeze=False, layout="constrained")
        for r, (finding, name) in enumerate(chunk):
            class_index = LABELS.index(finding)
            pil = Image.open(images / name).convert("L")
            image = np.asarray(pil)
            x = transform(pil).unsqueeze(0).to(device)
            truth = boxes_for(boxes, name, finding)
            ranks.append((finding, float((scores[finding] >= scores.at[name, finding]).mean())))

            draw_case(axes[r, 0], image, None, truth)
            axes[r, 0].set_ylabel(f"{finding.replace('_', ' ')}\n{score_label(scores, name, finding)}", rotation=0, ha="right",
                                  va="center", labelpad=12, color=INK, fontsize=10)
            axes[r, 0].set_xlabel("radiologist box in blue", color=INK_2, fontsize=8.5)
            for c, method in enumerate(columns, start=1):
                draw_case(axes[r, c], image, explanation(model, x, class_index, method), truth)
                row = detail[(detail["method"] == method) & (detail["finding"] == finding)
                             & (detail[IMAGE_COL] == name)].iloc[0]
                axes[r, c].set_xlabel(f"{'hit' if row['pointing_hit'] else 'miss'}, IoU {row['iou']:.2f}",
                                      color=INK_2, fontsize=8.5)
        for c, heading in enumerate(["X-ray", *[METHOD_NAMES[m] for m in columns]]):
            axes[0, c].set_title(heading, loc="center", fontsize=10)
        heat_scale(fig, axes)
        fig.suptitle(f"Explanations on typical test cases, page {page + 1} of {pages}", x=0.01, ha="left",
                     fontsize=12, fontweight="bold")
        weakest, weakest_top = max(ranks, key=lambda item: item[1])
        # Only say a case went unflagged when one on this page actually did.
        unflagged = (
            f"The {weakest.replace('_', ' ')} case ranks in the top {weakest_top:.0%}, below most test "
            "images, so the model has not flagged it. Its maps are still drawn, and show where the model "
            "sees the most evidence for the finding, however weak, which is how a map can land in the "
            "right place while the score stays low. "
            if weakest_top > 0.5 else ""
        )
        caption(fig, "Each row is the median case for its finding, ranked by Grad-CAM++ overlap, so these are "
                     "neither the best nor the worst examples. p is the model's probability for the finding. "
                     f"The scores are not calibrated, so below p is the share of the {len(scores):,} test "
                     "images scoring at least as high. Maps are drawn for the finding the radiologist boxed, "
                     "whatever the model predicted, which is how the localisation scores are computed. "
                     + unflagged
                     + "'hit' means the hottest point falls inside the radiologist's box; IoU is the overlap "
                     "of the box derived from the map with the radiologist's box.")
        save(fig, out, f"7_gallery_typical_{page + 1}.png")


def gallery_range(model, xai: Path, data: Path, images: Path, out: Path, device) -> None:
    detail = pd.read_csv(xai / "xai_per_image.csv")
    boxes = load_boxes(data)
    transform = build_transforms(512, train=False)
    chooser = detail[detail["method"] == "gradcam++"]
    findings = ["Cardiomegaly", "Nodule"]

    fig, axes = plt.subplots(len(findings), 3, figsize=(7.4, 2.5 * len(findings) + 0.9),
                             squeeze=False, layout="constrained")
    for r, finding in enumerate(findings):
        rows = chooser[chooser["finding"] == finding].sort_values(["iou", IMAGE_COL]).reset_index(drop=True)
        picks = [("best", rows.iloc[-1]), ("median", rows.iloc[len(rows) // 2]), ("worst", rows.iloc[0])]
        class_index = LABELS.index(finding)
        for c, (tag, row) in enumerate(picks):
            name = row[IMAGE_COL]
            pil = Image.open(images / name).convert("L")
            x = transform(pil).unsqueeze(0).to(device)
            draw_case(axes[r, c], np.asarray(pil), explanation(model, x, class_index, "gradcam++"),
                      boxes_for(boxes, name, finding))
            axes[r, c].set_xlabel(f"{'hit' if row['pointing_hit'] else 'miss'}, IoU {row['iou']:.2f}",
                                  color=INK_2, fontsize=8.5)
            if r == 0:
                axes[r, c].set_title(f"{tag} case", loc="center", fontsize=10)
        axes[r, 0].set_ylabel(f"{finding}\n(n={len(rows)})", rotation=0, ha="right", va="center",
                              labelpad=12, color=INK, fontsize=10)
    heat_scale(fig, axes)
    fig.suptitle("Best, median and worst case for Grad-CAM++", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    caption(fig, "Ranked by overlap with the radiologist's box. A nodule box is about one cell of the 16 by 16 "
                 "explanation grid, so overlap cannot credit a map that lands on a nodule: the median nodule case "
                 "here peaks within one grid cell of it and still scores as a miss. The best nodule case by "
                 "overlap is a diffuse map that happens to cover the box, the same weakness from the other side.")
    save(fig, out, "8_gallery_range.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Make the figures for the first review.")
    parser.add_argument("--run", type=Path, default=Path("runs/densenet121_512"))
    parser.add_argument("--xai", type=Path, default=Path("runs/xai"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--images", type=Path, default=Path("data/processed/nih512"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    training_curves(args.run, args.out)
    published_comparison(args.run, args.out)
    auroc_vs_auprc(args.run, args.out)
    method_disagreement(args.xai, args.out)
    localisation_by_finding(args.xai, args.out)
    roc_grid(args.run, args.data, args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.run, device)
    scores = pd.read_csv(args.run / "test_predictions.csv").set_index(IMAGE_COL)
    gallery_typical(model, scores, args.xai, args.data, args.images, args.out, device)
    gallery_range(model, args.xai, args.data, args.images, args.out, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
