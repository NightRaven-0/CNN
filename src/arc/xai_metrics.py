"""Scoring explanations against radiologist boxes, and testing them for faithfulness.

Three questions that the literature often runs together, kept separate here:

* Does the map point at the finding? :func:`pointing_game`, :func:`iou`, :func:`iobb`
* Is the map faithful to the model?  :func:`deletion_curve`, :func:`insertion_curve`
* Does the map depend on the model?  the randomisation test, which lives in xai.py

Box coordinates come in the NIH release's 1024-pixel space, while heatmaps arrive
at whatever resolution the method produced (16x16 for CAM on a 512 px input).
:func:`resize_map` puts both in one frame before anything is compared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy import ndimage

#: A box as (x, y, w, h), matching the columns of BBox_List_2017.csv.
Box = tuple[float, float, float, float]


def resize_map(heatmap: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly resize a 2-D map to ``size`` x ``size``."""
    t = torch.as_tensor(np.asarray(heatmap, dtype=np.float32))[None, None]
    out = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def normalise(heatmap: np.ndarray) -> np.ndarray:
    """Scale to 0..1. A flat map stays flat rather than dividing by zero."""
    m = np.asarray(heatmap, dtype=np.float32)
    lo, hi = float(m.min()), float(m.max())
    return np.zeros_like(m) if hi - lo < 1e-12 else (m - lo) / (hi - lo)


def boxes_to_mask(boxes: Sequence[Box], size: int, *, source_size: int = 1024) -> np.ndarray:
    """Boolean mask of the union of ``boxes``, rescaled from ``source_size`` to ``size``."""
    mask = np.zeros((size, size), dtype=bool)
    scale = size / source_size
    for x, y, w, h in boxes:
        x0 = max(int(round(x * scale)), 0)
        y0 = max(int(round(y * scale)), 0)
        x1 = min(int(round((x + w) * scale)), size)
        y1 = min(int(round((y + h) * scale)), size)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def pointing_game(heatmap: np.ndarray, boxes: Sequence[Box], *, size: int = 1024) -> bool:
    """True when the map's highest point falls inside any ground-truth box.

    The standard check from Zhang et al. (2018): forgiving, since it ignores the
    map's extent entirely, so treat a high score as necessary, not sufficient.
    """
    if not boxes:
        raise ValueError("pointing_game needs at least one box")
    full = resize_map(heatmap, size)
    y, x = np.unravel_index(int(np.argmax(full)), full.shape)
    return bool(boxes_to_mask(boxes, size)[y, x])


def map_to_box(heatmap: np.ndarray, *, size: int = 1024, quantile: float = 0.95) -> Box | None:
    """Threshold the map, then bound its largest connected blob.

    Wang et al. (2017) derive a box this way before scoring localisation. The
    threshold is a quantile of the map rather than a fixed value, so a diffuse
    map and a peaked one both yield a box.
    """
    full = normalise(resize_map(heatmap, size))
    cutoff = float(np.quantile(full, quantile))
    # Strictly greater. On a mostly-zero map the high quantile is itself zero, so
    # ">=" would mark every background pixel hot and return the whole frame; a
    # flat map would do the same instead of admitting it localises nothing.
    hot = full > cutoff
    if not hot.any():
        return None

    labels, count = ndimage.label(hot)
    if count == 0:
        return None
    largest = 1 + int(np.argmax(ndimage.sum(hot, labels, range(1, count + 1))))
    ys, xs = np.where(labels == largest)
    return float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def _corners(box: Box) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def _intersection(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = _corners(a)
    bx0, by0, bx1, by1 = _corners(b)
    dx = min(ax1, bx1) - max(ax0, bx0)
    dy = min(ay1, by1) - max(ay0, by0)
    return float(dx * dy) if dx > 0 and dy > 0 else 0.0


def iou(pred: Box, truth: Box) -> float:
    """Intersection over union of two boxes."""
    inter = _intersection(pred, truth)
    union = pred[2] * pred[3] + truth[2] * truth[3] - inter
    return inter / union if union > 0 else 0.0


def iobb(pred: Box, truth: Box) -> float:
    """Intersection over the predicted box's own area.

    Wang et al. report localisation accuracy at T(IoBB) thresholds. Their exact
    convention needs checking against the paper before our numbers are put
    beside theirs; this implementation divides by the predicted box, which
    rewards a small box inside a large finding and is the more forgiving reading.
    """
    return _intersection(pred, truth) / (pred[2] * pred[3]) if pred[2] * pred[3] > 0 else 0.0


def _ranked_pixels(heatmap: np.ndarray, size: int) -> np.ndarray:
    """Pixel indices ordered by importance, most important first.

    The copy is load-bearing: reversing a numpy array gives a view with a
    negative stride, and torch refuses to index with one.
    """
    return np.argsort(resize_map(heatmap, size).ravel())[::-1].copy()


def deletion_curve(
    predict: Callable[[torch.Tensor], float],
    image: torch.Tensor,
    heatmap: np.ndarray,
    *,
    steps: int = 20,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Scores as the map's most important pixels are progressively blanked out.

    A map that ranks pixels well makes the score fall fast, so a low area under
    this curve is good (Petsiuk et al., 2018). ``image`` is (C, H, W) and
    ``predict`` returns one score for the class being explained.
    """
    size = image.shape[-1]
    order = _ranked_pixels(heatmap, size)
    blank = torch.zeros_like(image) if baseline is None else baseline
    work = image.clone()
    total = order.size

    scores = [predict(work)]
    for step in range(1, steps + 1):
        upto = order[: int(total * step / steps)]
        work = image.clone()
        flat = work.reshape(work.shape[0], -1)
        flat[:, upto] = blank.reshape(blank.shape[0], -1)[:, upto]
        scores.append(predict(flat.reshape(image.shape)))
    return np.asarray(scores, dtype=np.float32)


def insertion_curve(
    predict: Callable[[torch.Tensor], float],
    image: torch.Tensor,
    heatmap: np.ndarray,
    *,
    steps: int = 20,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Scores as the map's most important pixels are added back to a blank image.

    The mirror of :func:`deletion_curve`; here a high area under the curve is good.
    """
    size = image.shape[-1]
    order = _ranked_pixels(heatmap, size)
    blank = torch.zeros_like(image) if baseline is None else baseline

    scores = [predict(blank)]
    for step in range(1, steps + 1):
        upto = order[: int(order.size * step / steps)]
        work = blank.clone()
        flat = work.reshape(work.shape[0], -1)
        flat[:, upto] = image.reshape(image.shape[0], -1)[:, upto]
        scores.append(predict(flat.reshape(image.shape)))
    return np.asarray(scores, dtype=np.float32)


def curve_auc(scores: np.ndarray) -> float:
    """Area under a deletion or insertion curve, normalised to 0..1."""
    return float(np.trapezoid(scores) / (len(scores) - 1)) if len(scores) > 1 else float("nan")
