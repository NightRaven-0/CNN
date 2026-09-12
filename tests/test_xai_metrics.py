"""Tests for explanation scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arc.xai_metrics import (
    boxes_to_mask,
    curve_auc,
    deletion_curve,
    insertion_curve,
    iobb,
    iou,
    map_to_box,
    normalise,
    pointing_game,
    resize_map,
)

BOX = (256.0, 256.0, 512.0, 512.0)  # centre quarter of a 1024 px image


def peaked_map(row: int, col: int, size: int = 16) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.float32)
    m[row, col] = 1.0
    return m


def test_boxes_scale_from_1024_to_the_map_frame():
    mask = boxes_to_mask([BOX], 512)
    assert mask.shape == (512, 512)
    assert mask[256, 256] and not mask[10, 10]
    assert mask.sum() == 256 * 256


def test_pointing_game_hit_and_miss():
    assert pointing_game(peaked_map(8, 8), [BOX]) is True
    assert pointing_game(peaked_map(0, 0), [BOX]) is False


def test_pointing_game_needs_a_box():
    with pytest.raises(ValueError, match="at least one box"):
        pointing_game(peaked_map(8, 8), [])


def test_map_to_box_bounds_the_hot_blob():
    m = np.zeros((32, 32), dtype=np.float32)
    m[8:16, 8:16] = 1.0
    x, y, w, h = map_to_box(m, size=1024, quantile=0.9)
    # The blob occupies the second quarter of the frame, so the box lands there.
    assert 200 < x < 300 and 200 < y < 300
    assert 200 < w < 320 and 200 < h < 320


def test_map_to_box_returns_none_for_a_flat_map():
    assert map_to_box(np.zeros((16, 16), dtype=np.float32)) is None


def test_iou_and_iobb_on_known_boxes():
    assert iou(BOX, BOX) == pytest.approx(1.0)
    assert iobb(BOX, BOX) == pytest.approx(1.0)
    # A small box fully inside a larger one: IoU is small, IoBB is 1.
    inner = (300.0, 300.0, 100.0, 100.0)
    assert iou(inner, BOX) == pytest.approx(10000 / (512 * 512), rel=1e-3)
    assert iobb(inner, BOX) == pytest.approx(1.0)


def test_disjoint_boxes_score_zero():
    assert iou((0.0, 0.0, 10.0, 10.0), BOX) == 0.0
    assert iobb((0.0, 0.0, 10.0, 10.0), BOX) == 0.0


def test_normalise_handles_a_flat_map():
    assert np.array_equal(normalise(np.full((4, 4), 3.0, dtype=np.float32)), np.zeros((4, 4)))


def test_resize_map_changes_shape_only():
    out = resize_map(np.ones((8, 8), dtype=np.float32), 64)
    assert out.shape == (64, 64)
    assert out == pytest.approx(np.ones((64, 64)), abs=1e-5)


def test_deletion_falls_and_insertion_rises_for_a_good_map():
    # Score is the mean pixel value, so a map that ranks bright pixels first
    # should strip signal quickly on deletion and restore it quickly on insertion.
    image = torch.zeros(1, 8, 8)
    image[0, :4, :] = 1.0
    heatmap = image[0].numpy().copy()

    def predict(x: torch.Tensor) -> float:
        return float(x.mean())

    deletion = deletion_curve(predict, image, heatmap, steps=4)
    insertion = insertion_curve(predict, image, heatmap, steps=4)

    assert deletion[0] > deletion[-1]
    assert insertion[0] < insertion[-1]
    assert np.all(np.diff(deletion) <= 1e-6)
    assert np.all(np.diff(insertion) >= -1e-6)


def test_curve_auc_of_a_constant_curve():
    assert curve_auc(np.full(11, 0.4, dtype=np.float32)) == pytest.approx(0.4)
