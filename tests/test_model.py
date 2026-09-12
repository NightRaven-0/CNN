"""Tests for the classifier."""

from __future__ import annotations

import torch

from arc.model import DenseNet121Classifier


def test_cam_is_an_exact_decomposition_of_the_logits():
    torch.manual_seed(0)
    model = DenseNet121Classifier(14, pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
    cam_logits, maps = model.cam(x)
    assert maps.shape == (2, 14, 2, 2)
    torch.testing.assert_close(cam_logits, logits, atol=1e-4, rtol=1e-4)


def test_target_layer_is_the_last_trunk_layer():
    model = DenseNet121Classifier(14, pretrained=False)
    assert model.target_layer is model.features[-1]
