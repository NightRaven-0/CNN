"""Tests for the explanation methods.

These use a small stand-in network rather than DenseNet-121. The randomisation
check re-initialises every parameterised module in turn and recomputes the map
each time, which on a 121-layer network would make the suite slow enough that
nobody runs it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from arc.xai import cam_explain, cascading_randomisation, integrated_gradients, occlusion


class TinyNet(nn.Module):
    """Same shape of head as the real model: trunk, ReLU, mean pool, one linear."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
        )
        self.classifier = nn.Linear(16, num_classes)

    @property
    def target_layer(self) -> nn.Module:
        return self.features[-1]

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_map(x).mean(dim=(2, 3)))

    @torch.no_grad()
    def cam(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.feature_map(x)
        maps = torch.einsum("bkhw,ck->bchw", f, self.classifier.weight)
        return maps.mean(dim=(2, 3)) + self.classifier.bias, maps


@pytest.fixture
def model() -> TinyNet:
    torch.manual_seed(0)
    return TinyNet().eval()


@pytest.fixture
def images() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 3, 32, 32)


def test_integrated_gradients_returns_one_value_per_pixel(model, images):
    attribution = integrated_gradients(model, images[0], 1, steps=4)
    assert attribution.shape == (32, 32)
    assert attribution.min() >= 0.0  # absolute attribution


def test_occlusion_grid_follows_patch_and_stride(model, images):
    drops = occlusion(model, images[0], 1, patch=16, stride=8)
    assert drops.shape == (3, 3)


def test_cam_explain_rejects_an_unknown_method(model, images):
    with pytest.raises(KeyError, match="unknown method"):
        cam_explain(model, images, 1, "not-a-method")


def test_randomisation_breaks_the_explanation(model, images):
    """A map that survives randomising the whole network is not explaining it."""
    trace = cascading_randomisation(model, images, 1, method="cam")
    assert trace, "expected one entry per parameterised module"
    assert all(-1.0 <= rho <= 1.0 or rho != rho for _, rho, _ in trace)
    assert all(0.0 <= flat <= 1.0 for _, _, flat in trace)
    # By the time every layer has been re-randomised, the map should have moved.
    assert abs(trace[-1][1]) < 0.9


def test_a_flat_map_is_counted_rather_than_crashing(model, images):
    """Grad-CAM can collapse to an all-zero map. That is a finding, not an error."""
    with torch.no_grad():
        model.classifier.weight.zero_()  # every activation map is now identically zero
    trace = cascading_randomisation(model, images, 1, method="cam")
    assert all(flat == 1.0 for _, _, flat in trace)
    assert all(rho != rho for _, rho, _ in trace)  # NaN: correlation is undefined
