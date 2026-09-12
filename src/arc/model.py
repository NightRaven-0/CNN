"""DenseNet-121 classifier in the CheXNet layout, with exact class activation maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 trunk, ReLU, global average pooling, one linear layer.

    The plain head matters for explainability. With global average pooling
    followed by a single linear layer, each class logit is exactly the spatial
    mean of that class's activation map plus a bias (Zhou et al., 2016). So
    :meth:`cam` is a decomposition of the prediction rather than an estimate of
    it, which makes it a useful reference point for the post-hoc methods.
    """

    def __init__(self, num_classes: int, *, pretrained: bool = True) -> None:
        super().__init__()
        net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features
        self.classifier = nn.Linear(net.classifier.in_features, num_classes)

    @property
    def target_layer(self) -> nn.Module:
        """Last layer of the trunk, where Grad-CAM-style methods attach their hooks."""
        return self.features[-1]

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # Out-of-place ReLU: an in-place one would overwrite the activations a
        # forward hook on target_layer has just captured.
        return F.relu(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_map(x).mean(dim=(2, 3)))

    @torch.no_grad()
    def cam(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and class activation maps of shape (B, C, h, w).

        ``logits == maps.mean(dim=(2, 3)) + bias`` holds exactly, up to float error.
        """
        f = self.feature_map(x)
        maps = torch.einsum("bkhw,ck->bchw", f, self.classifier.weight)
        logits = maps.mean(dim=(2, 3)) + self.classifier.bias
        return logits, maps
