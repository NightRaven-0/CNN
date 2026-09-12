"""Explanation methods, and the sanity check that decides whether to trust them.

Six methods, chosen because they fail in different ways:

* ``cam``        the exact decomposition this architecture allows, from :meth:`DenseNet121Classifier.cam`
* ``gradcam``    and ``gradcam++``, the usual choice in medical imaging papers
* ``scorecam``   gradient free, so it cannot be fooled by gradient saturation
* ``ig``         Integrated Gradients, pixel level rather than 16x16
* ``occlusion``  blanks patches and watches the score, measuring the model's own reaction
* ``random``     a control. Any metric that rates this well is too easy to pass.

Integrated Gradients and occlusion are written out here rather than pulled from
a library: both are short, and owning the baseline choice matters, since for a
normalised radiograph a black baseline means "no signal" rather than a picture.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, RandomCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import stats
from torch import nn

#: ScoreCAM runs one forward pass per channel, and DenseNet-121's last block has
#: 1024 of them, so it is for a subsample of images rather than the full test set.
CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
    "random": RandomCAM,
}


def cam_explain(model: nn.Module, images: torch.Tensor, class_index: int, method: str) -> np.ndarray:
    """Heatmaps of shape (B, H, W) at input resolution, from the CAM family."""
    if method not in CAM_METHODS:
        raise KeyError(f"unknown method {method!r}; have {sorted(CAM_METHODS)}")
    targets = [ClassifierOutputTarget(class_index)] * images.shape[0]
    with CAM_METHODS[method](model=model, target_layers=[model.target_layer]) as cam:
        return cam(input_tensor=images, targets=targets)


def builtin_cam(model, images: torch.Tensor, class_index: int) -> np.ndarray:
    """The model's own class activation maps, which sum exactly to its logits."""
    _, maps = model.cam(images)
    return maps[:, class_index].float().cpu().numpy()


def integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    class_index: int,
    *,
    steps: int = 32,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Per-pixel attribution accumulated along a path from a baseline to the image.

    Sundararajan et al. (2017). Returns (H, W): the absolute attribution summed
    over channels, since for a greyscale radiograph the three channels are copies.
    """
    batch = image.unsqueeze(0) if image.dim() == 3 else image
    base = torch.zeros_like(batch) if baseline is None else baseline
    total = torch.zeros_like(batch)

    for k in range(1, steps + 1):
        point = (base + (batch - base) * (k / steps)).detach().requires_grad_(True)
        score = model(point)[0, class_index]
        (grad,) = torch.autograd.grad(score, point)
        total += grad

    attribution = ((batch - base) * total / steps)[0]
    return attribution.abs().sum(0).detach().cpu().numpy()


@torch.no_grad()
def occlusion(
    model: nn.Module,
    image: torch.Tensor,
    class_index: int,
    *,
    patch: int = 64,
    stride: int = 32,
    fill: float = 0.0,
    batch_size: int = 32,
) -> np.ndarray:
    """How far the score drops when each patch is blanked out.

    The most direct attribution available: it measures what the model actually
    does when information is removed, instead of inferring it from gradients.
    Costs one forward pass per patch position, so ``stride`` sets the price.
    """
    batch = image.unsqueeze(0) if image.dim() == 3 else image
    _, _, height, width = batch.shape
    ys = list(range(0, max(height - patch, 0) + 1, stride))
    xs = list(range(0, max(width - patch, 0) + 1, stride))

    reference = float(model(batch)[0, class_index])
    drops = np.zeros((len(ys), len(xs)), dtype=np.float32)

    positions = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        variants = batch.repeat(len(chunk), 1, 1, 1)
        for slot, (i, j) in enumerate(chunk):
            variants[slot, :, ys[i] : ys[i] + patch, xs[j] : xs[j] + patch] = fill
        scores = model(variants)[:, class_index]
        for slot, (i, j) in enumerate(chunk):
            drops[i, j] = reference - float(scores[slot])
    return drops


def _reinitialise(module: nn.Module, generator: torch.Generator) -> None:
    """Re-randomise a module's own parameters, leaving its children alone.

    Values are drawn on the CPU and copied across: a generator only works with
    tensors on its own device, and the model is normally on the GPU.
    """
    for name, param in module.named_parameters(recurse=False):
        with torch.no_grad():
            if param.dim() >= 2:
                bound = 1.0 / max(param.shape[1:].numel(), 1) ** 0.5
                noise = torch.empty(param.shape, dtype=param.dtype, device="cpu")
                param.copy_(noise.uniform_(-bound, bound, generator=generator))
            elif "bias" in name:
                param.zero_()
            else:
                param.fill_(1.0)

    # A BatchNorm holds its running statistics as buffers, not parameters, so the
    # loop above never touches them. In eval mode those trained statistics
    # normalise activations that the randomised convolutions now produce, the
    # scale diverges, and across 121 layers it overflows to inf so every map
    # returns NaN. Reset the statistics along with the weights.
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        with torch.no_grad():
            if module.running_mean is not None:
                module.running_mean.zero_()
            if module.running_var is not None:
                module.running_var.fill_(1.0)
            if module.num_batches_tracked is not None:
                module.num_batches_tracked.zero_()


def cascading_randomisation(
    model,
    images: torch.Tensor,
    class_index: int,
    *,
    method: str = "gradcam",
    seed: int = 0,
) -> list[tuple[str, float, float]]:
    """Adebayo's sanity check: randomise the network downwards and watch the map.

    Returns (layer, similarity, flat_fraction) for each parameterised module,
    working from the output back to the input. Similarity is the Spearman rank
    correlation between the intact model's map and the map after that layer and
    everything above it has been re-randomised. A method whose correlation stays
    near 1 all the way down is not explaining the model, it is describing the
    image, and its localisation scores mean nothing.

    flat_fraction is the share of images whose map came out completely uniform.
    Grad-CAM clips at zero, so a sufficiently randomised network drives it to an
    all-zero map; the correlation is undefined there, and that collapse is worth
    reporting rather than averaging away. Those images are left out of the mean.
    """
    explain = builtin_cam if method == "cam" else (
        lambda m, x, c: cam_explain(m, x, c, method)
    )
    reference = explain(model, images, class_index).reshape(images.shape[0], -1)

    scratch = copy.deepcopy(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    named = [(n, m) for n, m in scratch.named_modules() if any(True for _ in m.parameters(recurse=False))]

    results: list[tuple[str, float, float]] = []
    for name, module in reversed(named):
        _reinitialise(module, generator)
        try:
            maps = explain(scratch, images, class_index).reshape(images.shape[0], -1)
        except Exception as exc:
            # EigenCAM takes an SVD of the activations, and randomised weights can
            # make it degenerate or non-finite so the decomposition fails to
            # converge. That a method breaks down under randomisation is itself
            # worth recording, so note it and carry on down the network.
            results.append((f"{name} [failed: {type(exc).__name__}]", float("nan"), 1.0))
            continue

        rhos: list[float] = []
        flat = 0
        for i in range(images.shape[0]):
            if not (np.isfinite(maps[i]).all() and np.isfinite(reference[i]).all()):
                # The map came back with NaN or inf in it. That is the method
                # breaking down, not agreeing with anything, and it counts as
                # unusable rather than being dropped silently.
                flat += 1
                continue
            if float(maps[i].std()) == 0.0 or float(reference[i].std()) == 0.0:
                flat += 1
                continue
            rhos.append(float(stats.spearmanr(reference[i], maps[i]).statistic))

        finite = [r for r in rhos if np.isfinite(r)]
        results.append((
            name,
            float(np.mean(finite)) if finite else float("nan"),
            flat / images.shape[0],
        ))
    return results
