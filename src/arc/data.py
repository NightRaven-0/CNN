"""NIH ChestX-ray14: labels, the official split, boxes, and a PyTorch dataset.

The official NIH test list is used untouched so results are comparable with
published work. Validation is carved out of the official train_val list by
patient. All 880 images with hand-drawn boxes are in the official test list,
so the boxes never touch training and stay an independent check on the
explanations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from arc.splits import assert_no_leakage, make_splits
from arc.zones import canonical, validate_against_labels

LABELS: list[str] = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
IMAGE_COL = "Image Index"
PATIENT_COL = "Patient ID"

#: Resolution of the NIH release. Box coordinates are given in this space.
ORIGINAL_SIZE = 1024

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_metadata(root: Path, *, val_fraction: float = 0.1, seed: int = 1337) -> pd.DataFrame:
    """One row per image, with 14 binary label columns and a ``split`` column."""
    root = Path(root)
    df = pd.read_csv(root / "Data_Entry_2017_v2020.csv")
    onehot = df["Finding Labels"].str.get_dummies(sep="|")
    validate_against_labels(onehot.columns.tolist())
    df[LABELS] = onehot.reindex(columns=LABELS, fill_value=0).to_numpy(dtype=np.int8)

    test = set((root / "test_list.txt").read_text().split())
    train_val = set((root / "train_val_list.txt").read_text().split())
    in_test = df[IMAGE_COL].isin(test)
    in_train_val = df[IMAGE_COL].isin(train_val)
    if not (in_test ^ in_train_val).all():
        raise ValueError("every image must appear in exactly one of the official lists")

    fit = make_splits(
        df[in_train_val],
        LABELS,
        group_col=PATIENT_COL,
        fractions={"train": 1 - val_fraction, "val": val_fraction},
        seed=seed,
    )
    out = pd.concat([fit, df[in_test].assign(split="test")], ignore_index=True)
    assert_no_leakage(out, group_col=PATIENT_COL)
    return out


def load_boxes(root: Path) -> pd.DataFrame:
    """Hand-drawn boxes, one row per box, in the original 1024-pixel coordinates."""
    raw = pd.read_csv(Path(root) / "BBox_List_2017.csv")
    boxes = pd.DataFrame(
        {
            IMAGE_COL: raw["Image Index"],
            "label": raw["Finding Label"].map(canonical),
            "x": raw["Bbox [x"],
            "y": raw["y"],
            "w": raw["w"],
            "h": raw["h]"],
        }
    )
    validate_against_labels(boxes["label"].unique().tolist())
    return boxes


def positive_weights(frame: pd.DataFrame, power: float = 0.5) -> torch.Tensor:
    """Per-class BCE weight on positives: ``(negatives / positives) ** power``.

    ``power=1`` is the full ratio, which for Hernia (0.2% of images) is a weight
    in the hundreds and a very noisy loss. ``power=0.5`` keeps most of the
    correction without the extremes. ``power=0`` is plain BCE.
    """
    pos = frame[LABELS].sum().to_numpy(dtype=np.float64)
    neg = len(frame) - pos
    return torch.tensor((neg / np.maximum(pos, 1.0)) ** power, dtype=torch.float32)


class RepeatChannels(nn.Module):
    """Grey (1xHxW) to three channels, for ImageNet-pretrained backbones.

    A module rather than a lambda so DataLoader workers can pickle it on Windows.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(3, -1, -1) if x.shape[0] == 1 else x


def build_transforms(size: int, *, train: bool) -> T.Compose:
    ops: list[nn.Module] = [T.PILToTensor(), T.Resize((size, size), antialias=True)]
    if train:
        # Mild geometry only. No horizontal flip: it puts the heart on the wrong
        # side, and left versus right is part of what the localisation reports.
        ops.append(T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)))
    ops += [
        T.ToDtype(torch.float32, scale=True),
        RepeatChannels(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(ops)


class ChestXray14(Dataset):
    """Images from ``image_dir`` paired with their 14-label target vector.

    Items are ``(image, target, index)``. The index points back into ``frame``,
    so any explanation or error can be traced to its file.
    """

    def __init__(
        self, frame: pd.DataFrame, image_dir: Path, transform: nn.Module | None = None
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        # torch.tensor copies; pandas hands back a read-only array under copy-on-write.
        self.targets = torch.tensor(self.frame[LABELS].to_numpy(dtype=np.float32))
        self.transform = transform or T.PILToTensor()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        with Image.open(self.image_dir / self.frame.at[i, IMAGE_COL]) as im:
            img = im.convert("L")
        return self.transform(img), self.targets[i], i
