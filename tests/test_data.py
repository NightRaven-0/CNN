"""Tests for NIH metadata loading, the official split, boxes and the dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from arc.data import (
    IMAGE_COL,
    LABELS,
    PATIENT_COL,
    ChestXray14,
    build_transforms,
    load_boxes,
    load_metadata,
    positive_weights,
)


@pytest.fixture
def nih_root(tmp_path):
    """A miniature NIH release: 300 patients, two images each, patient-wise official split."""
    rng = np.random.default_rng(0)
    rows, test, train_val = [], [], []
    for patient in range(1, 301):
        for k in range(2):
            name = f"{patient:08d}_{k:03d}.png"
            found = [label for label in LABELS if rng.random() < 0.12]
            rows.append(
                {
                    IMAGE_COL: name,
                    "Finding Labels": "|".join(found) or "No Finding",
                    "Follow-up #": k,
                    PATIENT_COL: patient,
                    "Patient Age": 50,
                    "Patient Sex": "M",
                    "View Position": "PA",
                }
            )
            (test if patient <= 60 else train_val).append(name)
    pd.DataFrame(rows).to_csv(tmp_path / "Data_Entry_2017_v2020.csv", index=False)
    (tmp_path / "test_list.txt").write_text("\n".join(test))
    (tmp_path / "train_val_list.txt").write_text("\n".join(train_val))
    # Same awkward header as the real file, including the trailing empty columns.
    (tmp_path / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "00000001_000.png,Infiltrate,100.5,200.25,50,60\n"
        "00000002_001.png,Effusion,10,20,30,40\n"
    )
    return tmp_path


def test_official_test_list_is_used_untouched(nih_root):
    df = load_metadata(nih_root)
    expected = set((nih_root / "test_list.txt").read_text().split())
    assert set(df.loc[df["split"] == "test", IMAGE_COL]) == expected


def test_validation_is_carved_from_train_val_by_patient(nih_root):
    df = load_metadata(nih_root)
    assert set(df["split"]) == {"train", "val", "test"}
    assert df.groupby(PATIENT_COL)["split"].nunique().max() == 1
    train_val = set((nih_root / "train_val_list.txt").read_text().split())
    assert set(df.loc[df["split"] == "val", IMAGE_COL]) <= train_val


def test_label_columns_match_the_finding_string(nih_root):
    df = load_metadata(nih_root)
    for _, row in df.head(50).iterrows():
        expected = set(row["Finding Labels"].split("|")) - {"No Finding"}
        assert {label for label in LABELS if row[label] == 1} == expected


def test_image_in_both_official_lists_is_rejected(nih_root):
    first_test = (nih_root / "test_list.txt").read_text().split()[0]
    with open(nih_root / "train_val_list.txt", "a") as f:
        f.write("\n" + first_test)
    with pytest.raises(ValueError, match="exactly one"):
        load_metadata(nih_root)


def test_boxes_use_canonical_label_names(nih_root):
    boxes = load_boxes(nih_root)
    assert boxes["label"].tolist() == ["Infiltration", "Effusion"]
    assert boxes.loc[0, ["x", "y", "w", "h"]].tolist() == [100.5, 200.25, 50, 60]


def test_positive_weight_powers(nih_root):
    df = load_metadata(nih_root)
    assert np.allclose(positive_weights(df, power=0).numpy(), 1.0)
    pos = df[LABELS].sum().to_numpy(dtype=np.float64)
    assert np.allclose(positive_weights(df, power=1).numpy(), (len(df) - pos) / pos, rtol=1e-5)


def test_dataset_items(nih_root):
    df = load_metadata(nih_root).head(4)
    image_dir = nih_root / "images"
    image_dir.mkdir()
    pixels = np.random.default_rng(0).integers(0, 255, (80, 80), dtype=np.uint8)
    for name in df[IMAGE_COL]:
        Image.fromarray(pixels, mode="L").save(image_dir / name)

    ds = ChestXray14(df, image_dir, build_transforms(64, train=False))
    x, y, i = ds[2]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert i == 2
    assert torch.equal(y, torch.tensor(df.iloc[2][LABELS].to_numpy(dtype=np.float32)))


def test_training_augmentation_never_flips():
    ops = build_transforms(64, train=True).transforms
    assert not any(isinstance(op, (T.RandomHorizontalFlip, T.RandomVerticalFlip)) for op in ops)
