"""Tests for the anatomical vocabulary."""

from __future__ import annotations

import pytest

from arc.zones import (
    NO_FINDING_LABELS,
    Region,
    findings_in,
    region_of,
    validate_against_labels,
)

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

# The eight classes with hand-drawn boxes, spelled as in the NIH box file.
NIH_BOX_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
]


def test_every_nih_label_has_a_region():
    assert validate_against_labels([*NIH_LABELS, "No Finding"]) == []


def test_box_file_spelling_is_normalised():
    assert validate_against_labels(NIH_BOX_LABELS) == []
    assert region_of("Infiltrate") is region_of("Infiltration") is Region.PARENCHYMA


def test_misspelt_label_raises():
    with pytest.raises(ValueError, match="no region assignment"):
        validate_against_labels([*NIH_LABELS, "Pnuemothorax"])


def test_non_strict_returns_unmapped():
    assert validate_against_labels(["Pnuemothorax"], strict=False) == ["Pnuemothorax"]


def test_findings_in_restricts_to_given_labels():
    assert findings_in(Region.PLEURA, NIH_LABELS) == [
        "Effusion", "Pleural_Thickening", "Pneumothorax",
    ]


def test_both_no_finding_spellings_accepted():
    assert validate_against_labels(sorted(NO_FINDING_LABELS)) == []
