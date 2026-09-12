"""Anatomical vocabulary for saying where a finding is.

Localisation is the project's second priority, after explainability, so this
module does not drive the model. It is the vocabulary the explanation layer uses
to turn a heatmap into something a radiologist would write: "effusion, left
lower zone" rather than a coloured blob.

Two levels:

* ``Region`` is the anatomical compartment a finding belongs to. Results tables
  are grouped by it, and it gives a cheap plausibility check on explanations: a
  cardiomegaly heatmap that peaks in the lung apex is wrong whatever its score.
* ``LungZone`` is the left/right x upper/middle/lower grid used in reports.

Label names cover NIH ChestX-ray14 (the primary dataset) and VinDr-CXR (kept as
a possible second evaluation set with better boxes). They are checked against
the real CSV at load time by :func:`validate_against_labels` instead of being
trusted, because a misspelt class would silently drop out of every table.
"""

from __future__ import annotations

from enum import Enum


class Region(str, Enum):
    """Anatomical compartment a finding is primarily located in."""

    PARENCHYMA = "parenchyma"
    PLEURA = "pleura"
    MEDIASTINUM = "mediastinum"
    SKELETAL = "skeletal"
    UNASSIGNED = "unassigned"


class LungZone(str, Enum):
    """Report-style zones for parenchymal and pleural findings."""

    RIGHT_UPPER = "right_upper"
    RIGHT_MIDDLE = "right_middle"
    RIGHT_LOWER = "right_lower"
    LEFT_UPPER = "left_upper"
    LEFT_MIDDLE = "left_middle"
    LEFT_LOWER = "left_lower"


#: Primary region for each finding label.
#:
#: "Primary" because several findings can occur in more than one compartment
#: (calcification, for one). Each is assigned where it occurs most often, which
#: is a convention to be stated in the write-up, not a claim about every case.
FINDING_TO_REGION: dict[str, Region] = {
    # --- shared by NIH ChestX-ray14 and VinDr-CXR ---
    "Atelectasis": Region.PARENCHYMA,
    "Cardiomegaly": Region.MEDIASTINUM,
    "Consolidation": Region.PARENCHYMA,
    "Edema": Region.PARENCHYMA,
    "Emphysema": Region.PARENCHYMA,
    # --- NIH ChestX-ray14 only ---
    "Effusion": Region.PLEURA,
    "Fibrosis": Region.PARENCHYMA,
    "Hernia": Region.MEDIASTINUM,  # hiatal hernia, retrocardiac on a PA film
    "Infiltration": Region.PARENCHYMA,
    "Mass": Region.PARENCHYMA,
    "Nodule": Region.PARENCHYMA,
    "Pleural_Thickening": Region.PLEURA,
    "Pneumonia": Region.PARENCHYMA,
    "Pneumothorax": Region.PLEURA,
    # --- VinDr-CXR only ---
    "Aortic enlargement": Region.MEDIASTINUM,
    "Calcification": Region.PARENCHYMA,
    "Clavicle fracture": Region.SKELETAL,
    "Enlarged PA": Region.MEDIASTINUM,
    "ILD": Region.PARENCHYMA,
    "Lung cavity": Region.PARENCHYMA,
    "Lung cyst": Region.PARENCHYMA,
    "Lung Opacity": Region.PARENCHYMA,
    "Mediastinal shift": Region.MEDIASTINUM,
    "Nodule/Mass": Region.PARENCHYMA,
    "Other lesion": Region.UNASSIGNED,
    "Pleural effusion": Region.PLEURA,
    "Pleural thickening": Region.PLEURA,
    "Pulmonary fibrosis": Region.PARENCHYMA,
    "Rib fracture": Region.SKELETAL,
}

#: Findings that are spread across the lungs rather than sitting in one place.
#: Box-based localisation scores are reported separately for these, since
#: scoring a whole-lung process against one box penalises a correct heatmap.
DIFFUSE_FINDINGS: frozenset[str] = frozenset(
    {"Edema", "Emphysema", "Fibrosis", "ILD", "Pulmonary fibrosis"}
)

#: The normal-study label. NIH capitalises it differently from VinDr.
NO_FINDING_LABELS: frozenset[str] = frozenset({"No Finding", "No finding"})

#: Spellings that differ between a dataset's label file and its box file.
#: The NIH box list is reported to use "Infiltrate" where the label file says
#: "Infiltration". Anything else that disagrees is caught by
#: :func:`validate_against_labels` on first load.
LABEL_ALIASES: dict[str, str] = {"Infiltrate": "Infiltration"}


def canonical(label: str) -> str:
    """Normalise a label spelling to the key used in :data:`FINDING_TO_REGION`."""
    return LABEL_ALIASES.get(label, label)


def region_of(label: str) -> Region:
    """Region for ``label``, accepting alias spellings."""
    return FINDING_TO_REGION[canonical(label)]


def findings_in(region: Region, labels: list[str] | None = None) -> list[str]:
    """Findings located in ``region``, optionally restricted to ``labels``."""
    pool = FINDING_TO_REGION if labels is None else [canonical(x) for x in labels]
    return sorted(name for name in pool if FINDING_TO_REGION.get(name) is region)


def validate_against_labels(observed: list[str], *, strict: bool = True) -> list[str]:
    """Check the taxonomy covers every label actually present in the data.

    Returns the observed labels with no region assignment. Raises when
    ``strict`` and anything is unmapped, since an unmapped finding would be
    silently left out of every grouped table.
    """
    known = set(FINDING_TO_REGION) | NO_FINDING_LABELS
    unmapped = sorted({canonical(x) for x in observed} - known)
    if unmapped and strict:
        raise ValueError(
            f"{len(unmapped)} label(s) in the data have no region assignment: "
            f"{unmapped}. Add them to FINDING_TO_REGION (or LABEL_ALIASES if it "
            "is a spelling difference) before training."
        )
    return unmapped
