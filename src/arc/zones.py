"""Anatomical taxonomy for ARC-CXR.

Two levels of locality, deliberately kept separate:

* ``Expert`` (4) — which region-specific head owns a finding. This is the
  routing target and it is fixed by anatomy, not learned, so a reviewer can
  audit it against a radiology textbook.
* ``LungZone`` (6) — the R/L x upper/mid/lower grid used to phrase parenchymal
  localisation the way a report does ("opacity, right lower zone").

The finding names below are the VinDr-CXR local-finding labels. They are
asserted against the real CSV at load time by :func:`validate_against_labels`
rather than trusted — label sets differ between the PhysioNet release (22
local findings) and the Kaggle competition subset (14), and silently mapping a
misspelled class to the wrong expert would corrupt every downstream number.
"""

from __future__ import annotations

from enum import Enum


class Expert(str, Enum):
    """Region-specific expert heads."""

    PARENCHYMA = "parenchyma"
    PLEURA = "pleura"
    MEDIASTINUM = "mediastinum"
    SKELETAL = "skeletal"
    UNASSIGNED = "unassigned"  # catch-all; routed to every expert, weighted


class LungZone(str, Enum):
    """Spatial zones used for report-style parenchymal localisation."""

    RIGHT_UPPER = "right_upper"
    RIGHT_MIDDLE = "right_middle"
    RIGHT_LOWER = "right_lower"
    LEFT_UPPER = "left_upper"
    LEFT_MIDDLE = "left_middle"
    LEFT_LOWER = "left_lower"


#: Primary expert for each VinDr-CXR local finding.
#:
#: "Primary" matters: calcification and consolidation can both appear pleurally,
#: but each finding is assigned to the expert that owns it in the majority of
#: cases. Multi-expert findings are handled by the router's soft weighting at
#: inference, not by duplicating labels here.
FINDING_TO_EXPERT: dict[str, Expert] = {
    # --- parenchymal ---
    "Atelectasis": Expert.PARENCHYMA,
    "Calcification": Expert.PARENCHYMA,
    "Consolidation": Expert.PARENCHYMA,
    "Edema": Expert.PARENCHYMA,
    "Emphysema": Expert.PARENCHYMA,
    "ILD": Expert.PARENCHYMA,
    "Infiltration": Expert.PARENCHYMA,
    "Lung cavity": Expert.PARENCHYMA,
    "Lung cyst": Expert.PARENCHYMA,
    "Lung Opacity": Expert.PARENCHYMA,
    "Nodule/Mass": Expert.PARENCHYMA,
    "Pulmonary fibrosis": Expert.PARENCHYMA,
    # --- pleural ---
    "Pleural effusion": Expert.PLEURA,
    "Pleural thickening": Expert.PLEURA,
    "Pneumothorax": Expert.PLEURA,
    # --- mediastinal / cardiac ---
    "Aortic enlargement": Expert.MEDIASTINUM,
    "Cardiomegaly": Expert.MEDIASTINUM,
    "Enlarged PA": Expert.MEDIASTINUM,
    "Mediastinal shift": Expert.MEDIASTINUM,
    # --- skeletal ---
    "Clavicle fracture": Expert.SKELETAL,
    "Rib fracture": Expert.SKELETAL,
    # --- unassignable ---
    "Other lesion": Expert.UNASSIGNED,
}

#: Findings that are diffuse by nature. Zone-level localisation metrics are
#: reported separately for these, because scoring a whole-lung process against a
#: single bounding box penalises a model that is behaving correctly.
DIFFUSE_FINDINGS: frozenset[str] = frozenset(
    {"Edema", "Emphysema", "ILD", "Pulmonary fibrosis"}
)

#: Label present in VinDr for a normal study. Never routed.
NO_FINDING = "No finding"


def experts_in_order() -> list[Expert]:
    """Expert heads in a fixed order, excluding the catch-all."""
    return [Expert.PARENCHYMA, Expert.PLEURA, Expert.MEDIASTINUM, Expert.SKELETAL]


def findings_for(expert: Expert) -> list[str]:
    """Findings owned by ``expert``, in stable sorted order."""
    return sorted(name for name, e in FINDING_TO_EXPERT.items() if e is expert)


def expert_index_matrix() -> dict[str, int]:
    """Map each finding to its expert's index in :func:`experts_in_order`.

    ``Expert.UNASSIGNED`` maps to ``-1``; callers must decide how to weight it.
    """
    order = {e: i for i, e in enumerate(experts_in_order())}
    return {name: order.get(e, -1) for name, e in FINDING_TO_EXPERT.items()}


def validate_against_labels(observed: list[str], *, strict: bool = True) -> list[str]:
    """Check the taxonomy against the label names actually present in the data.

    Returns the list of observed findings that have no expert assignment.
    Raises when ``strict`` and anything is unmapped — an unmapped finding would
    otherwise be silently dropped from routing and quietly deflate recall.
    """
    known = set(FINDING_TO_EXPERT) | {NO_FINDING}
    unmapped = sorted(set(observed) - known)
    unused = sorted(known - set(observed) - {NO_FINDING})

    if unmapped and strict:
        raise ValueError(
            f"{len(unmapped)} finding(s) in the data have no expert assignment: "
            f"{unmapped}. Add them to FINDING_TO_EXPERT before training. "
            f"(Findings declared here but absent from the data: {unused or 'none'} "
            "— expected if you are on the 14-class Kaggle subset.)"
        )
    return unmapped
