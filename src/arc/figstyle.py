"""Colours and base styling shared by the review figures and the prediction tool.

Values are the light-mode reference palette from the dataviz guidance. Slot 1
marks what a figure is about, slot 2 is a second series where there is one, and
anything that is only context is grey. Text uses the ink tokens, never a series
colour.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#c3c2b7"
SLOT_1 = "#2a78d6"
SLOT_2 = "#eb6834"

#: Heatmap overlay. A warm ramp built from neighbouring palette hues (yellow,
#: orange, red), the one multi-hue scale the guidance allows, and only ever shown
#: with a scale bar. Opacity rises with the value, so the X-ray stays readable
#: wherever the explanation is weak.
HEAT = LinearSegmentedColormap.from_list(
    "explanation_heat",
    [
        (0.00, (0.929, 0.631, 0.000, 0.00)),
        (0.35, (0.929, 0.631, 0.000, 0.30)),
        (0.70, (0.922, 0.408, 0.204, 0.62)),
        (1.00, (0.890, 0.286, 0.282, 0.85)),
    ],
)


def apply_style() -> None:
    """Matplotlib defaults to match the palette: quiet axes, hairline grid, sans type."""
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "legend.frameon": False,
        }
    )
