from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


COLORS = {
    "blue": "#2171b5",
    "orange": "#d94801",
    "green": "#238b45",
    "red": "#cb181d",
    "purple": "#6a51a3",
    "gray": "#636363",
    "cyan": "#0097a7",
    "pink": "#dd3497",
    "olive": "#6b6e23",
    "brown": "#8b4513",
}

COLOR_BIRTH = "#238b45"
COLOR_DEATH = "#cb181d"
COLOR_EDGE = "#4a4a4a"

CMAP_STABILITY = LinearSegmentedColormap.from_list(
    "stability_warm",
    [(0.00, "#fff3b0"), (0.25, "#ffcb69"), (0.50, "#e8871e"),
     (0.75, "#c9462a"), (1.00, "#4a1a2a")],
)

CMAP_PROB = LinearSegmentedColormap.from_list(
    "prob_diverging",
    [(0.00, "#1e4a7a"), (0.15, "#3a7ab8"), (0.35, "#8cb4d9"),
     (0.50, "#f0f0f0"), (0.65, "#e8a99a"), (0.85, "#c24a4a"),
     (1.00, "#8b1a1a")],
)

CMAP_DISC = LinearSegmentedColormap.from_list(
    "disc_diverging",
    [(0.00, "#1e4a7a"), (0.20, "#4a8ac4"), (0.40, "#a8cce8"),
     (0.50, "#f0f0f0"), (0.60, "#e8a8a8"), (0.80, "#c45a5a"),
     (1.00, "#8b1a1a")],
)

CMAP_DIVERGING = plt.cm.RdBu_r
CMAP_PROBABILITY = plt.cm.coolwarm


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def text_color_for_bg(rgba: Tuple[float, ...], threshold: float = 0.45) -> str:
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < threshold else "black"


def text_color_for_stability(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "black"
    norm = (value - vmin) / (vmax - vmin)
    return "black" if norm < 0.55 else "white"


def save_figure(fig: plt.Figure, path, formats=("pdf", "png"), dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=dpi)

