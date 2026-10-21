"""Shared figure style for validation plots (static PNG + PDF).

Categorical colours are the first slots of the validated reference palette, in fixed
order: the proposed/new method is always slot 1 (blue), the legacy/baseline is slot 2
(orange), and the exact reference is slot 3 (aqua). Text stays in ink colours, the grid
is recessive, and lines are 2 px.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NEW, LEGACY, EXACT, FOURTH = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 150,
    })


def save(fig, directory, name):
    os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(directory, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(directory, f"{name}.png")
