"""One visual language for every figure in this directory.

Colour carries exactly one thing here: **which composition rule** produced a
curve. There are three of them, so three categorical hues are needed and no
more, and the sampler variant that shares a rule (churn eta = 8 versus ten
Langevin correctors) is carried by line style instead. That keeps the reading
"same colour = same rule" true in every panel, including the scatter panels
where all pairs of series appear together.

The three hues are slots 1-3 of the validated categorical palette in the
data-visualisation reference (blue / orange / aqua). That prefix is the one
documented to clear the all-pairs colour-vision-deficiency and normal-vision
separation floors in both light and dark modes, which is the case that applies
to scatter and small-multiple panels; the alternative -- a fourth and fifth hue
for the corrector variants -- puts yellow next to orange and fails those floors.
Because the aqua slot sits below 3:1 contrast on a light surface, every series
is also directly labelled or legended, and every number behind every figure is
written to CSV alongside it, so identity is never carried by colour alone.

Exact references are drawn in ink, never in a series hue: they are not one of
the things being compared.
"""
from __future__ import annotations

import matplotlib as mpl

# Categorical slots 1-3.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Text and surface roles.
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#8a8f98"
SURFACE, GRID = "#fcfcfb", "#e3e2de"
# The exact-reference fill: ink at low opacity, so it reads as ground truth
# rather than as a fourth series.
REFERENCE_FILL = "#d8d7d2"

RULE_COLOUR = {
    "gauss_hierarchical": BLUE,
    "gauss_jacobian": ORANGE,
    "fnpe": AQUA,
}

# Distinguishes a truth curve from an estimate at a glance.
TRUTH_KWARGS = {"color": INK, "lw": 2.0, "zorder": 5}
ESTIMATE_LW = 2.0


def apply():
    """Set the shared rcParams. Call once, before any figure is created."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": SECONDARY,
        "lines.solid_capstyle": "round",
        "font.size": 9.5,
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    })


def despine(axis, keep=("left", "bottom")):
    """Recessive frame: keep only the spines that carry a scale."""
    for name, spine in axis.spines.items():
        spine.set_visible(name in keep)


def caption(figure, text, y=-0.01):
    """One explanatory line under a figure, in secondary ink."""
    figure.text(0.5, y, text, ha="center", va="top", fontsize=8.5,
                color=SECONDARY, wrap=True)
