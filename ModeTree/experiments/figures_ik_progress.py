#!/usr/bin/env python
"""Posterior panels from whichever IK cells have finished so far.

``experiment_ik.py`` writes its panel file after every cell, so the partial
state of a running ladder is already on disk. This merges the ModeTree run, the
baseline run and the n=4 projection probe into one figure per regime, drawing
only the cells that exist. Cells where ModeTree returned no branches are
labelled as such rather than left silently empty.

Run:  python ModeTree/experiments/figures_ik_progress.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FIGURES = ARTIFACTS / "figures"
# Later tags win on a per-curve basis, so the ModeTree run supplies `modetree`
# and the baseline run supplies the two samplers for the same (regime, n).
TAGS = ("ik_baselines", "ik_modetree", "ik_mt_t003")
LABELS = {
    "modetree": ("ModeTree (t=1e-3)", "#2ca02c", "--"),
    "modetree_t003": ("ModeTree (t=0.03)", "#8BC34A", ":"),
    "gauss_jacobian": ("gauss_jacobian", "#5E60CE", "-"),
    "langevin_fnpse": ("Langevin + F-NPSE", "#E76F51", "-"),
}
STYLE = {"figure.dpi": 140, "font.size": 9, "axes.grid": True,
         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}


def load():
    panels, rows = {}, []
    for tag in TAGS:
        try:
            with open(ARTIFACTS / f"{tag}_panels.json") as handle:
                for key, panel in json.load(handle).items():
                    slot = panels.setdefault(
                        key, {"grid": panel["grid"], "exact": panel["exact"],
                              "truth": panel["truth"], "curves": {}})
                    curves = dict(panel["curves"])
                    if tag == "ik_mt_t003" and "modetree" in curves:
                        # t = 0.03 is a different terminal time from the
                        # default 1e-3 run, so it is a different method and
                        # gets its own curve rather than overwriting.
                        curves["modetree_t003"] = curves.pop("modetree")
                    slot["curves"].update(curves)
            with open(ARTIFACTS / f"{tag}_modetree.json") as handle:
                rows.extend(json.load(handle))
        except FileNotFoundError:
            continue
    return panels, rows


def figure(panels, rows, regime):
    keys = sorted((k for k in panels if k.startswith(f"{regime}|")),
                  key=lambda k: int(k.split("|")[1]))
    if not keys:
        return False
    failures = {(r["regime"], r["n"]) for r in rows
                if r["method"] == "modetree" and "error" in r}
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.1),
                                    squeeze=False)
        for column, key in enumerate(keys):
            panel = panels[key]
            axis = axes[0][column]
            grid = np.asarray(panel["grid"])
            axis.plot(grid, panel["exact"], color="#22262B", lw=2.0, zorder=5,
                      label="ABC reference")
            for method, curve in panel["curves"].items():
                label, colour, style = LABELS[method]
                axis.plot(grid, curve, color=colour, ls=style, lw=1.4,
                          label=label, zorder=4 if method == "modetree" else 3)
            axis.axvline(panel["truth"], color="#B5179E", ls=":", lw=0.9)
            axis.axvline(-panel["truth"], color="#B5179E", ls=":", lw=0.9)
            n = int(key.split("|")[1])
            axis.set_title(f"n = {n}", fontsize=10)
            axis.set_xlabel("shared angle_1")
            axis.set_xlim(-1.2, 1.2)
            # Label only cells where *no* ModeTree configuration produced a
            # curve; a cell where one terminal time failed and another
            # succeeded is not a failed cell.
            if (regime, n) in failures and not any(
                    m.startswith("modetree") for m in panel["curves"]):
                axis.text(0.5, 0.94, "ModeTree: no branches", ha="center",
                          va="top", transform=axis.transAxes, fontsize=7.5,
                          color="#2ca02c")
        axes[0][0].set_ylabel("density")
        axes[0][-1].legend(fontsize=6.5, loc="upper right")
        figure.suptitle(
            f"Inverse kinematics, {regime} regime "
            "(dotted: +-true angle_1)", fontsize=10)
        figure.tight_layout()
        figure.savefig(FIGURES / f"ik_final_{regime}.png")
        plt.close(figure)
    return True


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    panels, rows = load()
    for regime in sorted({key.split("|")[0] for key in panels}):
        if figure(panels, rows, regime):
            print(f"wrote ik_final_{regime}.png")


if __name__ == "__main__":
    main()
