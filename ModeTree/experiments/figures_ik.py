#!/usr/bin/env python
"""Posterior panels and the runtime chart for the inverse-kinematics sweep.

Reads ``artifacts/ik_panels.json`` and ``artifacts/ik_modetree.json`` and draws
one posterior figure per data regime (one panel per n, every method against the
ABC reference) plus a runtime/accuracy figure.

Run:  python ModeTree/experiments/figures_ik.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FIGURES = ARTIFACTS / "figures"
STYLE = {"figure.dpi": 140, "font.size": 9, "axes.grid": True,
         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}
LABELS = {
    "modetree": ("ModeTree-JAC", "#2ca02c", "--"),
    "gauss_jacobian": ("gauss_jacobian", "#5E60CE", "-"),
    "langevin_fnpse": ("Langevin + F-NPSE", "#E76F51", "-"),
}
REGIME_TITLE = {
    "bimodal": "observations selected to |x1| < 0.15 (bimodal shared posterior)",
    "stock": "prior-predictive observations (unimodal shared posterior)",
}


def posterior_figure(panels, regime):
    keys = sorted((k for k in panels if k.startswith(f"{regime}|")),
                  key=lambda k: int(k.split("|")[1]))
    if not keys:
        return False
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.0),
                                    squeeze=False)
        for column, key in enumerate(keys):
            panel = panels[key]
            axis = axes[0][column]
            grid = np.asarray(panel["grid"])
            axis.plot(grid, panel["exact"], color="#22262B", lw=2.0, zorder=5,
                      label="ABC reference")
            for method, curve in panel["curves"].items():
                label, colour, style = LABELS[method]
                axis.plot(grid, curve, color=colour, ls=style, lw=1.3,
                          label=label)
            axis.axvline(panel["truth"], color="#B5179E", ls=":", lw=0.9,
                         label="true angle_1")
            axis.axvline(-panel["truth"], color="#B5179E", ls=":", lw=0.9)
            axis.set_title(f"n = {key.split('|')[1]}", fontsize=10)
            axis.set_xlabel("shared angle_1")
            axis.set_xlim(-1.4, 1.4)
        axes[0][0].set_ylabel("density")
        axes[0][-1].legend(fontsize=6.5, loc="upper right")
        figure.suptitle(f"Inverse kinematics, {REGIME_TITLE.get(regime, regime)}",
                        fontsize=10)
        figure.tight_layout()
        figure.savefig(FIGURES / f"ik_posterior_{regime}.png")
        plt.close(figure)
    return True


def runtime_figure(rows, regimes):
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(regimes) + 1,
                                    figsize=(4.0 * (len(regimes) + 1), 3.2),
                                    squeeze=False)
        for column, regime in enumerate(regimes):
            axis = axes[0][column]
            for method, (label, colour, _) in LABELS.items():
                series = sorted((r["n"], r["seconds"]) for r in rows
                                if r["method"] == method and r["regime"] == regime
                                and "error" not in r)
                if not series:
                    continue
                axis.plot([s[0] for s in series], [s[1] for s in series],
                          marker="o", ms=3.5, color=colour, label=label)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("observations n")
            axis.set_title(regime, fontsize=9)
        axes[0][0].set_ylabel("wall clock (s)")
        axes[0][0].legend(fontsize=7)

        axis = axes[0][-1]
        for method, (label, colour, _) in LABELS.items():
            for regime, style in zip(regimes, ("-", "--")):
                series = sorted((r["n"], r["w1_over_sigma"]) for r in rows
                                if r["method"] == method and r["regime"] == regime
                                and "error" not in r)
                if not series:
                    continue
                axis.plot([s[0] for s in series], [s[1] for s in series],
                          marker="o", ms=3.0, color=colour, ls=style,
                          alpha=0.9 if style == "-" else 0.55)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("observations n")
        axis.set_ylabel("W1 to the ABC reference / reference sd")
        axis.set_title(f"accuracy (solid: {regimes[0]}"
                       + (f", dashed: {regimes[1]}" if len(regimes) > 1 else "")
                       + ")", fontsize=9)
        figure.tight_layout()
        figure.savefig(FIGURES / "ik_runtime.png")
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="ik")
    args = parser.parse_args(argv)
    FIGURES.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / f"{args.tag}_panels.json") as handle:
        panels = json.load(handle)
    with open(ARTIFACTS / f"{args.tag}_modetree.json") as handle:
        rows = json.load(handle)
    regimes = sorted({key.split("|")[0] for key in panels})
    for regime in regimes:
        if posterior_figure(panels, regime):
            print(f"wrote ik_posterior_{regime}.png")
    runtime_figure(rows, regimes)
    print("wrote ik_runtime.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
