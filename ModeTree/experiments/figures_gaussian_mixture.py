#!/usr/bin/env python
"""Posterior panels and the runtime chart for the Gaussian-mixture sweep.

Reads ``artifacts/gaussian_mixture_panels.json`` and
``artifacts/gaussian_mixture_modetree.json`` (written by
``experiment_gaussian_mixture.py``) and draws

* one figure per score source: the exact tall posterior against every method,
  one panel per n;
* one runtime figure: seconds against n for every method and both sources.

Run:  python ModeTree/experiments/figures_gaussian_mixture.py
"""

import os
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
STYLE = {"figure.dpi": 140, "font.size": 9, "axes.grid": True,
         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}
LABELS = {
    "modetree": ("ModeTree-JAC", "#2ca02c", "--"),
    "gauss_jacobian": ("gauss_jacobian", "#5E60CE", "-"),
    "gauss_jacobian_mixture_prior": ("gauss_jacobian + mixture prior",
                                     "#B5179E", "-"),
    "langevin_fnpse": ("Langevin + F-NPSE", "#E76F51", "-"),
}
SOURCE_TITLE = {"oracle": "analytic (exact diffused) score",
                "learned": "learned COMPASS score network"}


def posterior_figure(panels, source):
    keys = sorted((k for k in panels if k.startswith(f"{source}|")),
                  key=lambda k: int(k.split("|")[1]))
    if not keys:
        return
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.0),
                                    squeeze=False)
        for column, key in enumerate(keys):
            panel = panels[key]
            axis = axes[0][column]
            grid = np.asarray(panel["grid"])
            axis.plot(grid, panel["exact"], color="k", lw=2.0, zorder=5,
                      label="exact tall posterior")
            for method, curve in panel["curves"].items():
                label, colour, style = LABELS[method]
                axis.plot(grid, curve, color=colour, ls=style, lw=1.3,
                          label=label)
            for mode in panel["exact_means"]:
                axis.axvline(mode, color="k", ls=":", lw=0.7)
            axis.set_title(f"n = {key.split('|')[1]}", fontsize=10)
            axis.set_xlabel("global parameter g")
            lo = min(panel["exact_means"]) - 2.0
            hi = max(panel["exact_means"]) + 2.0
            axis.set_xlim(lo, hi)
        axes[0][0].set_ylabel("density")
        axes[0][-1].legend(fontsize=6.5, loc="upper right")
        figure.suptitle(
            f"Gaussian-mixture benchmark, {SOURCE_TITLE[source]} "
            "(dotted: exact branch means)", fontsize=10)
        figure.tight_layout()
        figure.savefig(FIGURES / f"gaussian_mixture_posterior_{source}.png")
        plt.close(figure)


def runtime_figure(rows):
    with plt.rc_context(STYLE):
        sources = sorted({r["score_source"] for r in rows})
        figure, axes = plt.subplots(1, len(sources) + 1,
                                    figsize=(4.0 * (len(sources) + 1), 3.2),
                                    squeeze=False)
        for column, source in enumerate(sources):
            axis = axes[0][column]
            for method, (label, colour, _) in LABELS.items():
                series = sorted(
                    (r["n"], r["seconds"]) for r in rows
                    if r["method"] == method and r["score_source"] == source
                    and "error" not in r)
                if not series:
                    continue
                axis.plot([s[0] for s in series], [s[1] for s in series],
                          marker="o", ms=3.5, color=colour, label=label)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("n")
            axis.set_title(SOURCE_TITLE[source], fontsize=9)
        axes[0][0].set_ylabel("wall clock (s)")
        axes[0][0].legend(fontsize=7)

        axis = axes[0][-1]
        for method, (label, colour, _) in LABELS.items():
            series = sorted(
                (r["n"], r["w1"]) for r in rows
                if r["method"] == method and r["score_source"] == "learned"
                and "error" not in r)
            if not series:
                continue
            axis.plot([s[0] for s in series], [s[1] for s in series],
                      marker="o", ms=3.5, color=colour, label=label)
            series = sorted(
                (r["n"], r["w1"]) for r in rows
                if r["method"] == method and r["score_source"] == "oracle"
                and "error" not in r)
            if series:
                axis.plot([s[0] for s in series], [s[1] for s in series],
                          marker="s", ms=3.0, color=colour, ls="--", alpha=0.6)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("n")
        axis.set_ylabel("Wasserstein-1 to the exact posterior")
        axis.set_title("accuracy (solid: learned, dashed: analytic)", fontsize=9)
        figure.tight_layout()
        figure.savefig(FIGURES / "gaussian_mixture_runtime.png")
        plt.close(figure)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "gaussian_mixture_panels.json") as handle:
        panels = json.load(handle)
    with open(ARTIFACTS / "gaussian_mixture_modetree.json") as handle:
        rows = json.load(handle)
    for source in sorted({key.split("|")[0] for key in panels}):
        posterior_figure(panels, source)
        print(f"wrote gaussian_mixture_posterior_{source}.png")
    runtime_figure(rows)
    print("wrote gaussian_mixture_runtime.png")


if __name__ == "__main__":
    main()
