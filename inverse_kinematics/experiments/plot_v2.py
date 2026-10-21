#!/usr/bin/env python3
"""Inverse kinematics at the fast settings, with and without bagging.

Settings: 40 diffusion steps, 12,000 single-observation draws, batched EM on the
device, stage 3 stratified over 250 quantiles, K inferred per observation.

Bagging (``--bag``) averages the composed shared marginal over bootstrap
resamples of the observations. It targets the one thing that fails at large n --
the branch weights collapsing -- and leaves the locals untouched.
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from plots import INK, RAIL, _style_axis  # noqa: E402

SOURCE = ROOT / "artifacts" / "mixture_inference"
RULES = ["gauss_hierarchical", "gauss_jacobian", "langevin_fnpse"]
LABELS = {"gauss_hierarchical": "Gaussian hierarchical",
          "gauss_jacobian": "Gaussian Jacobian",
          "langevin_fnpse": "Langevin F-NPSE"}
COLOURS = {"gauss_hierarchical": "#A6761D", "gauss_jacobian": "#D95F02",
           "langevin_fnpse": "#7570B3"}
PLAIN, BAGGED = "#1B9E77", "#C71585"


def load(n, bag):
    directory = SOURCE / f"bag_n{n}_{bag}"
    stem = f"inverse_kinematics_n{n}"
    with np.load(directory / f"{stem}.npz") as stored:
        payload = {"grid": stored["grid"], "reference": stored["reference"],
                   "curves": {k: stored[f"curve_{k}"] for k in
                              RULES + ["mixture_product"]}}
    payload["table"] = json.loads((directory / f"{stem}.json").read_text())
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--output", type=Path,
                        default=SOURCE / "03_fast_settings.png")
    arguments = parser.parse_args(argv)

    figure, axes = plt.subplots(1, len(arguments.sizes),
                                figsize=(4.4 * len(arguments.sizes), 4.6),
                                layout="constrained", squeeze=False)
    for column, n in enumerate(arguments.sizes):
        plain, bagged = load(n, 0), load(n, 64)
        grid, reference = plain["grid"], plain["reference"]
        positive = float(np.trapezoid(reference[grid >= 0], grid[grid >= 0]))
        axis = axes[0][column]
        axis.plot(grid, reference, color=INK, linewidth=2.3, label="ABC reference")
        for name in RULES:
            axis.plot(grid, plain["curves"][name], color=COLOURS[name],
                      linewidth=1.3, linestyle=(0, (5, 2)),
                      label=f"{LABELS[name]} · "
                            f"{plain['table'][name]['w1_over_sigma']:.3f}")
        axis.plot(grid, plain["curves"]["mixture_product"], color=PLAIN,
                  linewidth=2.0,
                  label=f"mixture · "
                        f"{plain['table']['mixture_product']['w1_over_sigma']:.3f}")
        axis.plot(grid, bagged["curves"]["mixture_product"], color=BAGGED,
                  linewidth=2.0, linestyle=(0, (1, 1)),
                  label=f"mixture + bagging · "
                        f"{bagged['table']['mixture_product']['w1_over_sigma']:.3f}")
        axis.axvline(0.0, color=RAIL, linewidth=0.8, zorder=0)
        _style_axis(axis)
        axis.set_title(f"n = {n}   (reference branch mass {positive:.2f})",
                       fontsize=10, color=INK)
        axis.set_xlabel(r"global $\mathrm{angle}_1$", color=INK)
        axis.set_xlim(-1.6, 1.6)
        stack = [reference] + list(plain["curves"].values()) + [
            bagged["curves"]["mixture_product"]]
        axis.set_ylim(0.0, 1.5 * max(float(np.max(c)) for c in stack))
        if column == 0:
            axis.set_ylabel("density", color=INK)
        axis.legend(frameon=False, fontsize=6.8, loc="upper left")

    figure.suptitle(
        "Inverse kinematics at the fast settings — 40 steps, 12k single-obs "
        "draws, batched EM, stratified locals; K inferred at every n",
        fontsize=12, color=INK)
    figure.savefig(arguments.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
