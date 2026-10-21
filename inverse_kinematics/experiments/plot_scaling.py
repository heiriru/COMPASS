#!/usr/bin/env python3
"""Inverse kinematics as n grows: the global posterior, and the local residuals.

Top row -- the composed global posterior over ``angle_1`` at each n, against
that dataset's own ABC reference.

Bottom row -- the **endpoint residual** ``r = || f(theta) - x_j ||`` for the
locals: the distance from a drawn arm's tip to the observation it was
conditioned on. The forward map is deterministic, so ``r`` needs no reference,
and a method cannot do well on it by accident -- it has to return locals that
actually reproduce the data. Shown as the empirical CDF of ``r`` over every draw
and every observation, so the whole distribution is visible rather than its mean.
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

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
from plots import INK, RAIL, _style_axis  # noqa: E402

SOURCE = ROOT / "artifacts" / "mixture_inference"
METHODS = ["gauss_hierarchical", "gauss_jacobian", "langevin_fnpse",
           "mixture_product"]
LABELS = {"gauss_hierarchical": "Gaussian hierarchical",
          "gauss_jacobian": "Gaussian Jacobian",
          "langevin_fnpse": "Langevin F-NPSE",
          "mixture_product": "mixture product (K inferred)"}
COLOURS = {"gauss_hierarchical": "#A6761D", "gauss_jacobian": "#D95F02",
           "langevin_fnpse": "#7570B3", "mixture_product": "#1B9E77"}


def load(n):
    directory = SOURCE if n == 8 else SOURCE / f"n{n}"
    stem = "inverse_kinematics" if n == 8 else f"inverse_kinematics_n{n}"
    with np.load(directory / f"{stem}.npz") as stored:
        payload = {
            "grid": stored["grid"], "reference": stored["joint_reference"]
            if "joint_reference" in stored else stored["reference"],
            "x": stored["observations"],
            "curves": {m: stored[f"curve_{m}"] for m in METHODS},
            "samples": {m: stored[f"samples_{m}"] for m in METHODS},
        }
    payload["table"] = json.loads((directory / f"{stem}.json").read_text())
    return payload


def residuals(samples, x):
    """``r = || f(theta) - x_j ||`` over every draw and observation."""
    return np.concatenate([
        np.linalg.norm(
            sim.forward(H.to_simulator_order(samples[i, :, :4])) - x[i], axis=1)
        for i in range(samples.shape[0])
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--output", type=Path,
                        default=SOURCE / "02_scaling_with_n.png")
    arguments = parser.parse_args(argv)

    figure, axes = plt.subplots(2, len(arguments.sizes),
                                figsize=(4.3 * len(arguments.sizes), 8.0),
                                layout="constrained", squeeze=False)
    for column, n in enumerate(arguments.sizes):
        data = load(n)
        grid, reference = data["grid"], data["reference"]
        positive = float(np.trapezoid(reference[grid >= 0], grid[grid >= 0]))

        axis = axes[0][column]
        axis.plot(grid, reference, color=INK, linewidth=2.3, label="ABC reference")
        for name in METHODS:
            axis.plot(grid, data["curves"][name], color=COLOURS[name],
                      linewidth=2.0 if name == "mixture_product" else 1.4,
                      linestyle="-" if name == "mixture_product" else (0, (5, 2)),
                      label=f"{LABELS[name]} · "
                            f"{data['table'][name]['w1_over_sigma']:.3f}")
        axis.axvline(0.0, color=RAIL, linewidth=0.8, zorder=0)
        _style_axis(axis)
        axis.set_title(f"n = {n}   (reference branch mass {positive:.2f})",
                       fontsize=10, color=INK)
        axis.set_xlabel(r"global $\mathrm{angle}_1$", color=INK)
        axis.set_xlim(-1.6, 1.6)
        axis.set_ylim(0.0, 1.5 * max(float(np.max(c)) for c in
                                     [reference] + list(data["curves"].values())))
        if column == 0:
            axis.set_ylabel("density", color=INK)
        axis.legend(frameon=False, fontsize=6.6, loc="upper left")

        axis = axes[1][column]
        for name in METHODS:
            r = residuals(data["samples"][name], data["x"])
            order = np.sort(r)
            axis.plot(order, np.arange(1, len(order) + 1) / len(order),
                      color=COLOURS[name],
                      linewidth=2.0 if name == "mixture_product" else 1.4,
                      linestyle="-" if name == "mixture_product" else (0, (5, 2)),
                      label=f"{LABELS[name]} · median {np.median(r):.3f}")
        _style_axis(axis)
        axis.set_xscale("log")
        axis.set_xlim(2e-3, 1.0)
        axis.set_title(f"n = {n} — endpoint residual $r$", fontsize=10, color=INK)
        axis.set_xlabel(r"$r = \|f(\theta) - x_j\|$", color=INK)
        if column == 0:
            axis.set_ylabel("empirical CDF", color=INK)
        axis.legend(frameon=False, fontsize=6.6, loc="upper left")

    figure.suptitle(
        "Inverse kinematics as n grows — global posterior (top) and local "
        "endpoint residuals (bottom). K inferred at every n.",
        fontsize=12, color=INK)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
