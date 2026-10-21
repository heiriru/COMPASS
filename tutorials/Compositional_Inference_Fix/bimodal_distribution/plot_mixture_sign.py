#!/usr/bin/env python3
"""The sign model, where the global posterior really is a Gaussian mixture.

The exact joint posterior of this hierarchy is a ``2^n``-component Gaussian
mixture with shared covariance, in closed form -- so the black reference curve
here is not an estimate, and the object the mixture method builds is the same
kind of object as the truth it is graded against.

Per benchmark: the global posterior, the locals on the line, and cost.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "inverse_kinematics"))
from plots import INK, RAIL, _style_axis  # noqa: E402

SOURCE = HERE / "mixture_inference"
METHODS = ["gauss_hierarchical", "gauss_jacobian", "langevin_fnpse",
           "mixture_product"]
LABELS = {"gauss_hierarchical": "Gaussian hierarchical",
          "gauss_jacobian": "Gaussian Jacobian",
          "langevin_fnpse": "Langevin F-NPSE",
          "mixture_product": "mixture product (K inferred)"}
COLOURS = {"gauss_hierarchical": "#A6761D", "gauss_jacobian": "#D95F02",
           "langevin_fnpse": "#7570B3", "mixture_product": "#1B9E77"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    # One column per n, so the evolution of the shape is read left to right.
    figure, axes = plt.subplots(2, len(arguments.sizes),
                                figsize=(4.3 * len(arguments.sizes), 8.0),
                                layout="constrained", squeeze=False)
    for column, n in enumerate(arguments.sizes):
        with np.load(SOURCE / f"sign_n{n}.npz") as stored:
            grid, reference = stored["grid"], stored["reference"]
            curves = {m: stored[f"curve_{m}"] for m in METHODS}
            locals_ = {m: stored[f"locals_{m}"] for m in METHODS}
            truth = float(stored["global_truth"])
        table = json.loads((SOURCE / f"sign_n{n}.json").read_text())
        chosen = table["mixture_product"]["chosen_k"]

        axis = axes[0][column]
        axis.plot(grid, reference, color=INK, linewidth=2.3,
                  label=f"exact ($2^{{{n}}}$ components)")
        for name in METHODS:
            axis.plot(grid, curves[name], color=COLOURS[name],
                      linewidth=2.0 if name == "mixture_product" else 1.4,
                      linestyle="-" if name == "mixture_product" else (0, (5, 2)),
                      label=f"{LABELS[name]} · {table[name]['w1_over_sigma']:.3f}")
        axis.axvline(truth, color=RAIL, linewidth=1.1, linestyle=(0, (3, 2)),
                     zorder=0, label="true $g$")
        _style_axis(axis)
        axis.set_title(f"n = {n}   (K inferred: "
                       f"{sorted(set(chosen))})", fontsize=10, color=INK)
        axis.set_xlabel("shared parameter $g$", color=INK)
        axis.set_xlim(-5.5, 5.5)
        axis.set_ylim(0.0, 1.5 * max(float(np.max(c)) for c in
                                     [reference] + list(curves.values())))
        if column == 0:
            axis.set_ylabel("density", color=INK)
        axis.legend(frameon=False, fontsize=6.6, loc="upper left")

        # The width ratio is the quantity that moves with n here, so it gets
        # its own row rather than being buried in a legend.
        axis = axes[1][column]
        positions = np.arange(len(METHODS))
        widths = [table[m]["width_ratio"] for m in METHODS]
        axis.bar(positions, widths, 0.6, color=[COLOURS[m] for m in METHODS])
        axis.axhline(1.0, color=INK, linewidth=1.1, linestyle=(0, (3, 2)))
        for position, value in zip(positions, widths):
            axis.text(position, value, f"{value:.3f}", ha="center", va="bottom",
                      fontsize=7.6, color=INK)
        axis.set_xticks(positions)
        axis.set_xticklabels(["gauss\nhier.", "gauss\njac.", "F-NPSE",
                              "mixture"], fontsize=7.2)
        axis.set_ylim(0.0, 1.32)
        _style_axis(axis)
        axis.set_title(f"n = {n} — width / exact width", fontsize=10, color=INK)
        if column == 0:
            axis.set_ylabel("width ratio (1 = correct)", color=INK)

    figure.suptitle(
        "Sign model as n grows — the exact global posterior IS a Gaussian "
        "mixture ($2^n$ components). BIC picks K = 2 at every n, which is what "
        "the hidden sign puts there.", fontsize=12, color=INK)
    output = arguments.output or (SOURCE / "02_sign_scaling.png")
    figure.savefig(output, dpi=205, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
