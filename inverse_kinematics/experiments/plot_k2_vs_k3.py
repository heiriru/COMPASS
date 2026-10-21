#!/usr/bin/env python3
"""K = 2 against K = 3, including the instability that decided the default.

Panels 1-5 are the composed global posterior on each dataset, at the recommended
settings (Runnalls merge pruning at beam 8, EM on 5000 draws x 1 restart), with
K = 2 and K = 3 drawn over the ABC reference.

Panel 6 is the reason K = 3 is the default. At n = 16, K = 2 is **bistable**: EM
finds one of two solutions depending on how many draws it is given, they differ
only slightly per observation, and after composing 16 of them they are 30 times
apart. Both are drawn, from the draw counts that produce them, with K = 3 on the
same axes for contrast. More restarts do not rescue the bad branch -- they make
EM find it more reliably, because it is the better *per-observation* fit.
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
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
from plots import INK, RAIL, _style_axis  # noqa: E402
from composition_attribution import moments, w1_grid  # noqa: E402
from mixture_product import density_on_grid, fit_mixture  # noqa: E402
from mixture_refinements import sequential_product  # noqa: E402

ATTRIBUTION = ROOT / "artifacts" / "composition_attribution"
OUTPUT = ROOT / "artifacts" / "mixture_product"

DATASETS = [
    ("selected_n8_seed99", "selected, n = 8, seed 99"),
    ("selected_n8_seed101", "selected, n = 8, seed 101"),
    ("selected_n16_seed99", "selected, n = 16, seed 99"),
    ("prior_n8_seed3", "prior predictive, n = 8"),
    ("exact_zero_n8_seed99", "exactly symmetric control, n = 8"),
]
K2_COLOUR = "#D95F02"
K3_COLOUR = "#1B9E77"
BAD_COLOUR = "#C1121F"


def composed(draws, grid, k, count, restarts, beam, seed):
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    fits = [fit_mixture(draws[j, :count], k, restarts=restarts, seed=seed + j)
            for j in range(draws.shape[0])]
    return density_on_grid(*(
        (grid,) + sequential_product(fits, prior_precision, beam, "merge")[:3]
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--em-draws", type=int, default=5000)
    parser.add_argument("--em-restarts", type=int, default=1)
    parser.add_argument("--stable-draws", type=int, default=2000,
                        help="draw count at which K=2 finds its good solution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = arguments.output or OUTPUT / "03_k2_vs_k3.png"
    figure, axes = plt.subplots(2, 3, figsize=(14.6, 7.8), layout="constrained")

    for axis, (dataset, title) in zip(axes.ravel(), DATASETS):
        with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
            grid = stored["grid"]
            reference = stored["joint_reference"]
            draws = stored["single_draws"]
        reference_sd = moments(grid, reference)["sd"]

        def score(curve):
            return w1_grid(grid, curve, reference) / reference_sd

        curves = {k: composed(draws, grid, k, arguments.em_draws,
                              arguments.em_restarts, arguments.beam,
                              arguments.seed)
                  for k in (2, 3)}
        axis.plot(grid, reference, color=INK, linewidth=2.4,
                  label="ABC reference")
        axis.plot(grid, curves[2], color=K2_COLOUR, linewidth=1.7,
                  linestyle=(0, (4, 2)),
                  label=f"K = 2 · {score(curves[2]):.3f}")
        axis.plot(grid, curves[3], color=K3_COLOUR, linewidth=2.0,
                  label=f"K = 3 · {score(curves[3]):.3f}")
        axis.axvline(0.0, color=RAIL, linewidth=0.8, zorder=0)
        _style_axis(axis)
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_ylim(0.0, 1.4 * max(float(np.max(c)) for c in
                                     (reference, curves[2], curves[3])))
        axis.legend(frameon=False, fontsize=7.8, loc="upper left")

    # --- the instability, drawn ---------------------------------------------
    axis = axes.ravel()[-1]
    with np.load(ATTRIBUTION / "selected_n16_seed99" / "curves.npz") as stored:
        grid = stored["grid"]
        reference = stored["joint_reference"]
        draws = stored["single_draws"]
    reference_sd = moments(grid, reference)["sd"]

    def score(curve):
        return w1_grid(grid, curve, reference) / reference_sd

    good = composed(draws, grid, 2, arguments.stable_draws, 4,
                    arguments.beam, arguments.seed)
    bad = composed(draws, grid, 2, arguments.em_draws, 4,
                   arguments.beam, arguments.seed)
    stable = composed(draws, grid, 3, arguments.em_draws, 4,
                      arguments.beam, arguments.seed)
    axis.plot(grid, reference, color=INK, linewidth=2.4, label="ABC reference")
    axis.plot(grid, good, color=K2_COLOUR, linewidth=1.7, linestyle=(0, (4, 2)),
              label=f"K = 2, {arguments.stable_draws} draws · {score(good):.3f}")
    axis.plot(grid, bad, color=BAD_COLOUR, linewidth=2.0,
              label=f"K = 2, {arguments.em_draws} draws · {score(bad):.3f}")
    axis.plot(grid, stable, color=K3_COLOUR, linewidth=2.0,
              label=f"K = 3, {arguments.em_draws} draws · {score(stable):.3f}")
    axis.axvline(0.0, color=RAIL, linewidth=0.8, zorder=0)
    _style_axis(axis)
    axis.set_title("K = 2 is bistable at n = 16 — both solutions shown",
                   fontsize=10, color=INK)
    axis.set_ylim(0.0, 1.4 * max(float(np.max(c)) for c in
                                 (reference, good, bad, stable)))
    axis.legend(frameon=False, fontsize=7.6, loc="upper left")

    for axis in axes.ravel():
        axis.set_xlabel(r"shared $\mathrm{angle}_1$", color=INK)
        axis.set_ylabel("density", color=INK)
        axis.set_xlim(grid.min(), grid.max())

    figure.suptitle(
        "K = 2 against K = 3 — merge pruning at beam 8, EM on "
        f"{arguments.em_draws} draws; legend values are W1/σ (lower is better)",
        fontsize=12, color=INK,
    )
    figure.savefig(output, dpi=205, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
