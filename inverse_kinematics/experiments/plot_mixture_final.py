#!/usr/bin/env python3
"""The composed global posterior under the recommended mixture settings.

Everything here uses the configuration the follow-up experiments settled on and
nothing else:

    K = 3 fixed (no selection -- no selector beat a fixed K),
    Runnalls **merge** pruning at beam 8 (matches exact enumeration of all K^n
        components, and is flat in the beam),
    EM on 5000 draws with 1 restart (~3 s of CPU; restarts buy nothing and
        draws past ~5000 buy nothing systematic).

Each panel is one dataset. Four curves:

    ABC reference        the ground truth the whole table is graded against.
    Gaussian Jacobian    what the library returns today.
    mixture K = 1        the same product with a *single* Gaussian per
                         observation -- i.e. the Gaussian rules' summary, with
                         no sampler in it. The control that shows the failure is
                         the summary, not the sampler.
    mixture K = 3        the recommended setting.

The final panel summarises W1/sigma across datasets, so the curves and the
number they earn are on the same page.
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
MIXTURE_COLOUR = "#1B9E77"
SINGLE_COLOUR = "#D95F02"
RULE_COLOUR = "#7570B3"


def composed(draws, grid, k, arguments):
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    subset = draws[:, :arguments.em_draws]
    fits = [fit_mixture(subset[j], k, restarts=arguments.em_restarts,
                        seed=arguments.seed + j)
            for j in range(draws.shape[0])]
    return density_on_grid(*(
        (grid,) + sequential_product(fits, prior_precision,
                                     arguments.beam, "merge")[:3]
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--em-draws", type=int, default=5000)
    parser.add_argument("--em-restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = arguments.output or OUTPUT / "02_recommended_settings.png"

    figure, axes = plt.subplots(2, 3, figsize=(14.6, 7.8), layout="constrained")
    scores = {}

    for axis, (dataset, title) in zip(axes.ravel(), DATASETS):
        with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
            grid = stored["grid"]
            reference = stored["joint_reference"]
            draws = stored["single_draws"]
            rule = stored["curve_gauss_jacobian"]
        reference_sd = moments(grid, reference)["sd"]

        def score(curve):
            return w1_grid(grid, curve, reference) / reference_sd

        single = composed(draws, grid, 1, arguments)
        mixture = composed(draws, grid, arguments.k, arguments)
        scores[dataset] = {
            "Gaussian Jacobian": score(rule),
            "mixture K = 1": score(single),
            f"mixture K = {arguments.k}": score(mixture),
        }

        axis.plot(grid, reference, color=INK, linewidth=2.4,
                  label="ABC reference")
        axis.plot(grid, rule, color=RULE_COLOUR, linewidth=1.5,
                  linestyle=(0, (5, 2)),
                  label=f"Gaussian Jacobian · {score(rule):.3f}")
        axis.plot(grid, single, color=SINGLE_COLOUR, linewidth=1.5,
                  linestyle=(0, (2, 2)),
                  label=f"mixture K = 1 · {score(single):.3f}")
        axis.plot(grid, mixture, color=MIXTURE_COLOUR, linewidth=2.1,
                  label=f"mixture K = {arguments.k} · {score(mixture):.3f}")
        axis.axvline(0.0, color=RAIL, linewidth=0.8, zorder=0)
        _style_axis(axis)
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_xlabel(r"shared $\mathrm{angle}_1$", color=INK)
        axis.set_ylabel("density", color=INK)
        axis.set_xlim(grid.min(), grid.max())
        axis.set_ylim(0.0, 1.42 * max(float(np.max(c)) for c in
                                      (reference, rule, single, mixture)))
        axis.legend(frameon=False, fontsize=7.3, loc="upper left")

    # --- summary ------------------------------------------------------------
    axis = axes.ravel()[-1]
    names = list(next(iter(scores.values())))
    colours = {"Gaussian Jacobian": RULE_COLOUR, "mixture K = 1": SINGLE_COLOUR,
               f"mixture K = {arguments.k}": MIXTURE_COLOUR}
    positions = np.arange(len(DATASETS))
    width = 0.26
    for offset, name in enumerate(names):
        axis.bar(positions + (offset - 1) * width,
                 [scores[dataset][name] for dataset, _ in DATASETS],
                 width, color=colours[name], label=name)
    axis.set_xticks(positions)
    axis.set_xticklabels(["sel n8\ns99", "sel n8\ns101", "sel n16\ns99",
                          "prior\nn8", "exact\nzero"], fontsize=8)
    axis.set_ylabel(r"W1 / $\sigma$   (lower is better)", color=INK)
    axis.set_title("summary — the control is the last bar group",
                   fontsize=10, color=INK)
    _style_axis(axis)
    axis.legend(frameon=False, fontsize=7.6, loc="upper left")

    figure.suptitle(
        f"Composed global posterior — K = {arguments.k}, Runnalls merge pruning "
        f"at beam {arguments.beam}, EM on {arguments.em_draws} draws × "
        f"{arguments.em_restarts} restart (~3 s CPU)",
        fontsize=12, color=INK,
    )
    figure.savefig(output, dpi=205, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
