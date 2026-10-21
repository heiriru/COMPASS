#!/usr/bin/env python3
"""What actually changes with n: the target, the error, and their ratio.

Every table in ``scaling_robustness.py`` reports ``W1 / sigma`` -- the transport
distance between the sampled and exact shared posteriors, divided by the exact
posterior's own standard deviation. That is the right yardstick for "is the
posterior *shape* right", because a fixed absolute error matters more when the
posterior is narrow. But it is a **ratio of two moving quantities**, and reading
it as accuracy alone is misleading: as n grows the exact posterior contracts, so
sigma shrinks and the ratio rises even when the absolute error is flat.

This script separates the two. For each noise law it draws

    row 1   sigma(n)          the exact posterior sd -- the target, contracting
    row 2   W1(n)             the absolute error, in raw units of g
    row 3   W1(n) / sigma(n)  the ratio the tables reported

so row 3 is visibly row 2 divided by row 1, and one can see which factor drives
it. A method whose absolute error is flat while sigma falls still shows a rising
ratio -- it is not getting worse, it is failing to keep up with a target that is
getting harder.

It then draws the posteriors themselves: exact density against the sampled one
at n = 10, 50, 200, on a shared axis, which shows directly whether a method's
error is a displacement, a width error, or both.

Reads ``artifacts/scaling_robustness.csv`` for the curves and re-runs a handful
of cells for the overlays (cheap on the Gaussian). Nothing in ``compass`` or the
other scripts is modified.

Usage:
    python scaling_visualisation.py
    python scaling_visualisation.py --skip-overlays     # curves only, no GPU
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[3]))

import numpy as np  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CSV_PATH = ARTIFACTS / "scaling_robustness.csv"

# (rule, intervention, label, colour)
CURVES = (
    ("lang", "base", "Langevin + F-NPSE", "#4A4A4A"),
    ("jac", "base", "gauss_jacobian, no churn", "#C0C4C8"),
    ("jac", "churn2", "gauss_jacobian + churn2", "#7FBFC0"),
    ("jac", "churn8", "gauss_jacobian + churn8", "#0F8B8D"),
    ("hier", "churn8", "gauss_hierarchical + churn8", "#B0453A"),
)
PROBLEM_LABELS = {
    "gauss": "Gaussian noise",
    "laplace": "Laplace noise",
    "exponential": "Exponential noise (skew + wall)",
}
OVERLAY_METHODS = (
    ("lang", "base", "Langevin + F-NPSE", "#4A4A4A"),
    ("jac", "churn8", "gauss_jacobian + churn8", "#0F8B8D"),
)


def load(path):
    with path.open() as handle:
        return [row for row in csv.DictReader(handle) if row["status"] == "ok"]


def series(rows, problem, rule, intervention):
    selected = sorted(
        (row for row in rows if row["problem"] == problem
         and row["rule"] == rule and row["intervention"] == intervention),
        key=lambda row: int(row["observations"]),
    )
    n = np.array([int(row["observations"]) for row in selected], dtype=float)
    ratio = np.array([float(row["shared_w1_over_sigma"]) for row in selected])
    sigma = np.array([float(row["exact_std"]) for row in selected])
    return n, sigma, ratio * sigma, ratio


def plot_decomposition(rows, problems, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import compare

    compare.configure_style()
    figure, axes = plt.subplots(3, len(problems),
                                figsize=(4.5 * len(problems), 9.6),
                                squeeze=False, constrained_layout=True)
    for column, problem in enumerate(problems):
        # Row 1: the target. Identical for every method, so draw it once.
        n, sigma, _, _ = series(rows, problem, "lang", "base")
        axis = axes[0][column]
        axis.plot(n, sigma, marker="o", ms=5, lw=2, color="black")
        axis.set(xscale="log", yscale="log", xlabel="observations n")
        axis.set_title(PROBLEM_LABELS[problem], fontsize=11)
        if column == 0:
            axis.set_ylabel("1. THE TARGET\nexact posterior sd")
        axis.grid(alpha=0.25)
        axis.annotate("posterior contracts\n(the bar rises)", xy=(0.05, 0.12),
                      xycoords="axes fraction", fontsize=8, color="#555")

        for row_index, (field, label) in enumerate(
            ((2, "2. WHAT WE ACHIEVE\nabsolute error  W1"),
             (3, "3. THE RATIO REPORTED\nW1 / sigma")), start=1
        ):
            axis = axes[row_index][column]
            for rule, intervention, name, colour in CURVES:
                n, sigma, absolute, ratio = series(rows, problem, rule, intervention)
                if n.size == 0:
                    continue
                values = absolute if field == 2 else ratio
                axis.plot(n, values, marker="o", ms=4, lw=1.7, color=colour,
                          label=name if (row_index == 1 and column == 0) else None)
            axis.set(xscale="log", yscale="log", xlabel="observations n")
            if column == 0:
                axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            if field == 3:
                axis.axhline(1.0, color="black", lw=1, ls=":")
    figure.legend(loc="outside lower center", ncol=3, frameon=True)
    figure.suptitle(
        "Why 'accuracy' looks like it degrades: row 3 = row 2 / row 1.\n"
        "The posterior is a moving target -- a flat absolute error still gives "
        "a rising ratio.", fontsize=12,
    )
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def draw_overlay(problem_name, rule_name, intervention_name, count, arguments):
    """Re-run one cell and return the shared draws plus the exact reference."""
    import torch
    import compare
    import symmetric_row_scores as row_scores
    import stochastic_predictor_experiments as churn
    from symmetric_experiments import PROBLEMS, patched_problem_module
    from compass.SDE import VESDE
    from scaling_robustness import RULES, INTERVENTIONS

    specification = PROBLEMS[problem_name]
    module = specification["module"]
    rule = RULES[rule_name]
    intervention = INTERVENTIONS[intervention_name]

    sde = VESDE(sigma=compare.recipe.SDE_KWARGS["sigma"])
    model = row_scores.AnalyticModel(
        sde, specification["row_score"], module, device=arguments.device,
        nodes=769, seed=arguments.seed,
    )
    _, _, x = module.observations(count, arguments.seed)
    grid, density, _, mean, std = module.shared_reference(x)
    problem = {"data": torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1),
               "device": arguments.device}
    sample_kwargs = dict(rule["base"])
    sample_kwargs["correction"] = rule["correction"]
    sample_kwargs.update(rule.get("extra", {}))

    stack = contextlib.ExitStack()
    with stack:
        stack.enter_context(patched_problem_module(module))
        if intervention.get("eta"):
            stack.enter_context(churn.patched_dpm_sampler(
                churn.churned_dpm_sampler(intervention["eta"])))
        samples, _, _ = compare.draw(
            model, problem, sample_kwargs, arguments.num_samples, 100,
            arguments.seed, 5.0,
        )
    return samples[0, :, module.HIERARCHY[0]].numpy(), grid, density, mean, std


def plot_overlays(problem_name, counts, arguments, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import compare

    compare.configure_style()
    figure, axes = plt.subplots(1, len(counts), figsize=(4.6 * len(counts), 4.0),
                                squeeze=False, constrained_layout=True)
    for column, count in enumerate(counts):
        axis = axes[0][column]
        reference = None
        for rule_name, intervention_name, label, colour in OVERLAY_METHODS:
            draws, grid, density, mean, std = draw_overlay(
                problem_name, rule_name, intervention_name, count, arguments)
            if reference is None:
                reference = (grid, density, mean, std)
                axis.fill_between(grid, density, color="#B9C2C9", alpha=0.75,
                                  lw=0, label="exact posterior" if column == 0 else None,
                                  zorder=1)
            kde = gaussian_kde(draws)
            axis.plot(grid, kde(grid), lw=2, color=colour, zorder=3,
                      label=label if column == 0 else None)
        grid, density, mean, std = reference
        axis.axvline(mean, color="black", lw=1, ls="--", zorder=2)
        # Frame every panel on the exact posterior at +-6 sd so the contraction
        # across panels is visible rather than normalised away.
        axis.set_xlim(mean - 6 * std, mean + 6 * std)
        axis.set(xlabel="shared parameter g", title=f"n = {count}   (sd = {std:.3f})")
        if column == 0:
            axis.set_ylabel("density")
        axis.grid(alpha=0.2)
    figure.legend(loc="outside lower center", ncol=3, frameon=True)
    figure.suptitle(
        f"The shared posterior itself, {PROBLEM_LABELS[problem_name]}: "
        "exact (grey) vs sampled.\nEach panel is scaled to its own posterior, "
        "which is 4x narrower at n=200 than at n=10.", fontsize=12,
    )
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["gauss", "laplace", "exponential"])
    parser.add_argument("--overlay-problem", default="gauss")
    parser.add_argument("--overlay-counts", nargs="+", type=int,
                        default=[10, 50, 200])
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--skip-overlays", action="store_true")
    arguments = parser.parse_args()

    rows = load(arguments.csv)
    first = plot_decomposition(rows, arguments.problems,
                               ARTIFACTS / "10_scaling_decomposition.png")
    print(f"Wrote {first}")

    if not arguments.skip_overlays:
        if arguments.device.startswith("cuda"):
            from autocvd import autocvd
            autocvd(num_gpus=1, interval=5)
            arguments.device = "cuda"
        second = plot_overlays(
            arguments.overlay_problem, arguments.overlay_counts, arguments,
            ARTIFACTS / "11_posterior_overlays.png")
        print(f"Wrote {second}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
