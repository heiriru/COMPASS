#!/usr/bin/env python3
"""The corrector-free churned DPM2, on symmetric-noise twins of the hierarchy.

``stochastic_predictor_experiments.py`` established that DPM2 with churn
(``sigma_hat = sigma_i + eta (sigma_i - sigma_{i+1})``, no corrector steps)
reaches the exact shared posterior of the *exponential* hierarchy, whose tall
posterior is sharply skewed and pressed against a hard wall at ``min_j x_j``.
Two questions that leaves open, and this script answers:

1. Does the result survive when the posterior is **symmetric** -- i.e. was the
   original failure a property of the skew and the wall?
2. Does it hold for **both** composition rules, or only for ``gauss_jacobian``?
   ``artifacts/oracle_diagnostics/exact_curl.csv`` puts ``gauss_hierarchical``'s
   relative Jacobian antisymmetry at ~0.92 at every noise level against ~0.09 for
   ``gauss_jacobian``, and a Langevin-type term has no guaranteed invariant
   measure on a field carrying that much curl.

Two twins, because "symmetric" and "Gaussian" are different changes
-------------------------------------------------------------------
``hierarchy_gauss``   Normal noise. Symmetric, no wall, **and** Gaussian -- so
                      the Gaussian/Tweedie approximation every composition rule
                      rests on is *exact at every noise level*. Both rules should
                      be right, and whatever error remains is the sampler, the
                      pilot's Monte-Carlo noise, or the KDE. The control.
``hierarchy_laplace`` Laplace noise: the symmetrized exponential. Symmetric and
                      no wall, but still **non-Gaussian**, so the rules stay
                      approximate. This is the twin that isolates skew from
                      non-Gaussianity.

Both use ``Var(eps) = Var(eta) = 1``, matching ``Exponential(rate = 1)``, so
W1/sigma numbers are readable against the exponential table. Both are graded by
``validate_symmetric.py`` against brute-force 2-D quadrature, Monte-Carlo moments
and SBC before anything here runs.

Nothing in ``compass``, ``compare.py``, ``analytic_compare.py`` or
``hierarchy.py`` is modified. The problem module is swapped into ``compare``'s
namespace for the duration of a run, the churn predictor is patched onto
``MultiObsSampler`` exactly as in ``stochastic_predictor_experiments.py``, and
both are restored afterwards.

Usage:
    python symmetric_experiments.py --problem gauss laplace --device cuda
    python symmetric_experiments.py --problem laplace --variants C_eta4 --device cuda
    python symmetric_experiments.py --replot
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
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare  # noqa: E402
import hierarchy_gauss  # noqa: E402
import hierarchy_laplace  # noqa: E402
import stochastic_predictor_experiments as churn  # noqa: E402
import symmetric_row_scores as row_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

PROBLEMS = {
    "gauss": {
        "module": hierarchy_gauss,
        "label": "Gaussian noise (symmetric, and Gaussian: rules exact)",
        "row_score": row_scores.GaussianRowScoreNetwork,
    },
    "laplace": {
        "module": hierarchy_laplace,
        "label": "Laplace noise (symmetric, non-Gaussian: rules approximate)",
        "row_score": row_scores.LaplaceRowScoreNetwork,
    },
}

NAVY, BLUE, TEAL, CORAL, GOLD = (
    compare.NAVY, compare.BLUE, compare.TEAL, compare.CORAL, compare.GOLD
)
MUTED = "#8A8F98"

JACOBIAN_KWARGS = {
    "method": "dpm", "order": 2, "correction": "gauss_jacobian",
    "corrector_steps": 0, "final_corrector_steps": 0,
    "terminal_corrector_steps": 0,
}
HIERARCHICAL_KWARGS = {
    "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
    "precision_est_samples": compare.PRECISION_EST_SAMPLES,
    "precision_est_timesteps": compare.TIMESTEPS,
    "corrector_steps": 0, "final_corrector_steps": 0,
    "terminal_corrector_steps": 0,
}

VARIANTS = {
    # gauss_hierarchical with its constant pilot Lambda_j replaced by the
    # state-dependent kernel estimate of the same object (kernel_curvature.py).
    # Same rule, same pilot, no extra network evaluations.
    "K_eta4": {"label": "gauss_hierarchical + kernel curvature, eta 4",
               "eta": 4.0, "kwargs": HIERARCHICAL_KWARGS, "colour": "#7F4FC9",
               "kernel": True},
    "K_eta0": {"label": "gauss_hierarchical + kernel curvature, eta 0",
               "eta": 0.0, "kwargs": HIERARCHICAL_KWARGS, "colour": "#B8A1DC",
               "kernel": True},
    # Sigma_t,j measured by the law of total covariance instead of assumed
    # Gaussian (tweedie_pilot.py). Reduces to gauss_hierarchical algebraically
    # on a Gaussian problem, so the Gaussian twin is its identity test.
    "T_eta4": {"label": "gauss_hierarchical + measured Sigma_t, eta 4",
               "eta": 4.0, "kwargs": HIERARCHICAL_KWARGS, "colour": "#0F8B8D",
               "tweedie": True},
    "T_eta0": {"label": "gauss_hierarchical + measured Sigma_t, eta 0",
               "eta": 0.0, "kwargs": HIERARCHICAL_KWARGS, "colour": "#7FBFC0",
               "tweedie": True},
    "J_eta0": {"label": "gauss_jacobian, eta 0 (probability-flow ODE)",
               "eta": 0.0, "kwargs": JACOBIAN_KWARGS, "colour": MUTED},
    "J_eta4": {"label": "gauss_jacobian, eta 4 (churned, corrector-free)",
               "eta": 4.0, "kwargs": JACOBIAN_KWARGS, "colour": TEAL},
    "H_eta0": {"label": "gauss_hierarchical, eta 0 (probability-flow ODE)",
               "eta": 0.0, "kwargs": HIERARCHICAL_KWARGS, "colour": "#B0453A"},
    "H_eta4": {"label": "gauss_hierarchical, eta 4 (churned, corrector-free)",
               "eta": 4.0, "kwargs": HIERARCHICAL_KWARGS, "colour": CORAL},
}


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched_problem_module(module):
    """Point ``compare``'s ``hierarchy`` name at a different problem module.

    ``compare.py`` reaches the problem only through ``hierarchy.<name>``, so
    rebinding that one attribute redirects every reference in ``evaluate``,
    ``kde_map_pipeline`` and ``conditional_local_ascent`` at once. ``hierarchy``
    itself is untouched, and so is every other importer of it.
    """
    original = compare.hierarchy
    compare.hierarchy = module
    try:
        yield
    finally:
        compare.hierarchy = original


def build_problem(module, model, observations, seed, device):
    """``compare.build_problem`` for a symmetric twin.

    Reimplemented rather than reused for one reason: ``compare.build_problem``
    ends the local grid at ``max_j x_j``, which is exactly right for the
    exponential hierarchy (``l_j <= x_j`` there) and wrong here, where ``l_j`` is
    centred on ``(g + x_j)/2`` with symmetric spread and reaches past ``x_j``.
    Clipping it would silently bias every local W1.
    """
    global_truth, local_truth, x = module.observations(observations, seed)
    grid, density, weights, mean, std = module.shared_reference(x)
    local_mean, local_std = module.local_reference(x, grid, weights)
    # Cover every local marginal to +-8 of its own sd, on both sides.
    low = float((local_mean - 8.0 * local_std).min())
    high = float((local_mean + 8.0 * local_std).max())
    local_grid = np.linspace(min(low, float(grid[0])), high, 20001)
    local_weights = np.stack([
        module.local_marginal(value, grid, weights, local_grid) for value in x
    ])
    return {
        "model": model, "device": device, "observations": observations,
        "x": np.asarray(x, dtype=np.float64),
        "data": torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1),
        "global_truth": global_truth,
        "local_truth": np.asarray(local_truth, dtype=np.float64),
        "global_grid": grid, "global_density": density, "global_weights": weights,
        "global_mean": mean, "global_std": std,
        "global_mode": module.grid_mode(grid, density),
        "local_mean": local_mean, "local_std": local_std,
        "local_grid": local_grid, "local_weights": local_weights,
        "config": "analytic", "train_samples": 0,
    }


def run_variant(entry, module, problem, model, arguments):
    from compass.MultiObsSampler import MultiObsSampler
    import kernel_curvature

    builder = churn.churned_dpm_sampler(entry["eta"])
    record = [] if entry.get("kernel") else None
    stack = contextlib.ExitStack()
    with stack:
        stack.enter_context(churn.patched_dpm_sampler(builder))
        if entry.get("kernel"):
            stack.enter_context(kernel_curvature.kernel_curvature(
                MultiObsSampler, pilot_draws=arguments.kernel_pilot_draws,
                record=record,
            ))
        if entry.get("tweedie"):
            import tweedie_pilot
            stack.enter_context(tweedie_pilot.tweedie_pilot(
                MultiObsSampler, pilot_draws=arguments.kernel_pilot_draws,
                seed=arguments.seed,
            ))
        samples, sampler, runtime = compare.draw(
            model, problem, dict(entry["kwargs"]), arguments.num_samples,
            arguments.timesteps, arguments.seed, arguments.denoise_clamp,
        )
    if record:
        print("  kernel effective sample fraction by log10(lambda): " + ", ".join(
            f"1e{key}: {value:.3f} ({count})"
            for key, value, count in kernel_curvature.diagnostics(record)
        ))
    result = compare.kde_map_pipeline(model, problem, samples, arguments)
    metrics = compare.evaluate(problem, result, sampler, runtime, [])
    return result, metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_method(problem, module, name, entry, result, metrics, title, output):
    """The four-panel layout of ``compare.plot_method``, without the wall.

    Written out rather than reused because ``compare.plot_method`` draws and
    labels a vertical line at ``min_j x_j`` -- the hard wall of the exponential
    hierarchy. Neither symmetric twin has one, and a figure that claims a
    boundary that is not there is worse than no figure.
    """
    x = problem["x"]
    n = problem["observations"]
    indices = np.arange(n)
    local_draw_mean = result["kept_local"].mean(axis=1)
    local_draw_std = result["kept_local"].std(axis=1, ddof=1)

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.5))
    fig.suptitle(title, fontsize=13.0, fontweight="bold", color=NAVY)

    # 1. the observations. A strip plot: the vertical offsets separate the
    # markers and carry no meaning, so the axis is left unlabelled rather than
    # borrowing the exponential layout's "stacked per bin", which it is not.
    axes[0].plot(x, np.linspace(0.4, 0.6, n), "o", color=BLUE, ms=7, alpha=0.75)
    axes[0].axvline(float(np.mean(x)), color="red", ls=":", lw=1.8,
                    label="mean observation")
    axes[0].axvline(problem["global_mean"], color=NAVY, ls="--", lw=1.5,
                    label="E[g | x]")
    axes[0].set(xlabel="observed x", ylim=(0.3, 0.7),
                title=f"{n} observations (spread vertically for legibility)")
    axes[0].set_yticks([])
    axes[0].legend(fontsize=8)

    # 2. the shared posterior
    axes[1].hist(result["kept_shared"], bins=70, density=True, color=BLUE,
                 alpha=0.55, label="COMPASS draws")
    axes[1].plot(problem["global_grid"], problem["global_density"], "k--", lw=2.2,
                 label="exact")
    axes[1].plot(result["kde_grid"], result["kde_density"], color=GOLD, lw=2.0,
                 label="KDE of draws")
    axes[1].axvline(problem["global_truth"], color="red", ls=":", lw=1.8,
                    label="true global parameter")
    axes[1].axvline(result["global_map"], color=TEAL, ls="--", lw=1.8,
                    label="KDE MAP (this method)")
    axes[1].set_xlim(problem["global_mean"] - 5 * problem["global_std"],
                     problem["global_mean"] + 5 * problem["global_std"])
    axes[1].set(xlabel="global parameter g", ylabel="density",
                title="Shared posterior")
    axes[1].legend(fontsize=8)

    # 3. the local posteriors
    axes[2].errorbar(indices, local_draw_mean, yerr=local_draw_std, fmt="o",
                     color=CORAL, ms=4, elinewidth=1.4, alpha=0.75,
                     label="joint draws, mean ± σ")
    axes[2].plot(indices, problem["local_mean"], "_", color=NAVY, ms=13,
                 markeredgewidth=2.0, label="exact mean")
    axes[2].plot(indices, result["local_map"], "d", color=TEAL, ms=6,
                 label="ascent MAP at g = ĝ")
    axes[2].set(xlabel="observation", ylabel="local parameter ℓᵢ",
                title=f"{n} local posteriors")
    axes[2].legend(fontsize=8)

    # 4. local recovery
    axes[3].scatter(problem["local_mean"], result["local_map"], c=indices,
                    cmap="viridis", s=45, zorder=5)
    limits = [float(min(problem["local_mean"].min(), result["local_map"].min())),
              float(max(problem["local_mean"].max(), result["local_map"].max()))]
    pad = 0.05 * (limits[1] - limits[0])
    axes[3].plot([limits[0] - pad, limits[1] + pad],
                 [limits[0] - pad, limits[1] + pad], "k--", lw=1.6)
    axes[3].text(
        0.03, 0.95,
        f"mean |error| = {metrics['local_map_error_vs_posterior_mean_sigma']:.2f}"
        f" analytic σ\nvs exact p(ℓ|ĝ,x) mode: "
        f"{metrics['local_map_error_vs_conditional_mode_sigma']:.2f} σ",
        transform=axes[3].transAxes, va="top", fontsize=8.5, color=NAVY,
    )
    axes[3].set(xlabel="exact local posterior mean", ylabel="ascent MAP at g = ĝ",
                title="Local recovery")

    fig.text(0.5, -0.02,
             f"shared W1 {metrics['shared_w1_over_sigma']:.4f} σ   |   "
             f"width ratio {metrics['shared_width_ratio']:.3f}   |   "
             f"local W1 {metrics['local_w1_over_sigma']:.4f} σ   |   "
             f"{metrics['sample_seconds']:.0f} s, "
             f"{metrics['sampler_network_calls']} score calls",
             ha="center", fontsize=9.5, color=NAVY)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def plot_cross_problem_summary(artifacts, output):
    """The whole point, on one axis: three noise laws x two rules x churn on/off.

    Reads the exponential arms from ``stochastic_predictor_experiments.csv`` and
    the two symmetric twins from ``symmetric_experiments.csv``, so the figure is
    always consistent with the tables rather than with a remembered number.
    """
    exponential = {}
    path = artifacts / "stochastic_predictor_experiments.csv"
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                exponential[row["method"]] = row
    symmetric = {}
    path = artifacts / "symmetric_experiments.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            symmetric[(row["problem"], row["method"])] = row

    # (column label, is-non-Gaussian, {arm key: row})
    columns = [
        ("Gaussian\n(symmetric, Gaussian)", False, {
            "J_eta0": symmetric.get(("gauss", "J_eta0")),
            "J_eta4": symmetric.get(("gauss", "J_eta4")),
            "H_eta0": symmetric.get(("gauss", "H_eta0")),
            "H_eta4": symmetric.get(("gauss", "H_eta4")),
        }),
        ("Laplace\n(symmetric, non-Gaussian)", True, {
            "J_eta0": symmetric.get(("laplace", "J_eta0")),
            "J_eta4": symmetric.get(("laplace", "J_eta4")),
            "H_eta0": symmetric.get(("laplace", "H_eta0")),
            "H_eta4": symmetric.get(("laplace", "H_eta4")),
        }),
        ("Exponential\n(skewed, hard wall)", True, {
            "J_eta0": exponential.get("P0_baseline"),
            "J_eta4": exponential.get("C_eta4"),
            "H_eta0": exponential.get("H_eta0"),
            "H_eta4": exponential.get("H_eta4"),
        }),
    ]
    arms = [
        ("J_eta0", "gauss_jacobian, ODE", MUTED, "//"),
        ("J_eta4", "gauss_jacobian, churn η=4", TEAL, None),
        ("H_eta0", "gauss_hierarchical, ODE", "#B0453A", "//"),
        ("H_eta4", "gauss_hierarchical, churn η=4", CORAL, None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))
    fig.suptitle(
        "Corrector-free churned DPM2 across three noise laws\n"
        "(exact single-observation score throughout; W1 of the marginal / "
        "posterior σ, log scale; 3,000 draws, 100 steps)",
        fontsize=13.5, fontweight="bold", color=NAVY,
    )
    width = 0.2
    for axis, field, title in (
        (axes[0], "shared_w1_over_sigma", "Shared posterior"),
        (axes[1], "local_w1_over_sigma", "Local posteriors (mean over 30)"),
    ):
        base = np.arange(len(columns))
        for index, (key, label, colour, hatch) in enumerate(arms):
            values = [
                float(entry[key][field]) if entry[key] else np.nan
                for _, _, entry in columns
            ]
            offset = base + (index - 1.5) * width
            axis.bar(offset, values, width * 0.92, color=colour, hatch=hatch,
                     edgecolor="white", linewidth=1.0,
                     label=label if axis is axes[0] else None)
            for position, value in zip(offset, values):
                if np.isfinite(value):
                    axis.text(position, value * 1.10, f"{value:.3f}", ha="center",
                              fontsize=7.0, color=NAVY, rotation=90)
        # The sampling floor: W1 of 3,000 exact draws against their own law.
        axis.axhline(0.0235, color=NAVY, ls=":", lw=1.4)
        axis.text(-0.48, 0.0175, "Monte-Carlo floor (n = 3,000)",
                  fontsize=7.5, color=NAVY, ha="left", va="top")
        axis.set_yscale("log")
        axis.set_ylim(0.008, 20.0)
        axis.set_xticks(base)
        axis.set_xticklabels([name for name, _, _ in columns], fontsize=9)
        axis.set(ylabel="W1 / posterior σ", title=title)
        axis.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8.5, loc="upper left", ncol=2)

    fig.text(
        0.5, -0.04,
        "Removing the corrector steps costs 1.4-4.7 σ on every noise law, so the "
        "failure is not the skew or the wall — it is the initial law, which a "
        "deterministic map cannot forget.\nChurn repairs all six cases. The two "
        "rules are indistinguishable only where the problem is Gaussian and the "
        "Gaussian composition is therefore exact.",
        ha="center", fontsize=9.5, color=NAVY,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


FIELDS = ["problem", "method", "label", "eta", "correction", "seconds", "calls",
          "shared_w1_over_sigma", "shared_mean_error_sigma", "shared_width_ratio",
          "local_w1_over_sigma", "marginal_score_rel_rmse", "global_map",
          "global_map_error_sigma", "local_map_error_vs_posterior_mean_sigma"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", nargs="+", default=list(PROBLEMS),
                        choices=list(PROBLEMS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--observations", type=int, default=compare.OBSERVATIONS)
    parser.add_argument("--num-samples", type=int, default=compare.NUM_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=compare.TIMESTEPS)
    parser.add_argument("--quadrature-nodes", type=int, default=769)
    parser.add_argument("--kernel-pilot-draws", type=int, default=512,
                        help="pilot draws retained for the kernel curvature "
                             "estimate; its cost is linear in this count and "
                             "its effective sample size is bounded by it.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--excursion-sigma", type=float, default=15.0)
    parser.add_argument("--kde-bandwidth", default=None)
    parser.add_argument("--map-timesteps", type=int, default=200)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="redraw 07_symmetric_summary.png from the CSVs "
                             "alone; skips rebuilding the problems, whose "
                             "Laplace local marginals cost minutes.")
    arguments = parser.parse_args()

    from compass.SDE import VESDE

    compare.configure_style()
    if arguments.summary_only:
        plot_cross_problem_summary(
            arguments.artifacts, arguments.artifacts / "07_symmetric_summary.png"
        )
        return
    rows = []
    for problem_name in arguments.problem:
        specification = PROBLEMS[problem_name]
        module = specification["module"]
        sde = VESDE(sigma=compare.recipe.SDE_KWARGS["sigma"])
        model = row_scores.AnalyticModel(
            sde, specification["row_score"], module, device=arguments.device,
            nodes=arguments.quadrature_nodes, seed=arguments.seed,
        )

        with patched_problem_module(module):
            problem = build_problem(module, model, arguments.observations,
                                    arguments.seed, arguments.device)
            print(f"\n########## {problem_name}: {specification['label']}")
            print(f"true g = {problem['global_truth']:.5f}   exact posterior "
                  f"mean {problem['global_mean']:.5f}, sd "
                  f"{problem['global_std']:.5f}   x in "
                  f"[{problem['x'].min():.2f}, {problem['x'].max():.2f}]")

            for name in arguments.variants:
                entry = VARIANTS[name]
                folder = arguments.artifacts / f"symmetric_{problem_name}_{name}"
                folder.mkdir(parents=True, exist_ok=True)
                archive = folder / f"{name}.npz"

                if arguments.replot:
                    with np.load(archive, allow_pickle=True) as stored:
                        result = {key: stored[key] for key in stored.files}
                    result["global_map"] = float(result["global_map"])
                    result["excluded"] = int(result["excluded"])
                    metrics = json.loads(str(result["metrics_json"]))
                else:
                    print(f"\n=== {problem_name}: {entry['label']} ===")
                    result, metrics = run_variant(
                        entry, module, problem, model, arguments
                    )
                    print(f"  {metrics['sample_seconds']:.1f}s  "
                          f"{metrics['sampler_network_calls']} calls  "
                          f"W1 {metrics['shared_w1_over_sigma']:.4f}  "
                          f"width {metrics['shared_width_ratio']:.3f}  "
                          f"local W1 {metrics['local_w1_over_sigma']:.4f}")
                    scalar = {key: value for key, value in metrics.items()
                              if np.isscalar(value) or isinstance(value, str)}
                    np.savez_compressed(
                        archive,
                        x=problem["x"], global_grid=problem["global_grid"],
                        global_density=problem["global_density"],
                        global_truth=problem["global_truth"],
                        exact_local_mean=problem["local_mean"],
                        exact_local_std=problem["local_std"],
                        metrics_json=json.dumps(scalar, default=float),
                        score_grid=metrics["score_grid"],
                        implied_score=metrics["implied_score"],
                        exact_score=metrics["exact_score"],
                        **result,
                    )

                title = (
                    f"{entry['label']}  —  {specification['label']}\n"
                    f"(g → ℓⱼ → xⱼ, exact single-observation score, "
                    f"no corrector steps)"
                )
                plot_method(problem, module, name, entry, result, metrics,
                            title, folder / f"01_{problem_name}_{name}.png")

                rows.append({
                    "problem": problem_name, "method": name,
                    "label": entry["label"], "eta": entry["eta"],
                    "correction": entry["kwargs"]["correction"],
                    "seconds": metrics["sample_seconds"],
                    "calls": metrics["sampler_network_calls"],
                    **{key: metrics[key] for key in FIELDS[7:]},
                })

    path = arguments.artifacts / "symmetric_experiments.csv"
    merged = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                merged[(row["problem"], row["method"])] = row
    for row in rows:
        merged[(row["problem"], row["method"])] = row
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(merged, key=lambda item: (
            list(PROBLEMS).index(item[0]) if item[0] in PROBLEMS else 99,
            list(VARIANTS).index(item[1]) if item[1] in VARIANTS else 99,
        )):
            writer.writerow({f: merged[key].get(f, "") for f in FIELDS})
    print(f"\nWrote {path}")

    plot_cross_problem_summary(
        arguments.artifacts, arguments.artifacts / "07_symmetric_summary.png"
    )


if __name__ == "__main__":
    main()
