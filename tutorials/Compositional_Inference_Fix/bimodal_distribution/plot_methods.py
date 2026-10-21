#!/usr/bin/env python3
"""Figures 3-6: how each composition rule and sampler does on a bimodal target.

Reads the ``.npz`` files ``run_experiment.py`` wrote; runs no sampler itself, so
it can be re-run freely while tuning a figure.

``03_<name>_marginals.png``    the shared marginal each arm produced, against
                               the exact one, learned score on top and exact row
                               score below -- the panel that says whether the
                               two modes survived at all.
``04_<name>_joint.png``        the same draws in the (g, l_1) plane, where a
                               rule that keeps the modes but breaks the
                               shared/local coupling is visible and the
                               marginals alone would not show it.
``05_<name>_score_error.png``  the composed score's own error against the exact
                               diffused score, per noise level -- the rule
                               isolated from the sampler.
``06_<name>_summary.png``      four numbers per arm: distance to the exact
                               marginal, how the mass split between the modes,
                               how deep the recovered valley is, and whether the
                               local signs stayed coordinated.

Usage:
    python plot_methods.py
    python plot_methods.py --models sign
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
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mixture_reference as reference  # noqa: E402
import style  # noqa: E402

ARTIFACTS = ROOT / "artifacts"

SCORE_LABEL = {
    "learned": "learned row score (small net, 552K)",
    "learned_large": "learned row score (stock net, 5.2M)",
    "analytic": "exact row score (closed form, no network)",
}
# The short form, for axis labels and legends where the long one would not fit.
# It must still tell the two learned rungs apart -- "learned row score" alone
# would label both.
SCORE_SHORT = {
    "learned": "small net (552K)",
    "learned_large": "stock net (5.2M)",
    "analytic": "exact row score",
}
# Which composition rule each arm uses; this is what colour encodes.
RULE_OF = {
    "dpm2_churn8_gauss_hierarchical": "gauss_hierarchical",
    "dpm2_gauss_hierarchical": "gauss_hierarchical",
    "dpm2_churn8_gauss_jacobian": "gauss_jacobian",
    "dpm2_gauss_jacobian": "gauss_jacobian",
    "langevin_fnpe": "fnpe",
}


def load(model_name, score, method):
    path = ARTIFACTS / model_name / score / f"{method}.npz"
    return dict(np.load(path, allow_pickle=True)) if path.exists() else None


def available(model_name, methods, scores):
    """Only the (score, method) cells that actually have results on disk."""
    return {
        (score, method): load(model_name, score, method)
        for score in scores for method in methods
        if load(model_name, score, method) is not None
    }


def colour_of(method):
    return style.RULE_COLOUR[RULE_OF[method]]


def linestyle_of(method, experiment):
    return experiment.METHODS[method]["linestyle"]


def window(grid, density, pad=0.28):
    """Crop a wide reference grid to where the density actually lives."""
    mass = density / density.sum()
    inside = np.flatnonzero(mass > 1e-6 * mass.max())
    low, high = grid[inside[0]], grid[inside[-1]]
    margin = pad * (high - low)
    return low - margin, high + margin


# ---------------------------------------------------------------------------
# Figure 3: the shared marginal
# ---------------------------------------------------------------------------

def figure_marginals(module, model_name, results, methods, scores, problem, path,
                     experiment):
    rows = [score for score in scores
            if any(key[0] == score for key in results)]
    figure, axes = plt.subplots(
        len(rows), len(methods), figsize=(3.05 * len(methods), 3.0 * len(rows)),
        squeeze=False, sharex=True, sharey=True,
    )
    figure.suptitle(
        f"{module.TITLE}\nshared posterior $p(g \\mid y_1 \\dots y_n)$, "
        f"n = {problem['observations']}",
        fontsize=12.5, color=style.INK, y=1.0,
    )

    first = next(iter(results.values()))
    grid = first["global_grid"]
    density = first["global_density"]
    # What a perfect sampler's kernel density estimate would converge to: the
    # exact mixture convolved with the same kernel, in closed form. Comparing
    # the draws to *this* rather than to the raw density keeps the kernel's own
    # smoothing out of the verdict.
    smoothed = first["smoothed_density"]
    limits = window(grid, density)
    # One y-scale for every panel. Panel-local limits would silently rescale a
    # sampler that produced peaks half again too tall into looking like the rest.
    ceiling = 1.32 * max(
        [float(smoothed.max())]
        + [float(gaussian_kde(
            payload["shared_draws"],
            bw_method=float(payload["kde_bandwidth"])
            / float(payload["shared_draws"].std(ddof=1)),
        )(grid).max()) for payload in results.values()]
    )

    for row, score in enumerate(rows):
        for column, method in enumerate(methods):
            axis = axes[row][column]
            payload = results.get((score, method))
            axis.fill_between(grid, density, color=style.REFERENCE_FILL, zorder=1,
                              label="exact")
            axis.plot(grid, smoothed, color=style.INK, lw=1.4, zorder=4,
                      label="exact, same kernel")
            if payload is None:
                axis.text(0.5, 0.5, "not run", transform=axis.transAxes,
                          ha="center", color=style.MUTED)
                style.despine(axis)
                continue
            draws = payload["shared_draws"]
            # The same absolute bandwidth the metrics use, and the same one for
            # every panel -- Scott's rule would set it from the gap between the
            # modes and flatten exactly the structure being judged.
            bandwidth = float(payload["kde_bandwidth"])
            estimate = gaussian_kde(
                draws, bw_method=bandwidth / float(draws.std(ddof=1))
            )(grid)
            axis.plot(grid, estimate, color=colour_of(method),
                      lw=style.ESTIMATE_LW, ls=linestyle_of(method, experiment),
                      zorder=5, label="sampled")
            axis.axvline(float(problem["global_truth"]), color=style.MUTED,
                         lw=1.0, ls=":", zorder=2)
            axis.set_xlim(*limits)
            axis.set_ylim(0.0, ceiling)
            axis.text(
                0.03, 0.97,
                f"W1/$\\sigma$ {float(payload['shared_w1_over_sigma']):.3f}\n"
                f"modes {int(payload['modes_found'])}"
                f"/{int(payload['modes_exact'])}\n"
                f"P(g>0) {float(payload['positive_mass']):.2f} "
                f"(exact {float(payload['positive_mass_exact']):.2f})",
                transform=axis.transAxes, va="top", fontsize=8,
                color=style.SECONDARY,
            )
            if row == 0:
                axis.set_title(experiment.METHODS[method]["short"], fontsize=10)
            if column == 0:
                axis.set_ylabel(SCORE_LABEL[score].replace(" (", "\n("),
                                fontsize=8.5)
            if row == len(rows) - 1:
                axis.set_xlabel("shared parameter $g$")
            style.despine(axis)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, 0.02), fontsize=9)

    style.caption(figure, (
        "Filled grey is the exact posterior; the black line is that same "
        "posterior convolved with the estimator's kernel -- the curve a perfect "
        "sampler would produce -- and the coloured line is the draws through "
        "that kernel. Colour is the composition rule (blue gauss_hierarchical, "
        "orange gauss_jacobian, aqua F-NPSE); solid is the churn eta = 8 "
        "predictor, dashed the ten-corrector one."
    ), y=-0.01)
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    figure.savefig(path)
    plt.close(figure)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Figure 4: the (g, l_1) plane
# ---------------------------------------------------------------------------

def figure_joint(module, model_name, results, methods, scores, problem, path,
                 experiment):
    rows = [score for score in scores if any(key[0] == score for key in results)]
    figure, axes = plt.subplots(
        len(rows), len(methods), figsize=(3.05 * len(methods), 3.05 * len(rows)),
        squeeze=False, sharex=True, sharey=True,
    )
    figure.suptitle(
        f"{module.TITLE}\nshared and first local coordinate, $(g, \\ell_1)$",
        fontsize=12.5, color=style.INK, y=1.0,
    )

    y = np.asarray(problem["y"], dtype=np.float64)
    weights, means, covariance = module.mixture(y)
    keep = [0, 1]
    sub_means = means[:, keep]
    sub_covariance = covariance[np.ix_(keep, keep)]
    # Per-axis limits, not a symmetric box: with weak pooling the local
    # coordinate tracks its own observation rather than the shared sign, so a
    # box centred on zero would spend most of its area on empty space.
    margin = 3.4 * np.sqrt(np.diag(sub_covariance))
    low = sub_means.min(axis=0) - margin
    high = sub_means.max(axis=0) + margin
    grid_x = np.linspace(low[0], high[0], 260)
    grid_y = np.linspace(low[1], high[1], 260)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
    precision = np.linalg.inv(sub_covariance)
    residual = points[:, None, :] - sub_means[None, :, :]
    quadratic = np.einsum("ski,ij,skj->sk", residual, precision, residual)
    exact_density = (weights[None, :] * np.exp(-0.5 * quadratic)).sum(axis=1)
    exact_density = exact_density.reshape(mesh_x.shape)
    levels = np.array([0.05, 0.3, 0.7]) * exact_density.max()

    for row, score in enumerate(rows):
        for column, method in enumerate(methods):
            axis = axes[row][column]
            axis.contour(mesh_x, mesh_y, exact_density, levels=levels,
                         colors=[style.MUTED], linewidths=1.1, zorder=3)
            payload = results.get((score, method))
            if payload is None:
                axis.text(0.5, 0.5, "not run", transform=axis.transAxes,
                          ha="center", color=style.MUTED)
                style.despine(axis)
                continue
            axis.scatter(payload["shared_draws"], payload["local_draws"][0],
                         s=4, color=colour_of(method), alpha=0.22,
                         linewidths=0, zorder=2)
            axis.scatter([problem["global_truth"]],
                         [np.asarray(problem["local_truth"])[0]], s=42,
                         marker="*", color=style.INK, zorder=6)
            axis.set_xlim(low[0], high[0])
            axis.set_ylim(low[1], high[1])
            if row == 0:
                axis.set_title(experiment.METHODS[method]["short"], fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{SCORE_SHORT[score]}\nlocal $\\ell_1$",
                                fontsize=9)
            if row == len(rows) - 1:
                axis.set_xlabel("shared $g$")
            style.despine(axis)

    style.caption(figure, (
        "Grey contours are the exact joint; the star is the truth. The exact "
        "posterior puts its mass in two blobs on the diagonal -- a rule that "
        "bridges them along that diagonal has kept both modes but leaked mass "
        "through the valley, and one that fills the off-diagonal quadrants has "
        "lost the coupling between the shared sign and the local one."
        if module.NAME == "sign" else
        "Grey contours are the exact joint; the star is the truth. Pooling is "
        "weak here, so the local coordinate follows its own observation rather "
        "than the shared lobe: the two blobs differ almost only in g. What to "
        "look for is their width, which every arm here under-states."
    ), y=0.0)
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    figure.savefig(path)
    plt.close(figure)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Figure 5: the composed score itself
# ---------------------------------------------------------------------------

def figure_score_error(module, model_name, results, methods, scores, problem,
                       path, experiment, merge_lambda=None):
    rows = [score for score in scores if any(key[0] == score for key in results)]
    figure, axes = plt.subplots(1, len(rows) + 1,
                                figsize=(4.6 * (len(rows) + 1), 4.1),
                                squeeze=False)
    axes = axes[0]
    figure.suptitle(
        f"{module.TITLE}\ncomposed score against the exact diffused score",
        fontsize=12.5, color=style.INK, y=1.02,
    )

    for index, score in enumerate(rows):
        axis = axes[index]
        for method in methods:
            payload = results.get((score, method))
            if payload is None:
                continue
            axis.plot(payload["score_lambdas"], payload["score_shared_rel_rmse"],
                      color=colour_of(method), lw=style.ESTIMATE_LW,
                      ls=linestyle_of(method, experiment), marker="o", ms=4.5,
                      label=experiment.METHODS[method]["short"])
        axis.set_xscale("log")
        axis.set_yscale("log")
        if merge_lambda is not None:
            # Placed in axes fraction on y, so it neither rescales the log axis
            # nor depends on limits that are still being computed.
            axis.axvline(merge_lambda, color=style.INK, lw=1.1, ls=":")
            axis.text(merge_lambda, 0.02, " row modes merge",
                      transform=axis.get_xaxis_transform(), rotation=90,
                      va="bottom", ha="left", fontsize=8,
                      color=style.SECONDARY)
        axis.set_xlabel("noise level $\\lambda$")
        axis.set_ylabel("relative RMS error of $\\nabla_g \\log p_\\lambda$")
        axis.set_title(SCORE_LABEL[score])
        axis.legend(loc="upper left", ncol=1)
        style.despine(axis)

    # The last panel: composed against exact, state by state, at the smallest
    # probed noise level. A scatter against the identity rather than a curve over
    # g -- the score depends on all n + 1 coordinates, so plotting it against the
    # shared one alone would show scatter that is projection, not error.
    axis = axes[-1]
    reference_payload = next(iter(results.values()))
    lambdas = reference_payload["score_lambdas"]
    level = int(np.argmin(np.abs(lambdas - 0.2)))
    exact_field = reference_payload["probe_exact_shared"][level]
    span = float(np.abs(exact_field).max()) * 1.05
    axis.plot([-span, span], [-span, span], **style.TRUTH_KWARGS,
              label="exact (identity)")
    for method in methods:
        payload = results.get((rows[-1], method))
        if payload is None:
            continue
        axis.scatter(exact_field, payload["probe_composed_shared"][level],
                     s=7, color=colour_of(method), alpha=0.35, linewidths=0,
                     label=experiment.METHODS[method]["short"])
    axis.set_xlim(-span, span)
    axis.set_xlabel(f"exact $\\nabla_g \\log p_\\lambda$ at "
                    f"$\\lambda$ = {lambdas[level]:g}")
    axis.set_ylabel("composed $\\nabla_g \\log p_\\lambda$")
    axis.set_title(f"the field itself, {SCORE_SHORT[rows[-1]]}")
    axis.legend(loc="upper left", fontsize=8, markerscale=2.2)
    style.despine(axis)

    style.caption(figure, (
        "States are drawn from the exact diffused posterior, so the error is "
        "weighted by where the sampler actually is at that noise level. The "
        "left panels differ only in whether the row score being composed is the "
        "network's or the closed form, so the gap between them is network error "
        "and everything left in them is the rule. Two arms that share a "
        "composition rule share a field, so their curves coincide -- what "
        "separates them is the sampler, which this figure deliberately excludes."
    ), y=-0.02)
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(path)
    plt.close(figure)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Figure 6: the summary
# ---------------------------------------------------------------------------

# Two of these four are magnitudes with a meaningful zero, so they are bars; the
# other two live in a narrow band near their exact value, where a bar from zero
# would show five identical rectangles and a truncated bar axis would be a lie.
# Those are drawn as dots on their own scale with the exact value as a line.
PANELS = (
    ("shared_w1_over_sigma", "distance to the exact marginal",
     "$W_1 / \\sigma$  (lower is better)", None, "bar"),
    ("gap_draws", "draws that leaked into the gap between the modes",
     "draws", "gap_draws_exact", "bar"),
    ("runtime_seconds", "cost", "seconds", None, "bar"),
    ("positive_mass", "how the mass split between the modes",
     "$P(g > 0)$", "positive_mass_exact", "dot"),
    ("mode_width", "width of each mode",
     "within-mode s.d. of $g$", "mode_width_exact", "dot"),
    ("shared_local_correlation", "shared/local coupling kept",
     "mean corr($g$, $\\ell_j$)", "shared_local_correlation_exact", "dot"),
)

MARKER = {"learned": "o", "learned_large": "s", "analytic": "D"}
HATCH = {"learned": None, "learned_large": "..", "analytic": "///"}


def figure_summary(module, model_name, results, methods, scores, path,
                   experiment):
    rows = [score for score in scores if any(key[0] == score for key in results)]
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 7.8))
    figure.suptitle(f"{module.TITLE} -- summary", fontsize=13, color=style.INK,
                    y=0.99)

    positions = np.arange(len(methods), dtype=float)
    width = 0.8 / max(len(rows), 1)

    for axis, (key, title, ylabel, exact_key, form) in zip(axes.ravel(), PANELS):
        target = (float(next(iter(results.values()))[exact_key])
                  if exact_key is not None else None)
        for index, score in enumerate(rows):
            offset = (index - (len(rows) - 1) / 2) * width
            values = [float(results[(score, method)][key])
                      if (score, method) in results else np.nan
                      for method in methods]
            colours = [colour_of(method) for method in methods]
            if form == "bar":
                axis.bar(positions + offset, values, width=width * 0.9,
                         color=colours, edgecolor=style.SURFACE, linewidth=1.4,
                         hatch=HATCH[score],
                         label=SCORE_SHORT[score])
            else:
                if target is not None:
                    axis.vlines(positions + offset, target, values,
                                color=style.MUTED, lw=1.2, alpha=0.7, zorder=1)
                axis.scatter(positions + offset, values, s=64, c=colours,
                             marker=MARKER[score], edgecolor=style.SURFACE,
                             linewidth=1.0, zorder=3,
                             label=SCORE_SHORT[score])
        if target is not None:
            axis.axhline(target, **style.TRUTH_KWARGS,
                         label=f"exact = {target:.3g}")
            # In a right-hand margin kept clear for it, so it never lands on a
            # bar or a marker whatever the values turn out to be.
            axis.annotate(f"exact {target:.3g}", xy=(1.0, target),
                          xycoords=("axes fraction", "data"),
                          xytext=(-3, 3), textcoords="offset points",
                          ha="right", va="bottom", fontsize=8.5,
                          color=style.SECONDARY)
        axis.set_xticks(positions)
        axis.set_xticklabels([experiment.METHODS[name]["short"]
                              for name in methods], rotation=18, ha="right")
        if key == "gap_draws":
            total = int(float(next(iter(results.values()))["posterior_draws"]))
            ylabel = f"{ylabel} (of {total:,})"
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xlim(-0.6, len(methods) - 0.05
                      + (0.85 if target is not None else 0.0))
        style.despine(axis)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, unique = set(), []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique.append((handle, label))
    figure.legend([h for h, _ in unique], [l for _, l in unique],
                  loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035),
                  fontsize=9)

    style.caption(figure, (
        "Bar and marker colour is the composition rule. The top row is measured "
        "from zero, so it is drawn as bars; the bottom row lives in a narrow "
        "band around its exact value, so it is drawn as dots on their own scale "
        "with a leader to the exact line -- bars from zero there would be five "
        "near-identical rectangles, and a truncated bar axis would misstate the "
        "differences. Where an exact-row-score marker is still off target, the "
        "network is not the problem. Runtimes are wall clock on one RTX 2080 Ti "
        "and are an over-estimate wherever the draws had to be taken in chunks "
        "to fit in memory, since a pilot covariance is then re-estimated per "
        "chunk."
    ), y=0.0)
    figure.tight_layout(rect=(0, 0.10, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["sign", "control"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--scores", nargs="+",
                        default=["learned", "learned_large", "analytic"])
    arguments = parser.parse_args()

    import plot_model
    import run_experiment as experiment

    methods = arguments.methods or list(experiment.METHODS)
    style.apply()
    for name in arguments.models:
        module = experiment.MODELS[name]
        results = available(name, methods, arguments.scores)
        if not results:
            print(f"No results for {name}; run run_experiment.py first.")
            continue
        present = [method for method in methods
                   if any(key[1] == method for key in results)]
        with (ARTIFACTS / name / "problem.json").open() as handle:
            problem = json.load(handle)

        weights, means, covariance = module.row_mixture(
            float(np.asarray(problem["y"])[
                int(np.argmax(np.abs(problem["y"])))
            ])
        )
        merge = plot_model.merge_lambda(weights, means[:, 0],
                                        float(covariance[0, 0]))

        output = ARTIFACTS / name
        figure_marginals(module, name, results, present, arguments.scores,
                         problem, output / f"03_{name}_marginals.png", experiment)
        figure_joint(module, name, results, present, arguments.scores, problem,
                     output / f"04_{name}_joint.png", experiment)
        figure_score_error(module, name, results, present, arguments.scores,
                           problem, output / f"05_{name}_score_error.png",
                           experiment, merge_lambda=merge)
        figure_summary(module, name, results, present, arguments.scores,
                       output / f"06_{name}_summary.png", experiment)


if __name__ == "__main__":
    main()
