#!/usr/bin/env python3
"""Figures for the bimodal shared-posterior experiment.

Reads ``artifacts/bimodal/{posteriors.npz,metrics.json}``. Three figures, one per
question the experiment asks.

Usage:
    python inverse_kinematics/plots_bimodal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
from plots import (ACCENT, INK, RAIL, _draw_arm, _draw_rail,  # noqa: E402
                   _style_axis, plot_params_kinematic)

ROOT = Path(__file__).resolve().parent
PRIOR_GREY = "#B9C0C7"

# Plot styling is keyed by method here rather than read from the stored
# metrics, so the palette can change without re-running a sampler. The stored
# colour is kept as a fallback for arms this table does not know.
METHOD_COLOURS = {
    "langevin_fnpse": "#2A9D8F",     # teal
    "langevin_fnpse_arruda_damped": "#D28A00",  # amber
    "gauss_hierarchical": "#E76F51",  # coral
    "gauss_jacobian": "#5E60CE",      # indigo
}


def method_colour(method, row):
    return METHOD_COLOURS.get(method, row.get("colour", INK))



def plot_single_capability(store, metrics, output):
    """Question 1: can the network hold two modes with no composition at all?

    Drawn in the notebook's own four-panel form so it can be laid beside their
    figure 11 directly. If this panel is unimodal, nothing downstream about
    composition can be concluded.
    """
    draws = np.asarray(store["single_draws"])
    prior = np.random.default_rng(0).normal(
        loc=H.PRIOR_MEAN, scale=H.PRIOR_STD, size=(len(draws), 4)
    )
    # Node order -> simulator order, so the panels match theirs column for column.
    return plot_params_kinematic(
        H.to_simulator_order(draws[:, :4]), H.to_simulator_order(prior),
        output=output, plt_module=plt,
        title="COMPASS single-observation posterior at obs = [0, 1.5] "
              "(magenta: posterior, black: prior)",
    )


def plot_shared_posterior(store, metrics, output):
    """Question 2: does composition preserve the two modes?"""
    grid = np.asarray(store["grid"])
    density = np.asarray(store["density"])
    truth = float(store["truth"])

    figure, axis = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    prior = np.exp(-0.5 * (grid / H.GLOBAL_PRIOR_STD[0]) ** 2)
    prior /= np.trapezoid(prior, grid)
    axis.fill_between(grid, prior, color=PRIOR_GREY, alpha=0.45, linewidth=0,
                      label="Prior", zorder=1)
    # A white underlay makes the reference readable where a rule crosses it,
    # without needing a colour that competes with the three rules.
    import matplotlib.patheffects as path_effects

    axis.plot(grid, density, color=INK, linewidth=3.0, zorder=5,
              label="Reference (ABC, no network)",
              path_effects=[path_effects.Stroke(linewidth=5.2, foreground="white"),
                            path_effects.Normal()])
    for method, row in metrics["methods"].items():
        draws = store[f"samples_{method}"][0, :, H.HIERARCHY[0]]
        curve = gaussian_kde(np.asarray(draws, dtype=np.float64))(grid)
        curve /= np.trapezoid(curve, grid)
        axis.plot(grid, curve, color=method_colour(method, row), linewidth=1.9, zorder=3,
                  label=f"{row['label']}  (dip {row['dip_ratio']:.2f})")
    # There is exactly ONE true angle_1. The second line is its *reflection*
    # under (a1,a2,a3,h) -> (-a1,-a2,-a3, 2*x1-h): a different parameter vector
    # that produces byte-identical data, so the likelihood cannot separate them.
    # Labelling both "+/- truth" (as an earlier version did) wrongly suggests two
    # true values.
    axis.axvline(truth, color=INK, linestyle="-", linewidth=1.4, zorder=4)
    axis.axvline(-truth, color=INK, linestyle=(0, (2, 2)), linewidth=1.2,
                 alpha=0.75, zorder=4)
    # Annotations sit low: the legend occupies the upper left, and an earlier
    # version put the mirror label straight through it.
    top = axis.get_ylim()[1]
    axis.annotate("truth", xy=(truth + 0.04, top * 0.30), fontsize=8.5,
                  color=INK, ha="left")
    axis.annotate("mirror of truth\n(identical data)",
                  xy=(-truth - 0.04, top * 0.24), fontsize=8, color=INK,
                  ha="right", linespacing=1.3)

    _style_axis(axis)
    axis.set_xlim(-1.6, 1.6)
    axis.set_ylim(bottom=0)
    axis.set_xlabel(r"$\mathrm{angle}_1$ (shared)", color=INK)
    axis.set_ylabel("Density", color=INK)
    selection = ("every observation at $x_1 = 0$ exactly"
                 if metrics.get("exact_x1_zero")
                 else f"selected with $|x_1| < {metrics['select_x1']}$")
    axis.set_title(
        f"Shared angle, {metrics['observations']} observations, {selection}",
        fontsize=11, color=INK,
    )
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_bimodal_arms(store, metrics, output, method=None, ensemble=200):
    """The two branches as arms, for the first four observations."""
    method = method or ("gauss_jacobian" if "gauss_jacobian" in metrics["methods"]
                        else next(iter(metrics["methods"])))
    row = metrics["methods"][method]
    samples = store[f"samples_{method}"]
    observed = np.asarray(store["observations"])
    count = min(4, samples.shape[0])

    figure, axes = plt.subplots(1, count, figsize=(2.5 * count, 3.2),
                                layout="constrained", sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    stride = max(1, samples.shape[1] // ensemble)
    for index, axis in enumerate(axes):
        theta = samples[index, ::stride, :4]
        # Colour by branch, which is the whole point: the two lobes of the
        # shared angle should show up as two families of configurations.
        for chain, angle in zip(sim.joint_positions(H.to_simulator_order(theta)),
                                theta[:, 0]):
            _draw_arm(axis, chain, ACCENT if angle >= 0 else "#2A6F97",
                      1.0, 0.12)
        axis.plot(observed[index, 1], observed[index, 0], marker="X",
                  markersize=9, color=INK, markeredgecolor="white",
                  markeredgewidth=0.9, zorder=6)
        _draw_rail(axis, span=1.6)
        _style_axis(axis)
        axis.set_aspect("equal")
        axis.set_xlim(-1.8, 2.3)
        axis.set_ylim(-1.7, 1.7)
        axis.set_title(f"Observation {index}", fontsize=9, color=INK)
        axis.set_xlabel("$x_2$", color=INK)
    axes[0].set_ylabel("$x_1$", color=INK)
    figure.suptitle(
        f"Posterior arms by branch — {row['label']}\n"
        r"magenta: $\mathrm{angle}_1 \geq 0$ · blue: $\mathrm{angle}_1 < 0$ · "
        "cross: observed end position",
        fontsize=10.5, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    arguments = result.parse_args(argv)

    source = arguments.artifacts / "bimodal"
    store = np.load(source / "posteriors.npz")
    metrics = json.loads((source / "metrics.json").read_text())
    figures = arguments.artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    written = [
        plot_single_capability(store, metrics,
                               figures / "13_compass_posterior_single.png"),
        plot_shared_posterior(store, metrics,
                              figures / "14_bimodal_shared_posterior.png"),
        plot_bimodal_arms(store, metrics,
                          figures / "15_bimodal_arms.png"),
    ]
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
