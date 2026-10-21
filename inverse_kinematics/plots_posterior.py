#!/usr/bin/env python3
"""Posterior figures for the compositional inference run.

Reads ``artifacts/inference/{posteriors.npz,metrics.json}`` and writes three
figures. Kept separate from :mod:`run_inference` so the figures can be redrawn
without resampling.

Usage:
    python inverse_kinematics/plots_posterior.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

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
from plots import INK, RAIL, _draw_arm, _draw_rail, _style_axis  # noqa: E402

ROOT = Path(__file__).resolve().parent
PRIOR_GREY = "#B9C0C7"

# Plot styling is keyed by method here rather than read from the stored
# metrics, so the palette can change without re-running a sampler. The stored
# colour is kept as a fallback for arms this table does not know.
METHOD_COLOURS = {
    "langevin_fnpse": "#2A9D8F",     # teal
    "gauss_hierarchical": "#E76F51",  # coral
    "gauss_jacobian": "#5E60CE",      # indigo
}


def method_colour(method, row):
    return METHOD_COLOURS.get(method, row.get("colour", INK))



def _kde_curve(draws, grid):
    return gaussian_kde(np.asarray(draws, dtype=np.float64))(grid)


def plot_global_posterior(store, metrics, output):
    """The headline: three composed posteriors against the reference.

    The reference is the ABC posterior computed straight through the forward
    model, so this panel grades the composition rules rather than comparing them
    to each other's opinion.
    """
    grid = store["grid"]
    density = store["density"]
    truth = float(store["truth"])

    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4.0), layout="constrained",
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )

    main = axes[0]
    prior = np.exp(-0.5 * (grid / H.GLOBAL_PRIOR_STD[0]) ** 2)
    prior /= np.trapezoid(prior, grid)
    main.fill_between(grid, prior, color=PRIOR_GREY, alpha=0.45, linewidth=0,
                      label="Prior", zorder=1)
    main.plot(grid, density, color=INK, linewidth=2.4, zorder=4,
              label="Reference (ABC, no network)")

    for method, row in metrics["methods"].items():
        draws = store[f"samples_{method}"][0, :, H.HIERARCHY[0]]
        main.plot(grid, _kde_curve(draws, grid), color=method_colour(method, row),
                  linewidth=1.9, alpha=0.95, zorder=3, label=row["label"])
    main.axvline(truth, color=INK, linestyle=(0, (4, 3)), linewidth=1.2,
                 zorder=5, label="Truth")

    _style_axis(main)
    main.set_xlim(truth - 6 * metrics["reference"]["std"],
                  truth + 6 * metrics["reference"]["std"])
    main.set_ylim(bottom=0)
    main.set_xlabel(r"$\mathrm{angle}_1$ (shared)", color=INK)
    main.set_ylabel("Density", color=INK)
    main.set_title(
        f"Shared shoulder angle, {metrics['observations']} observations",
        fontsize=11, color=INK,
    )
    main.legend(frameon=False, fontsize=8.5, loc="upper left")

    # Right: the same thing as numbers, since three overlapping curves near the
    # reference are hard to rank by eye.
    table = axes[1]
    table.axis("off")
    names = list(metrics["methods"])
    rows = [
        [metrics["methods"][m]["label"].replace(" + ", "\n+ "),
         f"{metrics['methods'][m]['w1_over_sigma']:.3f}",
         f"{metrics['methods'][m]['width_ratio']:.3f}",
         f"{metrics['methods'][m]['mean_error_sigma']:+.3f}",
         f"{metrics['methods'][m]['seconds']:.0f}"]
        for m in names
    ]
    handle = table.table(
        cellText=rows,
        colLabels=["Rule", r"W1/$\sigma$", "Width", r"Bias/$\sigma$", "s"],
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.40, 0.15, 0.15, 0.17, 0.13],
    )
    handle.auto_set_font_size(False)
    handle.set_fontsize(8.5)
    handle.scale(1.0, 2.1)
    for (row_index, column), cell in handle.get_celld().items():
        cell.set_edgecolor(RAIL)
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_text_props(color=INK, fontweight="semibold")
            cell.set_facecolor("#F1F4F6")
        elif column == 0:
            cell.set_text_props(color=metrics["methods"][names[row_index - 1]]["colour"])
    table.set_title("Against the reference", fontsize=11, color=INK, pad=14)

    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_local_posteriors(store, metrics, output, observation=0):
    """One observation's private coordinates, per rule, against the truth.

    Included because a shared-coordinate figure alone cannot show whether the
    local block survived the composition: every rule here also applies a
    cross-correction to the row-local scores, and on the repository's other
    benchmarks that is exactly where the pilot rule loses.
    """
    local_truth = np.asarray(store["local_truth"])[observation]
    figure, axes = plt.subplots(1, 3, figsize=(10, 2.9), layout="constrained")

    for index, (axis, name) in enumerate(zip(axes, H.LOCAL_NAMES)):
        node = H.LOCAL_INDICES[index]
        lo, hi = np.inf, -np.inf
        for method, row in metrics["methods"].items():
            draws = store[f"samples_{method}"][observation, :, node]
            lo, hi = min(lo, draws.min()), max(hi, draws.max())
        grid = np.linspace(lo, hi, 400)
        for method, row in metrics["methods"].items():
            draws = store[f"samples_{method}"][observation, :, node]
            axis.plot(grid, _kde_curve(draws, grid), color=method_colour(method, row),
                      linewidth=1.8, label=row["label"])
        prior_scale = H.LOCAL_PRIOR_STD[index]
        prior = np.exp(-0.5 * (grid / prior_scale) ** 2)
        prior /= np.trapezoid(prior, grid)
        axis.fill_between(grid, prior, color=PRIOR_GREY, alpha=0.4, linewidth=0,
                          zorder=0)
        axis.axvline(local_truth[index], color=INK, linestyle=(0, (4, 3)),
                     linewidth=1.2)
        _style_axis(axis)
        axis.set_ylim(bottom=0)
        axis.set_xlabel(" ".join(name.title().split("_")), color=INK)
    axes[0].set_ylabel("Density", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=8.5,
                  loc="outside upper center", ncols=3)
    figure.suptitle(
        f"Local coordinates of observation {observation} "
        "(dashed: truth, grey: prior)",
        fontsize=11, color=INK, y=1.14,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_arm_posterior(store, metrics, output, method=None, count=4,
                       ensemble=150):
    """Posterior arms for the first few observations, drawn from one rule.

    The end-to-end check a coordinate histogram cannot make: every drawn
    configuration should terminate on its own observed end position while the
    shoulder angle stays common to all of them. A composition that got the
    shared coordinate wrong shows up here as arms whose tips miss their cross.
    """
    method = method or ("gauss_jacobian" if "gauss_jacobian" in metrics["methods"]
                        else next(iter(metrics["methods"])))
    row = metrics["methods"][method]
    samples = store[f"samples_{method}"]
    observations = np.asarray(store["observations"])
    local_truth = np.asarray(store["local_truth"])
    truth = float(store["truth"])
    count = min(count, samples.shape[0])

    figure, axes = plt.subplots(1, count, figsize=(2.5 * count, 3.1),
                                layout="constrained", sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    stride = max(1, samples.shape[1] // ensemble)

    for index, axis in enumerate(axes):
        theta = samples[index, ::stride, :4]           # node order
        joints = sim.joint_positions(H.to_simulator_order(theta))
        for chain in joints:
            _draw_arm(axis, chain, row["colour"], 1.0, 0.10)

        true_theta = np.concatenate([[truth], local_truth[index]])
        true_joints = sim.joint_positions(H.to_simulator_order(true_theta))[0]
        _draw_arm(axis, true_joints, INK, 1.8, 0.9, zorder=4, joint_size=2.8)

        # `observations` is (x1, x2); the plot is (x2, x1).
        axis.plot(observations[index, 1], observations[index, 0], marker="X",
                  markersize=9, color=INK, markeredgecolor="white",
                  markeredgewidth=0.9, zorder=6)
        _draw_rail(axis, span=1.6)
        _style_axis(axis)
        axis.set_aspect("equal")
        axis.set_xlim(-1.6, 2.3)
        axis.set_ylim(-1.6, 1.6)
        axis.set_title(f"Observation {index}", fontsize=9, color=INK)
        axis.set_xlabel("$x_2$", color=INK)
    axes[0].set_ylabel("$x_1$", color=INK)

    figure.suptitle(
        f"Posterior arm configurations — {row['label']}\n"
        "faint: posterior draws · solid: truth · cross: observed end position",
        fontsize=10.5, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    result.add_argument("--observation", type=int, default=0)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    source = arguments.artifacts / "inference"
    store = np.load(source / "posteriors.npz")
    metrics = json.loads((source / "metrics.json").read_text())

    figures = arguments.artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = [
        plot_global_posterior(store, metrics, figures / "04_global_posterior.png"),
        plot_local_posteriors(store, metrics,
                              figures / "05_local_posteriors.png",
                              observation=arguments.observation),
        plot_arm_posterior(store, metrics, figures / "06_arm_posterior.png"),
    ]
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def plot_observation_sweep(sweep, output, plt_module=None):
    """How the shared posterior behaves as observations are combined.

    Left: the reference contracts and every rule tracks its width. Right: the
    bias in units of that shrinking width. The ``n = 1`` column is the control --
    the arrow system collapses to the network's raw score there, so composition
    is provably a no-op and any bias is the network's own.
    """
    plt_module = plt_module or plt
    rows = sweep["rows"]
    counts = sorted({row["observations"] for row in rows})
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    colours = {"langevin_fnpse": "#2A9D8F",
               "gauss_hierarchical": "#E76F51",
               "gauss_jacobian": "#264653"}
    labels = {"langevin_fnpse": "Langevin + F-NPSE",
              "gauss_hierarchical": "gauss_hierarchical + churn",
              "gauss_jacobian": "gauss_jacobian + churn"}

    figure, axes = plt_module.subplots(1, 2, figsize=(10.5, 3.9),
                                       layout="constrained")

    reference = [next(r for r in rows if r["observations"] == n)["reference_std"]
                 for n in counts]
    axes[0].plot(counts, reference, color=INK, linewidth=2.2, marker="o",
                 markersize=5, label="Reference (SBC-validated)", zorder=4)
    for method in methods:
        widths = [next(r for r in rows if r["observations"] == n
                       and r["method"] == method)["posterior_std"]
                  for n in counts]
        axes[0].plot(counts, widths, color=colours[method], linewidth=1.7,
                     marker="s", markersize=4, label=labels[method])
    axes[0].set_ylabel("Posterior sd of $\\mathrm{angle}_1$", color=INK)
    axes[0].set_title("Contraction with observations", fontsize=10.5, color=INK)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(0.0, color=RAIL, linewidth=1.0, zorder=0)
    for method in methods:
        bias = [next(r for r in rows if r["observations"] == n
                     and r["method"] == method)["bias_sigma"]
                for n in counts]
        axes[1].plot(counts, bias, color=colours[method], linewidth=1.7,
                     marker="s", markersize=4, label=labels[method])
    axes[1].set_ylabel("Bias / reference sd", color=INK)
    axes[1].set_title("Bias grows as the posterior narrows",
                      fontsize=10.5, color=INK)
    axes[1].annotate(
        "n = 1: composition is a no-op",
        xy=(counts[0], 0.0), xytext=(1.35, -0.42), fontsize=8, color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.8),
    )

    for axis in axes:
        _style_axis(axis)
        axis.set_xscale("log", base=2)
        axis.set_xticks(counts, [str(n) for n in counts])
        axis.set_xlabel("Observations sharing $\\mathrm{angle}_1$", color=INK)

    figure.suptitle(
        "Shared-coordinate accuracy against observation count",
        fontsize=11.5, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt_module.close(figure)
    return output
