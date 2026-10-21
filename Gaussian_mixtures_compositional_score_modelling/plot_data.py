#!/usr/bin/env python3
"""Two figures about the data, drawn before any network exists.

``simulation_data``
    What one draw from the generative model looks like, and what a training set
    of them looks like: the graph, the prior, how the mixture propagates and
    blurs along ``g -> l -> x``, and the joint cloud the network is fitted to.

``dataset_regimes``
    What the *tall* datasets look like and, underneath each, the exact posterior
    the composition rules will be graded against. This is the figure that shows
    why the sweep is built the way it is: under ``natural`` data the target
    collapses onto one branch within a few observations, while the three
    constructed regimes hold the shape of the target fixed as ``n`` grows.

Usage:
    python plot_data.py
    python plot_data.py --draws 4000 --observations 1 4 16
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit()

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import datasets  # noqa: E402
from model import DEFAULT  # noqa: E402

ROOT = Path(__file__).resolve().parent

# One categorical pair for the two mixture components, in fixed order, plus the
# neutral ink the exact posterior is always drawn in. The component colours are
# deliberately *not* any of the three method colours used by ``plot_scaling``:
# nothing in this directory should invite reading a component as a method.
MINUS = "#3D5A80"
PLUS = "#EE6C4D"
INK = "#22262B"
RAIL = "#9AA3AB"
FAINT = "#DDE1E6"


def style(axis, grid=False):
    axis.set_facecolor("white")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(RAIL)
    axis.tick_params(colors=INK, labelsize=8, length=3, color=RAIL)
    if grid:
        axis.grid(True, color=FAINT, linewidth=0.6)
        axis.set_axisbelow(True)


def draw_graph(axis, mixture):
    """The graphical model ``C -> g -> l_i -> x_i``, drawn rather than described."""
    axis.set_axis_off()
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    nodes = {
        "C": (1.0, 3.0, "#FFFFFF", RAIL),
        "g": (3.2, 3.0, "#FFFFFF", INK),
        "l": (6.0, 3.0, "#FFFFFF", INK),
        "x": (8.6, 3.0, FAINT, INK),
    }
    labels = {"C": "$C$", "g": "$g$", "l": "$\\ell_i$", "x": "$x_i$"}
    for name, (px, py, fill, edge) in nodes.items():
        axis.add_patch(plt.Circle((px, py), 0.62, facecolor=fill,
                                  edgecolor=edge, linewidth=1.4, zorder=3))
        axis.text(px, py, labels[name], ha="center", va="center",
                  fontsize=13, color=INK, zorder=4)
    # The plate around the per-observation pair.
    axis.add_patch(plt.Rectangle((4.9, 1.5), 4.8, 3.0, fill=False,
                                 edgecolor=RAIL, linewidth=1.0, zorder=1))
    axis.text(9.55, 1.75, "$i = 1 \\dots n$", ha="right", va="bottom",
              fontsize=9, color=RAIL)
    for (start, end) in (("C", "g"), ("g", "l"), ("l", "x")):
        x0, y0 = nodes[start][0], nodes[start][1]
        x1, y1 = nodes[end][0], nodes[end][1]
        axis.annotate("", xy=(x1 - 0.72, y1), xytext=(x0 + 0.72, y0),
                      arrowprops=dict(arrowstyle="-|>", color=INK,
                                      linewidth=1.3, shrinkA=0, shrinkB=0))
    axis.text(2.1, 3.35, "$\\pm$", ha="center", fontsize=9, color=RAIL)
    axis.text(4.6, 3.35, "$\\tau_\\ell$", ha="center", fontsize=9, color=RAIL)
    axis.text(7.3, 3.35, "$\\sigma_x$", ha="center", fontsize=9, color=RAIL)
    axis.text(
        5.0, 0.55,
        f"$\\mu_0={mixture.mu_0}$,  $\\tau_g={mixture.tau_g}$,  "
        f"$\\tau_\\ell={mixture.tau_l}$,  $\\sigma_x={mixture.sigma_x}$",
        ha="center", va="center", fontsize=9.5, color=INK,
    )
    axis.set_title("the generative model", fontsize=10, color=INK, loc="left")


def draw_chain_densities(axis, mixture):
    """``p(g)``, ``p(l)`` and ``p(x)``: the same mixture, blurred twice."""
    grid = np.linspace(-6.5, 6.5, 1201)

    def mixture_density(scale):
        density = np.zeros_like(grid)
        for weight, centre in zip(mixture.weights, mixture.components):
            density += weight * np.exp(
                -0.5 * (grid - centre) ** 2 / scale ** 2
            ) / np.sqrt(2.0 * np.pi * scale ** 2)
        return density

    widths = {
        "$p(g)$": mixture.tau_g,
        "$p(\\ell_i)$": np.sqrt(mixture.tau_g ** 2 + mixture.tau_l ** 2),
        "$p(x_i)$": np.sqrt(mixture.tau_g ** 2 + mixture.v),
    }
    styles = [(INK, "-", 1.9), (RAIL, "--", 1.6), (RAIL, ":", 1.8)]
    for (label, scale), (colour, dash, width) in zip(widths.items(), styles):
        axis.plot(grid, mixture_density(scale), color=colour, linestyle=dash,
                  linewidth=width, label=label)
    # The two prior components, so the mixture is visibly a mixture.
    for weight, centre, colour in zip(mixture.weights, mixture.components,
                                      (MINUS, PLUS)):
        component = weight * np.exp(
            -0.5 * (grid - centre) ** 2 / mixture.tau_g ** 2
        ) / np.sqrt(2.0 * np.pi * mixture.tau_g ** 2)
        axis.fill_between(grid, component, color=colour, alpha=0.18, linewidth=0)
    axis.set_xlabel("value", fontsize=9, color=INK)
    axis.set_ylabel("density", fontsize=9, color=INK)
    axis.set_title("the mixture blurs down the chain", fontsize=10, color=INK,
                   loc="left")
    axis.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    style(axis, grid=True)


def draw_chain_lines(axis, mixture, rng, count=250):
    """One line per simulated row: ``g``, then ``l_i``, then ``x_i``.

    The single most informative picture of this simulator. Each line starts in
    one of the two prior modes and wanders right; by the ``x`` column the two
    populations overlap, which is exactly the ambiguity a single observation
    leaves and the composition has to resolve.
    """
    rows = mixture.simulate(count, rng)
    positions = np.array([0.0, 1.0, 2.0])
    for row in rows:
        colour = PLUS if row[0] > 0 else MINUS
        axis.plot(positions, row, color=colour, alpha=0.28, linewidth=0.7,
                  solid_capstyle="round")
    for position, values in zip(positions, rows.T):
        axis.scatter(np.full(len(values), position), values, s=4,
                     color=[PLUS if v > 0 else MINUS for v in rows[:, 0]],
                     alpha=0.55, linewidths=0, zorder=3)
    axis.set_xticks(positions)
    axis.set_xticklabels(["$g$", "$\\ell_i$", "$x_i$"], fontsize=11, color=INK)
    axis.set_xlim(-0.25, 2.25)
    axis.set_ylabel("value", fontsize=9, color=INK)
    axis.set_title(f"{count} simulated rows", fontsize=10, color=INK, loc="left")
    style(axis)


def draw_joint(axis, mixture, rng, count=40_000):
    """The cloud the network is actually fitted to: ``(g, x)`` pairs."""
    rows = mixture.simulate(count, rng)
    axis.hexbin(rows[:, 0], rows[:, 2], gridsize=60, cmap="Greys", mincnt=1,
                linewidths=0)
    line = np.linspace(rows[:, 0].min(), rows[:, 0].max(), 10)
    axis.plot(line, line, color=PLUS, linewidth=1.4, linestyle="--",
              label="$E[x_i \\mid g] = g$")
    axis.set_xlabel("$g$", fontsize=10, color=INK)
    axis.set_ylabel("$x_i$", fontsize=10, color=INK)
    axis.set_title(f"joint of {count // 1000}k training rows", fontsize=10,
                   color=INK, loc="left")
    axis.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    style(axis)


def simulation_figure(output, mixture=DEFAULT, seed=0, draws=250):
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), layout="constrained")
    draw_graph(axes[0, 0], mixture)
    draw_chain_densities(axes[0, 1], mixture)
    draw_chain_lines(axes[1, 0], mixture, rng, count=draws)
    draw_joint(axes[1, 1], mixture, rng)
    figure.suptitle(
        "A hierarchical Gaussian mixture: one global $g$, one local $\\ell_i$ "
        "and one observation $x_i$ per row",
        fontsize=12, color=INK,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def regimes_figure(output, mixture=DEFAULT, observations=(1, 4, 16, 32),
                   seed=2026, branch_mass=0.75):
    """Rows = regime, columns = ``n``. Observations on top, exact posterior below."""
    regimes = datasets.REGIMES
    figure, axes = plt.subplots(
        len(regimes), len(observations), figsize=(3.0 * len(observations), 8.4),
        layout="constrained", sharex=True,
    )
    grid = np.linspace(-4.2, 4.2, 1601)
    for row, regime in enumerate(regimes):
        for column, n in enumerate(observations):
            axis = axes[row, column]
            rng = np.random.default_rng(
                [seed, datasets.REGIMES.index(regime), n, 0]
            )
            data = datasets.make_dataset(regime, n, rng, mixture=mixture,
                                         branch_mass=branch_mass)
            x = data["x"]
            density = mixture.global_density(grid, x)
            weights = mixture.branch_weights(x)
            axis.fill_between(grid, density, color=INK, alpha=0.12, linewidth=0)
            axis.plot(grid, density, color=INK, linewidth=1.7)
            # The prior, faint, so the shrinkage from prior to posterior reads.
            axis.plot(grid, mixture.prior_density(grid), color=RAIL,
                      linewidth=1.0, linestyle=":")
            top = axis.get_ylim()[1]
            axis.scatter(np.clip(x, grid[0], grid[-1]),
                         np.full(len(x), top * 0.93), marker="|", s=90,
                         color=[PLUS if v > 0 else MINUS for v in x],
                         linewidths=1.1, zorder=4)
            axis.set_ylim(0, top * 1.06)
            axis.text(0.02, 0.97,
                      f"$w_+={weights[1]:.2f}$", transform=axis.transAxes,
                      ha="left", va="top", fontsize=8.5, color=INK)
            if row == 0:
                axis.set_title(f"$n = {n}$", fontsize=11, color=INK)
            if column == 0:
                axis.set_ylabel(regime, fontsize=10, color=INK)
            if row == len(regimes) - 1:
                axis.set_xlabel("$g$", fontsize=10, color=INK)
            style(axis)
            axis.set_yticks([])
    figure.suptitle(
        "Exact $p(g \\mid x_{1:n})$ per regime (ticks: the observations; "
        "dotted: the prior)", fontsize=12, color=INK,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--draws", type=int, default=250,
                        help="Rows drawn for the parallel-coordinates panel.")
    result.add_argument("--observations", type=int, nargs="+",
                        default=[1, 4, 16, 32])
    result.add_argument("--branch-mass", type=float, default=0.75)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    output = Path(arguments.artifacts) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    print(simulation_figure(output / "simulation_data", seed=arguments.seed,
                            draws=arguments.draws))
    print(regimes_figure(output / "dataset_regimes",
                         observations=tuple(arguments.observations),
                         seed=arguments.seed,
                         branch_mass=arguments.branch_mass))
    for regime in datasets.REGIMES:
        for n in arguments.observations:
            rng = np.random.default_rng(
                [arguments.seed, datasets.REGIMES.index(regime), n, 0]
            )
            print(datasets.describe(
                datasets.make_dataset(regime, n, rng,
                                      branch_mass=arguments.branch_mass)
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
