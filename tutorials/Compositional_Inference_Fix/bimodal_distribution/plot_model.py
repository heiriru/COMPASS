#!/usr/bin/env python3
"""Figures 1 and 2: what the model is, and where the bimodality comes from.

These need no trained network and no sampler -- every curve is a closed form
from ``mixture_reference`` -- so they can be drawn before, and independently of,
any inference run.

``01_<name>_model.png``      the generative process, the dataset, the n locally
                             ambiguous row posteriors, and the one globally
                             coordinated posterior they compose into.
``02_<name>_emergence.png``  the mechanism: which discrete configurations carry
                             the mass, how the valley deepens with n, how it
                             depends on tau/sigma_y, and how the row posterior's
                             own two modes merge as the diffusion noise grows.

Usage:
    python plot_model.py                 # both benchmarks
    python plot_model.py --models sign
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
import csv  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mixture_reference as reference  # noqa: E402
import style  # noqa: E402

ARTIFACTS = ROOT / "artifacts"

N_LADDER = (1, 2, 4, 8, 16)
FORWARD_DRAWS = 200_000


def shared_window(module, y, pad=1.9):
    """A g-axis window that frames both modes without wasting half the panel.

    The mixture's own standard deviation is dominated by the *gap* between the
    modes, so a fixed number of those would leave the interesting structure in
    the middle sixth of the axis.
    """
    weights, means, variance = reference.marginal(module, y, 0)
    reach = float(np.abs(means).max()) + pad * float(np.sqrt(variance))
    return -reach, reach


def merge_lambda(weights, centres, variance, upper=12.0, steps=60):
    """The VE noise level at which a 1-D mixture stops having two modes.

    A symmetric two-component mixture is bimodal exactly while the total
    variance is small against the gap; the transition is monotone in lambda, so
    a bisection on "does the peak-normalized density still dip at the midpoint"
    finds it to any precision. This is the level above which a diffusion sampler
    has no two branches left to keep apart, whatever its composition rule does.
    """
    reach = float(np.abs(centres).max()) + 3.0 * float(np.sqrt(variance + upper**2))
    grid = np.linspace(-reach, reach, 4001)

    def bimodal(lam):
        density = reference.component_density(
            grid, weights, centres, variance + lam**2
        )
        interior = ((density[1:-1] > density[:-2])
                    & (density[1:-1] >= density[2:]))
        return int(interior.sum()) >= 2

    if not bimodal(0.0):
        return 0.0
    low, high = 0.0, float(upper)
    if bimodal(high):
        return float("inf")
    for _ in range(steps):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if bimodal(middle) else (low, middle)
    return 0.5 * (low + high)


def dataset(module, count, seed):
    truth, local_truth, discrete, y = module.observations(count, seed)
    return {
        "truth": float(truth), "local_truth": np.asarray(local_truth),
        "discrete": np.asarray(discrete), "y": np.asarray(y, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Figure 1: the model and the dataset
# ---------------------------------------------------------------------------

def figure_model(module, data, path, table_path):
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    figure.suptitle(module.TITLE, fontsize=13, color=style.INK, y=0.98)
    y = data["y"]
    n = len(y)

    # -- A: the forward law, as the simulator actually draws it ------------
    axis = axes[0, 0]
    generator = torch.Generator().manual_seed(3)
    theta, observed = module.simulate(FORWARD_DRAWS, generator)
    shared = theta[:, 0].numpy()
    observed = observed[:, 0].numpy()
    limit = float(np.percentile(np.abs(np.concatenate([shared, observed])), 99.5))
    axis.hexbin(shared, observed, gridsize=110, cmap="Blues", bins="log",
                extent=(-limit, limit, -limit, limit), linewidths=0.0)
    axis.axhline(0.0, color=style.MUTED, lw=0.8, ls=":")
    axis.axvline(0.0, color=style.MUTED, lw=0.8, ls=":")
    axis.set_xlabel("shared parameter $g$")
    axis.set_ylabel("observation $y$")
    axis.set_title("A. The forward law, 200k draws")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    if module.NAME == "sign":
        axis.text(0.03, 0.97, "the hidden sign splits every $g$ into\n"
                              "two branches $y \\approx \\pm g$",
                  transform=axis.transAxes, va="top", ha="left", fontsize=8.5,
                  color=style.SECONDARY)
    else:
        axis.text(0.03, 0.97, "two prior lobes, no ambiguity:\n"
                              "$y$ tracks $g$ on both",
                  transform=axis.transAxes, va="top", ha="left", fontsize=8.5,
                  color=style.SECONDARY)
    style.despine(axis)

    # -- B: the one dataset every arm is graded on -------------------------
    axis = axes[0, 1]
    rows = np.arange(1, n + 1)
    axis.axvline(data["truth"], color=style.INK, lw=1.6, zorder=4,
                 label=f"true $g$ = {data['truth']:+.2f}")
    axis.axvline(-data["truth"], color=style.INK, lw=1.0, ls="--", zorder=4,
                 label="its mirror image $-g$")
    axis.axvline(0.0, color=style.MUTED, lw=0.8, ls=":")
    for row, (local, value) in enumerate(zip(data["local_truth"], y), start=1):
        axis.plot([local, value], [row, row], color=style.GRID, lw=1.2, zorder=1)
    axis.scatter(data["local_truth"], rows, s=34, facecolor="none",
                 edgecolor=style.SECONDARY, lw=1.2, zorder=3,
                 label="true local $\\ell_j$")
    axis.scatter(y, rows, s=40, color=style.BLUE, zorder=3,
                 label="observed $y_j$")
    axis.set_yticks(rows)
    axis.set_ylabel("observation $j$")
    axis.set_xlabel("value")
    axis.set_title(f"B. The dataset: n = {n} observations")
    axis.invert_yaxis()
    lower, upper = axis.get_xlim()
    axis.set_xlim(lower - 0.42 * (upper - lower), upper)
    axis.legend(loc="lower left", fontsize=8)
    style.despine(axis)

    # -- C: every row on its own -- ambiguous ------------------------------
    axis = axes[1, 0]
    reach, peak = 0.0, 0.0
    for index, value in enumerate(y):
        grid_g, mass_g, _, _ = reference.row_reference(module, value)
        density = mass_g / (grid_g[1] - grid_g[0])
        axis.plot(grid_g, density, color=style.BLUE, lw=1.3, alpha=0.55,
                  label="one row, $p(g \\mid y_j)$" if index == 0 else None)
        # A row posterior is far wider than the composed one, so this panel gets
        # its own window; the composed panel's would clip every tail.
        weights_row, means_row, covariance_row = module.row_mixture(value)
        reach = max(reach, float(np.abs(means_row[:, 0]).max())
                    + 2.6 * float(np.sqrt(covariance_row[0, 0])))
        peak = max(peak, float(density.max()))
    axis.axvline(0.0, color=style.MUTED, lw=0.8, ls=":")
    axis.axvline(data["truth"], **style.TRUTH_KWARGS, label="true $g$")
    axis.set_xlabel("shared parameter $g$")
    axis.set_ylabel("density")
    axis.set_title("C. Each observation alone")
    axis.set_xlim(-reach, reach)
    axis.set_ylim(0.0, 1.42 * peak)
    axis.legend(loc="upper right")
    style.despine(axis)
    note = ("every row is two-moded and says\nnothing about the sign"
            if module.NAME == "sign" else
            "every row is two-moded too -- but they are\n"
            "the prior's lobes, and each row already leans")
    axis.text(0.02, 0.97, note, transform=axis.transAxes, va="top",
              fontsize=8.5, color=style.SECONDARY, zorder=6)

    # -- D: all n rows composed -- coordinated -----------------------------
    axis = axes[1, 1]
    grid, density, _, _, _ = reference.shared_reference(module, y)
    axis.fill_between(grid, density, color=style.REFERENCE_FILL, zorder=1)
    axis.plot(grid, density, color=style.INK, lw=1.8, zorder=2,
              label="exact $p(g \\mid y_1 \\dots y_n)$")
    axis.axvline(data["truth"], color=style.INK, lw=1.2, ls="--", zorder=3,
                 label=f"true $g$ = {data['truth']:+.2f}")
    axis.axvline(0.0, color=style.MUTED, lw=0.8, ls=":")
    for mode in reference.mode_locations(module, y):
        axis.annotate(f"{mode:+.2f}", xy=(mode, np.interp(mode, grid, density)),
                      xytext=(0, 6), textcoords="offset points", ha="center",
                      fontsize=8.5, color=style.SECONDARY)
    axis.set_xlabel("shared parameter $g$")
    axis.set_ylabel("density")
    axis.set_title("D. All n composed")
    axis.set_xlim(*shared_window(module, y, pad=4.0))
    axis.set_ylim(0.0, 1.62 * float(density.max()))
    axis.legend(loc="upper right")
    style.despine(axis)
    weights_shared, means_shared, _ = reference.marginal(module, y, 0)
    detail = (
        f"coherent configurations: "
        f"{100 * reference.aligned_mass(module, y):.1f}% of the mass"
        if module.NAME == "sign" else
        "lobe weights: "
        + " / ".join(f"{value:.2f}" for value in weights_shared)
    )
    axis.text(0.02, 0.97,
              f"valley at $g = 0$: {reference.valley_depth(module, y):.1f} nats deep\n"
              + detail,
              transform=axis.transAxes, va="top", fontsize=8.5,
              color=style.SECONDARY)

    style.caption(figure, (
        "Panels C and D are the whole point: n rows that are individually "
        "ambiguous about the sign compose into one globally coordinated "
        "two-moded posterior, not into the 2^n-moded product that treating the "
        "rows as independent would give."
        if module.NAME == "sign" else
        "Panels C and D are the contrast case: the two modes were already in "
        "the prior and each row already leans to one of them, so composition "
        "only has to avoid destroying a bimodality it was handed."
    ), y=0.0)
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)

    with table_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation", "true_local", "observed_y",
                         "hidden_discrete"])
        for index in range(n):
            writer.writerow([index + 1, data["local_truth"][index], y[index],
                             data["discrete"][index]])
    print(f"Wrote {path}\nWrote {table_path}")


# ---------------------------------------------------------------------------
# Figure 2: the mechanism
# ---------------------------------------------------------------------------

def panel_a(module, axis, y, data):
    """Where the discrete mass sits -- a different question in each benchmark.

    In the sign model there are 2^n sign configurations and the interesting fact
    is that two of them dominate, so they are ranked on a log axis. In the
    control there are only ever two components and they were *given* by the
    prior, so the honest panel is the prior itself beside the posterior it
    produced: nothing had to be built.
    """
    if module.NAME == "sign":
        n = len(y)
        weights, _, _ = module.mixture(y)
        coherent = module.coherent_components(y)
        order = np.argsort(-weights)
        ranks = np.arange(1, len(weights) + 1)
        axis.bar(ranks[~coherent[order]], weights[order][~coherent[order]],
                 width=1.0, color=style.MUTED, label="mixed configurations")
        axis.bar(ranks[coherent[order]], weights[order][coherent[order]],
                 width=1.0, color=style.BLUE, label="coherent: all rows agree")
        axis.set_yscale("log")
        axis.set_xlabel("configuration, ranked by posterior weight")
        axis.set_ylabel("weight $w_s$")
        axis.set_title(f"A. {len(weights)} discrete configurations")
        axis.set_ylim(max(weights.min(), 1e-12) * 0.5, 2.0)
        axis.legend(loc="upper right")
        axis.text(0.30, 0.55,
                  "the two coherent ones hold\n"
                  f"{100 * reference.aligned_mass(module, y):.1f}% of the mass",
                  transform=axis.transAxes, fontsize=8.5, color=style.SECONDARY)
        style.despine(axis)
        return

    grid, density, _, _, _ = reference.shared_reference(module, y)
    prior = 0.5 * (
        np.exp(-0.5 * (grid - module.A) ** 2 / module.SIGMA_0**2)
        + np.exp(-0.5 * (grid + module.A) ** 2 / module.SIGMA_0**2)
    ) / math.sqrt(2 * math.pi * module.SIGMA_0**2)
    axis.plot(grid, prior, color=style.MUTED, lw=2.0, label="prior $p(g)$")
    axis.fill_between(grid, density, color=style.REFERENCE_FILL, zorder=1)
    axis.plot(grid, density, color=style.INK, lw=1.8, zorder=2,
              label="posterior $p(g \\mid y)$")
    axis.axvline(0.0, color=style.MUTED, lw=0.8, ls=":")
    axis.set_xlabel("shared parameter $g$")
    axis.set_ylabel("density")
    axis.set_title("A. The modes were already in the prior")
    axis.set_xlim(*shared_window(module, y, pad=4.0))
    axis.set_ylim(top=1.35 * float(max(density.max(), prior.max())))
    axis.legend(loc="upper center")
    axis.text(0.03, 0.97,
              "nothing has to be constructed here --\n"
              "the data only reweights two lobes\nthat the prior already had",
              transform=axis.transAxes, va="top", fontsize=8.5,
              color=style.SECONDARY)
    style.despine(axis)


def panel_c(module, axis, y, n, seed):
    """How fragile the bimodality is -- again a different question per benchmark.

    Sign model: sweep the pooling strength ``tau / sigma_y``, which is what sets
    how strongly the hierarchy insists the signs align.

    Control: sweep the *data*. The lobe weights depend on the observations only
    through their mean, so the posterior stays two-moded only while ``ybar`` sits
    in a narrow window around zero -- which is the reason this benchmark's
    dataset has to be chosen rather than simulated blind, and the reason it is a
    weaker test than the sign model, where every dataset is bimodal.
    """
    if module.NAME == "sign":
        ratios = np.geomspace(0.4, 12.0, 25)
        saved = module.TAU
        depths, masses = [], []
        try:
            for ratio in ratios:
                module.TAU = float(ratio * module.SIGMA_Y)
                _, _, _, subset = module.observations(n, seed)
                depths.append(reference.valley_depth(module, subset))
                masses.append(reference.aligned_mass(module, subset))
            shipped = saved / module.SIGMA_Y
        finally:
            module.TAU = saved
        depth_line, = axis.plot(ratios, depths, color=style.BLUE,
                                lw=style.ESTIMATE_LW,
                                label="valley depth (nats)")
        marker = axis.axvline(
            shipped, color=style.INK, lw=1.2, ls="--",
            label=f"this benchmark, $\\tau/\\sigma_y$ = {shipped:.1f}",
        )
        axis.axhline(0.0, color=style.MUTED, lw=0.8, ls=":")
        axis.set_xscale("log")
        axis.set_xlabel("pooling strength $\\tau / \\sigma_y$  "
                        "(weak $\\rightarrow$)")
        axis.set_ylabel("valley depth (nats)")
        axis.set_title("C. Tighter pooling, cleaner bimodality")
        # The right axis is a second *unit* for the same sweep, not a second
        # scale on the same quantity, and both curves are named in one legend so
        # neither can be read off the wrong side.
        twin = axis.twinx()
        mass_line, = twin.plot(ratios, masses, color=style.ORANGE,
                               lw=style.ESTIMATE_LW, ls="--",
                               label="coherent mass (right axis)")
        twin.set_ylabel("coherent mass", color=style.ORANGE)
        twin.tick_params(axis="y", colors=style.ORANGE)
        twin.grid(False)
        twin.set_ylim(-0.02, 1.02)
        axis.legend(handles=[depth_line, mass_line, marker], loc="upper right")
        style.despine(axis)
        return

    centred = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    offsets = np.linspace(-2.5, 2.5, 161)
    weights, bimodal = [], []
    for offset in offsets:
        subset = centred + offset
        component, _, _ = module.mixture(subset)
        weights.append(float(component[1]))
        # "Two modes" by the same prominence rule the sampled draws are graded
        # with -- a second peak carrying under 5% of the taller one is not a
        # mode a sampler could be blamed for missing.
        bimodal.append(len(reference.mode_locations(
            module, subset, prominence=0.05
        )) >= 2)
    weights = np.array(weights)
    bimodal = np.array(bimodal)
    weight_line, = axis.plot(offsets, weights, color=style.BLUE,
                             lw=style.ESTIMATE_LW,
                             label="weight of the $+$ lobe")
    axis.fill_between(offsets, 0, 1, where=bimodal, color=style.REFERENCE_FILL,
                      zorder=0, label="posterior still has two modes")
    marker = axis.axvline(float(np.mean(y)), color=style.INK, lw=1.2, ls="--",
                          label=f"this dataset, $\\bar{{y}}$ = "
                                f"{float(np.mean(y)):+.2f}")
    axis.set_xlabel("mean observation $\\bar{y}$ (data shifted rigidly)")
    axis.set_ylabel("posterior weight of the $+$ lobe")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("C. How narrow the bimodal window is")
    axis.legend(handles=[weight_line, marker], loc="upper left", fontsize=8)
    style.despine(axis)
    width = (float(offsets[bimodal].max() - offsets[bimodal].min())
             if bimodal.any() else 0.0)
    axis.text(0.97, 0.30,
              f"only a window of width {width:.1f} in $\\bar{{y}}$ leaves both\n"
              "modes standing; outside it the data resolves the\n"
              "lobe and there is no bimodality left to preserve",
              transform=axis.transAxes, va="bottom", ha="right", fontsize=8.5,
              color=style.SECONDARY)


def figure_emergence(module, data, path, table_path, seed):
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    figure.suptitle(f"{module.TITLE} -- where the two modes come from",
                    fontsize=13, color=style.INK, y=0.98)
    y = data["y"]
    n = len(y)

    panel_a(module, axis=axes[0, 0], y=y, data=data)
    panel_c(module, axis=axes[1, 0], y=y, n=n, seed=seed)

    # -- B: the valley deepens with n --------------------------------------
    axis = axes[0, 1]
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(N_LADDER)))
    ladder = []
    for colour, count in zip(shades, N_LADDER):
        # Nested datasets: the same seed draws the same g and the same first
        # ``count`` locals, so this is one dataset being extended, not five.
        _, _, _, subset = module.observations(count, seed)
        grid, density, _, _, _ = reference.shared_reference(module, subset)
        depth = reference.valley_depth(module, subset)
        axis.plot(grid, density, color=colour, lw=1.8,
                  label=f"n = {count}  ({depth:.1f} nats)")
        ladder.append((count, depth, reference.aligned_mass(module, subset)))
    axis.axvline(0.0, color=style.MUTED, lw=0.8, ls=":")
    axis.set_xlabel("shared parameter $g$")
    axis.set_ylabel("density")
    axis.set_title("B. Adding rows, and the valley they open"
                   if module.NAME == "sign"
                   else "B. More rows, and one lobe simply wins")
    axis.set_xlim(*shared_window(module, y, pad=4.0))
    axis.set_ylim(top=1.60 * axis.get_ylim()[1])
    axis.legend(loc="upper right", ncol=1, fontsize=8)
    style.despine(axis)
    if module.NAME == "sign":
        axis.text(0.02, 0.97,
                  "the valley deepens while the added\n"
                  "rows are informative, then stops: by\n"
                  "n = 16 several $|y_j|$ are near zero and\n"
                  "almost sign-indifferent (coherent mass\n"
                  "0.96 at n = 8, 0.27 at n = 16)",
                  transform=axis.transAxes, va="top", fontsize=8,
                  color=style.SECONDARY)

    # -- D: the row posterior's own two modes, under diffusion -------------
    axis = axes[1, 1]
    value = float(y[int(np.argmax(np.abs(y)))])
    weights_row, means_row, covariance = module.row_mixture(value)
    centres, variance = means_row[:, 0], float(covariance[0, 0])
    merged = merge_lambda(weights_row, centres, variance)
    lambdas = (0.0, 0.3, 0.6, 1.0, round(merged, 2), 2.5)
    reach = float(np.abs(centres).max()) + 2.4 * float(
        np.sqrt(variance + max(lambdas) ** 2)
    )
    grid_row = np.linspace(-reach, reach, 2001)
    shades = plt.get_cmap("Blues")(np.linspace(0.95, 0.30, len(lambdas)))
    for colour, lam in zip(shades, lambdas):
        density_row = reference.component_density(
            grid_row, weights_row, centres, variance + lam**2
        )
        axis.plot(grid_row, density_row / density_row.max(), color=colour,
                  lw=1.8, ls="--" if abs(lam - merged) < 5e-3 else "-",
                  label=f"$\\lambda$ = {lam:g}"
                        + (" (merge)" if abs(lam - merged) < 5e-3 else ""))
    axis.scatter(centres, [0.0, 0.0], s=30, color=style.INK, zorder=5,
                 label="the two row modes")
    axis.set_xlabel("diffused shared coordinate $g_\\lambda$")
    axis.set_ylabel("density (peak-normalized)")
    axis.set_title(f"D. One row, $y_j$ = {value:+.2f}, under VE diffusion")
    axis.set_ylim(0.0, 1.55)
    axis.legend(loc="upper left", fontsize=8)
    style.despine(axis)
    axis.text(0.98, 0.98,
              "this row's two modes merge into one blob\n"
              f"at $\\lambda$ = {merged:.2f}; above that noise level\n"
              "the composition rule has no two branches\nto keep apart",
              transform=axis.transAxes, va="top", ha="right", fontsize=8.5,
              color=style.SECONDARY)

    style.caption(figure, (
        "A and B are the ferromagnetic coupling: coherent configurations "
        "dominate, and each informative row deepens the gap. D is why that is "
        "hard for a diffusion sampler -- at the noise levels where the "
        "trajectory is decided, the two branches are not yet separate."
        if module.NAME == "sign" else
        "A and C are why this is the weaker of the two tests: the modes were "
        "handed over by the prior, and they only survive into the posterior for "
        "datasets whose mean lands in a narrow window. D still applies -- the "
        "row's two modes merge under diffusion here too."
    ), y=0.0)
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)

    with table_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observations", "valley_depth_nats", "coherent_mass"])
        writer.writerows(ladder)
    print(f"Wrote {path}\nWrote {table_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["sign", "control"])
    arguments = parser.parse_args()

    import run_experiment as experiment

    style.apply()
    for name in arguments.models:
        module = experiment.MODELS[name]
        seed = experiment.DATA_SEEDS[name]
        data = dataset(module, experiment.OBSERVATIONS, seed)
        output = ARTIFACTS / name
        output.mkdir(parents=True, exist_ok=True)
        figure_model(module, data, output / f"01_{name}_model.png",
                     output / f"01_{name}_dataset.csv")
        figure_emergence(module, data, output / f"02_{name}_emergence.png",
                         output / f"02_{name}_ladder.csv", seed)


if __name__ == "__main__":
    main()
