#!/usr/bin/env python3
"""What the mixture product actually looks like, next to the reference.

Four panels, all from saved arrays -- no model, no sampler, no GPU:

1. **What K means.** One observation's shared marginal, with the K=1 and K=2
   mixture fits drawn over the draws they were fitted to. This is the whole
   argument in one picture: the single Gaussian puts its mean in the valley
   between the two branches.
2. **What K does to the composed posterior.** The product of all n
   per-observation mixtures at K = 1..4, against the ABC reference.
3. **Against the rules.** The reference, the three composition rules, the exact
   grid product, and the best mixture.
4. **What pruning costs.** K=4 at beam 64 / 256 / 1024 against the exact
   enumeration of all K^n components.

Usage:
    python plot_mixture_product.py
    python plot_mixture_product.py --dataset selected_n16_seed99
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
from composition_attribution import grid_product, prior_density, w1_grid, moments  # noqa: E402
from mixture_product import (  # noqa: E402
    beam_product, density_on_grid, exact_product, fit_mixture,
)

ATTRIBUTION = ROOT / "artifacts" / "composition_attribution"
OUTPUT = ROOT / "artifacts" / "mixture_product"

COMPONENT_COLOURS = {1: "#D95F02", 2: "#1B9E77", 3: "#7570B3", 4: "#E7298A"}
RULE_COLOURS = {
    "gauss_jacobian": "#D95F02",
    "gauss_hierarchical": "#A6761D",
    "langevin_fnpse": "#7570B3",
}
RULE_LABELS = {
    "gauss_jacobian": "Gaussian Jacobian",
    "gauss_hierarchical": "Gaussian hierarchical",
    "langevin_fnpse": "Langevin F-NPSE",
}
BEAM_COLOURS = {64: "#CDB4DB", 256: "#8E7DBE", 1024: "#3F3D8F"}


def mixture_density(grid, weights, means, variances):
    """A fitted mixture tabulated on the grid, for display."""
    return (weights[None, :]
            / np.sqrt(2.0 * np.pi * variances[None, :])
            * np.exp(-0.5 * (grid[:, None] - means[None, :]) ** 2
                     / variances[None, :])).sum(axis=1)


def headroom(axis, curves, factor=1.35):
    axis.set_ylim(0.0, factor * max(float(np.max(c)) for c in curves))


def build(dataset, arguments):
    with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
        grid = stored["grid"]
        reference = stored["joint_reference"]
        draws = stored["single_draws"]
        per_observation_network = stored["per_observation_network"]
        rules = {name: stored[f"curve_{name}"] for name in RULE_LABELS
                 if f"curve_{name}" in stored}
    prior = prior_density(grid)
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    n = draws.shape[0]
    reference_sd = moments(grid, reference)["sd"]

    def score(curve):
        return w1_grid(grid, curve, reference) / reference_sd

    fits = {k: [fit_mixture(draws[j], k, seed=arguments.seed + j)
                for j in range(n)]
            for k in (1, 2, 3, 4)}

    composed, labels = {}, {}
    for k, mixtures in fits.items():
        combinations = int(np.prod([len(m[0]) for m in mixtures]))
        if combinations <= arguments.max_exact:
            components = exact_product(mixtures, prior_precision)
            labels[k] = f"K = {k} (exact, {combinations:,} comps)"
        else:
            components = beam_product(mixtures, prior_precision, 1024)[:3]
            labels[k] = f"K = {k} (beam 1024)"
        composed[k] = density_on_grid(grid, *components)

    beams = {}
    for beam in (64, 256, 1024):
        beams[beam] = density_on_grid(
            grid, *beam_product(fits[4], prior_precision, beam)[:3]
        )

    return {
        "grid": grid, "reference": reference, "draws": draws,
        "per_observation": per_observation_network, "rules": rules,
        "grid_product": grid_product(grid, per_observation_network, prior),
        "fits": fits, "composed": composed, "labels": labels, "beams": beams,
        "score": score, "n": n,
    }


def plot(data, dataset, output):
    grid = data["grid"]
    reference = data["reference"]
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.6), layout="constrained")

    # --- 1. What K means, on a single observation ------------------------------
    axis = axes[0, 0]
    # The observation whose own posterior is least symmetric, since that is the
    # one a single Gaussian misrepresents most.
    index = int(np.argmax([abs(np.mean(row > 0) - 0.5) for row in data["draws"]]))
    row = data["draws"][index]
    axis.hist(row, bins=90, density=True, color=RAIL, alpha=0.42,
              label=f"draws from $p(a\\,|\\,x_{{{index + 1}}})$")
    for k in (1, 2):
        weights, means, variances = data["fits"][k][index]
        axis.plot(grid, mixture_density(grid, weights, means, variances),
                  color=COMPONENT_COLOURS[k], linewidth=2.0,
                  label=f"K = {k} fit")
        if k == 2:
            for w, m, v in zip(weights, means, variances):
                axis.plot(grid, w / np.sqrt(2 * np.pi * v)
                          * np.exp(-0.5 * (grid - m) ** 2 / v),
                          color=COMPONENT_COLOURS[k], linewidth=0.9,
                          linestyle=(0, (4, 2)), alpha=0.85)
                axis.annotate(f"w = {w:.2f}", (m, 0.06), fontsize=7.4,
                              color=COMPONENT_COLOURS[k], ha="center")
    axis.set_title("1 · What K means: one observation's posterior",
                   fontsize=10, color=INK)
    axis.legend(frameon=False, fontsize=7.6, loc="upper left")

    # --- 2. What K does to the composed posterior ------------------------------
    axis = axes[0, 1]
    axis.plot(grid, reference, color=INK, linewidth=2.3, label="ABC reference")
    for k in (1, 2, 3, 4):
        axis.plot(grid, data["composed"][k], color=COMPONENT_COLOURS[k],
                  linewidth=1.6,
                  label=f"{data['labels'][k]}  ·  W1/σ {data['score'](data['composed'][k]):.3f}")
    axis.set_title(f"2 · Composed posterior, n = {data['n']} observations",
                   fontsize=10, color=INK)
    axis.legend(frameon=False, fontsize=7.4, loc="upper left")

    # --- 3. Against the rules --------------------------------------------------
    axis = axes[1, 0]
    axis.plot(grid, reference, color=INK, linewidth=2.3, label="ABC reference")
    for name, curve in data["rules"].items():
        axis.plot(grid, curve, color=RULE_COLOURS[name], linewidth=1.5,
                  linestyle=(0, (5, 2)),
                  label=f"{RULE_LABELS[name]}  ·  {data['score'](curve):.3f}")
    axis.plot(grid, data["grid_product"], color="#666666", linewidth=1.5,
              label=f"grid product  ·  {data['score'](data['grid_product']):.3f}")
    best = min((1, 2, 3, 4), key=lambda k: data["score"](data["composed"][k]))
    axis.plot(grid, data["composed"][best], color=COMPONENT_COLOURS[best],
              linewidth=2.0,
              label=f"mixture K = {best}  ·  {data['score'](data['composed'][best]):.3f}")
    axis.set_title("3 · Against the composition rules", fontsize=10, color=INK)
    axis.legend(frameon=False, fontsize=7.4, loc="upper left")

    # --- 4. What pruning costs -------------------------------------------------
    axis = axes[1, 1]
    axis.plot(grid, reference, color=INK, linewidth=2.3, label="ABC reference")
    axis.plot(grid, data["composed"][4], color="#E7298A", linewidth=2.0,
              label=f"K = 4 exact  ·  {data['score'](data['composed'][4]):.3f}")
    for beam, curve in data["beams"].items():
        axis.plot(grid, curve, color=BEAM_COLOURS[beam], linewidth=1.4,
                  linestyle=(0, (4, 2)),
                  label=f"K = 4, beam {beam}  ·  {data['score'](curve):.3f}")
    axis.set_title("4 · What pruning costs (K = 4)", fontsize=10, color=INK)
    axis.legend(frameon=False, fontsize=7.4, loc="upper left")

    for axis in axes.ravel():
        axis.axvline(0.0, color=RAIL, linewidth=0.8, zorder=0)
        _style_axis(axis)
        axis.set_xlabel(r"shared $\mathrm{angle}_1$", color=INK)
        axis.set_ylabel("density", color=INK)
        axis.set_xlim(grid.min(), grid.max())
    headroom(axes[0, 0], [np.histogram(data["draws"][index], bins=90,
                                       density=True)[0]])
    headroom(axes[0, 1], [reference] + [data["composed"][k] for k in (1, 2, 3, 4)])
    headroom(axes[1, 0], [reference, data["grid_product"]] + list(data["rules"].values()))
    headroom(axes[1, 1], [reference, data["composed"][4]] + list(data["beams"].values()))

    figure.suptitle(
        f"Gaussian-mixture composition on {dataset} — "
        "legend values are W1 in units of the reference sigma (lower is better)",
        fontsize=11.5, color=INK,
    )
    figure.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="selected_n8_seed99")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-exact", type=int, default=300_000)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = arguments.output or OUTPUT / f"01_mixture_{arguments.dataset}.png"
    data = build(arguments.dataset, arguments)
    print(f"Wrote {plot(data, arguments.dataset, output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
