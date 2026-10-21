#!/usr/bin/env python3
"""Figures for the shared (global) posterior under the exact marginal product.

Reads only what ``composition_attribution.py`` and
``evaluate_marginal_product.py`` already wrote -- no model, no GPU, no sampling
-- so the figures are guaranteed to be of the numbers in those tables rather
than of a fresh run that would differ by Monte Carlo noise.

Three figures:

* ``01_global_posteriors`` -- the shared posterior each method returns, laid over
  the ABC reference, one panel per dataset. This is the headline: it shows the
  Gaussian rules handing back a near-symmetric density where the reference is
  lopsided.
* ``02_product_construction`` -- how the product is built. The eight
  single-observation posteriors are individually almost uninformative about
  which branch is right; multiplied, they are decisive. This is what a mean and
  a covariance per observation cannot carry.
* ``03_diagnostics`` -- branch mass, width ratio and W1 across every dataset,
  plus the endpoint error of the locals.

Usage:
    python inverse_kinematics/experiments/plot_marginal_product.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plots import INK, RAIL, _style_axis  # noqa: E402

# The composition rules keep the colours they carry everywhere else in this
# benchmark; the two exact curves are deliberately achromatic so a reader never
# mistakes a referee for a method.
COLOURS = {
    "langevin_fnpse": "#2A9D8F",
    "gauss_hierarchical": "#E76F51",
    "gauss_jacobian": "#5E60CE",
    "marginal_product": "#B5179E",
    "network_product": "#6C757D",
    "reference_product": "#ADB5BD",
}
LABELS = {
    "gauss_jacobian": "DPM2 + gauss_jacobian",
    "gauss_hierarchical": "DPM2 + gauss_hierarchical",
    "langevin_fnpse": "Langevin + F-NPSE",
    "marginal_product": "marginal product (new)",
    "network_product": "network marginals, composed exactly",
}
RULES = ("gauss_jacobian", "gauss_hierarchical", "langevin_fnpse")
CASES = (
    ("selected_n8_seed99", "selected |x1|<0.15, n=8, seed 99"),
    ("selected_n8_seed101", "selected |x1|<0.15, n=8, seed 101"),
    ("selected_n16_seed99", "selected |x1|<0.15, n=16, seed 99"),
    ("exact_zero_n8_seed99", "x1 = 0 exactly, n=8  (symmetric control)"),
    ("prior_n8_seed3", "prior predictive, n=8  (no selection)"),
)


def load(directory):
    curves = np.load(directory / "curves.npz")
    product = np.load(directory / "marginal_product_curve.npz")
    metrics = json.loads((directory / "marginal_product.json").read_text())
    attribution = json.loads((directory / "metrics.json").read_text())
    return curves, product, metrics, attribution


def support(grid, density, floor=2e-3, pad=0.12):
    """x-limits that hold the reference, so tight posteriors are not a spike."""
    keep = np.flatnonzero(density > floor * density.max())
    low, high = grid[keep[0]], grid[keep[-1]]
    margin = pad * max(high - low, 1e-6)
    return low - margin, high + margin


def plot_global_posteriors(cases, output):
    figure, axes = plt.subplots(
        len(cases), 1, figsize=(7.6, 2.5 * len(cases)), layout="constrained"
    )
    for axis, (directory, title) in zip(np.atleast_1d(axes), cases):
        curves, product, metrics, _ = load(directory)
        grid = curves["grid"]
        reference = curves["joint_reference"]
        axis.fill_between(grid, reference, color=INK, alpha=0.13, zorder=1)
        axis.plot(grid, reference, color=INK, linewidth=2.0, zorder=5,
                  label="ABC reference")
        for name in RULES:
            row = metrics["methods"][name]
            axis.plot(grid, curves[f"curve_{name}"], color=COLOURS[name],
                      linewidth=1.4, alpha=0.95, zorder=3,
                      label=f"{LABELS[name]}   W1/$\\sigma$ "
                            f"{row['w1_over_sigma']:.3f}")
        row = metrics["methods"]["marginal_product"]
        axis.plot(grid, product["density"], color=COLOURS["marginal_product"],
                  linewidth=2.4, zorder=4,
                  label=f"{LABELS['marginal_product']}   W1/$\\sigma$ "
                        f"{row['w1_over_sigma']:.3f}")
        axis.axvline(0.0, color=RAIL, linewidth=0.8, linestyle=":", zorder=0)
        axis.set_xlim(*support(grid, reference))
        # Headroom, so the legend sits above the curves rather than on them.
        peak = max([reference.max(), product["density"].max()]
                   + [curves[f"curve_{n}"].max() for n in RULES])
        axis.set_ylim(0.0, 1.55 * peak)
        axis.set_title(title, color=INK, fontsize=9, loc="left")
        axis.set_ylabel("density", color=INK, fontsize=8)
        _style_axis(axis)
        # Put the legend over whichever half of the panel carries less density,
        # so a lopsided posterior never has its taller mode written over.
        stack = np.vstack([reference, product["density"]]
                          + [curves[f"curve_{n}"] for n in RULES])
        left, right = axis.get_xlim()
        middle = 0.5 * (left + right)
        busier_left = (stack[:, grid <= middle].max()
                       > stack[:, grid > middle].max())
        axis.legend(frameon=False, fontsize=7, labelcolor=INK,
                    loc="upper right" if busier_left else "upper left")
    np.atleast_1d(axes)[-1].set_xlabel(
        "shared angle_1", color=INK, fontsize=9
    )
    figure.suptitle(
        "Shared posterior p(angle_1 | x_1..x_n): the Gaussian rules return a "
        "near-symmetric density\nwhere the reference is lopsided",
        color=INK, fontsize=10,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_product_construction(directory, title, output):
    curves, product, metrics, _ = load(directory)
    grid = curves["grid"]
    reference = curves["joint_reference"]
    per_observation = product["per_observation"]
    figure, axes = plt.subplots(
        2, 1, figsize=(7.6, 5.4), layout="constrained", sharex=True
    )

    axis = axes[0]
    for index, density in enumerate(per_observation):
        axis.plot(grid, density, color=COLOURS["marginal_product"],
                  linewidth=1.0, alpha=0.45,
                  label="p(angle_1 | x_i), one per observation"
                        if index == 0 else None)
    axis.plot(grid, curves["prior"], color=RAIL, linewidth=1.6, linestyle="--",
              label="prior p(angle_1)")
    axis.set_title(
        f"{title}: each observation on its own is nearly uninformative "
        "about the branch", color=INK, fontsize=9, loc="left")
    axis.set_ylabel("density", color=INK, fontsize=8)
    axis.set_ylim(bottom=0.0)
    _style_axis(axis)
    axis.legend(frameon=False, fontsize=7, labelcolor=INK)

    axis = axes[1]
    axis.fill_between(grid, reference, color=INK, alpha=0.13)
    axis.plot(grid, reference, color=INK, linewidth=2.0, label="ABC reference")
    axis.plot(grid, product["density"], color=COLOURS["marginal_product"],
              linewidth=2.4, label=LABELS["marginal_product"])
    axis.plot(grid, curves["curve_gauss_jacobian"],
              color=COLOURS["gauss_jacobian"], linewidth=1.4,
              label=LABELS["gauss_jacobian"])
    axis.axvline(0.0, color=RAIL, linewidth=0.8, linestyle=":")
    axis.set_title(
        "multiplied (and divided by the prior n-1 times), they are decisive",
        color=INK, fontsize=9, loc="left")
    axis.set_xlabel("shared angle_1", color=INK, fontsize=9)
    axis.set_ylabel("density", color=INK, fontsize=8)
    axis.set_ylim(bottom=0.0)
    axis.set_xlim(*support(grid, reference))
    _style_axis(axis)
    axis.legend(frameon=False, fontsize=7, labelcolor=INK)

    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_diagnostics(cases, output):
    names = list(RULES) + ["marginal_product"]
    figure, axes = plt.subplots(4, 1, figsize=(8.4, 9.0), layout="constrained")
    positions = np.arange(len(cases))
    width = 0.19

    panels = [
        ("positive_mass", "mass on the positive branch",
         "the reference value is the black bar; equal branches is the failure"),
        ("width_ratio", "posterior width / reference width", None),
        ("w1_over_sigma", "W1 / reference sigma   (lower is better)", None),
        ("endpoint_median", "endpoint error of the locals (median)",
         "reference-free: a correct draw reproduces its own observation"),
    ]
    for axis, (key, label, note) in zip(axes, panels):
        for offset, name in enumerate(names):
            values = []
            for directory, _ in cases:
                _, _, metrics, _ = load(directory)
                values.append(metrics["methods"][name][key])
            axis.bar(positions + (offset - 1.5) * width, values, width * 0.92,
                     color=COLOURS[name], label=LABELS[name] if key ==
                     "positive_mass" else None)
        if key == "positive_mass":
            for index, (directory, _) in enumerate(cases):
                curves, _, _, _ = load(directory)
                grid, reference = curves["grid"], curves["joint_reference"]
                mass = float(np.trapezoid(reference[grid >= 0], grid[grid >= 0]))
                axis.plot([positions[index] - 2 * width,
                           positions[index] + 2 * width], [mass, mass],
                          color=INK, linewidth=2.0,
                          label="ABC reference" if index == 0 else None)
        if key == "width_ratio":
            axis.axhline(1.0, color=INK, linewidth=1.2, linestyle="--")
        axis.set_xticks(positions)
        axis.set_xticklabels([t.replace(", ", "\n") for _, t in cases],
                             fontsize=6.5, color=INK)
        axis.set_ylabel(label, color=INK, fontsize=8)
        if note:
            axis.set_title(note, color=RAIL, fontsize=7.5, loc="left")
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK, ncol=2)
    figure.suptitle(
        "Composition diagnostics across five datasets", color=INK, fontsize=10)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--attribution", type=Path,
                        default=ROOT / "artifacts/composition_attribution")
    result.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/marginal_product_figures")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    arguments.output.mkdir(parents=True, exist_ok=True)
    cases = [(arguments.attribution / name, title) for name, title in CASES]
    missing = [d.name for d, _ in cases
               if not (d / "marginal_product.json").exists()]
    if missing:
        raise SystemExit(
            "No marginal_product.json in: " + ", ".join(missing) +
            "\nRun evaluate_marginal_product.py on those directories first."
        )
    written = [
        plot_global_posteriors(cases, arguments.output / "01_global_posteriors.png"),
        plot_product_construction(
            cases[0][0], cases[0][1],
            arguments.output / "02_product_construction.png"),
        plot_diagnostics(cases, arguments.output / "03_diagnostics.png"),
    ]
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
