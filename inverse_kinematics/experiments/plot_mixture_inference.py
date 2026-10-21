#!/usr/bin/env python3
"""Global posterior, locals on the line, and cost -- for every benchmark.

Three panels per benchmark, all from saved arrays:

1. **Global posterior** as a density, against the benchmark's own reference
   (ABC for inverse kinematics, the exact closed-form Gaussian for the
   shared/local hierarchy).
2. **Locals on the line** -- one vertical interval per observation showing the
   local posterior's median and central 68%, with the truth marked. This is the
   check that a method cannot pass by getting the shared coordinate right while
   returning locals that do not reproduce the data.
3. **Cost against quality** -- seconds per method next to W1/sigma.
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from plots import INK, RAIL, _style_axis  # noqa: E402

METHODS = ["gauss_hierarchical", "gauss_jacobian", "langevin_fnpse",
           "mixture_product"]
LABELS = {"gauss_hierarchical": "Gaussian hierarchical",
          "gauss_jacobian": "Gaussian Jacobian",
          "langevin_fnpse": "Langevin F-NPSE",
          "mixture_product": "mixture product (K inferred)"}
COLOURS = {"gauss_hierarchical": "#A6761D", "gauss_jacobian": "#D95F02",
           "langevin_fnpse": "#7570B3", "mixture_product": "#1B9E77"}


def panel_global(axis, grid, reference, curves, table, title):
    axis.plot(grid, reference, color=INK, linewidth=2.4, label="reference")
    for name in METHODS:
        if name not in curves:
            continue
        axis.plot(grid, curves[name], color=COLOURS[name],
                  linewidth=2.1 if name == "mixture_product" else 1.5,
                  linestyle="-" if name == "mixture_product" else (0, (5, 2)),
                  label=f"{LABELS[name]} · {table[name]['w1_over_sigma']:.3f}")
    _style_axis(axis)
    axis.set_title(title, fontsize=10, color=INK)
    axis.set_xlabel("shared parameter", color=INK)
    axis.set_ylabel("density", color=INK)
    stack = [reference] + [curves[m] for m in METHODS if m in curves]
    axis.set_ylim(0.0, 1.42 * max(float(np.max(c)) for c in stack))
    axis.legend(frameon=False, fontsize=7.4, loc="upper left")


def panel_locals(axis, locals_by_method, truth, title, max_observations=16):
    """One interval per observation per method: median and central 68%."""
    count = min(locals_by_method[METHODS[0]].shape[0], max_observations)
    offsets = np.linspace(-0.28, 0.28, len(METHODS))
    for offset, name in zip(offsets, METHODS):
        draws = locals_by_method[name][:count]
        median = np.median(draws, axis=1)
        low = np.quantile(draws, 0.16, axis=1)
        high = np.quantile(draws, 0.84, axis=1)
        positions = np.arange(count) + offset
        axis.vlines(positions, low, high, color=COLOURS[name], linewidth=2.0,
                    alpha=0.85, label=LABELS[name])
        axis.plot(positions, median, "o", color=COLOURS[name], markersize=2.6)
    if truth is not None:
        axis.plot(np.arange(count), np.asarray(truth)[:count], "_",
                  color=INK, markersize=13, markeredgewidth=1.8, label="truth")
    _style_axis(axis)
    axis.set_title(title, fontsize=10, color=INK)
    axis.set_xlabel("observation", color=INK)
    axis.set_ylabel("local posterior", color=INK)
    axis.legend(frameon=False, fontsize=7.0, loc="upper left", ncol=2)


def panel_cost(axis, table, quality_key, quality_label):
    positions = np.arange(len(METHODS))
    seconds = [table[m]["seconds"] for m in METHODS]
    axis.bar(positions, seconds, 0.6,
             color=[COLOURS[m] for m in METHODS])
    for position, value in zip(positions, seconds):
        axis.text(position, value, f"{value:.0f}s", ha="center",
                  va="bottom", fontsize=8, color=INK)
    axis.set_xticks(positions)
    axis.set_xticklabels(["gauss\nhier.", "gauss\njacobian", "Langevin\nF-NPSE",
                          "mixture\n(K inferred)"], fontsize=7.6)
    axis.set_ylabel("seconds", color=INK)
    twin = axis.twinx()
    twin.plot(positions, [table[m][quality_key] for m in METHODS], "D--",
              color=INK, markersize=5, linewidth=1.2)
    twin.set_ylabel(quality_label, color=INK)
    twin.spines["top"].set_visible(False)
    _style_axis(axis)
    axis.set_title("cost, with quality overlaid (diamonds)", fontsize=10,
                   color=INK)


def inverse_kinematics(arguments):
    source = ROOT / "artifacts" / "mixture_inference"
    with np.load(source / "inverse_kinematics.npz") as stored:
        grid, reference = stored["grid"], stored["reference"]
        curves = {m: stored[f"curve_{m}"] for m in METHODS
                  if f"curve_{m}" in stored}
        samples = {m: stored[f"samples_{m}"] for m in METHODS
                   if f"samples_{m}" in stored}
    table = json.loads((source / "inverse_kinematics.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.4), layout="constrained")
    panel_global(axes[0], grid, reference, curves, table,
                 "inverse kinematics — global posterior over angle₁")
    panel_locals(axes[1], {m: samples[m][:, :, 2] for m in METHODS}, None,
                 "locals on the line — angle₂ per observation")
    panel_cost(axes[2], table, "endpoint_mean", "endpoint error (lower better)")
    figure.suptitle(
        "Inverse kinematics, n = 8 — K inferred per observation at inference "
        f"time: {table['mixture_product']['diagnostics']['chosen_k']}",
        fontsize=11.5, color=INK)
    output = source / "01_inverse_kinematics.png"
    figure.savefig(output, dpi=205, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def shared_local(label, title):
    source = (ROOT.parent / "tutorials" / "Compositional_Inference_Fix"
              / "New_Attempt" / "mixture_inference")
    with np.load(source / f"{label}.npz") as stored:
        grid, reference = stored["grid"], stored["reference"]
        curves = {m: stored[f"curve_{m}"] for m in METHODS
                  if f"curve_{m}" in stored}
        locals_ = {m: stored[f"locals_{m}"] for m in METHODS
                   if f"locals_{m}" in stored}
        exact_mean = stored["exact_mean"]
    table = json.loads((source / f"{label}.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.4), layout="constrained")
    panel_global(axes[0], grid, reference, curves, table,
                 f"{title} — global posterior")
    panel_locals(axes[1], locals_, exact_mean[1:],
                 "locals on the line — exact posterior mean marked")
    panel_cost(axes[2], table, "local_mean_error", "local mean error")
    chosen = table["mixture_product"]["chosen_k"]
    figure.suptitle(
        f"{title} — K inferred at inference time: "
        f"all {len(chosen)} observations chose K = "
        f"{sorted(set(chosen)) if len(set(chosen)) > 1 else chosen[0]}",
        fontsize=11.5, color=INK)
    output = source / f"01_{label}.png"
    figure.savefig(output, dpi=205, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", nargs="+",
                        default=["ik", "gaussian_mixture_base", "noise_0.171"])
    arguments = parser.parse_args(argv)
    for item in arguments.what:
        if item == "ik":
            inverse_kinematics(arguments)
        elif item == "gaussian_mixture_base":
            shared_local(item, "shared/local Gaussian mixture hierarchy")
        else:
            shared_local(item, f"shared/local Gaussian, obs noise "
                               f"{item.split('_')[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
