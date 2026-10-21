#!/usr/bin/env python3
"""Plot fixed-setting Gaussian-composition scaling on the symmetric arm task.

Reads the saved ``gauss_scaling_e10000/n{N}/metrics.json`` files created with
the frozen e10000 checkpoint.  Every dataset is constructed at ``x1 = 0``, so
the reference posterior is exactly symmetric and the correct positive-branch
mass is one half.
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

from plots import INK, RAIL, _style_axis  # noqa: E402
from plots_bimodal import METHOD_COLOURS  # noqa: E402

ROOT = Path(__file__).resolve().parent
COUNTS = (1, 2, 4, 8, 16)
METHODS = ("gauss_hierarchical", "gauss_jacobian")
LABELS = {
    "gauss_hierarchical": "Gaussian hierarchical",
    "gauss_jacobian": "Gaussian Jacobian",
}


def load_rows(source):
    """Read metrics into one row per composition rule and observation count."""
    rows = {method: [] for method in METHODS}
    for count in COUNTS:
        path = source / f"n{count}" / "metrics.json"
        payload = json.loads(path.read_text())
        if not payload.get("exact_x1_zero"):
            raise ValueError(f"{path} is not an exact-symmetry experiment.")
        for method in METHODS:
            metric = payload["methods"][method]
            rows[method].append({"observations": count, **metric})
    return rows


def plot(rows, output):
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), layout="constrained")
    panels = (
        ("w1_over_sigma", "W1 / reference $\\sigma$", 0.0,
         "Distributional error"),
        ("fraction_positive", r"$P(\mathrm{angle}_1 > 0)$", 0.5,
         "Positive-branch mass"),
        ("dip_ratio", "Dip ratio", 0.0, "Mode separation"),
    )
    for axis, (key, ylabel, target, title) in zip(axes, panels):
        axis.axhline(target, color=RAIL, linewidth=1.15, linestyle=(0, (3, 2)),
                    label="exact target" if key == "w1_over_sigma" else None)
        for method, method_rows in rows.items():
            x = [row["observations"] for row in method_rows]
            y = [row[key] for row in method_rows]
            axis.plot(x, y, marker="o", markersize=4.8, linewidth=1.9,
                      color=METHOD_COLOURS[method], label=LABELS[method])
        _style_axis(axis)
        axis.set_xscale("log", base=2)
        axis.set_xticks(COUNTS)
        axis.set_xticklabels([str(value) for value in COUNTS])
        axis.set_xlabel("Observations sharing $\\mathrm{angle}_1$", color=INK)
        axis.set_ylabel(ylabel, color=INK)
        axis.set_title(title, fontsize=10, color=INK)
    axes[0].set_ylim(bottom=-0.02)
    axes[1].set_ylim(0.45, 0.66)
    axes[2].set_ylim(bottom=-0.03)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(
        "Gaussian composition error compounds with observations — exact symmetry, "
        "e10000, DPM2 80 steps, churn $\\eta=2$",
        fontsize=11, color=INK,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "artifacts" / "gauss_scaling_e10000")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "figures" /
                        "18_gaussian_composition_scaling.png")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {plot(load_rows(args.source), args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
