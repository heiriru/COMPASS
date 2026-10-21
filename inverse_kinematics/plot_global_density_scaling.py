#!/usr/bin/env python3
"""Compare global-angle densities as the number of observations grows.

The reference is the independent numerical marginalisation saved by
``run_bimodal.py``.  Samples are smoothed only for display; all reported W1
and dip measurements remain sample-based in ``metrics.json``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from hierarchical import HIERARCHY  # noqa: E402
from plots import INK, RAIL, _style_axis  # noqa: E402
from plots_bimodal import METHOD_COLOURS  # noqa: E402

ROOT = Path(__file__).resolve().parent
COUNTS = (1, 2, 4, 8, 16)
TRUE_ANGLE = 0.6
METHODS = ("langevin_fnpse", "langevin_fnpse_arruda_damped",
           "gauss_hierarchical", "gauss_jacobian")
LABELS = {
    "langevin_fnpse": "Langevin F-NPSE",
    "langevin_fnpse_arruda_damped": "F-NPSE + Arruda damping",
    "gauss_hierarchical": "Gaussian hierarchical",
    "gauss_jacobian": "Gaussian Jacobian",
}


def smoothed_density(samples, grid):
    """A display-only KDE with a robust fallback for degenerate samples."""
    samples = np.asarray(samples, dtype=float)
    try:
        return gaussian_kde(samples)(grid)
    except np.linalg.LinAlgError:
        scale = max(float(np.std(samples)), 1e-3)
        bandwidth = max(0.06 * scale, 2e-3)
        return np.mean(np.exp(-0.5 * ((grid[:, None] - samples) / bandwidth) ** 2),
                       axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def plot(gaussian_source, fnpse_source, damped_source, output):
    figure, axes = plt.subplots(1, len(COUNTS), figsize=(15.2, 3.25),
                                sharey=True, layout="constrained")
    for axis, count in zip(axes, COUNTS):
        gaussian = np.load(gaussian_source / f"n{count}" / "posteriors.npz")
        fnpse = np.load(fnpse_source / f"n{count}" / "posteriors.npz")
        damped = np.load(damped_source / f"n{count}" / "posteriors.npz")
        grid, reference = gaussian["grid"], gaussian["density"]
        if not np.allclose(grid, fnpse["grid"]) or not np.allclose(grid, damped["grid"]):
            raise ValueError(f"incompatible grids at N={count}")
        axis.plot(grid, reference, color=INK, linewidth=2.15,
                  label="numerical reference" if count == COUNTS[0] else None)
        for method, source in (("langevin_fnpse", fnpse),
                               ("langevin_fnpse_arruda_damped", damped),
                               ("gauss_hierarchical", gaussian),
                               ("gauss_jacobian", gaussian)):
            samples = source[f"samples_{method}"][0, :, HIERARCHY[0]]
            axis.plot(grid, smoothed_density(samples, grid), linewidth=1.55,
                      color=METHOD_COLOURS[method],
                      label=LABELS[method] if count == COUNTS[0] else None)
        axis.axvline(0, color=RAIL, linewidth=.8, zorder=0)
        for angle in (-TRUE_ANGLE, TRUE_ANGLE):
            axis.axvline(
                angle, color=RAIL, linewidth=1.0, linestyle=(0, (3, 2)), zorder=0,
                label=("true / reflected parameter" if count == COUNTS[0]
                       and angle > 0 else None),
            )
        _style_axis(axis)
        axis.set_title(f"N = {count}", fontsize=10, color=INK)
        axis.set_xlabel(r"global $\mathrm{angle}_1$", color=INK)
    axes[0].set_ylabel("density", color=INK)
    axes[0].legend(frameon=False, fontsize=7.8, loc="upper left")
    figure.suptitle(
        "Global posterior under composition — exact symmetry, e10000, 80 steps",
        fontsize=11, color=INK,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-source", type=Path,
                        default=ROOT / "artifacts" / "gauss_scaling_e10000")
    parser.add_argument("--fnpse-source", type=Path,
                        default=ROOT / "artifacts" / "fnpse_scaling_e10000")
    parser.add_argument("--damped-source", type=Path,
                        default=ROOT / "artifacts" /
                        "fnpse_arruda_damped_scaling_e10000")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "figures" /
                        "20_global_density_scaling_arruda_damping.png")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {plot(args.gaussian_source, args.fnpse_source, args.damped_source, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
