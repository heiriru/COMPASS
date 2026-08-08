#!/usr/bin/env python3
"""Create publication-style dashboards for COMPASS global/local validation.

The script is plotting-only: it reads saved CSV/NPZ outputs and never trains or
samples. Generate the exact-score CSV first with tests/test_multiobs_analytic.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "compositional_inference_local_vs_global"
DEFAULT_SHARED = ROOT / "output" / "compositional_inference" / "06b_shared_local"
DEFAULT_HIERARCHY = ROOT / "output" / "compositional_inference" / "08_miniexperiment"
NAVY, BLUE, TEAL, CORAL, GOLD = "#17223B", "#3A86FF", "#2A9D8F", "#EF476F", "#FFB703"


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 240, "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: mpl.figure.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def mean(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values))


def plot_exact_score(csv_path: Path, output: Path) -> None:
    rows = read_rows(csv_path)
    n = np.asarray([int(row["n_observations"]) for row in rows])
    error = np.asarray([float(row["mean_error_in_analytic_std"]) for row in rows])
    ratio = np.asarray([float(row["posterior_std_ratio"]) for row in rows])
    limit = float(rows[0]["mean_error_limit"])
    lower = float(rows[0]["std_ratio_lower_limit"])
    upper = float(rows[0]["std_ratio_upper_limit"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25))
    fig.suptitle("Exact-score validation: one global parameter, many local nuisances",
                 fontsize=15, fontweight="bold", color=NAVY)
    axes[0].axhspan(0, limit, color=TEAL, alpha=0.10, label="acceptance region")
    axes[0].axhline(limit, color=TEAL, ls="--", lw=1.4)
    axes[0].plot(n, error, "o-", color=BLUE, lw=2.4, ms=7)
    for x, y in zip(n, error):
        axes[0].annotate(f"{y:.2f}σ", (x, y), xytext=(0, 9),
                         textcoords="offset points", ha="center", color=NAVY)
    axes[0].set(xscale="log", xlabel="observations N",
                ylabel="absolute mean error / analytic σ", title="Posterior location")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left")

    axes[1].axhspan(lower, upper, color=TEAL, alpha=0.10, label="acceptance region")
    axes[1].axhline(1.0, color=NAVY, ls=":", lw=1.5, label="analytic width")
    axes[1].plot(n, ratio, "o-", color=CORAL, lw=2.4, ms=7)
    for x, y in zip(n, ratio):
        axes[1].annotate(f"{y:.2f}×", (x, y), xytext=(0, 9),
                         textcoords="offset points", ha="center", color=NAVY)
    axes[1].set(xscale="log", xlabel="observations N",
                ylabel="COMPASS σ / analytic σ", title="Posterior uncertainty")
    axes[1].legend(loc="upper left")
    fig.text(0.5, -0.01,
             "The score is analytic, so deviations isolate composition and sampling error.",
             ha="center", color="#52616B")
    save(fig, output)


def plot_shared_local(
    npz_path: Path,
    output: Path,
    title: str = "Learned-score validation against the exact joint posterior",
) -> None:
    data = np.load(npz_path)
    exact = data["exact_joint_mean"]
    covariance = data["exact_joint_covariance"]
    global_truth = float(data["global_truth"])
    global_samples = data["compass_global_samples"]
    local_samples = data["compass_local_samples"]
    local_mean = local_samples.mean(axis=1)
    local_std = local_samples.std(axis=1, ddof=1)
    exact_local_std = np.sqrt(np.diag(covariance)[1:])
    indices = np.arange(len(local_mean))
    x_observed = np.asarray(data["x_observed"], dtype=float).reshape(-1)

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.3))
    fig.suptitle(title, fontsize=15, fontweight="bold", color=NAVY)

    # Leftmost panel: the raw observations themselves, stacked as a dot plot
    # in observational (x) space -- i.e. what actually went into inference,
    # before any composition into the shared/local parameter space shown by
    # the other three panels.
    n_bins = min(15, len(x_observed))
    counts, edges = np.histogram(x_observed, bins=n_bins)
    bin_idx = np.clip(np.digitize(x_observed, edges[1:-1]), 0, n_bins - 1)
    stack_height = np.zeros(n_bins, dtype=int)
    stack_position = np.empty(len(x_observed), dtype=int)
    for i in np.argsort(x_observed):
        b = bin_idx[i]
        stack_position[i] = stack_height[b]
        stack_height[b] += 1
    centers = (edges[:-1] + edges[1:]) / 2
    axes[0].scatter(centers[bin_idx], stack_position + 0.5, s=80, color=BLUE,
                    alpha=0.85, edgecolor="white", linewidth=0.6)
    axes[0].axvline(x_observed.mean(), color="red", linestyle=":", linewidth=2,
                    label="mean observation")
    axes[0].set(xlabel="observed x", ylabel="observations stacked per bin",
               title="Observations (observational space)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    axes[1].hist(global_samples, bins=46, density=True, color=BLUE, alpha=0.72,
                 label="COMPASS")
    grid = np.linspace(exact[0] - 4*np.sqrt(covariance[0, 0]),
                       exact[0] + 4*np.sqrt(covariance[0, 0]), 400)
    density = np.exp(-0.5*((grid-exact[0])**2/covariance[0, 0])) / np.sqrt(2*np.pi*covariance[0, 0])
    axes[1].plot(grid, density, color=NAVY, ls="--", lw=2, label="exact")
    axes[1].axvline(
        global_truth, color="red", linestyle=":", linewidth=2,
        label="true global parameter",
    )
    # Fix the x-range to the analytic +/-4 sigma band (identical across every
    # method run against the same reference) instead of letting it autoscale
    # to this run's own samples -- a few tail draws would otherwise stretch
    # one method's plot wider than another's, making side-by-side comparison
    # misleading.
    axes[1].set_xlim(grid[0], grid[-1])
    axes[1].set(xlabel="global parameter g", ylabel="density", title="Shared posterior")
    axes[1].legend()

    axes[2].errorbar(indices, local_mean, yerr=local_std, fmt="o", ms=4,
                     color=CORAL, ecolor="#F4A3B4", alpha=0.9, label="COMPASS mean ± σ")
    axes[2].plot(indices, exact[1:], "_", ms=9, color=NAVY, label="exact mean")
    axes[2].set(xlabel="observation", ylabel="local parameter ℓᵢ", title="Thirty local posteriors")
    axes[2].legend()

    axes[3].scatter(exact[1:], local_mean, c=indices, cmap="viridis", s=38,
                    edgecolor="white", linewidth=0.5)
    lo = min(exact[1:].min(), local_mean.min())
    hi = max(exact[1:].max(), local_mean.max())
    axes[3].plot([lo, hi], [lo, hi], color=NAVY, ls="--", lw=1.6)
    mae_sigma = np.mean(np.abs(local_mean-exact[1:]) / exact_local_std)
    axes[3].text(0.04, 0.94, f"mean |error| = {mae_sigma:.2f} analytic σ",
                 transform=axes[3].transAxes, va="top", color=NAVY)
    axes[3].set(xlabel="exact local posterior mean", ylabel="COMPASS posterior mean",
                title="Local recovery")
    save(fig, output)


def plot_hierarchy(csv_path: Path, output: Path) -> None:
    from tutorials.hierarchy_dashboard import plot_hierarchy_dashboard
    plot_hierarchy_dashboard(csv_path, output)


def plot_hierarchy_pairplot(output_dir: Path) -> None:
    from tutorials.hierarchy_dashboard import plot_model_pairplot
    plot_model_pairplot(
        output_dir / "04_linear_quadratic_model_pairplot.png",
        output_dir / "04_linear_quadratic_model_pairplot_data.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exact-score-csv", type=Path,
                        default=DEFAULT_OUTPUT / "analytic_exact_score_metrics.csv")
    parser.add_argument("--shared-local-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--hierarchy-dir", type=Path, default=DEFAULT_HIERARCHY)
    args = parser.parse_args()
    configure_style()
    if args.exact_score_csv.exists():
        plot_exact_score(args.exact_score_csv, args.output_dir / "01_exact_score_global_validation.png")
    else:
        print(f"Skipping exact-score plot; missing {args.exact_score_csv}")
    plot_shared_local(args.shared_local_dir / "raw_plot_data.npz",
                      args.output_dir / "02_learned_score_global_local_validation.png")
    plot_hierarchy(args.hierarchy_dir / "hierarchical_linear_quadratic_detailed.csv",
                   args.output_dir / "03_hierarchy_validation_dashboard.png")
    plot_hierarchy_pairplot(args.output_dir)


if __name__ == "__main__":
    main()
