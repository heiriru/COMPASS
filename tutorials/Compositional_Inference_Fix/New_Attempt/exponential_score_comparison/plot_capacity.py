#!/usr/bin/env python3
"""Turn artifacts/capacity.csv into the two figures the sizing decision rests on.

Left half -- **how small**: both fidelity metrics against parameter count, at a
fixed 200K simulations, so the only thing varying is the architecture.
Right half -- **how much data**: the same metrics against the simulation budget
at the chosen architecture, so the only thing varying is the training set.

The pass line drawn on both is ``W1/sigma = 0.05``: a network whose
single-observation marginals sit within a twentieth of a posterior standard
deviation of the closed form. That is the level at which composition, not the
network, becomes the thing being measured -- which is the whole point of the
checkpoint this directory trains.

Usage:
    python plot_capacity.py                 # both panels, config chosen by --config
    python plot_capacity.py --config h8d1
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

NAVY, BLUE, TEAL, CORAL, GOLD = "#17223B", "#3A86FF", "#2A9D8F", "#EF476F", "#FFB703"
PASS_W1 = 0.05
LAMBDAS = (0.05, 0.1, 0.3, 1.0, 3.0)


def read(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key != "config":
                row[key] = float(value) if value not in ("", None) else float("nan")
    return rows


def configure_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 240, "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="h8d1",
                        help="Architecture whose data ladder fills the right panels.")
    parser.add_argument("--reference-samples", type=int, default=200_000,
                        help="Simulation budget held fixed in the size panels.")
    parser.add_argument("--csv", type=Path, default=ARTIFACTS / "capacity.csv")
    parser.add_argument("--output", type=Path,
                        default=ARTIFACTS / "00_capacity.png")
    arguments = parser.parse_args()

    configure_style()
    rows = read(arguments.csv)
    size_rows = sorted(
        [row for row in rows if row["train_samples"] == arguments.reference_samples],
        key=lambda row: row["parameters"],
    )
    data_rows = sorted(
        [row for row in rows if row["config"] == arguments.config],
        key=lambda row: row["train_samples"],
    )

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.5))
    fig.suptitle(
        "Sizing the score network for g → ℓⱼ → xⱼ "
        "(single-observation fidelity; no composition in the loop)",
        fontsize=14, fontweight="bold", color=NAVY,
    )

    # Panel 1 -- distributional fidelity against capacity.
    parameters = [row["parameters"] for row in size_rows]
    axes[0].plot(parameters, [row["w1_g"] for row in size_rows], "o-",
                 color=BLUE, lw=2, label="W1 of p(g | x$_j$)")
    axes[0].plot(parameters, [row["w1_l"] for row in size_rows], "s-",
                 color=CORAL, lw=2, label="W1 of p(ℓ | x$_j$)")
    axes[0].axhline(PASS_W1, color=NAVY, ls=":", lw=1.4)
    axes[0].text(parameters[0], PASS_W1 * 1.1, f"pass line {PASS_W1:g} σ",
                 color=NAVY, fontsize=8, va="bottom")
    for row in size_rows:
        axes[0].annotate(row["config"], (row["parameters"], row["w1_g"]),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color=NAVY)
    axes[0].set(xscale="log", yscale="log", xlabel="parameters",
                ylabel="W1 / posterior σ",
                title=f"Capacity: fidelity\n({arguments.reference_samples:,} simulations)")
    axes[0].legend(fontsize=8.5)

    # Panel 2 -- score fidelity against capacity, resolved by noise level. This
    # is the metric the composition rules actually consume.
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(size_rows)))
    for row, colour in zip(size_rows, colours):
        axes[1].plot(LAMBDAS, [row[f"score_rmse_lam{lam:g}"] for lam in LAMBDAS],
                     "o-", lw=2, ms=4, color=colour,
                     label=f"{row['config']} ({row['parameters']/1e3:.0f}K)")
    axes[1].set(xscale="log", yscale="log", xlabel="noise level λ",
                ylabel="relative RMS score error",
                title="Capacity: score error\n(exact single-observation target)")
    axes[1].legend(fontsize=8)

    # Panel 3 -- distributional fidelity against the simulation budget.
    budgets = [row["train_samples"] for row in data_rows]
    axes[2].plot(budgets, [row["w1_g"] for row in data_rows], "o-", color=BLUE,
                 lw=2, label="W1 of p(g | x$_j$)")
    axes[2].plot(budgets, [row["w1_l"] for row in data_rows], "s-", color=CORAL,
                 lw=2, label="W1 of p(ℓ | x$_j$)")
    axes[2].axhline(PASS_W1, color=NAVY, ls=":", lw=1.4)
    axes[2].set(xscale="log", yscale="log", xlabel="training simulations",
                ylabel="W1 / posterior σ",
                title=f"Data: fidelity\n({arguments.config})")
    axes[2].legend(fontsize=8.5)

    # Panel 4 -- score fidelity against the simulation budget.
    colours = plt.cm.magma(np.linspace(0.15, 0.75, len(data_rows)))
    for row, colour in zip(data_rows, colours):
        axes[3].plot(LAMBDAS, [row[f"score_rmse_lam{lam:g}"] for lam in LAMBDAS],
                     "o-", lw=2, ms=4, color=colour,
                     label=f"{row['train_samples']/1000:.0f}K sims")
    axes[3].set(xscale="log", yscale="log", xlabel="noise level λ",
                ylabel="relative RMS score error",
                title=f"Data: score error\n({arguments.config})")
    axes[3].legend(fontsize=8)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(arguments.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
