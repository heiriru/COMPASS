#!/usr/bin/env python3
"""Bar-chart comparison of every method in combined_method_metrics.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAVY, BLUE, TEAL, CORAL, GOLD, GREY = (
    "#17223B", "#3A86FF", "#2A9D8F", "#EF476F", "#FFB703", "#8D99AE",
)
LABELS = {
    "dpm2_gaussian": "DPM2 + Gaussian\n(no hierarchy correction)",
    "langevin_fnpe": "Langevin + F-NPSE",
    "dpm2_gauss_global_local": "DPM2 + Gauss_global_local\n(oracle moments -- cheat)",
    "dpm2_gauss_moment": "DPM2 + moment projection\n(oracle moments -- cheat)",
    "gauss_hierarchical_dense_correctors": "gauss_hierarchical\n(dense correctors, ours)",
    "gauss_hierarchical_deterministic": "gauss_hierarchical\n(deterministic, ours)",
}
COLORS = {
    "dpm2_gaussian": GREY,
    "langevin_fnpe": GOLD,
    "dpm2_gauss_global_local": CORAL,
    "dpm2_gauss_moment": CORAL,
    "gauss_hierarchical_dense_correctors": BLUE,
    "gauss_hierarchical_deterministic": TEAL,
}
ORDER = list(LABELS)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_rows(ROOT / "combined_method_metrics.csv")
    by_method = {row["method"]: {} for row in rows}
    for row in rows:
        by_method[row["method"]][row["parameter"]] = row
    methods = [m for m in ORDER if m in by_method]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = range(len(methods))
    colors = [COLORS[m] for m in methods]
    labels = [LABELS[m] for m in methods]

    global_err = [float(by_method[m]["global"]["mean_error_in_exact_std"]) for m in methods]
    axes[0].bar(x, global_err, color=colors)
    axes[0].set(title="Global parameter: mean error / exact std", ylabel="|error| / analytic σ")
    axes[0].axhline(0.1057, color=GOLD, ls="--", lw=1.2, label="Langevin F-NPSE")
    axes[0].legend(fontsize=8)

    local_err = [float(by_method[m]["locals_mean"]["mean_error_in_exact_std"]) for m in methods]
    axes[1].bar(x, local_err, color=colors)
    axes[1].set(title="30 local parameters: mean |error| / exact std", ylabel="|error| / analytic σ")

    runtime = [float(by_method[m]["global"]["runtime_seconds"]) for m in methods]
    axes[2].bar(x, runtime, color=colors)
    axes[2].set(title="Wall-clock runtime", ylabel="seconds")

    for axis in axes:
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Hierarchical GAUSS composition: true compositional (network-only) score modeling\n"
        "vs. oracle-moment cheats and existing baselines -- 30 observations, shared/local Gaussian model",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    fig.tight_layout()
    fig.savefig(ROOT / "03_method_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ROOT / '03_method_comparison.png'}")


if __name__ == "__main__":
    main()
