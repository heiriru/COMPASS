#!/usr/bin/env python3
"""Benchmark the shared-posterior methods across integration grids.

The script writes one directory per grid point, a CSV of analytic global-posterior
errors, and an accuracy-versus-runtime figure.  It reuses the controlled problem,
pilot moments, checkpoint, and plotting convention of
``compare_shared_local_composition_methods.py``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_shared_local_composition_methods import (
    build_single_observation_covariance_bank,
    load_result,
    load_score_checkpoint,
    run_method,
)

ROOT = Path(__file__).resolve().parent
TUTORIALS_ROOT = ROOT if (ROOT / "output" / "compositional_inference").exists() else ROOT.parent
DEFAULT_REFERENCE = TUTORIALS_ROOT / "output" / "compositional_inference" / "06b_shared_local" / "raw_plot_data.npz"
DEFAULT_CHECKPOINT = TUTORIALS_ROOT / "output" / "compositional_inference" / "models" / "shared_local_mixture_full" / "Model_checkpoint.pt"

METHODS = (
    "dpm2_gaussian",
    "langevin_fnpe",
    "dpm2_gaussian_without_correctors",
    "dpm2_gauss_global_local",
    "dpm2_gauss_global_local_without_correctors",
)
LABELS = {
    "dpm2_gaussian": "DPM2 + Gaussian",
    "langevin_fnpe": "Langevin + F-NPSE",
    "dpm2_gaussian_without_correctors": "DPM2 + Gaussian (no correctors)",
    "dpm2_gauss_global_local": "DPM2 + global/local Gaussian",
    "dpm2_gauss_global_local_without_correctors": (
        "DPM2 + global/local Gaussian (no correctors)"
    ),
}


def parse_timesteps(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("timesteps must be comma-separated integers >= 2")
    return values


def global_error(raw: dict[str, np.ndarray]) -> float:
    analytic_mean = float(raw["exact_joint_mean"][0])
    analytic_std = float(np.sqrt(raw["exact_joint_covariance"][0, 0]))
    return abs(float(raw["compass_global_samples"].mean()) - analytic_mean) / analytic_std


def plot(rows: list[dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 6.2), constrained_layout=True)
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: int(row["timesteps"]))
        xs = [float(row["runtime_seconds"]) for row in method_rows]
        ys = [float(row["global_mean_error_in_analytic_std"]) for row in method_rows]
        line, = axis.plot(xs, ys, marker="o", linewidth=1.8, label=LABELS[method])
        for row, x, y in zip(method_rows, xs, ys):
            axis.annotate(str(row["timesteps"]), (x, y), xytext=(4, 4),
                          textcoords="offset points", fontsize=8, color=line.get_color())
    axis.set(
        xlabel="sampling time (s)",
        ylabel="|global sample mean − analytic mean| / analytic std",
        title="Global-posterior accuracy versus sampling cost",
        xscale="log",
        yscale="log",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--timesteps", type=parse_timesteps, default=(25, 50, 100))
    parser.add_argument("--posterior-samples", type=int, default=8_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_result(args.reference)
    covariance_bank, _ = build_single_observation_covariance_bank(reference["x_observed"])
    model = load_score_checkpoint(args.checkpoint)
    rows: list[dict[str, object]] = []
    for timesteps in args.timesteps:
        grid_output = args.output_dir / f"timesteps_{timesteps}"
        grid_output.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            run_method(method, model, reference, grid_output, args.force,
                       covariance_bank, timesteps, args.posterior_samples)
            raw = load_result(grid_output / f"{method}.npz")
            rows.append({
                "method": method,
                "label": LABELS[method],
                "timesteps": timesteps,
                "runtime_seconds": float(raw["runtime_seconds"]),
                "global_mean_error_in_analytic_std": global_error(raw),
            })
    csv_path = args.output_dir / "accuracy_vs_time.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output_dir / "accuracy_vs_time.png")


if __name__ == "__main__":
    main()
