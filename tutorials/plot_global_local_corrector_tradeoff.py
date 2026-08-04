#!/usr/bin/env python3
"""Compare global/local DPM2 accuracy and cost across corrector counts.

The benchmark includes Langevin + F-NPSE, DPM2 + global/local Gaussian with no
correctors, and three DPM2 + global/local Gaussian corrector counts.  Each
method is evaluated on each requested integration grid and compared with the
analytic global posterior.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_shared_local_composition_methods import (
    VARIANTS,
    build_single_observation_covariance_bank,
    load_result,
    load_score_checkpoint,
    run_method,
)

ROOT = Path(__file__).resolve().parent
TUTORIALS_ROOT = ROOT if (ROOT / "output" / "compositional_inference").exists() else ROOT.parent
DEFAULT_REFERENCE = TUTORIALS_ROOT / "output" / "compositional_inference" / "06b_shared_local" / "raw_plot_data.npz"
DEFAULT_CHECKPOINT = TUTORIALS_ROOT / "output" / "compositional_inference" / "models" / "shared_local_mixture_full" / "Model_checkpoint.pt"


def parse_csv_ints(value: str, *, minimum: int, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    if not values or any(item < minimum for item in values):
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated integers >= {minimum}"
        )
    return values


def parse_timesteps(value: str) -> tuple[int, ...]:
    return parse_csv_ints(value, minimum=2, name="timesteps")


def parse_correctors(value: str) -> tuple[int, ...]:
    return parse_csv_ints(value, minimum=1, name="corrector steps")


def global_error(raw: dict[str, np.ndarray]) -> float:
    mean = float(raw["exact_joint_mean"][0])
    std = float(np.sqrt(raw["exact_joint_covariance"][0, 0]))
    return abs(float(raw["compass_global_samples"].mean()) - mean) / std


def add_corrector_variants(corrector_steps: tuple[int, ...]) -> dict[str, str]:
    labels = {
        "langevin_fnpe": "Langevin + F-NPSE",
        "dpm2_gauss_global_local_without_correctors": (
            "DPM2 + global/local Gaussian (0 correctors)"
        ),
    }
    for steps in corrector_steps:
        key = f"dpm2_gauss_global_local_correctors_{steps}"
        VARIANTS[key] = {
            "title": f"DPM2 + global/local Gaussian ({steps} correctors)",
            "filename": f"DPM2_gauss_global_local_{steps}_correctors.png",
            "sample_kwargs": {
                "method": "dpm", "order": 2,
                "correction": "Gauss_global_local",
                "corrector_steps_interval": 1,
                "corrector_steps": steps,
                "final_corrector_steps": 3,
                "snr": 0.2,
            },
            "description": (
                "Corrected marginal-global composition with "
                f"{steps} Langevin corrector steps per DPM2 interval."
            ),
        }
        labels[key] = f"DPM2 + global/local Gaussian ({steps} correctors)"
    return labels


def plot(rows: list[dict[str, object]], methods: list[str], labels: dict[str, str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    for method in methods:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["timesteps"]),
        )
        xs = [float(row["runtime_seconds"]) for row in method_rows]
        ys = [float(row["global_mean_error_in_analytic_std"]) for row in method_rows]
        line, = axis.plot(xs, ys, marker="o", linewidth=1.8, label=labels[method])
        for row, x, y in zip(method_rows, xs, ys):
            axis.annotate(str(row["timesteps"]), (x, y), xytext=(4, 4),
                          textcoords="offset points", fontsize=8, color=line.get_color())
    axis.set(
        xlabel="sampling time (s)",
        ylabel="|global sample mean − analytic mean| / analytic std",
        title="Global-posterior accuracy versus cost: corrector sweep",
        xscale="log", yscale="log",
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
    parser.add_argument("--corrector-steps", type=parse_correctors, default=(1, 2, 3, 5, 10))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    labels = add_corrector_variants(args.corrector_steps)
    methods = [
        "langevin_fnpe",
        "dpm2_gauss_global_local_without_correctors",
        *(f"dpm2_gauss_global_local_correctors_{steps}" for steps in args.corrector_steps),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_result(args.reference)
    covariance_bank, _ = build_single_observation_covariance_bank(reference["x_observed"])
    model = load_score_checkpoint(args.checkpoint)
    rows: list[dict[str, object]] = []
    for timesteps in args.timesteps:
        grid_output = args.output_dir / f"timesteps_{timesteps}"
        grid_output.mkdir(parents=True, exist_ok=True)
        for method in methods:
            run_method(method, model, reference, grid_output, args.force,
                       covariance_bank, timesteps, args.posterior_samples)
            raw = load_result(grid_output / f"{method}.npz")
            rows.append({
                "method": method,
                "label": labels[method],
                "timesteps": timesteps,
                "runtime_seconds": float(raw["runtime_seconds"]),
                "global_mean_error_in_analytic_std": global_error(raw),
            })
    with (args.output_dir / "accuracy_vs_time_corrector_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, methods, labels, args.output_dir / "accuracy_vs_time_corrector_sweep.png")


if __name__ == "__main__":
    main()
