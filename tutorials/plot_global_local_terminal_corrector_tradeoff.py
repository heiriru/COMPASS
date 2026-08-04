#!/usr/bin/env python3
"""Compare global/local DPM2 accuracy and cost across terminal-only corrector counts.

The benchmark includes Langevin + F-NPSE, DPM2 + global/local Gaussian with no
correctors, and three DPM2 + global/local Gaussian corrector counts.  Each
method is evaluated on each requested integration grid and compared with the
analytic global posterior.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_shared_local_composition_methods import (
    COVARIANCE_SAMPLE_COUNTS,
    INFERENCE_SEED,
    VARIANTS,
    build_single_observation_covariance_bank,
    empirical_covariances,
    load_result,
    load_score_checkpoint,
    run_method,
)
from plot_local_vs_global_validation import plot_shared_local

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


def add_terminal_corrector_variants(corrector_steps: tuple[int, ...]) -> dict[str, str]:
    labels = {"langevin_fnpe": "Langevin + F-NPSE"}
    for steps in corrector_steps:
        key = f"dpm2_gauss_global_local_terminal_correctors_{steps}"
        VARIANTS[key] = {
            "title": f"DPM2 + global/local Gaussian ({steps} terminal correctors)",
            "filename": f"DPM2_gauss_global_local_terminal_{steps}_correctors.png",
            "sample_kwargs": {
                "method": "dpm", "order": 2,
                "correction": "Gauss_global_local",
                "corrector_steps_interval": 1,
                "corrector_steps": 0,
                "final_corrector_steps": 0,
                "terminal_corrector_steps": steps,
                "snr": 0.2,
            },
            "description": (
                "Corrected marginal-global composition with no interleaved correctors and "
                f"{steps} terminal Langevin corrector steps."
            ),
        }
        labels[key] = f"DPM2 + global/local Gaussian ({steps} terminal correctors)"
    return labels


def run_terminal_sweep(model, reference: dict[str, np.ndarray], output: Path, force: bool,
                       covariance_bank: np.ndarray, timesteps: int, posterior_samples: int,
                       corrector_counts: tuple[int, ...]) -> list[dict[str, object]]:
    """Run one DPM2 predictor and branch its endpoint into terminal chains."""
    result_paths = {
        count: output / f"dpm2_gauss_global_local_terminal_correctors_{count}.npz"
        for count in corrector_counts
    }
    if not force and all(path.exists() for path in result_paths.values()):
        return [load_result(result_paths[count]) for count in corrector_counts]

    # _prepare_data constructs (global, local, observation) rows internally,
    # so it receives only the single observed coordinate per subject.
    x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)
    moment_count = max(COVARIANCE_SAMPLE_COUNTS)
    covariance = empirical_covariances(covariance_bank, moment_count)
    posterior_mean = np.asarray(covariance_bank[:, :moment_count], dtype=np.float64).mean(axis=1)
    torch.manual_seed(INFERENCE_SEED)
    torch.cuda.manual_seed_all(INFERENCE_SEED)
    started = time.perf_counter()
    samples_by_count = model.multi_obs_sampler.sample(
        world_size=1, data=x,
        condition_mask=torch.tensor([0., 0., 1.]),
        timesteps=timesteps, num_samples=posterior_samples, device="cuda", verbose=True,
        hierarchy=[0], prior=([0.0], [1.0]), method="dpm", order=2,
        correction="Gauss_global_local", posterior_covariance=torch.from_numpy(covariance),
        global_posterior_mean=torch.from_numpy(posterior_mean[:, :1]),
        global_posterior_covariance=torch.from_numpy(covariance[:, :1, :1]),
        corrector_steps_interval=1, corrector_steps=0, final_corrector_steps=0,
        terminal_corrector_counts=corrector_counts, snr=0.2,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    sampler = model.multi_obs_sampler
    terminal_seconds = sampler.terminal_corrector_seconds
    shared_predictor_seconds = elapsed - terminal_seconds[max(corrector_counts)]
    diagnostics = sampler.covariance_diagnostics
    covariance_condition_number = max(map(np.linalg.cond, covariance))
    raws = []
    for count in corrector_counts:
        samples = samples_by_count[count].detach().cpu().numpy()
        synchronization_error = float(np.max(np.abs(samples[:, :, 0] - samples[0:1, :, 0])))
        if synchronization_error > 1e-6:
            raise AssertionError(f"terminal_correctors_{count}: global samples are not synchronized: {synchronization_error}")
        total_seconds = shared_predictor_seconds + terminal_seconds[count]
        path = result_paths[count]
        np.savez_compressed(
            path, x_observed=reference["x_observed"], global_truth=reference["global_truth"],
            local_truth=reference["local_truth"], exact_joint_mean=reference["exact_joint_mean"],
            exact_joint_covariance=reference["exact_joint_covariance"],
            compass_global_samples=samples[0, :, 0], compass_local_samples=samples[:, :, 1],
            shared_synchronization_max_abs=synchronization_error,
            covariance_condition_number=covariance_condition_number,
            pd_repair_fraction=diagnostics["repair_fraction"],
            pd_maximum_relative_repair=diagnostics["maximum_relative_repair"],
            pd_minimum_eigenvalue_before=diagnostics["minimum_eigenvalue_before"] if diagnostics["minimum_eigenvalue_before"] is not None else np.nan,
            pd_minimum_eigenvalue_after=diagnostics["minimum_eigenvalue_after"] if diagnostics["minimum_eigenvalue_after"] is not None else np.nan,
            predictor_runtime_seconds=shared_predictor_seconds,
            terminal_corrector_runtime_seconds=terminal_seconds[count],
            runtime_seconds=total_seconds,
        )
        plot_shared_local(path, output / VARIANTS[f"dpm2_gauss_global_local_terminal_correctors_{count}"]["filename"],
                          title=VARIANTS[f"dpm2_gauss_global_local_terminal_correctors_{count}"]["title"])
        raws.append(load_result(path))
    return raws

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
        title="Global-posterior accuracy versus cost: terminal-corrector sweep",
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

    labels = add_terminal_corrector_variants(args.corrector_steps)
    methods = [
        "langevin_fnpe",
        *(f"dpm2_gauss_global_local_terminal_correctors_{steps}" for steps in args.corrector_steps),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_result(args.reference)
    covariance_bank, _ = build_single_observation_covariance_bank(reference["x_observed"])
    model = load_score_checkpoint(args.checkpoint)
    rows: list[dict[str, object]] = []
    for timesteps in args.timesteps:
        grid_output = args.output_dir / f"timesteps_{timesteps}"
        grid_output.mkdir(parents=True, exist_ok=True)
        run_method("langevin_fnpe", model, reference, grid_output, args.force,
                   covariance_bank, timesteps, args.posterior_samples)
        langevin_raw = load_result(grid_output / "langevin_fnpe.npz")
        rows.append({
            "method": "langevin_fnpe", "label": labels["langevin_fnpe"],
            "timesteps": timesteps, "runtime_seconds": float(langevin_raw["runtime_seconds"]),
            "global_mean_error_in_analytic_std": global_error(langevin_raw),
        })
        terminal_raws = run_terminal_sweep(
            model, reference, grid_output, args.force, covariance_bank, timesteps,
            args.posterior_samples, args.corrector_steps,
        )
        for count, raw in zip(args.corrector_steps, terminal_raws):
            method = f"dpm2_gauss_global_local_terminal_correctors_{count}"
            rows.append({
                "method": method, "label": labels[method], "timesteps": timesteps,
                "runtime_seconds": float(raw["runtime_seconds"]),
                "global_mean_error_in_analytic_std": global_error(raw),
            })
    with (args.output_dir / "accuracy_vs_time_terminal_corrector_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, methods, labels, args.output_dir / "accuracy_vs_time_terminal_corrector_sweep.png")


if __name__ == "__main__":
    main()
