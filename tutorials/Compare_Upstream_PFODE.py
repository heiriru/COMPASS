#!/usr/bin/env python3
"""Compare current and bGuenes-upstream exact PF-ODE integration."""

from __future__ import annotations

import os

CPU_USAGE_LIMIT_FRACTION = 0.06
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def configure_cpu_usage_limit(fraction=CPU_USAGE_LIMIT_FRACTION):
    logical = os.cpu_count() or 1
    limit = int(logical * fraction)
    if limit < 1:
        raise RuntimeError(f"Cannot enforce a {fraction:.1%} CPU limit on {logical} CPUs.")
    selected = tuple(sorted(os.sched_getaffinity(0))[:limit])
    if not selected:
        raise RuntimeError("The process has no CPUs available.")
    os.sched_setaffinity(0, selected)
    for name in CPU_THREAD_ENV_VARS:
        os.environ[name] = str(len(selected))
    return logical, selected


CPU_LIMIT_INFO = configure_cpu_usage_limit() if __name__ == "__main__" else None

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from autocvd import autocvd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


TUTORIAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TUTORIAL_DIR.parent
OUTPUT_ROOT = TUTORIAL_DIR / "output" / "divergence_head" / "upstream_pfode_comparison"
DEFAULT_UPSTREAM_ROOT = Path("/tmp/compass-bguenes-pfode-926f580")
EPS = 1e-3
TIMESTEPS = (10, 50, 100, 200, 500)
DENSITY_TIMESTEPS = 100
SAMPLE_COUNT = 64
COLORS = {
    "analytic": "#222222",
    "current": "#4C78A8",
    "upstream": "#F58518",
    "difference": "#E45756",
    "centered": "#B279A2",
}


def log(message):
    print(message, flush=True)


def validate_cpu_cap():
    logical, expected = CPU_LIMIT_INFO
    active = tuple(sorted(os.sched_getaffinity(0)))
    if active != expected or len(active) / logical > CPU_USAGE_LIMIT_FRACTION:
        raise RuntimeError("CPU affinity no longer satisfies the 6% hard cap.")
    for variable in CPU_THREAD_ENV_VARS:
        if os.environ.get(variable) != str(len(active)):
            raise RuntimeError(f"{variable} no longer matches the CPU cap.")


def finite(name, values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains non-finite values.")


def metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite("metric prediction", prediction)
    finite("metric target", target)
    residual = prediction - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    centered = residual - residual.mean()
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if prediction.size > 1 and prediction.std() > 0 and target.std() > 0
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "centered_rmse": float(np.sqrt(np.mean(centered**2))),
        "bias": float(np.mean(residual)),
        "correlation": correlation,
    }


def checkpoint_for(problem):
    roots = {
        "gaussian": TUTORIAL_DIR / "data" / "divergence_head" /
                    "gaussian_identity_prior_noise_0p2",
        "line": TUTORIAL_DIR / "data" / "divergence_head" /
                "line_parabola" / "line",
        "parabola": TUTORIAL_DIR / "data" / "divergence_head" /
                    "line_parabola" / "parabola",
        "banana": TUTORIAL_DIR / "data" / "divergence_head" /
                  "banana" / "banana",
    }
    root = roots[problem]
    preferred = root / "stage1_exact_no_head.pt"
    return preferred if preferred.exists() else root / "stage1_score.pt"


def smoothing_sigma(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu")
    if payload["sde_type"] != "vesde":
        raise RuntimeError("This comparison currently expects the VESDE checkpoints.")
    sigma = float(payload["sigma"])
    return math.sqrt((sigma ** (2.0 * EPS) - 1.0) / (2.0 * math.log(sigma)))


def prepare_cases():
    import Divergence_Head as dh
    import Divergence_Head_Problems as problems

    cases = {}
    gaussian_reference = dh.LinearGaussianReference(**dh.GAUSSIAN_PROBLEM_CONFIG)
    gaussian_splits = dh.load_data_splits(
        TUTORIAL_DIR / "data" / "divergence_head" /
        "gaussian_identity_prior_noise_0p2" / "linear_gaussian_splits.npz"
    )
    cases["gaussian"] = {
        "reference": gaussian_reference,
        "splits": gaussian_splits,
        "masks": dh.MASKS,
        "grid_size": dh.DEFAULT_CONFIG["density_grid_size"],
        "grid_builder": "gaussian",
        "shift": lambda mask: 0.0,
    }

    for family in ("line", "parabola"):
        config = dict(problems.BASE_CONFIG)
        reference, splits = problems.load_curve_splits(family, config)
        cases[family] = {
            "reference": reference,
            "splits": splits,
            "masks": reference.masks,
            "grid_size": config["density_grid_size"],
            "grid_builder": "problem",
            "shift": reference.raw_log_density_shift,
        }

    banana_cache = (
        TUTORIAL_DIR / "data" / "divergence_head" / "banana" / "banana"
    )
    banana_config = dict(problems.BASE_CONFIG)
    banana_reference, banana_splits = problems.banana_splits(
        banana_config, banana_cache
    )
    cases["banana"] = {
        "reference": banana_reference,
        "splits": banana_splits,
        "masks": banana_reference.masks,
        "grid_size": banana_config["density_grid_size"],
        "grid_builder": "problem",
        "shift": banana_reference.raw_log_density_shift,
    }

    for problem, case in cases.items():
        checkpoint = checkpoint_for(problem)
        sigma_eps = smoothing_sigma(checkpoint)
        reference = case["reference"]
        splits = case["splits"]
        sample_rows = np.concatenate(splits["test"], axis=1).astype(np.float32)
        case["sample_rows"] = sample_rows[:SAMPLE_COUNT]
        case["checkpoint"] = checkpoint
        case["sigma_eps"] = sigma_eps
        case["grids"] = {}

        for mask_name, mask in case["masks"].items():
            mask = np.asarray(mask, dtype=np.float32)
            if case["grid_builder"] == "gaussian":
                fixed = sample_rows[0]
                joint, grid0, grid1, latent = dh.make_density_grid(
                    reference, fixed, mask, case["grid_size"], sigma_eps
                )
                grid = {
                    "dimension": 2,
                    "joint": joint,
                    "axis0": grid0,
                    "axis1": grid1,
                    "shape": grid0.shape,
                    "latent": latent,
                }
            else:
                grid = problems.density_grid(
                    reference, mask, case["grid_size"]
                )
            shift = float(case["shift"](mask))
            analytic_grid = (
                reference.conditional_log_prob(grid["joint"], mask, sigma_eps) + shift
            ).reshape(grid["shape"])
            analytic_samples = (
                reference.conditional_log_prob(case["sample_rows"], mask, sigma_eps)
                + shift
            )
            case["grids"][mask_name] = {
                **grid,
                "mask": mask,
                "shift": shift,
                "analytic_grid": analytic_grid,
                "analytic_samples": analytic_samples,
            }
    return cases


def write_worker_input(problem, case, path):
    arrays = {
        "sample_rows": case["sample_rows"],
        "timesteps": np.asarray(TIMESTEPS, dtype=np.int64),
    }
    for mask_name, grid in case["grids"].items():
        arrays[f"mask_{mask_name}"] = grid["mask"]
        arrays[f"grid_{mask_name}"] = grid["joint"]
    np.savez_compressed(path, **arrays)


def load_model(checkpoint_path, backend, device):
    from compass import ScoreBasedInferenceModel as SBIm

    if backend == "current":
        return SBIm.load(str(checkpoint_path), device=device)

    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["model_state_dict"] = {
        key: value
        for key, value in payload["model_state_dict"].items()
        if not key.startswith("divergence_head.")
    }
    payload.pop("divergence_head_trained", None)
    temporary = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    temporary.close()
    temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        return SBIm.load(str(temporary_path), device=device)
    finally:
        temporary_path.unlink(missing_ok=True)


def worker_main(args):
    validate_cpu_cap()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = np.load(args.input)
    model = load_model(Path(args.checkpoint), args.backend, device)
    results = {}
    mask_names = sorted(
        key.removeprefix("mask_") for key in payload.files if key.startswith("mask_")
    )
    for mask_name in mask_names:
        mask = torch.from_numpy(payload[f"mask_{mask_name}"]).float()
        grid = torch.from_numpy(payload[f"grid_{mask_name}"]).float()
        log(f"[{args.backend}] {args.problem} {mask_name}: density grid")
        results[f"grid_{mask_name}"] = model.log_prob(
            data=grid,
            condition_mask=mask,
            timesteps=DENSITY_TIMESTEPS,
            eps=EPS,
            divergence="exact",
            device=device,
            batch_size=512,
            verbose=False,
        ).numpy()
        samples = torch.from_numpy(payload["sample_rows"]).float()
        for step in payload["timesteps"]:
            count = int(step)
            log(f"[{args.backend}] {args.problem} {mask_name}: {count} steps")
            results[f"sample_{mask_name}_{count}"] = model.log_prob(
                data=samples,
                condition_mask=mask,
                timesteps=count,
                eps=EPS,
                divergence="exact",
                device=device,
                batch_size=SAMPLE_COUNT,
                verbose=False,
            ).numpy()
    np.savez_compressed(args.output, **results)


def run_worker(problem, backend, checkpoint, input_path, output_path, implementation_root):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(implementation_root) / "src")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend", backend,
        "--problem", problem,
        "--checkpoint", str(checkpoint),
        "--input", str(input_path),
        "--output", str(output_path),
    ]
    subprocess.run(command, check=True, env=environment)


def density_plot(problem, mask_name, grid, current, upstream, output_path):
    analytic = grid["analytic_grid"]
    shape = grid["shape"]
    current = (np.asarray(current) + grid["shift"]).reshape(shape)
    upstream = (np.asarray(upstream) + grid["shift"]).reshape(shape)
    finite(f"{problem} {mask_name} current grid", current)
    finite(f"{problem} {mask_name} upstream grid", upstream)

    if grid["dimension"] == 1:
        x = grid["axis0"]
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(x, np.exp(np.clip(analytic, -80, 50)), "k--", label="Analytic")
        axes[0].plot(
            x, np.exp(np.clip(current, -80, 50)),
            color=COLORS["current"], label="Current exact",
        )
        axes[0].plot(
            x, np.exp(np.clip(upstream, -80, 50)),
            color=COLORS["upstream"], label="bGuenes upstream exact",
        )
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[1].plot(
            x, current - analytic, color=COLORS["current"],
            label="Current - analytic",
        )
        axes[1].plot(
            x, upstream - analytic, color=COLORS["upstream"],
            label="Upstream - analytic",
        )
        axes[1].plot(
            x, current - upstream, color=COLORS["difference"],
            label="Current - upstream",
        )
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set(xlabel=f"Latent dimension {int(grid['latent'][0])}",
                    ylabel="Log-density difference")
        axes[1].legend()
    else:
        grid0, grid1 = grid["axis0"], grid["axis1"]
        densities = [
            np.exp(np.clip(values, -80, 50))
            for values in (analytic, current, upstream)
        ]
        errors = [current - analytic, upstream - analytic, current - upstream]
        density_upper = max(max(float(values.max()) for values in densities), 1e-12)
        error_limit = max(max(float(np.abs(values).max()) for values in errors), 1e-8)
        density_levels = np.linspace(0, density_upper, 19)
        error_levels = np.linspace(-error_limit, error_limit, 19)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
        titles = (
            "Analytic smoothed density",
            "Current exact density",
            "bGuenes upstream exact density",
            "Current - analytic log density",
            "Upstream - analytic log density",
            "Current - upstream log density",
        )
        for index, (axis, values, title) in enumerate(
            zip(axes.flat, densities + errors, titles)
        ):
            levels = density_levels if index < 3 else error_levels
            cmap = "viridis" if index < 3 else "coolwarm"
            filled = axis.contourf(grid0, grid1, values, levels=levels, cmap=cmap)
            fig.colorbar(filled, ax=axis, shrink=0.82)
            axis.set_title(title)
        for axis in axes[-1]:
            axis.set_xlabel(f"Latent dimension {int(grid['latent'][0])}")
        for axis in axes[:, 0]:
            axis.set_ylabel(f"Latent dimension {int(grid['latent'][1])}")
    fig.suptitle(f"{problem.capitalize()} {mask_name}: exact PF-ODE comparison", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return current, upstream


def timestep_plot(problem, case, current_data, upstream_data, output_path):
    mask_names = list(case["masks"])
    fig, axes = plt.subplots(1, len(mask_names), figsize=(12, 4.8), squeeze=False)
    metric_rows = []
    for column, mask_name in enumerate(mask_names):
        axis = axes[0, column]
        analytic = case["grids"][mask_name]["analytic_samples"]
        shift = case["grids"][mask_name]["shift"]
        current_mae = []
        upstream_mae = []
        disagreement = []
        centered_disagreement = []
        for count in TIMESTEPS:
            current = current_data[f"sample_{mask_name}_{count}"] + shift
            upstream = upstream_data[f"sample_{mask_name}_{count}"] + shift
            current_metrics = metrics(current, analytic)
            upstream_metrics = metrics(upstream, analytic)
            comparison = metrics(current, upstream)
            current_mae.append(current_metrics["mae"])
            upstream_mae.append(upstream_metrics["mae"])
            disagreement.append(comparison["rmse"])
            centered_disagreement.append(comparison["centered_rmse"])
            for method, values in (
                ("current_vs_analytic", current_metrics),
                ("upstream_vs_analytic", upstream_metrics),
                ("current_vs_upstream", comparison),
            ):
                metric_rows.append({
                    "problem": problem,
                    "mask": mask_name,
                    "timesteps": count,
                    "comparison": method,
                    **values,
                })
        axis.plot(TIMESTEPS, current_mae, marker="o", color=COLORS["current"],
                  label="Current MAE vs analytic")
        axis.plot(TIMESTEPS, upstream_mae, marker="o", color=COLORS["upstream"],
                  label="Upstream MAE vs analytic")
        axis.plot(TIMESTEPS, disagreement, marker="o", color=COLORS["difference"],
                  label="Current-upstream RMSE")
        axis.plot(TIMESTEPS, centered_disagreement, marker="o",
                  color=COLORS["centered"], linestyle="--",
                  label="Centered current-upstream RMSE")
        axis.set(
            title=mask_name,
            xlabel="Integration timesteps",
            ylabel="Log-density error",
            xscale="log",
            yscale="log",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(f"{problem.capitalize()}: exact PF-ODE timestep comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return metric_rows


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_plot(rows, output_path):
    selected = [
        row for row in rows
        if row["timesteps"] == DENSITY_TIMESTEPS
        and row["comparison"] in (
            "current_vs_analytic", "upstream_vs_analytic", "current_vs_upstream"
        )
    ]
    labels = sorted({f"{row['problem']}\n{row['mask']}" for row in selected})
    comparisons = (
        ("current_vs_analytic", "Current vs analytic", COLORS["current"]),
        ("upstream_vs_analytic", "Upstream vs analytic", COLORS["upstream"]),
        ("current_vs_upstream", "Current vs upstream", COLORS["difference"]),
    )
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(13, 5))
    for offset, (comparison, title, color) in enumerate(comparisons):
        lookup = {
            f"{row['problem']}\n{row['mask']}": max(float(row["rmse"]), 1e-12)
            for row in selected if row["comparison"] == comparison
        }
        values = [lookup[label] for label in labels]
        axis.bar(x + (offset - 1) * width, values, width, label=title, color=color)
    axis.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Log-density RMSE",
        yscale="log",
        title=f"Exact PF-ODE comparison at {DENSITY_TIMESTEPS} integration timesteps",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(args):
    validate_cpu_cap()
    selected_gpu = autocvd(num_gpus=1, interval=1)
    log(f"[setup] autocvd selected {selected_gpu}")
    upstream_root = Path(args.upstream_root).resolve()
    if not (upstream_root / "src" / "compass" / "PFODE.py").exists():
        raise FileNotFoundError(f"Missing upstream worktree at {upstream_root}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = prepare_cases()
    all_rows = []
    with tempfile.TemporaryDirectory(prefix="compass-pfode-comparison-", dir="/tmp") as temp:
        temp_root = Path(temp)
        for problem, case in cases.items():
            log(f"[compare] {problem}")
            problem_output = OUTPUT_ROOT / problem
            problem_output.mkdir(parents=True, exist_ok=True)
            input_path = temp_root / f"{problem}_input.npz"
            current_path = temp_root / f"{problem}_current.npz"
            upstream_path = temp_root / f"{problem}_upstream.npz"
            write_worker_input(problem, case, input_path)
            run_worker(
                problem, "current", case["checkpoint"], input_path, current_path,
                REPO_ROOT,
            )
            run_worker(
                problem, "upstream", case["checkpoint"], input_path, upstream_path,
                upstream_root,
            )
            current_data = np.load(current_path)
            upstream_data = np.load(upstream_path)
            for mask_name, grid in case["grids"].items():
                current, upstream = density_plot(
                    problem,
                    mask_name,
                    grid,
                    current_data[f"grid_{mask_name}"],
                    upstream_data[f"grid_{mask_name}"],
                    problem_output / f"{mask_name}_density_comparison.png",
                )
                analytic = grid["analytic_grid"]
                for comparison, values in (
                    ("current_grid_vs_analytic", metrics(current, analytic)),
                    ("upstream_grid_vs_analytic", metrics(upstream, analytic)),
                    ("current_grid_vs_upstream", metrics(current, upstream)),
                ):
                    all_rows.append({
                        "problem": problem,
                        "mask": mask_name,
                        "timesteps": DENSITY_TIMESTEPS,
                        "comparison": comparison,
                        **values,
                    })
            all_rows.extend(timestep_plot(
                problem, case, current_data, upstream_data,
                problem_output / "timestep_comparison.png",
            ))

    write_csv(OUTPUT_ROOT / "metrics.csv", all_rows)
    summary_plot(all_rows, OUTPUT_ROOT / "summary.png")
    manifest = {
        "current_repository": str(REPO_ROOT),
        "current_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "upstream_repository": "https://github.com/bGuenes/COMPASS.git",
        "upstream_worktree": str(upstream_root),
        "upstream_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream_root, text=True
        ).strip(),
        "checkpoint_role": "untouched score-only Stage-1",
        "eps": EPS,
        "density_timesteps": DENSITY_TIMESTEPS,
        "timestep_sweep": TIMESTEPS,
        "sample_count": SAMPLE_COUNT,
        "output": str(OUTPUT_ROOT),
    }
    with (OUTPUT_ROOT / "comparison_config.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    log(f"[done] comparison outputs: {OUTPUT_ROOT}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("current", "upstream"))
    parser.add_argument("--problem")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument(
        "--upstream-root", default=str(DEFAULT_UPSTREAM_ROOT),
        help="Detached bGuenes upstream worktree.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        worker_main(parsed)
    else:
        main(parsed)
