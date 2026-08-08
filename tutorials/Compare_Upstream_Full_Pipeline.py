#!/usr/bin/env python3
"""Train/evaluate the historical bGuenes pipeline against current exact PF-ODE."""

from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def configure_cpu_usage_limit(max_threads=CPU_THREAD_LIMIT):
    logical = os.cpu_count() or 1
    limit = min(int(max_threads), logical)
    if limit < 1:
        raise RuntimeError(f"Cannot enforce a {max_threads}-thread CPU limit on {logical} CPUs.")
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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from autocvd import autocvd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import Compare_Upstream_PFODE as common


TUTORIAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TUTORIAL_DIR.parent
DEFAULT_UPSTREAM_ROOT = common.DEFAULT_UPSTREAM_ROOT
OUTPUT_ROOT = (
    TUTORIAL_DIR / "output" / "divergence_head" /
    "upstream_full_pipeline_comparison"
)
CHECKPOINT_ROOT = (
    TUTORIAL_DIR / "data" / "divergence_head" / "upstream_full_pipeline"
)
BASELINE_CHECKPOINT_ROOT = (
    TUTORIAL_DIR / "data" / "divergence_head" /
    "upstream_full_pipeline_before_loss_fix"
)
TRAINING_SEED = 1729
COLORS = {
    "analytic": "#222222",
    "upstream": "#F58518",
    "stage1": "#4C78A8",
    "stage2": "#54A24B",
    "stage1_difference": "#E45756",
    "stage2_difference": "#B279A2",
}


def log(message):
    print(message, flush=True)


def validate_cpu_cap():
    logical, expected = CPU_LIMIT_INFO
    active = tuple(sorted(os.sched_getaffinity(0)))
    if active != expected or len(active) > CPU_THREAD_LIMIT:
        raise RuntimeError("CPU affinity no longer satisfies the 3-thread hard cap.")


def stage2_checkpoint(problem):
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
    path = roots[problem] / "stage2_joint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing current Stage-2 checkpoint: {path}")
    return path


def historical_training_config(problem, case):
    import Divergence_Head as dh
    import Divergence_Head_Problems as problems

    source = dh.DEFAULT_CONFIG if problem == "gaussian" else problems.BASE_CONFIG
    return {
        "seed": int(source["seed"]),
        "nodes_size": int(
            case["splits"]["train"][0].shape[1] +
            case["splits"]["train"][1].shape[1]
        ),
        "sde_type": source["sde_type"],
        "sigma": float(source["sigma"]),
        "hidden_size": int(source["hidden_size"]),
        "depth": int(source["depth"]),
        "num_heads": int(source["num_heads"]),
        "mlp_ratio": int(source["mlp_ratio"]),
        "batch_size": int(source["batch_size"]),
        "max_epochs": int(source["stage1_max_epochs"]),
        "learning_rate": float(source["learning_rate"]),
        "early_stopping_patience": int(source["early_stopping_patience"]),
        "time_sampling": source["time_sampling"],
    }


def write_training_input(case, path):
    arrays = {}
    for split in ("train", "validation"):
        theta, x = case["splits"][split]
        arrays[f"{split}_theta"] = np.asarray(theta, dtype=np.float32)
        arrays[f"{split}_x"] = np.asarray(x, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def validate_historical_checkpoint(path, config):
    payload = torch.load(path, map_location="cpu")
    expected = {
        key: config[key]
        for key in (
            "nodes_size", "sde_type", "sigma", "hidden_size", "depth",
            "num_heads", "mlp_ratio",
        )
    }
    actual = {key: payload[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"Historical checkpoint mismatch at {path}: {actual} != {expected}")
    if any(key.startswith("divergence_head.") for key in payload["model_state_dict"]):
        raise RuntimeError(f"Historical checkpoint unexpectedly contains a divergence head: {path}")


def training_worker(args):
    validate_cpu_cap()
    from compass import ScoreBasedInferenceModel as SBIm

    with Path(args.training_config).open() as handle:
        config = json.load(handle)
    final_path = Path(args.checkpoint)
    checkpoint_path = final_path.with_name(f"{final_path.stem}_checkpoint.pt")
    history_path = final_path.with_name("training_history.json")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() and not args.force_retrain:
        validate_historical_checkpoint(final_path, config)
        log(f"[upstream cache] {args.problem}: {final_path}")
        return
    if args.force_retrain:
        final_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)

    data = np.load(args.training_input)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SBIm(
        nodes_size=config["nodes_size"],
        sde_type=config["sde_type"],
        sigma=config["sigma"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        device=device,
    )
    log(
        f"[upstream train] {args.problem}: up to {config['max_epochs']} epochs "
        f"on {device} with the historical score-only model"
    )
    training_start = perf_counter()
    model.train(
        theta=torch.from_numpy(data["train_theta"]),
        x=torch.from_numpy(data["train_x"]),
        theta_val=torch.from_numpy(data["validation_theta"]),
        x_val=torch.from_numpy(data["validation_x"]),
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        lr=config["learning_rate"],
        device=device,
        verbose=False,
        path=str(final_path.parent),
        name=final_path.stem,
        early_stopping_patience=config["early_stopping_patience"],
        time_sampling=config["time_sampling"],
    )
    training_seconds = perf_counter() - training_start
    if not checkpoint_path.exists():
        raise RuntimeError(f"Historical training did not produce {checkpoint_path}")
    shutil.copy2(checkpoint_path, final_path)
    validate_historical_checkpoint(final_path, config)
    history = {
        "train_loss": [float(value) for value in model.trainer.train_loss],
        "validation_loss": [float(value) for value in model.trainer.val_loss],
        "training_seconds": float(training_seconds),
        "epochs_completed": len(model.trainer.train_loss),
        "loss_reduction": "per_example_before_sigma_weight",
    }
    with history_path.open("w") as handle:
        json.dump(history, handle, indent=2)
    log(f"[upstream train done] {args.problem}: {final_path}")


def run_training_worker(problem, input_path, config_path, checkpoint, upstream_root,
                        force_retrain=False):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(upstream_root / "src")
    command = [
        sys.executable, str(Path(__file__).resolve()), "--train-worker",
        "--problem", problem,
        "--training-input", str(input_path),
        "--training-config", str(config_path),
        "--checkpoint", str(checkpoint),
    ]
    if force_retrain:
        command.append("--force-retrain")
    subprocess.run(command, check=True, env=environment)


def checkpoint_comparison(reference_path, candidate_path):
    reference = torch.load(reference_path, map_location="cpu")["model_state_dict"]
    candidate = torch.load(candidate_path, map_location="cpu")["model_state_dict"]
    reference = {
        key: value for key, value in reference.items()
        if not key.startswith("divergence_head.")
    }
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        unexpected = sorted(set(candidate) - set(reference))
        raise RuntimeError(
            f"Historical score architecture mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    squared_sum = 0.0
    count = 0
    max_abs = 0.0
    for key in sorted(reference):
        difference = reference[key].double() - candidate[key].double()
        squared_sum += float(torch.sum(difference * difference))
        count += difference.numel()
        max_abs = max(max_abs, float(torch.max(torch.abs(difference))))
    return {"parameter_rmse": float(np.sqrt(squared_sum / count)), "parameter_max_abs": max_abs}


def density_plot(problem, mask_name, grid, upstream, stage1, stage2, output_path):
    analytic = np.asarray(grid["analytic_grid"])
    shape = grid["shape"]
    shift = grid["shift"]
    values = {
        "upstream": (np.asarray(upstream) + shift).reshape(shape),
        "stage1": (np.asarray(stage1) + shift).reshape(shape),
        "stage2": (np.asarray(stage2) + shift).reshape(shape),
    }
    for name, value in values.items():
        common.finite(f"{problem} {mask_name} {name}", value)

    if grid["dimension"] == 1:
        x = grid["axis0"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(x, np.exp(np.clip(analytic, -80, 50)), "k--", label="Analytic")
        for name, label in (
            ("upstream", "Upstream loss-fixed model + upstream PF-ODE"),
            ("stage1", "Current Stage-1 model + current exact PF-ODE"),
            ("stage2", "Current Stage-2 model + current exact PF-ODE"),
        ):
            axes[0].plot(x, np.exp(np.clip(values[name], -80, 50)),
                         color=COLORS[name], label=label)
            axes[1].plot(x, values[name] - analytic, color=COLORS[name],
                         label=f"{label} - analytic")
        axes[0].set_ylabel("Density")
        axes[0].legend(fontsize=8)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set(
            xlabel=f"Latent dimension {int(grid['latent'][0])}",
            ylabel="Log-density difference",
        )
        axes[1].legend(fontsize=7)
    else:
        grid0, grid1 = grid["axis0"], grid["axis1"]
        density_values = [
            np.exp(np.clip(item, -80, 50))
            for item in (analytic, values["upstream"], values["stage1"], values["stage2"])
        ]
        error_values = [
            values["upstream"] - analytic,
            values["stage1"] - analytic,
            values["stage2"] - analytic,
            values["stage2"] - values["upstream"],
        ]
        density_upper = max(max(float(item.max()) for item in density_values), 1e-12)
        error_limit = max(max(float(np.abs(item).max()) for item in error_values[:3]), 1e-8)
        density_levels = np.linspace(0, density_upper, 19)
        error_levels = np.linspace(-error_limit, error_limit, 19)
        fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
        titles = (
            "Analytic smoothed density",
            "Upstream loss-fixed model\nupstream exact PF-ODE",
            "Current Stage-1 model\ncurrent exact PF-ODE",
            "Current Stage-2 model\ncurrent exact PF-ODE",
            "Upstream - analytic\nlog density",
            "Current Stage-1 - analytic\nlog density",
            "Current Stage-2 - analytic\nlog density",
            "Current Stage-2 - upstream\nlog density",
        )
        for index, (axis, item, title) in enumerate(
            zip(axes.flat, density_values + error_values, titles)
        ):
            levels = density_levels if index < 4 else error_levels
            cmap = "viridis" if index < 4 else "coolwarm"
            filled = axis.contourf(grid0, grid1, item, levels=levels, cmap=cmap, extend="both")
            fig.colorbar(filled, ax=axis, shrink=0.8)
            axis.set_title(title)
        for axis in axes[-1]:
            axis.set_xlabel(f"Latent dimension {int(grid['latent'][0])}")
        for axis in axes[:, 0]:
            axis.set_ylabel(f"Latent dimension {int(grid['latent'][1])}")
    fig.suptitle(f"{problem.capitalize()} {mask_name}: full historical-pipeline comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return values


def timestep_plot(problem, case, upstream_data, stage1_data, stage2_data, output_path):
    fig, axes = plt.subplots(1, len(case["masks"]), figsize=(13, 5), squeeze=False)
    rows = []
    for column, mask_name in enumerate(case["masks"]):
        axis = axes[0, column]
        analytic = case["grids"][mask_name]["analytic_samples"]
        shift = case["grids"][mask_name]["shift"]
        series = {key: [] for key in (
            "upstream", "stage1", "stage2", "stage1_upstream", "stage2_upstream"
        )}
        for count in common.TIMESTEPS:
            values = {
                "upstream": upstream_data[f"sample_{mask_name}_{count}"] + shift,
                "stage1": stage1_data[f"sample_{mask_name}_{count}"] + shift,
                "stage2": stage2_data[f"sample_{mask_name}_{count}"] + shift,
            }
            comparisons = {
                "upstream_vs_analytic": common.metrics(values["upstream"], analytic),
                "current_stage1_vs_analytic": common.metrics(values["stage1"], analytic),
                "current_stage2_vs_analytic": common.metrics(values["stage2"], analytic),
                "current_stage1_vs_upstream": common.metrics(values["stage1"], values["upstream"]),
                "current_stage2_vs_upstream": common.metrics(values["stage2"], values["upstream"]),
            }
            series["upstream"].append(comparisons["upstream_vs_analytic"]["mae"])
            series["stage1"].append(comparisons["current_stage1_vs_analytic"]["mae"])
            series["stage2"].append(comparisons["current_stage2_vs_analytic"]["mae"])
            series["stage1_upstream"].append(comparisons["current_stage1_vs_upstream"]["rmse"])
            series["stage2_upstream"].append(comparisons["current_stage2_vs_upstream"]["rmse"])
            for comparison, metric in comparisons.items():
                rows.append({
                    "problem": problem, "mask": mask_name, "timesteps": count,
                    "comparison": comparison, **metric,
                })
        axis.plot(common.TIMESTEPS, series["upstream"], "o-", color=COLORS["upstream"],
                  label="Upstream loss-fixed MAE vs analytic")
        axis.plot(common.TIMESTEPS, series["stage1"], "o-", color=COLORS["stage1"],
                  label="Current Stage-1 MAE vs analytic")
        axis.plot(common.TIMESTEPS, series["stage2"], "o-", color=COLORS["stage2"],
                  label="Current Stage-2 exact MAE vs analytic")
        axis.plot(common.TIMESTEPS, series["stage1_upstream"], "o--",
                  color=COLORS["stage1_difference"], label="Stage-1/upstream RMSE")
        axis.plot(common.TIMESTEPS, series["stage2_upstream"], "o--",
                  color=COLORS["stage2_difference"], label="Stage-2/upstream RMSE")
        axis.set(title=mask_name, xlabel="Integration timesteps",
                 ylabel="Log-density error", xscale="log", yscale="log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(f"{problem.capitalize()}: historical training + exact PF-ODE comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def summary_plot(rows, output_path):
    comparisons = (
        ("upstream_vs_analytic", "Upstream loss-fixed vs analytic", COLORS["upstream"]),
        ("current_stage1_vs_analytic", "Current Stage-1 vs analytic", COLORS["stage1"]),
        ("current_stage2_vs_analytic", "Current Stage-2 exact vs analytic", COLORS["stage2"]),
        ("current_stage1_vs_upstream", "Stage-1 vs upstream loss-fixed", COLORS["stage1_difference"]),
        ("current_stage2_vs_upstream", "Stage-2 vs upstream loss-fixed", COLORS["stage2_difference"]),
    )
    selected = [row for row in rows if row["timesteps"] == common.DENSITY_TIMESTEPS]
    labels = sorted({f"{row['problem']}\n{row['mask']}" for row in selected})
    x = np.arange(len(labels))
    width = 0.16
    fig, axis = plt.subplots(figsize=(15, 5.5))
    for offset, (comparison, title, color) in enumerate(comparisons):
        lookup = {
            f"{row['problem']}\n{row['mask']}": max(float(row["rmse"]), 1e-12)
            for row in selected if row["comparison"] == comparison
        }
        axis.bar(x + (offset - 2) * width, [lookup[label] for label in labels],
                 width, label=title, color=color)
    axis.set(xticks=x, xticklabels=labels, ylabel="Log-density RMSE", yscale="log",
             title="Full historical upstream pipeline vs current exact PF-ODE (100 steps)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def loss_fix_effect_plot(rows, output_path):
    selected = [
        row for row in rows
        if row["timesteps"] == common.DENSITY_TIMESTEPS
        and row["comparison"] in (
            "before_vs_analytic", "after_vs_analytic", "after_vs_before"
        )
    ]
    labels = sorted({"{}\n{}".format(row["problem"], row["mask"]) for row in selected})
    comparisons = (
        ("before_vs_analytic", "Before fix vs analytic", "#F58518"),
        ("after_vs_analytic", "After fix vs analytic", "#54A24B"),
        ("after_vs_before", "After fix vs before", "#B279A2"),
    )
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(14, 5.5))
    for offset, (comparison, title, color) in enumerate(comparisons):
        lookup = {
            "{}\n{}".format(row["problem"], row["mask"]): max(float(row["rmse"]), 1e-12)
            for row in selected if row["comparison"] == comparison
        }
        axis.bar(x + (offset - 1) * width, [lookup[label] for label in labels],
                 width, label=title, color=color)
    axis.set(xticks=x, xticklabels=labels, ylabel="Held-out log-density RMSE",
             yscale="log", title="Effect of per-example score-loss reduction (100 PF-ODE steps)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def timing_plot(training_rows, inference_rows, output_path):
    problems = ["gaussian", "line", "parabola", "banana"]
    train_lookup = {row["problem"]: row for row in training_rows}
    before_runtime = {
        row["problem"]: row["seconds"] for row in inference_rows
        if row["method"] == "upstream_before_loss_fix"
    }
    after_runtime = {
        row["problem"]: row["seconds"] for row in inference_rows
        if row["method"] == "upstream_after_loss_fix"
    }
    x = np.arange(len(problems))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - width / 2, [train_lookup[p]["before_seconds_estimated"] / 60 for p in problems],
                width, label="Before fix (epoch-normalized estimate)", color="#F58518")
    axes[0].bar(x + width / 2, [train_lookup[p]["after_seconds_measured"] / 60 for p in problems],
                width, label="After fix (measured)", color="#54A24B")
    axes[0].set(xticks=x, xticklabels=problems, ylabel="Training minutes", title="Training time")
    axes[0].legend(fontsize=8)
    axes[1].bar(x - width / 2, [before_runtime[p] for p in problems], width,
                label="Before fix", color="#F58518")
    axes[1].bar(x + width / 2, [after_runtime[p] for p in problems], width,
                label="After fix", color="#54A24B")
    axes[1].set(xticks=x, xticklabels=problems, ylabel="Seconds for identical evaluation workload",
                title="Exact PF-ODE inference time")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(args):
    validate_cpu_cap()
    selected_gpu = autocvd(num_gpus=1, interval=1)
    log(f"[setup] autocvd selected {selected_gpu}")
    upstream_root = Path(args.upstream_root).resolve()
    if not (upstream_root / "src" / "compass" / "Trainer.py").exists():
        raise FileNotFoundError(f"Missing upstream worktree at {upstream_root}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = common.prepare_cases()
    all_rows = []
    parameter_rows = []
    loss_fix_rows = []
    inference_runtime_rows = []
    training_timing_rows = []
    training_configs = {}
    checkpoint_manifest = {}
    with tempfile.TemporaryDirectory(prefix="compass-upstream-full-", dir="/tmp") as temp:
        temp_root = Path(temp)
        for problem, case in cases.items():
            log(f"[full pipeline] {problem}")
            problem_output = OUTPUT_ROOT / problem
            problem_output.mkdir(parents=True, exist_ok=True)
            config = historical_training_config(problem, case)
            training_configs[problem] = config
            training_input = temp_root / f"{problem}_training.npz"
            config_path = temp_root / f"{problem}_training.json"
            evaluation_input = temp_root / f"{problem}_evaluation.npz"
            upstream_result = temp_root / f"{problem}_upstream.npz"
            baseline_result = temp_root / f"{problem}_upstream_before_loss_fix.npz"
            stage1_result = temp_root / f"{problem}_stage1.npz"
            stage2_result = temp_root / f"{problem}_stage2.npz"
            upstream_checkpoint = CHECKPOINT_ROOT / problem / "upstream_score.pt"
            baseline_checkpoint = BASELINE_CHECKPOINT_ROOT / problem / "upstream_score.pt"
            if not baseline_checkpoint.exists():
                raise FileNotFoundError(f"Missing preserved before-fix checkpoint: {baseline_checkpoint}")
            write_training_input(case, training_input)
            with config_path.open("w") as handle:
                json.dump(config, handle, indent=2, sort_keys=True)
            run_training_worker(
                problem, training_input, config_path, upstream_checkpoint,
                upstream_root, force_retrain=args.force_retrain,
            )
            validate_historical_checkpoint(upstream_checkpoint, config)
            validate_historical_checkpoint(baseline_checkpoint, config)
            with upstream_checkpoint.with_name("training_history.json").open() as handle:
                fixed_history = json.load(handle)
            with (BASELINE_CHECKPOINT_ROOT / problem / "training_history.json").open() as handle:
                baseline_history = json.load(handle)
            fixed_epochs = int(fixed_history["epochs_completed"])
            baseline_epochs = len(baseline_history["train_loss"])
            fixed_seconds = float(fixed_history["training_seconds"])
            estimated_baseline_seconds = fixed_seconds * baseline_epochs / max(fixed_epochs, 1)
            training_timing_rows.append({
                "problem": problem,
                "before_epochs": baseline_epochs,
                "after_epochs": fixed_epochs,
                "before_seconds_estimated": estimated_baseline_seconds,
                "after_seconds_measured": fixed_seconds,
                "before_time_basis": "estimated_from_after_seconds_times_epoch_ratio",
            })
            common.write_worker_input(problem, case, evaluation_input)
            current_stage1 = case["checkpoint"]
            current_stage2 = stage2_checkpoint(problem)
            inference_start = perf_counter()
            common.run_worker(
                problem, "upstream", baseline_checkpoint, evaluation_input,
                baseline_result, upstream_root,
            )
            baseline_inference_seconds = perf_counter() - inference_start
            inference_start = perf_counter()
            common.run_worker(
                problem, "upstream", upstream_checkpoint, evaluation_input,
                upstream_result, upstream_root,
            )
            upstream_inference_seconds = perf_counter() - inference_start
            inference_start = perf_counter()
            common.run_worker(
                problem, "current", current_stage1, evaluation_input,
                stage1_result, REPO_ROOT,
            )
            stage1_inference_seconds = perf_counter() - inference_start
            inference_start = perf_counter()
            common.run_worker(
                problem, "current", current_stage2, evaluation_input,
                stage2_result, REPO_ROOT,
            )
            stage2_inference_seconds = perf_counter() - inference_start
            evaluated_points = sum(
                grid["joint"].shape[0] + len(case["sample_rows"]) * len(common.TIMESTEPS)
                for grid in case["grids"].values()
            )
            for method, seconds in (
                ("upstream_before_loss_fix", baseline_inference_seconds),
                ("upstream_after_loss_fix", upstream_inference_seconds),
                ("current_stage1", stage1_inference_seconds),
                ("current_stage2", stage2_inference_seconds),
            ):
                inference_runtime_rows.append({
                    "problem": problem, "method": method, "seconds": seconds,
                    "evaluated_points": evaluated_points,
                    "microseconds_per_point": seconds * 1e6 / evaluated_points,
                })
            upstream_data = np.load(upstream_result)
            baseline_data = np.load(baseline_result)
            stage1_data = np.load(stage1_result)
            stage2_data = np.load(stage2_result)
            for role, checkpoint in (
                ("current_stage1_vs_upstream", current_stage1),
                ("current_stage2_vs_upstream", current_stage2),
            ):
                parameter_rows.append({
                    "problem": problem, "comparison": role,
                    **checkpoint_comparison(checkpoint, upstream_checkpoint),
                })
            for mask_name, grid in case["grids"].items():
                density_values = density_plot(
                    problem, mask_name, grid,
                    upstream_data[f"grid_{mask_name}"],
                    stage1_data[f"grid_{mask_name}"],
                    stage2_data[f"grid_{mask_name}"],
                    problem_output / f"{mask_name}_density_comparison.png",
                )
                analytic = grid["analytic_grid"]
                baseline_values = (
                    np.asarray(baseline_data[f"grid_{mask_name}"]) + grid["shift"]
                ).reshape(grid["shape"])
                for comparison, metric in (
                    ("before_grid_vs_analytic", common.metrics(baseline_values, analytic)),
                    ("after_grid_vs_analytic", common.metrics(density_values["upstream"], analytic)),
                    ("after_grid_vs_before", common.metrics(density_values["upstream"], baseline_values)),
                ):
                    loss_fix_rows.append({
                        "problem": problem, "mask": mask_name,
                        "timesteps": common.DENSITY_TIMESTEPS,
                        "comparison": comparison, **metric,
                    })
                comparisons = {
                    "upstream_grid_vs_analytic": common.metrics(density_values["upstream"], analytic),
                    "current_stage1_grid_vs_analytic": common.metrics(density_values["stage1"], analytic),
                    "current_stage2_grid_vs_analytic": common.metrics(density_values["stage2"], analytic),
                    "current_stage1_grid_vs_upstream": common.metrics(density_values["stage1"], density_values["upstream"]),
                    "current_stage2_grid_vs_upstream": common.metrics(density_values["stage2"], density_values["upstream"]),
                }
                for comparison, metric in comparisons.items():
                    all_rows.append({
                        "problem": problem, "mask": mask_name,
                        "timesteps": common.DENSITY_TIMESTEPS,
                        "comparison": comparison, **metric,
                    })
            for mask_name, grid in case["grids"].items():
                analytic_samples = grid["analytic_samples"]
                shift = grid["shift"]
                for count in common.TIMESTEPS:
                    before_values = baseline_data[f"sample_{mask_name}_{count}"] + shift
                    after_values = upstream_data[f"sample_{mask_name}_{count}"] + shift
                    for comparison, metric in (
                        ("before_vs_analytic", common.metrics(before_values, analytic_samples)),
                        ("after_vs_analytic", common.metrics(after_values, analytic_samples)),
                        ("after_vs_before", common.metrics(after_values, before_values)),
                    ):
                        loss_fix_rows.append({
                            "problem": problem, "mask": mask_name,
                            "timesteps": count, "comparison": comparison, **metric,
                        })
            all_rows.extend(timestep_plot(
                problem, case, upstream_data, stage1_data, stage2_data,
                problem_output / "timestep_comparison.png",
            ))
            checkpoint_manifest[problem] = {
                "upstream_loss_fixed": str(upstream_checkpoint),
                "upstream_before_loss_fix": str(baseline_checkpoint),
                "current_stage1": str(current_stage1),
                "current_stage2": str(current_stage2),
            }

    common.write_csv(OUTPUT_ROOT / "metrics.csv", all_rows)
    common.write_csv(OUTPUT_ROOT / "parameter_comparison.csv", parameter_rows)
    common.write_csv(OUTPUT_ROOT / "loss_fix_effects.csv", loss_fix_rows)
    common.write_csv(OUTPUT_ROOT / "inference_runtime.csv", inference_runtime_rows)
    common.write_csv(OUTPUT_ROOT / "training_timing.csv", training_timing_rows)
    summary_plot(all_rows, OUTPUT_ROOT / "summary.png")
    loss_fix_effect_plot(loss_fix_rows, OUTPUT_ROOT / "loss_fix_effects.png")
    timing_plot(training_timing_rows, inference_runtime_rows, OUTPUT_ROOT / "timing_comparison.png")
    manifest = {
        "comparison_role": (
            "upstream score-only training with per-example loss reduction + upstream exact PF-ODE versus "
            "current Stage-1 and Stage-2 models + current exact PF-ODE"
        ),
        "current_repository": str(REPO_ROOT),
        "current_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "upstream_repository": "https://github.com/bGuenes/COMPASS.git",
        "upstream_variant": "Trainer.loss_fn per-example reduction before sigma weighting",
        "unchanged_components": ["condition-mask sampling", "latent attention mask"],
        "upstream_worktree": str(upstream_root),
        "upstream_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream_root, text=True
        ).strip(),
        "training_configs": training_configs,
        "checkpoints": checkpoint_manifest,
        "eps": common.EPS,
        "density_timesteps": common.DENSITY_TIMESTEPS,
        "timestep_sweep": common.TIMESTEPS,
        "sample_count": common.SAMPLE_COUNT,
        "output": str(OUTPUT_ROOT),
    }
    with (OUTPUT_ROOT / "comparison_config.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    log(f"[done] full historical comparison outputs: {OUTPUT_ROOT}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--problem")
    parser.add_argument("--training-input")
    parser.add_argument("--training-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--upstream-root", default=str(DEFAULT_UPSTREAM_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.train_worker:
        training_worker(parsed)
    else:
        main(parsed)
