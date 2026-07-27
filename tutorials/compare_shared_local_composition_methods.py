#!/usr/bin/env python3
"""Compare three compositional samplers on one learned global/local posterior.

All variants reuse the same trained score network, 30 observations, analytic
posterior, 3,000 posterior draws, 100 noise levels, and random seed. Results are
saved incrementally, so an interrupted run can resume without repeating methods.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

CPU_FRACTION = 0.06
logical_cpus = os.cpu_count() or 1
cpu_limit = max(1, int(logical_cpus * CPU_FRACTION))
available_cpus = tuple(sorted(os.sched_getaffinity(0)))
selected_cpus = available_cpus[:min(cpu_limit, len(available_cpus))]
os.sched_setaffinity(0, selected_cpus)
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(len(selected_cpus))

from autocvd import autocvd

# Reserve before importing Torch so CUDA_VISIBLE_DEVICES exposes only one GPU.
autocvd(num_gpus=1, interval=1)

import numpy as np
import torch

from compass import ScoreBasedInferenceModel as SBIm
from plot_local_vs_global_validation import plot_shared_local

ROOT = Path(__file__).resolve().parent
BASE_OUTPUT = ROOT / "output" / "compositional_inference"
DEFAULT_REFERENCE = BASE_OUTPUT / "06b_shared_local" / "raw_plot_data.npz"
DEFAULT_CHECKPOINT = BASE_OUTPUT / "models" / "shared_local_mixture_full" / "Model_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "output" / "compositional_inference_local_vs_global" / "method_comparison"
N_SAMPLES = 3_000
TIMESTEPS = 100
INFERENCE_SEED = 1_208

VARIANTS = {
    "dpm2_gaussian": {
        "title": "DPM2 + Gaussian composition",
        "filename": "02a_dpm2_gaussian.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss",
            "corrector_steps_interval": 1, "corrector_steps": 10,
            "final_corrector_steps": 3, "snr": 0.2,
        },
        "description": "Second-order probability-flow predictor with dense Langevin correctors and Gaussian score composition.",
    },
    "pfode_gaussian": {
        "title": "PF-ODE + Gaussian composition",
        "filename": "02b_pfode_gaussian.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss",
            "corrector_steps_interval": 1, "corrector_steps": 0,
            "final_corrector_steps": 0, "snr": 0.2,
        },
        "description": "Deterministic second-order probability-flow ODE integration in noise-scale space; all Langevin correctors disabled.",
    },
    "langevin_fnpe": {
        "title": "Annealed Langevin + F-NPSE composition",
        "filename": "02c_langevin_fnpe.png",
        "sample_kwargs": {
            "method": "langevin", "correction": "fnpe",
            "corrector_steps": 10, "snr": 0.2,
        },
        "description": "Pure annealed Langevin dynamics targeting the F-NPSE bridging densities.",
    },
}


def metric_rows(method: str, raw: dict[str, np.ndarray], runtime: float) -> list[dict[str, object]]:
    exact = raw["exact_joint_mean"]
    covariance = raw["exact_joint_covariance"]
    global_samples = raw["compass_global_samples"]
    local_samples = raw["compass_local_samples"]
    exact_std = np.sqrt(np.diag(covariance))
    local_means = local_samples.mean(axis=1)
    rows = [{
        "method": method,
        "parameter": "global",
        "mean_error_in_exact_std": abs(global_samples.mean() - exact[0]) / exact_std[0],
        "std_ratio_to_exact": global_samples.std(ddof=1) / exact_std[0],
        "runtime_seconds": runtime,
        "shared_synchronization_max_abs": float(raw["shared_synchronization_max_abs"]),
    }]
    rows.append({
        "method": method,
        "parameter": "locals_mean",
        "mean_error_in_exact_std": np.mean(np.abs(local_means - exact[1:]) / exact_std[1:]),
        "std_ratio_to_exact": np.mean(local_samples.std(axis=1, ddof=1) / exact_std[1:]),
        "runtime_seconds": runtime,
        "shared_synchronization_max_abs": float(raw["shared_synchronization_max_abs"]),
    })
    return rows


def load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def write_metrics(output: Path, methods: list[str]) -> None:
    rows: list[dict[str, object]] = []
    for method in methods:
        result_path = output / f"{method}.npz"
        if result_path.exists():
            raw = load_result(result_path)
            rows.extend(metric_rows(method, raw, float(raw["runtime_seconds"])))
    if not rows:
        return
    with (output / "method_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_method(
    method: str,
    model: SBIm,
    reference: dict[str, np.ndarray],
    output: Path,
    force: bool,
) -> None:
    variant = VARIANTS[method]
    result_path = output / f"{method}.npz"
    if result_path.exists() and not force:
        print(f"Reusing {result_path}")
    else:
        x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)
        n_observations = len(x)
        precision = 1.0 + 1.0 / (1.0 + 0.5**2)
        sample_kwargs = dict(variant["sample_kwargs"])
        if sample_kwargs["correction"] == "gauss":
            sample_kwargs["posterior_precision"] = torch.full((n_observations, 1), precision)
        torch.manual_seed(INFERENCE_SEED)
        torch.cuda.manual_seed_all(INFERENCE_SEED)
        started = time.perf_counter()
        samples = model.sample(
            x=x, multi_obs_inference=True, hierarchy=[0], prior=([0.0], [1.0]),
            num_samples=N_SAMPLES, timesteps=TIMESTEPS,
            device="cuda", verbose=True, **sample_kwargs,
        )
        runtime = time.perf_counter() - started
        samples = samples.detach().cpu().numpy()
        synchronization_error = float(np.max(np.abs(samples[:, :, 0] - samples[0:1, :, 0])))
        if synchronization_error > 1e-6:
            raise AssertionError(f"{method}: global samples are not synchronized: {synchronization_error}")
        np.savez_compressed(
            result_path,
            x_observed=reference["x_observed"],
            global_truth=reference["global_truth"],
            local_truth=reference["local_truth"],
            exact_joint_mean=reference["exact_joint_mean"],
            exact_joint_covariance=reference["exact_joint_covariance"],
            compass_global_samples=samples[0, :, 0],
            compass_local_samples=samples[:, :, 1],
            shared_synchronization_max_abs=synchronization_error,
            runtime_seconds=runtime,
        )
        print(f"{method} finished in {runtime:.1f}s; wrote {result_path}")
    plot_shared_local(result_path, output / variant["filename"], title=variant["title"])


def load_score_checkpoint(path: Path) -> SBIm:
    """Load score-backbone weights, tolerating an unused divergence head."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SBIm(
        nodes_size=checkpoint["nodes_size"], sde_type=checkpoint["sde_type"],
        sigma=checkpoint["sigma"], beta_min=checkpoint["beta_min"],
        beta_max=checkpoint["beta_max"], hidden_size=checkpoint["hidden_size"],
        depth=checkpoint["depth"], num_heads=checkpoint["num_heads"],
        mlp_ratio=checkpoint["mlp_ratio"], device="cuda",
    )
    score_state = {
        key: value for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith("divergence_head.")
    }
    model.model.load_state_dict(score_state, strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", default=",".join(VARIANTS),
                        help="comma-separated method keys")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(VARIANTS))
    if unknown:
        parser.error("unknown methods: " + ", ".join(unknown))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_result(args.reference)
    print(f"Loading shared score model from {args.checkpoint}")
    model = load_score_checkpoint(args.checkpoint)
    for method in methods:
        run_method(method, model, reference, args.output_dir, args.force)
        write_metrics(args.output_dir, methods)
    metadata = {
        "checkpoint": str(args.checkpoint), "reference": str(args.reference),
        "n_observations": int(len(reference["x_observed"])),
        "posterior_samples": N_SAMPLES, "timesteps": TIMESTEPS,
        "inference_seed": INFERENCE_SEED,
        "variants": {method: VARIANTS[method] for method in methods},
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Comparison outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
