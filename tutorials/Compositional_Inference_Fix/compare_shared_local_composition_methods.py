#!/usr/bin/env python3
"""Compare compositional samplers on one learned global/local posterior.

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

CPU_THREAD_LIMIT = 3
logical_cpus = os.cpu_count() or 1
cpu_limit = max(1, min(CPU_THREAD_LIMIT, logical_cpus))
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
    "dpm2_gaussian_without_correctors": {
        "title": "DPM2 + Gaussian composition (no correctors)",
        "filename": "DPM2_gaussian_without_correctors.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss",
            "corrector_steps_interval": 1, "corrector_steps": 0,
            "final_corrector_steps": 0, "snr": 0.2,
        },
        "description": "Second-order probability-flow predictor without Langevin corrector steps.",
    },
    "dpm2_gauss_global_local_without_correctors": {
        "title": "DPM2 + global/local Gaussian composition (no correctors)",
        "filename": "DPM2_gauss_global_local_without_correctors.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "Gauss_global_local",
            "corrector_steps_interval": 1, "corrector_steps": 0,
            "final_corrector_steps": 0, "snr": 0.2,
        },
        "description": "Corrected marginal-global composition with no Langevin corrector steps.",
    },
    "dpm2_full_gaussian": {
        "title": "DPM2 + full Gaussian composition",
        "filename": "02d_dpm2_full_gaussian.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "full_gaussian",
            "corrector_steps_interval": 1, "corrector_steps": 10,
            "final_corrector_steps": 3, "snr": 0.2,
        },
        "description": "Second-order probability-flow predictor with dense Langevin correctors and full-covariance Gaussian score composition.",
    },
    "dpm2_schur_global": {
        "title": "DPM2 + Schur-global Gaussian composition",
        "filename": "02e_dpm2_schur_global.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "Gauss_schur_global",
            "corrector_steps_interval": 1, "corrector_steps": 10,
            "final_corrector_steps": 3, "snr": 0.2,
        },
        "description": "DPM2 with joint global/local covariance reduced to the shared score.",
    },
    "dpm2_gauss_global_local": {
        "title": "DPM2 + global/local Gaussian composition",
        "filename": "02f_dpm2_gauss_global_local.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "Gauss_global_local",
            "corrector_steps_interval": 1, "corrector_steps": 10,
            "final_corrector_steps": 3, "snr": 0.2,
        },
        "description": "DPM2 with Schur-global and implied global/local cross-score corrections.",
    },
    "dpm2_gauss_moment": {
        "title": "DPM2 + Gaussian moment projection",
        "filename": "02g_dpm2_gauss_moment.png",
        "moment_projection": True,
        "sample_kwargs": {
            "method": "dpm", "order": 2,
            "correction": "Gauss_global_local",
            "corrector_steps_interval": 1, "corrector_steps": 0,
            "final_corrector_steps": 0, "snr": 0.2,
            "denoise_clamp": None,
        },
        "description": (
            "Deterministic DPM2 with joint Gaussian posterior means and "
            "covariances projected before global/local composition."
        ),
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
COVARIANCE_SAMPLE_COUNTS = (128, 256, 512, 1024, 2048, 4096)
COVARIANCE_BANK_SEED = 34_711


def regularize_covariances(
    covariance: np.ndarray, shrinkage: float = 0.01, nugget: float = 1e-6,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    diagonal = np.zeros_like(covariance)
    indices = np.arange(covariance.shape[-1])
    diagonal[..., indices, indices] = covariance[..., indices, indices]
    regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    scale = np.maximum(
        1.0, np.trace(regularized, axis1=-2, axis2=-1) / covariance.shape[-1]
    )
    return regularized + nugget * scale[..., None, None] * np.eye(
        covariance.shape[-1]
    )


def build_single_observation_covariance_bank(
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one reusable bank from the toy model's exact p(global, local_j | x_j)."""
    joint_precision = np.array([[5.0, 4.0], [4.0, 5.0]])
    exact_covariance = np.linalg.solve(joint_precision, np.eye(2))
    natural = 4.0 * np.repeat(
        np.asarray(observations, dtype=np.float64).reshape(-1, 1), 2, axis=1
    )
    means = natural @ exact_covariance.T
    rng = np.random.default_rng(COVARIANCE_BANK_SEED)
    noise = rng.standard_normal(
        (len(means), max(COVARIANCE_SAMPLE_COUNTS), 2)
    )
    draws = means[:, None, :] + noise @ np.linalg.cholesky(exact_covariance).T
    return draws, exact_covariance


def empirical_covariances(bank: np.ndarray, count: int) -> np.ndarray:
    draws = np.asarray(bank[:, :count], dtype=np.float64)
    centered = draws - draws.mean(axis=1, keepdims=True)
    covariance = np.einsum("nsi,nsj->nij", centered, centered) / (count - 1)
    return regularize_covariances(covariance)


def write_covariance_convergence(
    output: Path, bank: np.ndarray, exact_covariance: np.ndarray,
) -> None:
    rows = []
    exact_norm = np.linalg.norm(exact_covariance)
    for count in COVARIANCE_SAMPLE_COUNTS:
        covariance = empirical_covariances(bank, count)
        for subject, estimate in enumerate(covariance):
            rows.append({
                "samples": count,
                "subject": subject,
                "joint_relative_frobenius_error": (
                    np.linalg.norm(estimate - exact_covariance) / exact_norm
                ),
                "global_variance_relative_error": abs(
                    estimate[0, 0] / exact_covariance[0, 0] - 1.0
                ),
                "condition_number": np.linalg.cond(estimate),
            })
    with (output / "covariance_sample_convergence.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)




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
        "covariance_condition_number": float(raw.get("covariance_condition_number", np.nan)),
        "pd_repair_fraction": float(raw.get("pd_repair_fraction", 0.0)),
        "pd_maximum_relative_repair": float(raw.get("pd_maximum_relative_repair", 0.0)),
        # Share of per-observation information eigenvalues that came out
        # negative, i.e. that claimed an observation makes the shared
        # parameters *less* certain than the prior alone. Nonzero means the
        # pilot covariance left the model class and plain GAUSS would have
        # composed against an indefinite precision.
        "negative_information_fraction": float(
            raw.get("negative_information_fraction", 0.0)
        ),
        "minimum_information_eigenvalue": float(
            raw.get("minimum_information_eigenvalue", np.nan)
        ),
        "maximum_relative_adaptation": float(
            raw.get("maximum_relative_adaptation", 0.0)
        ),
    }]
    rows.append({
        "method": method,
        "parameter": "locals_mean",
        "mean_error_in_exact_std": np.mean(np.abs(local_means - exact[1:]) / exact_std[1:]),
        "covariance_condition_number": float(raw.get("covariance_condition_number", np.nan)),
        "pd_repair_fraction": float(raw.get("pd_repair_fraction", 0.0)),
        "pd_maximum_relative_repair": float(raw.get("pd_maximum_relative_repair", 0.0)),
        "negative_information_fraction": float(
            raw.get("negative_information_fraction", 0.0)
        ),
        "minimum_information_eigenvalue": float(
            raw.get("minimum_information_eigenvalue", np.nan)
        ),
        "maximum_relative_adaptation": float(
            raw.get("maximum_relative_adaptation", 0.0)
        ),
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
    covariance_bank: np.ndarray,
    timesteps: int = TIMESTEPS,
    posterior_samples: int = N_SAMPLES,
) -> None:
    variant = VARIANTS[method]
    result_path = output / f"{method}.npz"
    if result_path.exists() and not force:
        print(f"Reusing {result_path}")
    else:
        x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)
        n_observations = len(x)
        moment_count = max(COVARIANCE_SAMPLE_COUNTS)
        covariance = empirical_covariances(covariance_bank, moment_count)
        posterior_mean = np.asarray(
            covariance_bank[:, :moment_count], dtype=np.float64
        ).mean(axis=1)
        sample_kwargs = dict(variant["sample_kwargs"])
        if sample_kwargs["correction"] == "gauss":
            sample_kwargs["posterior_precision"] = torch.from_numpy(
                1.0 / covariance[:, :1, 0]
            ).to(torch.float32)
        elif sample_kwargs["correction"] == "full_gaussian":
            sample_kwargs["posterior_covariance"] = torch.from_numpy(
                covariance[:, :1, :1]
            )
        elif sample_kwargs["correction"] in {"Gauss_schur_global", "Gauss_global_local"}:
            sample_kwargs["posterior_covariance"] = torch.from_numpy(covariance)
            if sample_kwargs["correction"] == "Gauss_global_local":
                # Pilot draws supply p(g_0 | x_j) directly, with local latents
                # integrated out before the shared score is composed.
                sample_kwargs["global_posterior_mean"] = torch.from_numpy(
                    posterior_mean[:, :1]
                )
                sample_kwargs["global_posterior_covariance"] = torch.from_numpy(
                    covariance[:, :1, :1]
                )
            if variant.get("moment_projection", False):
                sample_kwargs["posterior_mean"] = torch.from_numpy(posterior_mean)
        torch.manual_seed(INFERENCE_SEED)
        torch.cuda.manual_seed_all(INFERENCE_SEED)
        started = time.perf_counter()
        samples = model.sample(
            x=x, multi_obs_inference=True, hierarchy=[0], prior=([0.0], [1.0]),
            num_samples=posterior_samples, timesteps=timesteps,
            device="cuda", verbose=True, **sample_kwargs,
        )
        diagnostics = model.multi_obs_sampler.covariance_diagnostics
        covariance_used = (
            covariance if sample_kwargs["correction"] in {
                "Gauss_schur_global", "Gauss_global_local"
            } else covariance[:, :1, :1]
        )
        runtime = time.perf_counter() - started
        covariance_condition_number = max(map(np.linalg.cond, covariance_used))
        samples = samples.detach().cpu().numpy()
        synchronization_error = float(np.max(np.abs(samples[:, :, 0] - samples[0:1, :, 0])))
        if synchronization_error > 1e-6:
            raise AssertionError(f"{method}: global samples are not synchronized: {synchronization_error}")
        np.savez_compressed(
            result_path,
            x_observed=reference["x_observed"],
            global_truth=reference["global_truth"],
            covariance_condition_number=covariance_condition_number,
            pd_repair_fraction=diagnostics["repair_fraction"],
            pd_maximum_relative_repair=diagnostics["maximum_relative_repair"],
            pd_minimum_eigenvalue_before=diagnostics["minimum_eigenvalue_before"] if diagnostics["minimum_eigenvalue_before"] is not None else np.nan,
            pd_minimum_eigenvalue_after=diagnostics["minimum_eigenvalue_after"] if diagnostics["minimum_eigenvalue_after"] is not None else np.nan,
            negative_information_fraction=diagnostics["negative_information_fraction"],
            minimum_information_eigenvalue=diagnostics["minimum_information_eigenvalue"] if diagnostics["minimum_information_eigenvalue"] is not None else np.nan,
            maximum_relative_adaptation=diagnostics["maximum_relative_adaptation"],
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
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS,
                        help="number of noise-schedule points")
    parser.add_argument("--posterior-samples", type=int, default=N_SAMPLES,
                        help="posterior draws per method")
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(VARIANTS))
    if unknown:
        parser.error("unknown methods: " + ", ".join(unknown))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_result(args.reference)
    covariance_bank, exact_single_covariance = (
        build_single_observation_covariance_bank(reference["x_observed"])
    )
    write_covariance_convergence(
        args.output_dir, covariance_bank, exact_single_covariance
    )
    print(f"Loading shared score model from {args.checkpoint}")
    model = load_score_checkpoint(args.checkpoint)
    for method in methods:
        run_method(
            method, model, reference, args.output_dir, args.force,
            covariance_bank, args.timesteps, args.posterior_samples,
        )
        write_metrics(args.output_dir, methods)
    metadata = {
        "checkpoint": str(args.checkpoint), "reference": str(args.reference),
        "n_observations": int(len(reference["x_observed"])),
        "posterior_samples": args.posterior_samples, "timesteps": args.timesteps,
        "covariance_sample_counts": COVARIANCE_SAMPLE_COUNTS,
        "covariance_bank_seed": COVARIANCE_BANK_SEED,
        "inference_seed": INFERENCE_SEED,
        "variants": {method: VARIANTS[method] for method in methods},
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Comparison outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
