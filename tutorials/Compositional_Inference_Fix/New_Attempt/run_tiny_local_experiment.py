#!/usr/bin/env python3
"""Retrain the shared/local score model with independently controllable local-
parameter and observation-noise scales, then evaluate "gauss_hierarchical".

The generative model has two separate noise scales, easy to conflate:

    x_j = g + l_j + epsilon_j
    l_j ~ N(0, S0L^2)        <- the local parameter's own spread across
                                 observations ("--local-std")
    epsilon_j ~ N(0, SXH^2)  <- observation noise added on top of g+l_j to
                                 produce the observed x_j ("--obs-noise")

Both are exposed independently here:
  - Shrinking --local-std makes the true local parameters themselves tiny
    (a degenerate-hierarchy limit; the derivation's own sanity check says
    the arrow-precision system must collapse to the ordinary GAUSS formula
    there, so gauss_hierarchical should become essentially exact).
  - Shrinking --obs-noise instead leaves the local parameters' own spread
    untouched, but makes each x_j an almost noise-free measurement of
    g+l_j -- a different diagnostic (highly informative individual
    likelihoods) than the degenerate-hierarchy limit above.

This script:
  1. Monkeypatches Compositional_Inference.S0L and .SXH so every reused
     helper (simulate_hierarchical_pairs, exact_hierarchical_posterior) is
     consistent with the new generative model. (The generative model
     changed, so a *fresh* score network must be trained from scratch --
     the old checkpoint's targets are for the wrong distribution.)
  2. Trains that fresh network (same architecture/training recipe as the
     existing "shared_local_mixture_full" HQ checkpoint by default, or the
     smaller --model-size small ablation; --quick shrinks data/epochs only,
     independent of --model-size).
  3. Draws a new set of --n-observations observations from the new model
     and computes their exact joint Gaussian posterior analytically.
  4. Runs, all against this new truth and this new network:
       - dpm2_gaussian            (hierarchy-blind "gauss" baseline)
       - dpm2_gauss_global_local  (oracle-moment cheat, for reference only)
       - dpm2_gauss_moment        (oracle-moment cheat, deterministic)
       - gauss_hierarchical_dense_correctors   (ours, dense Langevin correctors)
       - gauss_hierarchical_deterministic      (ours, zero correctors)
  5. Writes per-method .npz/plots, a combined metrics CSV, and a bar chart
     into --output-dir (default: a name encoding local-std/obs-noise/model-size
     next to this file).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

CPU_THREAD_LIMIT = 3
logical_cpus = os.cpu_count() or 1
cpu_limit = max(1, min(CPU_THREAD_LIMIT, logical_cpus))
available_cpus = tuple(sorted(os.sched_getaffinity(0)))
selected_cpus = available_cpus[:min(cpu_limit, len(available_cpus))]
os.sched_setaffinity(0, selected_cpus)
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(len(selected_cpus))

from autocvd import autocvd

autocvd(num_gpus=1, interval=1)

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))

import Compositional_Inference as ci  # noqa: E402
from compare_shared_local_composition_methods import (  # noqa: E402
    VARIANTS, COVARIANCE_SAMPLE_COUNTS, empirical_covariances,
    load_result, metric_rows, run_method as run_oracle_method,
)
from plot_local_vs_global_validation import plot_shared_local  # noqa: E402

MODEL_SIZES = {
    # ~5.2M parameters -- COMPASS's generic HQ backbone, reused unchanged
    # from the real (non-toy) applications.
    "full": None,  # resolved to ci.SHARED_LOCAL_HQ_MODEL_KWARGS at runtime
    # ~552K parameters -- a capacity ablation for this 3-scalar Gaussian toy
    # problem, to test whether the HQ backbone's size is masking or causing
    # any of gauss_hierarchical's accuracy gap.
    "small": {
        "sde_type": "vesde", "sigma": 8.0, "hidden_size": 32,
        "depth": 3, "num_heads": 4, "mlp_ratio": 2,
    },
    # ~109K parameters -- roughly 5x fewer than "small" (and ~48x fewer than
    # "full"), to push the capacity ablation further and see whether
    # gauss_hierarchical's accuracy gap grows once the backbone is this thin.
    "very_small": {
        "sde_type": "vesde", "sigma": 8.0, "hidden_size": 16,
        "depth": 1, "num_heads": 2, "mlp_ratio": 1,
    },
    # ~20.4K parameters (<25K) -- the floor is set by the fixed
    # time_embedding_size=256 adaLN-modulation layers (not exposed as a
    # size knob), which cost ~O(hidden_size) each; hidden_size=3/depth=1/
    # num_heads=1/mlp_ratio=1 is the largest hidden_size that still clears
    # the <25K budget.
    "extremely_small": {
        "sde_type": "vesde", "sigma": 8.0, "hidden_size": 3,
        "depth": 1, "num_heads": 1, "mlp_ratio": 1,
    },
}

NEW_VARIANTS = {
    "gauss_hierarchical_dense_correctors": {
        "title": "DPM2 + true compositional hierarchical GAUSS (tiny local std)",
        "filename": "02h_dpm2_gauss_hierarchical.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "corrector_steps_interval": 1, "corrector_steps": 10,
            "final_corrector_steps": 3, "snr": 0.2,
        },
    },
    "gauss_hierarchical_deterministic": {
        "title": "DPM2 + true compositional hierarchical GAUSS, deterministic (tiny local std)",
        "filename": "02i_dpm2_gauss_hierarchical_deterministic.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "corrector_steps_interval": 1, "corrector_steps": 0,
            "final_corrector_steps": 0, "snr": 0.2, "denoise_clamp": None,
        },
    },
}

BAR_CHART_LABELS = {
    "dpm2_gaussian": "DPM2 + Gaussian\n(no hierarchy correction)",
    "dpm2_gauss_global_local": "DPM2 + Gauss_global_local\n(oracle moments -- cheat)",
    "dpm2_gauss_moment": "DPM2 + moment projection\n(oracle moments -- cheat)",
    "gauss_hierarchical_dense_correctors": "gauss_hierarchical\n(dense correctors, ours)",
    "gauss_hierarchical_deterministic": "gauss_hierarchical\n(deterministic, ours)",
}
BAR_CHART_COLORS = {
    "dpm2_gaussian": "#8D99AE",
    "dpm2_gauss_global_local": "#EF476F",
    "dpm2_gauss_moment": "#EF476F",
    "gauss_hierarchical_dense_correctors": "#3A86FF",
    "gauss_hierarchical_deterministic": "#2A9D8F",
}
BAR_CHART_ORDER = list(BAR_CHART_LABELS)


def build_generalized_covariance_bank(
    observations: np.ndarray, s0g: float, s0l: float, sxh: float,
    sample_counts: tuple[int, ...], seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalization of build_single_observation_covariance_bank to arbitrary S0G/S0L/SXH.

    The original hardcodes joint_precision=[[5,4],[4,5]], which is only the
    exact p(g,l_j|x_j) precision for the original S0G=S0L=1, SXH=0.5 model
    (diag(1/S0G^2, 1/S0L^2) + (1/SXH^2)*ones(2,2)). Reduces to the original
    formula exactly when s0g=s0l=1, sxh=0.5.
    """
    joint_precision = np.diag([1.0 / s0g**2, 1.0 / s0l**2]) + (1.0 / sxh**2) * np.ones((2, 2))
    exact_covariance = np.linalg.solve(joint_precision, np.eye(2))
    natural = (1.0 / sxh**2) * np.repeat(
        np.asarray(observations, dtype=np.float64).reshape(-1, 1), 2, axis=1
    )
    means = natural @ exact_covariance.T
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(means), max(sample_counts), 2))
    draws = means[:, None, :] + noise @ np.linalg.cholesky(exact_covariance).T
    return draws, exact_covariance


def run_gauss_hierarchical_variant(
    method: str, variant: dict, model, reference: dict[str, np.ndarray],
    output: Path, force: bool, timesteps: int, posterior_samples: int,
    precision_est_samples: int, precision_est_timesteps: int,
) -> None:
    """Same logic as run_gauss_hierarchical_experiment.run_no_oracle_method:
    no oracle mean/covariance is ever supplied; every covariance is estimated
    from the trained network's own DDIM draws, and gauss_hierarchical's API
    refuses posterior_mean/global_posterior_mean outright.
    """
    result_path = output / f"{method}.npz"
    if result_path.exists() and not force:
        print(f"Reusing {result_path}")
        plot_shared_local(result_path, output / variant["filename"], title=variant["title"])
        return

    x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)
    sample_kwargs = dict(variant["sample_kwargs"])
    torch.manual_seed(1_208)
    torch.cuda.manual_seed_all(1_208)
    started = time.perf_counter()
    samples = model.sample(
        x=x, multi_obs_inference=True, hierarchy=[0], prior=([0.0], [1.0]),
        num_samples=posterior_samples, timesteps=timesteps,
        precision_est_samples=precision_est_samples,
        precision_est_timesteps=precision_est_timesteps,
        device="cuda", verbose=True, **sample_kwargs,
    )
    diagnostics = model.multi_obs_sampler.covariance_diagnostics
    covariance_used = model.multi_obs_sampler.posterior_covariance.detach().cpu().numpy()
    runtime = time.perf_counter() - started
    covariance_condition_number = max(np.linalg.cond(covariance_used[i]) for i in range(covariance_used.shape[0]))
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


def write_combined_metrics(output: Path, methods: list[str]) -> Path | None:
    rows: list[dict[str, object]] = []
    for method in methods:
        result_path = output / f"{method}.npz"
        if result_path.exists():
            raw = load_result(result_path)
            rows.extend(metric_rows(method, raw, float(raw["runtime_seconds"])))
    if not rows:
        return None
    path = output / "combined_method_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    return path


def plot_bar_chart(output: Path, metrics_csv: Path, subtitle: str) -> None:
    with metrics_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method: dict[str, dict[str, dict]] = {row["method"]: {} for row in rows}
    for row in rows:
        by_method[row["method"]][row["parameter"]] = row
    methods = [m for m in BAR_CHART_ORDER if m in by_method]
    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = range(len(methods))
    colors = [BAR_CHART_COLORS[m] for m in methods]
    labels = [BAR_CHART_LABELS[m] for m in methods]

    global_err = [float(by_method[m]["global"]["mean_error_in_exact_std"]) for m in methods]
    axes[0].bar(x, global_err, color=colors)
    axes[0].set(title="Global parameter: mean error / exact std", ylabel="|error| / analytic sigma")

    local_err = [float(by_method[m]["locals_mean"]["mean_error_in_exact_std"]) for m in methods]
    axes[1].bar(x, local_err, color=colors)
    axes[1].set(title="Local parameters: mean |error| / exact std", ylabel="|error| / analytic sigma")

    runtime = [float(by_method[m]["global"]["runtime_seconds"]) for m in methods]
    axes[2].bar(x, runtime, color=colors)
    axes[2].set(title="Wall-clock runtime", ylabel="seconds")

    for axis in axes:
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"gauss_hierarchical composition accuracy\n{subtitle}",
        fontsize=12, fontweight="bold", color="#17223B",
    )
    fig.tight_layout()
    out_path = output / "03_method_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-std", type=float, default=ci.S0L,
                         help="New S0L (local parameter's own prior std). Original model uses 1.0. "
                              "Independent of --obs-noise.")
    parser.add_argument("--obs-noise", type=float, default=ci.SXH,
                         help="New SXH (observation noise added on top of g+l_j to produce x_j). "
                              "Original model uses 0.5. Independent of --local-std.")
    parser.add_argument("--n-observations", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=ci.SHARED_LOCAL_HQ_TRAIN_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=ci.SHARED_LOCAL_HQ_VALIDATION_SAMPLES)
    parser.add_argument("--max-epochs", type=int, default=ci.SHARED_LOCAL_HQ_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=ci.SHARED_LOCAL_HQ_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=ci.SHARED_LOCAL_HQ_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=ci.SHARED_LOCAL_HQ_LR)
    parser.add_argument("--quick", action="store_true",
                         help="Few training samples / epochs, for a fast pipeline smoke test. "
                              "Independent of --model-size.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZES), default="full",
                         help="Score-network capacity: 'full' (~5.2M params, the HQ backbone "
                              "used everywhere else) or 'small' (~552K params, a capacity "
                              "ablation for this toy problem). Independent of --quick.")
    parser.add_argument("--posterior-samples", type=int, default=3_000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--precision-est-samples", type=int, default=4_096)
    parser.add_argument("--precision-est-timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if a checkpoint exists.")
    parser.add_argument("--force-rerun", action="store_true", help="Resample every method even if its .npz exists.")
    args = parser.parse_args()

    run_tag = f"local_std_{args.local_std:g}_obs_noise_{args.obs_noise:g}_{args.model_size}"
    output_dir = args.output_dir or (ROOT / run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The reused helpers (simulate_hierarchical_pairs, exact_hierarchical_posterior)
    # read Compositional_Inference.S0L/.SXH as module globals at call time, so
    # patching them here keeps every downstream call consistent without
    # duplicating the generative-model code.
    original_s0l, original_sxh = ci.S0L, ci.SXH
    ci.S0L = args.local_std
    ci.SXH = args.obs_noise
    print(f"Patched Compositional_Inference.S0L = {ci.S0L} (was {original_s0l})")
    print(f"Patched Compositional_Inference.SXH = {ci.SXH} (was {original_sxh})")

    train_samples = 2_000 if args.quick else args.train_samples
    validation_samples = 500 if args.quick else args.validation_samples
    max_epochs = 3 if args.quick else args.max_epochs

    cfg = ci.RunConfig(
        output_dir=output_dir, device="cuda", seed=args.seed,
        train_samples=train_samples, validation_samples=validation_samples,
        max_epochs=max_epochs, patience=args.patience, batch_size=args.batch_size,
        posterior_samples=args.posterior_samples, timesteps=args.timesteps,
    )
    model_dir = ci.checkpoint_dir(cfg, f"shared_local_{run_tag}")
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists() and not args.force_retrain:
        print(f"Loading existing checkpoint: {checkpoint}")
        model = ci.SBIm.load(str(checkpoint), device=cfg.device)
    else:
        ci.seed_all(cfg.seed + 1_190)
        theta_train, x_train = ci.simulate_hierarchical_pairs(train_samples)
        theta_val, x_val = ci.simulate_hierarchical_pairs(validation_samples)
        model_kwargs = MODEL_SIZES[args.model_size] or ci.SHARED_LOCAL_HQ_MODEL_KWARGS
        model = ci.SBIm(nodes_size=3, device=cfg.device, **model_kwargs)
        param_count = sum(p.numel() for p in model.model.parameters())
        print(
            f"Training shared/local score model from scratch with S0L={args.local_std}, "
            f"SXH={args.obs_noise}, model_size={args.model_size} ({param_count:,} params) "
            f"on {train_samples:,} simulations ({'quick' if args.quick else 'full'} data/epoch settings)..."
        )
        started = time.perf_counter()
        model.train(
            theta=theta_train, x=x_train, theta_val=theta_val, x_val=x_val,
            batch_size=args.batch_size, max_epochs=max_epochs,
            early_stopping_patience=args.patience, lr=args.lr, time_sampling="mixture",
            device=cfg.device, verbose=True, path=str(model_dir),
        )
        print(f"  trained in {(time.perf_counter() - started) / 60:.1f} minutes")

    ci.seed_all(cfg.seed + 1_200)
    n = args.n_observations
    global_true = float(ci.MUG + ci.S0G * torch.randn(()))
    local_true = args.local_std * torch.randn(n)
    x = global_true + local_true[:, None] + ci.SXH * torch.randn(n, 1)
    exact_mean, exact_covariance = ci.exact_hierarchical_posterior(x)
    reference = {
        "x_observed": ci.tensor_numpy(x),
        "global_truth": np.asarray(global_true),
        "local_truth": ci.tensor_numpy(local_true),
        "exact_joint_mean": exact_mean,
        "exact_joint_covariance": exact_covariance,
    }
    ci.save_raw_data(output_dir / "reference.npz", **reference)
    print(
        f"New reference: n={n} observations, S0L={args.local_std}, SXH={args.obs_noise}, "
        f"global truth={global_true:.3f}, analytic global std={exact_covariance[0, 0] ** 0.5:.4f}"
    )

    covariance_bank, _ = build_generalized_covariance_bank(
        reference["x_observed"], ci.S0G, args.local_std, args.obs_noise,
        COVARIANCE_SAMPLE_COUNTS, seed=34_711,
    )

    for method in ("dpm2_gaussian", "dpm2_gauss_global_local", "dpm2_gauss_moment"):
        run_oracle_method(
            method, model, reference, output_dir, args.force_rerun,
            covariance_bank, args.timesteps, args.posterior_samples,
        )

    for method, variant in NEW_VARIANTS.items():
        run_gauss_hierarchical_variant(
            method, variant, model, reference, output_dir, args.force_rerun,
            args.timesteps, args.posterior_samples,
            args.precision_est_samples, args.precision_est_timesteps,
        )

    all_methods = [
        "dpm2_gaussian", "dpm2_gauss_global_local", "dpm2_gauss_moment",
        "gauss_hierarchical_dense_correctors", "gauss_hierarchical_deterministic",
    ]
    metrics_csv = write_combined_metrics(output_dir, all_methods)
    if metrics_csv is not None:
        subtitle = (
            f"S0L={args.local_std:g} (local prior std), SXH={args.obs_noise:g} (obs. noise), "
            f"model_size={args.model_size}"
        )
        plot_bar_chart(output_dir, metrics_csv, subtitle)

    metadata = {
        "local_std": args.local_std, "obs_noise": args.obs_noise, "n_observations": n,
        "model_size": args.model_size,
        "train_samples": train_samples, "validation_samples": validation_samples,
        "max_epochs": max_epochs, "quick": args.quick,
        "posterior_samples": args.posterior_samples, "timesteps": args.timesteps,
        "precision_est_samples": args.precision_est_samples,
        "precision_est_timesteps": args.precision_est_timesteps,
        "seed": args.seed, "checkpoint": str(checkpoint),
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Experiment outputs written to {output_dir}")


if __name__ == "__main__":
    main()
