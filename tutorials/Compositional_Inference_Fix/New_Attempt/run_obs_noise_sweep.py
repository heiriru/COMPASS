#!/usr/bin/env python3
"""Sweep observation noise (SXH) across three compositional samplers.

For each --obs-noise value in the sweep, this script:

  1. Retrains a fresh shared/local score network from scratch (the generative
     model changes with SXH, so a checkpoint trained at one obs-noise level
     is not valid at another) at the requested --model-size ("full", "small",
     "very_small", or "extremely_small" -- see MODEL_SIZES in
     run_tiny_local_experiment.py, imported below).
  2. Draws a fresh set of --n-observations observations and their exact
     joint Gaussian posterior.
  3. Runs three methods against that network/reference:
       - gauss_hierarchical_dpm50_deterministic: DPM2, 50 timesteps, our
         true compositional hierarchical GAUSS, zero Langevin correctors.
       - gauss_hierarchical_dpm50_langevin2: DPM2, 50 timesteps, hierarchical
         GAUSS with 2 Langevin corrector steps per predictor step.
       - langevin_fnpe: annealed Langevin + F-NPSE composition, 100 timesteps.
  4. Saves every checkpoint, every method's raw .npz samples, and a 4-panel
     plot_shared_local dashboard (observations / shared posterior / local
     posteriors / local recovery) per method, all under
     --output-dir/obs_noise_<value>/.
  5. After the sweep, writes a combined sweep_metrics.csv and a single
     obs_noise-vs-global-accuracy summary plot across all methods.

Everything is written under a dedicated subfolder (default:
New_Attempt/obs_noise_sweep_<model-size>/) so repeated sweeps at different
model sizes don't collide.
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
    COVARIANCE_SAMPLE_COUNTS, load_result, metric_rows,
    run_method as run_oracle_method,
)
from plot_local_vs_global_validation import plot_shared_local  # noqa: E402
from run_tiny_local_experiment import (  # noqa: E402
    MODEL_SIZES, build_generalized_covariance_bank, run_gauss_hierarchical_variant,
)

DEFAULT_OBS_NOISE_VALUES = tuple(round(float(v), 4) for v in np.geomspace(0.05, 2.0, 10))

# Maps our friendly method key -> the .npz stem it is actually saved under.
# The two gauss_hierarchical methods save under their own friendly key
# (run_gauss_hierarchical_variant uses the `method` argument as the stem);
# langevin_fnpe reuses compare_shared_local_composition_methods.VARIANTS,
# keyed by its own pre-existing method name.
NPZ_STEM = {
    "gauss_hierarchical_dpm50_deterministic": "gauss_hierarchical_dpm50_deterministic",
    "gauss_hierarchical_dpm50_langevin2": "gauss_hierarchical_dpm50_langevin2",
    "langevin_fnpe": "langevin_fnpe",
}
METHOD_ORDER = list(NPZ_STEM)
METHOD_LABELS = {
    "gauss_hierarchical_dpm50_deterministic": "hierarchical GAUSS\n(DPM2-50, deterministic)",
    "gauss_hierarchical_dpm50_langevin2": "hierarchical GAUSS\n(DPM2-50, 2 correctors/step)",
    "langevin_fnpe": "Langevin + F-NPSE\n(100 steps)",
}
METHOD_COLORS = {
    "gauss_hierarchical_dpm50_deterministic": "#2A9D8F",
    "gauss_hierarchical_dpm50_langevin2": "#3A86FF",
    "langevin_fnpe": "#8D99AE",
}


def build_variants(obs_noise: float) -> dict:
    """Per-obs_noise sample_kwargs/titles for the two gauss_hierarchical variants.

    (langevin_fnpe reuses compare_shared_local_composition_methods.VARIANTS'
    sample_kwargs unchanged -- only its title/filename is set here.)
    """
    tag = f"obs_noise={obs_noise:g}"
    return {
        "gauss_hierarchical_dpm50_deterministic": {
            "title": f"DPM2 (50 steps) + hierarchical GAUSS, deterministic ({tag})",
            "filename": "01_gauss_hierarchical_dpm50_deterministic.png",
            "sample_kwargs": {
                "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
                "corrector_steps_interval": 1, "corrector_steps": 0,
                "final_corrector_steps": 0, "snr": 0.2, "denoise_clamp": None,
            },
            "timesteps": 50,
        },
        "gauss_hierarchical_dpm50_langevin2": {
            "title": f"DPM2 (50 steps) + hierarchical GAUSS, 2 Langevin correctors/step ({tag})",
            "filename": "02_gauss_hierarchical_dpm50_langevin2.png",
            "sample_kwargs": {
                "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
                "corrector_steps_interval": 1, "corrector_steps": 2,
                "final_corrector_steps": 2, "snr": 0.2,
            },
            "timesteps": 50,
        },
        "langevin_fnpe": {
            "title": f"Annealed Langevin + F-NPSE, 100 steps ({tag})",
            "filename": "03_langevin_fnpe.png",
            "timesteps": 100,
        },
    }


def train_or_load_model(
    obs_noise: float, local_std: float, model_size: str, cfg: "ci.RunConfig",
    force_retrain: bool, quick: bool,
) -> "ci.SBIm":
    model_tag = f"shared_local_{model_size}_obs_noise_{obs_noise:g}"
    model_dir = ci.checkpoint_dir(cfg, model_tag)
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists() and not force_retrain:
        print(f"Loading existing checkpoint: {checkpoint}")
        return ci.SBIm.load(str(checkpoint), device=cfg.device)

    ci.seed_all(cfg.seed + 1_190)
    theta_train, x_train = ci.simulate_hierarchical_pairs(cfg.train_samples)
    theta_val, x_val = ci.simulate_hierarchical_pairs(cfg.validation_samples)
    model_kwargs = MODEL_SIZES[model_size] or ci.SHARED_LOCAL_HQ_MODEL_KWARGS
    model = ci.SBIm(nodes_size=3, device=cfg.device, **model_kwargs)
    param_count = sum(p.numel() for p in model.model.parameters())
    print(
        f"Training shared/local score model from scratch: obs_noise={obs_noise:g}, "
        f"local_std={local_std:g}, model_size={model_size} ({param_count:,} params) on "
        f"{cfg.train_samples:,} simulations ({'quick' if quick else 'full'} data/epoch settings)..."
    )
    started = time.perf_counter()
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_val, x_val=x_val,
        batch_size=cfg.batch_size, max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.patience, lr=ci.SHARED_LOCAL_HQ_LR, time_sampling="mixture",
        device=cfg.device, verbose=True, path=str(model_dir),
    )
    print(f"  trained in {(time.perf_counter() - started) / 60:.1f} minutes")
    return model


def run_one_obs_noise(
    obs_noise: float, args: argparse.Namespace, output_dir: Path,
) -> list[dict[str, object]]:
    tag = f"obs_noise_{obs_noise:g}"
    subdir = output_dir / tag
    subdir.mkdir(parents=True, exist_ok=True)

    original_s0l, original_sxh = ci.S0L, ci.SXH
    ci.S0L = args.local_std
    ci.SXH = obs_noise
    print(f"\n=== {tag}: S0L={ci.S0L}, SXH={ci.SXH}, model_size={args.model_size} ===")

    train_samples = 2_000 if args.quick else args.train_samples
    validation_samples = 500 if args.quick else args.validation_samples
    max_epochs = 3 if args.quick else args.max_epochs
    cfg = ci.RunConfig(
        output_dir=subdir, device="cuda", seed=args.seed,
        train_samples=train_samples, validation_samples=validation_samples,
        max_epochs=max_epochs, patience=args.patience, batch_size=args.batch_size,
        posterior_samples=args.posterior_samples,
    )

    model = train_or_load_model(
        obs_noise, args.local_std, args.model_size, cfg, args.force_retrain, args.quick,
    )

    reference_path = subdir / "reference.npz"
    if reference_path.exists() and not args.force_rerun:
        with np.load(reference_path) as archive:
            reference = {key: archive[key] for key in archive.files}
    else:
        ci.seed_all(cfg.seed + 1_200)
        n = args.n_observations
        global_true = float(ci.MUG + ci.S0G * torch.randn(()))
        local_true = args.local_std * torch.randn(n)
        x = global_true + local_true[:, None] + obs_noise * torch.randn(n, 1)
        exact_mean, exact_covariance = ci.exact_hierarchical_posterior(x)
        reference = {
            "x_observed": ci.tensor_numpy(x),
            "global_truth": np.asarray(global_true),
            "local_truth": ci.tensor_numpy(local_true),
            "exact_joint_mean": exact_mean,
            "exact_joint_covariance": exact_covariance,
        }
        ci.save_raw_data(reference_path, **reference)
        print(
            f"New reference: n={n}, global truth={global_true:.3f}, "
            f"analytic global std={exact_covariance[0, 0] ** 0.5:.4f}"
        )

    covariance_bank, _ = build_generalized_covariance_bank(
        reference["x_observed"], ci.S0G, args.local_std, obs_noise,
        COVARIANCE_SAMPLE_COUNTS, seed=34_711,
    )

    variants = build_variants(obs_noise)

    for method in ("gauss_hierarchical_dpm50_deterministic", "gauss_hierarchical_dpm50_langevin2"):
        run_gauss_hierarchical_variant(
            method, variants[method], model, reference, subdir, args.force_rerun,
            variants[method]["timesteps"], args.posterior_samples,
            args.precision_est_samples, args.precision_est_timesteps,
        )

    # langevin_fnpe doesn't use oracle moments, but run_oracle_method's
    # signature still takes a covariance_bank argument (unused for the
    # "fnpe" correction branch); reuse the one built above.
    run_oracle_method(
        "langevin_fnpe", model, reference, subdir, args.force_rerun,
        covariance_bank, variants["langevin_fnpe"]["timesteps"], args.posterior_samples,
    )
    # run_oracle_method plots using VARIANTS["langevin_fnpe"]'s generic title/
    # filename; replot (no recompute -- the .npz already has the samples)
    # with this sweep's own obs_noise-labeled title/filename for consistency
    # with the other two methods.
    result_path = subdir / "langevin_fnpe.npz"
    plot_shared_local(result_path, subdir / variants["langevin_fnpe"]["filename"], title=variants["langevin_fnpe"]["title"])

    rows: list[dict[str, object]] = []
    for method, stem in NPZ_STEM.items():
        result_path = subdir / f"{stem}.npz"
        raw = load_result(result_path)
        for row in metric_rows(method, raw, float(raw["runtime_seconds"])):
            row["obs_noise"] = obs_noise
            rows.append(row)

    with (subdir / "combined_method_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {subdir / 'combined_method_metrics.csv'}")

    ci.S0L, ci.SXH = original_s0l, original_sxh
    return rows


def plot_global_accuracy_vs_obs_noise(all_rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method in METHOD_ORDER:
        points = sorted(
            (row["obs_noise"], row["mean_error_in_exact_std"])
            for row in all_rows if row["method"] == method and row["parameter"] == "global"
        )
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method], lw=2, ms=6)
    ax.set_xscale("log")
    ax.set(
        xlabel="observation noise SXH",
        ylabel="|mean(global samples) - exact posterior mean| / analytic σ",
        title="Global parameter recovery accuracy vs. observation noise",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZES), default="full",
                         help="Score-network capacity for every point in the sweep.")
    parser.add_argument("--obs-noise-values", type=str, default=None,
                         help="Comma-separated SXH values. Default: 10 log-spaced values "
                              f"from {DEFAULT_OBS_NOISE_VALUES[0]:g} to {DEFAULT_OBS_NOISE_VALUES[-1]:g}.")
    parser.add_argument("--local-std", type=float, default=ci.S0L,
                         help="S0L (local parameter's own prior std), fixed across the sweep.")
    parser.add_argument("--n-observations", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=ci.SHARED_LOCAL_HQ_TRAIN_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=ci.SHARED_LOCAL_HQ_VALIDATION_SAMPLES)
    parser.add_argument("--max-epochs", type=int, default=ci.SHARED_LOCAL_HQ_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=ci.SHARED_LOCAL_HQ_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=ci.SHARED_LOCAL_HQ_BATCH_SIZE)
    parser.add_argument("--quick", action="store_true",
                         help="Few training samples/epochs, for a fast pipeline smoke test.")
    parser.add_argument("--posterior-samples", type=int, default=3_000)
    parser.add_argument("--precision-est-samples", type=int, default=4_096)
    parser.add_argument("--precision-est-timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if a checkpoint exists.")
    parser.add_argument("--force-rerun", action="store_true", help="Resample every method even if its .npz exists.")
    args = parser.parse_args()

    if args.obs_noise_values is None:
        obs_noise_values = list(DEFAULT_OBS_NOISE_VALUES)
    else:
        obs_noise_values = [float(v) for v in args.obs_noise_values.split(",") if v.strip()]

    output_dir = args.output_dir or (ROOT / f"obs_noise_sweep_{args.model_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for obs_noise in obs_noise_values:
        all_rows.extend(run_one_obs_noise(obs_noise, args, output_dir))

    with (output_dir / "sweep_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {output_dir / 'sweep_metrics.csv'}")

    plot_global_accuracy_vs_obs_noise(all_rows, output_dir / "05_obs_noise_vs_global_accuracy.png")

    metadata = {
        "model_size": args.model_size, "obs_noise_values": obs_noise_values,
        "local_std": args.local_std, "n_observations": args.n_observations,
        "train_samples": (2_000 if args.quick else args.train_samples),
        "validation_samples": (500 if args.quick else args.validation_samples),
        "max_epochs": (3 if args.quick else args.max_epochs),
        "quick": args.quick, "posterior_samples": args.posterior_samples,
        "precision_est_samples": args.precision_est_samples,
        "precision_est_timesteps": args.precision_est_timesteps, "seed": args.seed,
        "methods": METHOD_ORDER,
    }
    (output_dir / "sweep_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Sweep outputs written to {output_dir}")


if __name__ == "__main__":
    main()
