#!/usr/bin/env python3
"""Evaluate the "gauss_hierarchical" correction on the shared/local Gaussian problem.

This reuses the exact checkpoint, reference truth, and 30 observations from
``compare_shared_local_composition_methods.py`` (the source of
``02a_dpm2_gaussian.png`` .. ``02g_dpm2_gauss_moment.png``), so results are
directly comparable to those figures. Unlike "Gauss_global_local" as invoked
in that script (which is handed an *analytic* single-observation covariance
bank built from the toy model's true generative parameters -- see
``build_single_observation_covariance_bank``), every covariance used here is
estimated from the trained score network's own DDIM draws
(``posterior_covariance=None`` triggers ``estimate_posterior_moments``/
``_estimate_posterior_covariance`` in ``MultiObsSampler``). No posterior_mean
or global_posterior_mean is ever supplied -- "gauss_hierarchical" refuses them
outright (see ``derivation.md``).

Two variants are run, matching the two existing reference settings:

- ``gauss_hierarchical_dense_correctors``: matches 02a/02f's dense-corrector
  DPM-2 sampler (corrector_steps_interval=1, corrector_steps=10,
  final_corrector_steps=3, snr=0.2).
- ``gauss_hierarchical_deterministic``: matches 02g's zero-corrector,
  deterministic DPM-2 sampler (no Langevin correctors, denoise_clamp=None),
  for a direct, apples-to-apples comparison against the moment-projection cheat.

For reference, 02g's own npz is regenerated here too (unmodified from
``compare_shared_local_composition_methods.py``'s "dpm2_gauss_moment" variant)
since its ``.npz`` was not left on disk, only its plot -- we want a real
number to compare against, not just an old picture.
"""
from __future__ import annotations

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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))

from compare_shared_local_composition_methods import (  # noqa: E402
    VARIANTS, load_result, load_score_checkpoint, metric_rows,
)
from plot_local_vs_global_validation import plot_shared_local  # noqa: E402

DEFAULT_REFERENCE = PARENT.parent / "output" / "compositional_inference" / "06b_shared_local" / "raw_plot_data.npz"
DEFAULT_CHECKPOINT = PARENT.parent / "output" / "compositional_inference" / "models" / "shared_local_mixture_full" / "Model_checkpoint.pt"
N_SAMPLES = 3_000
TIMESTEPS = 100
INFERENCE_SEED = 1_208
PRECISION_EST_SAMPLES = 4096
PRECISION_EST_TIMESTEPS = 100

NEW_VARIANTS = {
    "gauss_hierarchical_dense_correctors": {
        "title": "DPM2 + true compositional hierarchical GAUSS",
        "filename": "02h_dpm2_gauss_hierarchical.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "corrector_steps_interval": 1, "corrector_steps": 10,
            "final_corrector_steps": 3, "snr": 0.2,
        },
        "description": (
            "Second-order probability-flow predictor with dense Langevin "
            "correctors; the shared/local composition uses only the trained "
            "network's own scores, weighted by covariances estimated from the "
            "network's own DDIM draws (no oracle moments)."
        ),
    },
    "gauss_hierarchical_deterministic": {
        "title": "DPM2 + true compositional hierarchical GAUSS (deterministic)",
        "filename": "02i_dpm2_gauss_hierarchical_deterministic.png",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "corrector_steps_interval": 1, "corrector_steps": 0,
            "final_corrector_steps": 0, "snr": 0.2, "denoise_clamp": None,
        },
        "description": (
            "Deterministic DPM2, no Langevin correctors -- directly comparable "
            "to 02g_dpm2_gauss_moment.png's settings, but composing only real "
            "network scores instead of an oracle Gaussian moment projection."
        ),
    },
}


def run_no_oracle_method(
    method: str,
    variant: dict,
    model,
    reference: dict[str, np.ndarray],
    output: Path,
    force: bool,
    timesteps: int = TIMESTEPS,
    posterior_samples: int = N_SAMPLES,
) -> None:
    """Run one method with covariances estimated from the trained network."""
    result_path = output / f"{method}.npz"
    if result_path.exists() and not force:
        print(f"Reusing {result_path}")
        plot_shared_local(result_path, output / variant["filename"], title=variant["title"])
        return

    x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)
    sample_kwargs = dict(variant["sample_kwargs"])
    torch.manual_seed(INFERENCE_SEED)
    torch.cuda.manual_seed_all(INFERENCE_SEED)
    started = time.perf_counter()
    samples = model.sample(
        x=x, multi_obs_inference=True, hierarchy=[0], prior=([0.0], [1.0]),
        num_samples=posterior_samples, timesteps=timesteps,
        precision_est_samples=PRECISION_EST_SAMPLES,
        precision_est_timesteps=PRECISION_EST_TIMESTEPS,
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


def run_reference_method(
    method: str, model, reference: dict[str, np.ndarray], output: Path, force: bool,
    timesteps: int = TIMESTEPS, posterior_samples: int = N_SAMPLES,
) -> None:
    """Rerun one of the pre-existing (unmodified) VARIANTS, without the oracle bank.

    Only "dpm2_gauss_moment" needs regenerating here (its .npz was not left on
    disk); "dpm2_gaussian", "langevin_fnpe" and "dpm2_gauss_global_local" are
    copied in from the parent directory instead of rerun, to reuse the exact
    already-computed results the requested comparison is against.
    """
    from compare_shared_local_composition_methods import (
        COVARIANCE_BANK_SEED, COVARIANCE_SAMPLE_COUNTS,
        build_single_observation_covariance_bank, empirical_covariances,
    )
    variant = VARIANTS[method]
    result_path = output / f"{method}.npz"
    if result_path.exists() and not force:
        print(f"Reusing {result_path}")
        plot_shared_local(result_path, output / variant["filename"], title=variant["title"])
        return
    covariance_bank, _ = build_single_observation_covariance_bank(reference["x_observed"])
    x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)
    moment_count = max(COVARIANCE_SAMPLE_COUNTS)
    covariance = empirical_covariances(covariance_bank, moment_count)
    posterior_mean = np.asarray(covariance_bank[:, :moment_count], dtype=np.float64).mean(axis=1)
    sample_kwargs = dict(variant["sample_kwargs"])
    sample_kwargs["posterior_covariance"] = torch.from_numpy(covariance)
    sample_kwargs["global_posterior_mean"] = torch.from_numpy(posterior_mean[:, :1])
    sample_kwargs["global_posterior_covariance"] = torch.from_numpy(covariance[:, :1, :1])
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
    runtime = time.perf_counter() - started
    covariance_condition_number = max(map(np.linalg.cond, covariance))
    samples = samples.detach().cpu().numpy()
    synchronization_error = float(np.max(np.abs(samples[:, :, 0] - samples[0:1, :, 0])))
    np.savez_compressed(
        result_path,
        x_observed=reference["x_observed"],
        global_truth=reference["global_truth"],
        covariance_condition_number=covariance_condition_number,
        pd_repair_fraction=diagnostics["repair_fraction"],
        pd_maximum_relative_repair=diagnostics["maximum_relative_repair"],
        pd_minimum_eigenvalue_before=diagnostics["minimum_eigenvalue_before"] if diagnostics["minimum_eigenvalue_before"] is not None else np.nan,
        pd_minimum_eigenvalue_after=diagnostics["minimum_eigenvalue_after"] if diagnostics["minimum_eigenvalue_after"] is not None else np.nan,
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


def write_combined_metrics(output: Path, methods: list[str]) -> None:
    rows: list[dict[str, object]] = []
    for method in methods:
        result_path = output / f"{method}.npz"
        if result_path.exists():
            raw = load_result(result_path)
            rows.extend(metric_rows(method, raw, float(raw["runtime_seconds"])))
    if not rows:
        return
    with (output / "combined_method_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output / 'combined_method_metrics.csv'}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--posterior-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_result(args.reference)
    print(f"Loading shared score model from {args.checkpoint}")
    model = load_score_checkpoint(args.checkpoint)

    for method, variant in NEW_VARIANTS.items():
        run_no_oracle_method(
            method, variant, model, reference, args.output_dir, args.force,
            args.timesteps, args.posterior_samples,
        )
    run_reference_method(
        "dpm2_gauss_moment", model, reference, args.output_dir, args.force,
        args.timesteps, args.posterior_samples,
    )

    # Copy the three already-computed reference results in, unmodified, so the
    # combined metrics table and comparison plot cover every method requested.
    parent_results = {
        "dpm2_gaussian": PARENT / "dpm2_gaussian.npz",
        "langevin_fnpe": PARENT / "langevin_fnpe.npz",
        "dpm2_gauss_global_local": PARENT / "dpm2_gauss_global_local.npz",
    }
    for method, path in parent_results.items():
        target = args.output_dir / f"{method}.npz"
        if path.exists() and not target.exists():
            target.write_bytes(path.read_bytes())
            print(f"Copied {path} -> {target}")

    all_methods = [
        "dpm2_gaussian", "langevin_fnpe", "dpm2_gauss_global_local",
        "dpm2_gauss_moment", "gauss_hierarchical_dense_correctors",
        "gauss_hierarchical_deterministic",
    ]
    write_combined_metrics(args.output_dir, all_methods)

    metadata = {
        "checkpoint": str(args.checkpoint), "reference": str(args.reference),
        "n_observations": int(len(reference["x_observed"])),
        "posterior_samples": args.posterior_samples, "timesteps": args.timesteps,
        "inference_seed": INFERENCE_SEED,
        "precision_est_samples": PRECISION_EST_SAMPLES,
        "precision_est_timesteps": PRECISION_EST_TIMESTEPS,
        "new_variants": {k: v for k, v in NEW_VARIANTS.items()},
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"New_Attempt outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
