#!/usr/bin/env python3
"""Shared/local Gaussian hierarchy: mixture composition with K inferred.

Runs the three composition rules in ``MultiObsSampler`` and the Gaussian-mixture
product on the same checkpoint, observations and seed, and grades all four
against the **exact** joint posterior this model admits in closed form.

**K is never supplied and never swept**: ``mixture_inference.choose_and_fit``
picks it per observation by BIC at inference time.

This hierarchy is the honest control for the mixture method. Its per-observation
posteriors are unimodal Gaussians, so a single Gaussian summary is *sufficient*
and the mixture has nothing to gain -- the question here is only whether it loses
anything, and what it costs.
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "inverse_kinematics" / "experiments"))

from compare_shared_local_composition_methods import load_score_checkpoint  # noqa: E402
from mixture_inference import compose, choose_and_fit, sample_from  # noqa: E402
from mixture_product import density_on_grid  # noqa: E402

PRIOR_MEAN, PRIOR_STD = 0.0, 1.0
RULES = {
    "gauss_hierarchical": {"correction": "gauss_hierarchical", "method": "dpm",
                           "order": 2, "corrector_steps": 0,
                           "final_corrector_steps": 0},
    "gauss_jacobian": {"correction": "gauss_jacobian", "method": "dpm",
                       "order": 2, "corrector_steps": 0,
                       "final_corrector_steps": 0},
    "langevin_fnpse": {"correction": "fnpe", "method": "langevin",
                       "corrector_steps": 10, "snr": 0.1},
}


def w1_grid(grid, left, right):
    def cdf(density):
        c = np.concatenate([[0.0], np.cumsum(
            0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
        return c / c[-1]
    return float(np.trapezoid(np.abs(cdf(left) - cdf(right)), grid))


def kde(draws, grid):
    from scipy.stats import gaussian_kde
    density = gaussian_kde(np.asarray(draws, dtype=np.float64))(grid)
    return density / np.trapezoid(density, grid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--single-draws", type=int, default=5000)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--churn-eta", type=float, default=2.0)
    parser.add_argument("--pilot-draws", type=int, default=4096)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=HERE / "mixture_inference")
    arguments = parser.parse_args(argv)

    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly: shared host.")

    with np.load(arguments.reference) as stored:
        x = np.asarray(stored["x_observed"], dtype=np.float64)
        exact_mean = np.asarray(stored["exact_joint_mean"], dtype=np.float64)
        exact_cov = np.asarray(stored["exact_joint_covariance"], dtype=np.float64)
    n = len(x)
    shared_mean = float(exact_mean[0])
    shared_sd = float(np.sqrt(exact_cov[0, 0]))
    grid = np.linspace(shared_mean - 8 * shared_sd, shared_mean + 8 * shared_sd, 1601)
    reference = np.exp(-0.5 * ((grid - shared_mean) / shared_sd) ** 2)
    reference = reference / np.trapezoid(reference, grid)

    model = load_score_checkpoint(arguments.checkpoint)
    data = torch.as_tensor(x, dtype=torch.float32)
    results, curves = {}, {}

    for name, kwargs in RULES.items():
        extra = dict(kwargs)
        if extra["correction"] == "gauss_hierarchical":
            extra.update(precision_est_samples=arguments.pilot_draws,
                         precision_est_timesteps=arguments.timesteps)
        if extra["method"] == "dpm":
            extra["churn_eta"] = arguments.churn_eta
        torch.manual_seed(arguments.seed)
        torch.cuda.manual_seed_all(arguments.seed)
        started = time.perf_counter()
        samples = model.sample(
            x=data, multi_obs_inference=True, hierarchy=[0],
            prior=([PRIOR_MEAN], [PRIOR_STD]),
            num_samples=arguments.draws, timesteps=arguments.timesteps,
            device=arguments.device, verbose=False, **extra,
        ).detach().cpu().numpy()
        seconds = time.perf_counter() - started
        shared = samples[0, :, 0].astype(np.float64)
        curves[name] = kde(shared, grid)
        results[name] = {
            "seconds": seconds,
            "w1_over_sigma": w1_grid(grid, curves[name], reference) / shared_sd,
            "mean_error_sigma": (float(shared.mean()) - shared_mean) / shared_sd,
            "width_ratio": float(shared.std(ddof=1)) / shared_sd,
            "local_mean_error": float(np.mean(np.abs(
                samples[:, :, 1].mean(axis=1) - exact_mean[1:]))),
        }
        results[name]["_locals"] = samples[:, :, 1]
        print(f"  {name:<22}{seconds:7.1f}s  W1/sig "
              f"{results[name]['w1_over_sigma']:.3f}  width "
              f"{results[name]['width_ratio']:.3f}")

    # --- the mixture product, K inferred -------------------------------------
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    started = time.perf_counter()
    single = model.sample(
        x=data, num_samples=arguments.single_draws,
        timesteps=arguments.timesteps, method="dpm", order=2,
        churn_eta=arguments.churn_eta, corrector_steps=0,
        final_corrector_steps=0, device=arguments.device, verbose=False,
    ).detach().cpu().numpy()
    shared_draws = single[:, :, 0].astype(np.float64)
    fits, chosen = choose_and_fit(shared_draws, arguments.k_max, 1, arguments.seed)
    precision, precision_mean, log_constant = compose(
        fits, 1.0 / PRIOR_STD ** 2, arguments.beam
    )
    density = density_on_grid(grid, precision, precision_mean, log_constant)
    rng = np.random.default_rng(arguments.seed)
    shared = sample_from(precision, precision_mean, log_constant,
                         arguments.draws, rng)

    # Locals: p(l_j | g, x_j), the network's own conditional with g conditioned.
    condition_mask = torch.tensor([1.0, 0.0, 1.0])
    conditioned = torch.cat([
        torch.as_tensor(np.tile(shared, n), dtype=torch.float32)[:, None],
        data.repeat_interleave(arguments.draws, dim=0),
    ], dim=1)
    filled = []
    for start in range(0, conditioned.shape[0], 8192):
        block = conditioned[start:start + 8192]
        filled.append(model.sample(
            x=block, condition_mask=condition_mask, num_samples=1,
            timesteps=arguments.timesteps, method="dpm", order=2,
            churn_eta=arguments.churn_eta, corrector_steps=0,
            final_corrector_steps=0, device=arguments.device, verbose=False,
        ).detach().cpu().numpy()[:, 0, :])
    local_draws = np.concatenate(filled, axis=0).reshape(n, arguments.draws, -1)[:, :, 0]
    seconds = time.perf_counter() - started

    curves["mixture_product"] = density
    results["mixture_product"] = {
        "seconds": seconds,
        "w1_over_sigma": w1_grid(grid, density, reference) / shared_sd,
        "mean_error_sigma": (float(shared.mean()) - shared_mean) / shared_sd,
        "width_ratio": float(shared.std(ddof=1)) / shared_sd,
        "local_mean_error": float(np.mean(np.abs(
            local_draws.mean(axis=1) - exact_mean[1:]))),
        "chosen_k": chosen,
        "components": int(len(precision)),
        "cancellation_ratio": float(precision.min() / ((n - 1) / PRIOR_STD ** 2)),
        "_locals": local_draws,
    }
    print(f"  {'mixture_product':<22}{seconds:7.1f}s  W1/sig "
          f"{results['mixture_product']['w1_over_sigma']:.3f}  width "
          f"{results['mixture_product']['width_ratio']:.3f}")
    print(f"    K chosen: {chosen}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output / f"{arguments.label}.npz",
        grid=grid, reference=reference, exact_mean=exact_mean,
        exact_covariance=exact_cov, x_observed=x,
        **{f"curve_{k}": v for k, v in curves.items()},
        **{f"locals_{k}": v["_locals"] for k, v in results.items()},
    )
    (arguments.output / f"{arguments.label}.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         for k, v in results.items()}, indent=1, default=float))
    print(f"Wrote {arguments.output / (arguments.label + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
