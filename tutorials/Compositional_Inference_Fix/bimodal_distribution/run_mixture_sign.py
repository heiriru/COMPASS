#!/usr/bin/env python3
"""The sign model: a posterior that *is* a Gaussian mixture, composed as one.

This is the honest test of the mixture product. The exact joint posterior here is
a **2^n-component Gaussian mixture with shared covariance** (``sign_model.mixture``),
available in closed form -- so the reference is not an estimate, and the object the
method builds is the same kind of object as the truth.

Every other benchmark run so far had a unimodal per-observation posterior (the
shared/local Gaussian hierarchy) or a bimodal one with no closed form (inverse
kinematics). This one has both: genuinely multimodal, and exactly known.

**K is never supplied and never swept** -- ``mixture_inference.choose_and_fit``
picks it per observation by BIC at inference time. The interesting question is
whether it finds the 2 that the generative model actually puts there.
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
sys.path.insert(0, str(HERE.parent.parent.parent / "inverse_kinematics" / "experiments"))

import mixture_reference  # noqa: E402
import sign_model  # noqa: E402
from mixture_inference import choose_and_fit, compose, sample_from  # noqa: E402
from mixture_product import density_on_grid  # noqa: E402

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


def grid_reference(y, grid):
    """Exact ``p(g | y)`` on a grid, without enumerating the 2^n components.

    Integrating each local out gives a closed-form two-component emission

        p(y_j | g) = 1/2 N(y_j; g, s^2) + 1/2 N(y_j; -g, s^2),  s^2 = tau^2 + sigma_y^2

    so the shared posterior is the prior times a product of ``n`` such terms.
    ``sign_model.mixture`` enumerates 2^n components and is exact, but that is
    4.3 billion at n = 32; this agrees with it to 9e-15 at n = 4, 8 and 16 and
    costs ``len(grid) * n``.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    variance = sign_model.TAU ** 2 + sign_model.SIGMA_Y ** 2
    difference = y[None, :] - grid[:, None]
    total = y[None, :] + grid[:, None]
    log_emission = (
        -0.5 * np.log(2.0 * np.pi * variance)
        + np.logaddexp(-0.5 * difference ** 2 / variance,
                       -0.5 * total ** 2 / variance)
        - np.log(2.0)
    ).sum(axis=1)
    log_density = (-0.5 * ((grid - sign_model.MU_G) / sign_model.SIGMA_G) ** 2
                   + log_emission)
    density = np.exp(log_density - log_density.max())
    return density / np.trapezoid(density, grid)


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
    parser.add_argument("--checkpoint", type=Path,
                        default=HERE / "models" / "sign_h128d6_log_sigma"
                        / "Model_checkpoint.pt")
    parser.add_argument("--observations", type=int, default=8)
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--single-draws", type=int, default=5000)
    parser.add_argument("--pilot-draws", type=int, default=4096)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--churn-eta", type=float, default=2.0)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=HERE / "mixture_inference")
    arguments = parser.parse_args(argv)

    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly: shared host.")

    from compass import ScoreBasedInferenceModel as SBIm
    from compass.MultiObsSampler import MultiObsSampler

    model = SBIm.load(str(arguments.checkpoint), device=arguments.device)
    truth, local_truth, signs, y = sign_model.observations(
        arguments.observations, arguments.seed)
    n = len(y)
    # The grid is taken from the moments of the exact mixture where enumerating
    # it is affordable, and from a prior-scaled window otherwise; the reference
    # density itself always comes from the closed form above.
    if n <= 20:
        grid, _, _, reference_mean, reference_sd = (
            mixture_reference.shared_reference(sign_model, y))
        components = 2 ** n
    else:
        span = 4.0 * sign_model.SIGMA_G
        grid = np.linspace(-span, span, 40001)
        reference_mean = reference_sd = None
        components = 2 ** n
    reference = grid_reference(y, grid)
    weights_grid = reference / reference.sum()
    reference_mean = float((weights_grid * grid).sum())
    reference_sd = float(np.sqrt(max(
        (weights_grid * grid ** 2).sum() - reference_mean ** 2, 0.0)))
    print(f"n = {n}: exact posterior has 2^{n} = {components} components, "
          f"true g = {truth:+.3f}, reference sd {reference_sd:.3f}")

    data = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)
    prior = (torch.tensor([sign_model.MU_G]),
             torch.tensor([sign_model.SIGMA_G]))
    local_prior = (torch.tensor([sign_model.LOCAL_PRIOR_MEAN]),
                   torch.tensor([sign_model.LOCAL_PRIOR_STD]))
    results, curves = {}, {}

    for name, kwargs in RULES.items():
        extra = dict(kwargs)
        if extra["correction"] == "gauss_hierarchical":
            extra.update(precision_est_samples=arguments.pilot_draws,
                         precision_est_timesteps=arguments.timesteps)
        if extra["method"] == "dpm":
            extra["churn_eta"] = arguments.churn_eta
        sampler = MultiObsSampler(model)
        torch.manual_seed(arguments.seed)
        torch.cuda.manual_seed_all(arguments.seed)
        started = time.perf_counter()
        samples = sampler.sample(
            world_size=1, data=data, condition_mask=sign_model.CONDITION_MASK,
            timesteps=arguments.timesteps, num_samples=arguments.draws,
            hierarchy=list(sign_model.HIERARCHY), prior=prior,
            local_prior=local_prior, device=arguments.device, verbose=False,
            **extra,
        ).detach().cpu().numpy()
        seconds = time.perf_counter() - started
        shared = samples[0, :, sign_model.GLOBAL_INDEX].astype(np.float64)
        curves[name] = kde(shared, grid)
        results[name] = {
            "seconds": seconds,
            "w1_over_sigma": w1_grid(grid, curves[name], reference) / reference_sd,
            "mean_error_sigma": (float(shared.mean()) - reference_mean) / reference_sd,
            "width_ratio": float(shared.std(ddof=1)) / reference_sd,
            "_locals": samples[:, :, sign_model.LOCAL_INDEX],
        }
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
    shared_draws = single[:, :, sign_model.GLOBAL_INDEX].astype(np.float64)
    fits, chosen = choose_and_fit(shared_draws, arguments.k_max, 1, arguments.seed)
    precision, precision_mean, log_constant = compose(
        fits, 1.0 / sign_model.SIGMA_G ** 2, arguments.beam)
    density = density_on_grid(grid, precision, precision_mean, log_constant)
    rng = np.random.default_rng(arguments.seed)
    shared = sample_from(precision, precision_mean, log_constant,
                         arguments.draws, rng)

    condition_mask = torch.tensor([1.0, 0.0, 1.0])
    conditioned = torch.cat([
        torch.as_tensor(np.tile(shared, n), dtype=torch.float32)[:, None],
        data.repeat_interleave(arguments.draws, dim=0),
    ], dim=1)
    filled = []
    for start in range(0, conditioned.shape[0], 8192):
        filled.append(model.sample(
            x=conditioned[start:start + 8192], condition_mask=condition_mask,
            num_samples=1, timesteps=arguments.timesteps, method="dpm", order=2,
            churn_eta=arguments.churn_eta, corrector_steps=0,
            final_corrector_steps=0, device=arguments.device, verbose=False,
        ).detach().cpu().numpy()[:, 0, :])
    local_draws = np.concatenate(filled, axis=0).reshape(n, arguments.draws, -1)[:, :, 0]
    seconds = time.perf_counter() - started

    curves["mixture_product"] = density
    results["mixture_product"] = {
        "seconds": seconds,
        "w1_over_sigma": w1_grid(grid, density, reference) / reference_sd,
        "mean_error_sigma": (float(shared.mean()) - reference_mean) / reference_sd,
        "width_ratio": float(shared.std(ddof=1)) / reference_sd,
        "chosen_k": chosen, "components": int(len(precision)),
        "exact_components": int(components),
        "cancellation_ratio": float(precision.min()
                                    / ((n - 1) / sign_model.SIGMA_G ** 2)),
        "_locals": local_draws,
    }
    print(f"  {'mixture_product':<22}{seconds:7.1f}s  W1/sig "
          f"{results['mixture_product']['w1_over_sigma']:.3f}  width "
          f"{results['mixture_product']['width_ratio']:.3f}")
    print(f"    K chosen: {chosen}   (exact posterior: 2^{n} = {components} "
          f"components; composed mixture kept {len(precision)})")

    arguments.output.mkdir(parents=True, exist_ok=True)
    label = f"sign_n{n}"
    np.savez_compressed(
        arguments.output / f"{label}.npz", grid=grid, reference=reference,
        y=np.asarray(y), global_truth=truth, local_truth=np.asarray(local_truth),
        **{f"curve_{k}": v for k, v in curves.items()},
        **{f"locals_{k}": v["_locals"] for k, v in results.items()})
    (arguments.output / f"{label}.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         for k, v in results.items()}, indent=1, default=float))
    print(f"Wrote {arguments.output / (label + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
