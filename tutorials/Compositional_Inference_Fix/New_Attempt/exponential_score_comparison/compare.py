#!/usr/bin/env python3
"""Three composition rules on the exponential hierarchy, graded against exact answers.

    g -> l_j -> x_j,   l_j = g + Exp(rate),   x_j = l_j + Exp(rate)

One trained score network, three ways of composing it across the 30 observations:

``gauss_hierarchical``   DPM-2 + ``correction="gauss_hierarchical"`` -- the paper's
                         GAUSS rule generalized to a global/local hierarchy, weighted
                         by a **pilot** covariance estimated from the network's own
                         draws (4096 per observation, a whole extra reverse pass).
``gauss_jacobian``       DPM-2 + ``correction="gauss_jacobian"`` -- the same arrow
                         elimination, but the per-observation backward covariance
                         comes from the network's Jacobian by Tweedie's second-order
                         identity. Pilot-free. Its MAP is additionally reported via
                         ``newton_map_estimate(curvature="jacobian")``, so that arm
                         is pilot-free end to end.
``langevin_fnpe``        Annealed Langevin + ``correction="fnpe"`` -- Geffner et al.'s
                         bridging-density composition, the algorithm it is derived for.

The same network, observations, seed, sample count and step count throughout, so
every difference is the composition rule.

What is measured, in three layers
---------------------------------
**1. The composed score itself** (``--stage score``). Each rule is a formula for
the score of the diffused tall posterior, and on this model that score is exactly
computable at every noise level by one-dimensional quadrature (see
``hierarchy.diffused_score``). So the rules are compared *before* any sampler
error enters: at states drawn from the exact diffused joint, at a ladder of
``lam`` spanning the schedule. This is the cleanest of the three layers -- no
sampler, no Monte Carlo error in the reference, no burn-in.

  A caveat that must be read with the F-NPSE row: ``fnpe`` composes the score of
  the *bridging* densities of Geffner et al., which are deliberately **not** the
  diffusion marginals of the tall posterior. They agree only as ``lam -> 0``. So
  a large mid-``lam`` error for F-NPSE is not by itself a defect -- it is the
  algorithm doing what it says. The rungs that grade all three on equal terms are
  the small-``lam`` ones, where every rule must agree with the same target.

**2. The sampled posterior** (``--stage sample``). W1, mean error and width ratio
of the shared marginal against the exact ``p(g | x)``, plus the same for the 30
local marginals. Also the *implied* marginal score ``d/dg log KDE(draws)``
against the closed form

    d/dg log p(g | x) = -(g - mu_g)/sigma_g^2 + N rate - sum_j 1/(x_j - g)

which is the quantity in the derivation, read back out of the draws.

**3. The MAP** (``--stage sample``). One procedure for all three rules, so it
compares them rather than the estimators: marginalize the draws onto ``g``, take
a Gaussian-KDE maximum (parabolic-refined) for ``g_hat``, clamp ``g = g_hat``,
then run annealed score ascent on each local under ``p(l_j | g_hat, x_j)``. With
``g`` conditioned the locals are independent, so stage 3 has no composition in it
at all and every compositional error collapses onto the single scalar ``g_hat``.
For the Jacobian arm, ``newton_map_estimate`` is reported alongside as its native
pilot-free alternative.

Usage:
    python compare.py                          # everything
    python compare.py --stage score            # composed-score layer only
    python compare.py --methods gauss_jacobian
    python compare.py --replot                 # redraw from saved .npz
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compass.MultiObsSampler import MultiObsSampler  # noqa: E402

import hierarchy  # noqa: E402
import train as recipe  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

NAVY, BLUE, TEAL, CORAL, GOLD = "#17223B", "#3A86FF", "#2A9D8F", "#EF476F", "#FFB703"

OBSERVATIONS = 30
NUM_SAMPLES = 3_000
TIMESTEPS = 100
PRECISION_EST_SAMPLES = 4096
SCORE_LAMBDAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
SCORE_STATES = 512

CORRECTOR = {"corrector_steps_interval": 1, "corrector_steps": 10,
             "final_corrector_steps": 3, "snr": 0.2}

METHODS = {
    "gauss_hierarchical": {
        "label": "DPM2 + gauss_hierarchical",
        "colour": CORAL,
        "note": "pilot covariance, 4096 draws/observation",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "precision_est_samples": PRECISION_EST_SAMPLES,
            "precision_est_timesteps": TIMESTEPS, **CORRECTOR,
        },
    },
    "gauss_hierarchical_correctors2": {
        "label": "DPM2 + gauss_hierarchical (2 correctors)",
        "colour": CORAL,
        "note": "4096-draw pilot covariance; two correctors per level, not ten",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "precision_est_samples": PRECISION_EST_SAMPLES,
            "precision_est_timesteps": TIMESTEPS,
            "corrector_steps_interval": 1, "corrector_steps": 2,
            "final_corrector_steps": 3, "snr": 0.2,
        },
    },
    "gauss_hierarchical_predictor_only": {
        "label": "DPM2 + gauss_hierarchical (predictor only)",
        "colour": BLUE,
        "note": "same 4096-draw pilot covariance; no compositional correctors",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "precision_est_samples": PRECISION_EST_SAMPLES,
            "precision_est_timesteps": TIMESTEPS,
            "corrector_steps": 0, "final_corrector_steps": 0,
            "terminal_corrector_steps": 0,
        },
    },
    "gauss_jacobian": {
        "label": "DPM2 + gauss_jacobian (Newton MAP)",
        "colour": TEAL,
        "note": "pilot-free; curvature from the network Jacobian",
        "newton_map": True,
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_jacobian",
            **CORRECTOR,
        },
    },
    # Two correctors per level rather than ten. The corrector sweep in
    # ../../global_local_corrector_tradeoff shows a Gaussian rule reaching its
    # accuracy at one or two correctors on a problem that suits it, at a third
    # of Langevin+F-NPSE's cost; this arm asks the same cost question here.
    "gauss_jacobian_correctors2": {
        "label": "DPM2 + gauss_jacobian (2 correctors)",
        "colour": TEAL,
        "note": "pilot-free; two Langevin correctors per level instead of ten",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_jacobian",
            "corrector_steps_interval": 1, "corrector_steps": 2,
            "final_corrector_steps": 3, "snr": 0.2,
        },
    },
    "gauss_jacobian_predictor_only": {
        "label": "DPM2 + gauss_jacobian (predictor only)",
        "colour": BLUE,
        "note": "pilot-free; no interleaved or terminal Langevin correctors",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_jacobian",
            "corrector_steps": 0, "final_corrector_steps": 0,
            "terminal_corrector_steps": 0,
        },
    },
    "langevin_fnpe": {
        "label": "Langevin + F-NPSE",
        "colour": GOLD,
        "note": "bridging densities; equals the diffusion target only as lam -> 0",
        "sample_kwargs": {
            "method": "langevin", "correction": "fnpe",
            "corrector_steps": 10, "snr": 0.2,
        },
    },
}


# ---------------------------------------------------------------------------
# The problem: observations, exact references, the trained network
# ---------------------------------------------------------------------------

def build_problem(model_dir, config, train_samples, observations, seed, device):
    model, _ = recipe.train_or_load(
        model_dir, config, train_samples, device, verbose=False,
    )
    global_truth, local_truth, x = hierarchy.observations(observations, seed)
    grid, density, weights, mean, std = hierarchy.shared_reference(x)
    local_mean, local_std = hierarchy.local_reference(x, grid, weights)
    local_grid = np.linspace(grid[0], float(np.max(x)), 20001)
    local_weights = np.stack([
        hierarchy.local_marginal(value, grid, weights, local_grid) for value in x
    ])
    return {
        "model": model, "device": device, "observations": observations,
        "x": np.asarray(x, dtype=np.float64),
        "data": torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1),
        "global_truth": global_truth,
        "local_truth": np.asarray(local_truth, dtype=np.float64),
        "global_grid": grid, "global_density": density, "global_weights": weights,
        "global_mean": mean, "global_std": std,
        "global_mode": hierarchy.grid_mode(grid, density),
        "local_mean": local_mean, "local_std": local_std,
        "local_grid": local_grid, "local_weights": local_weights,
        "config": config, "train_samples": train_samples,
    }


# ---------------------------------------------------------------------------
# Layer 1: the composed score against the exact diffused score
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_composed_score(sampler, problem, lambdas, states, seed):
    """Relative RMS error of each rule's composed score, per noise level.

    ``sampler`` must be one that has already run :meth:`sample`, so it carries
    the rule's own configuration -- including, for ``gauss_hierarchical``, the
    pilot covariance it actually used. Probing that object rather than a fresh
    one guarantees the field measured here is the field that steered the draws.

    States come from :func:`hierarchy.sample_diffused`, i.e. exactly from
    ``p_lam(g_t, l_t | x)``, so the error is weighted by where the sampler
    really is at that noise level rather than over an arbitrary box.
    """
    device = problem["device"]
    x = problem["x"]
    n = problem["observations"]
    grid = hierarchy.quadrature_grid(x, device=device)
    mask = hierarchy.CONDITION_MASK.to(device).reshape(1, 1, -1).repeat(n, states, 1)
    indices = torch.arange(n, device=device)

    rows = []
    for lam in lambdas:
        shared, local, _, _ = hierarchy.sample_diffused(x, lam, states, seed)
        exact_shared, exact_local = hierarchy.diffused_score(
            shared, local, x, lam, grid
        )

        state = torch.zeros(n, states, hierarchy.NODES, dtype=torch.float32,
                            device=device)
        state[:, :, hierarchy.GLOBAL_INDEX] = torch.as_tensor(
            shared, dtype=torch.float32, device=device
        )
        state[:, :, hierarchy.LOCAL_INDEX] = torch.as_tensor(
            local.T.copy(), dtype=torch.float32, device=device
        )
        state[:, :, hierarchy.OBSERVED_INDEX] = torch.as_tensor(
            x, dtype=torch.float32, device=device
        ).reshape(-1, 1)

        t = sampler.sde.time_of_lambda(
            torch.tensor(float(lam))
        ).reshape(1, 1).to(device)
        # Curvature caches are keyed by refresh counter / time; clear them so a
        # probe never reuses the state left behind by the sampling run.
        sampler._reset_jacobian_state()
        sampler._covariance_time_cache = {}
        composed = sampler._get_score(state, t, mask, indices).to(torch.float64)

        composed_shared = composed[0, :, hierarchy.GLOBAL_INDEX].cpu()
        composed_local = composed[:, :, hierarchy.LOCAL_INDEX].cpu().T
        exact_shared = exact_shared.cpu()
        exact_local = exact_local.cpu()

        def relative(estimate, exact):
            return float(
                (estimate - exact).pow(2).mean().sqrt()
                / exact.pow(2).mean().sqrt().clamp_min(1e-30)
            )

        rows.append({
            "lambda": float(lam),
            "shared_rel_rmse": relative(composed_shared, exact_shared),
            "local_rel_rmse": relative(composed_local, exact_local),
            "shared_bias": float((composed_shared - exact_shared).mean()),
            "shared_exact_rms": float(exact_shared.pow(2).mean().sqrt()),
            "composed_shared": composed_shared.numpy(),
            "exact_shared": exact_shared.numpy(),
            "state_shared": shared,
        })
    return rows


# ---------------------------------------------------------------------------
# Layer 2: sampling, and the marginal score implied by the draws
# ---------------------------------------------------------------------------

def draw(model, problem, sample_kwargs, num_samples, timesteps, seed,
         denoise_clamp, verbose=False):
    """Run one composition rule and return its joint draws plus the sampler."""
    sampler = MultiObsSampler(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    samples = sampler.sample(
        world_size=1, data=problem["data"],
        condition_mask=hierarchy.CONDITION_MASK, timesteps=timesteps,
        num_samples=num_samples, hierarchy=list(hierarchy.HIERARCHY),
        prior=(torch.tensor([hierarchy.MU_G]), torch.tensor([hierarchy.SIGMA_G])),
        local_prior=(torch.tensor([hierarchy.LOCAL_PRIOR_MEAN]),
                     torch.tensor([hierarchy.LOCAL_PRIOR_STD])),
        denoise_clamp=denoise_clamp, device=problem["device"], verbose=verbose,
        **sample_kwargs,
    )
    runtime = time.perf_counter() - started
    samples = samples.detach().cpu()
    synchronization = float(
        (samples[:, :, hierarchy.HIERARCHY]
         - samples[:1, :, hierarchy.HIERARCHY]).abs().max()
    )
    if synchronization > 1e-5:
        raise AssertionError(
            f"shared draws are not synchronized across observations: "
            f"{synchronization:.3e}"
        )
    return samples, sampler, runtime


def kde_mode(draws, bandwidth=None, grid_size=4001, pad=0.15):
    """Peak of a Gaussian KDE over 1-D draws, refined by a parabolic fit."""
    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    kde = gaussian_kde(draws, bw_method=bandwidth)
    span = draws.max() - draws.min()
    grid = np.linspace(draws.min() - pad * span, draws.max() + pad * span, grid_size)
    density = kde(grid)
    peak = int(np.argmax(density))
    mode = float(grid[peak])
    if 0 < peak < len(grid) - 1:
        left, centre, right = density[peak - 1], density[peak], density[peak + 1]
        curvature = left - 2 * centre + right
        if curvature < 0:
            mode = float(grid[peak] + 0.5 * (left - right) / curvature
                         * (grid[1] - grid[0]))
    return mode, grid, density, float(kde.factor * draws.std(ddof=1))


def kde_score(draws, grid, bandwidth=None):
    """``d/dg log KDE(g)``: the marginal shared score implied by the draws.

    Analytic in the kernel rather than a finite difference of the density, so
    the estimate stays clean where the density is small -- which on this problem
    is exactly where the interesting behaviour (the wall) lives.
    """
    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    kde = gaussian_kde(draws, bw_method=bandwidth)
    width = float(kde.factor * draws.std(ddof=1))
    delta = (grid[:, None] - draws[None, :]) / width
    log_kernel = -0.5 * delta**2
    weight = np.exp(log_kernel - log_kernel.max(axis=1, keepdims=True))
    weight /= weight.sum(axis=1, keepdims=True)
    return -(weight * delta).sum(axis=1) / width


def wasserstein(samples, grid, weights):
    return recipe.wasserstein_1d(samples, grid, weights)


# ---------------------------------------------------------------------------
# Layer 3: the MAP -- one procedure for every rule
# ---------------------------------------------------------------------------

@torch.no_grad()
def conditional_local_ascent(model, rows, condition_mask, timesteps=200,
                             iterations=3, eps=1e-3, sigma_start=1.0,
                             device="cpu"):
    """Annealed Tweedie ascent on the local column with ``g`` clamped.

    ``z <- z + lam^2 s(z, lam)`` converges to a mode of the ``lam``-smoothed
    density; annealing ``lam`` down tracks it to the unsmoothed MAP. With ``g``
    conditioned, ``p(l_j | g_hat, x_j)`` involves observation ``j`` alone, so
    every row is an independent single-observation problem and no composition,
    correction or shared precision enters this stage at all.
    """
    sde = model.sde
    z = torch.as_tensor(rows, dtype=torch.float32).clone().to(device)
    mask = torch.as_tensor(condition_mask, dtype=torch.float32).to(device)
    latent = 1.0 - mask

    one = torch.ones(1, device=device)
    lam_min = sde.lambda_t(eps * one).item()
    lam_max = sde.lambda_t(one).item()
    lam_hi = min(max(float(sigma_start), 2 * lam_min), lam_max)
    lams = torch.logspace(
        math.log10(lam_hi), math.log10(lam_min), int(timesteps), device=device
    )
    times = sde.time_of_lambda(lams)

    for index in range(int(timesteps)):
        t = times[index].reshape(1, 1)
        alpha = sde.alpha_t(t).to(device)
        lam_squared = float(lams[index].item()) ** 2
        for _ in range(int(iterations)):
            state = z * (alpha * latent + mask)
            score = model.output_scale_function(t, model.model(x=state, t=t, c=mask))
            z = z + lam_squared * (alpha * score) * latent
    alpha_final = sde.alpha_t(times[-1]).to(device)
    return (z * (alpha_final * latent + mask)).cpu()


def kde_map_pipeline(model, problem, samples, arguments):
    """``g_hat`` by KDE over the shared marginal, then locals by ascent at ``g_hat``."""
    device = problem["device"]
    n = problem["observations"]
    shared = samples[0, :, hierarchy.GLOBAL_INDEX].to(torch.float64).numpy()
    local = samples[:, :, hierarchy.LOCAL_INDEX].to(torch.float64).numpy()

    # One solver excursion would set the KDE bandwidth (Scott's rule scales with
    # the sample std) wider than the whole posterior and flatten the peak.
    limit = arguments.excursion_sigma * problem["global_std"]
    keep = np.isfinite(shared) & (
        np.abs(shared - problem["global_mean"]) <= limit
    )
    excluded = int((~keep).sum())
    kept_shared, kept_local = shared[keep], local[:, keep]

    global_map, kde_grid, kde_density, bandwidth = kde_mode(
        kept_shared, arguments.kde_bandwidth
    )

    rows = torch.zeros(n, hierarchy.NODES)
    rows[:, hierarchy.OBSERVED_INDEX] = torch.as_tensor(
        problem["x"], dtype=torch.float32
    )
    rows[:, hierarchy.GLOBAL_INDEX] = float(global_map)
    rows[:, hierarchy.LOCAL_INDEX] = torch.as_tensor(
        kept_local.mean(axis=1), dtype=torch.float32
    )
    mask = hierarchy.CONDITION_MASK.unsqueeze(0).repeat(n, 1).clone()
    mask[:, hierarchy.GLOBAL_INDEX] = 1.0            # the shared coordinate is fixed

    started = time.perf_counter()
    ascended = conditional_local_ascent(
        model, rows, mask, timesteps=arguments.map_timesteps,
        iterations=arguments.map_iterations, eps=arguments.map_eps,
        sigma_start=max(float(kept_local.std(axis=1).max()), 1e-3), device=device,
    )
    ascent_seconds = time.perf_counter() - started
    return {
        "shared_draws": shared, "local_draws": local,
        "kept_shared": kept_shared, "kept_local": kept_local,
        "excluded": excluded, "kde_grid": kde_grid, "kde_density": kde_density,
        "kde_bandwidth": bandwidth, "global_map": global_map,
        "local_map": ascended[:, hierarchy.LOCAL_INDEX].to(torch.float64).numpy(),
        "ascent_seconds": ascent_seconds,
    }


def newton_map(model, problem, samples, sample_kwargs, arguments):
    """``newton_map_estimate`` with Jacobian curvature -- the pilot-free MAP.

    Reported only for the ``gauss_jacobian`` arm, where both the composition
    weights and the Newton curvature come from the same network Jacobian, so
    nothing has to be carried over from a pilot run.
    """
    sampler = MultiObsSampler(model)
    n = problem["observations"]
    initial = torch.zeros(n, 1, hierarchy.NODES)
    initial[:, 0, hierarchy.OBSERVED_INDEX] = torch.as_tensor(
        problem["x"], dtype=torch.float32
    )
    initial[:, 0, hierarchy.GLOBAL_INDEX] = float(
        samples[0, :, hierarchy.GLOBAL_INDEX].mean()
    )
    initial[:, 0, hierarchy.LOCAL_INDEX] = samples[
        :, :, hierarchy.LOCAL_INDEX
    ].mean(dim=1)

    data = torch.zeros(n, hierarchy.NODES)
    data[:, hierarchy.OBSERVED_INDEX] = torch.as_tensor(
        problem["x"], dtype=torch.float32
    )
    started = time.perf_counter()
    estimate = sampler.newton_map_estimate(
        data=data, condition_mask=hierarchy.CONDITION_MASK,
        init=initial, hierarchy=list(hierarchy.HIERARCHY),
        prior=(torch.tensor([hierarchy.MU_G]), torch.tensor([hierarchy.SIGMA_G])),
        local_prior=(torch.tensor([hierarchy.LOCAL_PRIOR_MEAN]),
                     torch.tensor([hierarchy.LOCAL_PRIOR_STD])),
        correction=sample_kwargs["correction"], curvature="jacobian",
        timesteps=arguments.map_timesteps, device=problem["device"],
    ).detach().cpu()
    seconds = time.perf_counter() - started
    return {
        "newton_global_map": float(estimate[0, 0, hierarchy.GLOBAL_INDEX]),
        "newton_local_map": estimate[:, 0, hierarchy.LOCAL_INDEX]
        .to(torch.float64).numpy(),
        "newton_seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(problem, result, sampler, runtime, score_rows):
    x, n = problem["x"], problem["observations"]
    std, mean = problem["global_std"], problem["global_mean"]
    kept_shared, kept_local = result["kept_shared"], result["kept_local"]
    global_map = result["global_map"]

    local_w1 = [
        wasserstein(kept_local[j], problem["local_grid"], problem["local_weights"][j])
        / problem["local_std"][j] for j in range(n)
    ]
    conditional_mode, conditional_std = hierarchy.conditional_local(x, global_map)

    # The KDE-implied marginal score, compared where the exact score is finite
    # and the draws actually support an estimate: the central 90% of the exact
    # posterior. At the wall the exact score diverges and no kernel estimate can
    # follow it, so scoring there would grade the KDE, not the sampler.
    grid, weights = problem["global_grid"], problem["global_weights"]
    cumulative = np.cumsum(weights)
    low = float(np.interp(0.05, cumulative, grid))
    high = float(np.interp(0.95, cumulative, grid))
    score_grid = np.linspace(low, high, 401)
    implied = kde_score(kept_shared, score_grid)
    exact = hierarchy.exact_global_score(score_grid, x)
    marginal_score_rel = float(
        np.sqrt(np.mean((implied - exact) ** 2) / np.mean(exact**2))
    )

    metrics = {
        "observations": n, "num_samples": kept_shared.size,
        "excluded_draws": result["excluded"],
        "shared_w1_over_sigma": wasserstein(kept_shared, grid, weights) / std,
        "shared_mean_error_sigma": abs(float(kept_shared.mean()) - mean) / std,
        "shared_width_ratio": float(kept_shared.std(ddof=1)) / std,
        "local_w1_over_sigma": float(np.mean(local_w1)),
        "local_draw_mean_error_sigma": float(np.mean(
            np.abs(kept_local.mean(axis=1) - problem["local_mean"])
            / problem["local_std"]
        )),
        "marginal_score_rel_rmse": marginal_score_rel,
        "global_map": global_map,
        "global_map_error_sigma": abs(global_map - problem["global_mode"]) / std,
        "global_map_error_from_truth_sigma":
            abs(global_map - problem["global_truth"]) / std,
        "local_map_error_vs_posterior_mean_sigma": float(np.mean(
            np.abs(result["local_map"] - problem["local_mean"])
            / problem["local_std"]
        )),
        "local_map_error_vs_conditional_mode_sigma": float(np.mean(
            np.abs(result["local_map"] - conditional_mode) / conditional_std
        )),
        "sample_seconds": runtime,
        "ascent_seconds": result["ascent_seconds"],
        "sampler_network_calls": int(sampler.score_network_calls),
    }
    for row in score_rows:
        metrics[f"score_rel_rmse_lam{row['lambda']:g}"] = row["shared_rel_rmse"]
        metrics[f"local_score_rel_rmse_lam{row['lambda']:g}"] = row["local_rel_rmse"]
    metrics["score_rel_rmse_mean"] = float(np.mean(
        [row["shared_rel_rmse"] for row in score_rows]
    )) if score_rows else float("nan")
    if "newton_global_map" in result:
        newton_mode, newton_std = hierarchy.conditional_local(
            x, result["newton_global_map"]
        )
        metrics["newton_global_map"] = result["newton_global_map"]
        metrics["newton_global_map_error_sigma"] = abs(
            result["newton_global_map"] - problem["global_mode"]
        ) / std
        metrics["newton_local_map_error_vs_conditional_mode_sigma"] = float(np.mean(
            np.abs(result["newton_local_map"] - newton_mode) / newton_std
        ))
        metrics["newton_seconds"] = result["newton_seconds"]
    metrics.update({
        "conditional_mode": conditional_mode, "conditional_std": conditional_std,
        "score_grid": score_grid, "implied_score": implied, "exact_score": exact,
    })
    return metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def configure_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 240, "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def plot_method(problem, name, result, metrics, output):
    """The four-panel layout of ``03_new_method_nongaussian.png``, per method."""
    x = problem["x"]
    n = problem["observations"]
    indices = np.arange(n)
    local_draw_mean = result["kept_local"].mean(axis=1)
    local_draw_std = result["kept_local"].std(axis=1, ddof=1)

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.5))
    # Which single-observation score fed the composition. Every checkpointed
    # method uses the network; analytic_compare.py registers methods that swap
    # it for the closed-form N = 1 score, and must say so on the figure.
    score_note = METHODS[name].get("score_note", "learned score")
    fig.suptitle(
        f"{METHODS[name]['label']} -> KDE global MAP -> conditional local ascent"
        f"\n(non-Gaussian exponential hierarchy g → ℓⱼ → xⱼ, {score_note})",
        fontsize=14, fontweight="bold", color=NAVY,
    )

    # Panel 1 -- the observations, stacked as a dot plot in observational space.
    bins = min(15, n)
    _, edges = np.histogram(x, bins=bins)
    bin_index = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    height = np.zeros(bins, dtype=int)
    position = np.empty(n, dtype=int)
    for i in np.argsort(x):
        position[i] = height[bin_index[i]]
        height[bin_index[i]] += 1
    centres = (edges[:-1] + edges[1:]) / 2
    axes[0].scatter(centres[bin_index], position + 0.5, s=80, color=BLUE,
                    alpha=0.85, edgecolor="white", linewidth=0.6)
    axes[0].axvline(x.mean(), color="red", linestyle=":", linewidth=2,
                    label="mean observation")
    axes[0].axvline(x.min(), color=NAVY, linestyle="-.", linewidth=1.6,
                    label="wall at min$_j$ x$_j$")
    axes[0].set(xlabel="observed x", ylabel="observations stacked per bin",
                title="Observations (observational space)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8.5)

    # Panel 2 -- the shared posterior, its exact reference, and the two modes.
    axes[1].hist(result["kept_shared"], bins=46, density=True, color=BLUE,
                 alpha=0.72, label="COMPASS draws")
    axes[1].plot(problem["global_grid"], problem["global_density"], color=NAVY,
                 ls="--", lw=2, label="exact")
    axes[1].plot(result["kde_grid"], result["kde_density"], color=GOLD, lw=1.8,
                 label="KDE of draws")
    axes[1].axvline(problem["global_truth"], color="red", linestyle=":",
                    linewidth=2, label="true global parameter")
    axes[1].axvline(result["global_map"], color=TEAL, linestyle="--", linewidth=2,
                    label="KDE MAP (this method)")
    if "newton_global_map" in result:
        axes[1].axvline(float(result["newton_global_map"]), color=CORAL,
                        linestyle="-.", linewidth=2, label="newton_map_estimate")
    axes[1].set_xlim(problem["global_mean"] - 5 * problem["global_std"],
                     problem["global_mean"] + 5 * problem["global_std"])
    axes[1].set(xlabel="global parameter g", ylabel="density",
                title="Shared posterior")
    axes[1].legend(fontsize=8.5)
    note = f"KDE MAP off exact mode by {metrics['global_map_error_sigma']:.2f} σ"
    if result["excluded"]:
        note += f"; {result['excluded']} solver excursion(s) excluded"
    axes[1].text(0.5, -0.26, note, transform=axes[1].transAxes, ha="center",
                 va="top", color=NAVY, fontsize=9)

    # Panel 3 -- the locals. The joint draws give a spread; the pipeline's own
    # answer is a point estimate conditioned on g_hat, so it is drawn as such.
    axes[2].errorbar(indices, local_draw_mean, yerr=local_draw_std, fmt="o", ms=3,
                     color=CORAL, ecolor="#F4A3B4", alpha=0.75,
                     label="joint draws, mean ± σ")
    axes[2].plot(indices, result["local_map"], "d", ms=5, color=TEAL,
                 label="ascent MAP at g = ĝ")
    axes[2].plot(indices, problem["local_mean"], "_", ms=9, color=NAVY,
                 label="exact mean")
    if "newton_local_map" in result:
        axes[2].plot(indices, result["newton_local_map"], "x", ms=5, color=CORAL,
                     label="newton_map_estimate")
    axes[2].set(xlabel="observation", ylabel="local parameter ℓᵢ",
                title=f"{n} local posteriors")
    axes[2].legend(fontsize=8.5)

    # Panel 4 -- recovery of the locals against their exact posterior means.
    axes[3].scatter(problem["local_mean"], result["local_map"], c=indices,
                    cmap="viridis", s=38, edgecolor="white", linewidth=0.5)
    lo = min(problem["local_mean"].min(), result["local_map"].min())
    hi = max(problem["local_mean"].max(), result["local_map"].max())
    axes[3].plot([lo, hi], [lo, hi], color=NAVY, ls="--", lw=1.6)
    axes[3].text(
        0.04, 0.94,
        "mean |error| = "
        f"{metrics['local_map_error_vs_posterior_mean_sigma']:.2f} analytic σ\n"
        "vs exact p(ℓ|ĝ,x) mode: "
        f"{metrics['local_map_error_vs_conditional_mode_sigma']:.2f} σ",
        transform=axes[3].transAxes, va="top", color=NAVY, fontsize=9,
    )
    axes[3].set(xlabel="exact local posterior mean",
                ylabel="ascent MAP at g = ĝ", title="Local recovery")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def plot_comparison(problem, results, metrics, score_rows, output, arguments):
    """The head-to-head: composed score, sampled posterior, marginal score, MAP."""
    names = list(results)
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.5))
    fig.suptitle(
        "Three composition rules on g → ℓⱼ → xⱼ, one network"
        f"  ({problem['config']}, {problem['train_samples']:,} simulations; "
        f"n = {problem['observations']}, {arguments.num_samples:,} draws, "
        f"{arguments.timesteps} steps)",
        fontsize=14, fontweight="bold", color=NAVY,
    )

    # Panel 1 -- the composition rules graded before any sampler runs. The grey
    # line is the same network's *single-observation* score error, as a
    # reference for how much of a composed error is the network underneath.
    # It is deliberately NOT called a floor: composing n observations averages
    # their independent network errors down by ~sqrt(n) while leaving coherent
    # (bias) error untouched, so a rule may legitimately land below the line --
    # gauss_jacobian does at small lambda. Below the line means the network's
    # error was largely independent across observations and the rule averaged
    # it away; far above it means the rule is adding error of its own.
    if problem.get("network_reference"):
        reference = problem["network_reference"]
        axes[0].plot(sorted(reference),
                     [reference[lam] for lam in sorted(reference)],
                     "--", color="#8A8F98", lw=2,
                     label="single-observation score error (network)")
    for name in names:
        rows = score_rows.get(name)
        if not rows:
            continue
        axes[0].plot([row["lambda"] for row in rows],
                     [row["shared_rel_rmse"] for row in rows],
                     "o-", color=METHODS[name]["colour"], lw=2, ms=5,
                     label=METHODS[name]["label"])
    axes[0].set(xscale="log", yscale="log", xlabel="noise level λ",
                ylabel="relative RMS error of the composed shared score",
                title="Composed score vs exact")
    axes[0].legend(fontsize=8)
    axes[0].text(
        0.5, -0.26,
        "exact target from 1-D quadrature of p$_λ$(g$_t$, ℓ$_t$ | x);\n"
        "F-NPSE targets bridging densities, so read its small-λ rungs",
        transform=axes[0].transAxes, ha="center", va="top", color=NAVY, fontsize=8,
    )

    # Panel 2 -- what the draws actually look like against the exact posterior.
    axes[1].plot(problem["global_grid"], problem["global_density"], color=NAVY,
                 ls="--", lw=2, label="exact", zorder=5)
    for name in names:
        axes[1].hist(results[name]["kept_shared"], bins=60, density=True,
                     histtype="step", lw=1.8, color=METHODS[name]["colour"],
                     label=METHODS[name]["label"])
    axes[1].axvline(problem["global_truth"], color="red", linestyle=":", lw=2,
                    label="true g")
    axes[1].set_xlim(problem["global_mean"] - 5 * problem["global_std"],
                     problem["global_mean"] + 5 * problem["global_std"])
    axes[1].set(xlabel="global parameter g", ylabel="density",
                title="Shared posterior")
    # Below the axes: the posterior is pressed against the wall at min_j x_j, so
    # the mass sits hard right and no in-panel corner stays clear across seeds.
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    # Panel 3 -- the closed-form marginal score, read back out of the draws.
    grid = metrics[names[0]]["score_grid"]
    axes[2].plot(grid, metrics[names[0]]["exact_score"], color=NAVY, ls="--",
                 lw=2, label="exact d/dg log p(g|x)")
    for name in names:
        axes[2].plot(grid, metrics[name]["implied_score"], lw=1.8,
                     color=METHODS[name]["colour"],
                     label=f"{METHODS[name]['label']} "
                           f"({metrics[name]['marginal_score_rel_rmse']:.2f})")
    axes[2].set(xlabel="global parameter g",
                ylabel="d/dg log p(g | x$_{1:N}$)",
                title="Implied marginal score")
    axes[2].legend(fontsize=8)
    axes[2].text(0.5, -0.26,
                 "from d/dg log KDE(draws), over the central 90% of p(g|x);\n"
                 "bracketed number is the relative RMS error",
                 transform=axes[2].transAxes, ha="center", va="top",
                 color=NAVY, fontsize=8)

    # Panel 4 -- the three error budgets side by side, in posterior sigmas.
    labels = ["ĝ vs exact\nmode", "ĝ vs true g", "locals vs\nexact mean",
              "locals vs\np(ℓ|ĝ,x) mode"]
    keys = ["global_map_error_sigma", "global_map_error_from_truth_sigma",
            "local_map_error_vs_posterior_mean_sigma",
            "local_map_error_vs_conditional_mode_sigma"]
    positions = np.arange(len(keys))
    width = 0.8 / len(names)
    for offset, name in enumerate(names):
        values = [metrics[name][key] for key in keys]
        axes[3].bar(positions + offset * width - 0.4 + width / 2, values,
                    width=width * 0.92, color=METHODS[name]["colour"],
                    label=METHODS[name]["label"])
    axes[3].set_xticks(positions)
    axes[3].set_xticklabels(labels, fontsize=8)
    axes[3].set(yscale="log", ylabel="error / posterior σ", title="MAP error budget")
    axes[3].axhline(1.0, color=NAVY, ls=":", lw=1.2)
    # Below the axes: bars fill the panel at every scale, so no corner inside is
    # reliably empty across runs.
    axes[3].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SCALAR_FIELDS = [
    "method", "observations", "num_samples", "excluded_draws",
    "score_rel_rmse_mean",
    *[f"score_rel_rmse_lam{lam:g}" for lam in SCORE_LAMBDAS],
    *[f"local_score_rel_rmse_lam{lam:g}" for lam in SCORE_LAMBDAS],
    "shared_w1_over_sigma", "shared_mean_error_sigma", "shared_width_ratio",
    "local_w1_over_sigma", "local_draw_mean_error_sigma",
    "marginal_score_rel_rmse", "global_map", "global_map_error_sigma",
    "global_map_error_from_truth_sigma",
    "local_map_error_vs_posterior_mean_sigma",
    "local_map_error_vs_conditional_mode_sigma",
    "newton_global_map", "newton_global_map_error_sigma",
    "newton_local_map_error_vs_conditional_mode_sigma",
    "sample_seconds", "ascent_seconds", "newton_seconds", "sampler_network_calls",
]


def write_metrics(path, rows):
    """Merge this run's rows into the table, keyed by method."""
    merged = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                merged[row["method"]] = row
    for row in rows:
        merged[row["method"]] = {
            field: row.get(field, "") for field in SCALAR_FIELDS
        }
    order = list(METHODS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCALAR_FIELDS)
        writer.writeheader()
        for name in sorted(merged, key=lambda item: (order + [item]).index(item)):
            writer.writerow({f: merged[name].get(f, "") for f in SCALAR_FIELDS})
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        choices=list(METHODS))
    parser.add_argument("--stage", nargs="+", default=["score", "sample"],
                        choices=["score", "sample"])
    parser.add_argument("--config", default="h8d1")
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Checkpoint directory; defaults to the one "
                             "--config/--train-samples names under artifacts/.")
    parser.add_argument("--observations", type=int, default=OBSERVATIONS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--score-states", type=int, default=SCORE_STATES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--excursion-sigma", type=float, default=15.0)
    parser.add_argument("--kde-bandwidth", default=None)
    parser.add_argument("--map-timesteps", type=int, default=200)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-eps", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--replot", action="store_true")
    arguments = parser.parse_args()

    if arguments.kde_bandwidth is not None:
        try:
            arguments.kde_bandwidth = float(arguments.kde_bandwidth)
        except ValueError:
            pass
    if arguments.device is None:
        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"

    configure_style()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = arguments.model_dir or recipe.model_directory(
        ARTIFACTS, arguments.config, arguments.train_samples
    )
    problem = build_problem(
        model_dir, arguments.config, arguments.train_samples,
        arguments.observations, arguments.seed, arguments.device,
    )
    print(f"true g = {problem['global_truth']:.5f}   exact posterior: "
          f"mean {problem['global_mean']:.5f}, sd {problem['global_std']:.5f}, "
          f"mode {problem['global_mode']:.5f}, wall at {problem['x'].min():.5f}")

    reference_path = arguments.output_dir / "network_floor.json"
    if "score" in arguments.stage and not arguments.replot:
        # The network's own single-observation score error on the same ladder --
        # context for the composed numbers, not a bound on them (see
        # plot_comparison).
        print("network reference: single-observation score error")
        reference = recipe.score_fidelity(
            problem["model"], problem["x"], SCORE_LAMBDAS,
            arguments.score_states, arguments.device, seed=arguments.seed + 101,
        )
        reference.pop("mean")
        reference_path.write_text(json.dumps(reference, indent=2) + "\n")
        for lam, value in sorted(reference.items()):
            print(f"  lam={lam:<5g} relative score RMSE {value:.4f}")
    if reference_path.exists():
        problem["network_reference"] = {
            float(key): value
            for key, value in json.loads(reference_path.read_text()).items()
        }

    results, metrics, score_rows = {}, {}, {}
    for name in arguments.methods:
        archive_path = arguments.output_dir / f"{name}.npz"
        if arguments.replot:
            with np.load(archive_path, allow_pickle=True) as archive:
                stored = {key: archive[key] for key in archive.files}
            results[name] = stored
            results[name]["global_map"] = float(stored["global_map"])
            results[name]["excluded"] = int(stored["excluded"])
            metrics[name] = json.loads(str(stored["metrics_json"]))
            metrics[name]["score_grid"] = stored["score_grid"]
            metrics[name]["implied_score"] = stored["implied_score"]
            metrics[name]["exact_score"] = stored["exact_score"]
            score_rows[name] = json.loads(str(stored["score_rows_json"]))
            plot_method(problem, name, results[name], metrics[name],
                        arguments.output_dir / f"01_{name}.png")
            continue

        print(f"\n=== {METHODS[name]['label']} ===")
        kwargs = dict(METHODS[name]["sample_kwargs"])
        samples, sampler, runtime = draw(
            problem["model"], problem, kwargs, arguments.num_samples,
            arguments.timesteps, arguments.seed, arguments.denoise_clamp,
            verbose=arguments.verbose,
        )
        print(f"[{name}] sampled in {runtime:.1f}s, "
              f"{sampler.score_network_calls} network calls")

        rows = []
        if "score" in arguments.stage:
            print(f"[{name}] probing the composed score against exact quadrature")
            rows = probe_composed_score(
                sampler, problem, SCORE_LAMBDAS, arguments.score_states,
                arguments.seed + 101,
            )
            for row in rows:
                print(f"[{name}]   lam={row['lambda']:<5g} shared "
                      f"{row['shared_rel_rmse']:.4f}  local "
                      f"{row['local_rel_rmse']:.4f}")
        score_rows[name] = rows

        result = kde_map_pipeline(problem["model"], problem, samples, arguments)
        if METHODS[name].get("newton_map"):
            print(f"[{name}] newton_map_estimate (jacobian curvature)")
            result.update(newton_map(
                problem["model"], problem, samples, kwargs, arguments
            ))
        results[name] = result
        metrics[name] = evaluate(problem, result, sampler, runtime, rows)
        metrics[name]["method"] = name

        scalar = {key: value for key, value in metrics[name].items()
                  if np.isscalar(value) or isinstance(value, str)}
        print(json.dumps(scalar, indent=2, default=float))

        np.savez_compressed(
            archive_path,
            x=problem["x"], global_grid=problem["global_grid"],
            global_density=problem["global_density"],
            global_truth=problem["global_truth"],
            exact_local_mean=problem["local_mean"],
            exact_local_std=problem["local_std"],
            metrics_json=json.dumps(scalar, default=float),
            score_rows_json=json.dumps(
                [{key: value for key, value in row.items()
                  if not isinstance(value, np.ndarray)} for row in rows],
                default=float,
            ),
            score_grid=metrics[name]["score_grid"],
            implied_score=metrics[name]["implied_score"],
            exact_score=metrics[name]["exact_score"],
            **{key: value for key, value in result.items()},
        )
        plot_method(problem, name, result, metrics[name],
                    arguments.output_dir / f"01_{name}.png")

    if len(results) > 1:
        plot_comparison(problem, results, metrics, score_rows,
                        arguments.output_dir / "02_method_comparison.png",
                        arguments)
    if not arguments.replot:
        write_metrics(arguments.output_dir / "methods.csv",
                      [metrics[name] for name in arguments.methods])


if __name__ == "__main__":
    main()
