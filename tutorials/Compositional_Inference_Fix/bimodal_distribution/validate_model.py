#!/usr/bin/env python3
"""Grade the closed-form references against independent derivations.

Nothing in this directory is trustworthy if the "exact" posterior is not exact,
so every reference the experiment relies on is checked here by a second route:

1. ``V_g`` and the aligned-branch mean against the design note's closed forms
   ``V_g = (1/sigma_g^2 + n/v)^-1`` and ``m_s = V_g/v sum_j s_j y_j``, reached
   through the arrow covariance rather than through the marginalized form.
2. ``p(g | y)`` against direct numerical normalisation of

       p(g | y) ~ N(g; 0, sigma_g^2) prod_j 1/2 [N(y_j; g, v) + N(y_j; -g, v)]

   which comes from integrating each ``l_j`` out analytically -- a derivation
   that never builds the 2^n mixture at all.
3. ``p(l_1 | y)`` against a two-dimensional grid over ``(g, l_1)`` with the
   remaining locals integrated out by the same one-dimensional identity.
4. The diffused score against a central difference of the diffused log density,
   at several noise levels.
5. Coverage: over many forward-simulated datasets, the true ``g`` should fall
   below its own posterior quantile uniformly.
6. The control model's posterior against the same treatment.

Usage:
    python validate_model.py
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import kstest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import control_model  # noqa: E402
import mixture_reference as reference  # noqa: E402
import sign_model  # noqa: E402

TOLERANCE = 1e-6


def _normal(x, mean, variance):
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / math.sqrt(2 * math.pi * variance)


def marginal_likelihood(module, g, value):
    """``p(y_j | g)`` with ``l_j`` integrated out analytically."""
    v = module.TAU**2 + module.SIGMA_Y**2
    if module is sign_model:
        return 0.5 * (_normal(value, g, v) + _normal(value, -g, v))
    return _normal(value, g, v)


def prior_density(module, g):
    if module is sign_model:
        return _normal(g, 0.0, module.SIGMA_G**2)
    return 0.5 * (_normal(g, module.A, module.SIGMA_0**2)
                  + _normal(g, -module.A, module.SIGMA_0**2))


def check_shared_marginal(module, y, points=200001, span=8.0):
    """Route 2: the shared marginal from an ``l``-free derivation."""
    grid, density, _, mean, std = reference.shared_reference(module, y)
    wide = np.linspace(mean - span * std, mean + span * std, points)
    direct = prior_density(module, wide)
    for value in np.asarray(y, dtype=np.float64).reshape(-1):
        direct = direct * marginal_likelihood(module, wide, value)
    direct /= np.trapezoid(direct, wide)
    comparison = np.interp(grid, wide, direct)
    scale = float(density.max())
    return float(np.abs(comparison - density).max() / scale)


def check_local_marginal(module, y, index=0, points=2001, span=8.0):
    """Route 3: one local marginal from a 2-D grid over ``(g, l_index)``."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    grid, density, _, mean, std = reference.local_reference(module, y, index)
    g_grid = np.linspace(-8.0, 8.0, points)
    l_grid = np.linspace(mean - span * std, mean + span * std, points)

    joint = prior_density(module, g_grid)[:, None] * np.ones(points)[None, :]
    for position, value in enumerate(y):
        if position == index:
            continue
        joint = joint * marginal_likelihood(module, g_grid, value)[:, None]
    # The retained local is kept explicit: p(l | g) p(y_index | l).
    joint = joint * _normal(l_grid[None, :], g_grid[:, None], module.TAU**2)
    if module is sign_model:
        emission = 0.5 * (_normal(y[index], l_grid, module.SIGMA_Y**2)
                          + _normal(y[index], -l_grid, module.SIGMA_Y**2))
    else:
        emission = _normal(y[index], l_grid, module.SIGMA_Y**2)
    joint = joint * emission[None, :]

    direct = np.trapezoid(joint, g_grid, axis=0)
    direct /= np.trapezoid(direct, l_grid)
    comparison = np.interp(grid, l_grid, direct)
    return float(np.abs(comparison - density).max() / float(density.max()))


def check_diffused_score(module, y, lambdas, states=64, step=1e-4, seed=4):
    """Route 4: the diffused score against central differences of log p_lam."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    weights, means, covariance = module.mixture(y)
    worst = 0.0
    for lam in lambdas:
        shared, local, _, _ = reference.sample_diffused(module, y, lam, states, seed)
        state = np.concatenate([shared[:, None], local], axis=1)
        precision = np.linalg.inv(covariance + lam**2 * np.eye(n + 1))

        def log_density(points):
            residual = points[:, None, :] - means[None, :, :]
            quadratic = np.einsum("ski,ij,skj->sk", residual, precision, residual)
            shifted = np.log(weights)[None, :] - 0.5 * quadratic
            top = shifted.max(axis=1, keepdims=True)
            return (top[:, 0] + np.log(np.exp(shifted - top).sum(axis=1)))

        finite = np.zeros_like(state)
        for axis in range(n + 1):
            offset = np.zeros(n + 1)
            offset[axis] = step
            finite[:, axis] = (log_density(state + offset)
                               - log_density(state - offset)) / (2 * step)
        exact_g, exact_l = reference.diffused_score(module, shared, local, y, lam)
        exact = np.concatenate([exact_g[:, None], exact_l], axis=1)
        worst = max(worst, float(
            np.sqrt(((exact - finite) ** 2).mean() / (finite**2).mean())
        ))
    return worst


def check_coverage(module, n, datasets=300, draws=2000, seed=2024):
    """Route 5: the true ``g`` should be uniform under its own posterior CDF."""
    quantiles = []
    for index in range(datasets):
        truth, _, _, y = module.observations(n, seed=seed + index)
        shared, _ = reference.joint_draws(module, y, draws, seed=7 * index + 1)
        quantiles.append(float((shared < truth).mean()))
    return float(kstest(quantiles, "uniform").pvalue)


def check_design_note():
    """Reproduce the worked example of the design note, number for number.

    ``sigma_g = 1.5, tau = 0.8, sigma_y = 0.3`` with ``y = (1.4, 1.55, 1.50,
    1.45)`` should give ``v = 0.73``, ``V_g ~ 0.169``, modes at ``+-1.36`` and
    ~93% of the sign mass on the two coherent configurations. Those are not the
    parameters ``sign_model`` ships (see its constants), so they are set for the
    duration of this check and restored afterwards.
    """
    y = [1.4, 1.55, 1.50, 1.45]
    saved = (sign_model.TAU, sign_model.SIGMA_Y)
    try:
        sign_model.TAU, sign_model.SIGMA_Y = 0.8, 0.3
        v = sign_model.TAU**2 + sign_model.SIGMA_Y**2
        expected_variance = 1.0 / (1.0 / sign_model.SIGMA_G**2 + len(y) / v)
        expected_mean = expected_variance / v * float(np.sum(y))
        weights, means, variance = reference.marginal(sign_model, y, 0)
        coherent = reference.aligned_mass(sign_model, y)
        print(f"  design note: v {v:.4f} (0.73), V_g {variance:.4f} "
              f"(0.169), modes +-{abs(means[int(np.argmax(weights))]):.3f} "
              f"(1.36), coherent mass {coherent:.3f} (0.93)")
        ok = (abs(variance - expected_variance) < TOLERANCE
              and abs(abs(means[int(np.argmax(weights))]) - abs(expected_mean))
              < TOLERANCE
              and abs(variance - 0.169) < 5e-4
              and abs(abs(expected_mean) - 1.36) < 5e-3
              and abs(coherent - 0.93) < 5e-3)
    finally:
        sign_model.TAU, sign_model.SIGMA_Y = saved
    return ok


def main():
    failures = []
    print("\n=== the design note's worked example ===")
    if not check_design_note():
        failures.append("sign: design-note closed forms")

    for module in (sign_model, control_model):
        print(f"\n=== {module.TITLE} ===")
        _, _, _, y = module.observations(6, seed=11)
        shared_error = check_shared_marginal(module, y)
        local_error = check_local_marginal(module, y)
        score_error = check_diffused_score(module, y, (0.05, 0.2, 1.0, 3.0))
        coverage = check_coverage(module, 4)
        print(f"  p(g | y)   vs direct integration: {shared_error:.2e} (relative to peak)")
        print(f"  p(l_1 | y) vs 2-D grid:           {local_error:.2e}")
        print(f"  diffused score vs finite differences: {score_error:.2e}")
        print(f"  coverage of the true g, KS p-value:   {coverage:.3f}")
        if shared_error > 1e-4 or local_error > 1e-3:
            failures.append(f"{module.NAME}: marginals")
        if score_error > 1e-5:
            failures.append(f"{module.NAME}: diffused score")
        if coverage < 0.01:
            failures.append(f"{module.NAME}: coverage")

    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))
    print("\nEvery reference agrees with its independent derivation.")


if __name__ == "__main__":
    main()
