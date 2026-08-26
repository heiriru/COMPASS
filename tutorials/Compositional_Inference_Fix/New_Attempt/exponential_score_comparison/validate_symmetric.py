#!/usr/bin/env python3
"""Grade every closed form in ``hierarchy_gauss`` and ``hierarchy_laplace``.

Each check reaches the same quantity by a route the module does not use, so
agreement grades the module rather than restating it:

``simulator``      the noise law's own mean/variance, and the hierarchy's
                   within-dataset spread, from forward draws only.
``global score``   central differences of ``log_shared_posterior``.
``diffused score`` central differences of a **brute-force** 2-D quadrature of
                   ``p_lam(g_t, l_t | x) = int int p(g, l | x) N(g_t; g, lam^2)
                   N(l_t; l, lam^2)``, which uses no analytic convolution at all.
                   This is the check that matters: it is the only independent
                   route to the object the composition rules are graded against.
``locals``         ``conditional_local`` / ``local_reference`` /
                   ``local_marginal`` against Monte-Carlo moments of
                   ``sample_diffused`` at lam = 0.
``SBC``            the exact posterior CDF at the true ``g`` is Uniform(0, 1)
                   over replicate datasets -- the end-to-end check that the
                   posterior is the posterior *of this simulator*.

Usage:
    python validate_symmetric.py
    python validate_symmetric.py --module laplace --verbose
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
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hierarchy_gauss  # noqa: E402
import hierarchy_laplace  # noqa: E402

MODULES = {"gauss": hierarchy_gauss, "laplace": hierarchy_laplace}

failures = []


def check(name, value, tolerance, detail=""):
    ok = bool(np.isfinite(value)) and abs(value) <= tolerance
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<52s} {value:.3e} "
          f"(tol {tolerance:.0e}) {detail}")
    if not ok:
        failures.append(name)
    return ok


# ---------------------------------------------------------------------------

def noise_draws(module, count, seed=0):
    """``eps`` alone, recovered as ``l - g`` from the simulator."""
    generator = torch.Generator().manual_seed(seed)
    theta, _ = module.simulate(count, generator)
    return (theta[:, 1] - theta[:, 0]).numpy()


def validate_simulator(module, name):
    print(f"{name}: simulator")
    epsilon = noise_draws(module, 400_000, seed=11)
    check("noise mean (target 0)", float(epsilon.mean()), 6e-3)
    check("noise variance (target 1)", float(epsilon.var()) - 1.0, 1e-2)
    check("noise skew (target 0, symmetric)",
          float((epsilon**3).mean()) / float(epsilon.var()) ** 1.5, 2e-2)

    # The hierarchy must be real: with g shared, Var_j(x_j) within a dataset is
    # 2 (eps + eta), whereas a fresh g per row would give 3.
    spreads = []
    for seed in range(300):
        _, _, x = module.observations(40, 5000 + seed)
        spreads.append(float(np.var(x)))
    check("within-dataset Var_j(x_j) - 2  [fresh g would give 3]",
          float(np.mean(spreads)) - 2.0, 6e-2)


def validate_global_score(module, name, seed=0):
    print(f"{name}: d/dg log p(g | x) vs central differences")
    _, _, x = module.observations(30, seed)
    grid, _, weights, mean, sd = module.shared_reference(x)
    probe = np.linspace(mean - 2.5 * sd, mean + 2.5 * sd, 41)
    step = 1e-5
    reference = (module.log_shared_posterior(x, probe + step)
                 - module.log_shared_posterior(x, probe - step)) / (2 * step)
    analytic = module.exact_global_score(probe, x)
    relative = float(np.sqrt(np.mean((analytic - reference) ** 2)
                             / np.mean(reference**2)))
    check("relative RMS", relative, 1e-6)

    # The normalized grid must integrate to the same moments the grid reports.
    check("grid weights sum - 1", float(weights.sum()) - 1.0, 1e-12)
    # The grid must not clip the posterior: the tails it drops must be negligible
    # on both sides, or the reported sd is a property of the window.
    edge = max(float(weights[0]), float(weights[-1])) / float(weights.max())
    check("relative weight at the grid edge", edge, 1e-10, f"sd = {sd:.5f}")


def _log_joint(module, g, l, value):
    """log p(g, l | x) for ONE observation, up to a constant. No convolution."""
    if module is hierarchy_gauss:
        s = module.NOISE_STD
        return (-0.5 * ((g - module.MU_G) / module.SIGMA_G) ** 2
                - 0.5 * ((l - g) / s) ** 2 - 0.5 * ((value - l) / s) ** 2)
    b = module.SCALE
    return (-0.5 * ((g - module.MU_G) / module.SIGMA_G) ** 2
            - np.abs(l - g) / b - np.abs(value - l) / b)


def _brute_force_log_diffused(module, g_t, l_t, value, lam, points=4001, span=14.0):
    """log p_lam(g_t, l_t | x) by direct 2-D quadrature, no analytic integrals."""
    axis = np.linspace(-span, span, points)
    g = axis[:, None]
    l = axis[None, :]
    log_joint = _log_joint(module, g, l, value)
    log_kernel = (-0.5 * ((g_t - g) / lam) ** 2 - 0.5 * ((l_t - l) / lam) ** 2)
    total = log_joint + log_kernel
    peak = total.max()
    return float(peak + np.log(np.exp(total - peak).sum()))


def validate_diffused_score(module, name, lambdas=(0.05, 0.2, 1.0, 3.0)):
    print(f"{name}: diffused score vs brute-force 2-D quadrature (N = 1)")
    _, _, x = module.observations(30, 0)
    value = float(x[0])
    grid = module.quadrature_grid([value], device="cpu")
    for lam in lambdas:
        rng = np.random.default_rng(7)
        shared, local, _, _ = module.sample_diffused([value], lam, 6, 3)
        errors_g, errors_l, magnitude = [], [], []
        for index in range(len(shared)):
            g_t, l_t = float(shared[index]), float(local[index, 0])
            analytic_g, analytic_l = module.diffused_score(
                np.array([g_t]), np.array([[l_t]]), [value], lam, grid
            )
            step = max(1e-4, 1e-3 * lam)
            numeric_g = (
                _brute_force_log_diffused(module, g_t + step, l_t, value, lam)
                - _brute_force_log_diffused(module, g_t - step, l_t, value, lam)
            ) / (2 * step)
            numeric_l = (
                _brute_force_log_diffused(module, g_t, l_t + step, value, lam)
                - _brute_force_log_diffused(module, g_t, l_t - step, value, lam)
            ) / (2 * step)
            errors_g.append((float(analytic_g[0]) - numeric_g) ** 2)
            errors_l.append((float(analytic_l[0, 0]) - numeric_l) ** 2)
            magnitude.append(numeric_g**2 + numeric_l**2)
        relative = math.sqrt((sum(errors_g) + sum(errors_l))
                             / max(sum(magnitude), 1e-30))
        check(f"lam = {lam:<5g} relative RMS (shared + local)", relative, 2e-4)


def validate_locals(module, name, seed=0):
    print(f"{name}: local references vs Monte-Carlo from sample_diffused")
    _, _, x = module.observations(30, seed)
    grid, _, weights, mean_g, _ = module.shared_reference(x)
    mean_l, std_l = module.local_reference(x, grid, weights)

    _, _, g_clean, local_clean = module.sample_diffused(x, 0.0, 200_000, 21)
    check("E[l_j | x] max abs error",
          float(np.abs(local_clean.mean(axis=0) - mean_l).max()), 2e-2)
    check("sd[l_j | x] max rel error",
          float(np.abs(local_clean.std(axis=0) / std_l - 1.0).max()), 2e-2)

    # p(l | g, x) is symmetric, so its mean equals the midpoint mode.
    mode, cond_sd = module.conditional_local(x, mean_g)
    inside = np.abs(g_clean - mean_g) < 0.02
    if inside.sum() > 2000:
        conditional = local_clean[inside]
        check("E[l_j | g, x] - midpoint, max abs",
              float(np.abs(conditional.mean(axis=0) - mode).max()), 4e-2)
        check("sd[l_j | g, x] max rel error",
              float(np.abs(conditional.std(axis=0) / cond_sd - 1.0).max()), 5e-2)

    # local_marginal must reproduce local_reference's moments.
    local_grid = np.linspace(float(np.min(x)) - 8.0, float(np.max(x)) + 8.0, 20001)
    marginal_mean, marginal_sd = [], []
    for value in x:
        w = module.local_marginal(value, grid, weights, local_grid)
        m = float((w * local_grid).sum())
        marginal_mean.append(m)
        marginal_sd.append(math.sqrt(max((w * local_grid**2).sum() - m**2, 0.0)))
    check("local_marginal mean vs local_reference, max abs",
          float(np.abs(np.array(marginal_mean) - mean_l).max()), 5e-3)
    check("local_marginal sd vs local_reference, max rel",
          float(np.abs(np.array(marginal_sd) / std_l - 1.0).max()), 5e-3)


def validate_sbc(module, name, replicates=400, observations=30):
    print(f"{name}: simulation-based calibration ({replicates} datasets)")
    ranks = []
    for seed in range(replicates):
        truth, _, x = module.observations(observations, 20_000 + seed)
        grid, _, weights, _, _ = module.shared_reference(x)
        ranks.append(float(np.interp(truth, grid, np.cumsum(weights))))
    ranks = np.array(ranks)
    check("mean rank - 0.5", float(ranks.mean()) - 0.5, 3.5 / math.sqrt(replicates))
    grid = np.linspace(0, 1, 201)
    empirical = np.searchsorted(np.sort(ranks), grid, side="right") / len(ranks)
    check("KS distance from Uniform(0,1)",
          float(np.abs(empirical - grid).max()), 2.5 / math.sqrt(replicates))


def validate_gaussian_extra():
    """The Gaussian module also admits a fully independent linear-algebra route."""
    print("gauss: joint moments vs the tall marginal")
    module = hierarchy_gauss
    _, _, x = module.observations(30, 0)
    mean, covariance = module.joint_moments(x)
    marginal_mean, marginal_variance = module.shared_moments(x)
    check("E[g | x] from the arrow Gaussian", float(mean[0]) - marginal_mean, 1e-9)
    check("Var[g | x] from the arrow Gaussian",
          float(covariance[0, 0]) - marginal_variance, 1e-9)
    grid, _, weights, _, _ = module.shared_reference(x)
    local_mean, local_std = module.local_reference(x, grid, weights)
    check("E[l_j | x] from the arrow Gaussian, max abs",
          float(np.abs(mean[1:] - local_mean).max()), 1e-9)
    check("sd[l_j | x] from the arrow Gaussian, max abs",
          float(np.abs(np.sqrt(np.diag(covariance)[1:]) - local_std).max()), 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", nargs="+", default=list(MODULES),
                        choices=list(MODULES))
    arguments = parser.parse_args()

    for name in arguments.module:
        module = MODULES[name]
        print(f"\n=== {name} ===")
        validate_simulator(module, name)
        validate_global_score(module, name)
        validate_diffused_score(module, name)
        validate_locals(module, name)
        validate_sbc(module, name)
        if module is hierarchy_gauss:
            validate_gaussian_extra()

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
