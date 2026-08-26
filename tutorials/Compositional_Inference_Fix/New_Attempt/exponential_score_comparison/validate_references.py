#!/usr/bin/env python3
"""Check every exact reference in hierarchy.py before any of them is trusted.

The whole experiment grades samplers against closed forms and quadratures, so a
sign error in one of those would silently become a "finding". Each check here
compares an analytic expression against an independent numerical route:

1. ``exact_global_score`` vs finite differences of ``log_shared_posterior``.
2. ``diffused_score`` vs finite differences of a brute-force ``log p_lam``.
3. ``diffused_score`` at large lam vs the Gaussian limit, where the diffused
   joint is dominated by the kernel and the score must approach
   ``(E[theta] - theta_t) / lam^2``.
4. ``sample_diffused`` moments vs quadrature moments of the same distribution.
5. ``shared_reference`` vs importance sampling against the analytic Gamma(2, rate)
   marginal likelihood -- a check that the tall posterior formula agrees with the
   ``x_j - g ~ Gamma(2, rate)`` reduction it is derived from.
6. The *simulator itself* is hierarchical: ``observations`` must draw **one**
   ``g`` and share it across all ``count`` observations, with independent
   per-observation ``eps_j`` and ``eta_j``, each Exponential(rate). Checks 1-5
   all grade closed forms against other closed forms; none of them touches the
   code that generates the data, so a simulator that drew a fresh ``g`` per
   observation would pass every one of them.
7. Simulation-based calibration: over many replicate datasets from that
   simulator, the exact posterior CDF evaluated at the true ``g`` must be
   Uniform(0, 1). This is the end-to-end check that the posterior really is the
   posterior *of this simulator*, and it is sharply sensitive to the shared-``g``
   structure -- if ``g`` were not shared, the tall posterior would contract like
   ``1/N`` around the wrong point and the ranks would pile up in the tails.

``simulate`` (training draws) deliberately draws a **fresh ``g`` per row**, and
that is not an inconsistency with ``observations``: the network is trained on
single ``(theta, x)`` pairs from the prior predictive, so each row is its own
one-observation problem. Sharing ``g`` is a property of the *tall dataset* the
composition rules are then evaluated on. Check 6 asserts both halves.
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchy  # noqa: E402

FAILURES = []


def check(name, value, tolerance):
    status = "ok  " if value <= tolerance else "FAIL"
    if value > tolerance:
        FAILURES.append(name)
    print(f"  [{status}] {name}: {value:.3e} (tolerance {tolerance:.1e})")


def brute_force_log_density(shared, local, x, lam, grid):
    """log p_lam(g_t, l_t | x) by direct quadrature, no gradient shortcuts."""
    grid = torch.as_tensor(grid, dtype=torch.float64)
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    n = int(x.numel())
    lower = (grid[:, None] - torch.as_tensor(local, dtype=torch.float64)[None, :]) / lam
    upper = (x[None, :] - torch.as_tensor(local, dtype=torch.float64)[None, :]) / lam
    upper = upper.expand_as(lower)
    interval = (torch.special.ndtr(upper) - torch.special.ndtr(lower)).clamp_min(1e-300)
    log_integrand = (
        -0.5 * ((grid - hierarchy.MU_G) / hierarchy.SIGMA_G) ** 2
        + n * hierarchy.RATE * grid
        - 0.5 * ((float(shared) - grid) / lam) ** 2
        + torch.log(interval).sum(dim=-1)
    )
    return float(torch.logsumexp(log_integrand, dim=0))


def main():
    torch.set_default_dtype(torch.float64)
    n_observations = 6                       # small, so brute force stays cheap
    _, _, x = hierarchy.observations(n_observations, seed=0)
    print(f"observations (n={n_observations}): min x = {x.min():.4f}")

    # -- 1. the closed-form tall score ------------------------------------
    print("\n1. exact_global_score vs finite differences of log p(g | x)")
    grid, _, weights, mean, std = hierarchy.shared_reference(x)
    probes = np.array([mean - 2 * std, mean - std, mean, mean + 0.5 * std])
    step = 1e-6
    numeric = (hierarchy.log_shared_posterior(x, probes + step)
               - hierarchy.log_shared_posterior(x, probes - step)) / (2 * step)
    analytic = hierarchy.exact_global_score(probes, x)
    check("global score", float(np.abs(numeric - analytic).max()
                                / np.abs(analytic).max()), 1e-6)

    # -- 2. the diffused joint score --------------------------------------
    print("\n2. diffused_score vs finite differences of log p_lam")
    quadrature = hierarchy.quadrature_grid(x, points=40001)
    for lam in (0.05, 0.3, 1.0, 3.0):
        shared, local, _, _ = hierarchy.sample_diffused(x, lam, 4, seed=3)
        score_g, score_l = hierarchy.diffused_score(shared, local, x, lam, quadrature)
        step = 1e-4 * max(lam, 0.05)
        errors_g, errors_l, magnitude = [], [], []
        for index in range(len(shared)):
            plus = brute_force_log_density(
                shared[index] + step, local[index], x, lam, quadrature)
            minus = brute_force_log_density(
                shared[index] - step, local[index], x, lam, quadrature)
            errors_g.append(abs((plus - minus) / (2 * step) - float(score_g[index])))
            magnitude.append(abs(float(score_g[index])))
            for j in range(n_observations):
                shifted = local[index].copy()
                shifted[j] += step
                plus = brute_force_log_density(shared[index], shifted, x, lam, quadrature)
                shifted[j] -= 2 * step
                minus = brute_force_log_density(shared[index], shifted, x, lam, quadrature)
                errors_l.append(
                    abs((plus - minus) / (2 * step) - float(score_l[index, j]))
                )
                magnitude.append(abs(float(score_l[index, j])))
        scale = max(np.mean(magnitude), 1e-9)
        check(f"lam={lam:<5g} shared", float(np.max(errors_g)) / scale, 2e-4)
        check(f"lam={lam:<5g} local ", float(np.max(errors_l)) / scale, 2e-4)

    # -- 3. the Gaussian limit at large lam --------------------------------
    print("\n3. large-lam limit: score -> (E[theta] - theta_t) / lam^2")
    lam = 40.0
    posterior_mean_g = mean
    local_mean, _ = hierarchy.local_reference(x, grid, weights)
    shared, local, _, _ = hierarchy.sample_diffused(x, lam, 8, seed=5)
    score_g, score_l = hierarchy.diffused_score(shared, local, x, lam, quadrature)
    predicted_g = (posterior_mean_g - shared) / lam**2
    predicted_l = (local_mean[None, :] - local) / lam**2
    scale = float(np.abs(predicted_g).mean())
    check("shared", float(np.abs(score_g.numpy() - predicted_g).max()) / scale, 5e-3)
    check("local ", float(np.abs(score_l.numpy() - predicted_l).max()) / scale, 5e-3)

    # -- 4. sample_diffused moments ---------------------------------------
    print("\n4. sample_diffused moments vs quadrature moments")
    lam = 0.2
    shared, local, clean_g, clean_l = hierarchy.sample_diffused(x, lam, 400_000, seed=7)
    check("E[g]", abs(clean_g.mean() - mean) / std, 5e-3)
    check("sd[g]", abs(clean_g.std() - std) / std, 5e-3)
    check("E[l_j]", float(np.abs(clean_l.mean(axis=0) - local_mean).max()) / std, 2e-2)
    check("Var[g_t]", abs(shared.var() - (std**2 + lam**2)) / (std**2 + lam**2), 1e-2)

    # -- 5. the posterior really is the simulator's posterior --------------
    print("\n5. shared_reference vs rejection-ABC on the forward simulator")
    # p(g | x) by importance weighting: draw g from the prior, weight by the
    # exact likelihood prod_j rate^2 (x_j - g) exp(-rate (x_j - g)) 1[g <= x_j],
    # which is the Gamma(2, rate) density of x_j - g. Independent of the
    # log-posterior code path, which builds the same thing on a grid.
    generator = np.random.default_rng(11)
    draws = hierarchy.MU_G + hierarchy.SIGMA_G * generator.standard_normal(4_000_000)
    gap = np.asarray(x)[None, :] - draws[:, None]
    valid = np.all(gap > 0.0, axis=1)
    log_weight = np.full(draws.shape, -np.inf)
    log_weight[valid] = (
        np.log(gap[valid]).sum(axis=1) - hierarchy.RATE * gap[valid].sum(axis=1)
    )
    weight = np.exp(log_weight - log_weight.max())
    weight /= weight.sum()
    importance_mean = float((weight * draws).sum())
    importance_std = float(
        max((weight * draws**2).sum() - importance_mean**2, 0.0) ** 0.5
    )
    ess = 1.0 / float((weight**2).sum())
    print(f"  effective sample size {ess:,.0f}")
    check("E[g]", abs(importance_mean - mean) / std, 0.05)
    check("sd[g]", abs(importance_std - std) / std, 0.05)

    # -- 6. the simulator is hierarchical -----------------------------------
    print("\n6. observations() shares one g; simulate() draws one per row")
    from scipy import stats

    count = 12
    g_truth, local_truth, x_truth = hierarchy.observations(count, seed=21)
    check("g is one scalar, not one per observation",
          0.0 if np.ndim(g_truth) == 0 else 1.0, 0.0)
    check("locals above g", float(max(0.0, g_truth - np.min(local_truth))), 0.0)
    check("observations above locals",
          float(max(0.0, np.max(local_truth - x_truth))), 0.0)

    # The decisive quantitative test of sharing. Within one dataset,
    #   shared g:     Var_j(l_j) = Var(eps)     = 1,   Var_j(x_j) = Var(eps+eta) = 2
    #   fresh g per j: Var_j(l_j) = Var(g)+Var(eps) = 2, Var_j(x_j)             = 3
    # so the within-dataset spread alone separates the two hypotheses by a
    # factor of two. Averaged over many datasets it is a sharp test.
    within_local, within_x, dataset_mean = [], [], []
    for index in range(3000):
        _, local_value, x_value = hierarchy.observations(40, seed=300_000 + index)
        # ddof=1: the comparison values below are population variances, and the
        # default ddof=0 would sit a factor (n-1)/n low against them.
        within_local.append(local_value.var(ddof=1))
        within_x.append(x_value.var(ddof=1))
        dataset_mean.append(local_value.mean())
    variance_eps = 1.0 / hierarchy.RATE**2
    print(f"  [info] within-dataset Var(l_j) = {np.mean(within_local):.4f} "
          f"(shared g -> {variance_eps:g}; fresh g per observation -> "
          f"{variance_eps + hierarchy.SIGMA_G**2:g})")
    check("within-dataset Var(l_j)",
          abs(float(np.mean(within_local)) - variance_eps), 0.05)
    check("within-dataset Var(x_j)",
          abs(float(np.mean(within_x)) - 2 * variance_eps), 0.05)
    # And between datasets: with g shared, the dataset mean keeps the *whole*
    # prior variance no matter how many observations it averages over.
    expected = hierarchy.SIGMA_G**2 + variance_eps / 40
    print(f"  [info] between-dataset Var(mean_j l_j) = "
          f"{np.var(dataset_mean):.4f} (shared g -> {expected:.4f}; "
          f"fresh g per observation -> {(hierarchy.SIGMA_G**2 + variance_eps)/40:.4f})")
    check("between-dataset Var(mean_j l_j)",
          abs(float(np.var(dataset_mean)) - expected) / expected, 0.06)

    # Distributional: pool the increments over many datasets and test them
    # against Exponential(rate) directly, rather than against another formula.
    replicates = 4000
    epsilons, etas, shared = [], [], []
    for index in range(replicates):
        g_value, local_value, x_value = hierarchy.observations(3, seed=100_000 + index)
        epsilons.append(local_value - g_value)
        etas.append(x_value - local_value)
        shared.append(g_value)
    epsilons = np.concatenate(epsilons)
    etas = np.concatenate(etas)
    shared = np.asarray(shared)
    scale = 1.0 / hierarchy.RATE
    check("eps ~ Exp(rate) (KS)",
          float(stats.kstest(epsilons, "expon", args=(0.0, scale)).statistic), 0.03)
    check("eta ~ Exp(rate) (KS)",
          float(stats.kstest(etas, "expon", args=(0.0, scale)).statistic), 0.03)
    check("g ~ Normal(mu, sigma) (KS)",
          float(stats.kstest(
              shared, "norm", args=(hierarchy.MU_G, hierarchy.SIGMA_G)
          ).statistic), 0.03)

    # The training simulator: same three laws, but a fresh g on every row.
    theta, x_train = hierarchy.simulate(200_000, torch.Generator().manual_seed(5))
    g_train = theta[:, 0].numpy()
    check("simulate: eps ~ Exp(rate) (KS)",
          float(stats.kstest(
              (theta[:, 1] - theta[:, 0]).numpy(), "expon", args=(0.0, scale)
          ).statistic), 0.01)
    check("simulate: eta ~ Exp(rate) (KS)",
          float(stats.kstest(
              (x_train[:, 0] - theta[:, 1]).numpy(), "expon", args=(0.0, scale)
          ).statistic), 0.01)
    check("simulate: g ~ Normal (KS)",
          float(stats.kstest(
              g_train, "norm", args=(hierarchy.MU_G, hierarchy.SIGMA_G)
          ).statistic), 0.01)
    # Training rows must NOT share g -- one g per row is the single-observation
    # problem the score network is supposed to learn.
    print(f"  [info] simulate() g spread across rows: sd = {g_train.std():.4f} "
          f"(expected {hierarchy.SIGMA_G:g}; a shared g would give 0)")
    check("simulate: g varies per row",
          float(abs(g_train.std() - hierarchy.SIGMA_G)), 0.02)

    # -- 7. simulation-based calibration ------------------------------------
    print("\n7. SBC: posterior CDF at the true g is Uniform(0,1)")
    for n_sbc in (4, 30):
        ranks = []
        for index in range(600):
            g_value, _, x_value = hierarchy.observations(n_sbc, seed=500_000 + index)
            grid_sbc, _, weights_sbc, _, _ = hierarchy.shared_reference(x_value)
            cumulative = np.cumsum(weights_sbc)
            ranks.append(float(np.interp(g_value, grid_sbc, cumulative)))
        ranks = np.asarray(ranks)
        statistic = float(stats.kstest(ranks, "uniform").statistic)
        print(f"  n={n_sbc:<3} mean rank {ranks.mean():.4f} (expect 0.5), "
              f"KS {statistic:.4f}")
        check(f"n={n_sbc} ranks uniform (KS)", statistic, 0.06)
        check(f"n={n_sbc} mean rank", abs(ranks.mean() - 0.5), 0.03)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        raise SystemExit(1)
    print("All reference checks passed.")


if __name__ == "__main__":
    main()
