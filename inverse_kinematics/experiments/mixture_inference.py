#!/usr/bin/env python3
"""Composed-posterior inference by Gaussian-mixture product, K inferred.

This is the production form of the method developed in `docs/mixture_product.md`.
Nothing here is swept and nothing is given: the component count is chosen from
the data at inference time, and the composition is guarded against the one
failure mode that was measured.

The method
----------
For a hierarchy with one shared block ``a`` and independent locals,

    p(a | x_1..n)  proportional to  p(a)^(1-n) * prod_j p(a | x_j)

is exact. Each ``p(a | x_j)`` is summarised as a Gaussian *mixture* rather than a
single Gaussian -- one Gaussian cannot carry unequal branch weights, which is the
measured defect of every rule in ``MultiObsSampler``. Mixtures are closed under
products and under diffusion, so the composition is closed form in any dimension.

Choosing K without being told it
--------------------------------
Per observation, by BIC over ``K = 1..K_max``. BIC is known to be the wrong norm
for this problem -- composed error is the *sum* of per-observation log-density
errors, so it is the sup-norm that matters, while BIC estimates an average
weighted by the density itself (`docs/mixture_product.md`, follow-up 5). It is
used anyway because every alternative measured did no better, tail-weighted
Anderson-Darling included, for a reason no statistic can dodge: the region that
controls the composed error holds ~1e-5 of each observation's draws. The
information is not in the sample.

What makes that acceptable is that the damage is *bounded* rather than predicted,
by the guard below. K then only has to be good, not right.

The guard, and why it exists
----------------------------
A composed component has precision ``P - (n-1)/sigma_0^2`` after the prior
division. A component whose partial-product precision approaches ``(n-1)/sigma_0^2``
is nearly cancelled: it becomes broad, and because a component's log-mass carries
``h^2 / (2P)``, it simultaneously acquires enormous weight. Measured at n = 16,
one such component reached 87% of the composed mass centred at ``a = 1.5``, where
no observation had a single draw, and the composed W1/sigma went to 3.2.

So merging -- which is variance-*increasing* by construction -- is forbidden from
producing any partial-product component with precision below
``margin * (n-1)/sigma_0^2``. This is not a heuristic threshold: it is the
condition for the prior division to leave a non-degenerate component.

Two reference-free diagnostics are returned alongside the answer:

``cancellation_ratio``  min composed precision over ``(n-1)/sigma_0^2``. Small
                        means the answer is dominated by a near-cancelled
                        component and should not be trusted.
``outside_support``     composed mass outside the intersection of the
                        per-observation sampled ranges. Since the composed
                        support *is* that intersection, any mass there is
                        provably wrong.
"""
from __future__ import annotations

import numpy as np

from mixture_product import bic, density_on_grid, fit_mixture, to_natural
from mixture_refinements import from_moments, to_moments

DEFAULT_K_MAX = 5
DEFAULT_BEAM = 64
DEFAULT_MARGIN = 4.0


def choose_and_fit(draws, k_max=DEFAULT_K_MAX, restarts=1, seed=0, device=None):
    """Per-observation mixture with K chosen by BIC. Returns (fits, chosen).

    ``device`` sends the fitting to ``batched_em.fit_all``, which fits every
    observation and every K at once instead of looping. Measured at n = 8 with
    20,000 draws: 43.7 s in this loop against 1.29 s batched on the GPU, a 33.8x
    difference, with the composed W1/sigma unchanged inside EM fit noise. The
    loop is kept as the reference implementation and the default.
    """
    if device is not None:
        from batched_em import fit_all
        return fit_all(draws, k_max=k_max, seed=seed, device=device)
    fits, chosen = [], []
    for index, row in enumerate(draws):
        candidates = {}
        for k in range(1, k_max + 1):
            fit = fit_mixture(row, k, restarts=restarts, seed=seed + index)
            candidates[k] = (bic(row, *fit), fit)
        best = min(candidates, key=lambda k: candidates[k][0])
        fits.append(candidates[best][1])
        chosen.append(best)
    return fits, chosen


def _merge_guarded(weight, mean, variance, target, precision_floor):
    weight = np.array(weight, dtype=np.float64)
    mean = np.array(mean, dtype=np.float64)
    variance = np.array(variance, dtype=np.float64)
    while len(weight) > target:
        total = weight[:, None] + weight[None, :]
        merged_mean = (weight[:, None] * mean[:, None]
                       + weight[None, :] * mean[None, :]) / total
        merged_variance = (
            weight[:, None] * (variance[:, None] + (mean[:, None] - merged_mean) ** 2)
            + weight[None, :] * (variance[None, :] + (mean[None, :] - merged_mean) ** 2)
        ) / total
        cost = 0.5 * (total * np.log(merged_variance)
                      - weight[:, None] * np.log(variance)[:, None]
                      - weight[None, :] * np.log(variance)[None, :])
        np.fill_diagonal(cost, np.inf)
        # The guard: never create a component the prior division would cancel.
        cost = np.where(1.0 / merged_variance < precision_floor, np.inf, cost)
        if not np.isfinite(cost).any():
            # Every admissible merge is forbidden; fall back to dropping, which
            # cannot invent mass even though it discards some.
            order = np.argsort(weight)[-target:]
            return weight[order], mean[order], variance[order]
        first, second = divmod(int(np.argmin(cost)), len(weight))
        keep = np.ones(len(weight), dtype=bool)
        keep[[first, second]] = False
        weight = np.concatenate([weight[keep], [total[first, second]]])
        mean = np.concatenate([mean[keep], [merged_mean[first, second]]])
        variance = np.concatenate([variance[keep], [merged_variance[first, second]]])
    return weight, mean, variance


def compose(fits, prior_precision, beam=DEFAULT_BEAM, margin=DEFAULT_MARGIN):
    """Guarded product of the per-observation mixtures, in natural parameters."""
    naturals = [to_natural(*fit) for fit in fits]
    n = len(naturals)
    floor = margin * (n - 1) * prior_precision
    precision, precision_mean, log_constant = np.zeros(1), np.zeros(1), np.zeros(1)
    for natural in naturals:
        precision = (precision[:, None] + natural[0][None, :]).reshape(-1)
        precision_mean = (precision_mean[:, None] + natural[1][None, :]).reshape(-1)
        log_constant = (log_constant[:, None] + natural[2][None, :]).reshape(-1)
        if len(precision) > beam:
            precision, precision_mean, log_constant = from_moments(
                *_merge_guarded(*to_moments(precision, precision_mean,
                                            log_constant), beam, floor)
            )
    return precision - (n - 1) * prior_precision, precision_mean, log_constant


def bagged_compose(fits, prior_precision, grid, beam=DEFAULT_BEAM,
                   margin=DEFAULT_MARGIN, replicates=32, seed=0):
    """Average the composition over bootstrap resamples of the observations.

    The measured high-n failure is not width and not K: it is the **branch
    weights collapsing**. The composed log-odds between two modes is a sum

        L = sum_j l_j

    of one term per observation, so a per-observation error of ``e`` becomes
    ``n e`` and by n = 64 that reliably sends a near-even posterior entirely onto
    one branch (inverse kinematics: mass 0.024 against an exact 0.50; sign model:
    0.126 against an exact 0.500).

    Resampling the observations with replacement and averaging the resulting
    *densities* attacks exactly that. Each replicate sees a different multiset of
    the ``l_j``, so L varies by roughly ``sqrt(n) sd(l)`` across replicates;
    averaging densities is a convex combination, so a replicate that collapses
    onto the wrong branch cannot cancel one that did not -- the average retains
    both. It is bagging, and its effect is confined to the quantity that is
    over-confident.

    Three properties make it worth doing here rather than something cleverer:

    * **It is free.** No network call: the per-observation mixtures are already
      fitted, and a replicate is a re-multiplication of the same K components.
    * **It stays in the model class.** A uniform average of Gaussian mixtures is
      a Gaussian mixture, so the result diffuses in closed form exactly as the
      unbagged composition does.
    * **It leaves the locals alone.** Only the shared marginal is bagged; the
      conditional stage is untouched, and the endpoint residual measures that.

    What it cannot do is remove a bias shared by every observation -- bootstrap
    resampling estimates variance, not bias. It shrinks confident collapse driven
    by how few observations actually carry the branch information; a systematic
    per-observation offset survives it.
    """
    grid = np.asarray(grid, dtype=np.float64)
    n = len(fits)
    # Each replicate is a product of the *same* n factors in a different
    # multiset, so tabulate each factor's log density once and let a replicate
    # be a sum of n rows. Composing in the mixture representation instead costs
    # a merge-pruned product per replicate -- 64 of those took 645 s at n = 64,
    # against milliseconds here. The output is a grid density rather than a
    # mixture, which the shared stage samples by inverse CDF anyway; what it
    # gives up is the closed-form diffusion of the unbagged composition.
    log_factors = np.empty((n, len(grid)), dtype=np.float64)
    for index, (weights, means, variances) in enumerate(fits):
        density = (weights[None, :] / np.sqrt(2.0 * np.pi * variances[None, :])
                   * np.exp(-0.5 * (grid[:, None] - means[None, :]) ** 2
                            / variances[None, :])).sum(axis=1)
        area = np.trapezoid(density, grid)
        log_factors[index] = np.log(np.maximum(density / max(area, 1e-300),
                                              1e-300))
    log_prior = -0.5 * (grid ** 2) * prior_precision
    log_prior -= np.log(np.trapezoid(np.exp(log_prior - log_prior.max()), grid))
    log_prior -= log_prior.max() - log_prior.max()

    rng = np.random.default_rng(seed)
    total = np.zeros(len(grid), dtype=np.float64)
    for _ in range(replicates):
        picked = rng.integers(0, n, size=n)
        log_density = log_factors[picked].sum(axis=0) - (n - 1) * log_prior
        log_density -= log_density.max()
        density = np.exp(log_density)
        total += density / max(np.trapezoid(density, grid), 1e-300)
    total = total / replicates
    return total / max(np.trapezoid(total, grid), 1e-300)


def infer(draws, grid, prior_std, k_max=DEFAULT_K_MAX, beam=DEFAULT_BEAM,
          margin=DEFAULT_MARGIN, restarts=1, seed=0, quantile=1e-3,
          device=None, bag=0):
    """Full shared-block inference. ``draws`` is (observations, samples)."""
    draws = np.asarray(draws, dtype=np.float64)
    prior_precision = 1.0 / float(prior_std) ** 2
    fits, chosen = choose_and_fit(draws, k_max, restarts, seed, device)
    precision, precision_mean, log_constant = compose(
        fits, prior_precision, beam, margin
    )
    if bag:
        density = bagged_compose(fits, prior_precision, grid, beam, margin,
                                 replicates=int(bag), seed=seed)
    else:
        density = density_on_grid(grid, precision, precision_mean, log_constant)

    lower = max(float(np.quantile(row, quantile)) for row in draws)
    upper = min(float(np.quantile(row, 1.0 - quantile)) for row in draws)
    inside = (grid >= lower) & (grid <= upper)
    total = float(np.trapezoid(density, grid))
    outside = max(0.0, 1.0 - float(np.trapezoid(density[inside], grid[inside])) / total)

    n = draws.shape[0]
    return density, {
        "chosen_k": chosen,
        "k_median": float(np.median(chosen)),
        "components": int(len(precision)),
        "cancellation_ratio": float(precision.min()
                                    / max((n - 1) * prior_precision, 1e-300)),
        "outside_support": outside,
        "support": [lower, upper],
    }


def sample_from(precision, precision_mean, log_constant, count, rng):
    """Draws from the composed mixture -- no grid, no inverse CDF."""
    from mixture_product import log_mass
    mass = log_mass(precision, precision_mean, log_constant)
    finite = np.isfinite(mass)
    weight = np.exp(mass[finite] - mass[finite].max())
    weight = weight / weight.sum()
    variance = 1.0 / precision[finite]
    mean = precision_mean[finite] * variance
    picked = rng.choice(len(weight), size=count, p=weight)
    return mean[picked] + np.sqrt(variance[picked]) * rng.standard_normal(count)
