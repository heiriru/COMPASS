"""Exact references shared by both benchmarks.

Both ``sign_model`` and ``control_model`` reduce to the same object: a Gaussian
mixture over ``X = (g, l_1..l_n)`` whose components share one covariance,

    p(X | y) = sum_k w_k N(X; m_k, C),

with ``(w, m, C) = module.mixture(y)``. Everything a figure or a metric needs --
marginals, diffused scores, exact draws -- follows from that triple alone, so it
is written once here and parameterised by the module rather than duplicated.

The shared covariance is what makes the diffused case free: the VE kernel adds
``lam^2 I`` to every component at once, so

    p_lam(X | y) = sum_k w_k N(X; m_k, C + lam^2 I)
    grad log p_lam(X) = (C + lam^2 I)^{-1} (sum_k r_k(X) m_k - X)

with ``r_k`` the responsibilities. No quadrature, no Monte Carlo, at any noise
level -- which is exactly what makes these two problems usable as gold standards
for a composition rule.
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def component_density(grid, weights, means, variance):
    """A 1-D Gaussian mixture with a shared variance, evaluated on ``grid``."""
    grid = np.asarray(grid, dtype=np.float64)
    exponent = -0.5 * (grid[None, :] - np.asarray(means)[:, None]) ** 2 / variance
    density = (np.asarray(weights)[:, None] * np.exp(exponent)).sum(axis=0)
    return density / math.sqrt(TWO_PI * variance)


def marginal(module, y, coordinate):
    """``(weights, means, variance)`` of one coordinate of p(X | y).

    ``coordinate`` is 0 for the shared ``g`` and ``j + 1`` for local ``l_j``.
    """
    weights, means, covariance = module.mixture(y)
    return weights, means[:, coordinate], float(covariance[coordinate, coordinate])


def moments(weights, means, variance):
    """Mean and standard deviation of a 1-D shared-variance mixture."""
    mean = float((np.asarray(weights) * np.asarray(means)).sum())
    second = float((np.asarray(weights) * (np.asarray(means) ** 2 + variance)).sum())
    return mean, math.sqrt(max(second - mean**2, 0.0))


def reference(module, y, coordinate, points=40001, span=6.0):
    """``(grid, density, normalized mass, mean, std)`` for one marginal.

    ``span`` is in standard deviations of the *whole* mixture, so the window
    always covers both modes and the valley between them.
    """
    weights, means, variance = marginal(module, y, coordinate)
    mean, std = moments(weights, means, variance)
    grid = np.linspace(mean - span * std, mean + span * std, points)
    density = component_density(grid, weights, means, variance)
    mass = density * float(grid[1] - grid[0])
    return grid, density, mass / mass.sum(), mean, std


def shared_reference(module, y, points=40001, span=6.0):
    """The exact marginal posterior over the shared coordinate."""
    return reference(module, y, 0, points=points, span=span)


def local_reference(module, y, index, points=20001, span=6.0):
    """The exact marginal posterior over local coordinate ``index``."""
    return reference(module, y, index + 1, points=points, span=span)


def aligned_mass(module, y):
    """Posterior mass on the fully *coherent* discrete configurations.

    Which components count as coherent is the module's own question -- for the
    sign model it is the two out of 2^n in which every row points the same way,
    for the control it is both of the two prior lobes -- so the module answers
    it with ``coherent_components``.
    """
    weights, _, _ = module.mixture(y)
    return float(weights[module.coherent_components(y)].sum())


def valley_depth(module, y):
    """``log p(g at the tallest mode) - log p(g = 0)``: how deep the gap is."""
    weights, means, variance = marginal(module, y, 0)
    peak_grid = np.array([float(means[int(np.argmax(weights))])])
    peak = component_density(peak_grid, weights, means, variance)[0]
    centre = component_density(np.array([0.0]), weights, means, variance)[0]
    return float(np.log(max(peak, 1e-300)) - np.log(max(centre, 1e-300)))


def mode_locations(module, y, points=40001, span=6.0, prominence=1e-3):
    """Prominent interior local maxima of the exact shared marginal, sorted.

    ``prominence`` is a fraction of the tallest peak. It is not cosmetic: in a
    deep-valley mixture the density underflows to float noise in the far tails,
    and plateaus there produce dozens of spurious "modes" if every local maximum
    is counted. The same threshold rule grades the sampled draws, so exact and
    recovered mode counts are comparable.
    """
    grid, density, _, _, _ = shared_reference(module, y, points=points, span=span)
    interior = (density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])
    positions, heights = grid[1:-1][interior], density[1:-1][interior]
    if len(heights) == 0:
        return positions
    return np.sort(positions[heights >= prominence * heights.max()])


def diffused_score(module, shared, local, y, lam):
    """Exact ``grad_X log p_lam(X | y)``, split into shared and local blocks."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    shared = np.asarray(shared, dtype=np.float64).reshape(-1)
    local = np.asarray(local, dtype=np.float64).reshape(len(shared), n)
    state = np.concatenate([shared[:, None], local], axis=1)

    weights, means, covariance = module.mixture(y)
    precision = np.linalg.inv(covariance + float(lam) ** 2 * np.eye(n + 1))

    residual = state[:, None, :] - means[None, :, :]
    quadratic = np.einsum("ski,ij,skj->sk", residual, precision, residual)
    log_responsibility = np.log(np.maximum(weights, 1e-300))[None, :] - 0.5 * quadratic
    log_responsibility -= log_responsibility.max(axis=1, keepdims=True)
    responsibility = np.exp(log_responsibility)
    responsibility /= responsibility.sum(axis=1, keepdims=True)

    score = (np.einsum("sk,ki->si", responsibility, means) - state) @ precision
    return score[:, 0], score[:, 1:]


def sample_diffused(module, y, lam, count, seed):
    """Exact draws from ``p_lam(X | y)``; also returns the clean draws."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    rng = np.random.default_rng(seed)
    weights, means, covariance = module.mixture(y)
    component = rng.choice(len(weights), size=count, p=weights)
    factor = np.linalg.cholesky(covariance)
    clean = means[component] + rng.standard_normal((count, n + 1)) @ factor.T
    noisy = clean + float(lam) * rng.standard_normal(clean.shape)
    return noisy[:, 0], noisy[:, 1:], clean[:, 0], clean[:, 1:]


def joint_draws(module, y, count, seed):
    """Exact draws from the clean joint posterior -- the reference sample."""
    _, _, shared, local = sample_diffused(module, y, 0.0, count, seed)
    return shared, local


def row_reference(module, value, points=20001, span=6.0):
    """``(grid_g, mass_g, grid_l, mass_l)`` for the one-observation posterior."""
    weights, means, covariance = module.row_mixture(value)
    out = []
    for axis in (0, 1):
        variance = float(covariance[axis, axis])
        mean, std = moments(weights, means[:, axis], variance)
        grid = np.linspace(mean - span * std, mean + span * std, points)
        density = component_density(grid, weights, means[:, axis], variance)
        mass = density * float(grid[1] - grid[0])
        out.extend([grid, mass / mass.sum()])
    return tuple(out)


def row_sample(module, value, count, rng):
    """Exact draws from the one-observation posterior p(g, l | y)."""
    weights, means, covariance = module.row_mixture(value)
    component = rng.choice(len(weights), size=count, p=np.asarray(weights))
    factor = np.linalg.cholesky(covariance)
    return means[component] + rng.standard_normal((count, 2)) @ factor.T


def wasserstein_1d(samples, grid, mass):
    """W1 between draws and a density tabulated as normalized ``mass``."""
    samples = np.sort(np.asarray(samples, dtype=np.float64))
    cumulative = np.cumsum(mass)
    cumulative /= cumulative[-1]
    quantiles = (np.arange(len(samples)) + 0.5) / len(samples)
    positions = np.clip(np.searchsorted(cumulative, quantiles), 0, len(grid) - 1)
    return float(np.abs(samples - grid[positions]).mean())
