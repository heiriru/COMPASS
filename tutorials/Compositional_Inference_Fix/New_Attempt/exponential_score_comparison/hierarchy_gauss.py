"""The *Gaussian* twin of ``hierarchy.py``: g -> l_j -> x_j with symmetric noise.

    g        ~ Normal(mu_g, sigma_g^2)                mu_g = 0, sigma_g = 1
    l_j | g  = g + eps_j,     eps_j ~ Normal(0, s^2)  s = 1
    x_j | l_j = l_j + eta_j,  eta_j ~ Normal(0, s^2)

Same node layout, same API, same prior on ``g`` as ``hierarchy.py``; the only
change is the noise law. ``s = 1`` is chosen so ``Var(eps) = Var(eta) = 1``
exactly matches ``Exponential(rate = 1)``, which makes W1/sigma numbers readable
against the exponential table.

What this variant is *for*
--------------------------
It removes three things at once relative to the exponential hierarchy: the skew,
the hard wall at ``min_j x_j``, and non-Gaussianity. The last one matters most,
because every composition rule in ``MultiObsSampler`` is built on a
Gaussian/Tweedie approximation of the per-observation backward covariance. Here
that approximation is **exact at every noise level**, so ``gauss_jacobian`` and
``gauss_hierarchical`` should both be exactly right and any residual error is the
sampler, the pilot's Monte-Carlo noise, or the KDE. That makes this the control
that isolates sampler error -- and the reason it cannot, on its own, separate
"symmetric" from "Gaussian". ``hierarchy_laplace.py`` is the variant that can.

Everything is closed form
-------------------------
``(g, l_1..l_N)`` given ``x`` is jointly Gaussian with arrow precision

    Lambda_gg     = 1/sigma_g^2 + N/s^2
    Lambda_g,l_j  = -1/s^2
    Lambda_l_j,l_j = 2/s^2
    b_g = mu_g/sigma_g^2,   b_l_j = x_j/s^2,   m = Lambda^{-1} b

and the VESDE kernel just adds ``lam^2 I`` to its covariance, so the exact
diffused joint score is ``-(Sigma + lam^2 I)^{-1} (z - m)`` -- no quadrature
anywhere, unlike the exponential case. The tall marginal follows immediately:

    p(g | x) = Normal(mean_g, 1 / (1/sigma_g^2 + N/(2 s^2)))
    p(l_j | g, x_j) = Normal((g + x_j)/2, s^2/2)
    p(l_j | x)      = Normal((E[g|x] + x_j)/2, s^2/2 + Var[g|x]/4)

``p(l_j | g, x_j)`` being symmetric about the midpoint is the exact analogue of
the exponential hierarchy's ``Uniform(g, x_j)``: mode = mean = ``(g + x_j)/2``,
so stage 3 of the pipeline grades the same quantity in both problems.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hierarchy import grid_mode  # noqa: F401  (pure grid utility, re-exported)

MU_G, SIGMA_G = 0.0, 1.0
# Noise standard deviation on both levels. Var = 1 matches Exponential(rate=1).
NOISE_STD = 1.0

# Marginal moments of l = g + Normal(0, s^2); the sampler's Gaussian stand-in
# for p(l) when it clamps denoised local predictions.
LOCAL_PRIOR_MEAN = MU_G
LOCAL_PRIOR_STD = math.sqrt(SIGMA_G**2 + NOISE_STD**2)

NODES = 3
GLOBAL_INDEX, LOCAL_INDEX, OBSERVED_INDEX = 0, 1, 2
CONDITION_MASK = torch.tensor([0.0, 0.0, 1.0])
HIERARCHY = [GLOBAL_INDEX]


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def simulate(count, generator=None):
    """Forward draws of (g, l, x), shaped for SBIm.train's theta/x split."""
    g = MU_G + SIGMA_G * torch.randn(count, 1, generator=generator)
    local = g + NOISE_STD * torch.randn(count, 1, generator=generator)
    return (torch.cat([g, local], dim=1),
            local + NOISE_STD * torch.randn(count, 1, generator=generator))


def observations(count, seed):
    """One dataset: a single true g, its locals, and the observed x_j."""
    generator = torch.Generator().manual_seed(seed)
    g = float(MU_G + SIGMA_G * torch.randn((), generator=generator))
    local = g + NOISE_STD * torch.randn(count, generator=generator).numpy()
    return g, local, local + NOISE_STD * torch.randn(
        count, generator=generator
    ).numpy()


# ---------------------------------------------------------------------------
# The exact tall posterior over the shared parameter
# ---------------------------------------------------------------------------

def shared_moments(x):
    """``(mean, variance)`` of p(g | x_1..N); x_j | g ~ Normal(g, 2 s^2)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    precision = 1.0 / SIGMA_G**2 + len(x) / (2.0 * NOISE_STD**2)
    mean = (MU_G / SIGMA_G**2 + x.sum() / (2.0 * NOISE_STD**2)) / precision
    return float(mean), float(1.0 / precision)


def log_shared_posterior(x, grid):
    """log p(g | x_1..N) on a grid, up to a constant."""
    mean, variance = shared_moments(x)
    grid = np.asarray(grid, dtype=np.float64)
    return -0.5 * (grid - mean) ** 2 / variance


def shared_reference(x, points=40001, span=8.0):
    """Normalized p(g | x_1..N) on a grid, with its mean, std and mode.

    ``span`` is in posterior standard deviations here rather than in absolute
    units: this posterior is unbounded and narrow, so a fixed-width window would
    either clip it or waste every grid point.
    """
    mean, variance = shared_moments(x)
    sd = math.sqrt(variance)
    grid = np.linspace(mean - span * sd, mean + span * sd, points)
    weights = np.exp(-0.5 * (grid - mean) ** 2 / variance)
    weights /= weights.sum()
    spacing = float(grid[1] - grid[0])
    empirical_mean = float((weights * grid).sum())
    empirical_std = float(
        max((weights * grid**2).sum() - empirical_mean**2, 0.0) ** 0.5
    )
    return grid, weights / spacing, weights, empirical_mean, empirical_std


def exact_global_score(g, x):
    """d/dg log p(g | x_1..N): linear, with no wall and no divergence."""
    mean, variance = shared_moments(x)
    return -(np.asarray(g, dtype=np.float64) - mean) / variance


# ---------------------------------------------------------------------------
# Local references
# ---------------------------------------------------------------------------

def conditional_local(x, g):
    """Mode and std of p(l_j | g, x_j) = Normal((g + x_j)/2, s^2/2).

    Symmetric, so mode = mean = midpoint -- the same quantity the exponential
    hierarchy's Uniform(g, x_j) hands stage 3.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return 0.5 * (g + x), np.full(x.shape, NOISE_STD / math.sqrt(2.0))


def local_reference(x, grid, weights):
    """E[l_j | x_1..N] and its std, in closed form."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mean_g = float((weights * grid).sum())
    variance_g = float(max((weights * grid**2).sum() - mean_g**2, 0.0))
    variance = NOISE_STD**2 / 2.0 + variance_g / 4.0
    return 0.5 * (mean_g + x), np.full(x.shape, math.sqrt(variance))


def local_marginal(value, shared_grid, weights, local_grid):
    """p(l_j | x_1..N) on ``local_grid``, as weights summing to one.

    Gaussian in closed form, so unlike the exponential case this needs no
    marginalization over the shared grid; the arguments are kept for API parity.
    """
    mean_g = float((weights * shared_grid).sum())
    variance_g = float(max((weights * shared_grid**2).sum() - mean_g**2, 0.0))
    mean = 0.5 * (mean_g + float(value))
    variance = NOISE_STD**2 / 2.0 + variance_g / 4.0
    density = np.exp(-0.5 * (np.asarray(local_grid) - mean) ** 2 / variance)
    total = density.sum()
    return density / total if total > 0 else density


def single_observation_reference(x, points=20001, span=8.0):
    """Grids and normalized weights for p(g | x) and p(l | x), one observation."""
    mean_g, variance_g = shared_moments([x])
    sd_g = math.sqrt(variance_g)
    grid_g = np.linspace(mean_g - span * sd_g, mean_g + span * sd_g, points)
    weights_g = np.exp(-0.5 * (grid_g - mean_g) ** 2 / variance_g)
    weights_g /= weights_g.sum()

    mean_l = 0.5 * (mean_g + float(x))
    variance_l = NOISE_STD**2 / 2.0 + variance_g / 4.0
    sd_l = math.sqrt(variance_l)
    grid_l = np.linspace(mean_l - span * sd_l, mean_l + span * sd_l, points)
    weights_l = np.exp(-0.5 * (grid_l - mean_l) ** 2 / variance_l)
    weights_l /= weights_l.sum()
    return grid_g, weights_g, grid_l, weights_l


# ---------------------------------------------------------------------------
# The exact *diffused* joint score -- ground truth for composition rules
# ---------------------------------------------------------------------------

def joint_moments(x):
    """Mean and covariance of p(g, l_1..l_N | x_1..N); the arrow Gaussian."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    inverse_noise = 1.0 / NOISE_STD**2
    precision = np.zeros((n + 1, n + 1))
    precision[0, 0] = 1.0 / SIGMA_G**2 + n * inverse_noise
    precision[0, 1:] = -inverse_noise
    precision[1:, 0] = -inverse_noise
    precision[np.arange(1, n + 1), np.arange(1, n + 1)] = 2.0 * inverse_noise
    linear = np.concatenate([[MU_G / SIGMA_G**2], x * inverse_noise])
    covariance = np.linalg.inv(precision)
    return covariance @ linear, covariance


def quadrature_grid(x, points=8001, span=8.0, device="cpu"):
    """A ``g``-grid covering p(g | x). Kept for API parity with hierarchy.py."""
    mean, variance = shared_moments(x)
    sd = math.sqrt(variance)
    grid = np.linspace(mean - span * sd, mean + span * sd, points)
    return torch.as_tensor(grid, dtype=torch.float64, device=device)


def diffused_score(shared, local, x, lam, grid=None, chunk=16):
    """Exact score of p_lam(g_t, l_1t..l_Nt | x_1..N).

    Closed form: the joint is Gaussian, the VESDE kernel adds ``lam^2 I`` to its
    covariance, so the score is ``-(Sigma + lam^2 I)^{-1} (z - m)``. ``grid`` and
    ``chunk`` are accepted and ignored -- no quadrature is needed here.
    """
    mean, covariance = joint_moments(x)
    n = len(np.asarray(x, dtype=np.float64).reshape(-1))
    shared = np.asarray(shared, dtype=np.float64).reshape(-1)
    local = np.asarray(local, dtype=np.float64).reshape(len(shared), n)
    state = np.concatenate([shared[:, None], local], axis=1)

    precision = np.linalg.inv(covariance + float(lam) ** 2 * np.eye(n + 1))
    score = -(state - mean[None, :]) @ precision
    device = grid.device if isinstance(grid, torch.Tensor) else "cpu"
    return (torch.as_tensor(score[:, 0], dtype=torch.float64, device=device),
            torch.as_tensor(score[:, 1:], dtype=torch.float64, device=device))


def sample_diffused(x, lam, count, seed, points=40001, span=8.0):
    """Exact draws from the diffused joint p_lam(g_t, l_t | x_1..N)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    mean_g, variance_g = shared_moments(x)
    g = mean_g + math.sqrt(variance_g) * rng.standard_normal(count)
    local = 0.5 * (g[:, None] + x[None, :]) + (
        NOISE_STD / math.sqrt(2.0)
    ) * rng.standard_normal((count, len(x)))
    return (g + lam * rng.standard_normal(count),
            local + lam * rng.standard_normal(local.shape),
            g, local)
