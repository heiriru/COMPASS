"""The *symmetric non-Gaussian* twin of ``hierarchy.py``: Laplace noise.

    g        ~ Normal(mu_g, sigma_g^2)                  mu_g = 0, sigma_g = 1
    l_j | g  = g + eps_j,     eps_j ~ Laplace(0, b)     b = 1/sqrt(2)
    x_j | l_j = l_j + eta_j,  eta_j ~ Laplace(0, b)

``b = 1/sqrt(2)`` gives ``Var(eps) = Var(eta) = 2 b^2 = 1``, matching
``Exponential(rate = 1)`` exactly, so W1/sigma numbers are readable against the
exponential table.

Why Laplace and not Gaussian
----------------------------
The Laplace law is the *symmetrized exponential*: same tail shape, same scale
family, mirrored. Swapping Exponential for Laplace changes the skew and removes
the hard wall while keeping the distribution non-Gaussian -- so the Gaussian /
Tweedie-second-order approximation every composition rule in ``MultiObsSampler``
rests on stays an approximation. ``hierarchy_gauss.py`` removes non-Gaussianity
too, which makes the rules exact by construction; running both is what separates
"the rules struggled with the skew and the wall" from "the rules struggled with
non-Gaussianity".

Everything remains exactly solvable
-----------------------------------
``x_j - g = eps_j + eta_j`` is a sum of two i.i.d. Laplace variables, with the
closed-form density ``f(z) = (1/4b) e^{-|z|/b} (1 + |z|/b)``, so the tall
posterior integrates the locals out exactly:

    log p(g | x) = -(g - mu_g)^2/(2 sigma_g^2)
                   + sum_j [ -|x_j - g|/b + log(1 + |x_j - g|/b) ]

    d/dg log p(g | x) = -(g - mu_g)/sigma_g^2
                        + sum_j sign(x_j - g) |u_j| / (b (1 + |u_j|)),
                        u_j = (x_j - g)/b

Unlike the exponential hierarchy this score is bounded everywhere -- no wall, no
divergence -- and it vanishes smoothly at ``g = x_j``.

The local conditional is the direct analogue of ``Uniform(g, x_j)``
--------------------------------------------------------------------
    p(l | g, x) prop exp[-(|l - g| + |x - l|)/b]

is **exactly flat on the interval between g and x** (the two absolute values sum
to the constant ``w = |x - g|`` there) and decays as ``e^{-2 d/b}`` outside, with
``d`` the distance to the nearer edge. So it is a plateau with exponential
shoulders, symmetric about the midpoint: mode = mean = ``(g + x)/2``, exactly as
in the exponential hierarchy. Its normalizer is ``e^{-w/b} (w + b)``, whose
``g``-dependence reproduces ``f(w)`` -- the consistency check that ties the local
conditional to the tall posterior above.

The diffused local integral, in closed form
-------------------------------------------
Under the VESDE kernel ``N(., lam^2)`` the per-observation integral splits into a
plateau term and two exponential shoulders, all of which convolve with a Gaussian
analytically. With ``kappa = 2/b`` and edges ``a_lo <= a_hi`` (the sorted pair
``g, x``):

    log P = log[Phi((a_hi - l_t)/lam) - Phi((a_lo - l_t)/lam)]
    log L = kappa (l_t - a_lo) + kappa^2 lam^2/2 + log Phi((a_lo - l_t - kappa lam^2)/lam)
    log R = -kappa (l_t - a_hi) + kappa^2 lam^2/2 + log Phi((l_t - a_hi - kappa lam^2)/lam)
    log K = -w/b + logsumexp(log P, log L, log R)

and -- after the ``exp(A) phi(a) = phi(v)`` cancellations that make the shoulder
derivatives collapse onto the plateau's -- the local score is simply

    d/dl_t log K = kappa (L - R) / (P + L + R)

i.e. a softmax-weighted difference of the two shoulders. The shared score is the
same quadrature form the exponential hierarchy uses,
``E_w[(g - g_t)] / lam^2``, over a one-dimensional grid in ``g``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hierarchy import grid_mode  # noqa: F401  (pure grid utility, re-exported)
from hierarchy import _LOG_SQRT_2PI, _log_gaussian_interval  # noqa: F401

MU_G, SIGMA_G = 0.0, 1.0
# Laplace scale; Var = 2 b^2 = 1 matches Exponential(rate = 1).
SCALE = 1.0 / math.sqrt(2.0)
# Decay rate of the shoulders of p(l | g, x): the two absolute values add.
KAPPA = 2.0 / SCALE

LOCAL_PRIOR_MEAN = MU_G
LOCAL_PRIOR_STD = math.sqrt(SIGMA_G**2 + 2.0 * SCALE**2)

NODES = 3
GLOBAL_INDEX, LOCAL_INDEX, OBSERVED_INDEX = 0, 1, 2
CONDITION_MASK = torch.tensor([0.0, 0.0, 1.0])
HIERARCHY = [GLOBAL_INDEX]


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def _laplace(shape, generator=None):
    """Laplace(0, SCALE) as the difference of two exponentials."""
    uniform = torch.rand(shape, generator=generator)
    return SCALE * torch.sign(uniform - 0.5) * torch.log1p(
        -2.0 * torch.abs(uniform - 0.5)
    ) * -1.0


def simulate(count, generator=None):
    """Forward draws of (g, l, x), shaped for SBIm.train's theta/x split."""
    g = MU_G + SIGMA_G * torch.randn(count, 1, generator=generator)
    local = g + _laplace((count, 1), generator)
    return torch.cat([g, local], dim=1), local + _laplace((count, 1), generator)


def observations(count, seed):
    """One dataset: a single true g, its locals, and the observed x_j."""
    generator = torch.Generator().manual_seed(seed)
    g = float(MU_G + SIGMA_G * torch.randn((), generator=generator))
    local = g + _laplace((count,), generator).numpy()
    return g, local, local + _laplace((count,), generator).numpy()


# ---------------------------------------------------------------------------
# The exact tall posterior over the shared parameter
# ---------------------------------------------------------------------------

def log_shared_posterior(x, grid):
    """log p(g | x_1..N) on a grid, up to a constant.

    ``x_j - g`` is a sum of two i.i.d. Laplace variables, whose density is
    ``(1/4b) e^{-|z|/b} (1 + |z|/b)``.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64)
    gap = np.abs(x[None, :] - grid[:, None]) / SCALE
    return (
        -0.5 * ((grid - MU_G) / SIGMA_G) ** 2
        + (-gap + np.log1p(gap)).sum(axis=1)
    )


def _support(x, points=20001, floor=1e-14, pad=0.5):
    """Numeric support of the tall posterior: where its weight is not negligible.

    There is no wall here, so the bracket is found rather than known. The scan
    is deliberately wide (the prior and the data can disagree) and the kept
    interval is padded before it is used as a grid.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    low = min(float(x.min()), MU_G) - 10.0 * SIGMA_G
    high = max(float(x.max()), MU_G) + 10.0 * SIGMA_G
    coarse = np.linspace(low, high, points)
    weights = np.exp(log_shared_posterior(x, coarse)
                     - log_shared_posterior(x, coarse).max())
    keep = np.flatnonzero(weights > floor)
    lower, upper = float(coarse[keep[0]]), float(coarse[keep[-1]])
    width = max(upper - lower, 1e-6)
    return lower - pad * width, upper + pad * width


def shared_reference(x, points=40001, span=8.0):
    """Normalized p(g | x_1..N) on a grid, with its mean, std and mode."""
    lower, upper = _support(x)
    grid = np.linspace(lower, upper, points)
    log_density = log_shared_posterior(x, grid)
    weights = np.exp(log_density - log_density.max())
    weights /= weights.sum()
    mean = float((weights * grid).sum())
    std = float(max((weights * grid**2).sum() - mean**2, 0.0) ** 0.5)
    spacing = float(grid[1] - grid[0])
    return grid, weights / spacing, weights, mean, std


def exact_global_score(g, x):
    """d/dg log p(g | x_1..N). Bounded everywhere: no wall, no divergence."""
    g = np.asarray(g, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    difference = x[None, :] - g[..., None]
    gap = np.abs(difference) / SCALE
    return (
        -(g - MU_G) / SIGMA_G**2
        + (np.sign(difference) * gap / (SCALE * (1.0 + gap))).sum(axis=-1)
    )


# ---------------------------------------------------------------------------
# Local references
# ---------------------------------------------------------------------------

def _conditional_variance(width):
    """Var of the plateau-with-shoulders p(l | g, x), width ``w = |x - g|``.

    Half-width ``h = w/2``, shoulders of scale ``b/2``:
    ``Var = [2h^3/3 + b^3/2 + h b^2 + h^2 b] / (2h + b)``. Reduces to ``b^2/2``
    (a two-sided exponential) at ``h = 0`` and to ``h^2/3`` (uniform) as
    ``h -> inf``.
    """
    h = np.asarray(width, dtype=np.float64) / 2.0
    b = SCALE
    return (2.0 * h**3 / 3.0 + b**3 / 2.0 + h * b**2 + h**2 * b) / (2.0 * h + b)


def conditional_local(x, g):
    """Mode and std of p(l_j | g, x_j): symmetric, so mode = mean = midpoint."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return 0.5 * (g + x), np.sqrt(_conditional_variance(np.abs(x - g)))


def local_reference(x, grid, weights):
    """E[l_j | x_1..N] and its std, by quadrature over the shared posterior."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mean_g = float((weights * grid).sum())
    variance_g = float(max((weights * grid**2).sum() - mean_g**2, 0.0))
    # Law of total variance: E_g[Var(l | g, x)] + Var_g(midpoint).
    inner = np.array([
        float((weights * _conditional_variance(np.abs(value - grid))).sum())
        for value in x
    ])
    return 0.5 * (mean_g + x), np.sqrt(inner + variance_g / 4.0)


def _conditional_density(local_grid, g, value):
    """p(l | g, x) on a grid: flat between g and x, ``e^{-2d/b}`` outside."""
    low, high = np.minimum(g, value), np.maximum(g, value)
    distance = np.maximum(np.maximum(low - local_grid, local_grid - high), 0.0)
    return np.exp(-KAPPA * distance) / (high - low + SCALE)


def local_marginal(value, shared_grid, weights, local_grid, chunk=512):
    """p(l_j | x_1..N) on ``local_grid``, as weights summing to one.

    Marginalizes the conditional above over the shared posterior. Chunked over
    the shared grid: the full outer product would be ~10^9 entries.
    """
    value = float(value)
    density = np.zeros_like(np.asarray(local_grid, dtype=np.float64))
    for start in range(0, len(shared_grid), chunk):
        stop = min(start + chunk, len(shared_grid))
        block = _conditional_density(
            local_grid[None, :], shared_grid[start:stop, None], value
        )
        density += weights[start:stop] @ block
    total = density.sum()
    return density / total if total > 0 else density


def single_observation_reference(x, points=20001, span=8.0):
    """Grids and normalized weights for p(g | x) and p(l | x), one observation."""
    lower, upper = _support([x])
    grid_g = np.linspace(lower, upper, points)
    log_density = log_shared_posterior([x], grid_g)
    weights_g = np.exp(log_density - log_density.max())
    weights_g /= weights_g.sum()

    # The local reaches beyond the shared support by a few shoulder lengths.
    reach = 8.0 / KAPPA
    grid_l = np.linspace(min(lower, float(x)) - reach,
                         max(upper, float(x)) + reach, points)
    weights_l = local_marginal(x, grid_g, weights_g, grid_l)
    return grid_g, weights_g, grid_l, weights_l


# ---------------------------------------------------------------------------
# The exact *diffused* joint score -- ground truth for composition rules
# ---------------------------------------------------------------------------

def log_local_kernel(nodes, l_t, value, lam):
    """``log K(g, l_t)`` and ``d/dl_t log K``, the diffused local integral.

    ``K(g, l_t) = int dl exp[-(|l - g| + |x - l|)/b] N(l_t; l, lam^2)``: a
    plateau between ``g`` and ``x`` plus two exponential shoulders, each
    convolved with the VESDE kernel in closed form. Written in log space
    throughout -- ``kappa^2 lam^2 / 2`` reaches 60 at the top of the schedule, so
    the shoulder terms overflow if formed directly.

    Args:
        nodes: quadrature nodes in ``g``, broadcastable against ``l_t``.
        l_t:   diffused local state.
        value: the observation ``x``.
        lam:   noise level.

    Returns:
        ``(log_kernel, derivative)`` with ``derivative = d/dl_t log K``.
    """
    lam = float(lam)
    width = torch.abs(value - nodes)
    low = torch.minimum(nodes, value)
    high = torch.maximum(nodes, value)

    lower = (low - l_t) / lam
    upper = (high - l_t) / lam
    log_plateau = _log_gaussian_interval(lower, upper)

    shoulder = KAPPA**2 * lam**2 / 2.0
    log_left = (KAPPA * (l_t - low) + shoulder
                + torch.special.log_ndtr((low - l_t - KAPPA * lam**2) / lam))
    log_right = (-KAPPA * (l_t - high) + shoulder
                 + torch.special.log_ndtr((l_t - high - KAPPA * lam**2) / lam))

    stacked = torch.stack([log_plateau, log_left, log_right], dim=-1)
    total = torch.logsumexp(stacked, dim=-1)
    # d(P + L + R)/dl_t = kappa (L - R): the shoulders' Gaussian-density terms
    # cancel the plateau's exactly (see the module docstring).
    derivative = KAPPA * (torch.exp(log_left - total)
                          - torch.exp(log_right - total))
    return total - width / SCALE, derivative


def quadrature_grid(x, points=8001, span=8.0, device="cpu"):
    """A ``g``-grid for the diffused quadrature, over the numeric support."""
    lower, upper = _support(x)
    grid = np.linspace(lower, upper, points)
    return torch.as_tensor(grid, dtype=torch.float64, device=device)


def diffused_score(shared, local, x, lam, grid, chunk=16):
    """Exact score of the diffused joint p_lam(g_t, l_1t..l_Nt | x_1..N).

    Same structure as ``hierarchy.diffused_score``: a one-dimensional quadrature
    in ``g``, with the per-observation local integral supplied in closed form by
    :func:`log_local_kernel` instead of a truncated-Gaussian interval.
    """
    device = grid.device
    shared = torch.as_tensor(shared, dtype=torch.float64, device=device).reshape(-1)
    local = torch.as_tensor(local, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1)
    lam = float(lam)

    prior_term = -0.5 * ((grid - MU_G) / SIGMA_G) ** 2

    out_shared = torch.empty(shared.shape, dtype=torch.float64, device=device)
    out_local = torch.empty(local.shape, dtype=torch.float64, device=device)
    for start in range(0, shared.numel(), chunk):
        stop = min(start + chunk, shared.numel())
        g_t = shared[start:stop]                               # (B,)
        l_t = local[start:stop]                                # (B, N)

        log_kernel, derivative = log_local_kernel(
            grid[None, :, None], l_t[:, None, :], x[None, None, :], lam
        )                                                      # (B, G, N)

        log_weight = (
            prior_term[None, :]
            - 0.5 * ((g_t[:, None] - grid[None, :]) / lam) ** 2
            + log_kernel.sum(dim=-1)
        )
        weight = torch.softmax(log_weight, dim=1)              # (B, G)

        out_shared[start:stop] = (
            weight * (grid[None, :] - g_t[:, None])
        ).sum(dim=1) / lam**2
        out_local[start:stop] = (weight[:, :, None] * derivative).sum(dim=1)
    return out_shared, out_local


def sample_diffused(x, lam, count, seed, points=40001, span=8.0):
    """Exact draws from the diffused joint p_lam(g_t, l_t | x_1..N).

    Ancestral: ``g`` from the tall posterior by inverse CDF, then ``l_j | g``
    from the plateau-with-shoulders conditional by its own inverse CDF (choose
    plateau or shoulder by mass, then place within it), then one VESDE kernel
    step on both.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    grid, _, weights, _, _ = shared_reference(x, points=points, span=span)
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    g = np.interp(rng.random(count), cumulative, grid)

    low = np.minimum(g[:, None], x[None, :])
    high = np.maximum(g[:, None], x[None, :])
    width = high - low
    total = width + SCALE
    choice = rng.random((count, len(x))) * total
    tail = rng.exponential(1.0 / KAPPA, size=(count, len(x)))
    local = np.where(
        choice < SCALE / 2.0, low - tail,
        np.where(choice < SCALE / 2.0 + width,
                 low + (choice - SCALE / 2.0), high + tail),
    )
    return (g + lam * rng.standard_normal(count),
            local + lam * rng.standard_normal(local.shape),
            g, local)
