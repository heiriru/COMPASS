"""The exponential hierarchy g -> l_j -> x_j, and every exact reference it admits.

The generative model
--------------------
    g        ~ Normal(mu_g, sigma_g^2)
    l_j | g  = g + eps_j,    eps_j ~ Exponential(rate)
    x_j | l_j = l_j + eta_j, eta_j ~ Exponential(rate)

One global parameter, one local parameter per observation, one observed value per
observation -- the (g, l, x) node layout every experiment in this directory uses.

What makes it worth measuring
-----------------------------
``x_j - g = eps_j + eta_j`` is a sum of two i.i.d. exponentials, so the local
parameter integrates out exactly and the tall posterior is closed form:

    p(g | x_1..N) prop exp[-(g - mu_g)^2 / (2 sigma_g^2)] exp(N rate g)
                       prod_j (x_j - g) 1[g <= min_j x_j]

    d/dg log p(g | x) = -(g - mu_g)/sigma_g^2 + N rate - sum_j 1/(x_j - g)

The support has a **hard wall** at ``min_j x_j`` where the score diverges, and
``exp(N rate g)`` presses the mass right up against it, so the posterior is
sharply skewed and (at N = 30) roughly ``1/N`` wide. Nothing about it is Gaussian.

The joint score is degenerate -- and that is the point
------------------------------------------------------
    p(g, l_1..N | x) prop exp[-(g-mu_g)^2/(2 sigma_g^2)] exp(N rate g)
                          prod_j 1[g <= l_j <= x_j]

is **flat in every l_j**, and its g-score is the affine ``-(g-mu)/sigma^2 + N rate``
with *no* ``-sum_j 1/(x_j - g)`` term. All of the interesting structure lives in
the indicator, i.e. in the boundary, not in the gradient. So a comparison of
composed scores at t = 0 is vacuous here: every rule that is asymptotically
correct returns the same affine function.

At a finite noise level ``lam`` the boundary is smoothed into the gradient and
the comparison becomes sharp. Diffusing with the VESDE kernel ``N(., lam^2)``
(alpha_t = 1), the local integrals still factor given ``g``:

    p_lam(g_t, l_t | x) prop int dg exp[-(g-mu)^2/(2 sigma^2) + N rate g]
                             N(g_t; g, lam^2)
                             prod_j [Phi((x_j - l_jt)/lam) - Phi((g - l_jt)/lam)]

a **one-dimensional quadrature**, so the exact diffused joint score is available
at every noise level the sampler visits. That is the ground truth this directory
grades composition rules against; see :func:`diffused_score`.

Local references
----------------
``p(l_j | g, x_j) = Uniform(g, x_j)``: the exponentials cancel. It is flat, so
"the mode" means the maximizer of its ``lam``-smoothed version, which by symmetry
is the midpoint ``(g + x_j)/2`` -- also its mean, and what annealed score ascent
converges to. Marginalizing over the shared posterior,

    E[l_j | x_1..N]   = (E[g | x] + x_j) / 2
    Var[l_j | x_1..N] = E[(x_j - g)^2] / 12 + Var[g | x] / 4
"""
from __future__ import annotations

import math

import numpy as np
import torch

MU_G, SIGMA_G, RATE = 0.0, 1.0, 1.0

# Marginal moments of l = g + Exponential(rate). The sampler wants a Gaussian
# stand-in for p(l) when it clamps denoised local predictions; these are that
# distribution's true mean and standard deviation.
LOCAL_PRIOR_MEAN = MU_G + 1.0 / RATE
LOCAL_PRIOR_STD = math.sqrt(SIGMA_G**2 + 1.0 / RATE**2)

NODES = 3
GLOBAL_INDEX, LOCAL_INDEX, OBSERVED_INDEX = 0, 1, 2
CONDITION_MASK = torch.tensor([0.0, 0.0, 1.0])
HIERARCHY = [GLOBAL_INDEX]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def simulate(count, generator=None):
    """Forward draws of (g, l, x), shaped for SBIm.train's theta/x split."""
    g = MU_G + SIGMA_G * torch.randn(count, 1, generator=generator)
    epsilon = -torch.log(torch.rand(count, 1, generator=generator)) / RATE
    eta = -torch.log(torch.rand(count, 1, generator=generator)) / RATE
    local = g + epsilon
    return torch.cat([g, local], dim=1), local + eta


def observations(count, seed):
    """One dataset: a single true g, its locals, and the observed x_j."""
    generator = torch.Generator().manual_seed(seed)
    g = float(MU_G + SIGMA_G * torch.randn((), generator=generator))
    epsilon = -np.log(torch.rand(count, generator=generator).numpy()) / RATE
    eta = -np.log(torch.rand(count, generator=generator).numpy()) / RATE
    local = g + epsilon
    return g, local, local + eta


# ---------------------------------------------------------------------------
# The exact tall posterior over the shared parameter
# ---------------------------------------------------------------------------

def log_shared_posterior(x, grid):
    """log p(g | x_1..N) on a grid, up to a constant. -inf past the wall."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64)
    n = len(x)
    gap = x[None, :] - grid[:, None]
    inside = np.all(gap > 0.0, axis=1)
    log_density = np.full(grid.shape, -np.inf)
    log_density[inside] = (
        -0.5 * ((grid[inside] - MU_G) / SIGMA_G) ** 2
        + n * RATE * grid[inside]
        + np.log(gap[inside]).sum(axis=1)
    )
    return log_density


def shared_reference(x, points=40001, span=8.0):
    """Normalized p(g | x_1..N) on a grid, with its mean, std and mode.

    Returns ``(grid, density, weights, mean, std)`` where ``weights`` sum to one
    and ``density`` is the same thing divided by the grid spacing.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    upper = x.min()
    grid = np.linspace(upper - span, upper, points)
    log_density = log_shared_posterior(x, grid)
    weights = np.exp(log_density - log_density.max())
    weights /= weights.sum()
    mean = float((weights * grid).sum())
    std = float(max((weights * grid**2).sum() - mean**2, 0.0) ** 0.5)
    spacing = float(grid[1] - grid[0])
    return grid, weights / spacing, weights, mean, std


def exact_global_score(g, x):
    """d/dg log p(g | x_1..N), the closed form this experiment grades against.

    ``-inf`` outside the support, since the density vanishes there.
    """
    g = np.asarray(g, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    gap = x[None, :] - g[..., None]
    score = (
        -(g - MU_G) / SIGMA_G**2
        + len(x) * RATE
        - np.where(gap > 0.0, 1.0 / np.where(gap > 0.0, gap, 1.0), 0.0).sum(axis=-1)
    )
    return np.where(np.all(gap > 0.0, axis=-1), score, -np.inf)


def grid_mode(grid, density):
    """Continuous argmax of a density tabulated on a grid (parabolic refine)."""
    peak = int(np.argmax(density))
    if 0 < peak < len(grid) - 1:
        left, centre, right = density[peak - 1], density[peak], density[peak + 1]
        curvature = left - 2 * centre + right
        if curvature < 0:
            return float(grid[peak] + 0.5 * (left - right) / curvature
                         * (grid[1] - grid[0]))
    return float(grid[peak])


# ---------------------------------------------------------------------------
# Local references
# ---------------------------------------------------------------------------

def local_reference(x, grid, weights):
    """E[l_j | x_1..N] and its std, by quadrature over the shared posterior."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mean_g = float((weights * grid).sum())
    variance_g = float(max((weights * grid**2).sum() - mean_g**2, 0.0))
    gap_squared = np.array([
        float((weights * (value - grid) ** 2).sum()) for value in x
    ])
    return 0.5 * (mean_g + x), np.sqrt(gap_squared / 12.0 + variance_g / 4.0)


def conditional_local(x, g):
    """Mode and std of p(l_j | g, x_j) = Uniform(g, x_j).

    Flat, so "mode" is the maximizer of the lam-smoothed density: the midpoint.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    width = np.maximum(x - g, 0.0)
    return 0.5 * (g + x), width / math.sqrt(12.0)


def local_marginal(value, shared_grid, weights, local_grid):
    """p(l_j | x_1..N) on ``local_grid``, from the tall shared posterior.

    ``l_j | g ~ Uniform(g, x_j)``, so for each ``l`` the density is the shared
    mass below it with each unit weighted by ``1 / (x_j - g)``. The local grid
    is separate from the shared one because ``l_j`` reaches up to ``x_j``, which
    for every observation but the minimizing one lies above the shared support.
    Returned as weights summing to one, matching :func:`shared_reference`.
    """
    contribution = np.where(shared_grid < value,
                            weights / np.maximum(value - shared_grid, 1e-12), 0.0)
    cumulative = np.cumsum(contribution)
    density = np.zeros_like(local_grid)
    inside = local_grid < value
    indices = np.searchsorted(shared_grid, local_grid[inside], side="right") - 1
    density[inside] = np.where(indices >= 0, cumulative[np.clip(indices, 0, None)], 0.0)
    total = density.sum()
    return density / total if total > 0 else density


def single_observation_reference(x, points=20001, span=8.0):
    """Grids and normalized weights for p(g | x) and p(l | x), one observation.

    ``p(l | x) = int p(g | x) Uniform(l; g, x) dg``: for each ``l``, the shared
    mass below it, each unit weighted by ``1 / (x - g)``.
    """
    grid_g = np.linspace(x - span, x, points)
    log_density = log_shared_posterior([x], grid_g)
    weights = np.exp(log_density - log_density.max())
    weights /= weights.sum()

    grid_l = np.linspace(x - span, x, points)
    density = np.zeros_like(grid_l)
    contribution = weights / np.maximum(x - grid_g, 1e-12)
    cumulative = np.cumsum(contribution)
    inside = grid_l < x
    indices = np.searchsorted(grid_g, grid_l[inside], side="right") - 1
    density[inside] = np.where(indices >= 0, cumulative[np.clip(indices, 0, None)], 0.0)
    total = density.sum()
    return grid_g, weights, grid_l, (density / total if total > 0 else density)


# ---------------------------------------------------------------------------
# The exact *diffused* joint score -- the ground truth for composition rules
# ---------------------------------------------------------------------------

def _log_gaussian_interval(lower, upper):
    """log[Phi(upper) - Phi(lower)] for lower <= upper, without cancellation.

    Both tails are handled by factoring out the larger term, so the interval
    probability stays accurate when both endpoints sit deep in the same tail --
    which is exactly what happens as ``lam -> 0`` and the wall sharpens.
    """
    log_ndtr = torch.special.log_ndtr
    # Central case: the interval straddles zero, so a plain difference is fine.
    central = torch.log(
        (torch.special.ndtr(upper) - torch.special.ndtr(lower)).clamp_min(1e-300)
    )
    # Upper tail: Phi(u) - Phi(l) = Q(l) [1 - Q(u)/Q(l)], Q(z) = Phi(-z).
    upper_tail = log_ndtr(-lower) + torch.log1p(
        -torch.exp((log_ndtr(-upper) - log_ndtr(-lower)).clamp(max=-1e-12))
    )
    # Lower tail: Phi(u) - Phi(l) = Phi(u) [1 - Phi(l)/Phi(u)].
    lower_tail = log_ndtr(upper) + torch.log1p(
        -torch.exp((log_ndtr(lower) - log_ndtr(upper)).clamp(max=-1e-12))
    )
    result = torch.where(lower >= 0.0, upper_tail, central)
    return torch.where(upper <= 0.0, lower_tail, result)


def quadrature_grid(x, points=8001, span=8.0, device="cpu"):
    """A ``g``-grid for the diffused quadrature, packed where the mass is.

    ``p(g | x)`` is pressed against the wall at ``min_j x_j`` and is ~1/N wide,
    so a uniform grid over the prior would spend every point where the density
    is zero. This trims to the support of the *clean* tall posterior at double
    precision and grids that.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    upper = float(x.min())
    coarse = np.linspace(upper - span, upper, 20001)
    log_density = log_shared_posterior(x, coarse)
    weights = np.exp(log_density - log_density.max())
    keep = np.flatnonzero(weights > 1e-14)
    lower = float(coarse[keep[0]]) if len(keep) else upper - span
    # A little slack below, and stop just short of the wall where log(0) lives.
    grid = np.linspace(lower - 0.5 * (upper - lower), upper, points + 1)[:-1]
    return torch.as_tensor(grid, dtype=torch.float64, device=device)


def diffused_score(shared, local, x, lam, grid, chunk=16):
    """Exact score of the diffused joint p_lam(g_t, l_1t..l_Nt | x_1..N).

    Args:
        shared: (S,) diffused shared states g_t.
        local:  (S, N) diffused local states l_jt.
        x:      (N,) observations.
        lam:    scalar noise level of the VESDE kernel N(., lam^2).
        grid:   (G,) quadrature nodes in g, from :func:`quadrature_grid`.
        chunk:  states evaluated at once; the working set is chunk x G x N.

    Returns:
        ``(score_shared, score_local)`` of shapes (S,) and (S, N), the exact
        gradient of ``log p_lam`` with respect to ``g_t`` and each ``l_jt``.

    The shared component follows from differentiating the Gaussian kernel under
    the integral -- ``E_w[(g - g_t)] / lam^2`` with ``w`` the posterior over the
    quadrature node -- and the local component from differentiating the interval
    probability ``Phi((x_j - l_jt)/lam) - Phi((g - l_jt)/lam)``.
    """
    device = grid.device
    shared = torch.as_tensor(shared, dtype=torch.float64, device=device).reshape(-1)
    local = torch.as_tensor(local, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1)
    lam = float(lam)
    n = int(x.numel())

    # The g-only part of the integrand: the prior, the exp(N rate g) tilt.
    prior_term = (
        -0.5 * ((grid - MU_G) / SIGMA_G) ** 2 + n * RATE * grid
    )

    out_shared = torch.empty(shared.shape, dtype=torch.float64, device=device)
    out_local = torch.empty(local.shape, dtype=torch.float64, device=device)
    for start in range(0, shared.numel(), chunk):
        stop = min(start + chunk, shared.numel())
        g_t = shared[start:stop]                              # (B,)
        l_t = local[start:stop]                               # (B, N)

        lower = (grid[None, :, None] - l_t[:, None, :]) / lam   # (B, G, N)
        upper = ((x[None, None, :] - l_t[:, None, :]) / lam).expand_as(lower)
        log_interval = _log_gaussian_interval(lower, upper)

        log_weight = (
            prior_term[None, :]
            - 0.5 * ((g_t[:, None] - grid[None, :]) / lam) ** 2
            + log_interval.sum(dim=-1)
        )
        weight = torch.softmax(log_weight, dim=1)             # (B, G)

        out_shared[start:stop] = (
            weight * (grid[None, :] - g_t[:, None])
        ).sum(dim=1) / lam**2

        # d/dl_j log[Phi(upper) - Phi(lower)] = [phi(lower) - phi(upper)] / (lam * D)
        log_phi_lower = -0.5 * lower**2 - _LOG_SQRT_2PI
        log_phi_upper = -0.5 * upper**2 - _LOG_SQRT_2PI
        derivative = (
            torch.exp(log_phi_lower - log_interval)
            - torch.exp(log_phi_upper - log_interval)
        ) / lam
        out_local[start:stop] = (weight[:, :, None] * derivative).sum(dim=1)
    return out_shared, out_local


def sample_diffused(x, lam, count, seed, points=40001, span=8.0):
    """Exact draws from the diffused joint p_lam(g_t, l_t | x_1..N).

    Ancestral and exact: ``g`` from the tall posterior by inverse CDF on its
    grid, ``l_j | g ~ Uniform(g, x_j)``, then one VESDE kernel step on both.
    These are the states the sampler actually visits at noise level ``lam``, so
    a score error measured here is the error that steers the trajectory.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    grid, _, weights, _, _ = shared_reference(x, points=points, span=span)
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    g = np.interp(rng.random(count), cumulative, grid)
    local = g[:, None] + rng.random((count, len(x))) * (x[None, :] - g[:, None])
    return (g + lam * rng.standard_normal(count),
            local + lam * rng.standard_normal(local.shape),
            g, local)
