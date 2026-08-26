"""A genuinely generative non-Gaussian hierarchy, with a closed-form posterior.

    g       ~ Normal(mu_g, sigma_g^2)
    l_j | g = g + eps_j,   eps_j ~ Exponential(rate)
    x_j | l_j = l_j + eta_j,  eta_j ~ Exponential(rate)

so the hierarchy is g -> l_j -> x_j, with exactly one global parameter and one
local parameter per observation -- the same node layout (g, l, x) every other
experiment in this directory uses.

Why this model is worth training a network on
---------------------------------------------
The mixture problem of ``experiments.py`` is *specified* as a set of posteriors:
its per-observation shifts are drawn independently, so there is no data-generating
``g`` behind them and the "shared" parameter is shared only by stipulation. Here
everything is generated forward from one true ``g``, and the exact posterior is still
available in closed form -- so the figure can carry a true-parameter line and the
composed posterior can be judged on recovery, not only on arithmetic.

The exact posterior
-------------------
Because ``x_j - g = eps_j + eta_j`` is a sum of two i.i.d. exponentials, the local
parameter integrates out exactly:

    x_j - g | g  ~  Gamma(2, rate)
    p(x_j | g)   =  rate^2 (x_j - g) exp(-rate (x_j - g)) 1[g <= x_j]

and therefore

    p(g | x_1..N) prop exp(-(g - mu_g)^2 / (2 sigma_g^2)) exp(N rate g)
                       prod_j (x_j - g) 1[g <= min_j x_j].

Two features make this a hard, honest test for a diffusion-composition rule: the
support has a **hard boundary** at ``min_j x_j`` (the true score diverges there),
and the density is strongly **skewed**, with the ``exp(N rate g)`` factor pushing
up against a product that vanishes linearly at the wall.

The locals are exact too, and in an unusual way:

    p(l_j | g, x_j) prop rate e^{-rate(l_j - g)} · rate e^{-rate(x_j - l_j)}
                     = rate^2 e^{-rate(x_j - g)}      for g <= l_j <= x_j

i.e. **Uniform(g, x_j)** -- the exponentials cancel and the conditional is flat.
So conditioned on a fixed ``g``, the local posterior has no unique argmax; its
lambda-smoothed mode (what annealed score ascent actually converges to) is the
midpoint ``(g + x_j)/2`` by symmetry, which is also its mean. That is the
reference this module reports for the conditional stage, and it makes the flat
ridge an interesting stress case for stage 4 rather than a degenerate one.

Marginalizing over the shared posterior then gives, for each observation,

    E[l_j | x_1..N]   = (E[g | x_1..N] + x_j) / 2
    Var[l_j | x_1..N] = E[(x_j - g)^2] / 12 + Var[g | x_1..N] / 4

both by one-dimensional quadrature over the shared grid.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

MU_G, SIGMA_G, RATE = 0.0, 1.0, 1.0

# Marginal moments of the local parameter, l = g + Exponential(rate): the
# sampler wants a Gaussian stand-in for p(l) when it clamps denoised
# predictions, and these are that distribution's true mean and std.
LOCAL_PRIOR_MEAN = MU_G + 1.0 / RATE
LOCAL_PRIOR_STD = math.sqrt(SIGMA_G**2 + 1.0 / RATE**2)

MODEL_KWARGS = {
    "sde_type": "vesde", "sigma": 8.0, "hidden_size": 128,
    "depth": 6, "num_heads": 8, "mlp_ratio": 4,
}
TRAIN_SAMPLES = 200_000
VALIDATION_SAMPLES = 20_000
MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 512
LR = 3e-4


def simulate(count, generator=None):
    """Forward draws of (g, l, x), shaped for SBIm.train's theta/x split."""
    g = MU_G + SIGMA_G * torch.randn(count, 1, generator=generator)
    epsilon = torch.distributions.Exponential(RATE).sample((count, 1))
    eta = torch.distributions.Exponential(RATE).sample((count, 1))
    local = g + epsilon
    x = local + eta
    return torch.cat([g, local], dim=1), x


def observations(count, seed):
    """One dataset: a single true g, its locals, and the observed x_j."""
    generator = torch.Generator().manual_seed(seed)
    g = float(MU_G + SIGMA_G * torch.randn((), generator=generator))
    epsilon = -np.log(
        torch.rand(count, generator=generator).numpy()
    ) / RATE
    eta = -np.log(torch.rand(count, generator=generator).numpy()) / RATE
    local = g + epsilon
    return g, local, local + eta


def log_shared_posterior(x, grid):
    """log p(g | x_1..N) on a grid, up to a constant. -inf past the boundary."""
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
    """Normalized p(g | x_1..N) on a grid, with its mean, std and mode."""
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


def local_reference(x, grid, weights):
    """E[l_j | x_1..N] and its std, from the shared posterior on the grid.

    l_j | g, x_j is Uniform(g, x_j), so the conditional mean is (g + x_j)/2 and
    the conditional variance (x_j - g)^2 / 12; both are then averaged over the
    shared posterior by the law of total expectation/variance.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mean_g = float((weights * grid).sum())
    variance_g = float(max((weights * grid**2).sum() - mean_g**2, 0.0))
    mean = 0.5 * (mean_g + x)
    gap_squared = np.array([
        float((weights * (value - grid) ** 2).sum()) for value in x
    ])
    variance = gap_squared / 12.0 + variance_g / 4.0
    return mean, np.sqrt(variance)


def conditional_local(x, g):
    """Mode and std of p(l_j | g, x_j) = Uniform(g, x_j).

    The density is flat, so "mode" means the maximizer of its lambda-smoothed
    version -- the midpoint, by symmetry -- which is what annealed score ascent
    converges to and also the conditional mean.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    width = np.maximum(x - g, 0.0)
    return 0.5 * (g + x), width / math.sqrt(12.0)


def train_or_load(directory, device, seed=7, force=False, quick=False,
                  verbose=True):
    """Load the checkpoint for this generative model, training it if absent."""
    from compass import ScoreBasedInferenceModel as SBIm

    directory = Path(directory)
    checkpoint = directory / "Model_checkpoint.pt"
    if checkpoint.exists() and not force:
        print(f"Loading exponential-hierarchy checkpoint: {checkpoint}")
        return SBIm.load(str(checkpoint), device=device)

    train_samples = 4_000 if quick else TRAIN_SAMPLES
    validation_samples = 1_000 if quick else VALIDATION_SAMPLES
    max_epochs = 3 if quick else MAX_EPOCHS

    torch.manual_seed(seed)
    np.random.seed(seed)
    theta_train, x_train = simulate(train_samples)
    theta_validation, x_validation = simulate(validation_samples)
    model = SBIm(nodes_size=3, device=device, **MODEL_KWARGS)
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"Training exponential-hierarchy score model ({parameters:,} params) "
          f"on {train_samples:,} simulations, up to {max_epochs} epochs")
    directory.mkdir(parents=True, exist_ok=True)
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_validation,
        x_val=x_validation, batch_size=BATCH_SIZE, max_epochs=max_epochs,
        early_stopping_patience=PATIENCE, lr=LR, time_sampling="mixture",
        device=device, verbose=verbose, path=str(directory),
    )
    return SBIm.load(str(checkpoint), device=device)
