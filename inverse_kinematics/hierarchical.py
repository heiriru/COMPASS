"""The tall-data version of the arm: ``angle_1`` shared, everything else local.

The single-arm benchmark in :mod:`simulator` has four private parameters. Here
``angle_1`` -- the shoulder joint -- is promoted to a **global** coordinate held
in common by every observation, while each observation keeps its own
``(height_arm, angle_2, angle_3)``. Physically: one shoulder setting, several
elbow/wrist configurations, several observed end positions.

That makes it a genuine compositional-inference problem of exactly the shape
``MultiObsSampler`` composes -- one shared block, one private block per
observation -- and, unlike the partial-pooling benchmark, one small enough to
carry an independent reference posterior.

Node layout
-----------
``COMPASS`` wants one flat vector per observation with the shared coordinates
first, then the row-local latents, then the conditioned observables::

    index   0        1            2         3         4     5
            angle_1  height_arm   angle_2   angle_3   x1    x2
            <shared> <------ row-local ------------>  <observed>

Note this is *not* :mod:`simulator`'s parameter order, which is
``(height_arm, angle_1, angle_2, angle_3)``. :func:`to_simulator_order` and
:func:`from_simulator_order` convert, and ``tests/test_hierarchical.py`` pins the
round trip -- a silent transposition here would train a network on a subtly
different forward model and nothing downstream would notice.

Why the network is trained on single rows
-----------------------------------------
Each training row is one ``(theta, x)`` pair drawn from the prior predictive with
its *own* ``angle_1``. Sharing ``angle_1`` is a property of the tall dataset the
composition rules are evaluated on, not of the training set -- the same split the
exponential bench documents. The network only ever learns the single-observation
joint; every cross-observation statement is the composition rule's doing.
"""

import numpy as np
import torch

import simulator as sim

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

GLOBAL_NAMES = ["angle_1"]
LOCAL_NAMES = ["height_arm", "angle_2", "angle_3"]
OBSERVED_NAMES = ["x1", "x2"]
NODE_NAMES = GLOBAL_NAMES + LOCAL_NAMES + OBSERVED_NAMES

NODES = len(NODE_NAMES)                       # 6
HIERARCHY = [0]                               # angle_1
LOCAL_INDICES = [1, 2, 3]
OBSERVED_INDICES = [4, 5]
LATENT_INDICES = HIERARCHY + LOCAL_INDICES

CONDITION_MASK = torch.tensor(
    [0.0] * len(LATENT_INDICES) + [1.0] * len(OBSERVED_INDICES)
)

# Prior scales in *node* order, read off ``simulator.PRIOR_SCALES`` rather than
# retyped, so the two modules cannot drift apart.
_SIM_INDEX_OF_NODE = [1, 0, 2, 3]             # angle_1, height, angle_2, angle_3
PRIOR_MEAN = np.zeros(4)
PRIOR_STD = sim.PRIOR_SCALES[_SIM_INDEX_OF_NODE].copy()

GLOBAL_PRIOR_MEAN = PRIOR_MEAN[:1]
GLOBAL_PRIOR_STD = PRIOR_STD[:1]
LOCAL_PRIOR_MEAN = PRIOR_MEAN[1:]
LOCAL_PRIOR_STD = PRIOR_STD[1:]


def to_simulator_order(theta):
    """Node order ``(angle_1, height, angle_2, angle_3)`` -> simulator order."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    out = np.empty_like(theta)
    out[:, 0] = theta[:, 1]      # height_arm
    out[:, 1] = theta[:, 0]      # angle_1
    out[:, 2] = theta[:, 2]
    out[:, 3] = theta[:, 3]
    return out


def from_simulator_order(theta):
    """Simulator order -> node order. The inverse of :func:`to_simulator_order`."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    out = np.empty_like(theta)
    out[:, 0] = theta[:, 1]      # angle_1
    out[:, 1] = theta[:, 0]      # height_arm
    out[:, 2] = theta[:, 2]
    out[:, 3] = theta[:, 3]
    return out


def forward(theta_nodes):
    """End positions for parameters given in *node* order."""
    return sim.forward(to_simulator_order(theta_nodes))


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------

def simulate(count, rng):
    """``count`` independent training rows as float32 ``(theta, x)`` tensors.

    A fresh ``angle_1`` per row, deliberately: see the module docstring.
    """
    theta = rng.normal(loc=PRIOR_MEAN, scale=PRIOR_STD, size=(int(count), 4))
    x = forward(theta)
    return (torch.as_tensor(theta, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32))


def simulate_dataset(observations, rng, angle_1=None):
    """One tall dataset: a single shared ``angle_1`` and ``n`` observations.

    Returns ``(angle_1, locals, x)`` with ``locals`` shaped ``(n, 3)`` and ``x``
    shaped ``(n, 2)``.
    """
    observations = int(observations)
    if angle_1 is None:
        angle_1 = float(rng.normal(0.0, GLOBAL_PRIOR_STD[0]))
    locals_ = rng.normal(
        loc=LOCAL_PRIOR_MEAN, scale=LOCAL_PRIOR_STD, size=(observations, 3)
    )
    theta = np.concatenate(
        [np.full((observations, 1), angle_1), locals_], axis=1
    )
    return float(angle_1), locals_, forward(theta)


# ---------------------------------------------------------------------------
# Reference posterior for the shared coordinate
# ---------------------------------------------------------------------------

def reference_global(x, grid=None, draws=40_000, bandwidth=0.05, seed=11,
                     chunk=None, budget=40_000_000, local_std=None,
                     global_std=None, symmetrize=False):
    """``p(angle_1 | x_1..n)`` on a grid, computed without any trained model.

    The forward map is deterministic, so ``p(x_j | angle_1)`` is the pushforward
    of the local prior through ``f(angle_1, .)`` -- a density on the plane with no
    closed form. It is estimated here by Monte Carlo with a Gaussian kernel of
    width ``bandwidth``:

        p_h(x_j | a) = E_{l ~ prior} N(x_j ; f(a, l), h^2 I)

    and the shared posterior follows from conditional independence given ``a``::

        log p(a | x_1..n) = log N(a; 0, sigma_a^2) + sum_j log p_h(x_j | a) + c

    This is an ABC posterior with a Gaussian kernel: exact as ``h -> 0``, and its
    bias at finite ``h`` is a *smoothing* of the likelihood, which is the same
    direction of error for every composition rule being graded. It involves no
    network, so it can referee them.

    Cost is ``len(grid) * draws * n`` kernel evaluations. The grid is chunked so
    the largest temporary stays near ``budget`` elements: that temporary is
    ``(chunk, draws, n)``, which at 200,000 draws and eight observations reaches
    1.6 GB for a chunk of 64 and dominates the runtime. ``chunk`` overrides the
    derived value if you need to pin it.

    Returns ``(grid, density)`` with ``density`` normalized to integrate to one.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if grid is None:
        span = 4.0 * GLOBAL_PRIOR_STD[0]
        grid = np.linspace(-span, span, 801)
    grid = np.asarray(grid, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # `local_std` / `global_std` let a caller vary the priors without editing the
    # module: the height prior is the only term that breaks the arm's reflection
    # symmetry, so widening it is how a bimodal shared posterior is produced (see
    # `bimodality_study.py`).
    local_scale = (LOCAL_PRIOR_STD if local_std is None
                   else np.broadcast_to(np.asarray(local_std, dtype=np.float64), (3,)))
    global_scale = float(GLOBAL_PRIOR_STD[0] if global_std is None else global_std)

    # One local sample set, reused at every grid point. Common random numbers:
    # the Monte Carlo error is then a smooth function of `a` rather than
    # independent noise per grid point, so the recovered curve is not jagged.
    locals_ = rng.normal(
        loc=LOCAL_PRIOR_MEAN, scale=local_scale, size=(int(draws), 3)
    )

    if chunk is None:
        chunk = max(1, int(budget // (len(locals_) * max(1, x.shape[0]))))
    log_likelihood = np.empty(len(grid))
    two_h2 = 2.0 * float(bandwidth) ** 2
    for start in range(0, len(grid), chunk):
        block = grid[start:start + chunk]
        theta = np.concatenate([
            np.repeat(block, len(locals_))[:, None],
            np.tile(locals_, (len(block), 1)),
        ], axis=1)
        predicted = forward(theta).reshape(len(block), len(locals_), 2)
        # (block, draws, n) squared distances, then log-mean-exp over draws.
        squared = ((predicted[:, :, None, :] - x[None, None, :, :]) ** 2).sum(-1)
        scaled = -squared / two_h2
        peak = scaled.max(axis=1, keepdims=True)
        log_kernel = (
            peak[:, 0, :]
            + np.log(np.exp(scaled - peak).mean(axis=1))
        )
        log_likelihood[start:start + chunk] = log_kernel.sum(axis=1)

    log_posterior = log_likelihood - 0.5 * (grid / global_scale) ** 2
    log_posterior -= log_posterior.max()
    density = np.exp(log_posterior)
    if symmetrize:
        # Only for data with x1 = 0 exactly, where the reflection
        # (a1,a2,a3,h) -> (-a1,-a2,-a3,-h) leaves the likelihood and every prior
        # invariant, so p(a | x) = p(-a | x) is a theorem rather than a hope.
        # Averaging the estimate with its mirror is then an exact-in-expectation
        # variance reduction: measured, the residual L1 asymmetry of the raw
        # estimate falls only as 1/sqrt(draws) -- 0.78 at 30k draws, 0.11 at
        # 480k -- which is enough to swamp the differences between composition
        # rules if left in.
        if not np.allclose(grid, -grid[::-1]):
            raise ValueError(
                "symmetrize=True needs a grid symmetric about zero."
            )
        density = 0.5 * (density + density[::-1])
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(
            "The reference posterior did not normalize; widen `bandwidth` or "
            "raise `draws` -- no local draw landed near an observation."
        )
    return grid, density / area


def grid_moments(grid, density):
    """Mean, standard deviation and mode of a density tabulated on a grid."""
    mean = float(np.trapezoid(grid * density, grid))
    variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    return mean, float(np.sqrt(max(variance, 0.0))), float(grid[np.argmax(density)])
