"""The sign-ambiguous hierarchy: bimodality *constructed* by composition.

    g          ~ Normal(0, sigma_g^2)
    l_j | g    ~ Normal(g, tau^2)
    s_j        ~ Uniform{-1, +1}                      (never observed)
    y_j | l_j  ~ Normal(s_j l_j, sigma_y^2)

Integrating the sign out,

    p(y_j | l_j) = 1/2 N(y_j; l_j, sigma_y^2) + 1/2 N(y_j; -l_j, sigma_y^2)

so a single observation fixes the *magnitude* of l_j and says nothing about its
sign. Every row posterior p(g, l_j | y_j) is therefore an exactly symmetric
two-component Gaussian mixture: "both positive" or "both negative", equally
weighted, with the same covariance.

Why this is the composition test
--------------------------------
Each row is ambiguous in the same way, but the hierarchy l_j ~ Normal(g, tau^2)
couples the signs ferromagnetically: after integrating g out, the probability of
a sign configuration s is

    w_s  ~  exp[ sigma_g^2 (sum_j s_j y_j)^2 / (2 v (v + n sigma_g^2)) ],
    v = tau^2 + sigma_y^2,

which is maximised by the two *aligned* configurations (+,...,+) and (-,...,-).
So n locally two-moded factors have to be composed into one globally two-moded
posterior with a deep valley at zero -- not into the 2^n-moded product a naive
composition would give. That is the thing a compositional score rule has to get
right, and it is a strictly harder question than "can the sampler preserve a
bimodality that was already in the prior" (see ``control_model.py``).

Everything is closed form
-------------------------
Conditional on a sign configuration s the whole model is linear Gaussian, with
precision (the usual arrow matrix)

    Lambda_gg     = 1/sigma_g^2 + n/tau^2
    Lambda_g,l_j  = -1/tau^2
    Lambda_l_j,l_j = 1/tau^2 + 1/sigma_y^2
    b_g = 0,   b_l_j = s_j y_j / sigma_y^2,   m_s = Lambda^{-1} b_s

and -- crucially -- ``C = Lambda^{-1}`` does not depend on s. Hence

    p(g, l_1..l_n | y) = sum_s w_s N(m_s, C),     w_s ~ exp(b_s^T C b_s / 2)

exactly, and under the VESDE kernel the *diffused* joint posterior is the same
mixture with ``C + lam^2 I``. So the exact composed score is available at every
noise level, with no quadrature and no Monte Carlo, for n up to ~16.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Generative parameters. The design note's worked example is sigma_g = 1.5,
# tau = 0.8, sigma_y = 0.3, where at n = 4 the two coherent sign configurations
# already carry ~93% of the posterior sign mass (validate_model.py reproduces
# that number exactly). The tighter tau below is the note's own suggestion for
# sharpening the modes: at n = 8 it puts ~96% of the mass on the two coherent
# configurations and opens a ~11 nat valley at g = 0, which is the regime where
# a composition rule's handling of a deep low-density region is actually
# tested.
# ---------------------------------------------------------------------------
SIGMA_G = 1.5
TAU = 0.5
SIGMA_Y = 0.25

MU_G = 0.0  # the prior over g is centred; the symmetry of the posterior needs it

# Marginal moments of l = g + Normal(0, tau^2), used as the sampler's Gaussian
# stand-in for p(l) when it clamps denoised local predictions.
LOCAL_PRIOR_MEAN = MU_G
LOCAL_PRIOR_STD = math.sqrt(SIGMA_G**2 + TAU**2)

NAME = "sign"
TITLE = "Sign-ambiguous hierarchy (bimodality built by composition)"

NODES = 3
GLOBAL_INDEX, LOCAL_INDEX, OBSERVED_INDEX = 0, 1, 2
CONDITION_MASK = torch.tensor([0.0, 0.0, 1.0])
HIERARCHY = [GLOBAL_INDEX]


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def simulate(count, generator=None):
    """Forward draws of (g, l), y, shaped for ``SBIm.train``'s theta/x split."""
    g = MU_G + SIGMA_G * torch.randn(count, 1, generator=generator)
    local = g + TAU * torch.randn(count, 1, generator=generator)
    sign = torch.where(
        torch.rand(count, 1, generator=generator) < 0.5,
        -torch.ones(count, 1), torch.ones(count, 1),
    )
    y = sign * local + SIGMA_Y * torch.randn(count, 1, generator=generator)
    return torch.cat([g, local], dim=1), y


def observations(count, seed):
    """One dataset: the true g, its locals, the hidden signs and the observed y."""
    generator = torch.Generator().manual_seed(seed)
    g = float(MU_G + SIGMA_G * torch.randn((), generator=generator))
    local = g + TAU * torch.randn(count, generator=generator).numpy()
    sign = np.where(
        torch.rand(count, generator=generator).numpy() < 0.5, -1.0, 1.0
    )
    y = sign * local + SIGMA_Y * torch.randn(count, generator=generator).numpy()
    return g, local, sign, y


# ---------------------------------------------------------------------------
# The exact joint posterior: a 2^n-component Gaussian mixture with shared cov
# ---------------------------------------------------------------------------

def _arrow(n):
    """Precision and covariance of p(g, l_1..l_n | y, s); both independent of s."""
    precision = np.zeros((n + 1, n + 1))
    precision[0, 0] = 1.0 / SIGMA_G**2 + n / TAU**2
    precision[0, 1:] = -1.0 / TAU**2
    precision[1:, 0] = -1.0 / TAU**2
    index = np.arange(1, n + 1)
    precision[index, index] = 1.0 / TAU**2 + 1.0 / SIGMA_Y**2
    return precision, np.linalg.inv(precision)


def sign_configurations(n):
    """All 2^n sign vectors, as a (2^n, n) array of +-1."""
    return np.array(list(itertools.product((-1.0, 1.0), repeat=n)))


def coherent_components(y):
    """Mask of the two configurations in which every row points the same way.

    Coherence is a statement about ``s_j sign(y_j)``, not about ``s_j``: the
    likelihood is even in ``y``, so the sign the simulator happened to draw for
    an observation carries no information and only ``|y_j|`` does. These two
    configurations are the ferromagnetic ground states the hierarchy prefers.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    effective = sign_configurations(len(y)) * np.sign(y)[None, :]
    return np.abs(effective.sum(axis=1)) == len(y)


def mixture(y):
    """``(weights, means, covariance)`` of the exact p(g, l_1..l_n | y).

    ``means`` is (2^n, n+1) with column 0 the shared coordinate; ``covariance``
    is the single (n+1, n+1) matrix every component shares. Weights come from
    the evidence ``p(y | s) ~ exp(b_s^T C b_s / 2)``: the s-dependence of the
    Gaussian normaliser is only that term, because ``sum_j (s_j y_j)^2`` and the
    determinant of ``C`` are both independent of ``s``.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    _, covariance = _arrow(n)
    signs = sign_configurations(n)
    linear = np.zeros((len(signs), n + 1))
    linear[:, 1:] = signs * y[None, :] / SIGMA_Y**2
    means = linear @ covariance.T
    log_weights = 0.5 * np.einsum("ki,ki->k", linear, means)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum(), means, covariance


# The rest of the exact reference surface -- marginals, diffused scores, exact
# draws -- depends on nothing but the ``(weights, means, covariance)`` triple
# above, so it lives once in ``mixture_reference`` and is called from there as
# ``mixture_reference.shared_reference(sign_model, y)``.


# ---------------------------------------------------------------------------
# The single-observation (row) posterior -- what the network is trained on
# ---------------------------------------------------------------------------

def row_mixture(value):
    """``(weights, means, covariance)`` of p(g, l | y) for one observation.

    Two equally weighted components at ``+-m`` with a common covariance, exactly
    as in the design note: the sign is the only unknown discrete variable and
    the model is symmetric under ``(g, l, s) -> (-g, -l, -s)``. Identical to the
    ``n = 1`` case of :func:`mixture`, written out because it is the object the
    row-score network has to reproduce.
    """
    prior = np.array([[SIGMA_G**2, SIGMA_G**2],
                      [SIGMA_G**2, SIGMA_G**2 + TAU**2]])
    a = np.array([0.0, 1.0])
    prior_a = prior @ a
    denominator = float(a @ prior_a) + SIGMA_Y**2
    covariance = prior - np.outer(prior_a, prior_a) / denominator
    mean = prior_a * float(value) / denominator
    return np.array([0.5, 0.5]), np.stack([mean, -mean]), covariance


def conditional_local_mixture(g, value):
    """``(weights, means, variance)`` of p(l | g, y): the conditioned branch.

    ``p(l | g, y) ~ N(l; g, tau^2) [N(y; l, sigma_y^2) + N(y; -l, sigma_y^2)]``,
    a two-component mixture whose weights -- unlike the row posterior's -- are
    *not* equal: knowing ``g`` breaks the symmetry and picks a sign.
    """
    g = np.asarray(g, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    variance = 1.0 / (1.0 / TAU**2 + 1.0 / SIGMA_Y**2)
    plus = variance * (g / TAU**2 + value / SIGMA_Y**2)
    minus = variance * (g / TAU**2 - value / SIGMA_Y**2)
    spread = TAU**2 + SIGMA_Y**2
    log_weights = np.stack([-0.5 * (g - value) ** 2 / spread,
                            -0.5 * (g + value) ** 2 / spread])
    log_weights -= log_weights.max(axis=0, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=0, keepdims=True)
    return weights, np.stack([plus, minus]), variance

# ---------------------------------------------------------------------------
# The same row posterior, vectorized in torch -- what the exact row-score
# network differentiates.  Kept next to the numpy forms above so the two can be
# checked against each other (``python row_scores.py`` does exactly that);
# everything here must stay differentiable for ``torch.func.jvp``,
# because ``correction="gauss_jacobian"`` gets its curvature by
# forward-differentiating the row score.
# ---------------------------------------------------------------------------

def row_components(obs):
    """``(log_weights, means, covariance)`` of p(g, l | y) for a batch of y.

    Shapes ``(B, 2)``, ``(B, 2, 2)`` and ``(2, 2)``. The means are exactly
    linear in ``y`` and antisymmetric between the components, and the weights
    are equal at every ``y`` -- the whole content of the sign ambiguity.
    """
    obs = torch.as_tensor(obs, dtype=torch.float64)
    prior = torch.tensor([[SIGMA_G**2, SIGMA_G**2],
                          [SIGMA_G**2, SIGMA_G**2 + TAU**2]], dtype=torch.float64,
                         device=obs.device)
    prior_a = prior[:, 1]
    denominator = float(prior[1, 1]) + SIGMA_Y**2
    covariance = prior - torch.outer(prior_a, prior_a) / denominator
    direction = prior_a / denominator
    plus = obs[:, None] * direction[None, :]
    means = torch.stack([plus, -plus], dim=1)
    return torch.zeros(obs.shape[0], 2, dtype=torch.float64, device=obs.device), \
        means, covariance


def conditional_local_components(g, obs):
    """``(log_weights, means, variance)`` of p(l | g, y), vectorized in torch.

    Shapes ``(B, 2)``, ``(B, 2)`` and a scalar. Conditioning on ``g`` is what
    breaks the sign symmetry: the weights below are equal only at ``g = 0``.
    """
    variance = 1.0 / (1.0 / TAU**2 + 1.0 / SIGMA_Y**2)
    plus = variance * (g / TAU**2 + obs / SIGMA_Y**2)
    minus = variance * (g / TAU**2 - obs / SIGMA_Y**2)
    spread = TAU**2 + SIGMA_Y**2
    log_weights = torch.stack([-0.5 * (g - obs) ** 2 / spread,
                               -0.5 * (g + obs) ** 2 / spread], dim=1)
    return log_weights, torch.stack([plus, minus], dim=1), variance
