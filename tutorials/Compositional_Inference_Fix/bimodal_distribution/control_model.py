"""The control twin: bimodality *given* in the prior, not built by composition.

    z          ~ Bernoulli(1/2)
    g | z      ~ Normal(-a, sigma_0^2) if z = 0 else Normal(+a, sigma_0^2)
    l_j | g    ~ Normal(g, tau^2)
    y_j | l_j  ~ Normal(l_j, sigma_y^2)          -- no sign ambiguity

Here every *row* posterior p(g, l_j | y_j) is already two-moded, for the trivial
reason that the prior is, and each row's evidence points at one of the two modes
directly. Nothing has to be constructed: a composition rule only has to avoid
destroying a bimodality that was handed to it. Contrast ``sign_model.py``, where
each row is ambiguous and the coordination between rows is the whole answer.

Running both with the same samplers separates

    "can COMPASS preserve bimodality?"   (this file)
    "can COMPASS construct bimodality?"  (sign_model.py)

Two caveats, stated once here and repeated in the figures. First:
``MultiObsSampler``
takes a *Gaussian* prior over the shared coordinate for its ``(1 - n)``
prior-score term, and this model's prior is not Gaussian. Every arm is given the
moment-matched Normal(0, a^2 + sigma_0^2), so this benchmark carries a
prior-score approximation the sign model does not -- there the prior over ``g``
really is Normal(0, sigma_g^2) and the composition rule is graded with nothing
else mis-specified. The exact references below use the true bimodal prior.
Second: this model's parameters are *not* the sign model's, for the reason
spelled out at the constants below -- a well-separated prior plus informative
data gives a unimodal posterior, so the control has to be run in the regime
where the likelihood cannot resolve the two lobes.

Everything is again closed form: conditional on ``z`` the model is linear
Gaussian with the same arrow precision for both components (only the prior mean
of ``g`` differs), so the posterior is a *two*-component Gaussian mixture with a
shared covariance, and the diffused posterior adds ``lam^2 I`` to it.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# These are not sign_model's tau and sigma_y, and cannot be. With locals that
# tight, eight observations pin g to within ~0.1 and the prior's two lobes stop
# being a posterior question at all: the data simply says which lobe, and there
# is no bimodality left to preserve. The prior gap only survives into the
# posterior while the likelihood cannot resolve it, i.e. while sqrt(v/n) is
# comparable to A with v = tau^2 + sigma_y^2. tau = 3 puts sqrt(v/8) at ~1.1
# against lobes at +-2, which is that regime.
#
# This asymmetry is itself a result, and the reason the two benchmarks are not
# the same experiment: in the control, bimodality is a property of the *dataset*
# -- most draws resolve the lobe and leave a unimodal posterior -- whereas in
# the sign model every dataset is bimodal, because the ambiguity is in the
# likelihood rather than in the prior. ``run_experiment.py`` therefore picks a
# control dataset whose posterior does retain both lobes; otherwise the arms
# would be graded on a target with nothing to preserve.
A = 2.0
SIGMA_0 = 0.4
TAU = 3.0
SIGMA_Y = 0.5

MU_G = 0.0
# Moment-matched Gaussian stand-in handed to the sampler's prior-score term.
PRIOR_STD = math.sqrt(A**2 + SIGMA_0**2)
LOCAL_PRIOR_MEAN = MU_G
LOCAL_PRIOR_STD = math.sqrt(PRIOR_STD**2 + TAU**2)

NAME = "control"
TITLE = "Control: bimodality already present in the prior"

NODES = 3
GLOBAL_INDEX, LOCAL_INDEX, OBSERVED_INDEX = 0, 1, 2
CONDITION_MASK = torch.tensor([0.0, 0.0, 1.0])
HIERARCHY = [GLOBAL_INDEX]


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def simulate(count, generator=None):
    """Forward draws of (g, l), y, shaped for ``SBIm.train``'s theta/x split."""
    sign = torch.where(
        torch.rand(count, 1, generator=generator) < 0.5,
        -torch.ones(count, 1), torch.ones(count, 1),
    )
    g = sign * A + SIGMA_0 * torch.randn(count, 1, generator=generator)
    local = g + TAU * torch.randn(count, 1, generator=generator)
    y = local + SIGMA_Y * torch.randn(count, 1, generator=generator)
    return torch.cat([g, local], dim=1), y


def observations(count, seed):
    """One dataset: the true g, its locals, the prior component and observed y.

    The third return value is the prior component as a ``+-1`` array of length
    ``count`` so the signature matches ``sign_model.observations``; unlike there
    it is a single shared latent, repeated.
    """
    generator = torch.Generator().manual_seed(seed)
    component = 1.0 if float(torch.rand((), generator=generator)) < 0.5 else -1.0
    g = float(component * A + SIGMA_0 * torch.randn((), generator=generator))
    local = g + TAU * torch.randn(count, generator=generator).numpy()
    y = local + SIGMA_Y * torch.randn(count, generator=generator).numpy()
    return g, local, np.full(count, component), y


# ---------------------------------------------------------------------------
# The exact joint posterior: two Gaussians sharing one covariance
# ---------------------------------------------------------------------------

def _arrow(n):
    precision = np.zeros((n + 1, n + 1))
    precision[0, 0] = 1.0 / SIGMA_0**2 + n / TAU**2
    precision[0, 1:] = -1.0 / TAU**2
    precision[1:, 0] = -1.0 / TAU**2
    index = np.arange(1, n + 1)
    precision[index, index] = 1.0 / TAU**2 + 1.0 / SIGMA_Y**2
    return precision, np.linalg.inv(precision)


def sign_configurations(n):
    """The two prior components, in ``sign_model``'s array layout."""
    return np.array([[-1.0] * n, [1.0] * n])


def coherent_components(y):
    """Both components: the two prior lobes are all there is to be coherent about.

    Present so ``mixture_reference.aligned_mass`` has the same question to ask
    of either benchmark; here the answer is trivially "all of them", which is
    exactly the difference between the two problems.
    """
    return np.ones(2, dtype=bool)


def mixture(y):
    """``(weights, means, covariance)`` of the exact p(g, l_1..l_n | y).

    Only the shared coordinate's prior mean differs between components, so both
    share ``C = Lambda^{-1}`` and the log weight of component ``c`` is
    ``b_c^T C b_c / 2`` (the ``-a^2 / 2 sigma_0^2`` term is common to both).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    _, covariance = _arrow(n)
    linear = np.zeros((2, n + 1))
    linear[:, 0] = np.array([-A, A]) / SIGMA_0**2
    linear[:, 1:] = y[None, :] / SIGMA_Y**2
    means = linear @ covariance.T
    log_weights = 0.5 * np.einsum("ki,ki->k", linear, means)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum(), means, covariance

# The remaining reference surface -- marginals, diffused scores, exact draws --
# needs nothing but the triple above, so it comes from ``mixture_reference``
# exactly as ``sign_model``'s does.


# ---------------------------------------------------------------------------
# The single-observation (row) posterior
# ---------------------------------------------------------------------------

def row_mixture(value):
    """``(weights, means, covariance)`` of p(g, l | y) for one observation.

    Two components with *different* weights and different means: the datum
    already prefers one of the prior's two lobes. The covariance is shared.
    """
    return mixture([value])


def conditional_local_mixture(g, value):
    """``(weights, means, variance)`` of p(l | g, y): a single Gaussian.

    With the sign ambiguity gone this branch is unimodal;
    ``p(l | g, y) = Normal(v (g/tau^2 + y/sigma_y^2), v)``. It is returned in
    mixture form (one component of weight one) so the row-score classes can
    share a code path with ``sign_model``.
    """
    g = np.asarray(g, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    variance = 1.0 / (1.0 / TAU**2 + 1.0 / SIGMA_Y**2)
    mean = variance * (g / TAU**2 + value / SIGMA_Y**2)
    return np.ones((1,) + np.shape(mean)), mean[None, ...], variance

# ---------------------------------------------------------------------------
# The same row posterior, vectorized in torch -- see the note in sign_model.py.
# ---------------------------------------------------------------------------

def row_components(obs):
    """``(log_weights, means, covariance)`` of p(g, l | y) for a batch of y.

    Both components share ``C``; only the prior mean of ``g`` differs, so the
    log weight of component ``k`` is ``b_k^T C b_k / 2`` with the ``y``-only
    term dropped (common to both, and the caller normalizes).
    """
    obs = torch.as_tensor(obs, dtype=torch.float64)
    _, covariance = _arrow(1)
    covariance = torch.as_tensor(covariance, dtype=torch.float64, device=obs.device)
    centres = torch.tensor([-A, A], dtype=torch.float64,
                           device=obs.device) / SIGMA_0**2
    linear = torch.stack([
        centres[None, :].expand(obs.shape[0], 2),
        (obs / SIGMA_Y**2)[:, None].expand(obs.shape[0], 2),
    ], dim=2)                                             # (B, 2 components, 2)
    means = linear @ covariance.T
    log_weights = (0.5 * float(covariance[0, 0]) * centres[None, :] ** 2
                   + float(covariance[0, 1]) * centres[None, :]
                   * (obs / SIGMA_Y**2)[:, None])
    return log_weights, means, covariance


def conditional_local_components(g, obs):
    """``(log_weights, means, variance)`` of p(l | g, y): one Gaussian.

    Returned in the same one-component-mixture form ``sign_model`` uses, so the
    row-score network needs no special case.
    """
    variance = 1.0 / (1.0 / TAU**2 + 1.0 / SIGMA_Y**2)
    mean = variance * (g / TAU**2 + obs / SIGMA_Y**2)
    return torch.zeros(g.shape[0], 1, dtype=g.dtype, device=g.device), \
        mean[:, None], variance
