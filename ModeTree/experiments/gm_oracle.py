"""Exact analytic machinery for the oracle bridge experiment on the GM benchmark.

Everything here is closed form.  No network is loaded, no Monte-Carlo reference
is used, and every score, Jacobian and density is the exact one for the
hierarchical Gaussian mixture of
``Gaussian_mixtures_compositional_score_modelling``::

    C ~ {-,+};  g | C ~ N(C mu_0, tau_g^2);  l_i | g ~ N(g, tau_l^2);
    x_i | l_i ~ N(l_i, sigma_x^2)

The state the samplers move is the full tall vector

    theta = (g, l_1, ..., l_n)   in   R^{n+1}

which is what ``MultiObsSampler`` also moves (its hierarchy dimension is kept
synchronised across the observation rows; here it is simply stored once).

Three families of exact object are provided.

``TallPosterior``
    The exact tall posterior is a **two-component Gaussian in n+1 dimensions**
    with a branch-independent covariance, so its ``lambda``-diffused version is
    the same mixture with ``C -> C + lambda^2 I`` and both its density and its
    score are exact at every noise level.  This is the reference the
    "true diffused tall score" arms are built from, and the object
    ``TRUE-SCORE-DPM2`` integrates.

``RowPosteriors``
    ``p(g, l_i | x_i)`` for every row at once: a two-component Gaussian mixture
    in two dimensions per row, with the row-independent conditional covariance
    written out.  Diffusion again only inflates the covariance, so row scores
    *and* row Jacobians are exact.

``MixturePrior1D``
    The exact two-component prior ``p(g)`` and its diffusion.  This is the term
    every composition rule subtracts ``n - 1`` copies of.  It is deliberately
    **not** moment-matched to a Gaussian: the whole point of the oracle
    experiment is that no approximation of any kind sits between the rules and
    the truth.

The factorisation the rules exploit,

    p(g, l_1..l_n | X)  =  p(g)^{1-n} prod_i p(g, l_i | x_i)  x  const,

is exact for this model and is checked in ``tests/test_gm_oracle.py``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class VESchedule:
    """The COMPASS ``VESDE``: ``lambda(t) = sqrt((s^{2t} - 1) / (2 log s))``.

    ``sigma = 15`` is the value the Gaussian-mixture benchmark trains and
    samples with, so the noise range the bridges are compared over is the one
    the project actually uses.
    """

    def __init__(self, sigma=15.0):
        self.sigma = float(sigma)
        self.log_sigma = math.log(self.sigma)

    def lam(self, t):
        t = np.asarray(t, dtype=np.float64)
        return np.sqrt((self.sigma ** (2.0 * t) - 1.0) / (2.0 * self.log_sigma))

    def time_of_lam(self, lam):
        lam = np.asarray(lam, dtype=np.float64)
        return np.log1p(2.0 * self.log_sigma * lam ** 2) / (2.0 * self.log_sigma)


# ---------------------------------------------------------------------------
# Exact pieces
# ---------------------------------------------------------------------------

def joint_covariance(mixture):
    """``Cov[(g, l, x) | C]`` -- the same for both components."""
    g = mixture.tau_g ** 2
    l = g + mixture.tau_l ** 2
    x = l + mixture.sigma_x ** 2
    return np.array([[g, g, g], [g, l, l], [g, l, x]], dtype=np.float64)


class MixturePrior1D:
    """The exact two-component prior ``p(g)`` and its ``lambda``-diffusion."""

    def __init__(self, mixture):
        self.centres = torch.as_tensor(mixture.components, dtype=torch.float64)
        self.log_weights = torch.log(
            torch.as_tensor(mixture.weights, dtype=torch.float64)
        )
        self.variance = float(mixture.tau_g ** 2)

    def _terms(self, g, lam):
        """``(log component weights, component scores)`` at ``g``; ``g`` is (...,)."""
        total = self.variance + float(lam) ** 2
        delta = g.unsqueeze(-1) - self.centres                    # (..., 2)
        logits = self.log_weights - 0.5 * delta ** 2 / total
        return logits, -delta / total, total

    def score(self, g, lam=0.0):
        logits, component, _ = self._terms(g, lam)
        return (torch.softmax(logits, dim=-1) * component).sum(-1)

    def hessian(self, g, lam=0.0):
        """``d^2/dg^2 log p_lambda(g)`` -- exact, and *not* ``-1/total``."""
        logits, component, total = self._terms(g, lam)
        r = torch.softmax(logits, dim=-1)
        mean = (r * component).sum(-1)
        second = (r * (component ** 2 - 1.0 / total)).sum(-1)
        return second - mean ** 2

    def log_prob(self, g, lam=0.0):
        logits, _, total = self._terms(g, lam)
        return torch.logsumexp(logits, dim=-1) - 0.5 * math.log(2.0 * math.pi * total)


class RowPosteriors:
    """``p(g, l_i | x_i)`` for all ``n`` rows: two components, two dimensions.

    ``means`` is ``(n, 2, 2)`` and the conditional covariance is shared by every
    row and both components, so a single ``2 x 2`` inverse serves the whole
    batch at a given noise level.
    """

    def __init__(self, mixture, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        sigma = joint_covariance(mixture)
        v_tot = sigma[2, 2]
        latent = sigma[:2, :2]
        cross = sigma[:2, 2]
        self.cov = torch.as_tensor(
            latent - np.outer(cross, cross) / v_tot, dtype=torch.float64
        )
        centres = np.asarray(mixture.components, dtype=np.float64)     # (2,)
        # mean_{i,c} = (c, c) + cross (x_i - c) / v_tot
        base = centres[None, :, None] * np.ones((1, 1, 2))             # (1, 2, 2)
        residual = (x[:, None] - centres[None, :])[..., None]          # (n, 2, 1)
        self.means = torch.as_tensor(
            base + residual * (cross / v_tot)[None, None, :], dtype=torch.float64
        )
        log_w = (np.log(np.asarray(mixture.weights, dtype=np.float64))[None, :]
                 - 0.5 * (x[:, None] - centres[None, :]) ** 2 / v_tot)
        log_w = log_w - log_w.max(axis=1, keepdims=True)
        weights = np.exp(log_w)
        weights /= weights.sum(axis=1, keepdims=True)
        self.log_weights = torch.as_tensor(np.log(weights), dtype=torch.float64)
        self.n = len(x)

    def _prepared(self, lam):
        cov = self.cov + (float(lam) ** 2) * torch.eye(2, dtype=torch.float64)
        return torch.linalg.inv(cov), cov

    def _terms(self, state, lam):
        """``state`` is ``(..., n, 2)``; returns responsibilities and ``-P delta``."""
        precision, _ = self._prepared(lam)
        delta = state.unsqueeze(-2) - self.means                       # (..., n, 2, 2)
        solved = delta @ precision                                     # symmetric P
        quad = (delta * solved).sum(-1)                                # (..., n, 2)
        logits = self.log_weights + (-0.5) * quad
        return torch.softmax(logits, dim=-1), -solved, precision

    def scores(self, state, lam):
        r, component, _ = self._terms(state, lam)
        return (r.unsqueeze(-1) * component).sum(-2)                   # (..., n, 2)

    def hessians(self, state, lam):
        """Exact ``grad^2 log p_lambda(g, l_i | x_i)``, shape ``(..., n, 2, 2)``."""
        r, component, precision = self._terms(state, lam)
        outer = component.unsqueeze(-1) * component.unsqueeze(-2)      # (...,n,2,2,2)
        first = (r[..., None, None] * (outer - precision)).sum(-3)
        mean = (r.unsqueeze(-1) * component).sum(-2)
        return first - mean.unsqueeze(-1) * mean.unsqueeze(-2)

    def log_prob(self, state, lam):
        precision, cov = self._prepared(lam)
        delta = state.unsqueeze(-2) - self.means
        quad = (delta @ precision * delta).sum(-1)
        logits = self.log_weights + (-0.5) * quad
        return (torch.logsumexp(logits, dim=-1)
                - 0.5 * torch.logdet(cov) - math.log(2.0 * math.pi))

    def global_marginal_log_prob(self, g, lam):
        """``log p_lambda(g | x_i)`` for every row: the exact ``l``-marginal.

        Used for the bridge pictures, where the object of interest is the
        one-dimensional shape each rule builds on the shared coordinate.
        ``g`` is ``(...,)`` and the result is ``(..., n)``.
        """
        total = float(self.cov[0, 0]) + float(lam) ** 2
        delta = g[..., None, None] - self.means[..., 0]                # (..., n, 2)
        logits = self.log_weights + (-0.5) * delta ** 2 / total
        return (torch.logsumexp(logits, dim=-1)
                - 0.5 * math.log(2.0 * math.pi * total))


class TallPosterior:
    """The exact tall posterior and every diffused quantity derived from it."""

    def __init__(self, mixture, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n = len(x)
        self.dim = self.n + 1
        weights, _, variance = mixture.global_posterior(x)
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.log_weights = torch.log(self.weights)
        self.means = torch.as_tensor(
            np.stack([mixture.branch_mean_vector(x, c) for c in range(2)]),
            dtype=torch.float64,
        )                                                              # (2, n+1)
        self.cov = torch.as_tensor(mixture.branch_covariance(self.n),
                                   dtype=torch.float64)
        self.global_variance = float(variance)
        self.global_means = self.means[:, 0].clone()
        self._cache = {}

    # -- diffused quantities ------------------------------------------------
    def _prepared(self, lam):
        key = float(lam)
        if key not in self._cache:
            cov = self.cov + (key ** 2) * torch.eye(self.dim, dtype=torch.float64)
            factor = torch.linalg.cholesky(cov)
            self._cache[key] = factor
        return self._cache[key]

    def score(self, state, lam):
        """Exact ``grad log p_lambda(theta | x_{1:n})``, shape ``(..., n+1)``."""
        factor = self._prepared(lam)
        delta = state.unsqueeze(-2) - self.means                       # (..., 2, n+1)
        solved = torch.cholesky_solve(delta.unsqueeze(-1), factor).squeeze(-1)
        quad = (delta * solved).sum(-1)                                # (..., 2)
        r = torch.softmax(self.log_weights - 0.5 * quad, dim=-1)
        return (r.unsqueeze(-1) * (-solved)).sum(-2)

    def global_log_prob(self, g, lam=0.0):
        """``log p_lambda(g | x_{1:n})`` -- the exact shared marginal."""
        total = self.global_variance + float(lam) ** 2
        delta = g.unsqueeze(-1) - self.global_means
        logits = self.log_weights - 0.5 * delta ** 2 / total
        return (torch.logsumexp(logits, dim=-1)
                - 0.5 * math.log(2.0 * math.pi * total))

    def global_score(self, g, lam=0.0):
        total = self.global_variance + float(lam) ** 2
        delta = g.unsqueeze(-1) - self.global_means
        logits = self.log_weights - 0.5 * delta ** 2 / total
        r = torch.softmax(logits, dim=-1)
        return (r * (-delta / total)).sum(-1)

    # -- exact reference summaries -----------------------------------------
    def global_cdf(self, points):
        total = self.global_variance
        z = (points.unsqueeze(-1) - self.global_means) / math.sqrt(2.0 * total)
        return (self.weights * 0.5 * (1.0 + torch.erf(z))).sum(-1)

    def global_moments(self):
        mean = float((self.weights * self.global_means).sum())
        second = float((self.weights
                        * (self.global_means ** 2 + self.global_variance)).sum())
        return mean, math.sqrt(max(second - mean ** 2, 0.0))

    def global_quantiles(self, probabilities, points=200_001):
        sd = math.sqrt(self.global_variance)
        low = float(self.global_means.min()) - 12.0 * sd
        high = float(self.global_means.max()) + 12.0 * sd
        grid = torch.linspace(low, high, points, dtype=torch.float64)
        cdf = self.global_cdf(grid)
        return interp(probabilities, cdf, grid)

    def sample_global(self, count, generator=None):
        """Exact draws of ``g`` -- the numerical floor for every W1 reported."""
        pick = torch.multinomial(
            self.weights, count, replacement=True, generator=generator
        )
        noise = torch.randn(count, dtype=torch.float64, generator=generator)
        return self.global_means[pick] + math.sqrt(self.global_variance) * noise


def interp(query, xp, fp):
    """``numpy.interp`` for monotone float64 tensors."""
    query = torch.as_tensor(query, dtype=torch.float64)
    index = torch.searchsorted(xp.contiguous(), query.contiguous()).clamp(1, len(xp) - 1)
    lo, hi = index - 1, index
    span = (xp[hi] - xp[lo]).clamp_min(1e-300)
    weight = (query - xp[lo]) / span
    return fp[lo] + weight * (fp[hi] - fp[lo])


# ---------------------------------------------------------------------------
# The oracle bundle
# ---------------------------------------------------------------------------

class GMOracle:
    """Every exact object one dataset needs, plus evaluation counters."""

    def __init__(self, mixture, x, sigma=15.0):
        self.mixture = mixture
        self.x = np.asarray(x, dtype=np.float64).ravel()
        self.n = len(self.x)
        self.dim = self.n + 1
        self.schedule = VESchedule(sigma)
        self.rows = RowPosteriors(mixture, self.x)
        self.prior = MixturePrior1D(mixture)
        self.tall = TallPosterior(mixture, self.x)
        self.reset_counters()

    def reset_counters(self):
        self.row_score_evaluations = 0
        self.row_jacobian_evaluations = 0
        self.field_calls = 0

    # -- state layout -------------------------------------------------------
    def rows_state(self, state):
        """``(..., n+1)`` tall state -> ``(..., n, 2)`` per-row states."""
        g = state[..., :1]
        locals_ = state[..., 1:]
        return torch.stack([g.expand_as(locals_), locals_], dim=-1)

    def assemble(self, global_score, local_scores):
        return torch.cat([global_score.unsqueeze(-1), local_scores], dim=-1)

    # -- counted accessors --------------------------------------------------
    def row_scores(self, rows_state, lam, samples=1):
        self.row_score_evaluations += self.n * samples
        return self.rows.scores(rows_state, lam)

    def row_hessians(self, rows_state, lam, samples=1):
        self.row_jacobian_evaluations += self.n * samples
        return self.rows.hessians(rows_state, lam)
