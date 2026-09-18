"""Milestone A -- analytical multimodal benchmarks with exact ground truth.

Everything here is a Gaussian-mixture *factor* model, which is the smallest
class that is simultaneously

* genuinely multimodal in the global parameters,
* exact in closed form for the row posteriors, their scores and Jacobians,
* exact on a grid for the tall posterior at any ``n``, including ``n = 100``,
  because the tall log density is a plain sum of ``n`` log-sum-exps and never
  needs the ``prod_i K_i`` expansion the task explicitly forbids.

Model
-----
Global ``g in R^{d_g}`` with Gaussian prior ``p(g) = N(mu_0, Sigma_0)``.
Optional row-local ``l_i in R^{d_l}`` with Gaussian conditional prior
``p(l_i | g) = N(M g + m, Sigma_l)``.

Row ``i`` contributes an unnormalised factor (its "likelihood")

    L_i(y_i) = sum_k w_ik N(y_i; m_ik, S_ik),      y_i = (g, l_i)

so the row posterior

    p(y_i | x_i)  ∝  p(g) p(l_i | g) L_i(y_i)

is itself a Gaussian mixture in closed form, and the tall posterior

    p(g, l_{1:n} | X)  ∝  p(g) prod_i [ p(l_i | g) L_i(g, l_i) ]

has the exact factorisation the compositional rules assume.  Any positive
``L_i`` defines a valid model, so specifying the factors directly costs no
generality.

All computation is float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

TWO_PI_LOG = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------------
# Gaussian mixture primitives (log density, score, Hessian -- all analytic)
# ----------------------------------------------------------------------------

def _cholesky(cov):
    return torch.linalg.cholesky(cov)


class GaussianMixture:
    """``sum_k w_k N(y; m_k, S_k)`` with exact log-density / score / Hessian.

    Weights need not sum to one: the class is used for unnormalised factors as
    well as for proper densities.  ``means`` is ``(K, D)``, ``covs`` ``(K, D, D)``.
    """

    def __init__(self, weights, means, covs):
        self.weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        self.means = torch.as_tensor(means, dtype=torch.float64)
        self.covs = torch.as_tensor(covs, dtype=torch.float64)
        if self.means.dim() == 1:
            self.means = self.means.unsqueeze(-1)
        if self.covs.dim() == 1:
            self.covs = torch.diag_embed(self.covs.unsqueeze(-1))
        if self.covs.dim() == 2 and self.means.shape[0] == self.covs.shape[0]:
            # (K, D) diagonal variances
            self.covs = torch.diag_embed(self.covs)
        K, D = self.means.shape
        assert self.covs.shape == (K, D, D), (self.covs.shape, (K, D, D))
        assert self.weights.shape == (K,)
        self.dim = D
        self._chol = _cholesky(self.covs)
        # log |S_k| and the constant -D/2 log 2pi - 1/2 log|S_k| + log w_k
        logdet = 2.0 * torch.log(torch.diagonal(self._chol, dim1=-2, dim2=-1)).sum(-1)
        self._log_const = (
            torch.log(self.weights) - 0.5 * D * TWO_PI_LOG - 0.5 * logdet
        )

    # -- component quantities -------------------------------------------------
    def _component_terms(self, y):
        """Return ``(log_k, u_k)`` for points ``y`` of shape ``(..., D)``.

        ``log_k`` is ``(..., K)``: the log of ``w_k N(y; m_k, S_k)``.
        ``u_k``  is ``(..., K, D)``: ``-S_k^{-1}(y - m_k)``, the component score.
        """
        y = torch.as_tensor(y, dtype=torch.float64)
        delta = y.unsqueeze(-2) - self.means                      # (..., K, D)
        # solve with the cached Cholesky factors; never an explicit inverse
        sol = torch.cholesky_solve(delta.unsqueeze(-1), self._chol).squeeze(-1)
        quad = (delta * sol).sum(-1)                              # (..., K)
        log_k = self._log_const + (-0.5) * quad
        return log_k, -sol

    def log_prob(self, y):
        log_k, _ = self._component_terms(y)
        return torch.logsumexp(log_k, dim=-1)

    def responsibilities(self, y):
        log_k, _ = self._component_terms(y)
        return torch.softmax(log_k, dim=-1)

    def score(self, y):
        log_k, u = self._component_terms(y)
        r = torch.softmax(log_k, dim=-1)
        return (r.unsqueeze(-1) * u).sum(-2)

    def hessian(self, y):
        """``grad^2 log p``; exact, via the log-sum-exp second derivative."""
        log_k, u = self._component_terms(y)
        r = torch.softmax(log_k, dim=-1)                          # (..., K)
        inv = torch.cholesky_inverse(self._chol)                  # (K, D, D)
        outer = u.unsqueeze(-1) * u.unsqueeze(-2)                 # (..., K, D, D)
        first = (r[..., None, None] * (-inv + outer)).sum(-3)
        mean_u = (r.unsqueeze(-1) * u).sum(-2)
        return first - mean_u.unsqueeze(-1) * mean_u.unsqueeze(-2)

    # -- algebra --------------------------------------------------------------
    def multiply(self, other):
        """Exact product of two Gaussian mixtures (K1*K2 components).

        Used only where one factor has a single component (prior x factor), so
        the component count stays linear.
        """
        K1, K2, D = self.means.shape[0], other.means.shape[0], self.dim
        weights, means, covs = [], [], []
        for a in range(K1):
            Pa = torch.cholesky_inverse(self._chol[a])
            for b in range(K2):
                Pb = torch.cholesky_inverse(other._chol[b])
                P = Pa + Pb
                cov = torch.linalg.inv(P)
                cov = 0.5 * (cov + cov.mT)
                mean = cov @ (
                    Pa @ self.means[a] + Pb @ other.means[b]
                )
                # normalisation: w_a w_b N(m_a; m_b, S_a + S_b)
                sum_cov = self.covs[a] + other.covs[b]
                chol = _cholesky(sum_cov)
                delta = (self.means[a] - other.means[b]).unsqueeze(-1)
                sol = torch.cholesky_solve(delta, chol)
                logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
                log_c = -0.5 * (delta * sol).sum() - 0.5 * D * TWO_PI_LOG - 0.5 * logdet
                weights.append(self.weights[a] * other.weights[b] * torch.exp(log_c))
                means.append(mean)
                covs.append(cov)
        return GaussianMixture(
            torch.stack(weights), torch.stack(means), torch.stack(covs)
        )

    def marginal(self, keep):
        """Marginalise onto the coordinate indices ``keep`` (exact for a GMM)."""
        keep = list(keep)
        idx = torch.as_tensor(keep)
        return GaussianMixture(
            self.weights, self.means[:, idx], self.covs[:, idx][:, :, idx]
        )


# ----------------------------------------------------------------------------
# The benchmark model
# ----------------------------------------------------------------------------

@dataclass
class FactorModel:
    """A tall-data benchmark with exact everything.

    Attributes
    ----------
    prior_mean, prior_cov : global Gaussian prior ``N(mu_0, Sigma_0)`` on ``g``.
    factors : list of ``GaussianMixture`` over ``y_i = (g, l_i)`` (dimension
        ``d_g + d_l``); one per observation.
    local_map, local_mean, local_cov : the conditional prior
        ``p(l_i | g) = N(local_map @ g + local_mean, local_cov)``.  ``None``
        for an all-global model (``d_l = 0``).
    """

    prior_mean: torch.Tensor
    prior_cov: torch.Tensor
    factors: list
    local_map: torch.Tensor | None = None
    local_mean: torch.Tensor | None = None
    local_cov: torch.Tensor | None = None
    name: str = "factor-model"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.prior_mean = torch.as_tensor(self.prior_mean, dtype=torch.float64).reshape(-1)
        self.prior_cov = torch.as_tensor(self.prior_cov, dtype=torch.float64)
        if self.prior_cov.dim() == 0:
            self.prior_cov = self.prior_cov.reshape(1, 1)
        if self.prior_cov.dim() == 1:
            self.prior_cov = torch.diag(self.prior_cov)
        self.d_g = self.prior_mean.numel()
        self.d_l = 0 if self.local_cov is None else int(
            torch.as_tensor(self.local_cov).shape[-1]
        )
        if self.d_l:
            self.local_map = torch.as_tensor(self.local_map, dtype=torch.float64).reshape(
                self.d_l, self.d_g
            )
            self.local_mean = torch.as_tensor(self.local_mean, dtype=torch.float64).reshape(-1)
            self.local_cov = torch.as_tensor(self.local_cov, dtype=torch.float64).reshape(
                self.d_l, self.d_l
            )
        self.dim = self.d_g + self.d_l
        for f in self.factors:
            assert f.dim == self.dim, (f.dim, self.dim)
        self._prior_gm = GaussianMixture(
            torch.ones(1, dtype=torch.float64),
            self.prior_mean.unsqueeze(0),
            self.prior_cov.unsqueeze(0),
        )
        self._row_cache: dict[int, GaussianMixture] = {}

    # -- sizes ----------------------------------------------------------------
    @property
    def n(self):
        return len(self.factors)

    def subset(self, n):
        """A copy using only the first ``n`` observations."""
        return FactorModel(
            self.prior_mean, self.prior_cov, self.factors[:n],
            self.local_map, self.local_mean, self.local_cov,
            name=f"{self.name}[n={n}]", meta=dict(self.meta),
        )

    # -- prior ---------------------------------------------------------------
    def log_prior(self, g):
        return self._prior_gm.log_prob(g)

    def prior_score(self, g):
        return self._prior_gm.score(g)

    def prior_hessian(self, g):
        return self._prior_gm.hessian(g)

    def _joint_prior(self):
        """``p(g) p(l|g)`` as a single Gaussian over ``(g, l)``."""
        if not self.d_l:
            return self._prior_gm
        M, S0 = self.local_map, self.prior_cov
        mean = torch.cat([self.prior_mean, M @ self.prior_mean + self.local_mean])
        top = torch.cat([S0, S0 @ M.mT], dim=1)
        bot = torch.cat([M @ S0, M @ S0 @ M.mT + self.local_cov], dim=1)
        cov = torch.cat([top, bot], dim=0)
        return GaussianMixture(
            torch.ones(1, dtype=torch.float64), mean.unsqueeze(0), cov.unsqueeze(0)
        )

    # -- row posteriors -------------------------------------------------------
    def row_posterior(self, i):
        """``p(g, l_i | x_i)`` as an exact (normalised) Gaussian mixture."""
        if i not in self._row_cache:
            gm = self._joint_prior().multiply(self.factors[i])
            gm = GaussianMixture(gm.weights / gm.weights.sum(), gm.means, gm.covs)
            self._row_cache[i] = gm
        return self._row_cache[i]

    def row_score(self, i, y):
        return self.row_posterior(i).score(y)

    def row_hessian(self, i, y):
        return self.row_posterior(i).hessian(y)

    def row_global_marginal(self, i):
        """``p(g | x_i)``, exact (the GMM marginal)."""
        return self.row_posterior(i).marginal(range(self.d_g))

    # -- exact tall posterior -------------------------------------------------
    def tall_log_prob(self, state):
        """Unnormalised ``log p(g, l_1..l_n | X)``.

        ``state`` is ``(..., d_g + n*d_l)``: the global block followed by the
        ``n`` local blocks.  Evaluated as a sum of ``n`` log-sum-exps, never as
        an expanded mixture.
        """
        state = torch.as_tensor(state, dtype=torch.float64)
        g = state[..., : self.d_g]
        total = self.log_prior(g)
        joint_prior = self._joint_prior()
        for i, factor in enumerate(self.factors):
            y = self._row_state(state, i)
            total = total + factor.log_prob(y)
            if self.d_l:
                # p(l_i | g) = joint prior / marginal prior
                total = total + joint_prior.log_prob(y) - self.log_prior(g)
        return total

    def _row_state(self, state, i):
        g = state[..., : self.d_g]
        if not self.d_l:
            return g
        start = self.d_g + i * self.d_l
        return torch.cat([g, state[..., start:start + self.d_l]], dim=-1)

    def tall_score(self, state):
        """Exact ``grad log p`` by autograd on :meth:`tall_log_prob`."""
        state = torch.as_tensor(state, dtype=torch.float64).clone().requires_grad_(True)
        value = self.tall_log_prob(state).sum()
        grad, = torch.autograd.grad(value, state)
        return grad.detach()

    def tall_hessian(self, state):
        state = torch.as_tensor(state, dtype=torch.float64).reshape(-1)
        return torch.autograd.functional.hessian(
            lambda s: self.tall_log_prob(s), state
        )

    # -- the exact global marginal posterior ---------------------------------
    def global_log_prob(self, g):
        """Unnormalised ``log p(g | X)`` with the locals marginalised out.

        Exact: ``int p(l_i|g) L_i(g, l_i) dl_i`` is a Gaussian-mixture integral,
        so the marginalised row factor is again a mixture in ``g`` alone.  We
        get it as ``p(g | x_i) / p(g)`` up to a constant, using the exact GMM
        marginal of the row posterior.
        """
        g = torch.as_tensor(g, dtype=torch.float64)
        total = self.log_prior(g)
        for i in range(self.n):
            total = total + self.row_global_marginal(i).log_prob(g) - self.log_prior(g)
        return total

    def global_score(self, g):
        g = torch.as_tensor(g, dtype=torch.float64).clone().requires_grad_(True)
        value = self.global_log_prob(g).sum()
        grad, = torch.autograd.grad(value, g)
        return grad.detach()


# ----------------------------------------------------------------------------
# Reference grid: exact modes, masses, moments, quantiles
# ----------------------------------------------------------------------------

@dataclass
class GridReference:
    grid: torch.Tensor          # (G, d) or (G,) for d=1
    log_density: torch.Tensor   # (G,) normalised log density
    modes: torch.Tensor         # (B, d) local maxima, descending log density
    mode_log_heights: torch.Tensor
    mode_masses: torch.Tensor   # basin masses, same order
    separators: torch.Tensor    # (B-1,) 1-D only: minima between modes
    mean: torch.Tensor
    covariance: torch.Tensor
    quantiles: dict | None = None
    d: int = 1

    @property
    def map(self):
        return self.modes[0]


def reference_1d(model, lo=None, hi=None, points=400_001):
    """Dense 1-D reference for an all-global ``d_g == 1`` model."""
    assert model.d_g == 1 and model.d_l == 0
    if lo is None or hi is None:
        s = float(model.prior_cov.reshape(()).sqrt())
        lo, hi = float(model.prior_mean) - 8 * s, float(model.prior_mean) + 8 * s
    grid = torch.linspace(lo, hi, points, dtype=torch.float64)
    logp = model.global_log_prob(grid.unsqueeze(-1))
    step = float(grid[1] - grid[0])
    logz = torch.logsumexp(logp, 0) + math.log(step)
    logp = logp - logz
    dens = logp.exp()

    # local maxima / minima of the log density on the interior
    left, mid, right = logp[:-2], logp[1:-1], logp[2:]
    is_max = (mid > left) & (mid > right)
    is_min = (mid < left) & (mid < right)
    max_idx = torch.nonzero(is_max).reshape(-1) + 1
    min_idx = torch.nonzero(is_min).reshape(-1) + 1

    # parabolic refinement of each maximum
    def refine(idx):
        y0, y1, y2 = logp[idx - 1], logp[idx], logp[idx + 1]
        denom = (y0 - 2 * y1 + y2)
        shift = torch.where(
            denom.abs() > 1e-300, 0.5 * (y0 - y2) / denom, torch.zeros_like(denom)
        ).clamp(-0.5, 0.5)
        return grid[idx] + shift * step, y1 - 0.25 * (y0 - y2) * shift

    mode_pos, mode_h = refine(max_idx)

    # basin masses: integrate between consecutive separating minima
    edges = torch.cat([
        torch.tensor([0], dtype=torch.long),
        min_idx,
        torch.tensor([points - 1], dtype=torch.long),
    ])
    masses = []
    for b in range(len(max_idx)):
        a, c = int(edges[b]), int(edges[b + 1])
        masses.append(torch.trapz(dens[a:c + 1], grid[a:c + 1]))
    masses = torch.stack(masses) if masses else torch.zeros(0, dtype=torch.float64)

    mean = torch.trapz(dens * grid, grid)
    var = torch.trapz(dens * (grid - mean) ** 2, grid)

    cdf = torch.cumulative_trapezoid(dens, grid, dim=0)
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf])
    quantiles = {
        float(q): float(grid[int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)))])
        for q in (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
    }

    order = torch.argsort(mode_h, descending=True)
    return GridReference(
        grid=grid, log_density=logp,
        modes=mode_pos[order].unsqueeze(-1),
        mode_log_heights=mode_h[order],
        mode_masses=masses[order] if len(masses) else masses,
        separators=grid[min_idx],
        mean=mean.reshape(1), covariance=var.reshape(1, 1),
        quantiles=quantiles, d=1,
    )


def reference_2d(model, lo, hi, points=1201):
    """Dense 2-D reference (grid modes + watershed-free basin masses).

    Basin masses are assigned by steepest-ascent on the grid, which is exact in
    the limit of grid resolution and needs no analytic separatrix.
    """
    assert model.d_g == 2 and model.d_l == 0
    axes = [torch.linspace(lo[k], hi[k], points, dtype=torch.float64) for k in range(2)]
    G0, G1 = torch.meshgrid(axes[0], axes[1], indexing="ij")
    pts = torch.stack([G0.reshape(-1), G1.reshape(-1)], dim=-1)
    # chunked: at n = 100 a 401^2 grid times n rows times K components is
    # gigabytes in one call
    chunks = []
    for start in range(0, pts.shape[0], 20_000):
        chunks.append(model.global_log_prob(pts[start:start + 20_000]))
    logp = torch.cat(chunks).reshape(points, points)
    cell = float(axes[0][1] - axes[0][0]) * float(axes[1][1] - axes[1][0])
    logz = torch.logsumexp(logp.reshape(-1), 0) + math.log(cell)
    logp = logp - logz
    dens = logp.exp()

    # local maxima over the 8-neighbourhood
    pad = torch.nn.functional.pad(logp, (1, 1, 1, 1), value=-float("inf"))
    stacks = [
        pad[1 + dx: 1 + dx + points, 1 + dy: 1 + dy + points]
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
    ]
    is_max = torch.stack([logp > s for s in stacks]).all(0)
    idx = torch.nonzero(is_max)
    modes = torch.stack([axes[0][idx[:, 0]], axes[1][idx[:, 1]]], dim=-1)
    heights = logp[idx[:, 0], idx[:, 1]]

    # steepest-ascent basin labelling
    labels = -torch.ones(points, points, dtype=torch.long)
    for b in range(idx.shape[0]):
        labels[idx[b, 0], idx[b, 1]] = b
    # iterative propagation: each cell adopts the label of its steepest neighbour
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    neigh = torch.stack(stacks)                            # (8, P, P)
    best = neigh.argmax(0)
    # A local maximum must be a *fixed point* of the ascent map.  Its neighbours
    # are all lower, so argmax still names one of them and the chain walks back
    # out of the basin; the pointer doubling below then terminates wherever the
    # walk stalls and the basin masses come out badly wrong (measured
    # 0.625/0.125/0.125/0.125 on a four-fold symmetric target that must be
    # 0.25 each).
    at_maximum = logp >= neigh.max(0).values
    step_x = torch.tensor([o[0] for o in offsets])
    step_y = torch.tensor([o[1] for o in offsets])
    ii = torch.arange(points).reshape(-1, 1).expand(points, points)
    jj = torch.arange(points).reshape(1, -1).expand(points, points)
    tgt_i = torch.where(at_maximum, ii,
                        (ii + step_x[best]).clamp(0, points - 1))
    tgt_j = torch.where(at_maximum, jj,
                        (jj + step_y[best]).clamp(0, points - 1))
    flat_tgt = (tgt_i * points + tgt_j).reshape(-1)
    flat_labels = labels.reshape(-1).clone()
    # pointer doubling until every cell points at a maximum
    for _ in range(int(math.ceil(math.log2(2 * points))) + 2):
        flat_tgt = flat_tgt[flat_tgt]
    flat_labels = labels.reshape(-1)[flat_tgt]
    masses = torch.zeros(idx.shape[0], dtype=torch.float64)
    masses.index_add_(0, flat_labels.clamp_min(0), dens.reshape(-1) * cell)

    mean = (dens.unsqueeze(-1) * torch.stack([G0, G1], -1)).reshape(-1, 2).sum(0) * cell
    delta = pts - mean
    covariance = (dens.reshape(-1, 1, 1) * delta.unsqueeze(-1) * delta.unsqueeze(-2)
                  ).sum(0) * cell

    order = torch.argsort(heights, descending=True)
    return GridReference(
        grid=pts, log_density=logp.reshape(-1),
        modes=modes[order], mode_log_heights=heights[order],
        mode_masses=masses[order], separators=torch.zeros(0),
        mean=mean, covariance=covariance, d=2,
    )


# ----------------------------------------------------------------------------
# Regime constructors (A1)
# ----------------------------------------------------------------------------

def _mix(weights, means, sigmas, d=1):
    means = torch.as_tensor(means, dtype=torch.float64).reshape(-1, d)
    sig = torch.as_tensor(sigmas, dtype=torch.float64).reshape(-1)
    covs = torch.stack([torch.eye(d, dtype=torch.float64) * s ** 2 for s in sig])
    return GaussianMixture(torch.as_tensor(weights, dtype=torch.float64), means, covs)


def bimodal_1d(n, regime="symmetric", tau=3.0, seed=0, jitter=0.0):
    """The A1 family: ``n`` bimodal factors over a scalar global parameter.

    Regimes
    -------
    ``symmetric``      two equal modes at +-mu
    ``asymmetric``     unequal mixture weights (unequal mode heights)
    ``narrow_broad``   one narrow high peak, one broad high-mass mode
    ``vanishing``      per-row asymmetry that kills one mode as n grows
    ``nearly_tied``    mode heights within a few 1e-2 of each other
    ``separated``      widely separated modes with an empty valley
    """
    rng = torch.Generator().manual_seed(seed)
    factors = []
    for i in range(n):
        if regime == "symmetric":
            w, m, s = [0.5, 0.5], [2.0, -2.0], [1.0, 1.0]
        elif regime == "asymmetric":
            w, m, s = [0.62, 0.38], [2.0, -2.0], [1.0, 1.0]
        elif regime == "narrow_broad":
            w, m, s = [0.35, 0.65], [2.0, -2.0], [0.45, 1.6]
        elif regime == "vanishing":
            w, m, s = [0.5 + 0.035, 0.5 - 0.035], [1.8, -1.8], [1.0, 1.0]
        elif regime == "nearly_tied":
            w, m, s = [0.5 + 0.0015, 0.5 - 0.0015], [2.0, -2.0], [1.0, 1.0]
        elif regime == "separated":
            w, m, s = [0.5, 0.5], [5.0, -5.0], [0.7, 0.7]
        else:
            raise ValueError(f"unknown regime {regime!r}")
        m = torch.tensor(m, dtype=torch.float64)
        if jitter:
            m = m + jitter * torch.randn(m.shape, generator=rng, dtype=torch.float64)
        factors.append(_mix(w, m, s))
    return FactorModel(
        prior_mean=torch.zeros(1, dtype=torch.float64),
        prior_cov=torch.tensor([[tau ** 2]], dtype=torch.float64),
        factors=factors, name=f"bimodal1d/{regime}",
        meta={"regime": regime, "tau": tau},
    )


def _calibrate_two_mode_weight(mu, s1, s2, n, delta):
    """Mixture weight ``w`` whose *tall* two-mode log-height gap is ``delta``.

    For a product of ``n`` identical two-component factors the tall log density
    is ``n log L(g) + log p(g)``, so the gap between the two modes grows like
    ``n`` for any fixed factor.  Keeping several modes alive at ``n = 100``
    therefore requires an ``O(1/n)`` per-observation tilt; this solves for it
    exactly by bisection on the factor log-density gap rather than using the
    well-separated asymptotic.  ``delta > 0`` puts the higher peak at ``+mu``.
    """
    target = delta / n

    def gap(w):
        f = _mix([w, 1.0 - w], [mu, -mu], [s1, s2])
        pts = torch.tensor([[mu], [-mu]], dtype=torch.float64)
        v = f.log_prob(pts)
        return float(v[0] - v[1])

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gap(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tilted_two_mode_1d(n, delta=0.0, mu=2.0, s1=1.0, s2=1.0, tau=4.0):
    """Two global modes whose tall log-height gap is ``delta`` at every ``n``.

    This is the regime in which mode *weights* stay non-degenerate as ``n``
    grows, which is what Milestones F/G actually have to get right.  The fixed
    per-observation tilts (``bimodal_1d('asymmetric')``) are the complementary
    "mode disappears with n" test of section 16.
    """
    w = _calibrate_two_mode_weight(mu, s1, s2, n, delta)
    factors = [_mix([w, 1.0 - w], [mu, -mu], [s1, s2]) for _ in range(n)]
    return FactorModel(
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([[tau ** 2]], dtype=torch.float64),
        factors, name=f"tilted2/d{delta}",
        meta={"delta": delta, "weight": w, "s1": s1, "s2": s2, "mu": mu},
    )


def map_vs_mass_1d(n, tau=4.0, delta=0.5, s1=0.40, s2=1.30, mu=2.0):
    """Section 13: mode A is the higher peak, mode B carries more mass.

    ``A`` at ``+mu`` is narrow (``s1``) and by construction ``delta`` higher in
    log density; ``B`` at ``-mu`` is broad (``s2``).  Since the tall mode widths
    scale as ``s/sqrt(n)`` the Laplace mass ratio is ``exp(delta) * s1/s2``,
    which is below 1 for the defaults at every ``n`` -- so the MAP branch and
    the largest-mass branch are different objects at all ``n``, which is exactly
    what the implementation must not conflate.
    """
    return tilted_two_mode_1d(n, delta=delta, mu=mu, s1=s1, s2=s2, tau=tau)


def empty_valley_1d(n, tau=6.0, separation=5.0):
    """Section 14: two modes with essentially zero density between them."""
    factors = [
        _mix([0.5, 0.5], [separation, -separation], [0.6, 0.6]) for _ in range(n)
    ]
    return FactorModel(
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([[tau ** 2]], dtype=torch.float64),
        factors, name="empty_valley", meta={"separation": separation},
    )


# ----------------------------------------------------------------------------
# 2-D regimes (A2)
# ----------------------------------------------------------------------------

def calibrate_component_weights(means, covs, targets, n):
    """Component weights giving prescribed *tall* mode log-height gaps.

    For well-separated components the factor's log density at mode ``k`` is
    ``log w_k - 1/2 log|2 pi S_k|``, so the tall gap between modes ``k`` and
    ``0`` is ``n`` times the difference of those.  Solving for ``log w_k``,

        log w_k = targets[k] / n + 1/2 ( d log 2pi + log|S_k| ) + const.

    Tempering the *weights* alone is not enough: the ``-1/2 log|S_k|`` volume
    term is multiplied by ``n`` as well, so two components differing only in
    covariance (``narrow_broad``, ``unequal_covariance``) still lose a mode
    exponentially fast.  Putting the volume term into the calibration is what
    keeps a 2-D benchmark genuinely multimodal at ``n = 100``; ``targets`` are
    in nats of *tall* log height relative to component 0, so the mode structure
    is fixed along the whole n ladder by construction.
    """
    covs = torch.as_tensor(covs, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64).reshape(-1)
    logdet = torch.linalg.slogdet(covs).logabsdet
    dim = covs.shape[-1]
    log_w = targets / n + 0.5 * (dim * math.log(2.0 * math.pi) + logdet)
    return torch.softmax(log_w, dim=0)


def four_mode_2d(n, regime="symmetric", tau=3.0, radius=2.0):
    """2-D benchmark with 2-4 separated global modes.

    Component weights come from :func:`calibrate_component_weights`, so the
    prescribed tall mode heights hold at every ``n``.
    """
    corners = torch.tensor(
        [[radius, radius], [-radius, radius], [-radius, -radius], [radius, -radius]],
        dtype=torch.float64,
    )
    if regime == "symmetric":
        targets = torch.zeros(4, dtype=torch.float64)
        covs = torch.stack([torch.eye(2, dtype=torch.float64)] * 4)
    elif regime == "unequal_weights":
        targets = torch.tensor([0.0, -0.8, -1.5, -2.5], dtype=torch.float64)
        covs = torch.stack([torch.eye(2, dtype=torch.float64)] * 4)
    elif regime == "unequal_covariance":
        # heights exactly tied, masses set purely by the mode volumes: the
        # cleanest possible separation of "MAP" from "largest mode mass"
        targets = torch.zeros(4, dtype=torch.float64)
        covs = torch.stack([
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.4, 0.15], [0.15, 1.6]], dtype=torch.float64),
            torch.tensor([[1.8, -0.5], [-0.5, 0.6]], dtype=torch.float64),
            torch.tensor([[0.7, 0.0], [0.0, 0.7]], dtype=torch.float64),
        ])
    elif regime == "narrow_broad":
        # the 2-D map_vs_mass: the narrow mode is the higher peak, the broad one
        # carries the larger mass
        targets = torch.tensor([0.6, 0.0], dtype=torch.float64)
        corners = corners[:2]
        covs = torch.stack([
            0.25 * torch.eye(2, dtype=torch.float64),
            1.60 * torch.eye(2, dtype=torch.float64),
        ])
    elif regime == "remote_small":
        # section 15: a dominant central mode plus a remote mode carrying a few
        # per cent of the mass that no bifurcation of the dominant mode reaches.
        corners = torch.tensor([[0.0, 0.0], [4.2, 3.4]], dtype=torch.float64)
        # targets are *likelihood* gaps; the global prior contributes once (not
        # n times), and at radius 5.4 with tau = 3 that alone is -1.6 nats, so a
        # zero target here lands the remote mode at ~10% of the mass at every n
        targets = torch.tensor([0.0, 0.0], dtype=torch.float64)
        covs = torch.stack([
            1.0 * torch.eye(2, dtype=torch.float64),
            0.55 * torch.eye(2, dtype=torch.float64),
        ])
    else:
        raise ValueError(regime)
    w = calibrate_component_weights(corners, covs, targets, n)
    factors = [GaussianMixture(w, corners, covs) for _ in range(n)]
    return FactorModel(
        torch.zeros(2, dtype=torch.float64),
        (tau ** 2) * torch.eye(2, dtype=torch.float64),
        factors, name=f"four_mode_2d/{regime}",
        meta={"regime": regime, "targets": targets.tolist(),
              "weights": w.tolist()},
    )


# ----------------------------------------------------------------------------
# Hierarchical regimes (A3)
# ----------------------------------------------------------------------------

def hierarchical(n, d_g=1, d_l=1, modes=2, tau=3.0, local_scale=1.0,
                 coupling=0.6, seed=0, separation=2.0):
    """Hierarchical benchmark ``y_i = (g, l_i)`` with a multimodal global block.

    The row factors are Gaussian mixtures over ``(g, l_i)`` whose components sit
    at alternating global offsets, so the *global marginal* posterior is
    multimodal while the local block stays conditionally Gaussian.  The exact
    global marginal is available through :meth:`FactorModel.global_log_prob`.
    """
    rng = torch.Generator().manual_seed(seed)
    dim = d_g + d_l
    signs = torch.tensor(
        [(-1.0) ** k for k in range(modes)], dtype=torch.float64
    )
    factors = []
    for i in range(n):
        means, covs, weights = [], [], []
        for k in range(modes):
            centre = torch.zeros(dim, dtype=torch.float64)
            centre[:d_g] = separation * signs[k] * torch.ones(d_g, dtype=torch.float64)
            centre[:d_g] += 0.15 * torch.randn(d_g, generator=rng, dtype=torch.float64)
            centre[d_g:] = 0.3 * torch.randn(d_l, generator=rng, dtype=torch.float64)
            cov = torch.eye(dim, dtype=torch.float64)
            cov[:d_g, :d_g] *= 1.0
            cov[d_g:, d_g:] *= local_scale ** 2
            for a in range(min(d_g, d_l)):
                cov[a, d_g + a] = coupling * local_scale
                cov[d_g + a, a] = coupling * local_scale
            means.append(centre)
            covs.append(cov)
            weights.append(1.0 / modes)
        factors.append(GaussianMixture(
            torch.tensor(weights, dtype=torch.float64),
            torch.stack(means), torch.stack(covs),
        ))
    return FactorModel(
        prior_mean=torch.zeros(d_g, dtype=torch.float64),
        prior_cov=(tau ** 2) * torch.eye(d_g, dtype=torch.float64),
        factors=factors,
        local_map=torch.zeros(d_l, d_g, dtype=torch.float64),
        local_mean=torch.zeros(d_l, dtype=torch.float64),
        local_cov=(local_scale ** 2) * torch.eye(d_l, dtype=torch.float64),
        name=f"hier/dg{d_g}_dl{d_l}_K{modes}",
        meta={"d_g": d_g, "d_l": d_l, "modes": modes},
    )


REGIMES_1D = (
    "symmetric", "asymmetric", "narrow_broad",
    "vanishing", "nearly_tied", "separated",
)
N_LADDER = (1, 2, 4, 8, 16, 32, 64, 100)
