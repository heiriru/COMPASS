"""Milestone B -- the exact clean (t = 0) compositional score.

The tall posterior of the hierarchical model

    p(g, l_1..l_n | X)  ∝  p(g)^{1-n} prod_i p(g, l_i | x_i)

(verified against the benchmark model in ``tests/test_clean_composition.py``)
gives the block score

    S_0,g   = sum_i s_i,g - (n-1) s_prior,g
    S_0,l_i = s_i,l_i

which is evaluated here in the cancellation-free form

    S_0,g = s_prior,g + sum_i [ s_i,g - s_prior,g ]

so that the two separately ``O(n)`` terms are never formed.  ``s_prior`` is the
score of the **marginal** global prior ``p(g)``: the conditional prior
``p(l_i|g)`` enters once per row inside ``s_i`` and needs no extra correction.

The same file carries the arrowhead curvature of the composed field,

    H = [[A, B_1 .. B_n], [B_i^T, C_i]],
    A = H_prior + sum_i (H_i,gg - H_prior),  B_i = H_i,gl,  C_i = H_i,ll,

and its Schur-eliminated marginal global precision
``P_g = -(A - sum_i B_i C_i^-1 B_i^T)``, built with ``torch.linalg.solve`` /
Cholesky and never an explicit inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


class RowScoreSource:
    """Interface the composer needs: per-row clean scores (and curvature).

    ``score(i, y)`` takes ``y`` of shape ``(..., d_g + d_l)`` and returns the
    same shape.  ``hessian`` may be omitted, in which case it is obtained by
    autograd on ``score`` (Jacobian of a vector field, symmetrised on request).
    """

    d_g: int
    d_l: int
    n: int

    def score(self, i, y):                       # pragma: no cover - interface
        raise NotImplementedError

    def hessian(self, i, y):
        return _autograd_jacobian(lambda z: self.score(i, z), y)

    def prior_score(self, g):                    # pragma: no cover - interface
        raise NotImplementedError

    def prior_hessian(self, g):
        return _autograd_jacobian(self.prior_score, g)


def _autograd_jacobian(fn, y):
    """Jacobian of a vector field at a single point (float64)."""
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    return torch.autograd.functional.jacobian(fn, y, vectorize=True)


class OracleRowSource(RowScoreSource):
    """Exact per-row scores/Hessians of a :class:`benchmarks.FactorModel`."""

    def __init__(self, model):
        self.model = model
        self.d_g, self.d_l, self.n = model.d_g, model.d_l, model.n

    def score(self, i, y):
        return self.model.row_score(i, y)

    def hessian(self, i, y):
        return self.model.row_hessian(i, y)

    def prior_score(self, g):
        return self.model.prior_score(g)

    def prior_hessian(self, g):
        return self.model.prior_hessian(g)


# ----------------------------------------------------------------------------


@dataclass
class ArrowBlocks:
    """Arrowhead curvature blocks of the composed clean log density."""

    A: torch.Tensor                    # (d_g, d_g)  = global block of the Hessian
    B: torch.Tensor | None             # (n, d_g, d_l)
    C: torch.Tensor | None             # (n, d_l, d_l)

    def global_precision(self):
        """``P_g = -(A - sum_i B_i C_i^-1 B_i^T)``, SPD at a genuine mode."""
        marginal = self.A
        if self.B is not None:
            solved = torch.linalg.solve(self.C, self.B.mT)          # (n, d_l, d_g)
            marginal = marginal - torch.einsum("igl,ilh->gh", self.B, solved)
        precision = -marginal
        return 0.5 * (precision + precision.mT)

    def log_det_negative_hessian(self):
        """``log|H|`` with ``H = -Hessian``, via ``log|P_g| + sum_i log|-C_i|``.

        Uses Cholesky log-determinants; never ``torch.linalg.det``.
        """
        total = _chol_logdet(self.global_precision())
        if self.C is not None:
            total = total + _chol_logdet(-self.C).sum()
        return total

    def dimension(self):
        d_g = self.A.shape[-1]
        return d_g + (0 if self.C is None else self.C.shape[0] * self.C.shape[-1])


def _chol_logdet(matrix):
    factor, info = torch.linalg.cholesky_ex(matrix)
    if torch.any(info != 0):
        raise RuntimeError("log-determinant requested for a non-SPD matrix")
    return 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)


class CleanComposer:
    """Exact clean compositional score, curvature and Schur elimination."""

    def __init__(self, source: RowScoreSource, n=None):
        self.source = source
        self.d_g = source.d_g
        self.d_l = source.d_l
        self.n = source.n if n is None else int(n)
        self.score_evaluations = 0
        self.hessian_evaluations = 0

    # -- state layout ---------------------------------------------------------
    @property
    def dim(self):
        return self.d_g + self.n * self.d_l

    def row_state(self, state, i):
        g = state[..., : self.d_g]
        if not self.d_l:
            return g
        start = self.d_g + i * self.d_l
        return torch.cat([g, state[..., start:start + self.d_l]], dim=-1)

    def split(self, state):
        g = state[..., : self.d_g]
        locals_ = None
        if self.d_l:
            locals_ = state[..., self.d_g:].reshape(
                state.shape[:-1] + (self.n, self.d_l)
            )
        return g, locals_

    # -- the composed score ---------------------------------------------------
    def score(self, state):
        """``S_0`` on the full state ``(..., d_g + n*d_l)``."""
        state = torch.as_tensor(state, dtype=torch.float64)
        g = state[..., : self.d_g]
        prior = self.source.prior_score(g)
        increment = torch.zeros_like(prior)
        local_parts = []
        for i in range(self.n):
            s_i = self.source.score(i, self.row_state(state, i))
            increment = increment + (s_i[..., : self.d_g] - prior)
            if self.d_l:
                local_parts.append(s_i[..., self.d_g:])
        self.score_evaluations += self.n
        out = prior + increment
        if self.d_l:
            out = torch.cat([out] + local_parts, dim=-1)
        return out

    def global_score_marginal(self, g):
        """``S_0`` of the *marginal* global posterior, for all-global models."""
        assert self.d_l == 0
        return self.score(g)

    # -- curvature ------------------------------------------------------------
    def blocks(self, state):
        """Arrowhead blocks at a single point ``state`` of shape ``(dim,)``."""
        state = torch.as_tensor(state, dtype=torch.float64).reshape(-1)
        g = state[: self.d_g]
        H_prior = self.source.prior_hessian(g)
        A = H_prior.clone()
        B = [] if self.d_l else None
        C = [] if self.d_l else None
        for i in range(self.n):
            H_i = self.source.hessian(i, self.row_state(state, i))
            A = A + (H_i[: self.d_g, : self.d_g] - H_prior)
            if self.d_l:
                B.append(H_i[: self.d_g, self.d_g:])
                C.append(H_i[self.d_g:, self.d_g:])
        self.hessian_evaluations += self.n
        A = 0.5 * (A + A.mT)
        if self.d_l:
            B = torch.stack(B)
            C = torch.stack(C)
            C = 0.5 * (C + C.mT)
        return ArrowBlocks(A=A, B=B, C=C)

    def dense_hessian(self, state):
        """Full ``dim x dim`` Hessian; for tests and tiny problems only."""
        blocks = self.blocks(state)
        dim = self.dim
        H = torch.zeros(dim, dim, dtype=torch.float64)
        H[: self.d_g, : self.d_g] = blocks.A
        if self.d_l:
            for i in range(self.n):
                a = self.d_g + i * self.d_l
                b = a + self.d_l
                H[: self.d_g, a:b] = blocks.B[i]
                H[a:b, : self.d_g] = blocks.B[i].mT
                H[a:b, a:b] = blocks.C[i]
        return H

    def newton_step(self, state, blocks=None):
        """Arrowhead Newton step ``-H^{-1} S_0`` using block elimination.

        Never forms the dense Hessian and never inverts ``C_i``.
        """
        state = torch.as_tensor(state, dtype=torch.float64).reshape(-1)
        blocks = self.blocks(state) if blocks is None else blocks
        s = self.score(state)
        s_g = s[: self.d_g]
        if not self.d_l:
            return torch.linalg.solve(-blocks.A, s_g)
        s_l = s[self.d_g:].reshape(self.n, self.d_l)
        solved_gradient = torch.linalg.solve(
            blocks.C, s_l.unsqueeze(-1)
        )                                                    # (n, d_l, 1)
        solved_cross = torch.linalg.solve(blocks.C, blocks.B.mT)   # (n, d_l, d_g)
        marginal = blocks.A - torch.einsum("igl,ilh->gh", blocks.B, solved_cross)
        rhs = s_g - torch.einsum("igl,ilj->gj", blocks.B, solved_gradient).reshape(-1)
        delta_g = torch.linalg.solve(-marginal, rhs)
        delta_l = -(solved_gradient.squeeze(-1)
                    + torch.einsum("ilg,g->il", solved_cross, delta_g))
        return torch.cat([delta_g, delta_l.reshape(-1)])


# ----------------------------------------------------------------------------
# Marginal (locals-integrated) composer for all-global reasoning
# ----------------------------------------------------------------------------

class MarginalGlobalComposer(CleanComposer):
    """Composer on the global block alone, with each row's locals profiled out.

    Uses the row *marginal* posteriors ``p(g | x_i)``, which the oracle model
    provides in closed form.  This is the object the mode weights and the
    Laplace masses in Milestones F/G are defined on for hierarchical models
    when only the global posterior is wanted.
    """

    def __init__(self, model):
        self.model = model
        source = _MarginalSource(model)
        super().__init__(source)

    @property
    def dim(self):
        return self.d_g


class _MarginalSource(RowScoreSource):
    def __init__(self, model):
        self.model = model
        self.d_g, self.d_l, self.n = model.d_g, 0, model.n
        self._marginals = [model.row_global_marginal(i) for i in range(model.n)]

    def score(self, i, y):
        return self._marginals[i].score(y)

    def hessian(self, i, y):
        return self._marginals[i].hessian(y)

    def prior_score(self, g):
        return self.model.prior_score(g)

    def prior_hessian(self, g):
        return self.model.prior_hessian(g)


# ----------------------------------------------------------------------------
# Vectorised oracle source: all n rows evaluated in one batched call
# ----------------------------------------------------------------------------

class BatchedGaussianMixtures:
    """``n`` Gaussian mixtures with a common component count, evaluated at once.

    Shapes: ``weights (n, K)``, ``means (n, K, D)``, ``covs (n, K, D, D)``.
    All methods take ``y`` of shape ``(..., n, D)`` and keep the leading batch.
    """

    def __init__(self, weights, means, covs):
        self.weights = weights.to(torch.float64)
        self.means = means.to(torch.float64)
        self.covs = covs.to(torch.float64)
        self.n, self.K, self.dim = self.means.shape
        self.chol = torch.linalg.cholesky(self.covs)
        logdet = 2.0 * torch.log(
            torch.diagonal(self.chol, dim1=-2, dim2=-1)
        ).sum(-1)
        self._log_const = (
            torch.log(self.weights)
            - 0.5 * self.dim * TWO_PI_LOG_C - 0.5 * logdet
        )
        self._inv = torch.cholesky_inverse(self.chol)

    def _terms(self, y):
        delta = y.unsqueeze(-2) - self.means                 # (..., n, K, D)
        sol = torch.cholesky_solve(delta.unsqueeze(-1), self.chol).squeeze(-1)
        quad = (delta * sol).sum(-1)
        return self._log_const + (-0.5) * quad, -sol

    def log_prob(self, y):
        log_k, _ = self._terms(y)
        return torch.logsumexp(log_k, dim=-1)

    def score(self, y):
        log_k, u = self._terms(y)
        r = torch.softmax(log_k, dim=-1)
        return (r.unsqueeze(-1) * u).sum(-2)

    def hessian(self, y):
        log_k, u = self._terms(y)
        r = torch.softmax(log_k, dim=-1)
        outer = u.unsqueeze(-1) * u.unsqueeze(-2)
        first = (r[..., None, None] * (-self._inv + outer)).sum(-3)
        mean_u = (r.unsqueeze(-1) * u).sum(-2)
        return first - mean_u.unsqueeze(-1) * mean_u.unsqueeze(-2)

    def marginal(self, keep):
        idx = torch.as_tensor(list(keep))
        return BatchedGaussianMixtures(
            self.weights, self.means[:, :, idx],
            self.covs[:, :, idx][:, :, :, idx],
        )


TWO_PI_LOG_C = math.log(2.0 * math.pi)


def _batched_row_posteriors(model):
    """``p(y_i | x_i)`` for every row at once, as a batched Gaussian mixture."""
    joint = model._joint_prior()
    mu0 = joint.means[0]
    S0 = joint.covs[0]
    P0 = torch.cholesky_inverse(torch.linalg.cholesky(S0))
    K = model.factors[0].means.shape[0]
    weights = torch.stack([f.weights for f in model.factors])
    means = torch.stack([f.means for f in model.factors])
    covs = torch.stack([f.covs for f in model.factors])
    chol = torch.linalg.cholesky(covs)
    Pb = torch.cholesky_inverse(chol)
    P = P0 + Pb
    cov = torch.cholesky_inverse(torch.linalg.cholesky(P))
    cov = 0.5 * (cov + cov.mT)
    rhs = (P0 @ mu0) + torch.einsum("nkij,nkj->nki", Pb, means)
    mean = torch.einsum("nkij,nkj->nki", cov, rhs)
    # normalisation N(m_k; mu0, S0 + S_k)
    total = S0 + covs
    tc = torch.linalg.cholesky(total)
    delta = (means - mu0).unsqueeze(-1)
    sol = torch.cholesky_solve(delta, tc)
    logdet = 2.0 * torch.log(torch.diagonal(tc, dim1=-2, dim2=-1)).sum(-1)
    log_c = (
        -0.5 * (delta * sol).sum((-2, -1))
        - 0.5 * means.shape[-1] * TWO_PI_LOG_C - 0.5 * logdet
    )
    log_w = torch.log(weights) + log_c
    log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
    return BatchedGaussianMixtures(log_w.exp(), mean, cov)


class BatchedOracleRowSource(RowScoreSource):
    """Exact row scores/Hessians for every row in one vectorised call.

    Mathematically identical to :class:`OracleRowSource`; it exists because the
    per-row Python loop is what makes ``n = 100`` slow, and the runtime numbers
    this project reports must not be dominated by that.
    """

    batched = True

    def __init__(self, model, marginal=False):
        self.model = model
        self.d_g = model.d_g
        self.d_l = 0 if marginal else model.d_l
        self.n = model.n
        posteriors = _batched_row_posteriors(model)
        self.rows = (posteriors.marginal(range(model.d_g)) if marginal
                     else posteriors)

    def scores_all(self, rows_state):
        return self.rows.score(rows_state)

    def hessians_all(self, rows_state):
        return self.rows.hessian(rows_state)

    def score(self, i, y):
        raise NotImplementedError("use scores_all")

    def prior_score(self, g):
        return self.model.prior_score(g)

    def prior_hessian(self, g):
        return self.model.prior_hessian(g)


def _rows_state(composer, state):
    """Stack the per-row states ``(..., n, d_g + d_l)``."""
    g = state[..., : composer.d_g]
    expanded = g.unsqueeze(-2).expand(g.shape[:-1] + (composer.n, composer.d_g))
    if not composer.d_l:
        return expanded
    locals_ = state[..., composer.d_g:].reshape(
        state.shape[:-1] + (composer.n, composer.d_l)
    )
    return torch.cat([expanded, locals_], dim=-1)


class BatchedCleanComposer(CleanComposer):
    """:class:`CleanComposer` for sources that evaluate all rows at once."""

    def score(self, state):
        state = torch.as_tensor(state, dtype=torch.float64)
        g = state[..., : self.d_g]
        prior = self.source.prior_score(g)
        rows = self.source.scores_all(_rows_state(self, state))   # (..., n, D)
        self.score_evaluations += self.n
        global_part = prior + (rows[..., : self.d_g] - prior.unsqueeze(-2)).sum(-2)
        if not self.d_l:
            return global_part
        locals_ = rows[..., self.d_g:].reshape(
            state.shape[:-1] + (self.n * self.d_l,)
        )
        return torch.cat([global_part, locals_], dim=-1)

    def global_curvature(self, states):
        """Batched global Hessian block ``A`` for all-global states.

        ``states`` is ``(..., d_g)``; returns ``(..., d_g, d_g)``.  Used by
        :func:`modetree.refine.refine_batch` so a whole probe set is refined in
        one pass.
        """
        assert not self.d_l
        states = torch.as_tensor(states, dtype=torch.float64)
        H_prior = self.source.prior_hessian(states)
        H = self.source.hessians_all(_rows_state(self, states))
        self.hessian_evaluations += self.n
        A = H_prior + (H - H_prior.unsqueeze(-3)).sum(-3)
        return 0.5 * (A + A.mT)

    def blocks(self, state):
        state = torch.as_tensor(state, dtype=torch.float64).reshape(-1)
        g = state[: self.d_g]
        H_prior = self.source.prior_hessian(g)
        H = self.source.hessians_all(_rows_state(self, state))     # (n, D, D)
        self.hessian_evaluations += self.n
        A = H_prior + (H[:, : self.d_g, : self.d_g] - H_prior).sum(0)
        A = 0.5 * (A + A.mT)
        if not self.d_l:
            return ArrowBlocks(A=A, B=None, C=None)
        B = H[:, : self.d_g, self.d_g:]
        C = H[:, self.d_g:, self.d_g:]
        C = 0.5 * (C + C.mT)
        return ArrowBlocks(A=A, B=B, C=C)


def make_composer(model, marginal=False, batched=True):
    """The composer used everywhere: batched by default, loop-based on request."""
    if batched:
        return BatchedCleanComposer(BatchedOracleRowSource(model, marginal=marginal))
    if marginal:
        return MarginalGlobalComposer(model)
    return CleanComposer(OracleRowSource(model))
