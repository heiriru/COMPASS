"""Milestones F and G -- structured Laplace mode masses, and Laplace+.

F.  For a converged mode ``b`` the arrowhead curvature gives
``H_b = -grad^2 log p`` and, by block elimination,
``P_g,b = A_b - sum_i B_ib C_ib^-1 B_ib^T`` and
``log|H_b| = log|P_g,b| + sum_i log|-C_ib|``.  The branch Laplace log evidence
up to a constant common to all branches is

    log Z_b = E_b - 1/2 log|H_b|,        w_b = softmax_b(log Z_b)

with ``E_b`` the reconstructed mode energy from :mod:`modetree.energy`.  The
MAP branch (``argmax_b E_b``) and the largest-mass branch (``argmax_b log Z_b``)
are computed and reported separately and are never conflated.

G.  Laplace+ corrects the within-branch non-Gaussianity without sampling.  With
``Delta_b(g) = log p(g) - log p(g_b)`` from score integration and the Gaussian
prediction ``-1/2 (g-g_b)^T P_b (g-g_b)``, the residual ``R_b`` vanishes exactly
for a Gaussian mode, and

    C_b = E_{g ~ q_b}[ exp(R_b(g)) ],     log Z_b^+ = log Z_b + log C_b

estimated with a randomised-Sobol (QMC) point set and a log-sum-exp.  The same
weights ``omega_m ∝ exp(R_b(g_m))`` reweight within-branch draws, and their ESS
is reported -- if the ESS collapses, Laplace+ is not working and this module
says so rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .energy import composite_nodes, integrate_polyline, mode_energies


LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class BranchLaplace:
    theta: torch.Tensor
    global_state: torch.Tensor
    energy: float
    log_det_H: float
    log_det_global: float
    global_precision: torch.Tensor
    global_covariance: torch.Tensor
    log_evidence: float
    log_evidence_plus: float | None = None
    log_correction: float | None = None
    ess: float | None = None
    max_weight: float | None = None
    log_weight_variance: float | None = None
    diagnostics: dict = field(default_factory=dict)


def branch_laplace(composer, mode_result, energy):
    """Structured Laplace quantities for one converged branch."""
    blocks = mode_result.blocks
    P = blocks.global_precision()
    factor, info = torch.linalg.cholesky_ex(P)
    if torch.any(info != 0):
        raise RuntimeError("branch global precision is not positive definite")
    log_det_global = float(2.0 * torch.log(torch.diagonal(factor)).sum())
    log_det_H = float(blocks.log_det_negative_hessian())
    covariance = torch.cholesky_inverse(factor)
    covariance = 0.5 * (covariance + covariance.mT)
    return BranchLaplace(
        theta=mode_result.theta,
        global_state=mode_result.theta[: composer.d_g],
        energy=float(energy),
        log_det_H=log_det_H,
        log_det_global=log_det_global,
        global_precision=P,
        global_covariance=covariance,
        log_evidence=float(energy) - 0.5 * log_det_H,
        diagnostics={
            "score_norm": mode_result.score_norm,
            "precision_eigenvalues": torch.linalg.eigvalsh(P).tolist(),
        },
    )


def softmax_weights(log_values):
    values = torch.as_tensor(log_values, dtype=torch.float64)
    return torch.softmax(values, dim=0)


# ----------------------------------------------------------------------------
# Laplace+  (Milestone G)
# ----------------------------------------------------------------------------

def _sobol_points(dim, count, seed=0):
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    u = engine.draw(count).to(torch.float64)
    # map the unit cube to standard normals; clamp away the exact 0/1 endpoints
    u = u.clamp(1e-12, 1.0 - 1e-12)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)


def laplace_plus(composer, branch, samples=32, seed=0, q=16, tol=1e-8,
                 antithetic=True, basin_guard=True, guard_points=33,
                 guard_tol=1e-6):
    """Local normalisation correction ``log C_b`` and its weight diagnostics.

    The residual needs ``Delta_b(g_m) = log p(g_m) - log p(g_b)``, which is a
    score line integral from the mode to each QMC point.  All ``M`` segments and
    all ``Q`` quadrature nodes are evaluated in a single batched call to the
    composed score, so the whole correction costs two batched evaluations.

    Basin guard
    -----------
    ``C_b = E_{q_b}[exp(R_b)]`` integrates over the *whole* space, while the
    quantity wanted is the mass of branch ``b``'s own basin.  When the branches
    are well separated the two agree, because ``q_b`` puts no mass beyond the
    saddle.  When they are not -- small ``n``, or modes within a couple of
    standard deviations -- the uncorrected estimator picks up the neighbouring
    mode and every branch converges on the *total* mass, driving the weights
    toward uniform.  That is not a subtlety: without the guard Laplace+ is
    reliably **worse** than plain Laplace at small ``n`` (measured, see the
    report).

    The guard keeps a point only if the log density decreases monotonically
    along the segment from the mode to it: a segment that dips and rises again
    has crossed a separating minimum and left the basin.  The test reuses the
    scores that are being integrated anyway, so it is nearly free, and it is
    exactly a no-op for well-separated branches.  In ``d > 1`` it is
    conservative (a straight chord can dip without leaving the basin), which
    biases ``C_b`` down rather than double-counting a neighbour.
    """
    d = composer.d_g
    z = _sobol_points(d, samples, seed=seed)
    if antithetic:
        z = torch.cat([z, -z], dim=0)
    chol = torch.linalg.cholesky(branch.global_covariance)
    points = branch.global_state.unsqueeze(0) + z @ chol.mT       # (M, d)
    delta = points - branch.global_state                          # (M, d)
    M = points.shape[0]

    def batched_delta(panels):
        nodes, weights = composite_nodes(q, panels)
        grid = (branch.global_state.reshape(1, 1, d)
                + nodes.reshape(1, -1, 1) * delta.unsqueeze(1))   # (M, Q, d)
        scores = _global_score(composer, branch, grid)
        integrand = (scores * delta.unsqueeze(1)).sum(-1)         # (M, Q)
        return (weights.reshape(1, -1) * integrand).sum(-1), M * nodes.numel()

    deltas, evaluations = batched_delta(1)
    refined, extra = batched_delta(2)
    evaluations += extra
    quadrature_error = float((refined - deltas).abs().max())
    deltas = refined

    accepted = torch.ones(M, dtype=torch.bool)
    if basin_guard:
        u = torch.linspace(0.0, 1.0, guard_points, dtype=torch.float64)
        grid = (branch.global_state.reshape(1, 1, d)
                + u.reshape(1, -1, 1) * delta.unsqueeze(1))
        scores = _global_score(composer, branch, grid)
        evaluations += M * guard_points
        integrand = (scores * delta.unsqueeze(1)).sum(-1)         # (M, U)
        profile = torch.cumulative_trapezoid(integrand, u, dim=-1)
        profile = torch.cat([torch.zeros(M, 1, dtype=torch.float64), profile], -1)
        interior_min = profile[:, :-1].min(dim=-1).values
        # monotone descent => the running minimum is the endpoint itself
        accepted = interior_min >= profile[:, -1] - guard_tol

    residuals_all = deltas + 0.5 * torch.einsum(
        "md,de,me->m", delta, branch.global_precision, delta
    )
    residuals = residuals_all[accepted]
    if residuals.numel() == 0:
        residuals = torch.zeros(1, dtype=torch.float64)
    log_c = torch.logsumexp(residuals, dim=0) - math.log(M)
    log_w = residuals - torch.logsumexp(residuals, dim=0)
    w = log_w.exp()
    ess = float(1.0 / (w * w).sum())
    branch.log_correction = float(log_c)
    branch.log_evidence_plus = branch.log_evidence + float(log_c)
    branch.ess = ess
    branch.max_weight = float(w.max())
    branch.log_weight_variance = float(residuals.var(unbiased=False))
    branch.diagnostics["laplace_plus"] = {
        "samples": M,
        "accepted": int(accepted.sum()),
        "ess_fraction": ess / M,
        "max_residual": float(residuals.abs().max()),
        "score_evaluations": evaluations,
        "quadrature_error": quadrature_error,
        "log_weights": residuals,
    }
    return branch


def _global_score(composer, branch, batch):
    """Composed score on the global block for an all-global composer.

    For hierarchical models the locals must be profiled out first; the
    ``MarginalGlobalComposer`` of :mod:`modetree.composition` does that exactly
    for the oracle benchmarks, and the profile-likelihood variant is used for
    the learned network (see ``modetree.hierarchical``).
    """
    if composer.d_l:
        raise NotImplementedError(
            "Laplace+ is defined on the global block; wrap a hierarchical model "
            "in MarginalGlobalComposer (or the profiled composer) first."
        )
    return composer.score(batch)


# ----------------------------------------------------------------------------
# Whole-posterior assembly
# ----------------------------------------------------------------------------

@dataclass
class MultimodalPosterior:
    branches: list
    weights: torch.Tensor
    weights_plus: torch.Tensor | None
    map_index: int
    mass_index: int
    energy_graph: object
    diagnostics: dict = field(default_factory=dict)

    @property
    def map_global(self):
        return self.branches[self.map_index].global_state

    def sample(self, count, generator=None, use_plus=True):
        """Draw from ``sum_b w_b N(g_b, P_b^-1)``."""
        w = self.weights_plus if (use_plus and self.weights_plus is not None) else self.weights
        counts = torch.multinomial(w, count, replacement=True, generator=generator)
        draws = []
        for b, branch in enumerate(self.branches):
            k = int((counts == b).sum())
            if not k:
                continue
            chol = torch.linalg.cholesky(branch.global_covariance)
            z = torch.randn(k, branch.global_state.numel(), dtype=torch.float64,
                            generator=generator)
            draws.append(branch.global_state.unsqueeze(0) + z @ chol.mT)
        return torch.cat(draws, dim=0) if draws else torch.zeros(0, 1)


def assemble(composer, mode_results, multiple_paths=True, laplace_plus_samples=0,
             seed=0, tol=1e-10, laplace_plus_kwargs=None):
    """Milestones D-G end to end for a set of already-refined modes."""
    modes = [r.theta[: composer.d_g] for r in mode_results]

    def score_fn(batch):
        if composer.d_l:
            raise NotImplementedError
        return composer.score(batch)

    graph = mode_energies(
        score_fn, modes, multiple_paths=multiple_paths, tol=tol
    )
    branches = [
        branch_laplace(composer, r, graph.energies[b])
        for b, r in enumerate(mode_results)
    ]
    log_z = torch.tensor([b.log_evidence for b in branches], dtype=torch.float64)
    weights = softmax_weights(log_z)
    weights_plus = None
    if laplace_plus_samples:
        for b, branch in enumerate(branches):
            laplace_plus(composer, branch, samples=laplace_plus_samples,
                         seed=seed + b, **(laplace_plus_kwargs or {}))
        weights_plus = softmax_weights(
            torch.tensor([b.log_evidence_plus for b in branches], dtype=torch.float64)
        )
    energies = torch.tensor([b.energy for b in branches], dtype=torch.float64)
    return MultimodalPosterior(
        branches=branches, weights=weights, weights_plus=weights_plus,
        map_index=int(torch.argmax(energies)),
        mass_index=int(torch.argmax(
            torch.tensor([b.log_evidence for b in branches], dtype=torch.float64)
        )),
        energy_graph=graph,
        diagnostics={
            "score_evaluations": graph.evaluations,
            "n_branches": len(branches),
        },
    )
