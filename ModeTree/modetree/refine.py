"""Milestone C -- deterministic mode refinement on the clean composed field.

Solves ``S_0(theta) = 0`` from a seed with a damped / trust-region Newton
iteration on the arrowhead system of :mod:`modetree.composition`, falling back
to a steepest-descent step on the merit ``m(theta) = ||S_0(theta)||^2`` when the
Newton direction is not a descent direction (which is exactly what happens near
a saddle, and is the signal Milestone I later listens to).

The dense Hessian of the full hierarchical state is never formed: the step comes
from ``CleanComposer.newton_step``, which eliminates the local blocks with
``torch.linalg.solve``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .composition import arrow_newton_batched


@dataclass
class ModeResult:
    theta: torch.Tensor
    score_norm: float
    iterations: int
    converged: bool
    blocks: object | None = None
    history: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def global_state(self):
        return self.theta[: self.diagnostics["d_g"]]


def refine_mode(composer, seed, tol=1e-9, max_iter=100, radius=None,
                verbose=False, require_mode=True):
    """Newton-refine ``seed`` to a stationary point of the composed log density.

    Parameters
    ----------
    tol : convergence threshold on ``||S_0||`` (float64; ``1e-7`` or below is
        reachable for the oracle benchmarks).
    radius : trust-region cap on the max-abs component of a step.  Defaults to
        one prior standard deviation-ish scale of 1.0, which only ever binds in
        the first couple of iterations.
    require_mode : mark the result unconverged if the stationary point is not a
        local maximum (negative-definite Hessian on the global block).
    """
    theta = torch.as_tensor(seed, dtype=torch.float64).reshape(-1).clone()
    radius = 1.0 if radius is None else float(radius)
    history = []
    blocks = None
    n_newton = 0
    n_fallback = 0
    for iteration in range(max_iter):
        blocks = composer.blocks(theta)
        score = composer.score(theta)
        norm = float(torch.linalg.vector_norm(score))
        history.append(norm)
        if norm < tol:
            break
        step = composer.newton_step(theta, blocks=blocks)
        n_newton += 1
        # merit-decreasing line search with a trust-region cap
        cap = radius / max(float(step.abs().max()), 1e-300)
        step = step * min(1.0, cap)
        merit = norm ** 2
        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            trial = theta + damping * step
            trial_norm = float(torch.linalg.vector_norm(composer.score(trial)))
            if trial_norm ** 2 < merit * (1.0 - 1e-4 * damping):
                theta, accepted = trial, True
                break
        if not accepted:
            # steepest descent on ||S||^2: grad m = 2 H S
            n_fallback += 1
            direction = -_hessian_vector(blocks, score, composer)
            scale = radius / max(float(direction.abs().max()), 1e-300)
            direction = direction * min(1.0, scale)
            for damping in (1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3):
                trial = theta + damping * direction
                trial_norm = float(torch.linalg.vector_norm(composer.score(trial)))
                if trial_norm ** 2 < merit:
                    theta, accepted = trial, True
                    break
        if not accepted:
            break
        if verbose:
            print(f"  it {iteration:3d} |S| = {norm:.3e}")

    blocks = composer.blocks(theta)
    score = composer.score(theta)
    norm = float(torch.linalg.vector_norm(score))
    is_max = True
    curvature = None
    try:
        precision = blocks.global_precision()
        curvature = torch.linalg.eigvalsh(precision)
        is_max = bool((curvature > 0).all())
        # A joint maximum needs the *whole* negative Hessian positive definite,
        # not only its global Schur complement.  Checking the global block alone
        # lets a branch through whose local block is indefinite, and the
        # structured Laplace determinant log|H| = log|P_g| + sum_i log|-C_i|
        # then fails downstream with a non-SPD Cholesky rather than at the point
        # where the branch was wrongly accepted.
        if is_max and blocks.C is not None:
            is_max = bool((torch.linalg.eigvalsh(-blocks.C) > 0).all())
    except RuntimeError:
        is_max = False
    converged = norm < max(tol * 1e3, 1e-7) and (is_max or not require_mode)
    return ModeResult(
        theta=theta, score_norm=norm, iterations=len(history),
        converged=converged, blocks=blocks, history=history,
        diagnostics={
            "d_g": composer.d_g,
            "newton_steps": n_newton,
            "fallback_steps": n_fallback,
            "global_precision_eigenvalues":
                None if curvature is None else curvature.tolist(),
            "is_local_maximum": is_max,
        },
    )


def _hessian_vector(blocks, vector, composer):
    """``H v`` from the arrowhead blocks (no dense Hessian)."""
    d_g = composer.d_g
    out = torch.zeros_like(vector)
    v_g = vector[:d_g]
    out[:d_g] = blocks.A @ v_g
    if composer.d_l:
        v_l = vector[d_g:].reshape(composer.n, composer.d_l)
        out[:d_g] = out[:d_g] + torch.einsum("igl,il->g", blocks.B, v_l)
        out[d_g:] = (
            torch.einsum("ilg,g->il", blocks.B.mT, v_g)
            + torch.einsum("ilm,im->il", blocks.C, v_l)
        ).reshape(-1)
    return out


def metric_from_precision(precision, floor=1e-6):
    """A genuinely positive-definite metric built from a curvature matrix.

    A branch that is still sitting on a saddle has an *indefinite* precision,
    and the raw quadratic form ``d^T P d`` is then zero or negative along the
    unstable direction.  Clamping that to zero makes every candidate look like a
    duplicate of the saddle, which silently disables branch spawning entirely
    (measured: the symmetric benchmark ends with no branches at all).  Taking
    the spectral absolute value keeps the metric's scale in every direction
    while making it a metric.
    """
    precision = 0.5 * (precision + precision.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(precision)
    magnitude = eigenvalues.abs()
    magnitude = magnitude.clamp_min(float(magnitude.max()) * floor)
    return (eigenvectors * magnitude.unsqueeze(-2)) @ eigenvectors.mT


def mahalanobis(a, b, precision):
    """Symmetric Mahalanobis distance used for mode-equivalence tests."""
    delta = (a - b).reshape(-1, 1)
    metric = metric_from_precision(precision)
    return float(torch.sqrt((delta.mT @ metric @ delta).clamp_min(0.0)))


def deduplicate(results, precision_of, tol=0.5, euclidean_tol=1e-6):
    """Collapse refined seeds that landed in the same basin.

    Two modes are the same when their Mahalanobis separation under the *local*
    precision of either one is below ``tol``.  Euclidean proximity alone is
    deliberately not sufficient (section J).
    """
    unique = []
    for candidate in results:
        duplicate = False
        for kept in unique:
            g_a = candidate.theta[: candidate.diagnostics["d_g"]]
            g_b = kept.theta[: kept.diagnostics["d_g"]]
            if float((g_a - g_b).abs().max()) < euclidean_tol:
                duplicate = True
                break
            d1 = mahalanobis(g_a, g_b, precision_of(kept))
            d2 = mahalanobis(g_a, g_b, precision_of(candidate))
            if min(d1, d2) < tol:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


# ----------------------------------------------------------------------------
# Batched refinement (all-global states only)
# ----------------------------------------------------------------------------

DAMPINGS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)


def mahalanobis_residual(score, curvature):
    """``sqrt(s^T P^-1 s)`` with ``P = -A``: the Newton displacement measured in
    local posterior standard deviations.

    This, not ``||S||``, is the right convergence test for mode finding.  The
    composed score at ``n = 100`` has a curvature of several hundred, so a root
    located to 1e-4 in *position* still shows ``||S|| ~ 1e-2``; a fixed
    ``||S|| < 1e-4`` acceptance test then rejects perfectly good modes, and it
    rejects them systematically in exactly the regime that matters (measured:
    ``map_vs_mass`` and ``narrow_broad`` lose their dominant mode at n >= 32).
    The precision-weighted residual is scale-free and dimensionally correct.
    """
    P = -0.5 * (curvature + curvature.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(P)
    magnitude = eigenvalues.abs().clamp_min(
        eigenvalues.abs().amax(-1, keepdim=True) * 1e-12
    )
    projected = torch.einsum("...ij,...i->...j", eigenvectors, score)
    return ((projected ** 2) / magnitude).sum(-1).clamp_min(0.0).sqrt()


def refine_batch(composer, seeds, tol=1e-9, max_iter=60, radius=1.0,
                 step_tol=1e-9):
    """Newton-refine many seeds at once on the composed field.

    Mode discovery launches tens of probes per diffusion level, and refining
    them one at a time is what makes the mode tree expensive -- not the score
    network.  Every probe sees the same field, so the whole probe set goes
    through the composer in one batched call per iteration, including the line
    search (all damping factors are tried simultaneously and the first
    merit-decreasing one is selected per row).

    Handles hierarchical states through the batched arrowhead solve when the
    composer exposes ``blocks_batched``, and all-global ones through
    ``global_curvature``.  Returns ``(theta, score_norms, global_precision,
    residual)``; ``global_precision`` is the Schur-complemented ``P_g`` at the
    final point, positive definite exactly at a mode.
    """
    theta = torch.as_tensor(seeds, dtype=torch.float64).clone()
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    arrow = composer.d_l > 0
    if arrow and not hasattr(composer, "blocks_batched"):
        raise TypeError(
            "refine_batch needs blocks_batched for a hierarchical composer"
        )
    dampings = torch.tensor(DAMPINGS, dtype=torch.float64)
    rows = torch.arange(theta.shape[0])

    def newton(state):
        score = composer.score(state)
        if arrow:
            A, B, C = composer.blocks_batched(state)
            step, precision = arrow_newton_batched(
                A, B, C, score, composer.d_g, composer.d_l, composer.n
            )
        else:
            A = composer.global_curvature(state)
            precision = -0.5 * (A + A.mT)
            step = torch.linalg.solve(precision, score.unsqueeze(-1)).squeeze(-1)
        return score, step, precision

    # Per-row bookkeeping, not a global break.  Rows converge and stall at very
    # different iterations -- a probe launched four prior standard deviations
    # away is still travelling while its neighbour is already at the root -- and
    # stopping the whole batch the moment *any* row cannot improve leaves the
    # travelling rows with a single Newton step each.  That shows up as an
    # acceptance residual exactly proportional to the seed offset, which is what
    # first hid the dominant mode of the map_vs_mass benchmark at n = 100.
    done = torch.zeros(theta.shape[0], dtype=torch.bool)
    for _ in range(max_iter):
        score, step, precision = newton(theta)
        norm = torch.linalg.vector_norm(score, dim=-1)
        residual = mahalanobis_residual(score[..., : composer.d_g], -precision)
        done = done | (norm < tol) | (residual < step_tol)
        if bool(done.all()):
            break
        cap = (radius / step.abs().amax(-1).clamp_min(1e-300)).clamp(max=1.0)
        step = step * cap.unsqueeze(-1)
        trials = theta.unsqueeze(1) + dampings.reshape(1, -1, 1) * step.unsqueeze(1)
        trial_norm = torch.linalg.vector_norm(composer.score(trials), dim=-1)
        target = (norm ** 2).unsqueeze(-1) * (
            1.0 - 1e-4 * dampings.reshape(1, -1)
        )
        ok = trial_norm ** 2 < target
        first = torch.argmax(ok.to(torch.int64), dim=-1)
        any_ok = ok.any(dim=-1) & ~done
        chosen = trials[rows, first]
        theta = torch.where(any_ok.unsqueeze(-1), chosen, theta)
        done = done | ~any_ok            # a row that cannot improve is finished

    score, _, precision = newton(theta)
    return (theta, torch.linalg.vector_norm(score, dim=-1), precision,
            mahalanobis_residual(score[..., : composer.d_g], -precision))


