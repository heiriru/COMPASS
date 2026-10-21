#!/usr/bin/env python3
"""The arrow system with a mixture in it: K = 1 recovers ``gauss_hierarchical``.

Two changes to the mixture product, both following from one observation: a
Gaussian *mixture* marginalises in closed form, so there is no reason to throw
the local coordinates away before fitting.

**1. Schur instead of marginal draws.** The current estimator fits a 1-D mixture
to the shared column of each observation's draws and discards the local columns.
Here the full ``(g, l_j)`` joint is fitted instead and each component reduced to
its shared block. For a Gaussian component that reduction is the ``(g,g)`` block
of the covariance -- identically the Schur complement
``Lambda_gg - Lambda_gl Lambda_ll^-1 Lambda_lg`` of its precision, which is what
``gauss_hierarchical`` computes at every noise level (see
``New_Attempt/derivation.md`` section 3). The difference is not the algebra, it
is how much of each draw is used: a joint fit conditions on all ``1 + m_l``
coordinates, so the shared marginal it implies is a Rao-Blackwellised estimate of
the one a marginal-only fit produces from the same draws.

**2. No stage 3.** Keeping the joint means the locals are already described.
Given the arrow structure -- locals conditionally independent across
observations given ``g`` -- the conditional of observation ``j``'s joint mixture
is exactly

    p(l_j | g, x_j) = sum_k p(k | g, x_j) N( l_j ; mu_k^{l|g}, Sigma_k^{l|g} )
    p(k | g, x_j) proportional to w_k N( g ; mu_k^g, Sigma_k^{gg} )

so the locals come out of the same fitted object by algebra, with **no network
call at all**. That removes the stage that cost 55% of the pipeline before it was
stratified, and it needs no tuple bookkeeping: given ``g`` the choice of
component is independent across observations.

What this does not change is the ``(n-1)`` prior subtraction, so the high-n
branch collapse is inherited unchanged; bagging remains the answer to that.
"""
from __future__ import annotations

import numpy as np
import torch

VARIANCE_FLOOR_FRACTION = 1e-3


def _log_normal(draws, means, covariances):
    """Log density of each draw under each component. Shapes (o,s,d),(o,k,d),(o,k,d,d)."""
    count, samples, dimension = draws.shape
    chol = torch.linalg.cholesky(covariances)
    delta = draws[:, :, None, :] - means[:, None, :, :]          # (o,s,k,d)
    solved = torch.linalg.solve_triangular(
        chol[:, None, :, :, :].expand(count, samples, -1, -1, -1),
        delta[..., None], upper=False,
    )[..., 0]
    quadratic = (solved ** 2).sum(dim=-1)                        # (o,s,k)
    log_determinant = 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)      # (o,k)
    return -0.5 * (quadratic + log_determinant[:, None, :]
                   + dimension * np.log(2.0 * np.pi))


def _em_joint(draws, k, iterations, tolerance, generator, floor, tied=False):
    """``tied=True`` shares one covariance across components.

    The full-covariance joint fit is what makes the Schur route lose: at K = 5 in
    4 dimensions it carries ~74 free parameters against the 1-D marginal fit's
    14, and the resulting shared marginal is noisier than fitting the marginal
    directly. Tying the covariance cuts that to ``10 + 5K``, and on this family
    of problems it is not merely a regulariser -- the Gaussian-mixtures
    benchmark's exact joint posterior has **one covariance shared by both
    branches** (only the means differ), and the two elbow branches of the arm
    have the same local geometry reflected. So the tied model is closer to the
    truth *and* cheaper to estimate.
    """
    """Batched multivariate EM. ``draws`` is (observations, samples, dimension)."""
    count, samples, dimension = draws.shape
    picks = torch.stack([
        torch.randperm(samples, generator=generator, device=draws.device)[:k]
        for _ in range(count)
    ])
    means = torch.gather(
        draws, 1, picks[:, :, None].expand(count, k, dimension)).clone()
    spread = draws - draws.mean(dim=1, keepdim=True)
    covariance = torch.einsum("osi,osj->oij", spread, spread) / samples
    covariances = covariance[:, None, :, :].repeat(1, k, 1, 1).clone()
    weights = torch.full((count, k), 1.0 / k, device=draws.device,
                         dtype=draws.dtype)
    eye = torch.eye(dimension, device=draws.device, dtype=draws.dtype)

    previous = torch.full((count,), -float("inf"), device=draws.device,
                          dtype=draws.dtype)
    for _ in range(iterations):
        log_component = (torch.log(weights.clamp_min(1e-300))[:, None, :]
                         + _log_normal(draws, means, covariances))
        peak = log_component.max(dim=2, keepdim=True).values
        scaled = torch.exp(log_component - peak)
        total = scaled.sum(dim=2, keepdim=True)
        log_likelihood = (peak[:, :, 0] + torch.log(total[:, :, 0])).sum(dim=1)
        responsibility = scaled / total                           # (o,s,k)

        mass = responsibility.sum(dim=1).clamp_min(1e-300)        # (o,k)
        weights = mass / samples
        means = torch.einsum("osk,osi->oki", responsibility, draws) / mass[..., None]
        centred = draws[:, :, None, :] - means[:, None, :, :]     # (o,s,k,d)
        covariances = torch.einsum(
            "osk,oski,oskj->okij", responsibility, centred, centred
        ) / mass[..., None, None] + floor * eye
        if tied:
            pooled = (covariances * weights[..., None, None]).sum(dim=1,
                                                                  keepdim=True)
            covariances = pooled.expand_as(covariances).contiguous()
        if bool(((log_likelihood - previous).abs()
                 <= tolerance * log_likelihood.abs()).all()):
            break
        previous = log_likelihood
    return weights, means, covariances, log_likelihood


def fit_joint(draws, k_max=5, iterations=300, tolerance=1e-9, seed=0,
              device=None, tied=False):
    """Per-observation joint mixtures over ``(g, l)``, K chosen by BIC.

    ``draws`` is ``(observations, samples, dimension)`` with the shared
    coordinate first. Returns ``(fits, chosen)`` where each fit is
    ``(weights, means, covariances)``.
    """
    array = np.asarray(draws, dtype=np.float64)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
    count, samples, dimension = tensor.shape
    generator = torch.Generator(device=device).manual_seed(int(seed))
    floor = VARIANCE_FLOOR_FRACTION * float(
        tensor.var(dim=1).mean())

    best = {}
    for k in range(1, k_max + 1):
        weights, means, covariances, log_likelihood = _em_joint(
            tensor, k, iterations, tolerance, generator, floor)
        # Free parameters of a full-covariance mixture in `dimension` dimensions.
        parameters = (k * dimension + k - 1
                      + (1 if tied else k) * dimension * (dimension + 1) // 2)
        best[k] = (parameters * np.log(samples) - 2.0 * log_likelihood,
                   (weights, means, covariances))

    scores = torch.stack([best[k][0] for k in range(1, k_max + 1)])
    chosen = (scores.argmin(dim=0) + 1).cpu().numpy()
    fits = []
    for observation in range(count):
        k = int(chosen[observation])
        weights, means, covariances = best[k][1]
        fits.append((weights[observation].cpu().numpy(),
                     means[observation].cpu().numpy(),
                     covariances[observation].cpu().numpy()))
    return fits, [int(value) for value in chosen]


def shared_marginals(fits):
    """Reduce each joint component to its shared block.

    This is the Schur complement of the component's precision, taken in
    covariance form where it is the ``(g,g)`` block and needs no inversion.
    Returns the ``(weights, means, variances)`` triples the 1-D composition
    already consumes, so it is a drop-in replacement for the marginal-only fit.
    """
    return [(weights, means[:, 0].copy(), covariances[:, 0, 0].copy())
            for weights, means, covariances in fits]


def conditional_locals(fits, shared, rng):
    """Draw ``l_j ~ p(l_j | g, x_j)`` from the fitted joints, by algebra.

    ``shared`` is the array of composed shared draws. Given ``g`` the component
    responsibility is ``w_k N(g; mu_k^g, Sigma_k^gg)`` and the local conditional
    is the ordinary Gaussian conditional of that component. No network call.
    """
    shared = np.asarray(shared, dtype=np.float64)
    count = len(shared)
    out = []
    for weights, means, covariances in fits:
        variance = covariances[:, 0, 0]
        logits = (np.log(np.maximum(weights, 1e-300))[None, :]
                  - 0.5 * np.log(2.0 * np.pi * variance)[None, :]
                  - 0.5 * (shared[:, None] - means[None, :, 0]) ** 2
                  / variance[None, :])
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probability, axis=1)
        picked = (cumulative < rng.random((count, 1))).sum(axis=1)
        picked = np.clip(picked, 0, len(weights) - 1)

        # Gaussian conditional of each chosen component: the cross-covariance
        # column over the shared variance gives the regression coefficient.
        cross = covariances[picked, 1:, 0]                     # (count, m_l)
        beta = cross / variance[picked][:, None]
        mean = (means[picked, 1:]
                + beta * (shared[:, None] - means[picked, 0][:, None]))
        residual = (covariances[picked][:, 1:, 1:]
                    - beta[:, :, None] * cross[:, None, :])
        # One correlated normal per draw, from the conditional covariance.
        jitter = 1e-10 * np.eye(residual.shape[-1])
        chol = np.linalg.cholesky(residual + jitter)
        noise = rng.standard_normal((count, residual.shape[-1]))
        out.append(mean + np.einsum("cij,cj->ci", chol, noise))
    return np.stack(out)
