"""Measure ``gauss_hierarchical``'s ``Sigma_t,j`` instead of assuming it.

The defect
----------
``gauss_hierarchical`` weights observation ``j`` by
``Lambda_j(t) = Sigma_0j^-1 + lambda^-2 I``. That identity is *exact only if*
``p(theta_0 | x_j)`` is Gaussian: it is the closed form of the backward
covariance ``Sigma_t,j = Cov(theta_0 | theta_t, x_j)`` for a Gaussian. On a
non-Gaussian posterior the rule is therefore using the wrong ``lambda``-profile at
every noise level, and because the local cross-coefficient
``R_j = -A_ll^-1 A_lg`` is read off the same ``Lambda_j``, the error lands hardest
on the locals.

The fix, from the law of total covariance
-----------------------------------------
    Sigma_0j = E_theta_t[ Cov(theta_0 | theta_t) ] + Cov_theta_t( E[theta_0 | theta_t] )

Tweedie gives the inner conditional mean in closed form,
``mu(theta_t) = theta_t + lambda^2 s_j(theta_t)``, so

    Sigma_bar_t,j  :=  E[ Cov(theta_0 | theta_t) ]  =  Sigma_0j - Cov( mu(theta_t) )

Every term is measurable: ``Sigma_0j`` is the pilot covariance the rule already
computes, and ``Cov(mu)`` needs the network's *own* score evaluated at diffused
pilot draws -- one batched evaluation per rung of a ``lambda`` ladder.

Why this succeeds where the importance-weighted estimate failed
---------------------------------------------------------------
``kernel_curvature.py`` tried to estimate ``Sigma_t,j(theta_t)`` by reweighting
``t = 0`` pilot draws with the diffusion kernel. It collapsed: the effective
sample size fell to 0.007 by ``lambda ~ 1e-2`` because the weights concentrate on
the nearest draw, and the draw count required grows like ``(sigma/lambda)^D``.

This estimator has no density ratio. It is a plain sample covariance of ``M``
denoised predictions in ``D`` dimensions, so it needs ``M >> D`` -- not
``M >> (sigma/lambda)^D`` -- and it evaluates the network **at the diffused
state**, which is the only place the missing information lives.

It also passes the Gaussian-twin identity test *by construction*. For a Gaussian
single-observation posterior, ``mu(theta_t) = Sigma_0j (Sigma_0j + lambda^2 I)^-1
theta_t``, so

    Cov(mu) = Sigma_0j (Sigma_0j + lambda^2 I)^-1 Sigma_0j
    Sigma_0j - Cov(mu) = lambda^2 Sigma_0j (Sigma_0j + lambda^2 I)^-1

whose inverse is exactly ``Sigma_0j^-1 + lambda^-2 I`` -- today's rule. So on a
Gaussian problem it reduces to `gauss_hierarchical` algebraically, and any
difference measured there is Monte-Carlo error in ``Cov(mu)``, not a change of
method.

What it does *not* fix
----------------------
``Sigma_bar_t,j`` is an average over ``theta_t``: it removes the Gaussian
assumption from the ``lambda``-profile but stays constant *within* a noise level.
`gauss_jacobian` is state-dependent within a level too. So this is a strict
improvement over the pilot form and a strict weakening of the Jacobian form, and
the gap between them measures how much of the non-Gaussianity is
within-level rather than across-lambda -- which no experiment here has isolated
before.

Cost: one batched row-score evaluation per ladder rung (default 24), against the
~200 the sampler itself performs. No change to ``compass``.
"""
from __future__ import annotations

import contextlib

import torch

from kernel_curvature import _condition


def _ladder(sde, rungs, eps=1e-3, device="cpu"):
    """Log-spaced ``lambda`` rungs spanning the sampler's schedule."""
    high = float(sde.lambda_t(torch.ones(1)))
    low = float(sde.lambda_t(torch.full((1,), float(eps))))
    return torch.logspace(
        torch.log10(torch.tensor(low)).item(),
        torch.log10(torch.tensor(high)).item(),
        int(rungs), dtype=torch.float64, device=device,
    )


@torch.no_grad()
def _measured_backward_covariance(sampler, draws, condition_mask, lambdas,
                                  features, generator=None):
    """``Sigma_0j - Cov(mu(theta_t))`` on each rung, from the network's own score.

    Args:
        draws: ``(n, M, F)`` pilot draws with observed columns filled.
        condition_mask: ``(F,)`` latent/observed mask.
        lambdas: ``(L,)`` rungs.
        features: latent column order the composition uses.

    Returns:
        ``(L, n, D, D)`` measured backward covariances.
    """
    rows, count, width = draws.shape
    device = draws.device
    mask = condition_mask.reshape(1, 1, -1).to(device).expand(rows, count, width)
    latent = (1.0 - mask)
    clean = draws[:, :, features].to(torch.float64)
    baseline = torch.stack([
        torch.atleast_2d(torch.cov(clean[row].mT)) for row in range(rows)
    ])                                                       # (n, D, D)

    measured = []
    for lam in lambdas:
        lam = float(lam)
        noise = torch.randn(draws.shape, device=device, dtype=draws.dtype,
                            generator=generator)
        state = draws + lam * noise * latent
        time = sampler.sde.time_of_lambda(
            torch.tensor(lam, dtype=torch.float64)
        ).reshape(1, 1).to(device).to(draws.dtype)
        score = sampler._raw_row_scores(state, time, mask)
        denoised = (state + lam**2 * score)[:, :, features].to(torch.float64)
        spread = torch.stack([
            torch.atleast_2d(torch.cov(denoised[row].mT)) for row in range(rows)
        ])
        # Sigma_0j - Cov(mu); symmetrized, and floored at PSD because both terms
        # carry Monte-Carlo error and their difference need not be PSD at finite M.
        estimate = baseline - spread
        estimate = 0.5 * (estimate + estimate.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(estimate)
        eigenvalues = eigenvalues.clamp_min(1e-12)
        measured.append(
            (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.mT
        )
    return torch.stack(measured)


def _interpolate(lambdas, table, lam):
    """Linear interpolation of the covariance table in ``log lambda``."""
    logs = torch.log(lambdas)
    target = torch.log(torch.as_tensor(float(lam), dtype=torch.float64,
                                       device=lambdas.device))
    if target <= logs[0]:
        return table[0]
    if target >= logs[-1]:
        return table[-1]
    upper = int(torch.searchsorted(logs, target).item())
    lower = upper - 1
    weight = float((target - logs[lower]) / (logs[upper] - logs[lower]))
    return (1.0 - weight) * table[lower] + weight * table[upper]


@contextlib.contextmanager
def tweedie_pilot(sampler_class, pilot_draws=1024, rungs=24, floor=1e-3,
                  seed=0, record=None):
    """Replace ``gauss_hierarchical``'s assumed ``Sigma_t,j`` with a measured one."""
    original_factors = sampler_class._effective_global_factors
    original_moments = sampler_class.estimate_posterior_moments

    def estimate_posterior_moments(self, data, condition_mask, num_samples,
                                   timesteps, eps, batch_size, device,
                                   feature_indices=None):
        mean, covariance = original_moments(
            self, data, condition_mask, num_samples, timesteps, eps,
            batch_size, device, feature_indices=feature_indices,
        )
        draws = self.SBIm.sampler.sample(
            world_size=1, data=data, condition_mask=condition_mask,
            timesteps=timesteps, eps=eps, num_samples=int(pilot_draws),
            device=device, verbose=False, method="dpm",
        ).detach().to(device)
        features = list(self.full_gaussian_features)
        generator = torch.Generator(device=draws.device).manual_seed(int(seed))
        lambdas = _ladder(self.sde, rungs, eps=eps, device=draws.device)
        table = _measured_backward_covariance(
            self, draws, torch.as_tensor(condition_mask), lambdas, features,
            generator=generator,
        )
        self._tweedie_lambdas = lambdas
        self._tweedie_table = table
        if record is not None:
            record.append((lambdas.cpu(), table.cpu()))
        return mean, covariance

    def effective_global_factors(self, var_t, num_observations=None,
                                 state=None, t=None, condition_mask=None):
        table = getattr(self, "_tweedie_table", None)
        if self.correction in self.JACOBIAN_GAUSSIAN_CORRECTIONS or table is None:
            return original_factors(
                self, var_t, num_observations, state=state, t=t,
                condition_mask=condition_mask,
            )
        n = int(self.num_observations if num_observations is None
                else num_observations)
        variance = float(torch.as_tensor(var_t, dtype=torch.float64).reshape(()))
        h = len(self.hierarchy)

        # Constant within a level, so the same time cache the pilot rules use
        # applies unchanged.
        cache = getattr(self, "_tweedie_cache", {})
        key = (variance, n)
        if key in cache:
            return cache[key]

        covariance = _interpolate(self._tweedie_lambdas, table, variance**0.5)
        joint_precision_t = _condition(covariance, variance, floor)
        joint_precision_t = joint_precision_t + self._global_block_adaptation(
            joint_precision_t, 1.0 / variance, n, h,
        )
        result = self._schur_factors(joint_precision_t, h)
        cache[key] = result
        self._tweedie_cache = cache
        return result

    sampler_class._effective_global_factors = effective_global_factors
    sampler_class.estimate_posterior_moments = estimate_posterior_moments
    try:
        yield
    finally:
        sampler_class._effective_global_factors = original_factors
        sampler_class.estimate_posterior_moments = original_moments
