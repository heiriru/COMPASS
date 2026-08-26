"""State-dependent backward covariance for ``gauss_hierarchical``, from its pilot.

``gauss_hierarchical`` weights each observation by
``Lambda_j(t) = Sigma_0j^-1 + lambda^-2 I``, which is exact only when
``p(theta_0 | x_j)`` is Gaussian, and is constant in the state. ``gauss_jacobian``
replaces it with the exact ``Sigma_t,j(y) = lambda^2 (I - lambda^2 H_j(y))`` at
the price of one forward-mode JVP per latent coordinate.

This module estimates the *same* object from the pilot draws the rule already
pays for, at **zero extra network evaluations**. With draws
``theta_0^(i) ~ p(theta_0 | x_j)``, Bayes' rule against the diffusion kernel gives

    p(theta_0 | theta_t, x_j) prop p(theta_0 | x_j) N(theta_t; theta_0, lambda^2 I)

so self-normalized importance weights ``w_i prop exp(-||theta_t - theta_0^(i)||^2
/ (2 lambda^2))`` yield a state-dependent, assumption-free estimate of
``Sigma_t,j(theta_t)``.

The estimate is then pushed through the *identical* conditioning the Jacobian
path applies -- convert to ``H_j = (I - Sigma_t/lambda^2)/lambda^2``, symmetrize,
clamp eigenvalues into ``[0, (1 - floor)/lambda^2]``, rebuild
``Lambda_j = 1/(lambda^2 (1 - lambda^2 eig))`` -- so ``Lambda_j`` is bounded below
by ``lambda^-2 I`` and the composed precision is positive definite by
construction rather than by repair.

Limits, both correct:
  * ``lambda`` large: weights flatten, the estimate returns ``Sigma_0,j`` and the
    rule reduces to today's ``gauss_hierarchical``.
  * ``lambda`` small: weights concentrate, the covariance shrinks, and the clamp
    holds it inside the admissible band.

Nothing in ``compass`` is edited: ``_effective_global_factors`` and
``estimate_posterior_moments`` are swapped for the duration of one run and
restored afterwards.

Honest caveat: this is a nonparametric density-ratio estimate, so its effective
sample size collapses once ``lambda`` is small relative to the spacing of the
pilot draws, and it does so faster in higher latent dimension. The effective
sample size is therefore recorded, not assumed -- see :func:`diagnostics`.
"""
from __future__ import annotations

import contextlib

import torch


def _kernel_backward_covariance(pilot, state, variance, chunk=256):
    """``Sigma_t,j(theta_t)`` by self-normalized importance weights.

    Args:
        pilot: ``(n, M, D)`` draws from ``p(theta_0 | x_j)``, latent block only.
        state: ``(n, S, D)`` current diffused latent state.
        variance: ``lambda^2``.
        chunk: samples evaluated at once; the working set is ``n x chunk x M``.

    Returns:
        ``(covariance, effective_sample_size)`` of shapes ``(n, S, D, D)`` and
        ``(n, S)``. The second is ``1 / sum_i w_i^2`` in units of ``M``: it is the
        diagnostic that says whether the estimate is supported by the pilot at
        this noise level, and it is what degrades in high dimension.
    """
    n, samples, dimension = state.shape
    covariance = torch.empty(n, samples, dimension, dimension,
                             dtype=torch.float64, device=state.device)
    efficiency = torch.empty(n, samples, dtype=torch.float64, device=state.device)
    count = pilot.shape[1]

    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        block = state[:, start:stop]                                # (n, B, D)
        # (n, B, M, D)
        delta = block[:, :, None, :] - pilot[:, None, :, :]
        log_weight = -0.5 * delta.pow(2).sum(dim=-1) / variance      # (n, B, M)
        weight = torch.softmax(log_weight, dim=-1)                   # (n, B, M)
        mean = (weight[..., None] * pilot[:, None, :, :]).sum(dim=2)  # (n, B, D)
        centred = pilot[:, None, :, :] - mean[:, :, None, :]         # (n, B, M, D)
        covariance[:, start:stop] = torch.einsum(
            "nbm,nbmi,nbmj->nbij", weight, centred, centred
        )
        efficiency[:, start:stop] = 1.0 / (
            weight.pow(2).sum(dim=-1) * count
        ).clamp_min(1e-30)
    return covariance, efficiency


def _condition(covariance, variance, floor):
    """Convert ``Sigma_t`` to ``Lambda_j``, with the Jacobian path's clamp.

    ``Sigma_t = lambda^2 (I - lambda^2 H)`` defines ``H``; every lambda-smoothed
    density obeys ``0 <= H <= lambda^-2 I`` exactly, so clamping the eigenvalues
    of ``H`` into ``[0, (1 - floor)/lambda^2]`` bounds ``Lambda_j`` into
    ``[lambda^-2 I, (floor lambda^2)^-1 I]`` -- never wider than the pure-noise
    kernel, never a point mass.
    """
    dimension = covariance.shape[-1]
    identity = torch.eye(dimension, dtype=covariance.dtype,
                         device=covariance.device)
    curvature = (identity - covariance / variance) / variance
    curvature = 0.5 * (curvature + curvature.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(curvature)
    eigenvalues = eigenvalues.clamp(min=0.0, max=(1.0 - floor) / variance)
    precision = 1.0 / (variance * (1.0 - variance * eigenvalues))
    return (eigenvectors * precision.unsqueeze(-2)) @ eigenvectors.mT


@contextlib.contextmanager
def kernel_curvature(sampler_class, pilot_draws=512, floor=1e-3, chunk=256,
                     record=None):
    """Swap ``gauss_hierarchical``'s constant ``Lambda_j`` for the kernel estimate.

    ``record``, if given, is a list that receives one
    ``(lambda, mean_effective_sample_fraction)`` tuple per composed evaluation.
    """
    original_factors = sampler_class._effective_global_factors
    original_moments = sampler_class.estimate_posterior_moments

    def estimate_posterior_moments(self, data, condition_mask, num_samples,
                                   timesteps, eps, batch_size, device,
                                   feature_indices=None):
        mean, covariance = original_moments(
            self, data, condition_mask, num_samples, timesteps, eps,
            batch_size, device, feature_indices=feature_indices,
        )
        # Retain a (smaller) pilot set for the kernel estimate. Drawn from the
        # same single-observation sampler the covariance came from, so it
        # introduces no new approximation -- only Monte-Carlo error at a size we
        # control, since the kernel cost is quadratic in this count.
        indices = (list(range(covariance.shape[-1]))
                   if feature_indices is None else list(feature_indices))
        draws = self.SBIm.sampler.sample(
            world_size=1, data=data, condition_mask=condition_mask,
            timesteps=timesteps, eps=eps, num_samples=int(pilot_draws),
            device=device, verbose=False, method="dpm",
        )
        self._kernel_pilot = draws.detach()[:, :, indices].to(torch.float64)
        self._kernel_features = indices
        return mean, covariance

    def effective_global_factors(self, var_t, num_observations=None,
                                 state=None, t=None, condition_mask=None):
        pilot = getattr(self, "_kernel_pilot", None)
        if (self.correction in self.JACOBIAN_GAUSSIAN_CORRECTIONS
                or pilot is None or state is None):
            return original_factors(
                self, var_t, num_observations, state=state, t=t,
                condition_mask=condition_mask,
            )
        n = int(self.num_observations if num_observations is None
                else num_observations)
        variance = float(torch.as_tensor(var_t, dtype=torch.float64).reshape(()))
        h = len(self.hierarchy)
        features = list(self.full_gaussian_features)
        latent = state[:, :, features].to(torch.float64)

        covariance, efficiency = _kernel_backward_covariance(
            pilot.to(latent.device), latent, variance, chunk=chunk,
        )
        if record is not None:
            record.append((variance**0.5, float(efficiency.mean())))

        joint_precision_t = _condition(covariance, variance, floor)
        joint_precision_t = joint_precision_t + self._global_block_adaptation(
            joint_precision_t, 1.0 / variance, n, h,
        )
        return self._schur_factors(joint_precision_t, h)

    sampler_class._effective_global_factors = effective_global_factors
    sampler_class.estimate_posterior_moments = estimate_posterior_moments
    try:
        yield
    finally:
        sampler_class._effective_global_factors = original_factors
        sampler_class.estimate_posterior_moments = original_moments


def diagnostics(record):
    """Summarize the recorded effective sample size per noise decade."""
    if not record:
        return []
    buckets = {}
    for lam, fraction in record:
        key = round(float(torch.log10(torch.tensor(max(lam, 1e-12)))).__floor__())
        entry = buckets.setdefault(key, [])
        entry.append(fraction)
    return [
        (key, sum(values) / len(values), len(values))
        for key, values in sorted(buckets.items())
    ]
