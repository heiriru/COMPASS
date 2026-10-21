#!/usr/bin/env python3
"""Batched EM on the device: every observation and every K fitted at once.

Why this exists
---------------
Timing the mixture pipeline at ``n = 8`` with 20,000 draws per observation:

    diffusion steps | stage A | EM (K=1..5) | stage C | total
              100   |   15.9  |     43.2    |   19.5  |  78.6
               50   |    7.2  |     42.7    |   10.0  |  60.0
               25   |    3.6  |     43.3    |    5.1  |  52.0

The two sampler stages fall by 4x with the step count; **EM does not move**, and
at 25 steps it is 83% of the wall clock. It is not expensive arithmetic -- it is
``n * K_max`` independent fits run as a Python loop on the three CPU threads this
repository caps scripts to, while the GPU that just produced the draws sits idle.

Every one of those fits has the same shape, so they batch. Responsibilities are
``(observations, draws, components)``; the M-step is three reductions over the
draw axis. Fitting all K values for all observations is one pass per K.

``fit_all`` returns the same ``(weights, means, variances)`` triples per
observation that ``mixture_product.fit_mixture`` does, chosen by the same BIC, so
it is a drop-in replacement for ``mixture_inference.choose_and_fit``.
"""
from __future__ import annotations

import numpy as np
import torch

VARIANCE_FLOOR_FRACTION = 1e-3


def _em(draws, k, iterations, tolerance, generator, device, dtype):
    """One batched EM run. ``draws`` is ``(observations, samples)``."""
    count, samples = draws.shape
    floor = VARIANCE_FLOOR_FRACTION * draws.var(dim=1, keepdim=True)

    # Initialise from random draws, per observation, the same way the scalar
    # fitter does -- distinct points so no two components start identical.
    picks = torch.stack([
        torch.randperm(samples, generator=generator, device=device)[:k]
        for _ in range(count)
    ])
    means = torch.gather(draws, 1, picks).clone()
    variances = draws.var(dim=1, keepdim=True).expand(count, k).clone()
    weights = torch.full((count, k), 1.0 / k, device=device, dtype=dtype)

    previous = torch.full((count, 1), -float("inf"), device=device, dtype=dtype)
    for _ in range(iterations):
        log_component = (
            torch.log(weights.clamp_min(1e-300))[:, None, :]
            - 0.5 * torch.log(2.0 * np.pi * variances)[:, None, :]
            - 0.5 * (draws[:, :, None] - means[:, None, :]) ** 2
            / variances[:, None, :]
        )
        peak = log_component.max(dim=2, keepdim=True).values
        scaled = torch.exp(log_component - peak)
        total = scaled.sum(dim=2, keepdim=True)
        log_likelihood = (peak[:, :, 0] + torch.log(total[:, :, 0])).sum(
            dim=1, keepdim=True)
        responsibility = scaled / total

        mass = responsibility.sum(dim=1)
        weights = mass / samples
        means = torch.einsum("osk,os->ok", responsibility, draws) / mass.clamp_min(1e-300)
        variances = torch.clamp(
            torch.einsum("osk,osk->ok", responsibility,
                         (draws[:, :, None] - means[:, None, :]) ** 2)
            / mass.clamp_min(1e-300),
            min=floor,
        )
        if bool(((log_likelihood - previous).abs()
                 <= tolerance * log_likelihood.abs()).all()):
            break
        previous = log_likelihood
    return weights, means, variances, log_likelihood[:, 0]


def fit_all(draws, k_max=5, iterations=400, tolerance=1e-9, seed=0,
            device=None):
    """Fit K = 1..k_max for every observation at once; select by BIC.

    Returns ``(fits, chosen)`` matching ``mixture_inference.choose_and_fit``.
    """
    array = np.asarray(draws, dtype=np.float64)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
    count, samples = tensor.shape
    generator = torch.Generator(device=device).manual_seed(int(seed))

    best = {}
    for k in range(1, k_max + 1):
        if k == 1:
            weights = torch.ones(count, 1, dtype=torch.float64, device=device)
            means = tensor.mean(dim=1, keepdim=True)
            variances = tensor.var(dim=1, keepdim=True).clamp_min(1e-12)
            log_likelihood = (
                -0.5 * torch.log(2.0 * np.pi * variances)
                - 0.5 * (tensor - means) ** 2 / variances
            ).sum(dim=1)
        else:
            weights, means, variances, log_likelihood = _em(
                tensor, k, iterations, tolerance, generator, device,
                torch.float64)
        parameters = 3 * k - 1
        best[k] = (parameters * np.log(samples) - 2.0 * log_likelihood,
                   (weights, means, variances))

    scores = torch.stack([best[k][0] for k in range(1, k_max + 1)])
    chosen = (scores.argmin(dim=0) + 1).cpu().numpy()
    fits = []
    for observation in range(count):
        k = int(chosen[observation])
        weights, means, variances = best[k][1]
        fits.append((
            weights[observation].cpu().numpy(),
            means[observation].cpu().numpy(),
            variances[observation].cpu().numpy(),
        ))
    return fits, [int(value) for value in chosen]
