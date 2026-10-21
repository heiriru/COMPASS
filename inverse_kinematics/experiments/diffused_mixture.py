#!/usr/bin/env python3
"""Feed the composed mixture's analytic score into the sampler at every t.

A Gaussian mixture is closed under diffusion: convolving with ``N(0, lam_t^2)``
adds ``lam_t^2`` to every component variance. So once the shared marginal has
been composed as a mixture at ``t = 0``, its score is available **exactly** at
every noise level the sampler visits,

    s_t(a) = sum_k r_k(a) (mu_k - a) / (v_k + lam_t^2),
    r_k(a) proportional to w_k N(a; mu_k, v_k + lam_t^2),

with no Gaussian approximation and no ``(n-1)`` prior subtraction inside the
loop -- that subtraction already happened once, in closed form, when the mixture
was composed.

``K = 1`` makes this ``gauss_hierarchical``'s shared update exactly, so this is
the same rule with the component count as a dial.

Why run a sampler at all when the shared marginal is already known? Not for the
shared block -- sampling a mixture directly is strictly better there. It is for
the **locals**: driving the reverse diffusion with the exact composed shared
score lets the network's own local scores produce ``l_j`` inside one joint run,
instead of the separate conditional call the two-stage pipeline makes. Whether
that is better than the two-stage locals is the open question this answers.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch


def mixture_score(values, weights, means, variances, lam_squared):
    """Score of the mixture diffused to noise level ``sqrt(lam_squared)``."""
    spread = variances + lam_squared
    logits = (torch.log(weights.clamp_min(1e-300))[None, :]
              - 0.5 * torch.log(2.0 * np.pi * spread)[None, :]
              - 0.5 * (values[:, None] - means[None, :]) ** 2 / spread[None, :])
    responsibility = torch.softmax(logits, dim=1)
    return (responsibility * (means[None, :] - values[:, None])
            / spread[None, :]).sum(dim=1)


@contextlib.contextmanager
def composed_shared_score(sampler_class, weights, means, variances, device):
    """Override the shared block of every composed score with the mixture's.

    The local dimensions are left exactly as the underlying correction produced
    them, so the difference between running with and without this patch is
    attributable to the shared score alone.
    """
    original = sampler_class._compositional_score
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    means = torch.as_tensor(means, dtype=torch.float32, device=device)
    variances = torch.as_tensor(variances, dtype=torch.float32, device=device)

    def patched(self, scores, x, t, *args, **kwargs):
        composed = original(self, scores, x, t, *args, **kwargs)
        index = self.hierarchy[0]
        lam_squared = float(self.sde.lambda_t(t).reshape(-1)[0] ** 2)
        # The shared coordinate is synchronised across observation rows, so one
        # row carries it and the same score is written back to all of them.
        values = x[0, :, index].to(torch.float32)
        score = mixture_score(values, weights, means, variances, lam_squared)
        composed[:, :, index] = score.to(composed.dtype)[None, :]
        return composed

    sampler_class._compositional_score = patched
    try:
        yield
    finally:
        sampler_class._compositional_score = original
