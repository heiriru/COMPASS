"""Target score matching (TSM) targets for the partial-pooling globals.

The hierarchy is g ~ p~(g) (the training prior), l_i | g ~ N(g_i, exp(g_{3+i})^2) for
i = 0, 1, 2 (nu, log_alpha, log_t0), and the trials depend on (l, beta_raw) only. So for
the six globals other than beta_raw the clean joint score is analytic at a simulated
(g, l):

    T = grad_g [log p~(g) + log p(l | g)]
      mu_i:        -(g - m0)/s0^2 + (l_i - mu_i) / sigma_i^2
      log sigma_i: -(g - m0)/s0^2 + (l_i - mu_i)^2 / sigma_i^2 - 1

and E[T | noisy g, z] is the diffused score of the globals under every mask in which
they are latent (Trainer.train, global_target_fn). Network coordinates are the affine
normalisation theta_n = (theta - mean) / scale, so T_n = scale_g * T.

Blend weight: Trainer uses w = 1 / (1 + lambda^2 V) with V from ``variance``, an upper
bound on Var[T | g] under p(l | g) (the data only shrink it):

    mu_i:        scale^2 / sigma_i^2         log sigma_i:  2 scale^2

evaluated pessimistically from network inputs only (so the blend stays unbiased): log sigma
is read off the noisy g minus 4 noise sd (4 lambda in network units), clamped between the
training prior's -5 sd and its mean (never assume a sigma above the prior median: at low noise
that keeps V <= ~1, so w ~ 1 is unchanged, and it bounds w <= 1/(1 + 0.66 lambda^2) for the
rare noise draw that escapes the shift). Without the shift, at high noise a noisy g that overstates sigma gave w ~ 1 on
samples whose T is huge: the eps-unit target's mean square was ~30 (max 3e5) at lambda 3
instead of ~1, and the first TSM runs diverged (loss 600-2400 vs 45).
"""
from __future__ import annotations

import torch

INDICES = list(range(6))  # mu_nu, mu_log_alpha, mu_log_t0, log_sigma_nu, log_sigma_log_alpha, log_sigma_log_t0


def make(theta_normalizer, prior_mean, prior_std):
    """(target_fn, variance_fn) in network coordinates. ``prior_mean``/``prior_std``: the
    7 globals' training prior in raw units (the 1.5x-wide prior for the wide arms)."""
    mean = torch.as_tensor(theta_normalizer.mean, dtype=torch.float32)
    scale = torch.as_tensor(theta_normalizer.scale, dtype=torch.float32)
    m0 = torch.as_tensor(prior_mean, dtype=torch.float32)[:6]
    s0 = torch.as_tensor(prior_std, dtype=torch.float32)[:6]
    lo, hi = (m0[3:] - 5 * s0[3:]), m0[3:]

    def target(nodes):
        dev = nodes.device
        th = nodes[:, :10].float() * scale.to(dev) + mean.to(dev)
        g, l = th[:, :6], th[:, 7:10]
        mu, log_sigma = g[:, :3], g[:, 3:6]
        inv_var = torch.exp(-2 * log_sigma)
        r = l - mu
        t = -(g - m0.to(dev)) / s0.to(dev) ** 2
        t = t + torch.cat((r * inv_var, r * r * inv_var - 1.0), 1)
        return t * scale[:6].to(dev)

    def variance(noisy_globals, lam):
        dev = noisy_globals.device
        pessimistic = noisy_globals[:, 3:6].float() - 4 * lam.reshape(-1, 1).float()
        log_sigma = pessimistic * scale[3:6].to(dev) + mean[3:6].to(dev)
        log_sigma = torch.maximum(torch.minimum(log_sigma, hi.to(dev)), lo.to(dev))
        v_mu = scale[:3].to(dev) ** 2 * torch.exp(-2 * log_sigma)
        v_ls = (2 * scale[3:6].to(dev) ** 2).expand_as(v_mu)
        return torch.cat((v_mu, v_ls), 1)

    return target, variance
