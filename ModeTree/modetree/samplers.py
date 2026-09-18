"""Baseline samplers for the comparison (task section 7).

These reproduce what ``MultiObsSampler`` does, on the oracle composed fields of
:mod:`modetree.diffusion`, so every method in the comparison sees *identical*
per-observation scores and the differences measured are differences between
composition rules and inference procedures -- not between score networks.

* :func:`reverse_diffusion` -- the VE reverse SDE in ``y`` coordinates, the
  sampler ``correction="uncorrected"``, ``"gauss"`` and ``"gauss_jacobian"`` are
  valid inside.  One Euler-Maruyama step per level:

      y <- y + (lambda_k^2 - lambda_{k+1}^2) S_t(y) + sqrt(lambda_k^2 - lambda_{k+1}^2) z

* :func:`annealed_langevin` -- the sampler ``correction="fnpe"`` requires
  (Geffner et al. 2023): ``steps_per_level`` Langevin steps at each noise level
  against the bridging score.  The hierarchy-dimension noise is scaled by
  ``1/n`` exactly as ``MultiObsSampler._langevin_sampler`` does.

Both count their score evaluations so the comparison can report network
function evaluations rather than only wall clock.
"""

from __future__ import annotations

import math
import time

import torch


def lambda_schedule(schedule, levels, lam_max, lam_min):
    lams = torch.logspace(
        math.log10(lam_max), math.log10(lam_min), levels, dtype=torch.float64
    )
    return [(schedule.time_of_lambda(float(l)), float(l)) for l in lams]


def reverse_diffusion(field, num_samples, levels=200, lam_max=None, lam_min=1e-3,
                      prior_std=None, generator=None, denoise=True):
    """VE reverse-SDE sampling of the composed field."""
    schedule = field.oracle.schedule
    prior_std = (torch.ones(field.d_g, dtype=torch.float64) if prior_std is None
                 else torch.as_tensor(prior_std, dtype=torch.float64))
    lam_max = lam_max or float(prior_std.max()) * 4.0
    grid = lambda_schedule(schedule, levels, lam_max, lam_min)
    dim = field.d_g + field.n * field.d_l
    y = torch.randn(num_samples, dim, dtype=torch.float64, generator=generator) * lam_max
    started = time.perf_counter()
    evaluations = 0
    for k in range(levels - 1):
        t, lam = grid[k]
        _, lam_next = grid[k + 1]
        gap = lam ** 2 - lam_next ** 2
        score = field.score(y, t, lam).score
        evaluations += 1
        noise = torch.randn(y.shape, dtype=torch.float64, generator=generator)
        y = y + gap * score + math.sqrt(gap) * noise
    if denoise:
        t, lam = grid[-1]
        y = y + (lam ** 2) * field.score(y, t, lam).score
        evaluations += 1
    return y, {"seconds": time.perf_counter() - started,
               "score_field_calls": evaluations,
               "score_evaluations": evaluations * field.n * num_samples}


def annealed_langevin(field, num_samples, levels=100, steps_per_level=10,
                      snr=0.2, lam_max=None, lam_min=1e-2, prior_std=None,
                      generator=None):
    """Annealed Langevin against the F-NPSE bridging score.

    ``field.rule`` should be ``"fnpe"``; the bridging densities are not the
    diffusion marginals, so this is the only sampler they are consistent with.
    """
    schedule = field.oracle.schedule
    prior_std = (torch.ones(field.d_g, dtype=torch.float64) if prior_std is None
                 else torch.as_tensor(prior_std, dtype=torch.float64))
    lam_max = lam_max or float(prior_std.max()) * 4.0
    grid = lambda_schedule(schedule, levels, lam_max, lam_min)
    dim = field.d_g + field.n * field.d_l
    n = field.n
    # F-NPSE reference at the top of the schedule is N(0, lam_max^2 / n) on the
    # shared coordinates, exactly as MultiObsSampler scales it.
    y = torch.randn(num_samples, dim, dtype=torch.float64, generator=generator)
    y[:, : field.d_g] *= lam_max / math.sqrt(n)
    y[:, field.d_g:] *= lam_max
    started = time.perf_counter()
    evaluations = 0
    for t, lam in grid:
        step = (snr * lam) ** 2
        step_global = step / n
        for _ in range(steps_per_level):
            score = field.score(y, t, lam).score
            evaluations += 1
            noise = torch.randn(y.shape, dtype=torch.float64, generator=generator)
            scale = torch.full((dim,), step, dtype=torch.float64)
            scale[: field.d_g] = step_global
            y = y + scale * score + (2.0 * scale).sqrt() * noise
    return y, {"seconds": time.perf_counter() - started,
               "score_field_calls": evaluations,
               "score_evaluations": evaluations * field.n * num_samples}
