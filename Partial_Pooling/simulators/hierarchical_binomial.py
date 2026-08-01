"""Choice-only and full-observation hierarchical Bernoulli benchmarks."""

import numpy as np
from scipy.special import expit


def simulate_choice_only(rng, size, subjects, trials):
    globals_ = np.column_stack((rng.normal(0, 1, size), rng.normal(-0.7, 0.35, size)))
    locals_ = rng.normal(size=(size, subjects, 1))
    logits = globals_[:, None, 0] + np.exp(globals_[:, None, 1]) * locals_[..., 0]
    choices = rng.binomial(1, expit(logits)[..., None], size=(size, subjects, trials))
    return globals_, locals_, choices.astype(np.float64)


def simulate_full_observation(rng, size, subjects, trials):
    globals_ = np.column_stack((
        rng.normal(0, 1, size), rng.normal(-0.7, 0.35, size),
        rng.normal(-0.5, 0.5, size), rng.normal(-0.7, 0.35, size),
        rng.normal(-1.0, 0.25, size),
    ))
    locals_ = rng.normal(size=(size, subjects, 2))
    logits = globals_[:, None, 0] + np.exp(globals_[:, None, 1]) * locals_[..., 0]
    log_rt_mean = globals_[:, None, 2] + np.exp(globals_[:, None, 3]) * locals_[..., 1]
    choices = rng.binomial(1, expit(logits)[..., None], size=(size, subjects, trials))
    log_rt = rng.normal(
        log_rt_mean[..., None], np.exp(globals_[:, None, 4, None]),
        size=(size, subjects, trials),
    )
    observations = np.stack((choices, np.exp(log_rt), np.zeros_like(choices)), axis=-1)
    return globals_, locals_, observations.astype(np.float64)
