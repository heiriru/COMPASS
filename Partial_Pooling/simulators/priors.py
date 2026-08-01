"""Case-study priors and analytic Gaussian prior scores."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import beta as beta_distribution


GLOBAL_PRIOR_MEAN = np.array([0.5, 0.0, -1.0, -1.0, -3.0, -1.0, 0.0])
GLOBAL_PRIOR_STD = np.array([0.3, 0.05, 0.3, 1.0, 1.0, 0.3, 1.0])
FLAT_PRIOR_MEAN = np.array([0.5, 0.0, -1.0, 0.0])
FLAT_PRIOR_STD = np.array([np.exp(-1.0), np.exp(-3.0), np.exp(-1.0), 1.0])


def beta_from_normal(value, a=50.0, b=50.0):
    """Transform a standard normal variate to Beta(a,b), as upstream."""
    return beta_distribution.ppf(ndtr(value), a=a, b=b)


def sample_hierarchical_prior(rng, size, subjects):
    globals_ = rng.normal(GLOBAL_PRIOR_MEAN, GLOBAL_PRIOR_STD, size=(size, 7))
    local_mean = globals_[:, None, :3]
    local_std = np.exp(globals_[:, None, 3:6])
    locals_ = rng.normal(local_mean, local_std, size=(size, subjects, 3))
    return globals_, locals_


def sample_flat_prior(rng, size):
    nu = rng.normal(0.5, np.exp(-1.0), size)
    log_alpha = rng.normal(0.0, np.exp(-3.0), size)
    log_t0 = rng.normal(-1.0, np.exp(-1.0), size)
    beta_raw = rng.normal(0.0, 1.0, size)
    return np.stack((nu, log_alpha, log_t0, beta_raw), axis=-1)


def gaussian_prior_score(x, mean, std):
    return -(np.asarray(x) - np.asarray(mean)) / np.asarray(std) ** 2


def diffused_gaussian_prior_score(x_t, t, mean, std, sde):
    """Score in the SDE state x_t with mean alpha*mu and variance alpha²s²+sigma²."""
    alpha = np.asarray(sde.alpha_t(t))
    sigma = np.asarray(sde.sigma_t(t))
    variance = alpha**2 * np.asarray(std)**2 + sigma**2
    return -(np.asarray(x_t) - alpha * np.asarray(mean)) / variance
