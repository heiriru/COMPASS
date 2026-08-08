"""Case-study priors and analytic Gaussian prior scores."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import beta as beta_distribution


GLOBAL_PRIOR_MEAN = np.array([0.5, 0.0, -1.0, -1.0, -3.0, -1.0, 0.0])
GLOBAL_PRIOR_STD = np.array([0.3, 0.05, 0.3, 1.0, 1.0, 0.3, 1.0])

# Marginal (prior-predictive) moments of one subject's local parameters, after
# integrating out both the population mean and the population scale:
#   l_k = mu_k + exp(log_sigma_k) * z,   z ~ N(0, 1)
#   E[l_k]   = E[mu_k]
#   Var[l_k] = Var[mu_k] + E[exp(2 log_sigma_k)]
# with the second term the lognormal moment exp(2a + 2b^2) for
# log_sigma_k ~ N(a, b^2). These are what a local-latent box should be built
# from: the conditional scale exp(log_sigma_k) is itself inferred, so a box
# derived from it inherits any error in the shared coordinates.
LOCAL_PRIOR_MEAN = GLOBAL_PRIOR_MEAN[:3].copy()
LOCAL_PRIOR_STD = np.sqrt(
    GLOBAL_PRIOR_STD[:3] ** 2
    + np.exp(2.0 * GLOBAL_PRIOR_MEAN[3:6] + 2.0 * GLOBAL_PRIOR_STD[3:6] ** 2)
)
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
