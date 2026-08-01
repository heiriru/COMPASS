"""Canonical parameter orders and transformations."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import beta as beta_distribution


GLOBAL_NAMES = (
    "mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu",
    "log_sigma_log_alpha", "log_sigma_log_t0", "beta_raw",
)
LOCAL_NAMES = ("nu", "log_alpha", "log_t0")
JOINT_NAMES = GLOBAL_NAMES + LOCAL_NAMES
FLAT_NAMES = ("nu", "log_alpha", "log_t0", "beta_raw")
PHYSICAL_NAMES = ("nu", "alpha", "t0", "beta")
CHOICE_GLOBAL_NAMES = ("mu_p", "log_sigma_p")
CHOICE_LOCAL_NAMES = ("z",)
FULL_GLOBAL_NAMES = (
    "mu_p", "log_sigma_p", "mu_log_rt", "log_sigma_log_rt_mean",
    "log_sigma_log_rt",
)
FULL_LOCAL_NAMES = ("z_choice", "z_log_rt")
GLOBAL_INDICES = tuple(range(len(GLOBAL_NAMES)))


def physical_from_joint(globals_, locals_):
    """Transform case-study hierarchy to physical (nu, alpha, t0, beta)."""
    globals_ = np.asarray(globals_)
    locals_ = np.asarray(locals_)
    nu = locals_[..., 0]
    alpha = np.exp(locals_[..., 1])
    t0 = np.exp(locals_[..., 2])
    beta = beta_distribution.ppf(ndtr(globals_[..., 6]), a=50.0, b=50.0)
    beta = np.broadcast_to(beta, nu.shape)
    return np.stack((nu, alpha, t0, beta), axis=-1)


def physical_from_flat(theta):
    """Transform flat (nu, log_alpha, log_t0, beta_raw) parameters."""
    theta = np.asarray(theta)
    return np.stack((
        theta[..., 0], np.exp(theta[..., 1]), np.exp(theta[..., 2]),
        beta_distribution.ppf(ndtr(theta[..., 3]), a=50.0, b=50.0),
    ), axis=-1)


def transformed_observations(raw):
    """Return deterministic trial-major (choice, log_rt, censored) features."""
    raw = np.asarray(raw)
    if raw.shape[-1] != 3:
        raise ValueError(f"Expected final observation dimension 3, got {raw.shape}.")
    out = raw.astype(np.float64, copy=True)
    out[..., 1] = np.log(np.maximum(out[..., 1], np.finfo(float).tiny))
    return out.reshape(*out.shape[:-2], -1)
