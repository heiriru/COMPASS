"""Affine standardisation between raw model coordinates and network coordinates.

Parameters are standardised by their exact prior mean and sd (so the composition
prior of the globals is N(0, I)). Observations share one location/scale per family,
so the log-Jacobian ``-sum_d log x_sd[d]`` per row is the same for every model of the
family and WAIC differences need no correction.
"""

import functools

import numpy as np


@functools.lru_cache(maxsize=None)
def family_x_scale(family_name):
    """Per-dim (location, scale) of the observations, shared by a family's models."""
    from validation.a4_waic.families import family
    models = family(family_name)
    if family_name in ("offset", "noise"):
        n_obs = next(iter(models.values())).n_obs
        return np.zeros(n_obs), np.ones(n_obs)
    rng = np.random.default_rng(20261017)
    pooled = np.concatenate([m.training_data(20000, rng)[:, m.n_global + m.n_local:]
                             for m in models.values()])
    if family_name == "population_wide":
        # Heavy SP tail under the wide prior (max |x| ~ 560): scale by the bulk only.
        lo, hi = np.quantile(pooled, [0.001, 0.999], axis=0)
        bulk = np.clip(pooled, lo, hi)
        return bulk.mean(0), bulk.std(0)
    return pooled.mean(0), pooled.std(0)


class Scaler:
    def __init__(self, theta_mean, theta_sd, x_mean, x_sd):
        self.theta_mean, self.theta_sd = np.asarray(theta_mean, float), np.asarray(theta_sd, float)
        self.x_mean, self.x_sd = np.asarray(x_mean, float), np.asarray(x_sd, float)
        self.mean = np.r_[self.theta_mean, self.x_mean]
        self.sd = np.r_[self.theta_sd, self.x_sd]

    @classmethod
    def for_model(cls, family_name, model):
        x_mean, x_sd = family_x_scale(family_name)
        return cls(np.r_[model.global_mean, model.local_mean],
                   np.r_[model.global_std, model.local_std], x_mean, x_sd)

    def forward(self, nodes):
        return (nodes - self.mean) / self.sd

    def backward(self, nodes):
        return nodes * self.sd + self.mean

    def x_forward(self, x):
        return (x - self.x_mean) / self.x_sd

    def theta_backward(self, theta):
        return theta * self.theta_sd + self.theta_mean

    def waic_jacobian(self, n_rows):
        """Add to a standardised-x WAIC to get the raw-x WAIC."""
        return 2.0 * n_rows * float(np.log(self.x_sd).sum())

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("theta_mean", "theta_sd", "x_mean", "x_sd")}
