"""Model families for the WAIC validation (validation/TODO.md, item A4).

Every model is hierarchical with node layout ``[g (H globals), l (L locals), x (X)]``:
the globals are shared by all ``N`` rows of a dataset, the locals belong to one row.

    g ~ p(g)  (Gaussian, so the composition prior is exact)
    l_i ~ p(l)          (independent of g in every family here)
    x_i ~ p(x | g, l_i)

Each model provides the simulator (training data and test datasets), the exact
row log-likelihood ``log p(x_i | g, l_i)`` and the prior pieces a nested sampler
and an MCMC sampler need. Nothing here touches a network.

Families
--------
``offset`` (closed form; exact-score oracles exist):
    M1  x = (l + nu e1, g + nu e2),  g ~ N(0, s_g^2), l ~ N(0, s_l^2)
    M0  the same with g = 0 (no global).
``noise`` (non-Gaussian, non-nested; random-effects location model, 3 replicates):
    x_ij = g + l_i + nu * e_ij,  g ~ N(0, 1), l ~ N(0, tau^2)
    G: e ~ N(0, 1)       T: e ~ t_3 / sqrt(3)  (same noise variance nu^2)
``population`` is in population.py.
"""

import math

import numpy as np
from scipy import special, stats


class HierarchicalModel:
    """Base class; subclasses set the sizes and implement the densities."""

    name = "model"
    n_global = 0
    n_local = 0
    n_obs = 0

    # ----- layout -----------------------------------------------------------
    @property
    def nodes(self):
        return self.n_global + self.n_local + self.n_obs

    @property
    def hierarchy(self):
        return list(range(self.n_global))

    @property
    def condition_mask(self):
        return np.r_[np.zeros(self.n_global + self.n_local), np.ones(self.n_obs)]

    # ----- priors -----------------------------------------------------------
    global_mean = np.zeros(0)
    global_std = np.zeros(0)
    local_mean = np.zeros(0)
    local_std = np.zeros(0)

    def sample_global(self, n, rng):
        return self.global_mean + self.global_std * rng.standard_normal((n, self.n_global))

    def sample_local(self, n, rng):
        return self.local_mean + self.local_std * rng.standard_normal((n, self.n_local))

    def log_prior_global(self, g):
        return _normal_logpdf(g, self.global_mean, self.global_std).sum(-1)

    def log_prior_local(self, l):
        return _normal_logpdf(l, self.local_mean, self.local_std).sum(-1)

    def ppf_global(self, u):
        return self.global_mean + self.global_std * special.ndtri(u)

    def ppf_local(self, u):
        return self.local_mean + self.local_std * special.ndtri(u)

    # ----- simulator / likelihood (subclasses) -------------------------------
    def simulate_x(self, g, l, rng):
        raise NotImplementedError

    def loglik(self, g, l, x):
        """``log p(x | g, l)`` broadcasting over leading dims; returns ``(...)``."""
        raise NotImplementedError

    # ----- data ---------------------------------------------------------------
    def training_data(self, n, rng):
        """``n`` single-row joints ``[g, l, x]`` with g drawn from its prior per row."""
        g = self.sample_global(n, rng)
        l = self.sample_local(n, rng)
        return np.concatenate([g, l, self.simulate_x(g, l, rng)], 1)

    def dataset(self, n_rows, rng):
        """One dataset: a single global draw shared by ``n_rows`` rows."""
        g = self.sample_global(1, rng)
        l = self.sample_local(n_rows, rng)
        x = self.simulate_x(np.repeat(g, n_rows, 0), l, rng)
        return dict(g=g[0], l=l, x=x)

    # ----- flat parameter vector for MCMC / nested sampling -------------------
    def unpack(self, theta, n_rows):
        """``theta (..., H + N L)`` -> ``g (..., H)``, ``l (..., N, L)``."""
        g = theta[..., :self.n_global]
        l = theta[..., self.n_global:].reshape(theta.shape[:-1] + (n_rows, self.n_local))
        return g, l

    def log_lik_rows(self, theta, x):
        """Per-row ``log p(x_i | g, l_i)`` for flat ``theta (..., D)``: ``(..., N)``."""
        n_rows = x.shape[0]
        g, l = self.unpack(theta, n_rows)
        g = np.broadcast_to(g[..., None, :], l.shape[:-1] + (self.n_global,))
        return self.loglik(g, l, x)

    def log_posterior_unnorm(self, theta, x):
        n_rows = x.shape[0]
        g, l = self.unpack(theta, n_rows)
        return (self.log_lik_rows(theta, x).sum(-1) + self.log_prior_global(g)
                + self.log_prior_local(l).sum(-1))

    def prior_transform(self, u, n_rows):
        """Unit cube -> flat ``theta`` (for nested sampling)."""
        out = np.empty_like(u)
        out[..., :self.n_global] = self.ppf_global(u[..., :self.n_global])
        loc = u[..., self.n_global:].reshape(u.shape[:-1] + (n_rows, self.n_local))
        out[..., self.n_global:] = self.ppf_local(loc).reshape(u.shape[:-1] + (-1,))
        return out


def _normal_logpdf(x, mean, std):
    return -0.5 * np.log(2 * np.pi * std ** 2) - 0.5 * ((x - mean) / std) ** 2


# ---------------------------------------------------------------------------
# offset family
# ---------------------------------------------------------------------------

class OffsetM1(HierarchicalModel):
    name = "M1"
    n_global, n_local, n_obs = 1, 1, 2

    def __init__(self, s_g=0.5, s_l=1.0, nu=1.0):
        self.s_g, self.s_l, self.nu = s_g, s_l, nu
        self.global_mean, self.global_std = np.zeros(1), np.full(1, s_g)
        self.local_mean, self.local_std = np.zeros(1), np.full(1, s_l)

    def simulate_x(self, g, l, rng):
        e = self.nu * rng.standard_normal((len(l), 2))
        return np.concatenate([l[:, :1] + e[:, :1], g[:, :1] + e[:, 1:]], 1)

    def loglik(self, g, l, x):
        return (_normal_logpdf(x[..., 0], l[..., 0], self.nu)
                + _normal_logpdf(x[..., 1], g[..., 0], self.nu))


class OffsetM0(HierarchicalModel):
    name = "M0"
    n_global, n_local, n_obs = 0, 1, 2

    def __init__(self, s_l=1.0, nu=1.0):
        self.s_l, self.nu = s_l, nu
        self.local_mean, self.local_std = np.zeros(1), np.full(1, s_l)

    def simulate_x(self, g, l, rng):
        e = self.nu * rng.standard_normal((len(l), 2))
        return np.concatenate([l[:, :1] + e[:, :1], e[:, 1:]], 1)

    def loglik(self, g, l, x):
        return (_normal_logpdf(x[..., 0], l[..., 0], self.nu)
                + _normal_logpdf(x[..., 1], 0.0, self.nu))


# ---------------------------------------------------------------------------
# noise family
# ---------------------------------------------------------------------------

class NoiseModel(HierarchicalModel):
    n_global, n_local, n_obs = 1, 1, 3

    def __init__(self, kind, nu=0.5, tau=0.5, df=3.0):
        if kind not in ("G", "T"):
            raise ValueError(kind)
        self.kind, self.nu, self.tau, self.df = kind, nu, tau, df
        self.name = kind
        self.global_mean, self.global_std = np.zeros(1), np.ones(1)
        self.local_mean, self.local_std = np.zeros(1), np.full(1, tau)
        # Student-t scale with the same variance nu^2 as the Gaussian.
        self.t_scale = nu * math.sqrt((df - 2.0) / df)

    def simulate_x(self, g, l, rng):
        shape = (len(l), self.n_obs)
        if self.kind == "G":
            e = self.nu * rng.standard_normal(shape)
        else:
            e = self.t_scale * rng.standard_t(self.df, shape)
        return g[:, :1] + l[:, :1] + e

    def loglik(self, g, l, x):
        r = x - g[..., :1] - l[..., :1]
        if self.kind == "G":
            return _normal_logpdf(r, 0.0, self.nu).sum(-1)
        df, s = self.df, self.t_scale
        const = (special.gammaln((df + 1) / 2) - special.gammaln(df / 2)
                 - 0.5 * math.log(df * math.pi) - math.log(s))
        return (const - (df + 1) / 2 * np.log1p((r / s) ** 2 / df)).sum(-1)


FAMILIES = {
    "offset": lambda: {"M1": OffsetM1(), "M0": OffsetM0()},
    "noise": lambda: {"G": NoiseModel("G"), "T": NoiseModel("T")},
}

# Rows per dataset.
N_ROWS = {"offset": 16, "noise": 16, "population": 8, "population_wide": 8, "tutorial": 20}


def family(name):
    if name == "population":
        from validation.a4_waic.population import population_models
        return population_models()
    if name == "population_wide":
        from validation.a4_waic.population import population_wide_models
        return population_wide_models()
    if name == "tutorial":
        from validation.a4_waic.tutorial_family import tutorial_models
        return tutorial_models()
    return FAMILIES[name]()
