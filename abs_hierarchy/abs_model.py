"""Folded hierarchy: the GM benchmark's g -> l -> x chain with l centred on |g|.

    g          ~ N(0, sigma_g^2)
    l_j | g    ~ N(|g|, sigma_l^2)
    x_j | l_j  ~ N(l_j, sigma_x^2)

The likelihood depends on g only through |g|, so p(g | x_1..n) is exactly
symmetric in g for every dataset: the sign of g is never identified, and once
the data pull |g| away from 0 the shared posterior has two mirror modes at
+-|g|. The single-observation posterior p(g, l | x) is already bimodal in g.

Node layout matches ``Gaussian_mixtures_compositional_score_modelling.model``
(g, l, x), so that benchmark's training and sampling code is reused as is.

Exact posterior. Integrating l out, x_j | g ~ N(|g|, v) with v = sigma_l^2 +
sigma_x^2, and since the prior is even,

    p(g | x) ∝ N(|g|; m, V),   V = (1/sigma_g^2 + n/v)^-1,   m = V sum(x) / v,

two mirrored half-Gaussians. Given g the locals are independent Gaussians,
l_j | g, x_j ~ N(q(|g|/sigma_l^2 + x_j/sigma_x^2), q). The MCMC reference does
not use this collapse (it runs a plain two-block Gibbs sampler on (g, l)), so
the closed form is an independent check of it.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr

NODES = 3
HIERARCHY = [0]
LOCAL_INDICES = [1]
OBSERVED_INDICES = [2]


@dataclass(frozen=True)
class FoldedModel:
    sigma_g: float = 1.0
    sigma_l: float = 0.5
    sigma_x: float = 0.5
    label: str = "g~N(0,1), l~N(|g|,0.5²), x~N(l,0.5²)"

    @property
    def v(self):
        return self.sigma_l ** 2 + self.sigma_x ** 2

    @property
    def q(self):
        return 1.0 / (1.0 / self.sigma_l ** 2 + 1.0 / self.sigma_x ** 2)

    # The sampler API takes Gaussian priors. g's prior really is Gaussian;
    # l's marginal prior (a folded normal plus noise) is moment-matched.
    @property
    def global_prior_mean(self):
        return 0.0

    @property
    def global_prior_std(self):
        return float(self.sigma_g)

    @property
    def local_prior_mean(self):
        return float(self.sigma_g * np.sqrt(2.0 / np.pi))

    @property
    def local_prior_std(self):
        return float(np.sqrt(self.sigma_l ** 2 + self.sigma_g ** 2 * (1.0 - 2.0 / np.pi)))

    def simulate(self, count, rng):
        """``count`` independent (g, l, x) training rows, one fresh g each."""
        count = int(count)
        g = self.sigma_g * rng.standard_normal(count)
        l = np.abs(g) + self.sigma_l * rng.standard_normal(count)
        x = l + self.sigma_x * rng.standard_normal(count)
        return np.stack([g, l, x], axis=1)

    def simulate_dataset(self, n, rng, g):
        l = abs(g) + self.sigma_l * rng.standard_normal(int(n))
        x = l + self.sigma_x * rng.standard_normal(int(n))
        return l, x

    # -- exact posterior -------------------------------------------------
    def folded_moments(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        V = 1.0 / (1.0 / self.sigma_g ** 2 + len(x) / self.v)
        return V * x.sum() / self.v, V

    def global_density(self, grid, x):
        m, V = self.folded_moments(x)
        grid = np.asarray(grid, dtype=np.float64)
        log_norm = np.log(2.0) + 0.5 * np.log(2 * np.pi * V) + log_ndtr(m / np.sqrt(V))
        return np.exp(-0.5 * (np.abs(grid) - m) ** 2 / V - log_norm)

    def plot_half_width(self, x):
        m, V = self.folded_moments(x)
        return float(m + 5 * V ** 0.5 + 0.3)

    def local_density(self, grid, x, j):
        """p(l_j | x) on ``grid``: a mixture over |g| of Gaussians, by quadrature."""
        m, V = self.folded_moments(x)
        a = np.linspace(max(0.0, m - 9 * np.sqrt(V)), m + 9 * np.sqrt(V), 4001)
        w = np.exp(-0.5 * (a - m) ** 2 / V)
        w /= np.trapezoid(w, a)
        means = self.q * (a / self.sigma_l ** 2 + x[j] / self.sigma_x ** 2)
        kern = np.exp(-0.5 * (grid[:, None] - means[None]) ** 2 / self.q) / np.sqrt(2 * np.pi * self.q)
        return np.trapezoid(kern * w[None], a, axis=1)

    # -- MCMC reference --------------------------------------------------
    def gibbs(self, x, draws, rng, burn=2_000, thin=5, chains=8):
        """Two-block Gibbs on (g, l_1..n). Returns g ``(draws,)`` and l ``(draws, n)``.

        l | g, x is Gaussian. g | l is ∝ N(g; 0, sigma_g^2) prod N(l_j; |g|, sigma_l^2)
        = a half-Gaussian in |g| mirrored onto both signs, drawn exactly by a
        truncated normal for |g| and a fair coin for the sign. That block moves
        between the two modes freely, so the chain does not need tempering.
        """
        from scipy.stats import truncnorm

        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        per_chain = int(np.ceil(draws / chains))
        P = 1.0 / self.sigma_g ** 2 + n / self.sigma_l ** 2
        s = 1.0 / np.sqrt(P)
        g = self.sigma_g * rng.standard_normal(chains)
        out_g, out_l = [], []
        for step in range(burn + per_chain * thin):
            mean_l = self.q * (np.abs(g)[:, None] / self.sigma_l ** 2 + x[None] / self.sigma_x ** 2)
            l = mean_l + np.sqrt(self.q) * rng.standard_normal((chains, n))
            m = (l.sum(1) / self.sigma_l ** 2) / P
            a = truncnorm.rvs(-m / s, np.inf, loc=m, scale=s, random_state=rng)
            g = np.where(rng.random(chains) < 0.5, -a, a)
            if step >= burn and (step - burn) % thin == 0:
                out_g.append(g.copy())
                out_l.append(l.copy())
        g = np.stack(out_g, 1).reshape(-1)[:draws]
        l = np.stack(out_l, 1).reshape(-1, n)[:draws]
        return g, l


DEFAULT = FoldedModel()
