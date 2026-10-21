"""Step B: the folded hierarchy with an asymmetric link, l centred on |g| + a g.

    g          ~ N(0, sigma_g^2)
    l_j | g    ~ N(c(g), sigma_l^2),   c(g) = |g| + a g
    x_j | l_j  ~ N(l_j, sigma_x^2)

c is (1 + a) g on the positive branch and (1 - a) |g| on the negative one, so
the two sign hypotheses explain the same data with different |g|: the negative
branch needs |g| larger by (1 + a)/(1 - a), which the prior penalises. The exact
branch split is therefore data-dependent and away from 50/50 (about 0.73 for
g_true = 1.5, a = 0.2) and does not go to 0 or 1 with n: both branches fit the
data equally well, only the prior and the slope 1/(1 +- a) separate them. A rule
that symmetrises the branches (the IK failure) shows up directly as P(g>0).

Integrating l out, x_j | g ~ N(c(g), v) with v = sigma_l^2 + sigma_x^2, so within
each branch p(g | x) is a truncated Gaussian (c is linear there). Given g the
locals are Gaussian, l_j | g, x_j ~ N(q (c(g)/sigma_l^2 + x_j/sigma_x^2), q).

Reference: the exact density on a fine 1-D grid, and a two-block Gibbs sampler
whose g-block is an inverse-CDF draw from p(g | l) on that grid, as in
``square_model``.
"""

from dataclasses import dataclass

import numpy as np

GRID = np.linspace(-6.0, 6.0, 24_001)   # spacing 5e-4


@dataclass(frozen=True)
class LeanModel:
    sigma_g: float = 1.0
    sigma_l: float = 0.5
    sigma_x: float = 0.5
    a: float = 0.2
    label: str = "g~N(0,1), l~N(|g|+0.2g,0.5²), x~N(l,0.5²)"

    @property
    def v(self):
        return self.sigma_l ** 2 + self.sigma_x ** 2

    @property
    def q(self):
        return 1.0 / (1.0 / self.sigma_l ** 2 + 1.0 / self.sigma_x ** 2)

    def link(self, g):
        return np.abs(g) + self.a * g

    @property
    def global_prior_mean(self):
        return 0.0

    @property
    def global_prior_std(self):
        return float(self.sigma_g)

    @property
    def local_prior_mean(self):           # E|g| (E[a g] = 0)
        return float(self.sigma_g * np.sqrt(2.0 / np.pi))

    @property
    def local_prior_std(self):            # Var|g| + a^2 Var g (Cov(|g|, g) = 0)
        var_c = self.sigma_g ** 2 * (1.0 - 2.0 / np.pi + self.a ** 2)
        return float(np.sqrt(self.sigma_l ** 2 + var_c))

    def simulate(self, count, rng):
        count = int(count)
        g = self.sigma_g * rng.standard_normal(count)
        l = self.link(g) + self.sigma_l * rng.standard_normal(count)
        x = l + self.sigma_x * rng.standard_normal(count)
        return np.stack([g, l, x], axis=1)

    def simulate_dataset(self, n, rng, g):
        l = self.link(g) + self.sigma_l * rng.standard_normal(int(n))
        x = l + self.sigma_x * rng.standard_normal(int(n))
        return l, x

    # -- exact posterior (1-D grid) ---------------------------------------
    def _log_unnormalised(self, grid, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        c = self.link(np.asarray(grid, dtype=np.float64))
        n, s1, s2 = len(x), x.sum(), (x ** 2).sum()
        return -0.5 * np.asarray(grid) ** 2 / self.sigma_g ** 2 - (s2 - 2.0 * c * s1 + n * c ** 2) / (2.0 * self.v)

    def global_density(self, grid, x):
        log_ref = self._log_unnormalised(GRID, x)
        area = np.trapezoid(np.exp(log_ref - log_ref.max()), GRID)
        return np.exp(self._log_unnormalised(grid, x) - log_ref.max()) / area

    # -- the interface GM's product arms read (methods.draw_marginal_product) --
    def prior_density(self, grid):
        grid = np.asarray(grid, dtype=np.float64)
        return np.exp(-0.5 * grid ** 2 / self.sigma_g ** 2) / np.sqrt(2.0 * np.pi * self.sigma_g ** 2)

    def default_grid(self, x, points=4001):
        return np.linspace(-6.0 * self.sigma_g, 6.0 * self.sigma_g, int(points))

    def positive_mass(self, x):
        d = self.global_density(GRID, x)
        return float(np.trapezoid(d * (GRID > 0), GRID))

    def plot_half_width(self, x):
        d = self.global_density(GRID, x)
        cdf = np.cumsum(d) * (GRID[1] - GRID[0])
        lo = float(GRID[np.searchsorted(cdf, 1e-4)])
        hi = float(GRID[np.searchsorted(cdf, 1 - 1e-4)])
        return max(-lo, hi) + 0.25

    # -- MCMC reference -----------------------------------------------------
    def gibbs(self, x, draws, rng, burn=2_000, thin=5, chains=8):
        """Two-block Gibbs on (g, l_1..n); returns g ``(draws,)`` and l ``(draws, n)``."""
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        per_chain = int(np.ceil(draws / chains))
        c = self.link(GRID)
        dg = GRID[1] - GRID[0]
        g = self.sigma_g * rng.standard_normal(chains)
        out_g, out_l = [], []
        for step in range(burn + per_chain * thin):
            mean_l = self.q * (self.link(g)[:, None] / self.sigma_l ** 2 + x[None] / self.sigma_x ** 2)
            l = mean_l + np.sqrt(self.q) * rng.standard_normal((chains, n))
            s1 = l.sum(1, keepdims=True)
            s2 = (l ** 2).sum(1, keepdims=True)
            log_p = -0.5 * GRID ** 2 / self.sigma_g ** 2 - (s2 - 2 * c * s1 + n * c ** 2) / (2 * self.sigma_l ** 2)
            p = np.exp(log_p - log_p.max(1, keepdims=True))
            cdf = np.cumsum(p, 1)
            cdf /= cdf[:, -1:]
            index = (cdf < rng.random((chains, 1))).sum(1)
            g = GRID[np.minimum(index, len(GRID) - 1)] + dg * (rng.random(chains) - 0.5)
            if step >= burn and (step - burn) % thin == 0:
                out_g.append(g.copy())
                out_l.append(l.copy())
        return (np.stack(out_g, 1).reshape(-1)[:draws],
                np.stack(out_l, 1).reshape(-1, n)[:draws])


DEFAULT = LeanModel()
