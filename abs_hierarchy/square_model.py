"""Step A: the folded hierarchy with a curved link, l centred on g^2.

    g          ~ N(0, sigma_g^2)
    l_j | g    ~ N(g^2, sigma_l^2)
    x_j | l_j  ~ N(l_j, sigma_x^2)

Still exactly symmetric in g (the likelihood sees g only through g^2), so the
correct branch split is 50/50 at every n, as in ``abs_model``. What changes is
the shape *within* a branch: integrating l out, x_j | g ~ N(g^2, v) with
v = sigma_l^2 + sigma_x^2, so

    log p(g | x) = -g^2 / (2 sigma_g^2) - sum_j (x_j - g^2)^2 / (2 v) + const,

a quartic in g. Each mode is skewed and its curvature changes along it, so the
"locally Gaussian per observation" property that made ``abs_model`` easy for the
Gaussian rules is gone. Given g the locals are still Gaussian,
l_j | g, x_j ~ N(q (g^2/sigma_l^2 + x_j/sigma_x^2), q).

No closed form for the normaliser, but g is 1-D: the reference density is the
quartic on a fine grid, and the MCMC reference is a two-block Gibbs sampler
whose g-block is an exact inverse-CDF draw from p(g | l) on that grid (it moves
between the two modes freely).
"""

from dataclasses import dataclass

import numpy as np

GRID = np.linspace(-4.0, 4.0, 16_001)   # spacing 5e-4; mode sd >= 0.03 at n = 128


@dataclass(frozen=True)
class SquareModel:
    sigma_g: float = 1.0
    sigma_l: float = 0.5
    sigma_x: float = 0.5
    label: str = "g~N(0,1), l~N(g²,0.5²), x~N(l,0.5²)"

    @property
    def v(self):
        return self.sigma_l ** 2 + self.sigma_x ** 2

    @property
    def q(self):
        return 1.0 / (1.0 / self.sigma_l ** 2 + 1.0 / self.sigma_x ** 2)

    @property
    def global_prior_mean(self):
        return 0.0

    @property
    def global_prior_std(self):
        return float(self.sigma_g)

    @property
    def local_prior_mean(self):           # E[g^2]
        return float(self.sigma_g ** 2)

    @property
    def local_prior_std(self):            # Var[g^2] = 2 sigma_g^4
        return float(np.sqrt(self.sigma_l ** 2 + 2.0 * self.sigma_g ** 4))

    def simulate(self, count, rng):
        count = int(count)
        g = self.sigma_g * rng.standard_normal(count)
        l = g ** 2 + self.sigma_l * rng.standard_normal(count)
        x = l + self.sigma_x * rng.standard_normal(count)
        return np.stack([g, l, x], axis=1)

    def simulate_dataset(self, n, rng, g):
        l = g ** 2 + self.sigma_l * rng.standard_normal(int(n))
        x = l + self.sigma_x * rng.standard_normal(int(n))
        return l, x

    # -- exact posterior (1-D grid) ---------------------------------------
    def _log_unnormalised(self, grid, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        g2 = np.asarray(grid, dtype=np.float64) ** 2
        n, s1, s2 = len(x), x.sum(), (x ** 2).sum()
        # sum_j (x_j - g^2)^2 = s2 - 2 g^2 s1 + n g^4, without an (n, grid) array
        return -0.5 * g2 / self.sigma_g ** 2 - (s2 - 2.0 * g2 * s1 + n * g2 ** 2) / (2.0 * self.v)

    def global_density(self, grid, x):
        log_ref = self._log_unnormalised(GRID, x)
        area = np.trapezoid(np.exp(log_ref - log_ref.max()), GRID)
        return np.exp(self._log_unnormalised(grid, x) - log_ref.max()) / area

    def plot_half_width(self, x):
        d = self.global_density(GRID, x)
        cdf = np.cumsum(d) * (GRID[1] - GRID[0])
        return float(GRID[np.searchsorted(cdf, 1 - 1e-4)]) + 0.25

    # -- MCMC reference -----------------------------------------------------
    def gibbs(self, x, draws, rng, burn=2_000, thin=5, chains=8):
        """Two-block Gibbs on (g, l_1..n); returns g ``(draws,)`` and l ``(draws, n)``.

        l | g, x: Gaussian. g | l ∝ N(g; 0, sigma_g^2) prod N(l_j; g^2, sigma_l^2),
        a symmetric quartic, drawn exactly (to grid resolution) by inverse CDF.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        per_chain = int(np.ceil(draws / chains))
        g2 = GRID ** 2
        dg = GRID[1] - GRID[0]
        g = self.sigma_g * rng.standard_normal(chains)
        out_g, out_l = [], []
        for step in range(burn + per_chain * thin):
            mean_l = self.q * ((g ** 2)[:, None] / self.sigma_l ** 2 + x[None] / self.sigma_x ** 2)
            l = mean_l + np.sqrt(self.q) * rng.standard_normal((chains, n))
            s1 = l.sum(1, keepdims=True)
            s2 = (l ** 2).sum(1, keepdims=True)
            log_p = -0.5 * g2 / self.sigma_g ** 2 - (s2 - 2 * g2 * s1 + n * g2 ** 2) / (2 * self.sigma_l ** 2)
            p = np.exp(log_p - log_p.max(1, keepdims=True))
            cdf = np.cumsum(p, 1)
            cdf /= cdf[:, -1:]
            u = rng.random((chains, 1))
            index = (cdf < u).sum(1)
            g = GRID[np.minimum(index, len(GRID) - 1)] + dg * (rng.random(chains) - 0.5)
            if step >= burn and (step - burn) % thin == 0:
                out_g.append(g.copy())
                out_l.append(l.copy())
        return (np.stack(out_g, 1).reshape(-1)[:draws],
                np.stack(out_l, 1).reshape(-1, n)[:draws])


DEFAULT = SquareModel()
