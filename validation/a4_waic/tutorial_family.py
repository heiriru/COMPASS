"""The tutorials/Population_Dynamics.py setting with observation noise (improvement 1).

Exactly the tutorial's simulator -- the same four right-hand sides, the Euler
integrator with step 0.01 clamped at 0, initial state (30, 1), observations at
t = 0, 1, ..., 19 divided by 100 (40 dims), and every one of the four log-parameters
drawn per trajectory from N(MOCK_PARAMS, 0.1^2) -- plus Gaussian observation noise:

    theta_i = (log alpha, log beta, log gamma, log delta)_i ~ N(MOCK_PARAMS, 0.1^2)
    x_i = [N(t_k), P(t_k)]_{k=0..19} / 100 + SIGMA_OBS * e

Without the noise ``x`` is a deterministic function of ``theta`` and ``p(x | theta)`` is
not a density, which is what broke WAIC on the original tutorial networks. There are
no shared parameters (n_global = 0): each trajectory has its own posterior.
"""

import math

import numba
import numpy as np

from validation.a4_waic.families import HierarchicalModel, _normal_logpdf
from validation.a4_waic.population import KINDS, _rhs

MOCK_PARAMS = np.array([-0.1, -3.0, -0.1, -3.0])
PARAM_SD = np.full(4, 0.1)
INITIAL_STATE = (30.0, 1.0)
T_MAX, DT = 20, 0.01
SIGMA_OBS = 0.05
NAMES = {"LV": "Lotka-Volterra", "LP": "Logistic Prey", "SP": "Satiated Predator",
         "RM": "Rosenzweig-MacArthur"}


@numba.njit(cache=True)
def _solve_euler(kind, theta, n0, p0, t_max, dt):
    """``theta (M, 4)`` log rates -> ``(M, 2 t_max)`` = [N(0), P(0), N(1), P(1), ...] / 100."""
    m = theta.shape[0]
    out = np.empty((m, 2 * t_max))
    per_unit = int(round(1.0 / dt))
    for row in range(m):
        a = math.exp(theta[row, 0]); b = math.exp(theta[row, 1])
        c = math.exp(theta[row, 2]); d = math.exp(theta[row, 3])
        n, p = n0, p0
        out[row, 0] = n / 100.0
        out[row, 1] = p / 100.0
        for k in range(1, t_max):
            for _ in range(per_unit):
                dn, dp = _rhs(kind, n, p, a, b, c, d)
                n = max(n + dn * dt, 0.0)
                p = max(p + dp * dt, 0.0)
            out[row, 2 * k] = n / 100.0
            out[row, 2 * k + 1] = p / 100.0
    return out


class TutorialModel(HierarchicalModel):
    n_global, n_local = 0, 4
    n_obs = 2 * T_MAX

    def __init__(self, kind, sigma=SIGMA_OBS):
        self.name = kind
        self.long_name = NAMES[kind]
        self.kind = KINDS[kind]
        self.local_mean, self.local_std = MOCK_PARAMS.copy(), PARAM_SD.copy()
        self.sigma = sigma

    def trajectory(self, theta):
        shape = theta.shape[:-1]
        flat = np.ascontiguousarray(theta.reshape(-1, 4), dtype=np.float64)
        out = _solve_euler(self.kind, flat, INITIAL_STATE[0], INITIAL_STATE[1], T_MAX, DT)
        return out.reshape(shape + (self.n_obs,))

    def simulate_x(self, g, l, rng):
        mean = self.trajectory(l)
        return mean + self.sigma * rng.standard_normal(mean.shape)

    def loglik(self, g, l, x):
        out = _normal_logpdf(x, self.trajectory(l), self.sigma).sum(-1)
        return np.where(np.isfinite(out), out, -np.inf)


def tutorial_models():
    return {kind: TutorialModel(kind) for kind in KINDS}
