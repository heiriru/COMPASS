"""Population-dynamics family: the four predator-prey hypotheses of
tutorials/Population_Dynamics.py, made hierarchical and stochastic.

A dataset is ``N`` populations (sites) that share the dynamics but start from their
own initial state:

    g = (log alpha, log beta, log gamma, log delta)            4 globals, shared
        ~ N(GLOBAL_MEAN, GLOBAL_SD^2)  (GLOBAL_MEAN is the tutorial's mock parameter)
    l_i = (log N0_i, log P0_i)                                  2 locals per site
        ~ N((log 20, log 15), LOCAL_SD^2)
    x_i = [N(t_k)/100, P(t_k)/100]_{k} + SIGMA_OBS * e,  t_k = 1, 2, ..., 10   (20 dims)

The trajectory is the RK4 solution (step DT, clamped at 0 like the tutorial's
integrator) of one of the four right-hand sides below. Given the parameters the
observation noise is Gaussian, so the exact likelihood is closed form after one ODE
solve; numba keeps that cheap enough for nested sampling.

The four models (tutorial definitions, 4 parameters each, with different meanings):
    LV  Lotka-Volterra            dN = aN - bNP,                 dP = dNP - cP
    LP  logistic prey             dN = aN(1 - N/1000d) - bNP,    dP = 0.5 bNP - cP
    SP  satiated predator         q = bN/(1 + bdN); dN = aN - qP, dP = 0.5 qP - cP
    RM  Rosenzweig-MacArthur      q = bN/(1 + 0.1 bN); dN = aN(1 - N/1000d) - qP,
                                  dP = 0.5 qP - cP
(a, b, c, d) = (alpha, beta, gamma, delta).
"""

import math

import numba
import numpy as np

from validation.a4_waic.families import HierarchicalModel, _normal_logpdf

GLOBAL_MEAN = np.array([-0.1, -3.0, -0.1, -3.0])
GLOBAL_SD = np.array([0.1, 0.1, 0.1, 0.1])
LOCAL_MEAN = np.array([math.log(20.0), math.log(15.0)])
LOCAL_SD = np.array([0.2, 0.2])
T_OBS = np.arange(1.0, 10.0 + 1e-9, 1.0)
DT = 0.05
SIGMA_OBS = 0.05
KINDS = {"LV": 0, "LP": 1, "SP": 2, "RM": 3}


@numba.njit(cache=True, inline="always")
def _rhs(kind, n, p, a, b, c, d):
    if kind == 0:
        return a * n - b * n * p, d * n * p - c * p
    if kind == 1:
        return a * n * (1.0 - n / (1000.0 * d)) - b * n * p, 0.5 * b * n * p - c * p
    if kind == 2:
        q = b * n / (1.0 + b * d * n)
        return a * n - q * p, 0.5 * q * p - c * p
    q = b * n / (1.0 + 0.1 * b * n)
    return a * n * (1.0 - n / (1000.0 * d)) - q * p, 0.5 * q * p - c * p


@numba.njit(cache=True, parallel=False)
def _solve(kind, g, l, t_obs, dt):
    """``g (M, 4)`` log rates, ``l (M, 2)`` log initial states -> ``(M, 2 K)`` /100."""
    m = g.shape[0]
    k_obs = t_obs.shape[0]
    out = np.empty((m, 2 * k_obs))
    steps_total = int(round(t_obs[-1] / dt))
    for row in range(m):
        a = math.exp(g[row, 0]); b = math.exp(g[row, 1])
        c = math.exp(g[row, 2]); d = math.exp(g[row, 3])
        n = math.exp(l[row, 0]); p = math.exp(l[row, 1])
        k = 0
        for step in range(1, steps_total + 1):
            k1n, k1p = _rhs(kind, n, p, a, b, c, d)
            k2n, k2p = _rhs(kind, n + 0.5 * dt * k1n, p + 0.5 * dt * k1p, a, b, c, d)
            k3n, k3p = _rhs(kind, n + 0.5 * dt * k2n, p + 0.5 * dt * k2p, a, b, c, d)
            k4n, k4p = _rhs(kind, n + dt * k3n, p + dt * k3p, a, b, c, d)
            n = max(n + dt / 6.0 * (k1n + 2 * k2n + 2 * k3n + k4n), 0.0)
            p = max(p + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p), 0.0)
            if k < k_obs and abs(step * dt - t_obs[k]) < 0.5 * dt:
                out[row, 2 * k] = n / 100.0
                out[row, 2 * k + 1] = p / 100.0
                k += 1
    return out


class PopulationModel(HierarchicalModel):
    n_global, n_local = 4, 2
    n_obs = 2 * len(T_OBS)

    def __init__(self, kind):
        self.name = kind
        self.kind = KINDS[kind]
        self.global_mean, self.global_std = GLOBAL_MEAN.copy(), GLOBAL_SD.copy()
        self.local_mean, self.local_std = LOCAL_MEAN.copy(), LOCAL_SD.copy()
        self.sigma = SIGMA_OBS

    def trajectory(self, g, l):
        """Noise-free observation means for broadcastable ``g (..., 4)``, ``l (..., 2)``."""
        shape = np.broadcast_shapes(g.shape[:-1], l.shape[:-1])
        g2 = np.ascontiguousarray(np.broadcast_to(g, shape + (4,)).reshape(-1, 4), dtype=np.float64)
        l2 = np.ascontiguousarray(np.broadcast_to(l, shape + (2,)).reshape(-1, 2), dtype=np.float64)
        return _solve(self.kind, g2, l2, T_OBS, DT).reshape(shape + (self.n_obs,))

    def simulate_x(self, g, l, rng):
        mean = self.trajectory(g, l)
        return mean + self.sigma * rng.standard_normal(mean.shape)

    def loglik(self, g, l, x):
        mean = self.trajectory(g, l)
        out = _normal_logpdf(x, mean, self.sigma).sum(-1)
        return np.where(np.isfinite(out), out, -np.inf)


def population_models():
    return {kind: PopulationModel(kind) for kind in KINDS}


WIDE_FACTOR = 3.0


def population_wide_models():
    """Improvement 3b: the same four models, every prior sd x WIDE_FACTOR, so that the
    posteriors of misspecified models (4-6 narrow prior sds out in A4) stay inside the
    training support. Test data still come from the narrow ``population`` models
    (common.DATA_FAMILY)."""
    out = {}
    for kind in KINDS:
        m = PopulationModel(kind)
        m.global_std = GLOBAL_SD * WIDE_FACTOR
        m.local_std = LOCAL_SD * WIDE_FACTOR
        out[kind] = m
    return out
