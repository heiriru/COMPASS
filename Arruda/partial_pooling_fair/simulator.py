"""Upstream case_study4 simulator (upstream/ddm_simulator.py), vectorised.

The priors are upstream's own functions, imported unchanged. Only the trial loop is
re-implemented: upstream's ``simulate_ddm_trial`` is a numba ``@njit`` Python loop, and
numba does not support the numpy (2.5) vendored with BayesFlow here. The replica below
runs the identical Euler--Maruyama recursion for all trials at once:

    y = beta * alpha;  rt = t0
    while (alpha >= y >= 0) and rt <= max_time:  y += nu dt + sqrt(dt) z;  rt += dt
    c = 1 if y >= alpha else 0;  return (c, rt)

``check_replica`` compares it against upstream's own trial function (run as plain Python)
on several parameter sets; ``fair.py check-sim`` runs that check.
"""
from __future__ import annotations

import sys
import types

import numpy as np

# upstream/ddm_simulator.py imports numba only for @njit; the priors do not need it.
if "numba" not in sys.modules:
    _stub = types.ModuleType("numba")
    _stub.njit = lambda f=None, **_: f if f is not None else (lambda g: g)
    sys.modules["numba"] = _stub

from upstream import ddm_simulator as up  # noqa: E402

GLOBALS = ["mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu", "log_sigma_log_alpha",
           "log_sigma_log_t0", "beta_raw"]
LOCALS = ["nu", "alpha", "t0"]
# upstream sample_hierarchical_priors: (mean, sd) of each global
PRIOR = {"mu_nu": (0.5, 0.3), "mu_log_alpha": (0.0, 0.05), "mu_log_t0": (-1.0, 0.3),
         "log_sigma_nu": (-1.0, 1.0), "log_sigma_log_alpha": (-3.0, 1.0),
         "log_sigma_log_t0": (-1.0, 0.3), "beta_raw": (0.0, 1.0)}


def simulate_trials(nu, alpha, t0, beta, rng, dt=1e-3, scale=1.0, max_time=10.0):
    """Vectorised ``up.simulate_ddm_trial`` over arrays of equal shape -> (choice, rt)."""
    nu, alpha, t0 = (np.asarray(v, dtype=np.float64) for v in (nu, alpha, t0))
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), nu.shape)
    y = (beta * alpha).ravel().copy()
    rt = t0.ravel().copy()
    nu_f, alpha_f = nu.ravel(), alpha.ravel()
    const = scale * np.sqrt(dt)
    active = np.flatnonzero((alpha_f >= y) & (y >= 0) & (rt <= max_time))
    while active.size:
        y[active] += nu_f[active] * dt + const * rng.standard_normal(active.size)
        rt[active] += dt
        keep = (alpha_f[active] >= y[active]) & (y[active] >= 0) & (rt[active] <= max_time)
        active = active[keep]
    c = (y >= alpha_f).astype(np.float64)
    return c.reshape(nu.shape), rt.reshape(nu.shape)


def draw_priors(count, rng_np_seed, prior_scale=1.0, n_subjects=1):
    """``count`` draws of upstream's hierarchical prior; ``prior_scale`` widens the sd of
    the 7 globals (the training-prior ablation). Upstream draws with np.random, so it is
    seeded here and called unchanged when ``prior_scale == 1``."""
    np.random.seed(rng_np_seed)
    rows = []
    for _ in range(count):
        if prior_scale == 1.0:
            rows.append(up.sample_hierarchical_priors(n_subjects=n_subjects))
            continue
        g = {k: np.random.normal(m, prior_scale * s) for k, (m, s) in PRIOR.items()}
        rows.append({
            **g, "beta": up.beta_from_normal(g["beta_raw"], a=50, b=50),
            "nu": np.random.normal(g["mu_nu"], np.exp(g["log_sigma_nu"]), size=n_subjects),
            "alpha": np.exp(np.random.normal(g["mu_log_alpha"], np.exp(g["log_sigma_log_alpha"]), size=n_subjects)),
            "t0": np.exp(np.random.normal(g["mu_log_t0"], np.exp(g["log_sigma_log_t0"]), size=n_subjects)),
        })
    return rows


def simulate(count, n_subjects, n_trials, seed, prior_scale=1.0):
    """Upstream ``simulator_hierarchical.sample(count, n_subjects=..., n_trials=...)``, batched.
    Returns upstream's keys; scalars as (count, 1), locals as (count, n_subjects), and
    ``sim_data`` as (count, [n_subjects,] n_trials, 2) exactly as upstream's simulate_ddm
    squeezes it (n_subjects == 1 drops the subject axis)."""
    rows = draw_priors(count, seed, prior_scale, n_subjects)
    out = {k: np.array([[r[k]] for r in rows], dtype=np.float64) for k in GLOBALS + ["beta"]}
    for k in LOCALS:
        out[k] = np.array([r[k] for r in rows], dtype=np.float64)          # (count, n_subjects)
    rng = np.random.default_rng(seed + 1)
    shape = (count, n_subjects, n_trials)
    expand = lambda a: np.broadcast_to(a[:, :, None], shape)  # noqa: E731
    c, rt = simulate_trials(expand(out["nu"]), expand(out["alpha"]), expand(out["t0"]),
                            np.broadcast_to(out["beta"][:, :, None], shape), rng)
    data = np.stack((c, rt), axis=-1)
    out["sim_data"] = data[:, 0] if n_subjects == 1 else data
    return out


def check_replica(sets=6, trials=4000, seed=0):
    """Replica vs upstream's own trial function (plain Python) on random prior draws.
    Returns per-set (choice rate, RT quantiles) for both and the two-sample KS p-value."""
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    rows = []
    for draw in draw_priors(sets, seed + 7):
        nu, alpha, t0, beta = draw["nu"][0], draw["alpha"][0], draw["t0"][0], draw["beta"]
        ref = np.array([up.simulate_ddm_trial(nu, alpha, t0, beta) for _ in range(trials)])
        c, rt = simulate_trials(np.full(trials, nu), np.full(trials, alpha), np.full(trials, t0), beta, rng)
        rows.append({"nu": nu, "alpha": alpha, "t0": t0, "beta": beta,
                     "choice_up": ref[:, 0].mean(), "choice_ours": c.mean(),
                     "rt_q_up": np.quantile(ref[:, 1], [.1, .5, .9]).round(3).tolist(),
                     "rt_q_ours": np.quantile(rt, [.1, .5, .9]).round(3).tolist(),
                     "ks_p_rt": ks_2samp(ref[:, 1], rt).pvalue})
    return rows
