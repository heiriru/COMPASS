"""Hierarchical Two Moons (hsbibm), data and coordinates for the benchmark.

All simulations come from the hsbibm task itself
(``external/hsbibm``, ``sbibm.get_task("hierarchical_two_moons", n_l=N)``), so
the networks and the test datasets see exactly the benchmark's prior and
simulator. Layout of one hsbibm draw (``n_l = N``)::

    theta = [mu_0, mu_1, sigma_0, sigma_1, theta_1[0], theta_1[1], ..., theta_N[1]]
    x     = [x_1[0], x_1[1], ..., x_N[1]]

    mu_d    ~ U(-1, 1)            sigma_d ~ U(0.1, 3)            (global, 4)
    theta_i ~ TN(mu, sigma^2; [-1, 1]^2)                          (local, 2 each)
    x_i     = p_i + (-|z_0|, z_1),  z = R(-pi/4) theta_i,
              p_i = (r cos a + 0.25, r sin a),  a ~ U(-pi/2, pi/2),  r ~ N(0.1, 0.01^2)

Network coordinates (one group = 8 nodes, ``[g(4), l(2), x(2)]``):

* globals: probit of the uniform priors, ``z = Phi^-1(u)``. The prior is then
  exactly N(0, I), which is what every composition rule and BayesFlow's prior
  score assume, so no rule is handicapped by a moment-matched prior.
* locals: theta_i / LOCAL_SCALE (raw, bounded; the crescent geometry is kept).
* data: x standardized by fixed prior-predictive constants.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _path in (HERE / ".vendor", HERE / "external" / "hsbibm"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

GLOBALS = 4
LOCAL = 2
DATA = 2
NODES = GLOBALS + LOCAL + DATA
HIERARCHY = [0, 1, 2, 3]
GLOBAL_NAMES = ("mu_0", "mu_1", "sigma_0", "sigma_1")
MU_LOW, MU_HIGH = -1.0, 1.0
SIGMA_LOW, SIGMA_HIGH = 0.1, 3.0
# Simulator constants (hsbibm simulator_params).
A_LOW, A_HIGH = -math.pi / 2, math.pi / 2
BASE_OFFSET, R_LOC, R_SCALE = 0.25, 0.1, 0.01
ROTATION = -math.pi / 4

_TASKS = {}


def sbibm_task(n_l):
    """The hsbibm task object (cached per n_l)."""
    if n_l not in _TASKS:
        import sbibm
        _TASKS[n_l] = sbibm.get_task("hierarchical_two_moons", n_l=n_l)
    return _TASKS[n_l]


def simulate(n_l, count, seed):
    """``count`` joint draws from hsbibm with ``n_l`` groups.

    Returns physical (globals (count, 4), locals (count, n_l, 2), x (count, n_l, 2)).
    """
    import torch
    task = sbibm_task(n_l)
    state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        theta = task.get_prior()(num_samples=count)
        x = task.get_simulator()(theta)
    finally:
        torch.random.set_rng_state(state)
    theta = theta.double().numpy()
    x = x.double().numpy()
    return (theta[:, :GLOBALS], theta[:, GLOBALS:].reshape(count, n_l, LOCAL),
            x.reshape(count, n_l, DATA))


# Test regime: None = the full prior; a float = test (and BayesFlow tuning) datasets
# drawn from the prior conditioned on max(sigma) <= SIGMA_MAX. Training is unchanged.
SIGMA_MAX = None


def regime_tag():
    return "" if SIGMA_MAX is None else f"_smax{SIGMA_MAX:g}"


def conditioned_draw(n_l, seed):
    """One joint prior draw, rejected until max(sigma) <= SIGMA_MAX (if set)."""
    for attempt in range(100_000):
        g, locals_, x = simulate(n_l, 1, seed=seed + 7919 * attempt)
        if SIGMA_MAX is None or g[0, 2:].max() <= SIGMA_MAX:
            return g[0], locals_[0], x[0]
    raise RuntimeError("no draw with max(sigma) <= SIGMA_MAX")


def dataset(n_l, index):
    """Test dataset ``index`` for ``n_l`` groups: one draw from the (conditioned) prior."""
    return conditioned_draw(n_l, 1_000_003 * n_l + 17 * index + 5)


def mirror(g, locals_):
    """The exact symmetry of the task: theta -> (-theta_1, -theta_0) leaves x unchanged,
    so g -> (-mu_1, -mu_0, sigma_1, sigma_0) leaves the posterior unchanged."""
    g = np.asarray(g)
    gm = np.stack([-g[..., 1], -g[..., 0], g[..., 3], g[..., 2]], -1)
    lm = -np.asarray(locals_)[..., ::-1]
    return gm, lm


def symmetric_globals(g):
    """Globals in mirror-adapted coordinates (S, 4):
    u = (mu_0+mu_1)/sqrt2 and w = sigma_0-sigma_1 flip sign under the mirror (bimodal),
    v = (mu_1-mu_0)/sqrt2 and s = sigma_0+sigma_1 are invariant."""
    g = np.asarray(g)
    r2 = math.sqrt(2.0)
    return np.stack([(g[..., 0] + g[..., 1]) / r2, (g[..., 1] - g[..., 0]) / r2,
                     g[..., 2] - g[..., 3], g[..., 2] + g[..., 3]], -1)


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def _ndtr(z):
    from scipy.special import ndtr
    return ndtr(z)


def _ndtri(u):
    from scipy.special import ndtri
    return ndtri(u)


def globals_to_z(g):
    g = np.asarray(g, dtype=np.float64)
    u = np.empty_like(g)
    u[..., :2] = (g[..., :2] - MU_LOW) / (MU_HIGH - MU_LOW)
    u[..., 2:] = (g[..., 2:] - SIGMA_LOW) / (SIGMA_HIGH - SIGMA_LOW)
    return _ndtri(np.clip(u, 1e-12, 1 - 1e-12))


def z_to_globals(z):
    u = _ndtr(np.asarray(z, dtype=np.float64))
    g = np.empty_like(u)
    g[..., :2] = MU_LOW + (MU_HIGH - MU_LOW) * u[..., :2]
    g[..., 2:] = SIGMA_LOW + (SIGMA_HIGH - SIGMA_LOW) * u[..., 2:]
    return g


_NORM = None


def normalization():
    """Fixed standardization constants from a large prior-predictive draw (n_l=1)."""
    global _NORM
    if _NORM is None:
        _, locals_, x = simulate(1, 200_000, seed=12345)
        _NORM = {"local_scale": float(locals_.reshape(-1, LOCAL).std(0).mean()),
                 "x_mean": x.reshape(-1, DATA).mean(0),
                 "x_std": x.reshape(-1, DATA).std(0)}
    return _NORM


def x_to_net(x):
    n = normalization()
    return (np.asarray(x) - n["x_mean"]) / n["x_std"]


def to_network(g, locals_, x):
    """Rows ``[z(4), theta/scale(2), x_std(2)]`` for single-group draws."""
    n = normalization()
    return np.concatenate([globals_to_z(g), np.asarray(locals_) / n["local_scale"],
                           x_to_net(x)], axis=-1)


def simulate_wide(count, seed, global_scale):
    """Single-group simulations with the probit globals drawn from N(0, global_scale^2)
    instead of the prior's N(0, 1); locals and x come from hsbibm given those globals.
    p(l | g) and p(x | l) are unchanged, so a network trained on this learns the same
    per-group factor once the wider prior is divided out (MultiObsSampler.set_training_prior)."""
    import torch
    from sbibm.tasks.distributions import TruncatedNormal
    task = sbibm_task(1)
    state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        z = global_scale * torch.randn(count, GLOBALS, dtype=torch.float64)
        g = torch.as_tensor(z_to_globals(z.numpy()), dtype=torch.float32)
        locals_ = TruncatedNormal(g[:, :2], g[:, 2:], -1.0, 1.0, validate_args=False).sample()
        x = task.get_simulator()(torch.cat([g, locals_], 1))
    finally:
        torch.random.set_rng_state(state)
    return (g.double().numpy(), locals_.double().numpy()[:, None],
            x.double().numpy().reshape(count, 1, DATA))


def training_set(count, seed, global_scale=1.0):
    """``count`` single-group simulations in network coordinates, (count, 8).
    ``global_scale`` != 1: probit globals from N(0, global_scale^2) (see simulate_wide)."""
    if global_scale == 1.0:
        g, locals_, x = simulate(1, count, seed)
    else:
        g, locals_, x = simulate_wide(count, seed, global_scale)
    return to_network(g, locals_[:, 0], x[:, 0])


def global_score_target(nodes, prior_scale=1.0):
    """Clean joint score of the globals for target score matching, in network coordinates.

    T = grad_z [log N(z; 0, prior_scale^2) + log TN(theta; mu(z), sigma(z)^2, [-1, 1]^2)]
    for clean node vectors (B, 8) = [z (4), theta / local_scale (2), x (2)]. The likelihood
    p(x | theta) does not depend on z, so this is the z-part of grad log p(z, theta | x);
    E[T | noisy z, x] is the diffused global score (Trainer.train, global_target_fn).
    """
    import torch
    from reference import _log_diff_ndtr
    nodes = torch.as_tensor(nodes)
    with torch.enable_grad():
        z = nodes[:, :GLOBALS].detach().double().requires_grad_(True)
        theta = nodes[:, GLOBALS:GLOBALS + LOCAL].detach().double() * normalization()["local_scale"]
        u = torch.special.ndtr(z)
        mu = MU_LOW + (MU_HIGH - MU_LOW) * u[:, :2]
        sigma = SIGMA_LOW + (SIGMA_HIGH - SIGMA_LOW) * u[:, 2:]
        log_tn = (-0.5 * ((theta - mu) / sigma) ** 2 - torch.log(sigma)
                  - _log_diff_ndtr((-1 - mu) / sigma, (1 - mu) / sigma)).sum(-1)
        log_prior = -0.5 * (z / prior_scale).pow(2).sum(-1)
        (grad,) = torch.autograd.grad((log_tn + log_prior).sum(), z)
    return grad.to(nodes.dtype)


# ---------------------------------------------------------------------------
# Geometry helpers (used by the reference and the metrics)
# ---------------------------------------------------------------------------

def rotate(theta):
    """z = R(-pi/4) theta, the hsbibm two_moons_map rotation."""
    c, s = math.cos(ROTATION), math.sin(ROTATION)
    theta = np.asarray(theta)
    return np.stack([c * theta[..., 0] - s * theta[..., 1],
                     s * theta[..., 0] + c * theta[..., 1]], axis=-1)


def branch(theta):
    """Which of the two moons theta lies on: sign of z_0 (the |z_0| fold)."""
    return rotate(theta)[..., 0] > 0


def crescent_residual(theta, x):
    """Reference-free fit of theta to its own observation x.

    Inverts the simulator: p = x - (-|z_0|, z_1); u = p_0 - 0.25, v = p_1.
    A theta that could have produced x has u >= 0 and r = |(u, v)| ~ N(0.1, 0.01^2).
    Returns (|r - 0.1| / 0.01, valid) with valid = (u >= 0) and theta in the box.
    """
    z = rotate(theta)
    p0 = x[..., 0] + np.abs(z[..., 0])
    p1 = x[..., 1] - z[..., 1]
    u, v = p0 - BASE_OFFSET, p1
    r = np.sqrt(u ** 2 + v ** 2)
    inside = (np.abs(theta) <= 1.0).all(-1)
    return np.abs(r - R_LOC) / R_SCALE, (u >= 0) & inside
