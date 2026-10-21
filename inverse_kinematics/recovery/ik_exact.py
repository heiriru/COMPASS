"""Exact, noise-free likelihood pieces for the inverse-kinematics arm.

The forward map is deterministic, so ``p(x | theta)`` is a point mass and there
is no likelihood to hand an MCMC sampler over all four parameters. The fix used
here is exact, not an ABC kernel: integrate out two parameters analytically.

Given ``angle_1`` and ``height`` the end position fixes the remaining two joints
up to a discrete choice. With ``r = x - joint_2`` the classical two-link solution
is ``cos(a3) = (|r|^2 - L2^2 - L3^2) / (2 L2 L3)`` (elbow up / elbow down), and
``a2`` follows from the direction of ``r``. The change of variables
``(a2, a3) -> (x1, x2)`` has ``|det J| = L2 L3 |sin a3|``, so

    p(x | a1, h) = sum_solutions N(a2; 0, .5) N(a3; 0, .5) / (L2 L3 |sin a3|)

in closed form (:func:`log_lik_a1h`). It is zero outside the workspace annulus
``|L3 - L2| <= |r| <= L2 + L3`` and has integrable ``1/sqrt`` peaks on its edges.

For the ``angle_1``-only layout the height is local too and is integrated out:

    p(x | a1) = int N(h; 0, .25) p(x | a1, h) dh

The allowed ``h`` set is known analytically (two intervals mirrored about
``d = x1 - L1 sin a1``), and the substitution
``u = lo + (hi - lo)(1 - cos t)/2`` removes both edge singularities, so plain
Gauss-Legendre in ``t`` converges spectrally (:func:`log_marginal_a1`). That
quadrature is used only as an *independent* check on the MCMC, which works on
``(a1, h_1..h_n)`` with the closed form above and never calls it.

Parameters are in node order ``(angle_1, height, angle_2, angle_3)`` throughout.
"""

import numpy as np

import simulator as sim

L1, L2, L3 = sim.L1, sim.L2, sim.L3
SD_A1, SD_H, SD_A2, SD_A3 = 0.5, float(sim.PRIOR_SCALES[0]), 0.5, 0.5
PRIOR_SD = np.array([SD_A1, SD_H, SD_A2, SD_A3])
R_MIN, R_MAX = abs(L3 - L2), L2 + L3
# 2*pi wraps summed over for a2 and a3. The prior sd is 0.5, so a wrap costs at
# least (pi/0.5)^2/2 = 19.7 nats; kept for exactness, and checked in tests.
WRAPS = (-1, 0, 1)
_LOG_2PI = np.log(2.0 * np.pi)


def log_normal(value, sd):
    return -0.5 * (value / sd) ** 2 - np.log(sd) - 0.5 * _LOG_2PI


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def elbow_solutions(a1, h, x1, x2, wraps=WRAPS):
    """All ``(a2, a3)`` reaching ``(x1, x2)`` from ``(a1, h)``, with log weights.

    Broadcasts over its inputs and appends a solution axis of size
    ``2 * len(wraps)**2``. ``log_weight`` is ``log N(a2) + log N(a3) - log|det J|``
    and is ``-inf`` where the target is outside the workspace.
    """
    a1, h, x1, x2 = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a1, h, x1, x2)))
    r1 = x1 - h - L1 * np.sin(a1)
    r2 = x2 - L1 * np.cos(a1)
    kappa = (r1 * r1 + r2 * r2 - L2 ** 2 - L3 ** 2) / (2.0 * L2 * L3)
    inside = np.abs(kappa) < 1.0
    base = np.arccos(np.clip(kappa, -1.0, 1.0))
    beta = np.arctan2(r1, r2)
    log_det = np.log(L2 * L3 * np.maximum(np.sin(base), 1e-300))
    a2_all, a3_all, logw_all = [], [], []
    for sign in (1.0, -1.0):
        a3_0 = sign * base
        a2_0 = _wrap(beta - np.arctan2(L3 * np.sin(a3_0), L2 + L3 * np.cos(a3_0)) - a1)
        for k in wraps:
            a3 = a3_0 + 2.0 * np.pi * k
            for m in wraps:
                a2 = a2_0 + 2.0 * np.pi * m
                a2_all.append(a2)
                a3_all.append(a3)
                logw_all.append(np.where(inside, log_normal(a2, SD_A2) + log_normal(a3, SD_A3) - log_det,
                                         -np.inf))
    return np.stack(a2_all, -1), np.stack(a3_all, -1), np.stack(logw_all, -1)


def _logsumexp(values, axis=-1):
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        return np.log(np.sum(np.exp(values - peak), axis=axis)) + np.squeeze(peak, axis)


def log_lik_a1h(a1, h, x1, x2, wraps=WRAPS):
    """``log p(x | a1, h)``, the two joint angles integrated out exactly."""
    return _logsumexp(elbow_solutions(a1, h, x1, x2, wraps)[2])


def draw_elbow(a1, h, x1, x2, rng):
    """One ``(a2, a3)`` per entry from ``p(a2, a3 | a1, h, x)`` (a discrete law)."""
    a2, a3, logw = elbow_solutions(a1, h, x1, x2)
    logw = logw - _logsumexp(logw)[..., None]
    cumulative = np.cumsum(np.exp(logw), axis=-1)
    pick = (rng.random(cumulative.shape[:-1])[..., None] > cumulative).sum(-1)
    pick = np.minimum(pick, cumulative.shape[-1] - 1)[..., None]
    return np.take_along_axis(a2, pick, -1)[..., 0], np.take_along_axis(a3, pick, -1)[..., 0]


def height_support(a1, x1, x2):
    """``(d, lo, hi, reachable)``: allowed heights are ``d -+ u`` for ``u in [lo, hi]``."""
    d = x1 - L1 * np.sin(a1)
    c2 = (x2 - L1 * np.cos(a1)) ** 2
    reachable = c2 < R_MAX ** 2
    lo = np.sqrt(np.maximum(0.0, R_MIN ** 2 - c2))
    hi = np.sqrt(np.maximum(0.0, R_MAX ** 2 - c2))
    return d, lo, hi, reachable


_GL_CACHE = {}


def _gauss_legendre(nodes):
    if nodes not in _GL_CACHE:
        z, w = np.polynomial.legendre.leggauss(nodes)
        _GL_CACHE[nodes] = (0.5 * np.pi * (z + 1.0), 0.5 * np.pi * w)   # mapped to [0, pi]
    return _GL_CACHE[nodes]


def log_marginal_a1(a1, x1, x2, nodes=96, wraps=WRAPS, sd_h=SD_H):
    """``log p(x | a1) = log int N(h) p(x | a1, h) dh`` by regularised quadrature.

    ``a1`` broadcasts against ``x1``/``x2``; the quadrature axis is internal. ``sd_h`` is the
    height prior's scale (the stock 0.25 unless a wider-prior variant is being studied).
    """
    a1, x1, x2 = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a1, x1, x2)))
    d, lo, hi, reachable = height_support(a1, x1, x2)
    t, w = _gauss_legendre(nodes)
    u = lo[..., None] + (hi - lo)[..., None] * 0.5 * (1.0 - np.cos(t))
    log_du = np.log(np.maximum((hi - lo)[..., None] * 0.5 * np.sin(t), 1e-300)) + np.log(w)
    terms = []
    for side in (1.0, -1.0):
        h = d[..., None] - side * u
        terms.append(log_normal(h, sd_h) + log_lik_a1h(a1[..., None], h, x1[..., None], x2[..., None], wraps)
                     + log_du)
    total = _logsumexp(np.concatenate(terms, -1))
    return np.where(reachable, total, -np.inf)


def draw_height(a1, x1, x2, rng, cells=256, wraps=WRAPS, chunk=20_000):
    """Chunked front end of :func:`_draw_height`; broadcasts its inputs."""
    a1, x1, x2 = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a1, x1, x2)))
    flat = [v.reshape(-1) for v in (a1, x1, x2)]
    out = np.empty(flat[0].shape)
    for s in range(0, len(out), chunk):
        out[s:s + chunk] = _draw_height(*(v[s:s + chunk] for v in flat), rng, cells, wraps)
    return out.reshape(a1.shape)


def _draw_height(a1, x1, x2, rng, cells, wraps):
    """One ``h`` per entry from ``p(h | a1, x) propto N(h) p(x | a1, h)``.

    Inverse-CDF on ``cells`` equal cells of the regularised variable ``t`` (both
    sides of ``d``), uniform within the chosen cell. The density in ``t`` is
    smooth and bounded, so the only approximation is a piecewise-constant one
    at 1/cells resolution.
    """
    a1, x1, x2 = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a1, x1, x2)))
    d, lo, hi, _ = height_support(a1, x1, x2)
    half = cells // 2
    edges = np.linspace(0.0, np.pi, half + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    u = lo[..., None] + (hi - lo)[..., None] * 0.5 * (1.0 - np.cos(mid))
    log_du = np.log(np.maximum((hi - lo)[..., None] * 0.5 * np.sin(mid), 1e-300))
    logs = []
    for side in (1.0, -1.0):
        h = d[..., None] - side * u
        logs.append(log_normal(h, SD_H) + log_lik_a1h(a1[..., None], h, x1[..., None], x2[..., None], wraps) + log_du)
    logs = np.concatenate(logs, -1)
    prob = np.exp(logs - logs.max(-1, keepdims=True))
    cumulative = np.cumsum(prob, -1)
    cumulative /= cumulative[..., -1:]
    cell = np.minimum((rng.random(cumulative.shape[:-1])[..., None] > cumulative).sum(-1), cells - 1)
    side = np.where(cell < half, 1.0, -1.0)
    t = edges[cell % half] + (np.pi / half) * rng.random(cell.shape)
    return d - side * (lo + (hi - lo) * 0.5 * (1.0 - np.cos(t)))


# ---------------------------------------------------------------------------
# Deterministic grid posteriors, the independent check on the MCMC
# ---------------------------------------------------------------------------

def grid_posterior_a1(x, points=4001, nodes=96, span=5.0):
    """``p(a1 | x_1..n)`` for the angle_1-only layout, tabulated and normalised.

    Two passes: a coarse pass over +-span prior sd locates the mass, then a fine
    grid covers the region above 1e-14 of the peak.
    """
    x = np.atleast_2d(x)

    def log_post(grid):
        out = log_normal(grid, SD_A1)
        for start in range(0, len(x), 16):
            block = x[start:start + 16]
            out = out + log_marginal_a1(grid[:, None], block[None, :, 0], block[None, :, 1], nodes).sum(1)
        return out

    coarse = np.linspace(-span * SD_A1, span * SD_A1, 4001)
    lp = log_post(coarse)
    keep = np.flatnonzero(lp > lp.max() - 32.0)
    step = coarse[1] - coarse[0]
    grid = np.linspace(coarse[keep[0]] - 2 * step, coarse[keep[-1]] + 2 * step, points)
    lp = log_post(grid)
    density = np.exp(lp - lp.max())
    density /= np.trapezoid(density, grid)
    return grid, density


def grid_posterior_a1h(x, points=801, span=5.0):
    """``p(a1, h | x_1..n)`` for the (angle_1, height)-global layout on a 2-D grid.

    Returns ``(a1_grid, h_grid, density)`` with density normalised over the grid.
    The likelihood has integrable 1/sqrt edges, so this is a check to a few
    percent in moments, not a certificate to machine precision.
    """
    x = np.atleast_2d(x)

    def log_post(ga, gh):
        A, Hh = np.meshgrid(ga, gh, indexing="ij")
        out = log_normal(A, SD_A1) + log_normal(Hh, SD_H)
        for j in range(len(x)):
            out = out + log_lik_a1h(A, Hh, x[j, 0], x[j, 1])
        return out

    ga = np.linspace(-span * SD_A1, span * SD_A1, 401)
    gh = np.linspace(-span * SD_H, span * SD_H, 401)
    for _ in range(2):   # zoom twice onto the mass
        lp = log_post(ga, gh)
        live = lp > lp.max() - 32.0
        ia, ih = np.flatnonzero(live.any(1)), np.flatnonzero(live.any(0))
        da, dh = ga[1] - ga[0], gh[1] - gh[0]
        ga = np.linspace(ga[ia[0]] - 2 * da, ga[ia[-1]] + 2 * da, points)
        gh = np.linspace(gh[ih[0]] - 2 * dh, gh[ih[-1]] + 2 * dh, points)
    lp = log_post(ga, gh)
    density = np.exp(lp - lp.max())
    density /= density.sum() * (ga[1] - ga[0]) * (gh[1] - gh[0])
    return ga, gh, density
