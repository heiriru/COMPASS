"""Reference WAIC and evidence for a dataset: exact quadrature, MCMC, nested sampling.

All three use the exact row likelihood of the model (families.py), never a network.

* ``quadrature``: exact for models with at most one global and one local
  (``offset`` and ``noise`` families). The posterior factorises as
  ``p(g | x) prod_i p(l_i | g, x_i)``, so every WAIC expectation is a 2-D grid sum per
  row and the evidence a 1-D sum over g.
* ``mcmc``: emcee ensemble sampler (differential-evolution + stretch moves; emcee's DESnookerMove gave a 25%-too-narrow posterior on the offset model and is not used) over the flat
  ``(g, l_1..l_N)``; WAIC from its draws.
* ``nested``: dynesty static nested sampling (random-slice proposals); the log
  evidence with its error, and WAIC from the equally weighted posterior draws.
"""

import math
import time

import numpy as np
from scipy.special import logsumexp


def waic_from_loglik(ll):
    """WAIC pieces from an ``(N, S)`` matrix (numpy twin of compass.information_criteria.waic)."""
    s = ll.shape[1]
    lppd = logsumexp(ll, axis=1) - math.log(s)
    p = ll.var(axis=1, ddof=1)
    out = dict(waic=float(-2 * (lppd - p).sum()), lppd=float(lppd.sum()),
               p_waic=float(p.sum()), elpd_rows=(lppd - p).tolist())
    if s >= 10:
        from compass.information_criteria import psis_loo
        loo = psis_loo(ll, mc_groups=1)
        out.update(loo=float(loo["total"]), elpd_loo_rows=loo["elpd_rows"].tolist(),
                   k_max=float(loo["pareto_k"].max()), n_k_high=loo["n_k_high"])
    return out


# ---------------------------------------------------------------------------
# exact quadrature (H <= 1, L = 1)
# ---------------------------------------------------------------------------

def _row_tables(model, g_grid, l_grid, x_row):
    """log p(l) + log p(x | g, l) on the (G, Lg) grid."""
    g = g_grid.reshape(-1, 1, 1) if model.n_global else np.zeros((1, 1, 0))
    l = l_grid.reshape(1, -1, 1)
    ll = np.broadcast_to(model.loglik(g, l, x_row), (g.shape[0], len(l_grid)))
    return ll, model.log_prior_local(l_grid.reshape(-1, 1))


def quadrature(model, x, n_g=601, n_l=1601, width=8.0, with_loo=True):
    """Exact logZ and WAIC by grid quadrature (one global at most, one local)."""
    if model.n_global > 1 or model.n_local != 1:
        raise ValueError("quadrature needs n_global <= 1 and n_local == 1")
    sl = float(model.local_std[0])
    l_grid = np.linspace(model.local_mean[0] - width * sl, model.local_mean[0] + width * sl, n_l)
    dl = l_grid[1] - l_grid[0]

    def row_marginals(g_grid):
        out = []
        for x_row in x:
            ll, lp = _row_tables(model, g_grid, l_grid, x_row)
            out.append(logsumexp(ll + lp[None], axis=1) + math.log(dl))
        return np.stack(out, 1)                     # (G, N) log m_i(g)

    if model.n_global:
        sg, mg = float(model.global_std[0]), float(model.global_mean[0])
        coarse = np.linspace(mg - width * sg, mg + width * sg, 2001)
        post = model.log_prior_global(coarse[:, None]) + row_marginals(coarse).sum(1)
        keep = coarse[post > post.max() - 40.0]
        pad = 2 * (coarse[1] - coarse[0])
        g_grid = np.linspace(keep.min() - pad, keep.max() + pad, n_g)
        dg = g_grid[1] - g_grid[0]
        log_m = row_marginals(g_grid)
        log_joint_g = model.log_prior_global(g_grid[:, None]) + log_m.sum(1)
        log_z = float(logsumexp(log_joint_g) + math.log(dg))
        w_g = np.exp(log_joint_g + math.log(dg) - log_z)           # posterior weights of g
    else:
        g_grid = np.zeros(1)
        log_m = row_marginals(g_grid)
        log_z = float(log_m.sum())
        w_g = np.ones(1)

    lppd, p_rows = [], []
    for i, x_row in enumerate(x):
        ll, lp = _row_tables(model, g_grid, l_grid, x_row)
        # p(l | g, x_i) on the grid, per g.
        w_l = np.exp(ll + lp[None] + math.log(dl) - log_m[:, i:i + 1])
        weights = w_g[:, None] * w_l                                   # joint (g, l_i)
        lppd.append(logsumexp(ll, b=weights))
        m1 = (weights * ll).sum()
        m2 = (weights * ll ** 2).sum()
        p_rows.append(m2 - m1 ** 2)
    lppd, p_rows = np.array(lppd), np.array(p_rows)
    loo_rows = None
    if with_loo and len(x) > 1:
        loo_rows = [log_z - quadrature(model, np.delete(x, i, 0), n_g, n_l, width,
                                       with_loo=False)["log_z"] for i in range(len(x))]
    return dict(log_z=log_z,
                loo=None if loo_rows is None else float(-2 * np.sum(loo_rows)),
                elpd_loo_rows=loo_rows, waic=float(-2 * (lppd - p_rows).sum()),
                lppd=float(lppd.sum()), p_waic=float(p_rows.sum()),
                elpd_rows=(lppd - p_rows).tolist(),
                g_mean=float((w_g * g_grid).sum()) if model.n_global else None,
                g_sd=float(math.sqrt(max((w_g * g_grid ** 2).sum()
                                         - (w_g * g_grid).sum() ** 2, 0.0)))
                if model.n_global else None)


# ---------------------------------------------------------------------------
# MCMC (emcee)
# ---------------------------------------------------------------------------

def mcmc(model, x, seed=0, n_steps=6000, burn=2000, thin=10, walkers=None, init=None,
         restarts=0, restart_gap=20.0):
    """emcee with DE moves; returns WAIC from the draws and convergence diagnostics.

    ``init`` (optional, ``(W, D)``) sets the walkers' start; otherwise prior draws.
    ``restarts`` extra burn-in rounds of ``burn`` steps, each followed by moving walkers
    more than ``restart_gap`` nats below the best onto the good ones (then ``burn`` more
    steps are discarded from the main run as usual).
    """
    import emcee

    n_rows = x.shape[0]
    ndim = model.n_global + n_rows * model.n_local
    walkers = walkers or max(64, 4 * ndim)
    rng = np.random.default_rng(seed)
    if init is None:
        init = np.concatenate([model.sample_global(walkers, rng),
                               model.sample_local(walkers * n_rows, rng).reshape(walkers, -1)], 1)

    def log_prob(theta):
        out = model.log_posterior_unnorm(theta, x)
        return np.where(np.isfinite(out), out, -np.inf)

    sampler = emcee.EnsembleSampler(
        walkers, ndim, log_prob, vectorize=True,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.StretchMove(), 0.2)])
    t0 = time.time()
    sampler.random_state = np.random.RandomState(int(rng.integers(2 ** 31))).get_state()
    for _ in range(restarts):
        # Rugged likelihoods (misspecified ODE fits) strand walkers in local optima hundreds
        # of nats below the best; their mass is negligible but they inflate Var(log p).
        # Move walkers far below the best onto jittered copies of the good ones, re-burn.
        state = sampler.run_mcmc(init, burn, progress=False)
        lp = state.log_prob
        good = np.where(lp > lp.max() - restart_gap)[0]
        bad = np.where(lp <= lp.max() - restart_gap)[0]
        init = state.coords.copy()
        if len(bad):
            src = state.coords[good]
            scale = src.std(0) + 1e-6
            pick = rng.integers(0, len(good), len(bad))
            init[bad] = src[pick] + 0.1 * scale * rng.standard_normal((len(bad), ndim))
        sampler.reset()
    sampler.run_mcmc(init, n_steps, progress=False)
    seconds = time.time() - t0
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    ll = model.log_lik_rows(chain, x).T                               # (N, S)
    out = waic_from_loglik(ll)
    g_draws = chain[:, :model.n_global]
    out.update(seconds=seconds, n_draws=int(chain.shape[0]),
               tau_max=float(np.nanmax(tau)),
               n_eff=float(walkers * (n_steps - burn) / np.nanmax(tau)),
               acceptance=float(sampler.acceptance_fraction.mean()),
               g_mean=g_draws.mean(0).tolist(), g_sd=g_draws.std(0).tolist())
    return out, chain


def mcmc_rows(model, x, seed=0, n_steps=3000, burn=1000, thin=5, walkers=32, restarts=2):
    """MCMC for a model without globals: the posterior factorises over rows, so each
    row gets its own low-dimensional emcee run. Returns the same dict as ``mcmc`` and
    the draws ``(S, N L)`` (row draws paired by index; rows are independent)."""
    if model.n_global:
        raise ValueError("mcmc_rows needs a model without globals")
    lls, draws, taus, secs, accs = [], [], [], 0.0, []
    for i in range(x.shape[0]):
        out, chain = mcmc(model, x[i:i + 1], seed=seed * 7919 + i, n_steps=n_steps,
                          burn=burn, thin=thin, walkers=walkers, restarts=restarts)
        lls.append(model.log_lik_rows(chain, x[i:i + 1])[:, 0])
        draws.append(chain)
        taus.append(out["tau_max"]); secs += out["seconds"]; accs.append(out["acceptance"])
    ll = np.stack(lls)                                                  # (N, S)
    res = waic_from_loglik(ll)
    res.update(seconds=secs, n_draws=int(ll.shape[1]), tau_max=float(np.nanmax(taus)),
               n_eff=float(walkers * (n_steps - burn) / np.nanmax(taus)),
               acceptance=float(np.mean(accs)), g_mean=[], g_sd=[])
    return res, np.concatenate(draws, 1)


# ---------------------------------------------------------------------------
# nested sampling (dynesty)
# ---------------------------------------------------------------------------

def nested(model, x, seed=0, nlive=500, dlogz=0.05, sample="rslice"):
    """dynesty static nested sampler: log evidence and WAIC from its posterior draws."""
    import dynesty

    n_rows = x.shape[0]
    ndim = model.n_global + n_rows * model.n_local
    rng = np.random.default_rng(seed)

    def loglike(theta):
        value = float(model.log_lik_rows(theta, x).sum())
        return value if np.isfinite(value) else -1e300

    def transform(u):
        return model.prior_transform(u, n_rows)

    t0 = time.time()
    sampler = dynesty.NestedSampler(loglike, transform, ndim, nlive=nlive, bound="multi",
                                    sample=sample, rstate=rng)
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    res = sampler.results
    seconds = time.time() - t0
    draws = res.samples_equal(rstate=np.random.default_rng(seed + 1))
    ll = model.log_lik_rows(draws, x).T
    out = waic_from_loglik(ll)
    g_draws = draws[:, :model.n_global]
    out.update(log_z=float(res.logz[-1]), log_z_err=float(res.logzerr[-1]),
               n_calls=int(np.sum(res.ncall)), seconds=seconds, n_draws=int(len(draws)),
               g_mean=g_draws.mean(0).tolist(), g_sd=g_draws.std(0).tolist())
    return out, draws
