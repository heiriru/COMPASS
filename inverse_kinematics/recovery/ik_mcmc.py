"""MCMC reference posteriors for the inverse-kinematics arm, batched over datasets.

The target is the exact, noise-free posterior (see :mod:`ik_exact`): ``(a2, a3)``
of every observation are summed out in closed form, and the sampler runs on what
is left, with the discrete elbow choice drawn exactly afterwards.

* layout ``angle``        -- state ``(a1, h_1..h_n)``, dimension ``1 + n``.
* layout ``angle_height`` -- state ``(a1, h)``, dimension 2.

Every move is a Metropolis-Hastings step with a symmetric or explicitly
corrected proposal; step sizes adapt during warmup only and are frozen after it,
so the retained draws come from a fixed, reversible kernel.

Moves (``angle`` layout; ``angle_height`` uses the obvious scalar-``h`` analogues)

1. per-observation random walk on ``h_j`` (all ``j`` in parallel; independent
   given ``a1``, so each is its own accept/reject);
2. per-observation reflection ``h_j -> 2 d_j - h_j`` with ``d_j = x1_j - L1 sin a1``:
   the allowed ``h_j`` set is two intervals mirrored about ``d_j``, and a
   random walk alone need not cross the gap;
3. random walk on ``a1`` with the ``h_j`` held;
4. random walk on ``a1`` with every ``h_j`` shifted by ``-L1 (sin a1' - sin a1)``,
   which keeps each ``x1_j - h_j - L1 sin a1`` fixed (unit-Jacobian, self-inverse
   under ``delta -> -delta``);
5. the arm's exact mirror ``a1 -> -a1, h_j -> 2 x1_j - h_j``, an involution of
   the likelihood (README "Making the posterior bimodal"); it lets chains cross
   between mirror branches;
6. an independence proposal ``a1' ~ prior`` combined with the shift of move 4.
"""
import numpy as np

import ik_exact as E

MCMC_WRAPS = (0,)     # 2*pi wraps change log p by <1.3e-8 on prior draws (checked)


def _loglik(a1, h, x1, x2):
    return E.log_lik_a1h(a1, h, x1, x2, wraps=MCMC_WRAPS)


class _Adapter:
    """Robbins-Monro on log step size towards a target acceptance, warmup only."""

    def __init__(self, shape, initial, target=0.4):
        self.log_scale = np.full(shape, np.log(initial))
        self.target = target

    @property
    def scale(self):
        return np.exp(self.log_scale)

    def update(self, accept_rate, iteration):
        self.log_scale += (accept_rate - self.target) / np.sqrt(iteration + 1.0)
        np.clip(self.log_scale, np.log(1e-5), np.log(5.0), out=self.log_scale)


# ---------------------------------------------------------------------------
# layout "angle": state (a1, h_1..h_n)
# ---------------------------------------------------------------------------

def _init_angle(x, chains, rng, tries=4000):
    """Overdispersed valid starts: a1 from the prior restricted to reachability,
    each h_j uniform on its allowed set given that a1."""
    B, n = x.shape[:2]
    cand = rng.normal(0.0, E.SD_A1, (B, tries))
    _, _, _, reach = E.height_support(cand[:, :, None], x[:, None, :, 0], x[:, None, :, 1])
    ok = reach.all(-1)
    if not ok.any(1).all():
        raise RuntimeError("no reachable angle_1 among prior draws for some dataset")
    a1 = np.empty((B, chains))
    for b in range(B):
        a1[b] = rng.choice(cand[b, ok[b]], size=chains, replace=ok[b].sum() < chains)
    d, lo, hi, _ = E.height_support(a1[..., None], x[:, None, :, 0], x[:, None, :, 1])
    u = lo + (hi - lo) * rng.random(lo.shape)
    side = np.where(rng.random(lo.shape) < 0.5, 1.0, -1.0)
    return a1, d - side * u


def sample_angle(x, rng, chains=8, warmup=3000, draws=500, thin=10):
    """Posterior draws for layout ``angle``.

    ``x`` is ``(B, n, 2)``. Returns a dict with ``a1`` ``(B, chains, draws)``,
    ``h`` ``(B, chains, draws, n)`` and acceptance rates.
    """
    x = np.asarray(x, np.float64)
    B, n = x.shape[:2]
    x1, x2 = x[:, None, :, 0], x[:, None, :, 1]              # (B,1,n)
    a1, h = _init_angle(x, chains, rng)

    def ell(a, hh):                                          # (B,C,n)
        return E.log_normal(hh, E.SD_H) + _loglik(a[..., None], hh, x1, x2)

    cur = ell(a1, h)
    step_h = _Adapter((B, 1, n), 0.05)
    step_a = _Adapter((B, 1), 0.05)
    step_s = _Adapter((B, 1), 0.05)
    moves = ("h_walk", "h_reflect", "a1_walk", "a1_shift", "mirror", "a1_independent")
    accepted = {m: 0.0 for m in moves}
    out_a1 = np.empty((B, chains, draws))
    out_h = np.empty((B, chains, draws, n))
    total = warmup + draws * thin

    def joint_step(a_new, h_new, extra, name):
        nonlocal a1, h, cur
        new = ell(a_new, h_new)
        log_ratio = (np.nan_to_num(new.sum(-1) - cur.sum(-1), nan=-np.inf) + extra
                     + E.log_normal(a_new, E.SD_A1) - E.log_normal(a1, E.SD_A1))
        take = np.log(rng.random(a1.shape)) < log_ratio
        a1 = np.where(take, a_new, a1)
        h = np.where(take[..., None], h_new, h)
        cur = np.where(take[..., None], new, cur)
        if iteration >= warmup:
            accepted[name] += take.mean()
        return take

    for iteration in range(total):
        # 1. per-coordinate walk on h_j
        prop = h + step_h.scale * rng.normal(size=h.shape)
        new = ell(a1, prop)
        take = np.log(rng.random(h.shape)) < np.nan_to_num(new - cur, nan=-np.inf)
        h, cur = np.where(take, prop, h), np.where(take, new, cur)
        if iteration < warmup:
            step_h.update(take.mean(1, keepdims=True), iteration)
        else:
            accepted["h_walk"] += take.mean()
        # 2. per-coordinate reflection about d_j
        prop = 2.0 * (x1 - E.L1 * np.sin(a1)[..., None]) - h
        new = ell(a1, prop)
        take = np.log(rng.random(h.shape)) < np.nan_to_num(new - cur, nan=-np.inf)
        h, cur = np.where(take, prop, h), np.where(take, new, cur)
        if iteration >= warmup:
            accepted["h_reflect"] += take.mean()
        # 3. a1 walk, h held
        a_new = a1 + step_a.scale * rng.normal(size=a1.shape)
        take = joint_step(a_new, h, 0.0, "a1_walk")
        if iteration < warmup:
            step_a.update(take.mean(1, keepdims=True), iteration)
        # 4. a1 walk with the h_j shift
        a_new = a1 + step_s.scale * rng.normal(size=a1.shape)
        h_new = h - E.L1 * (np.sin(a_new) - np.sin(a1))[..., None]
        take = joint_step(a_new, h_new, 0.0, "a1_shift")
        if iteration < warmup:
            step_s.update(take.mean(1, keepdims=True), iteration)
        # 5. exact mirror involution
        joint_step(-a1, 2.0 * x1 - h, 0.0, "mirror")
        # 6. independence proposal from the prior, with the shift
        a_new = rng.normal(0.0, E.SD_A1, a1.shape)
        h_new = h - E.L1 * (np.sin(a_new) - np.sin(a1))[..., None]
        joint_step(a_new, h_new, E.log_normal(a1, E.SD_A1) - E.log_normal(a_new, E.SD_A1), "a1_independent")

        if iteration >= warmup and (iteration - warmup) % thin == thin - 1:
            k = (iteration - warmup) // thin
            out_a1[:, :, k] = a1
            out_h[:, :, k] = h
    kept = total - warmup
    return {"a1": out_a1, "h": out_h, "acceptance": {m: v / kept for m, v in accepted.items()}}


def sample_angle_collapsed(x, rng, chains=8, warmup=500, draws=500, thin=2, nodes=32):
    """Collapsed sampler for layout ``angle`` -- the production reference.

    Metropolis on ``a1`` alone against ``log N(a1) + sum_j log p(x_j | a1)``, the
    heights integrated out by :func:`ik_exact.log_marginal_a1`; then every
    ``h_j`` is drawn exactly from ``p(h_j | a1, x_j)``. Because ``h_j`` never sits
    in the state, the chain cannot be pinned by it (the failure of
    :func:`sample_angle` at large ``n``). Moves: adapted random walk, the mirror
    ``a1 -> -a1``, and an independence proposal from the prior.

    Returns the same dict as :func:`sample_angle`.
    """
    x = np.asarray(x, np.float64)
    B, n = x.shape[:2]
    x1, x2 = x[:, None, :, 0], x[:, None, :, 1]

    def target(a):
        ll = E.log_marginal_a1(a[..., None], x1, x2, nodes=nodes, wraps=MCMC_WRAPS).sum(-1)
        return np.nan_to_num(ll, nan=-np.inf) + E.log_normal(a, E.SD_A1)

    # Start from reachable prior draws (the likelihood is -inf elsewhere).
    cand = rng.normal(0.0, E.SD_A1, (B, 4000))
    _, _, _, reach = E.height_support(cand[:, :, None], x[:, None, :, 0], x[:, None, :, 1])
    a1 = np.empty((B, chains))
    for b in range(B):
        ok = cand[b, reach[b].all(-1)]
        if len(ok) == 0:
            raise RuntimeError(f"dataset {b}: no reachable angle_1 among prior draws")
        a1[b] = rng.choice(ok, size=chains, replace=len(ok) < chains)
    cur = target(a1)
    step = _Adapter((B, 1), 0.1)
    moves = ("walk", "mirror", "independent")
    accepted = {m: 0.0 for m in moves}
    out = np.empty((B, chains, draws))
    total = warmup + draws * thin
    for iteration in range(total):
        warm = iteration < warmup
        for name in moves:
            extra = 0.0
            if name == "walk":
                prop = a1 + step.scale * rng.normal(size=a1.shape)
            elif name == "mirror":
                prop = -a1
            else:
                prop = rng.normal(0.0, E.SD_A1, a1.shape)
                extra = E.log_normal(a1, E.SD_A1) - E.log_normal(prop, E.SD_A1)
            new = target(prop)
            take = np.log(rng.random(a1.shape)) < np.nan_to_num(new - cur, nan=-np.inf) + extra
            a1, cur = np.where(take, prop, a1), np.where(take, new, cur)
            if warm and name == "walk":
                step.update(take.mean(1, keepdims=True), iteration)
            elif not warm:
                accepted[name] += take.mean()
        if not warm and (iteration - warmup) % thin == thin - 1:
            out[:, :, (iteration - warmup) // thin] = a1
    h = E.draw_height(out[..., None], x[:, None, None, :, 0], x[:, None, None, :, 1], rng, wraps=MCMC_WRAPS)
    kept = total - warmup
    return {"a1": out, "h": h, "acceptance": {m: v / kept for m, v in accepted.items()}}


# ---------------------------------------------------------------------------
# layout "angle_height": state (a1, h)
# ---------------------------------------------------------------------------

def _init_angle_height(x, chains, rng, tries=200_000):
    B = x.shape[0]
    a1 = np.empty((B, chains))
    h = np.empty((B, chains))
    for b in range(B):
        ca = rng.normal(0.0, E.SD_A1, tries)
        ch = rng.normal(0.0, E.SD_H, tries)
        lp = _loglik(ca[:, None], ch[:, None], x[b, None, :, 0], x[b, None, :, 1]).sum(1)
        ok = np.flatnonzero(np.isfinite(lp))
        if len(ok) >= chains:
            pick = rng.choice(ok, size=chains, replace=False)
            a1[b], h[b] = ca[pick], ch[pick]
        else:
            # Large n: the feasible set (every |x_j - joint 2| in [R_MIN, R_MAX]) is too small for
            # prior draws to hit. Find it by minimising the reach violation, then jitter.
            a1[b], h[b] = _feasible_starts(x[b], chains, rng)
    return a1, h


def _reach_violation(a, v, x):
    """Total distance by which joint 2 = (v, L1 cos a) misses the reach annuli of all rows."""
    r = np.hypot(x[:, 0] - v, x[:, 1] - E.L1 * np.cos(a))
    return np.sum(np.maximum(r - E.R_MAX, 0) + np.maximum(E.R_MIN - r, 0))


def _feasible_starts(x, chains, rng, tries=20_000):
    from scipy.optimize import minimize
    A, V = np.meshgrid(np.linspace(-2.5, 2.5, 201), np.linspace(-2.5, 2.5, 201), indexing="ij")
    r = np.hypot(x[None, None, :, 0] - V[..., None], x[None, None, :, 1] - E.L1 * np.cos(A)[..., None])
    viol = (np.maximum(r - E.R_MAX, 0) + np.maximum(E.R_MIN - r, 0)).sum(-1)
    starts = np.argsort(viol.ravel())[:20]
    points = []
    for s in starts:
        res = minimize(lambda z: _reach_violation(z[0], z[1], x), [A.ravel()[s], V.ravel()[s]],
                       method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
        points.append(res.x)
    points = np.array(points)
    found = []
    for scale in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        cand = points[rng.integers(len(points), size=tries)] + scale * rng.normal(size=(tries, 2))
        ca, ch = cand[:, 0], cand[:, 1] - E.L1 * np.sin(cand[:, 0])
        lp = _loglik(ca[:, None], ch[:, None], x[None, :, 0], x[None, :, 1]).sum(1)
        good = np.flatnonzero(np.isfinite(lp))
        found.extend(zip(ca[good], ch[good]))
        if len(found) >= chains:
            break
    if not found:
        raise RuntimeError("no feasible (a1, h) found even by violation minimisation")
    found = np.array(found)
    pick = found[rng.choice(len(found), size=chains, replace=len(found) < chains)]
    return pick[:, 0], pick[:, 1]


def _t_logpdf(z, mean, chol, df):
    """Multivariate-t log density (up to a constant shared by all points) for 2-D ``z``."""
    diff = z - mean
    solved = np.linalg.solve(chol, diff[..., None])[..., 0]
    maha = (solved ** 2).sum(-1)
    log_det = np.log(np.abs(chol[..., 0, 0] * chol[..., 1, 1]))
    return -0.5 * (df + 2.0) * np.log1p(maha / df) - log_det


def sample_angle_height(x, rng, chains=8, warmup=4000, draws=500, thin=10, df=4.0, inflate=2.0):
    """Posterior draws for layout ``angle_height``: ``a1``, ``h`` each ``(B, chains, draws)``.

    The chain runs in ``(a1, v)`` with ``v = h + L1 sin a1``, the vertical position
    of joint 2 (unit Jacobian). The data pin joint 2 = ``(v, L1 cos a1)``, so in
    ``(a1, h)`` the posterior is a thin curved ridge; in ``(a1, v)`` it is nearly
    axis-aligned, and near ``a1 = 0`` it is two-lobed in ``+-a1`` (cos is even).

    Moves: walks in ``a1`` (v held), in ``v`` (a1 held), a joint walk with the
    warmup covariance, the involution ``a1 -> -a1`` at fixed ``v`` (joint 2 does
    not move), and, after warmup, an independence proposal from a multivariate
    t fitted to the second half of warmup (scale inflated by ``inflate``).
    """
    x = np.asarray(x, np.float64)
    B, n = x.shape[:2]
    x1, x2 = x[:, None, :, 0], x[:, None, :, 1]
    a1, h = _init_angle_height(x, chains, rng)
    v = h + E.L1 * np.sin(a1)

    def target(a, vv):
        hh = vv - E.L1 * np.sin(a)
        ll = _loglik(a[..., None], hh[..., None], x1, x2).sum(-1)
        return np.nan_to_num(ll, nan=-np.inf) + E.log_normal(a, E.SD_A1) + E.log_normal(hh, E.SD_H)

    cur = target(a1, v)
    steps = {m: _Adapter((B, 1), 0.05) for m in ("a1_walk", "v_walk", "joint")}
    moves = ("a1_walk", "v_walk", "joint", "flip", "independent")
    accepted = {m: 0.0 for m in moves}
    chol = np.broadcast_to(np.eye(2) * 0.05, (B, 1, 2, 2)).copy()
    fit_mean, fit_chol = None, None
    history = []
    out_a1 = np.empty((B, chains, draws))
    out_h = np.empty((B, chains, draws))
    total = warmup + draws * thin

    for iteration in range(total):
        warm = iteration < warmup
        for name in moves:
            extra = 0.0
            if name == "a1_walk":
                a_new, v_new = a1 + steps[name].scale * rng.normal(size=a1.shape), v
            elif name == "v_walk":
                a_new, v_new = a1, v + steps[name].scale * rng.normal(size=v.shape)
            elif name == "joint":
                dz = steps[name].scale[..., None] * np.einsum("bcij,bcj->bci", chol,
                                                              rng.normal(size=(B, chains, 2)))
                a_new, v_new = a1 + dz[..., 0], v + dz[..., 1]
            elif name == "flip":
                a_new, v_new = -a1, v
            else:
                if fit_mean is None:
                    continue
                z = rng.standard_normal((B, chains, 2))
                scale = np.sqrt(df / rng.chisquare(df, (B, chains)))[..., None]
                prop = fit_mean + scale * np.einsum("bcij,bcj->bci", fit_chol, z)
                a_new, v_new = prop[..., 0], prop[..., 1]
                old = np.stack([a1, v], -1)
                extra = _t_logpdf(old, fit_mean, fit_chol, df) - _t_logpdf(prop, fit_mean, fit_chol, df)
            new = target(a_new, v_new)
            take = np.log(rng.random(a1.shape)) < np.nan_to_num(new - cur, nan=-np.inf) + extra
            a1, v, cur = np.where(take, a_new, a1), np.where(take, v_new, v), np.where(take, new, cur)
            if warm and name in steps:
                steps[name].update(take.mean(1, keepdims=True), iteration)
            elif not warm:
                accepted[name] += take.mean()
        if warm:
            history.append(np.stack([a1, v], -1))
            if iteration + 1 in (warmup // 4, warmup // 2, 3 * warmup // 4, warmup):
                window = np.concatenate(history[len(history) // 2:], axis=1)       # (B, C*T, 2)
                mean = window.mean(1, keepdims=True)
                cov = np.einsum("bti,btj->bij", window - mean, window - mean) / window.shape[1]
                cov += 1e-12 * np.eye(2)
                chol = np.linalg.cholesky(cov)[:, None]
                if iteration + 1 == warmup:
                    fit_mean, fit_chol = mean, inflate * chol
                else:
                    steps["joint"] = _Adapter((B, 1), 1.0)
                history = []
        elif (iteration - warmup) % thin == thin - 1:
            k = (iteration - warmup) // thin
            out_a1[:, :, k], out_h[:, :, k] = a1, v - E.L1 * np.sin(a1)
    kept = total - warmup
    return {"a1": out_a1, "h": out_h, "acceptance": {m: v_ / kept for m, v_ in accepted.items()}}


# ---------------------------------------------------------------------------
# locals and diagnostics
# ---------------------------------------------------------------------------

def draw_locals(a1, h, x, rng, chunk=200_000):
    """Exact ``(a2_j, a3_j)`` for every draw and observation.

    ``a1`` is ``(..., )`` and ``h`` either ``(..., n)`` (angle layout) or
    ``(...)`` (angle_height). ``x`` is ``(n, 2)``. Returns two ``(..., n)`` arrays.
    """
    a1 = np.asarray(a1)
    h = np.asarray(h)
    if h.shape == a1.shape:
        h = np.repeat(h[..., None], len(x), -1)
    shape = h.shape
    A = np.repeat(a1.reshape(-1, 1), shape[-1], 1).ravel()
    Hf = h.reshape(-1)
    X1 = np.broadcast_to(x[:, 0], shape).reshape(-1)
    X2 = np.broadcast_to(x[:, 1], shape).reshape(-1)
    a2 = np.empty_like(Hf)
    a3 = np.empty_like(Hf)
    for s in range(0, len(Hf), chunk):
        sl = slice(s, s + chunk)
        a2[sl], a3[sl] = E.draw_elbow(A[sl], Hf[sl], X1[sl], X2[sl], rng)
    return a2.reshape(shape), a3.reshape(shape)


def _rank_normal(values):
    from scipy.special import ndtri
    from scipy.stats import rankdata
    flat = rankdata(values, method="average").reshape(values.shape)
    return ndtri((flat - 0.375) / (values.size + 0.25))


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, half:2 * half]], 0)


def _rhat_raw(chains):
    m, n = chains.shape
    means = chains.mean(1)
    within = chains.var(1, ddof=1).mean()
    between = n * means.var(ddof=1)
    return np.sqrt(((n - 1) / n * within + between / n) / within)


def rhat(chains):
    """Rank-normalised split R-hat (Vehtari et al. 2021); ``chains`` is ``(C, S)``."""
    split = _split(np.asarray(chains, np.float64))
    bulk = _rhat_raw(_rank_normal(split))
    folded = _rhat_raw(_rank_normal(np.abs(split - np.median(split))))
    return float(max(bulk, folded))


def ess_bulk(chains):
    """Bulk effective sample size (rank-normalised split chains, Geyer's monotone sequence)."""
    z = _rank_normal(_split(np.asarray(chains, np.float64)))
    m, n = z.shape
    centred = z - z.mean(1, keepdims=True)
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centred, size, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=1)[:, :n] / n
    chain_var = acov[:, 0] * n / (n - 1)
    within = chain_var.mean()
    var_plus = within * (n - 1) / n + z.mean(1).var(ddof=1)
    rho = 1.0 - (within - acov.mean(0)) / var_plus
    rho[0] = 1.0
    pairs = rho[:-1:2] + rho[1::2]
    positive = np.flatnonzero(pairs < 0)
    pairs = pairs[: positive[0]] if len(positive) else pairs
    pairs = np.minimum.accumulate(pairs)
    tau = -1.0 + 2.0 * pairs.sum()
    return float(m * n / max(tau, 1.0 / np.log10(m * n)))
