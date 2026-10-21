"""Check of the explicit three-level rule, the level-by-level N-level rule, and joint sampling.

Linear-Gaussian hierarchies, so every factor is exactly Gaussian and the composed score must
be the exact diffused tall-data score of the joint latent vector (globals, mids, locals).

  1. explicit 3-level passes (eqs. of three_level_composition.tex)  vs exact score
  2. level-by-level N-level recursion (N = 2..5)                     vs exact score
  3. joint sampling: probability-flow ODE driven by the composed 3-level score; sample
     moments (incl. cross-level covariances) vs the exact joint posterior, and the same
     sampler with the group-prior factors p(g,h_k)^(1-n_k) dropped.
"""
import os

CPU_THREAD_LIMIT = 3
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = str(CPU_THREAD_LIMIT)
try:
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT])
except (AttributeError, OSError):
    pass

import numpy as np

rng = np.random.default_rng(1)


# --------------------------------------------------------------------------- model
class NestedModel:
    """Perfectly nested linear-Gaussian hierarchy. branching[m] = children per level-(m+1) node;
    one observation per leaf. Node ids are tuples (multi-indices); () is the root."""

    def __init__(self, dims, branching, dx=2):
        self.N = len(dims)
        self.levels = [[()]]
        for b in branching:
            self.levels.append([p + (i,) for p in self.levels[-1] for i in range(b)])
        self.nodes = [v for lev in self.levels for v in lev]
        self.dim = {v: dims[len(v)] for v in self.nodes}
        off, self.sl = 0, {}
        for v in self.nodes:
            self.sl[v] = np.arange(off, off + self.dim[v]); off += self.dim[v]
        self.D = off
        W = np.zeros((self.D, self.D)); b = np.zeros(self.D); Q = np.zeros((self.D, self.D))
        for v in self.nodes:
            for a in self.path(v)[:-1]:           # CPD may depend on all ancestors
                W[np.ix_(self.sl[v], self.sl[a])] = rng.normal(size=(self.dim[v], self.dim[a])) * 0.6
            b[self.sl[v]] = rng.normal(size=self.dim[v])
            A = rng.normal(size=(self.dim[v],) * 2)
            Q[np.ix_(self.sl[v], self.sl[v])] = A @ A.T / self.dim[v] + 0.3 * np.eye(self.dim[v])
        T = np.linalg.inv(np.eye(self.D) - W)
        self.mu0, self.C0 = T @ b, T @ Q @ T.T
        theta = rng.multivariate_normal(self.mu0, self.C0)
        self.obs = {}
        for leaf in self.levels[-1]:
            I = self.idx(leaf)
            H = rng.normal(size=(dx, len(I)))
            x = H @ theta[I] + rng.normal(size=dx) * 0.7
            self.obs[leaf] = (H, 0.49 * np.eye(dx), x)

    def path(self, v):
        return [v[:m] for m in range(len(v) + 1)]

    def idx(self, v):
        return np.concatenate([self.sl[a] for a in self.path(v)])

    def children(self, v):
        if len(v) + 1 >= self.N:
            return []
        return [c for c in self.levels[len(v) + 1] if c[:-1] == v]

    def gauss(self, I, leaves):
        """Exact Gaussian p(theta_I | x_leaves) under the prior marginal on I."""
        m, C = self.mu0[I], self.C0[np.ix_(I, I)]
        P, h = np.linalg.inv(C), np.linalg.solve(C, m)
        for leaf in leaves:
            H, R, x = self.obs[leaf]
            Hf = np.zeros((len(x), self.D)); Hf[:, self.idx(leaf)] = H
            Hs = Hf[:, I]
            P, h = P + Hs.T @ np.linalg.solve(R, Hs), h + Hs.T @ np.linalg.solve(R, x)
        C = np.linalg.inv(P)
        return C @ h, C


def factor(m, C, Y, lam):
    """Diffused score at rows of Y (S x d) and backward precision of N(m, C)."""
    s = -np.linalg.solve(C + lam**2 * np.eye(len(m)), (Y - m).T).T
    return s, np.linalg.inv(C) + np.eye(len(m)) / lam**2


def exact_score(model, Y, lam):
    m, C = model.gauss(np.arange(model.D), list(model.obs))
    return factor(m, C, Y, lam)[0]


# --------------------------------------------------- level-by-level N-level rule
def nlevel_rule(model, Y, lam, drop_mid_priors=False):
    """Upward pass level N..1 (merge + Schur), then downward back-substitution."""
    N = model.N
    Q, sig, S, G = {}, {}, {}, {}
    for leaf in model.levels[-1]:                       # level N: network factors
        I = model.idx(leaf)
        Q[leaf] = None
        m, C = model.gauss(I, [leaf])
        sig[leaf], Q[leaf] = factor(m, C, Y[:, I], lam)
    for lev in range(N - 1, -1, -1):
        if lev < N - 1:                                 # merge children + prior factor on path(v)
            for v in model.levels[lev]:
                I = model.idx(v); ch = model.children(v); n = len(ch)
                Qv = sum(S[c] for c in ch)
                rv = sum(np.einsum("ab,sb->sa", S[c], sig[c][:, :len(I)]) for c in ch)
                if not (drop_mid_priors and lev > 0):
                    m0, C0 = model.mu0[I], model.C0[np.ix_(I, I)]
                    s0, P0 = factor(m0, C0, Y[:, I], lam)
                    Qv = Qv + (1 - n) * P0
                    rv = rv + (1 - n) * s0 @ P0.T
                Q[v], sig[v] = Qv, np.linalg.solve(Qv, rv.T).T
        if lev > 0:                                     # eliminate own block -> message to parent
            for v in model.levels[lev]:
                dv = model.dim[v]; Qv = Q[v]
                A, B, C = Qv[:-dv, :-dv], Qv[:-dv, -dv:], Qv[-dv:, -dv:]
                G[v] = -np.linalg.solve(C, B.T)
                S[v] = A + B @ G[v]
    u = np.zeros_like(Y)
    u[:, model.sl[()]] = sig[()]
    for lev in range(1, N):
        for v in model.levels[lev]:
            Ia = model.idx(v[:-1]); dv = model.dim[v]
            u[:, model.sl[v]] = sig[v][:, -dv:] + (u[:, Ia] - sig[v][:, :-dv]) @ G[v].T
    return u


# ----------------------------------------------- explicit three-level formulas
def three_level_rule(model, Y, lam):
    """Literal transcription of the boxed three-level rule (passes 1-3 + back-substitution)."""
    assert model.N == 3
    g = (); dg = model.dim[g]
    groups = model.levels[1]
    m, C = model.mu0[model.sl[g]], model.C0[np.ix_(model.sl[g], model.sl[g])]
    s0, P0 = factor(m, C, Y[:, model.sl[g]], lam)
    K = len(groups)
    T, tau, keep = [], [], {}
    for k in groups:
        a = model.idx(k); dh = model.dim[k]; leaves = model.children(k); nk = len(leaves)
        # pass 1: each local -> message (S_kj, s^a_kj) on a_k = (g, h_k)
        Qk = np.zeros((len(a), len(a))); qk = np.zeros((len(Y), len(a)))
        for leaf in leaves:
            I = model.idx(leaf)
            mj, Cj = model.gauss(I, [leaf])
            s, P = factor(mj, Cj, Y[:, I], lam)
            A, B, Cc = P[:len(a), :len(a)], P[:len(a), len(a):], P[len(a):, len(a):]
            Kkj = -np.linalg.solve(Cc, B.T)
            Skj = A + B @ Kkj
            keep[leaf] = (Kkj, s)
            Qk += Skj; qk += s[:, :len(a)] @ Skj.T
        # pass 2: merge with the group prior factor (1 - n_k) on (g, h_k)
        mk, Ck = model.mu0[a], model.C0[np.ix_(a, a)]
        s0k, P0k = factor(mk, Ck, Y[:, a], lam)
        Qk += (1 - nk) * P0k; qk += (1 - nk) * s0k @ P0k.T
        sigk = np.linalg.solve(Qk, qk.T).T
        Qgg, Qgh, Qhh = Qk[:dg, :dg], Qk[:dg, dg:], Qk[dg:, dg:]
        Gk = -np.linalg.solve(Qhh, Qgh.T)
        Tk = Qgg + Qgh @ Gk
        keep[k] = (Gk, sigk)
        T.append(Tk); tau.append(sigk[:, :dg] @ Tk.T)
    # pass 3: global
    L = sum(T) + (1 - K) * P0
    ug = np.linalg.solve(L, (sum(tau) + (1 - K) * s0 @ P0.T).T).T
    u = np.zeros_like(Y); u[:, model.sl[g]] = ug
    for k in groups:
        Gk, sigk = keep[k]
        u[:, model.sl[k]] = sigk[:, dg:] + (ug - sigk[:, :dg]) @ Gk.T
        ua = u[:, model.idx(k)]
        for leaf in model.children(k):
            Kkj, s = keep[leaf]
            na = ua.shape[1]
            u[:, model.sl[leaf]] = s[:, na:] + (ua - s[:, :na]) @ Kkj.T
    return u


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ------------------------------------------------------------------- checks
print("1) explicit three-level rule vs exact composed score")
m3 = NestedModel(dims=[2, 2, 1], branching=[3, 4])
for lam in (1e-3, 1e-2, 0.3, 1.0, 10.0):
    Y = m3.mu0 + rng.normal(size=(8, m3.D)) * (1 + lam)
    ue = exact_score(m3, Y, lam)
    print(f"   lam={lam:<6} explicit 3-level {rel(three_level_rule(m3, Y, lam), ue):.1e}"
          f"   level recursion {rel(nlevel_rule(m3, Y, lam), ue):.1e}"
          f"   drop p(g,h_k)^(1-n_k) {rel(nlevel_rule(m3, Y, lam, True), ue):.1e}")

print("2) level-by-level N-level rule vs exact composed score")
for dims, br in (([2, 1], [6]), ([2, 2, 1], [3, 3]), ([1, 2, 1, 1], [2, 3, 2]),
                 ([2, 1, 1, 1, 2], [2, 2, 2, 2])):
    mN = NestedModel(dims=dims, branching=br)
    err = 0.0
    for lam in (1e-3, 0.1, 1.0, 10.0):
        Y = mN.mu0 + rng.normal(size=(4, mN.D)) * (1 + lam)
        err = max(err, rel(nlevel_rule(mN, Y, lam), exact_score(mN, Y, lam)))
    print(f"   N={len(dims)} branching={br} dim={mN.D:3d} obs={len(mN.obs):2d}: max rel err {err:.1e}")


print("3) joint sampling with the composed three-level score (probability-flow ODE, Heun)")


def pf_ode(model, rule, n_samples=20000, lam_max=40.0, lam_min=1e-3, steps=400):
    """Start from the exact diffused tall posterior at lam_max, so that any error in the
    final moments is due to the score (plus ODE discretization), not to the initial law."""
    lams = np.geomspace(lam_max, lam_min, steps + 1)
    m, C = model.gauss(np.arange(model.D), list(model.obs))
    L = np.linalg.cholesky(C + lam_max**2 * np.eye(model.D))
    Y = m + rng.normal(size=(n_samples, model.D)) @ L.T
    for l0, l1 in zip(lams[:-1], lams[1:]):          # dy/dlam = -lam * u(y, lam)
        d0 = -l0 * rule(model, Y, l0)
        Yp = Y + (l1 - l0) * d0
        d1 = -l1 * rule(model, Yp, l1)
        Y = Y + (l1 - l0) * 0.5 * (d0 + d1)
    return Y


m_post, C_post = m3.gauss(np.arange(m3.D), list(m3.obs))
sd = np.sqrt(np.diag(C_post))
g_idx, h_idx, l_idx = m3.sl[()], m3.sl[(0,)], m3.sl[(0, 0)]
for name, rule in (("composed 3-level score", three_level_rule),
                   ("group priors dropped", lambda md, Y, l: nlevel_rule(md, Y, l, True))):
    Y = pf_ode(m3, rule)
    z = np.abs(Y.mean(0) - m_post) / sd
    Cs = np.cov(Y.T)
    corr_s = Cs / np.sqrt(np.outer(np.diag(Cs), np.diag(Cs)))
    corr_e = C_post / np.outer(sd, sd)
    print(f"   {name:24s}: max |mean err|/sd = {z.max():.3f}, "
          f"sd ratio in [{(np.sqrt(np.diag(Cs)) / sd).min():.3f}, {(np.sqrt(np.diag(Cs)) / sd).max():.3f}], "
          f"max |corr err| = {np.abs(corr_s - corr_e).max():.3f}, "
          f"corr(g1,h0) {corr_s[g_idx[0], h_idx[0]]:+.3f} (exact {corr_e[g_idx[0], h_idx[0]]:+.3f}), "
          f"corr(h0,l00) {corr_s[h_idx[0], l_idx[0]]:+.3f} (exact {corr_e[h_idx[0], l_idx[0]]:+.3f})")
print(f"   Monte-Carlo noise floor for 20000 draws: mean err/sd ~ {1/np.sqrt(20000):.3f}, corr ~ {1/np.sqrt(20000):.3f}")
