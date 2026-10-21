"""Numerical check of the region/counting-number composition rule on linear-Gaussian DAGs.

For a linear-Gaussian hierarchy every factor is exactly Gaussian, so the derived rule must
reproduce the exact diffused tall-data score at every noise level lambda. Checks:
  * dense solve of Lambda u = r          vs exact score
  * recursive tree (upward/downward) rule vs exact score   (tree-shaped DAGs only)
  * information-form identity and the "drop intermediate prior corrections" ablation
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

import itertools
import numpy as np

rng = np.random.default_rng(0)


class Model:
    def __init__(self, dims, parents, obs):
        """dims: {node: d}; parents: {node: [parents]} (topologically ordered dict);
        obs: list of (name, [latent parents], dx)."""
        self.nodes = list(dims)
        self.dims = dims
        self.parents = parents
        off, self.sl = 0, {}
        for v in self.nodes:
            self.sl[v] = slice(off, off + dims[v]); off += dims[v]
        self.D = off
        # linear-Gaussian CPDs theta_v = sum_p W_vp theta_p + b_v + N(0, Q_v)
        Wbig = np.zeros((self.D, self.D)); b = np.zeros(self.D); Qbig = np.zeros((self.D, self.D))
        for v in self.nodes:
            for p in parents[v]:
                Wbig[self.sl[v], self.sl[p]] = rng.normal(size=(dims[v], dims[p])) * 0.8
            b[self.sl[v]] = rng.normal(size=dims[v])
            A = rng.normal(size=(dims[v], dims[v]))
            Qbig[self.sl[v], self.sl[v]] = A @ A.T / dims[v] + 0.3 * np.eye(dims[v])
        T = np.linalg.inv(np.eye(self.D) - Wbig)
        self.mu0 = T @ b
        self.C0 = T @ Qbig @ T.T
        # observations x_o = H_o theta_{pa(o)} + N(0, R_o), rows of a big likelihood
        self.obs = []
        for name, pa, dx in obs:
            idx = np.concatenate([np.arange(self.D)[self.sl[p]] for p in pa])
            H = np.zeros((dx, self.D)); H[:, idx] = rng.normal(size=(dx, len(idx)))
            R = 0.5 * np.eye(dx)
            theta = rng.multivariate_normal(self.mu0, self.C0)
            x = H @ theta + rng.multivariate_normal(np.zeros(dx), R)
            self.obs.append(dict(name=name, pa=pa, H=H, R=R, x=x, region=self.ancestral(pa)))

    def ancestral(self, vs):
        out, stack = set(), list(vs)
        while stack:
            v = stack.pop()
            if v not in out:
                out.add(v); stack.extend(self.parents[v])
        return frozenset(out)

    def idx(self, region):
        return np.concatenate([np.arange(self.D)[self.sl[v]] for v in self.nodes if v in region])

    def posterior(self, region, obs_list):
        """Gaussian p(theta_R | x_obs) under the true prior marginal on R (R ancestral)."""
        I = self.idx(region)
        m, C = self.mu0[I], self.C0[np.ix_(I, I)]
        P, h = np.linalg.inv(C), np.linalg.solve(C, m)
        for o in obs_list:
            H = o["H"][:, I]
            assert np.allclose(np.delete(o["H"], I, axis=1), 0)
            P = P + H.T @ np.linalg.solve(o["R"], H)
            h = h + H.T @ np.linalg.solve(o["R"], o["x"])
        C = np.linalg.inv(P)
        return C @ h, C


def diffused(m, C, y, lam):
    s = -np.linalg.solve(C + lam**2 * np.eye(len(m)), y - m)
    P = np.linalg.inv(C) + lam**-2 * np.eye(len(m))
    return s, P


def factors(model, drop_internal_priors=False):
    """Observation factors (weight 1) + prior region factors (weight kappa) via Moebius."""
    fam = {o["region"] for o in model.obs}
    changed = True
    while changed:  # close under intersection
        changed = False
        for A, B in itertools.combinations(list(fam), 2):
            if A & B and (A & B) not in fam:
                fam.add(A & B); changed = True
    fam = sorted(fam, key=len, reverse=True)
    c = {}
    for R in fam:
        c[R] = 1 - sum(c[S] for S in fam if S > R)
    m = {R: sum(o["region"] == R for o in model.obs) for R in fam}
    kappa = {R: c[R] - m[R] for R in fam}
    # every node covered exactly once
    for v in model.nodes:
        tot = sum(1 for o in model.obs if v in o["region"]) + sum(k for R, k in kappa.items() if v in R)
        assert tot == 1, (v, tot)
    out = [(1.0, o["region"], ("obs", o)) for o in model.obs]
    root_size = min(len(R) for R in fam)
    for R, k in kappa.items():
        if k != 0 and not (drop_internal_priors and len(R) > root_size):
            out.append((float(k), R, ("prior", None)))
    return out, kappa


def factor_score(model, f, y, lam):
    w, R, (kind, o) = f
    I = model.idx(R)
    m, C = model.posterior(R, [o] if kind == "obs" else [])
    s, P = diffused(m, C, y[I], lam)
    return w, I, s, P


def dense_rule(model, fs, y, lam):
    Lam = np.zeros((model.D, model.D)); r = np.zeros(model.D)
    for f in fs:
        w, I, s, P = factor_score(model, f, y, lam)
        Lam[np.ix_(I, I)] += w * P
        r[I] += w * P @ s
    return np.linalg.solve(Lam, r), Lam


def tree_rule(model, fs, y, lam):
    """Upward Schur-message pass + downward back-substitution on a tree (each node <=1 parent)."""
    assert all(len(model.parents[v]) <= 1 for v in model.nodes)
    path = {v: model.ancestral([v]) for v in model.nodes}
    children = {v: [w for w in model.nodes if model.parents[w] == [v]] for v in model.nodes}
    # each factor lives on path(v) for exactly one v: the deepest node of its region
    at = {v: [] for v in model.nodes}
    for f in fs:
        v = [u for u in f[1] if path[u] == f[1]][0]
        at[v].append(factor_score(model, f, y, lam))
    msg, back = {}, {}

    def up(v):
        I = model.idx(path[v]); pos = {g: k for k, g in enumerate(I)}
        Q = np.zeros((len(I), len(I))); q = np.zeros(len(I))
        for w_, J, s, P in at[v]:
            Q += w_ * P; q += w_ * P @ s          # factors on path(v): scopes coincide
        for w in children[v]:
            up(w)
            S, sig, J = msg[w]
            k = [pos[g] for g in J]
            Q[np.ix_(k, k)] += S; q[k] += S @ sig
        sig = np.linalg.solve(Q, q)               # merged composed score on path(v)
        own = np.isin(I, np.arange(model.D)[model.sl[v]])
        if model.parents[v]:
            A, B, C = Q[np.ix_(~own, ~own)], Q[np.ix_(~own, own)], Q[np.ix_(own, own)]
            Rv = -np.linalg.solve(C, B.T)
            msg[v] = (A + B @ Rv, sig[~own], I[~own])
            back[v] = (sig[own], Rv, sig[~own], I[~own])
        else:
            back[v] = (sig[own], None, None, None)

    root = [v for v in model.nodes if not model.parents[v]][0]
    up(root)
    u = np.zeros(model.D)
    for v in model.nodes:                          # topological order = downward pass
        sv, Rv, sA, IA = back[v]
        u[model.sl[v]] = sv if Rv is None else sv + Rv @ (u[IA] - sA)
    return u


def exact(model, y, lam):
    m, C = model.posterior(frozenset(model.nodes), model.obs)
    return diffused(m, C, y, lam)[0]


def run(name, model, tree):
    fs, kappa = factors(model)
    fs_bad, _ = factors(model, drop_internal_priors=True)
    print(f"\n== {name}: dim {model.D}, {len(model.obs)} observations")
    for R, k in sorted(kappa.items(), key=lambda t: len(t[0])):
        if k:
            print(f"   prior factor on {sorted(R)}: kappa = {k:+d}")
    for lam in (1e-2, 0.3, 1.0, 5.0):
        y = model.mu0 + rng.normal(size=model.D) * (1 + lam)
        ue = exact(model, y, lam)
        ud, Lam = dense_rule(model, fs, y, lam)
        line = f"   lam={lam:<5} |dense-exact|/|exact| = {np.linalg.norm(ud-ue)/np.linalg.norm(ue):.1e}"
        if tree:
            ut = tree_rule(model, fs, y, lam)
            line += f"  |tree-exact| = {np.linalg.norm(ut-ue)/np.linalg.norm(ue):.1e}"
        ub, _ = dense_rule(model, fs_bad, y, lam)
        line += f"  | drop mid-level priors: {np.linalg.norm(ub-ue)/np.linalg.norm(ue):.1e}"
        line += f"  min eig Lam = {np.linalg.eigvalsh(Lam).min():.2f}"
        print(line)


# 1. three levels: g -> h_k -> l_kj -> x_kj, ragged groups, one obs on h_1, two trials on l_21
dims = {"g": 2}; par = {"g": []}; obs = []
for k, nk in enumerate([2, 3, 4]):
    dims[f"h{k}"] = 1; par[f"h{k}"] = ["g"]
    for j in range(nk):
        dims[f"l{k}{j}"] = 2; par[f"l{k}{j}"] = [f"h{k}"]
        obs.append((f"x{k}{j}", [f"l{k}{j}"], 2))
obs.append(("xh0", ["h0"], 1))
obs.append(("x21b", ["l21"], 2))
run("3-level tree", Model(dims, par, obs), tree=True)

# 2. random 5-level tree
dims = {"n0": 2}; par = {"n0": []}; obs = []; frontier = ["n0"]; c = 1
for level in range(4):
    new = []
    for p in frontier:
        for _ in range(rng.integers(1, 4)):
            v = f"n{c}"; c += 1; dims[v] = int(rng.integers(1, 3)); par[v] = [p]; new.append(v)
    frontier = new
for v in frontier:
    obs.append((f"x_{v}", [v], 2))
run("5-level random tree", Model(dims, par, obs), tree=True)

# 3. crossed random effects: x_ij | g, a_i, b_j
dims = {"g": 2}; par = {"g": []}; obs = []
for i in range(3):
    dims[f"a{i}"] = 1; par[f"a{i}"] = ["g"]
for j in range(4):
    dims[f"b{j}"] = 1; par[f"b{j}"] = ["g"]
for i in range(3):
    for j in range(4):
        obs.append((f"x{i}{j}", [f"a{i}", f"b{j}"], 1))
run("crossed design (3 x 4)", Model(dims, par, obs), tree=False)

# 4. DAG with a shared child: g -> (h1, h2) -> z (two parents) -> x; plus x on h1
dims = {"g": 1, "h1": 1, "h2": 1, "z1": 1, "z2": 1}
par = {"g": [], "h1": ["g"], "h2": ["g"], "z1": ["h1", "h2"], "z2": ["h2"]}
obs = [("a", ["z1"], 1), ("b", ["z1"], 1), ("c", ["z2"], 1), ("d", ["h1"], 1)]
run("non-tree DAG", Model(dims, par, obs), tree=False)
