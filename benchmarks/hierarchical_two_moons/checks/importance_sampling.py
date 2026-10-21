"""Can importance sampling (Dax et al.) repair the global posterior? Upper-bound-style test.

For each method/dataset: proposal q = density fitted to the method's 500 global draws
(Gaussian KDE in probit z, Scott bandwidth) + defensive components (10% broad Gaussian with
4x the draws' covariance, 5% prior N(0,I)); optionally mirror-symmetrized (KDE over the draws
and their exact mirror images). M proposal draws are weighted with the exact collapsed target
p(x|g) p(g) (reference quadrature); we report ESS/M and the quality of 500 resampled draws
against the reference (global C2ST, mode balance, mirror-invariant W1).
Note: q here is a density *fitted* to the samples, not the sampler's own density; that is
the "sample, fit, reweight" variant, which needs no probability-flow ODE.
"""
import os, sys, json, math, time
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:3]))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "3"
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, torch
import task as T, metrics as M
from reference import Reference

DEV = "cuda"
RES = HERE / "artifacts_smax0.6"
REF = HERE / "artifacts" / "reference_smax0.6"
MPROP = 20_000
METHODS = ["prior", "gauss_hierarchical", "gauss_jacobian_FD", "langevin_fnpe",
           "attn_hier:gauss_jacobian_FD", "attn_full:gauss_hierarchical",
           "attn_full:arruda_schur", "bayesflow_arruda", "bayesflow_arruda_diffprior"]
NS = [int(a) for a in sys.argv[1:]] or [5, 10, 25, 50, 100]
MIRROR_Z = torch.tensor([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float64)


def mvn_logpdf(z, means, chol):
    """log N(z; means_k, L L^T) averaged over k (equal weights): (M,)"""
    d = z.shape[-1]
    diff = z[:, None, :] - means[None]                                  # (M, K, d)
    sol = torch.linalg.solve_triangular(chol, diff.reshape(-1, d).T, upper=False).T.reshape(diff.shape)
    logdet = 2 * torch.log(torch.diagonal(chol)).sum()
    lp = -0.5 * (sol ** 2).sum(-1) - 0.5 * logdet - 0.5 * d * math.log(2 * math.pi)
    return torch.logsumexp(lp, 1) - math.log(means.shape[0])


def proposal(zd, gen):
    """Returns (sample(M), logq(z)) for the defensive KDE on draws zd (K, 4)."""
    K, d = zd.shape
    cov = torch.cov(zd.T) + 1e-6 * torch.eye(d, dtype=zd.dtype, device=zd.device)
    h = K ** (-1.0 / (d + 4))                                            # Scott
    L_k = torch.linalg.cholesky(h ** 2 * cov)
    L_b = torch.linalg.cholesky(4 * cov)
    mean = zd.mean(0, keepdim=True)
    L_p = torch.eye(d, dtype=zd.dtype, device=zd.device)
    zero = torch.zeros(1, d, dtype=zd.dtype, device=zd.device)
    w = torch.tensor([0.85, 0.10, 0.05], dtype=zd.dtype, device=zd.device)

    def logq(z):
        parts = torch.stack([mvn_logpdf(z, zd, L_k), mvn_logpdf(z, mean, L_b), mvn_logpdf(z, zero, L_p)], 1)
        return torch.logsumexp(parts + torch.log(w), 1)

    def sample(m):
        comp = torch.multinomial(w, m, replacement=True, generator=gen)
        eps = torch.randn(m, d, generator=gen, dtype=zd.dtype, device=zd.device)
        idx = torch.randint(0, K, (m,), generator=gen, device=zd.device)
        out = torch.where((comp == 0)[:, None], zd[idx] + eps @ L_k.T,
                          torch.where((comp == 1)[:, None], mean + eps @ L_b.T, eps))
        return out
    return sample, logq


def run(method, n, ds, symmetrize):
    ref = np.load(REF / f"N{n}_{ds}.npz", allow_pickle=True)
    ref_g = ref["ref_g"][500:]
    gen = torch.Generator(device=DEV).manual_seed(1000 * ds + n)
    target = Reference(ref["x"], device=DEV)
    if method == "prior":
        zd = torch.randn(500, 4, generator=gen, device=DEV, dtype=torch.float64)
    else:
        g = np.load(RES / "draws" / f"{method.replace(':', '__')}_N{n}_dataset{ds}.npz")["g"]
        g = g[np.isfinite(g).all(1)]
        zd = torch.as_tensor(np.clip(T.globals_to_z(g), -6, 6), device=DEV)
    if symmetrize:
        zd = torch.cat([zd, zd @ MIRROR_Z.to(DEV).T])
    sample, logq = proposal(zd, gen)
    z = sample(MPROP)
    logw = target.log_likelihood(z) + target.log_prior(z) - logq(z)
    logw = torch.nan_to_num(logw, nan=-math.inf)
    w = torch.softmax(logw, 0)
    ess = float(1.0 / (w ** 2).sum())
    pick = torch.multinomial(w, 500, replacement=True, generator=gen)
    g_is = T.z_to_globals(z[pick].cpu().numpy())
    sym_m, sym_r = T.symmetric_globals(g_is), T.symmetric_globals(ref_g)
    inv = np.mean([M.w1(np.abs(sym_m[:, k]), np.abs(sym_r[:, k])) / np.abs(sym_r[:, k]).std() for k in (0, 2)]
                  + [M.w1(sym_m[:, k], sym_r[:, k]) / sym_r[:, k].std() for k in (1, 3)])
    return {"ess_frac": ess / MPROP, "c2st": M.c2st(g_is, ref_g, ds), "inv_w1": float(inv),
            "mode_err": float(abs(np.mean(sym_m[:, 0] > 0) - np.mean(sym_r[:, 0] > 0)))}


out = {}
for n in NS:
    for method in METHODS:
        for sym in (False, True):
            t0 = time.perf_counter()
            rows = [run(method, n, ds, sym) for ds in range(10)]
            med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
            med["ess_frac_min"] = float(min(r["ess_frac"] for r in rows))
            out[f"{method}|N{n}|{'sym' if sym else 'raw'}"] = med
            print(f"N={n:<3} {method:32s} {'sym' if sym else 'raw'}  ESS/M={med['ess_frac']:.4f} "
                  f"(min {med['ess_frac_min']:.5f})  C2ST={med['c2st']:.3f}  invW1={med['inv_w1']:.3f}  "
                  f"mode_err={med['mode_err']:.3f}  [{time.perf_counter() - t0:.0f}s]", flush=True)
(RES / "importance_sampling.json").write_text(json.dumps(out, indent=1))
