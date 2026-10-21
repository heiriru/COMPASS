"""Unbiasedness of the TSM global target: E[T(g0, l0) | g_lam, x] == grad log p_lam(g_lam | x).

For one group x_i and a noisy point g_lam: g0 ~ p(g0 | g_lam, x_i) by importance weights
(proposal g_lam + lam * eps, weights p(x_i | g0) p(g0)), l0 ~ p(l | g0, x_i) exactly
(Reference.sample_locals), T = task.global_score_target. The weighted mean of T must match
the Tweedie score E_w[eps] / lam computed with the same weights, up to Monte-Carlo error.
"""
import os, sys
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:3]))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "3"
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, torch
import task as T
from reference import Reference

DEV = "cuda" if torch.cuda.is_available() else "cpu"
M = 8192
d = np.load(HERE / "artifacts" / "reference_smax0.6" / "N10_0.npz", allow_pickle=True)
scale = T.normalization()["local_scale"]
gen = torch.Generator(device=DEV).manual_seed(0)
for lam in (0.05, 0.5):
    rel = []
    for i in range(4):                                   # groups
        ref = Reference(d["x"][i:i + 1], device=DEV)
        for p in range(4):                               # points
            z = torch.as_tensor(T.globals_to_z(d["ref_g"][p]), device=DEV)
            z = z + lam * torch.randn(4, generator=gen, device=DEV, dtype=torch.float64)
            eps = torch.randn(M, 4, generator=gen, device=DEV, dtype=torch.float64)
            eps = torch.cat([eps, -eps])
            g0 = z + lam * eps
            logw = ref.log_likelihood(g0) + ref.log_prior(g0)
            w = torch.softmax(logw, 0)
            tweedie = (w[:, None] * eps).sum(0) / lam
            l0 = torch.as_tensor(ref.sample_locals(g0, seed=p), device=DEV)[:, 0]   # (2M, 2)
            nodes = torch.cat([g0, l0 / scale, torch.zeros(len(g0), 2, device=DEV, dtype=g0.dtype)], 1)
            tsm = (w[:, None] * T.global_score_target(nodes)).sum(0)
            lik = tweedie + z / (1 + lam ** 2)
            rel.append(float((tsm - tweedie).norm() / lik.norm()))
            if p == 0 and i < 2:
                print(f"lam={lam} group {i}: tweedie {np.round(tweedie.cpu().numpy(), 3)}  "
                      f"E[T] {np.round(tsm.cpu().numpy(), 3)}")
    print(f"lam={lam}: median |E[T] - tweedie| / |likelihood part| = {np.median(rel):.3f}")
