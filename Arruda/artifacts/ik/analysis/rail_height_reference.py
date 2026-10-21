"""High-accuracy ABC reference for a rail-height-selected IK dataset (see rail_height_test.py).

Same estimator as hierarchical.reference_global (Gaussian kernel h = 0.05, local-prior Monte Carlo,
log-mean-exp over draws, sum over observations, angle_1 prior), but with the draws pooled over
blocks so the 80k-draw branch-weight noise (seeds disagreed on which mode was taller) goes away.
Two independent halves are kept to measure what noise is left; the saved reference uses all draws.
Per-observation log-likelihoods are stored, so every nested n comes out of one pass.
"""
import os
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))   # 3-thread CPU cap, before numpy/torch
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "inverse_kinematics")]
import hierarchical as H
import natural_bimodal_search as S

TAG = sys.argv[1] if len(sys.argv) > 1 else "a0.6_tol0.05"
BLOCK = 250_000
BLOCKS_PER_HALF = int(sys.argv[2]) if len(sys.argv) > 2 else 4
BANDWIDTH = float(sys.argv[3]) if len(sys.argv) > 3 else S.BANDWIDTH   # ABC kernel = Gaussian observation noise
SUFFIX = "" if BANDWIDTH == S.BANDWIDTH else f"_h{BANDWIDTH:g}"
NS = (1, 2, 4, 8, 16, 32, 64, 128)
OUT = HERE.parent / "rail_height"
x = torch.as_tensor(np.load(OUT / f"data_{TAG}.npy"), device="cuda", dtype=torch.float32)
grid = np.linspace(-2.0, 2.0, 1_401)


def per_observation_loglik(seed_base):
    """(grid, n) log p_h(x_j | a), log-mean-exp over BLOCK * BLOCKS_PER_HALF local draws."""
    acc = torch.full((len(grid), len(x)), -np.inf, dtype=torch.float64, device="cuda")
    for b in range(BLOCKS_PER_HALF):
        rng = np.random.default_rng([seed_base, b])
        locals_ = torch.as_tensor(rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(BLOCK, 3)), device="cuda",
                                  dtype=torch.float32)
        for i, a in enumerate(grid):
            pred = S.forward(torch.tensor([a], device="cuda", dtype=torch.float32), locals_)[0]      # (D, 2)
            sq = ((pred[:, None, :] - x[None]) ** 2).sum(-1)                                          # (D, n)
            block = torch.logsumexp(-sq / (2 * BANDWIDTH ** 2), dim=0).double()
            acc[i] = torch.logaddexp(acc[i], block)
        print(f"  seed {seed_base} block {b + 1}/{BLOCKS_PER_HALF}", flush=True)
    return (acc - np.log(BLOCK * BLOCKS_PER_HALF)).cpu().numpy()


def posterior(loglik, n):
    logp = loglik[:, :n].sum(1) - 0.5 * (grid / H.GLOBAL_PRIOR_STD[0]) ** 2
    p = np.exp(logp - logp.max())
    return logp - logp.max(), p / np.trapezoid(p, grid)


halves = [per_observation_loglik(s) for s in (4001, 4002)]
pooled = np.logaddexp(*halves) - np.log(2)
np.savez(OUT / f"loglik_{TAG}{SUFFIX}.npz", grid=grid, half_a=halves[0], half_b=halves[1])
reference = {}
for n in NS:
    reference[f"n{n}"] = posterior(pooled, n)[1]
    rows = []
    for name, ll in (("half A", halves[0]), ("half B", halves[1]), ("pooled", pooled)):
        logp, p = posterior(ll, n)
        s = S.summary(grid, logp)
        rows.append(f"{name}: peaks {np.round(s['peaks'], 2).tolist()} P(a1>0) {np.trapezoid(np.where(grid > 0, p, 0), grid):.3f}")
    print(f"n={n:3d}  " + " | ".join(rows), flush=True)
np.savez(OUT / f"reference_{TAG}{SUFFIX}.npz", grid=grid, **reference)
