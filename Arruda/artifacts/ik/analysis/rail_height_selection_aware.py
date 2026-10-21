"""Selection-aware posterior for the rail-height dataset, next to the standard-model reference.

The rows were kept only if |x1 - h| < TOL, and x1 - h = sum of sines depends on (a1, b2, b3) only.
The selection-aware likelihood is therefore the ABC kernel averaged over local draws that pass the
selection at that a1, divided by the number that pass (the 1 / P(selected | a1) normalisation):
    p(x | a1, sel) = E[K(x - f(a1, l)) | |S(a1, b2, b3)| < TOL].
Same kernel (h = 0.05) and angle_1 prior as the reference; common random numbers across the grid.
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

TOL, BLOCK, BLOCKS = 0.05, 1_000_000, 8
NS = (1, 2, 4, 8, 16, 32, 64, 128)
OUT = HERE.parent / "rail_height"
x = torch.as_tensor(np.load(OUT / "data_a0.6_tol0.05.npy"), device="cuda", dtype=torch.float32)
grid = np.linspace(-2.0, 2.0, 801)
log_sum = torch.full((len(grid), len(x)), -np.inf, dtype=torch.float64, device="cuda")
kept = torch.zeros(len(grid), dtype=torch.float64, device="cuda")
for b in range(BLOCKS):
    rng = np.random.default_rng([4100, b])
    locals_ = torch.as_tensor(rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(BLOCK, 3)), device="cuda", dtype=torch.float32)
    for i, a in enumerate(grid):
        pred = S.forward(torch.tensor([a], device="cuda", dtype=torch.float32), locals_)[0]
        keep = (pred[:, 0] - locals_[:, 0]).abs() < TOL
        kept[i] += keep.sum()
        sq = ((pred[keep][:, None, :] - x[None]) ** 2).sum(-1)
        log_sum[i] = torch.logaddexp(log_sum[i], torch.logsumexp(-sq / (2 * S.BANDWIDTH ** 2), dim=0).double())
    print(f"block {b + 1}/{BLOCKS}", flush=True)
loglik = (log_sum - kept.log()[:, None]).cpu().numpy()
print(f"accepted draws per grid point: min {int(kept.min())}, at a1=0.6 {int(kept[np.argmin(abs(grid - 0.6))])}", flush=True)
ref = np.load(OUT / "reference_a0.6_tol0.05.npz")
out = {}
for n in NS:
    logp = loglik[:, :n].sum(1) - 0.5 * (grid / H.GLOBAL_PRIOR_STD[0]) ** 2
    p = np.exp(logp - logp.max()); p /= np.trapezoid(p, grid); out[f"n{n}"] = p
    s = S.summary(grid, logp - logp.max())
    r = ref[f"n{n}"]; rg = ref["grid"]
    print(f"n={n:3d}  selection-aware: peaks {np.round(s['peaks'], 2).tolist()} P(a1>0) {np.trapezoid(np.where(grid > 0, p, 0), grid):.2f} "
          f"E|a1| {np.trapezoid(abs(grid) * p, grid):.2f}   | standard reference: E|a1| {np.trapezoid(abs(rg) * r, rg):.2f}", flush=True)
np.savez(OUT / "selection_aware_a0.6_tol0.05.npz", grid=grid, **out)
