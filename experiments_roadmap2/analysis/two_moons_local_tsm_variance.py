"""Roadmap 2.9: is target score matching worth it for the two-moons LOCALS?
The local TSM target is T_l = grad_theta [log TN(theta; mu, sigma) + log p(x | theta)]; its
likelihood part comes from a crescent of radial sd 0.01. The inverse-variance blend
w = 1 / (1 + lambda^2 Var T) only moves off DSM where Var T is below the DSM target's 1/lambda^2.
Measured in network units (theta / local_scale) on prior-predictive draws."""
import os, sys
CPU = sorted(os.sched_getaffinity(0))[:3]; os.sched_setaffinity(0, CPU)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "3"
sys.path.insert(0, "/export/home/rheinric/COMPASS/benchmarks/hierarchical_two_moons")
import numpy as np, torch
import task as T
from reference import _log_diff_ndtr
g, th, x = T.simulate(1, 100_000, seed=3)
th, x = th.reshape(-1, 2), x.reshape(-1, 2)
scale = T.normalization()["local_scale"]
theta = torch.tensor(th, dtype=torch.float64, requires_grad=True)
X = torch.tensor(x, dtype=torch.float64)
c, s = np.cos(T.ROTATION), np.sin(T.ROTATION)
z0 = c * theta[:, 0] - s * theta[:, 1]; z1 = s * theta[:, 0] + c * theta[:, 1]
u = X[:, 0] + z0.abs() - T.BASE_OFFSET; v = X[:, 1] - z1
r = torch.sqrt(u ** 2 + v ** 2)
loglik = -0.5 * ((r - T.R_LOC) / T.R_SCALE) ** 2 - torch.log(r)
(glik,) = torch.autograd.grad(loglik.sum(), theta)
G = torch.tensor(g, dtype=torch.float64)
mu, sigma = G[:, :2], G[:, 2:]
gprior = -(theta.detach() - mu) / sigma ** 2
for name, part in (("likelihood part", glik), ("prior part", gprior), ("total", glik + gprior)):
    net = part * scale                      # d/d(theta/scale) = scale * d/dtheta
    print(f"{name:16s} per-coordinate var (network units): {net.var(0).numpy().round(1)}  "
          f"median |T| {net.abs().median():.1f}")
V = float(((glik + gprior) * scale).var(0).mean())
lam_star = 1 / np.sqrt(V)
print(f"blend weight w = 1/(1 + lambda^2 V) is 1/2 at lambda = {lam_star:.4f}; trained floor "
      f"lambda(t=1e-3) = 0.0317 (sigma 25: 0.032) -> w there = {1 / (1 + 0.0317 ** 2 * V):.3f}")
