"""Quadrature p(x_i | g) against a Monte-Carlo density of hsbibm's own simulator."""
import os, sys
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:3]))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import task as T
from reference import Reference

torch.manual_seed(0)
for trial in range(3):
    g = T.simulate(1, 1, seed=50 + trial)[0][0]
    # simulate many x from this fixed g via hsbibm (n_l=1)
    t = T.sbibm_task(1)
    th = torch.tensor(g)[None].repeat(2_000_000, 1)
    loc = t.prior_dist  # sample locals given g via the hierarchical prior's local fn
    from sbibm.tasks.distributions import TruncatedNormal
    tn = TruncatedNormal(th[:, :2], th[:, 2:], -1.0, 1.0)
    theta_l = tn.sample()
    x = t.get_simulator()(torch.cat([th, theta_l], 1)).double().numpy()
    # density at a few observed points by a small box count
    pts = x[-6:]; x = x[:-6]
    h = 0.03
    mc = [(np.abs(x - p).max(1) < h / 2).mean() / h**2 for p in pts]
    z = torch.tensor(T.globals_to_z(g))[None]
    quad = [float(torch.exp(Reference(p[None]).log_likelihood(z))) for p in pts]
    print("g", np.round(g, 3))
    for a, b in zip(mc, quad):
        print(f"   MC {a:8.4f}   quadrature {b:8.4f}   ratio {b/max(a,1e-12):.3f}")
