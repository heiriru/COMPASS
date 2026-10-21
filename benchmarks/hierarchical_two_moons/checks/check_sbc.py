"""SBC of the reference sampler: ranks of the true globals/locals among reference draws."""
import os, sys, time
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:3]))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import task as T
from reference import Reference
N, D, S = int(sys.argv[1]), int(sys.argv[2]), 199
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
g_true, l_true, x = T.simulate(N, D, seed=999)
ranks_g, ranks_l = [], []
t0 = time.time()
for d in range(D):
    ref = Reference(x[d], device=device)
    g, l, diag = ref.sample(S, seed=d, particles=1024)
    ranks_g.append((g < g_true[d]).sum(0))
    ranks_l.append((l < l_true[d]).sum(0).reshape(-1))
ranks_g, ranks_l = np.array(ranks_g), np.concatenate(ranks_l)
print(f"N={N} D={D} {time.time()-t0:.0f}s, last diag {diag}")
def chi2(r, bins=10):
    h = np.histogram(r, bins=bins, range=(0, S + 1))[0]
    e = len(r) / bins
    return ((h - e) ** 2 / e).sum(), h
for k, name in enumerate(T.GLOBAL_NAMES):
    c, h = chi2(ranks_g[:, k]); print(f"  {name:8s} chi2(9)={c:5.1f}  {h}")
c, h = chi2(ranks_l); print(f"  locals   chi2(9)={c:5.1f}  {h}  (n={len(ranks_l)})")
print("  95% chi2(9) critical value 16.9")
