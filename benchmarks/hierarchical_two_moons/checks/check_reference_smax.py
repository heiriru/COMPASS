"""Reference self-consistency in the max(sigma) <= 0.6 regime: stored reference vs an
independent, 4x larger / longer SMC-MCMC run (both mirror-symmetrized). C2ST ~ 0.5 = agree."""
import os, sys
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:3]))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "3"
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import task as T, metrics as M
from reference import Reference

T.SIGMA_MAX = 0.6
for n in map(int, sys.argv[1:] or ["25", "100"]):
    for i in range(10):
        d = np.load(HERE / f"artifacts/reference_smax0.6/N{n}_{i}.npz", allow_pickle=True)
        g, l, _ = Reference(d["x"], device="cuda").sample(2500, seed=500 + i, particles=4096,
                                                          final_moves=100, thin=20)
        flip = np.random.default_rng(i).random(len(g)) < 0.5
        g[flip], l[flip] = T.mirror(g[flip], l[flip])
        print(f"N={n} [{i}] global C2ST {M.c2st(g, d['ref_g'], i):.3f}  "
              f"local_1 C2ST {M.c2st(l[:, 0], d['ref_l'][:, 0], i):.3f}  "
              f"sd ratio {np.round(g.std(0) / d['ref_g'].std(0), 2)}", flush=True)
