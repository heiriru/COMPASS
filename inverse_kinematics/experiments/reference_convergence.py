"""Is the ABC reference's *branch mass* converged at the draw count we grade on?

`hierarchical.reference_global` carries a warning in its own source: the raw
estimate's residual L1 asymmetry falls only as 1/sqrt(draws) -- 0.78 at 30,000
draws, 0.11 at 480,000 -- which is "enough to swamp the differences between
composition rules if left in". The `symmetrize=True` variance reduction is only
licensed for x1 = 0 exactly, and this dataset is |x1| < 0.15, so it cannot be
used here and the raw asymmetry is what every branch-mass comparison rests on.

Every reference in this study was built at 40,000-60,000 draws. So: does the
branch mass move as the draw count rises? If it does, then grading branch mass
against it is grading against Monte-Carlo noise, and the conclusion that a
method "puts too little mass on the positive branch" may belong to the referee.
"""
import os
CPU=3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU]))
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v]=str(CPU)
import sys
import numpy as np
sys.path.insert(0,'/export/home/rheinric/COMPASS/inverse_kinematics')
sys.path.insert(0,'/export/home/rheinric/COMPASS/inverse_kinematics/experiments')
import hierarchical as H, simulator as sim
from bimodality_study import selected_dataset

grid = np.linspace(-4*float(H.GLOBAL_PRIOR_STD[0]), 4*float(H.GLOBAL_PRIOR_STD[0]), 801)

def summary(d):
    d = d/np.trapezoid(d, grid)
    i = np.arange(1, len(grid)-1)
    hit = i[(d[i] >= d[i-1]) & (d[i] >= d[i+1]) & (d[i] > 0.05*d.max())]
    mass = float(np.trapezoid(d[grid>0], grid[grid>0]))
    # L1 asymmetry against the mirror: a theorem-free diagnostic of MC noise,
    # since the selection |x1| < 0.15 makes the reflection nearly exact.
    asym = float(np.trapezoid(np.abs(d - d[::-1]), grid))
    return [float(grid[j]) for j in hit], mass, asym

print(f"{'n':>4}{'draws':>9}{'seed':>6}{'mass>0':>9}{'L1 asym':>10}{'modes':>26}")
for n in (8, 32):
    rng = sim.generator("bimodal-dataset", 99)
    x = np.asarray(selected_dataset(n, 0.6, 0.15, rng), dtype=np.float64)
    for draws in (40_000, 160_000, 480_000):
        for seed in (11, 12):
            _, d = H.reference_global(x, grid=grid, draws=draws, seed=seed)
            m, mass, asym = summary(d)
            print(f"{n:4d}{draws:9d}{seed:6d}{mass:9.3f}{asym:10.3f}"
                  f"{', '.join(f'{v:+.2f}' for v in m):>26}", flush=True)
