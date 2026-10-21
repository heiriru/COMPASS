"""The grading noise floor: W1 between two ABC references of the same data.

Every inverse-kinematics number in this study is a W1 against one ABC
reference. That reference is itself a Monte-Carlo estimate, and its own error is
amplified by n exactly the way a composition rule's is -- the shared log-posterior
is a sum of n estimated log-likelihood curves either way. So the honest question
is: how far apart are two references of the *same data* that differ only in
their Monte-Carlo seed? Differences between methods smaller than that are not
measurable with this referee.
"""
import os
CPU = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU]))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = str(CPU)
import sys
import numpy as np
sys.path.insert(0, '/export/home/rheinric/COMPASS/inverse_kinematics')
sys.path.insert(0, '/export/home/rheinric/COMPASS/inverse_kinematics/experiments')
import hierarchical as H, simulator as sim
from bimodality_study import selected_dataset

grid = np.linspace(-4 * float(H.GLOBAL_PRIOR_STD[0]),
                   4 * float(H.GLOBAL_PRIOR_STD[0]), 801)


def w1_over_sigma(a, b):
    """W1 between two densities on ``grid``, in the sd of the second."""
    step = np.diff(grid)
    ca = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * step)])
    cb = np.concatenate([[0.0], np.cumsum(0.5 * (b[1:] + b[:-1]) * step)])
    ca /= ca[-1]
    cb /= cb[-1]
    bn = b / np.trapezoid(b, grid)
    mean = float(np.trapezoid(grid * bn, grid))
    sd = float(np.sqrt(np.trapezoid((grid - mean) ** 2 * bn, grid)))
    return float(np.trapezoid(np.abs(ca - cb), grid)) / sd


def mass(d):
    d = d / np.trapezoid(d, grid)
    return float(np.trapezoid(d[grid > 0], grid[grid > 0]))


print(f"{'n':>4}{'draws':>9}{'W1(ref,ref)':>13}{'mass A':>9}{'mass B':>9}"
      f"{'|dmass|':>9}")
for n in (8, 32, 64):
    rng = sim.generator("bimodal-dataset", 99)
    x = np.asarray(selected_dataset(n, 0.6, 0.15, rng), dtype=np.float64)
    for draws in (40_000, 200_000):
        _, a = H.reference_global(x, grid=grid, draws=draws, seed=11)
        _, b = H.reference_global(x, grid=grid, draws=draws, seed=12)
        print(f"{n:4d}{draws:9d}{w1_over_sigma(a, b):13.3f}"
              f"{mass(a):9.3f}{mass(b):9.3f}{abs(mass(a) - mass(b)):9.3f}",
              flush=True)
