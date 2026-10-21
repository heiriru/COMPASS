#!/usr/bin/env python3
"""The tilt's misspecification on inverse kinematics, measured on real factors.

Builds the *per-observation* ABC references (no network, no stand-in) and asks:
how well can a degree-d polynomial represent each exact log-likelihood-ratio
``log p(a|x_j) - log p(a)``? The residual of the *best possible* degree-d fit is
a floor the tilt cannot get under no matter how many draws it is given, and the
composition multiplies it by n.

Contrast with the Gaussian-mixtures benchmark, where the same residual is
identically zero at d=2 because the true likelihood *is* log-quadratic. That is
the difference between the two benchmarks in one number, and it is why the tilt
is at the Monte-Carlo floor on one and not on the other.

Result at n = 32 (mean over eight observations, rms nats):

    d        2      4      6      8     10     12     16
    per factor   0.047  0.024  0.022  0.020  0.020  0.015  0.011
    x n          1.51   0.78   0.71   0.65   0.63   0.49   0.35

against a composed posterior whose own log-density spans only 3.0 nats over its
5%-of-peak support. So at the degree BIC actually selects (8-10) the irreducible
error is about a fifth of the entire shape information, and going to d = 16 buys
a factor of two for four times the parameters -- which is why widening the
candidate set did not move the measured W1.
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

grid = np.linspace(-4*float(H.GLOBAL_PRIOR_STD[0]), 4*float(H.GLOBAL_PRIOR_STD[0]), 1601)
log_prior = -0.5*(grid/float(H.GLOBAL_PRIOR_STD[0]))**2
log_prior -= np.log(np.trapezoid(np.exp(log_prior), grid))

n = 32
rng = sim.generator("bimodal-dataset", 99)
x = np.asarray(selected_dataset(n, 0.6, 0.15, rng), dtype=np.float64)
_, tall = H.reference_global(x, grid=grid, draws=40000)
tall = tall/np.trapezoid(tall, grid)
print("tall reference built", flush=True)

count = 8   # eight observations is enough to see the shape and the spread
factors = []
for j in range(count):
    _, f = H.reference_global(x[j:j+1], grid=grid, draws=40000)
    factors.append(f/np.trapezoid(f, grid))
    print(f"factor {j+1}/{count}", flush=True)
factors = np.asarray(factors)

def residual(target, weight, degree):
    keep = weight > weight.max()*1e-6
    u = (grid[keep]-grid[keep].mean())/max(grid[keep].std(), 1e-9)
    design = np.vander(u, degree+1, increasing=True)
    root = np.sqrt(weight[keep])[:,None]
    sol,*_ = np.linalg.lstsq(design*root, target[keep]*root[:,0], rcond=None)
    r = target[keep] - design@sol
    r = r - np.average(r, weights=weight[keep])
    return float(np.sqrt(np.average(r**2, weights=weight[keep])))

degrees = (2,4,6,8,10,12,16)
print("\nrms residual of the BEST degree-d fit to each exact log L_j, in nats")
print("weighted by the tall reference (the only region the composition uses)")
print(f"{'obs':>4}  " + "  ".join(f"d={d:<3}" for d in degrees))
table = []
for j, f in enumerate(factors):
    ratio = np.log(np.maximum(f,1e-300)) - log_prior
    row = [residual(ratio, tall/tall.max(), d) for d in degrees]
    table.append(row)
    print(f"{j:4d}  " + "  ".join(f"{v:5.3f}" for v in row))
table = np.asarray(table)
print(f"{'mean':>4}  " + "  ".join(f"{v:5.3f}" for v in table.mean(axis=0)))
print(f"\nmultiplied by n={n} (composed log-density error, nats):")
print(f"{'':4}  " + "  ".join(f"{n*v:5.2f}" for v in table.mean(axis=0)))
print("\nfor scale: the composed posterior's own log-density varies by "
      f"{np.log(tall.max()/np.maximum(tall,1e-300))[tall>tall.max()*0.05].max():.1f} nats "
      "across its 5%-of-peak support")
