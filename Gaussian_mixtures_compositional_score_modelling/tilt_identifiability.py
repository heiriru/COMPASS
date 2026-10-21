#!/usr/bin/env python3
"""The two claims behind the antisymmetric-regime answer, checked numerically.

(1) In that regime each observation sits on a prior mode, so its posterior is
almost entirely on one branch and the other branch is unobserved -- which is why
the first moment is a weak handle on beta.
(2) The composed branch log-odds is an exact cancellation of n large per-factor
tilts, so accumulated noise has no signal to hide behind.
"""

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import sys, numpy as np
sys.path.insert(0,'/export/home/rheinric/COMPASS/Gaussian_mixtures_compositional_score_modelling')
sys.path.insert(0,'/export/home/rheinric/COMPASS/inverse_kinematics/experiments')
import analytic_composition as A, model as M, datasets, tilt_fit
mix = M.DEFAULT
n = 100
for regime in ("asymmetric", "antisymmetric"):
    data = datasets.make_dataset(regime, n, np.random.default_rng([2026,datasets.REGIMES.index(regime),n,0]),
                                 mixture=mix, branch_mass=0.75)
    x = np.asarray(data["x"]).ravel()
    # per-observation branch weight, exactly
    w = np.asarray([mix.branch_weights(np.array([v]))[1] for v in x])
    beta_true = x/mix.v
    # sensitivity of the first moment to beta, by exact finite difference on the
    # closed-form tilted mixture
    def mean_of_tilt(beta, gamma):
        vt = 1.0/(1.0/mix.tau_g**2 + gamma)
        h = np.array([-mix.mu_0, mix.mu_0])/mix.tau_g**2 + beta
        logm = 0.5*vt*h**2
        logm -= logm.max()
        p = np.exp(logm); p /= p.sum()
        return float((p*(vt*h)).sum())
    g0 = 1.0/mix.v
    sens = np.array([(mean_of_tilt(b+1e-3, g0)-mean_of_tilt(b-1e-3, g0))/2e-3
                     for b in beta_true])
    print(f"{regime}:")
    print(f"   per-observation branch weight: min {w.min():.5f}  median {np.median(w):.5f}  max {w.max():.5f}")
    print(f"   |beta_true| median {np.median(np.abs(beta_true)):.3f}   sum(beta_true) {beta_true.sum():+.3f}")
    print(f"   d E[g] / d beta: median {np.median(sens):.4f}  (small = beta poorly identified)")
    print(f"   implied d beta for a 0.00265 error in E[g]: {0.00265/np.median(sens):.4f}")
    print(f"   sum of |beta_true| {np.abs(beta_true).sum():.1f} vs |sum| {abs(beta_true.sum()):.3f}"
          f"  -> cancellation factor {np.abs(beta_true).sum()/max(abs(beta_true.sum()),1e-9):.0f}x")
    w2 = mix.branch_weights(x); print(f"   exact composed log-odds {np.log(w2[1]/w2[0]):+.3f}")
