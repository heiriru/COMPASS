#!/usr/bin/env python3
"""Where does the tilt's residual error on this benchmark come from?

The true likelihood here is exactly a degree-2 tilt, so the family contains the
truth and the only possible sources are (a) estimation error in the two sample
moments and (b) how sharply theta responds to them. This measures both, per
regime, against the known answer.
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

def raw(th, c, s):
    beta = th[:,0]/s - 2.0*th[:,1]*c/s**2
    gamma = -2.0*th[:,1]/s**2
    return beta, gamma

print(f"{'regime':<14}{'n':>4}{'S':>7} | {'gamma bias%':>11}{'gamma sd%':>10} | "
      f"{'beta sd':>8} | {'sd ratio':>9}{'logodds err':>12}{'W1/sd':>8}")
for regime in ("asymmetric", "antisymmetric"):
    for n in (4, 16, 100):
        for S in (20000, 200000):
            data = datasets.make_dataset(regime, n, np.random.default_rng([2026,datasets.REGIMES.index(regime),n,0]),
                                         mixture=mix, branch_mass=0.75)
            x = np.asarray(data["x"]).ravel()
            grid = mix.default_grid(x, points=6001)
            prior = A._normalise(grid, mix.prior_density(grid))
            lp = np.log(np.maximum(prior, 1e-300))
            draws = A.exact_single_draws(mix, x, S, np.random.default_rng([2026,n,0,7]))
            th, deg, c, s, _ = tilt_fit.fit_all(draws, grid, lp, degrees=(2,))
            beta, gamma = raw(th, c, s)
            truth_b, truth_g = x/mix.v, 1.0/mix.v
            # composed object, in closed form for a 2-component prior
            gt, bt = gamma.sum(), beta.sum()
            var_t = 1.0/(1.0/mix.tau_g**2 + gt)
            sd_exact = np.sqrt(mix.posterior_variance(n))
            lo_fit = 2.0*var_t*bt*mix.mu_0/mix.tau_g**2
            w = mix.branch_weights(x); lo_true = np.log(w[1]/w[0])
            dens = tilt_fit.compose(th, grid, lp, c, s)
            print(f"{regime:<14}{n:4d}{S:7d} | "
                  f"{100*(gamma.mean()-truth_g)/truth_g:11.2f}{100*gamma.std()/truth_g:10.2f} | "
                  f"{(beta-truth_b).std():8.4f} | "
                  f"{np.sqrt(var_t)/sd_exact:9.4f}{lo_fit-lo_true:12.3f}"
                  f"{A.w1(grid,dens,mix,x):8.3f}", flush=True)
