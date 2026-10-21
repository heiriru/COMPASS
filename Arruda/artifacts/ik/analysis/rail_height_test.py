"""Rail-height-selected IK datasets: is the ABC reference bimodal at large n?

Selection (explicit, not natural): one true angle_1, locals (height, angle_2, angle_3) from the prior,
keep only observations whose endpoint is at the rail height, |x1 - h| < TOL, i.e. whose *true*
configuration has an exactly mirror-symmetric twin (-angles, same h). The first 128 kept rows form one
nested sequence (first n for every n). The reference is the standard model's ABC posterior for these x
(what every method targets), 3 Monte Carlo seeds, bimodal iff every seed has a second mode >= 5% mass.
"""
import os
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))   # 3-thread CPU cap, before numpy/torch
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "inverse_kinematics")]
import hierarchical as H
import natural_bimodal_search as S

SEED, N, DRAWS, MIN_MASS = 20260929, 128, 80_000, 0.05
NS = (1, 2, 4, 8, 16, 32, 64, 128)
OUT = HERE.parent / "rail_height"


def rail_dataset(a1, tol, seed):
    rng = np.random.default_rng(seed)
    kept = []
    while sum(len(k) for k in kept) < N:
        locals_ = rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(20_000, 3))
        theta = np.concatenate([np.full((len(locals_), 1), a1), locals_], axis=1)
        x = H.forward(theta)
        kept.append(x[np.abs(x[:, 0] - locals_[:, 0]) < tol])
    return np.concatenate(kept)[:N]


grid = np.linspace(-2.0, 2.0, 801)
report = []
for a1 in (0.3, 0.6):
    for tol in (0.05, 0.02):
        x = rail_dataset(a1, tol, [SEED, int(a1 * 100), int(tol * 1000)])
        tag = f"a{a1:g}_tol{tol:g}"
        np.save(OUT / f"data_{tag}.npy", x)
        print(f"== {tag}: |x1| mean {np.abs(x[:, 0]).mean():.2f}, x1 range [{x[:, 0].min():+.2f}, {x[:, 0].max():+.2f}]", flush=True)
        for n in NS:
            checks = [S.summary(grid, S.log_posterior(x[:n], grid, DRAWS, seed=3000 + j, device="cuda", chunk=1)) for j in range(3)]
            masses = [c["second"]["mass"] if c["second"] else 0.0 for c in checks]
            bimodal = all(m >= MIN_MASS for m in masses)
            report.append(dict(tag=tag, n=n, peaks=[c["peaks"] for c in checks], second_mass=masses,
                               mirror=[c["mirror_log_ratio"] for c in checks], mode=[c["mode"] for c in checks], bimodal=bimodal))
            print(f"  n={n:3d}: modes {[round(c['mode'], 2) for c in checks]} peaks {[np.round(c['peaks'], 2).tolist() for c in checks]} "
                  f"2nd mass {np.round(masses, 3).tolist()} log p(-mode)/p(mode) {np.round([c['mirror_log_ratio'] for c in checks], 1).tolist()}"
                  f" -> {'BIMODAL' if bimodal else 'no'}", flush=True)
        (OUT / "bimodality_test.json").write_text(json.dumps(report, indent=2) + "\n")
