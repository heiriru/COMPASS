"""Is the IK ABC reference stable across Monte Carlo seeds on the Arruda datasets?"""
import os
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "3"
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "inverse_kinematics")]
import hierarchical as H, bimodality_study as S
SEED = 20260923
grid = np.linspace(-2.0, 2.0, 1_401)
rows = []
for n in (8, 32, 64, 128):
    x = S.selected_dataset(n, true_angle=0.6, epsilon=0.15, rng=np.random.default_rng(SEED + n))
    for ref_seed in (SEED + n, 1, 2):          # the first seed is the one run_arruda used
        _, d = H.reference_global(x, grid=grid, draws=80_000, seed=ref_seed)
        d /= np.trapezoid(d, grid)
        pos = float(np.trapezoid(np.where(grid > 0, d, 0), grid))
        mean, sd, mode = H.grid_moments(grid, d)
        rows.append(dict(n=n, ref_seed=ref_seed, mass_positive=pos, mode=mode, sd=sd))
        print(json.dumps(rows[-1]), flush=True)
    print(f"n={n}: x1 mean {x[:,0].mean():+.4f}, x1 range [{x[:,0].min():+.3f},{x[:,0].max():+.3f}]", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
