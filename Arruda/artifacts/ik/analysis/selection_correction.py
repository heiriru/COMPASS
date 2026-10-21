"""Does the IK reference miss the true angle because the data were *selected* on |x1| < eps?

`selected_dataset` keeps prior-predictive draws at angle_1 = 0.6 only where |x1| < 0.15, so each
kept x follows p(x | a, S) = p(x | a) 1[S] / P(S | a), not p(x | a). The reference uses p(x | a),
so it ignores a factor prod_j 1 / P(S | a) = P(S | a)^-n that depends on the angle. Correct it and
see whether the posterior then covers 0.6.
"""
import os
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "inverse_kinematics")]
import hierarchical as H

EPS, TRUE = 0.15, 0.6
coarse = np.linspace(-2, 2, 161)
rng = np.random.default_rng(0)
locals_ = rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(400_000, 3))
p_sel = np.array([np.mean(np.abs(H.forward(np.column_stack([np.full(len(locals_), a), locals_]))[:, 0]) < EPS) for a in coarse])
print("P(|x1|<eps | a) at a = -0.6, 0, 0.6:", np.interp([-0.6, 0, 0.6], coarse, p_sel).round(4), flush=True)
rows = []
for regime in ("selected", "mirror_paired"):
    z = np.load(ROOT / f"Arruda/artifacts/ik/{regime}/reference.npz"); grid = z["grid"]
    log_p_sel = np.log(np.interp(grid, coarse, p_sel))
    for key in sorted((k for k in z.files if k != "grid"), key=lambda k: int(k[1:])):
        n = int(key[1:]); d = z[key]
        with np.errstate(divide="ignore"):
            logc = np.log(d) - n * log_p_sel
        c = np.exp(logc - logc[np.isfinite(logc)].max()); c[~np.isfinite(c)] = 0; c /= np.trapezoid(c, grid)
        def stats(f):
            mean, sd, mode = H.grid_moments(grid, f)
            cdf = np.cumsum(f) * (grid[1] - grid[0])
            pos = f * (grid > 0); pos_mode = grid[np.argmax(pos)]
            return dict(P_pos=round(float(np.trapezoid(pos, grid)), 3), pos_mode=round(float(pos_mode), 3),
                        pos_sd=round(float(np.sqrt(np.trapezoid(pos * (grid - np.trapezoid(pos * grid, grid) / np.trapezoid(pos, grid)) ** 2, grid) / np.trapezoid(pos, grid))), 3))
        row = dict(regime=regime, n=n, uncorrected=stats(d), selection_corrected=stats(c))
        rows.append(row); print(json.dumps(row), flush=True)
(HERE / "selection_correction.json").write_text(json.dumps(rows, indent=2) + "\n")
