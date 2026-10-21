"""Confirm the natural-bimodality screen: recompute the ABC reference for the most promising datasets
with more draws and three Monte Carlo seeds. A dataset counts as bimodal only if every seed shows a
second mode holding >= 5% of the mass (screen candidates at 20k draws were MC noise; see the log).

Promising = the screen's two-peak candidates plus the datasets whose mirror image -mode came closest
to the mode (the reflection is the only mechanism that gives this model a second mode).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "inverse_kinematics")]
import hierarchical as H
import natural_bimodal_search as S

SEED, DRAWS, TOP_MIRROR, MIN_MASS = 20260929, 80_000, 20, 0.05
rows = json.loads((HERE / "natural_bimodal_screen.json").read_text())
mirror = np.array([r["mirror_log_ratio"] for r in rows])
by_mirror = sorted(rows, key=lambda r: -r["mirror_log_ratio"])[:TOP_MIRROR]
candidates = [r for r in rows if r["second"] and r["second"]["mass"] >= MIN_MASS]
chosen = {r["dataset"]: r for r in by_mirror + candidates}
print(f"{len(rows)} screened; mirror log-ratio quantiles (1/10/50/90/99%): "
      f"{np.round(np.percentile(mirror, [1, 10, 50, 90, 99]), 1).tolist()}; max {mirror.max():.1f}; "
      f"{len(candidates)} two-peak candidates; checking {len(chosen)}", flush=True)
grid = np.linspace(-2.0, 2.0, 801)
results = []
for k, row in sorted(chosen.items()):
    rng = np.random.default_rng([SEED, k])
    a1 = float(rng.normal(0.0, H.GLOBAL_PRIOR_STD[0]))
    locals_ = rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(S.N, 3))
    x = H.forward(np.concatenate([np.full((S.N, 1), a1), locals_], axis=1))
    checks = [S.summary(grid, S.log_posterior(x, grid, DRAWS, seed=2000 + j, device="cuda", chunk=1)) for j in range(3)]
    masses = [c["second"]["mass"] if c["second"] else 0.0 for c in checks]
    bimodal = all(m >= MIN_MASS for m in masses)
    results.append(dict(dataset=k, true_angle=a1, screen_mirror=row["mirror_log_ratio"], screen_peaks=row["peaks"],
                        peaks=[c["peaks"] for c in checks], second_mass=masses,
                        mirror=[c["mirror_log_ratio"] for c in checks], bimodal=bimodal))
    print(f"dataset {k:5d} true {a1:+.3f}: peaks {[np.round(c['peaks'], 3).tolist() for c in checks]} "
          f"second-mode mass {np.round(masses, 3).tolist()} mirror {np.round([c['mirror_log_ratio'] for c in checks], 1).tolist()}"
          f" -> {'BIMODAL' if bimodal else 'no'}", flush=True)
    if bimodal:
        np.save(HERE / f"natural_bimodal_dataset_{k}.npy", x)
(HERE / "natural_bimodal_confirmed.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"confirmed bimodal: {[r['dataset'] for r in results if r['bimodal']]}", flush=True)
