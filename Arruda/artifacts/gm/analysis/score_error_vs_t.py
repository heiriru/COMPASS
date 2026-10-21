"""Learned vs exact single-observation score across noise levels, on the hier_g1.2 n=128 data."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
# command-line flags (e.g. --weights-tag) are parsed by the exec-ed hier_improve setup
sys.path[:0] = [str(HERE)]
src = (HERE / "hier_improve.py").read_text().split("rows, panels = [], {}")[0]   # reuse setup + score functions only
exec(compile(src, "hier_improve_setup", "exec"))
x, grid, exact = R.gm_tall_problem(128, SEED, mixture, "hierarchical", G_TRUE)
g = np.linspace(-2.5, 2.5, 101, dtype="float32")
for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    a, sg = (float(np.asarray(ops.convert_to_numpy(u)).ravel()[0]) for u in alpha_sigma(time=ops.convert_to_tensor([[t]], "float32")))
    z = (a * (g - m) / s)                                  # noisy-state coordinate of a clean g, in standardised units
    zz = np.repeat(z, len(x))[:, None]; xx = np.tile(x, (len(z), 1)); tt = ops.full((len(zz), 1), t)
    L = ops.convert_to_numpy(learned_score(ops.convert_to_tensor(zz), time=tt, conditions=ops.convert_to_tensor(xx))).reshape(len(z), -1)
    E = ops.convert_to_numpy(exact_local_score(ops.convert_to_tensor(zz), time=tt, conditions=ops.convert_to_tensor(xx))).reshape(len(z), -1)
    err = L - E
    print(f"t={t:.2f} sigma_g={sg * s / a:.3f}  rms|exact|={np.sqrt((E**2).mean()):.2f}  rms err={np.sqrt((err**2).mean()):.3f}  "
          f"|summed err over 128 obs| median={np.median(np.abs(err.sum(1))):.1f} max={np.abs(err.sum(1)).max():.1f}  "
          f"vs |summed exact| median={np.median(np.abs(E.sum(1))):.1f}", flush=True)
