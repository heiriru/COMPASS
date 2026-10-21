#!/usr/bin/env python3
"""gauss_jacobian_FD on a saved IK dataset, angle_1-global checkpoint, 4000 draws per n.

    run_rail_fd.py [DATA.npy [OUT_DIR]]   (defaults: the rail-height set -> rail_a0.6_compass)

One file per n, skipped if present, so it is resumable."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import run_network as RN  # noqa: E402  (CPU cap + autocvd)
import numpy as np  # noqa: E402
from layouts import LAYOUTS  # noqa: E402

ROOT = HERE.parents[1]
IK = ROOT / "Arruda" / "artifacts" / "ik"
x = np.load(sys.argv[1] if len(sys.argv) > 1 else IK / "rail_height" / "data_a0.6_tol0.05.npy")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else IK / "rail_a0.6_compass"
OUT.mkdir(parents=True, exist_ok=True)
layout = LAYOUTS["angle"]
model = RN.trainer.load_to(layout.checkpoint, "cuda")
for n in (1, 2, 4, 8, 16, 32, 64, 128):
    path = OUT / f"samples_gauss_jacobian_FD_n{n}.npy"
    if path.exists():
        continue
    theta = RN.composed(model, layout, x[:n], "gauss_jacobian_FD", 4000, 2026 + n)
    np.save(path, theta[:, 0, 0].astype(np.float32))
    print(f"n={n}: mean {theta[:, 0, 0].mean():+.3f} sd {theta[:, 0, 0].std():.3f}", flush=True)
