#!/usr/bin/env python3
"""Parameter budget and the data-dependent share across the te grid, N=6.

The share is the diagnostic this directory's notes identify as the real
constraint: at h16d2/te256 only 5,056 of 367,136 parameters -- 1.4% -- can
respond to the observation at all, the rest being adaLN modulation and the
timestep MLP, both functions of diffusion time alone. Shrinking te does not
touch the data-dependent parameters; it removes time-only ones, so the share
rises as the model shrinks.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import te_rewire as TE
import train_model as T

GRID = (256, 128, 64, 32, 16)


def main():
    for config in ("h16d2", "h32d3", "h64d4wide"):
        print(f"\n### {config}  ({T.CONFIGS[config]})")
        print(f"{'te':>5} {'total':>12} {'time-only':>12} {'data-dep':>10} {'data-dep %':>11} {'vs te256':>9}")
        base = None
        for te in GRID:
            model = T.build(config, "legacy")
            TE.rewire(model, te)
            r = TE.parameter_report(model.model)
            base = base or r["total"]
            print(f"{te:>5} {r['total']:>12,} {r['time_only']:>12,} "
                  f"{r['data_dependent']:>10,} {100 * r['data_dependent_share']:>10.1f}% "
                  f"{base / r['total']:>8.2f}x")


if __name__ == "__main__":
    main()
