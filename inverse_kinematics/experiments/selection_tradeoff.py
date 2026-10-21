#!/usr/bin/env python3
"""How much of the n=32 MAP gap is the dataset's selection rather than a method?

``bimodality_study.selected_dataset`` keeps only simulations whose tip satisfies
|x1| < eps. That selection is what frees the reflection
``(a1,a2,a3,h) -> (-a1,-a2,-a3,-h)`` and makes the shared posterior bimodal at
all -- but it also makes the data atypical for the generating angle, and the
posterior correctly follows them inward. Measured at eps = 0.15: mean |x1| is
0.090 against 0.971 unselected, and the reference's modes drift from
(-0.54, +0.60) at n = 8 to +-0.46 by n = 32 even at 480,000 draws.

So eps controls a genuine trade-off and nothing about the inference method can
escape it: smaller eps buys bimodality and costs bias, larger eps the reverse.
This sweeps it and reports, per eps, how far the *reference's own* MAP sits from
the generating angle and whether the posterior is still bimodal. That upper
bounds what any method can achieve on the dataset, so it says whether a MAP
error of 0.06 at n = 32 is a method failure or the data.

No network and no GPU: this is the ABC integral only.

Usage:
    python selection_tradeoff.py --observations 32 --epsilons 0.05 0.15 0.3 0.5
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
from bimodality_study import selected_dataset  # noqa: E402


def summarise(grid, density, truth):
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    density = density / np.trapezoid(density, grid)
    peak = float(grid[int(np.argmax(density))])
    interior = np.arange(1, len(grid) - 1)
    hit = interior[(density[interior] >= density[interior - 1])
                   & (density[interior] >= density[interior + 1])
                   & (density[interior] > 0.05 * density.max())]
    modes = [float(grid[i]) for i in hit] or [peak]
    if len(hit) >= 2:
        low, high = hit[0], hit[-1]
        dip = float(density[low:high + 1].min()
                    / min(density[low], density[high]))
    else:
        dip = 1.0
    mean = float(np.trapezoid(grid * density, grid))
    sd = float(np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid)))
    # Does the truth carry any mass at all? Reported as the density at the
    # generating angle relative to the peak, which is what "outside the
    # probability mass" means concretely.
    at_truth = float(np.interp(truth, grid, density) / density.max())
    return {"map": peak, "map_error": float(abs(abs(peak) - truth)),
            "modes": modes, "dip_ratio": dip, "sd": sd,
            "density_at_truth_over_peak": at_truth}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--observations", type=int, nargs="+", default=[8, 32])
    result.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.05, 0.15, 0.3, 0.5, 1.0])
    result.add_argument("--true-angle", type=float, default=0.6)
    result.add_argument("--seed", type=int, default=99)
    result.add_argument("--draws", type=int, default=60_000)
    result.add_argument("--grid-points", type=int, default=801)
    result.add_argument("--output", type=Path, default=ROOT / "artifacts"
                        / "mixture_inference" / "selection_tradeoff.json")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    span = 4.0 * float(H.GLOBAL_PRIOR_STD[0])
    grid = np.linspace(-span, span, int(arguments.grid_points))
    truth = float(arguments.true_angle)

    rows = []
    print(f"{'n':>4}{'eps':>7}{'mean|x1|':>10}{'MAP':>8}{'|MAP|-0.6':>11}"
          f"{'dip':>7}{'sd':>7}{'p(0.6)/peak':>13}{'modes':>22}")
    for n in arguments.observations:
        for epsilon in arguments.epsilons:
            rng = sim.generator("bimodal-dataset", arguments.seed)
            x = np.asarray(selected_dataset(n, truth, epsilon, rng),
                           dtype=np.float64)
            _, density = H.reference_global(x, grid=grid,
                                            draws=arguments.draws)
            record = summarise(grid, density, truth)
            record.update({"observations": int(n), "epsilon": float(epsilon),
                           "mean_abs_x1": float(np.abs(x[:, 0]).mean())})
            rows.append(record)
            print(f"{n:4d}{epsilon:7.2f}{record['mean_abs_x1']:10.3f}"
                  f"{record['map']:+8.3f}{record['map_error']:11.3f}"
                  f"{record['dip_ratio']:7.3f}{record['sd']:7.3f}"
                  f"{record['density_at_truth_over_peak']:13.3f}"
                  f"{', '.join(f'{v:+.2f}' for v in record['modes'][:3]):>22}",
                  flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
