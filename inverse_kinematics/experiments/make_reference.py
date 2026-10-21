#!/usr/bin/env python3
"""Build the joint ABC reference for a dataset size that has none yet.

``composition_attribution.py`` saves a full diagnostic bundle; this saves only
what grading needs -- the grid, the joint reference and the observations that
produced it -- in the same ``curves.npz`` layout, so ``run_mixture_ik.py`` can
consume it unchanged.

The reference involves no trained model: the forward map is deterministic, so
``p(x_j | a)`` is the pushforward of the local prior through ``f(a, .)``,
estimated by Monte Carlo with a Gaussian kernel. That is what lets it referee
the composition rules rather than share their error.

Cost is ``len(grid) * draws * n`` kernel evaluations, so it grows linearly in
the number of observations and is the expensive part of a large-n run.
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
from bimodality_study import selected_dataset  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--true-angle", type=float, default=0.6)
    parser.add_argument("--select-x1", type=float, default=0.15)
    parser.add_argument("--reference-draws", type=int, default=200_000)
    parser.add_argument("--reference-bandwidth", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    output = arguments.output or (
        ROOT / "artifacts" / "composition_attribution"
        / f"selected_n{arguments.observations}_seed{arguments.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    rng = sim.generator("bimodal-dataset", arguments.seed)
    x = np.asarray(selected_dataset(arguments.observations, arguments.true_angle,
                                    arguments.select_x1, rng), dtype=np.float64)
    span = 4.0 * H.GLOBAL_PRIOR_STD[0]
    grid = np.linspace(-span, span, 801)

    started = time.perf_counter()
    _, reference = H.reference_global(
        x, grid=grid, draws=arguments.reference_draws,
        bandwidth=arguments.reference_bandwidth, seed=arguments.seed,
    )
    weights = reference / reference.sum()
    mean = float((weights * grid).sum())
    sd = float(np.sqrt(max((weights * grid ** 2).sum() - mean ** 2, 0.0)))
    positive = float(np.trapezoid(reference[grid >= 0], grid[grid >= 0]))
    print(f"n = {arguments.observations}: reference mean {mean:+.4f}, sd {sd:.4f}, "
          f"mass on positive branch {positive:.3f}  "
          f"({time.perf_counter() - started:.0f}s)")

    np.savez_compressed(output / "curves.npz", grid=grid,
                        joint_reference=reference, observations=x)
    print(f"Wrote {output / 'curves.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
