#!/usr/bin/env python3
"""How cheap can the EM stage get without losing composed accuracy?

The product itself is free -- with merge pruning at beam 8 it costs 0.009 s at
K = 4.  **EM is the entire cost of the method**: 50-100 s for 8 observations at
the defaults (20,000 draws, 8 restarts).  That is what has to come down for the
mixture product to compete with ``gauss_hierarchical`` (16-46 s end to end in
this repository's saved runs).

Both defaults are obviously generous for fitting a 1-D mixture of at most a few
components, so this sweeps them jointly and reports composed accuracy *and*
wall clock, so the trade can be read off rather than guessed.
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import hierarchical as H  # noqa: E402
from composition_attribution import moments, w1_grid  # noqa: E402
from mixture_product import density_on_grid, fit_mixture  # noqa: E402
from mixture_refinements import sequential_product  # noqa: E402

ATTRIBUTION = ROOT / "artifacts" / "composition_attribution"
OUTPUT = ROOT / "artifacts" / "mixture_product"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["selected_n8_seed99", "selected_n8_seed101",
                                 "prior_n8_seed3"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--draws", type=int, nargs="+",
                        default=[1000, 2000, 5000, 20000])
    parser.add_argument("--restarts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    rows = []
    print(f"K = {arguments.k}, merge pruning at beam {arguments.beam}")
    print(f"{'draws':>7}{'restarts':>10}{'EM s':>9}" +
          "".join(f"{d.replace('_n8', ''):>22}" for d in arguments.datasets))
    for count in arguments.draws:
        for restarts in arguments.restarts:
            scores, seconds = [], 0.0
            for dataset in arguments.datasets:
                with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
                    grid = stored["grid"]
                    reference = stored["joint_reference"]
                    draws = stored["single_draws"]
                reference_sd = moments(grid, reference)["sd"]
                n = draws.shape[0]
                # A prefix of the saved draws: they are i.i.d., so taking the
                # first `count` is the same as having sampled only that many.
                subset = draws[:, :count]
                started = time.perf_counter()
                fits = [fit_mixture(subset[j], arguments.k,
                                    restarts=restarts, seed=arguments.seed + j)
                        for j in range(n)]
                seconds += time.perf_counter() - started
                curve = density_on_grid(*(
                    (grid,) + sequential_product(fits, prior_precision,
                                                 arguments.beam, "merge")[:3]
                ))
                scores.append(w1_grid(grid, curve, reference) / reference_sd)
            rows.append({"draws": count, "restarts": restarts,
                         "em_seconds_per_dataset": seconds / len(arguments.datasets),
                         "w1": scores})
            print(f"{count:>7}{restarts:>10}"
                  f"{seconds / len(arguments.datasets):>9.2f}" +
                  "".join(f"{s:>22.3f}" for s in scores))

    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / "em_budget.json"
    path.write_text(json.dumps({"config": vars(arguments), "rows": rows},
                               indent=1, default=str))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
