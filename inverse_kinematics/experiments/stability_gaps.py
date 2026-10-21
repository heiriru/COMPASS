#!/usr/bin/env python3
"""Is there a tolerance at which the stability criterion tracks the oracle K?

``mixture_refinements.py`` grades four ways of choosing K and all four pick
K = 4 everywhere, which is wrong on two of five datasets.  For BIC, AIC and
held-out likelihood that is the end of it -- they have no free knob.  The
stability criterion does: it takes the smallest K whose composed posterior stops
moving when K is incremented, and "stops moving" is a threshold.

So the question is not whether the default tolerance worked, but whether *any*
single tolerance would.  This prints the consecutive composed gaps
``W1(K, K+1) / sigma`` alongside the oracle choice, from which the answer can be
read directly: a workable tolerance must sit above every gap the oracle wants to
ignore and below every gap it wants to follow.
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
from composition_attribution import moments, w1_grid  # noqa: E402
from mixture_product import density_on_grid, fit_mixture  # noqa: E402
from mixture_refinements import sequential_product  # noqa: E402

ATTRIBUTION = ROOT / "artifacts" / "composition_attribution"
OUTPUT = ROOT / "artifacts" / "mixture_product"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--beam", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    datasets = arguments.datasets or sorted(
        path.name for path in ATTRIBUTION.iterdir()
        if (path / "curves.npz").exists()
    )
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    everything = {}

    print(f"{'dataset':<24}{'oracle K':>9}" +
          "".join(f"{'W1(' + str(k) + ',' + str(k + 1) + ')':>12}"
                  for k in arguments.components[:-1]) +
          "".join(f"{'score K=' + str(k):>12}" for k in arguments.components))
    for dataset in datasets:
        with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
            grid = stored["grid"]
            reference = stored["joint_reference"]
            draws = stored["single_draws"]
        reference_sd = moments(grid, reference)["sd"]
        n = draws.shape[0]

        composed = {}
        for k in arguments.components:
            fits = [fit_mixture(draws[j], k, seed=arguments.seed + j)
                    for j in range(n)]
            composed[k] = density_on_grid(*(
                (grid,) + sequential_product(fits, prior_precision,
                                             arguments.beam, "merge")[:3]
            ))
        scores = {k: w1_grid(grid, composed[k], reference) / reference_sd
                  for k in arguments.components}
        gaps = {k: w1_grid(grid, composed[k], composed[k + 1]) / reference_sd
                for k in arguments.components[:-1]}
        oracle = min(scores, key=lambda k: scores[k])
        everything[dataset] = {"oracle": oracle, "scores": scores, "gaps": gaps}
        print(f"{dataset:<24}{oracle:>9}" +
              "".join(f"{gaps[k]:>12.3f}" for k in arguments.components[:-1]) +
              "".join(f"{scores[k]:>12.3f}" for k in arguments.components))

    # A tolerance works on a dataset if it stops exactly at the oracle: every
    # gap before the oracle is above it, and the gap at the oracle is below.
    print("\ntolerance -> chosen K per dataset (oracle in brackets)")
    for tolerance in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        picks = []
        for dataset, entry in everything.items():
            chosen = arguments.components[-1]
            for k in arguments.components[:-1]:
                if entry["gaps"][k] < tolerance:
                    chosen = k
                    break
            picks.append(f"{chosen}[{entry['oracle']}]")
        agree = sum(p.split("[")[0] == p.split("[")[1][:-1] for p in picks)
        print(f"  {tolerance:<6}{'  '.join(picks)}   agree {agree}/{len(picks)}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / "stability_gaps.json"
    path.write_text(json.dumps(everything, indent=1, default=float))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
