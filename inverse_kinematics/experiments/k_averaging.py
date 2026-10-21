#!/usr/bin/env python3
"""If K cannot be selected, can it be averaged over instead?

No selector beat a fixed K (`docs/mixture_product.md`, follow-up 3), and the
oracle K wanders over 2-5 with no pattern, with K = 3, 4, 5 separated by about
the EM fit noise. That pattern -- several settings that are each sometimes best,
none reliably identifiable -- is the standard case for *averaging* rather than
choosing.

It costs nothing here. A uniform mixture over K of composed mixtures is itself a
Gaussian mixture: take the union of the component lists and divide the weights by
the number of K values. So the averaged answer stays exactly representable in the
same form, diffuses the same way, and needs no extra machinery.

Graded against fixed K and against `gauss_jacobian`, at the recommended settings
(merge pruning at beam 8, EM on 5000 draws x 1 restart).
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
DATASETS = ["selected_n8_seed99", "selected_n8_seed101", "selected_n16_seed99",
            "prior_n8_seed3", "exact_zero_n8_seed99"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--em-draws", type=int, default=5000)
    parser.add_argument("--em-restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    everything = {}
    labels = [f"K={k}" for k in arguments.components] + ["average", "jacobian"]
    print(f"{'dataset':<24}" + "".join(f"{lab:>11}" for lab in labels))

    for dataset in DATASETS:
        with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
            grid = stored["grid"]
            reference = stored["joint_reference"]
            draws = stored["single_draws"]
            rule = stored["curve_gauss_jacobian"]
        reference_sd = moments(grid, reference)["sd"]
        subset = draws[:, :arguments.em_draws]
        n = draws.shape[0]

        curves = {}
        for k in arguments.components:
            fits = [fit_mixture(subset[j], k, restarts=arguments.em_restarts,
                                seed=arguments.seed + j) for j in range(n)]
            curves[k] = density_on_grid(*(
                (grid,) + sequential_product(fits, prior_precision,
                                             arguments.beam, "merge")[:3]
            ))
        # The union-of-components mixture, formed on the grid: each composed
        # density is already normalized, so a uniform average of the densities
        # *is* the uniform mixture over K.
        averaged = np.mean([curves[k] for k in arguments.components], axis=0)

        scores = {f"K={k}": w1_grid(grid, curves[k], reference) / reference_sd
                  for k in arguments.components}
        scores["average"] = w1_grid(grid, averaged, reference) / reference_sd
        scores["jacobian"] = w1_grid(grid, rule, reference) / reference_sd
        everything[dataset] = scores
        print(f"{dataset:<24}" + "".join(f"{scores[lab]:>11.3f}" for lab in labels))

    print()
    for label in labels:
        values = [everything[d][label] for d in DATASETS]
        asymmetric = [everything[d][label] for d in DATASETS
                      if d != "exact_zero_n8_seed99"]
        print(f"  {label:<10} worst {max(values):.3f} | "
              f"worst excluding the symmetric control {max(asymmetric):.3f} | "
              f"mean {np.mean(values):.3f}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / "k_averaging.json"
    path.write_text(json.dumps(everything, indent=1, default=float))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
