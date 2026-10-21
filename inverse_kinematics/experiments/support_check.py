#!/usr/bin/env python3
"""A reference-free validity check for the composed mixture, and what it caught.

The check
---------
Composition is a product::

    p(a | x_1..n)  ∝  p(a)^(1-n) * prod_j p(a | x_j)

so the composed support is the **intersection** of the per-observation supports:
if even one factor is negligible at ``a``, the product is negligible at ``a``,
whatever the other factors do.  That is an identity, not a heuristic, and it
needs no reference -- each ``p(a | x_j)`` comes with draws, so its support is
directly observable.

So: composed mass outside ``∩_j [min_j draws, max_j draws]`` is **provably**
wrong.  Any approximation that puts mass there has failed, and the amount of
mass it puts there is a reference-free error bound in its own right.

What it caught
--------------
This was written to test a selection rule and instead falsified two claims in
``docs/mixture_product.md``:

* ``K = 2`` is *not* intrinsically unstable at ``n = 16``.  The exact product
  over all ``2^16`` components scores 0.231.
* Merge pruning at beam 8 is *not* safe at ``n = 16``.  It scores 3.246 there,
  and the check localises why: it puts mass far outside the intersection of the
  supports, where no observation has a single draw.

Merging preserves the moments of the pair it collapses, so collapsing two
*distant* components manufactures one wide component.  Over 16 sequential steps
squeezed to 8 components that compounds, and the manufactured width dominates the
tails -- where the ``prior^(1-n)`` division then multiplies it by ``exp(+70)`` at
``|a| = 1.5``.  Dropping cannot do this: it never invents mass.  Merging is still
the better estimator, but it has a failure mode dropping does not have, and the
beam has to be large enough to prevent it.
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


def support_violation(grid, density, draws, quantile):
    """Composed mass outside the intersection of the per-observation supports.

    The support of each factor is estimated by a two-sided quantile of its
    draws rather than by the extremes, so a single stray draw cannot widen it.
    ``quantile = 0`` would use the observed min and max.
    """
    lower = max(float(np.quantile(row, quantile)) for row in draws)
    upper = min(float(np.quantile(row, 1.0 - quantile)) for row in draws)
    inside = (grid >= lower) & (grid <= upper)
    total = float(np.trapezoid(density, grid))
    kept = float(np.trapezoid(density[inside], grid[inside]))
    return max(0.0, 1.0 - kept / total), (lower, upper)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["selected_n8_seed99", "selected_n16_seed99"])
    parser.add_argument("--components", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--beam", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--strategies", nargs="+", default=["merge", "drop"])
    parser.add_argument("--em-draws", type=int, default=5000)
    parser.add_argument("--em-restarts", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    everything = {}
    for dataset in arguments.datasets:
        with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
            grid = stored["grid"]
            reference = stored["joint_reference"]
            draws = stored["single_draws"]
        reference_sd = moments(grid, reference)["sd"]
        n = draws.shape[0]
        subset = draws[:, :arguments.em_draws]
        rows = []
        print(f"\n=== {dataset} (n = {n}) ===")
        print(f"{'K':>3}{'strategy':>9}{'beam':>7}{'W1/sigma':>11}"
              f"{'outside support':>18}")
        for k in arguments.components:
            fits = [fit_mixture(subset[j], k, restarts=arguments.em_restarts,
                                seed=arguments.seed + j) for j in range(n)]
            for strategy in arguments.strategies:
                for beam in arguments.beam:
                    curve = density_on_grid(*(
                        (grid,) + sequential_product(fits, prior_precision,
                                                     beam, strategy)[:3]
                    ))
                    violation, bounds = support_violation(
                        grid, curve, subset, arguments.quantile
                    )
                    score = w1_grid(grid, curve, reference) / reference_sd
                    rows.append({"k": k, "strategy": strategy, "beam": beam,
                                 "w1_over_sigma": score,
                                 "outside_support": violation,
                                 "support": bounds})
                    flag = "  <-- INVALID" if violation > 1e-3 else ""
                    print(f"{k:>3}{strategy:>9}{beam:>7}{score:>11.3f}"
                          f"{violation:>18.4f}{flag}")
        everything[dataset] = rows

    # Does the check separate the failures from the successes?
    print("\nscreening power (threshold 1e-3 of composed mass outside support)")
    for dataset, rows in everything.items():
        flagged = [r for r in rows if r["outside_support"] > 1e-3]
        clean = [r for r in rows if r["outside_support"] <= 1e-3]
        print(f"  {dataset}")
        print(f"    flagged {len(flagged):>3}: worst W1/sigma "
              f"{max((r['w1_over_sigma'] for r in flagged), default=float('nan')):.3f}, "
              f"best {min((r['w1_over_sigma'] for r in flagged), default=float('nan')):.3f}")
        print(f"    clean   {len(clean):>3}: worst W1/sigma "
              f"{max(r['w1_over_sigma'] for r in clean):.3f}, "
              f"best {min(r['w1_over_sigma'] for r in clean):.3f}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / "support_check.json"
    path.write_text(json.dumps(everything, indent=1, default=float))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
