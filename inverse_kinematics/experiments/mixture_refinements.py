#!/usr/bin/env python3
"""Three follow-ups on the Gaussian-mixture product.

1. **Can a low beam be made accurate?**  ``mixture_product.py`` prunes by
   *dropping* the lightest components, and at ``K = 4`` that needs a beam of
   1024 to stay near the exact enumeration.  Dropping removes that mass from the
   density entirely, and the components it removes are exactly the ones carrying
   the minority branch -- they are light at every intermediate step, long before
   the observations that would support them have been multiplied in.

   The alternative is to **merge** rather than delete: replace the two closest
   components with the single Gaussian that preserves their combined weight,
   mean and variance (Runnalls' KL-bound criterion, standard in multi-hypothesis
   tracking).  Mass is conserved instead of discarded.  In 1-D there is a second
   reason to expect this to help a lot: the product of Gaussians depends only on
   the *sums* of precisions and precision-means, so many distinct index tuples
   land on nearly identical components.  Merging deduplicates them; dropping
   keeps one copy and throws the rest away.

2. **What does it cost, against ``gauss_hierarchical``?**  The CPU side of the
   method -- EM plus product plus pruning -- is timed here exactly.  The sampler
   side is timed separately by ``time_against_gauss_hierarchical.py``, because it
   needs the checkpoint and a GPU.

3. **Can K be chosen without knowing it?**  BIC picks ``K = 4`` on
   ``prior_n8_seed3`` where ``K = 2`` is twice as good, so it is not usable as
   is.  Four selectors are graded here against the oracle choice -- the K that
   actually minimizes W1 to the ABC reference:

   ``bic``        the current selector, on all draws.
   ``aic``        the same, with the lighter penalty.
   ``holdout``    fit on half the draws, score log-likelihood on the other half.
   ``stability``  reference-free and tied to the *composed* object: take the
                  smallest K whose composed posterior stops moving when K is
                  incremented.  This is the only one of the four that grades
                  what the method actually produces.

Usage:
    python mixture_refinements.py
    python mixture_refinements.py --task pruning --beam 8 16 32 64
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
from composition_attribution import moments, prior_density, w1_grid  # noqa: E402
from mixture_product import (  # noqa: E402
    bic, density_on_grid, exact_product, fit_mixture, log_mass, to_natural,
)

ATTRIBUTION = ROOT / "artifacts" / "composition_attribution"
OUTPUT = ROOT / "artifacts" / "mixture_product"


# ---------------------------------------------------------------------------
# 1. Merge-based pruning
# ---------------------------------------------------------------------------

def to_moments(precision, precision_mean, log_constant):
    """Natural parameters -> (weight, mean, variance), weights normalized.

    Only *relative* weights matter: every later operation either multiplies all
    components by the same factor or normalizes at the end, so the discarded
    overall scale is unrecoverable and unneeded.
    """
    variance = 1.0 / precision
    mean = precision_mean * variance
    log_weight = log_mass(precision, precision_mean, log_constant)
    log_weight = log_weight - log_weight.max()
    weight = np.exp(log_weight)
    return weight / weight.sum(), mean, variance


def from_moments(weight, mean, variance):
    return to_natural(weight, mean, variance)


def merge_to(weight, mean, variance, target):
    """Runnalls merging down to ``target`` components, preserving moments.

    The cost of merging ``i`` and ``j`` is the KL upper bound

        0.5 [ (w_i + w_j) log v_ij - w_i log v_i - w_j log v_j ],

    which is zero for identical components and grows with how much variance the
    merge invents. Repeatedly merging the cheapest pair keeps the mixture's
    total weight, mean and variance exactly and only coarsens its shape.
    """
    weight = np.array(weight, dtype=np.float64)
    mean = np.array(mean, dtype=np.float64)
    variance = np.array(variance, dtype=np.float64)
    while len(weight) > target:
        total = weight[:, None] + weight[None, :]
        merged_mean = (weight[:, None] * mean[:, None]
                       + weight[None, :] * mean[None, :]) / total
        merged_variance = (
            weight[:, None] * (variance[:, None] + (mean[:, None] - merged_mean) ** 2)
            + weight[None, :] * (variance[None, :] + (mean[None, :] - merged_mean) ** 2)
        ) / total
        cost = 0.5 * (
            total * np.log(merged_variance)
            - weight[:, None] * np.log(variance)[:, None]
            - weight[None, :] * np.log(variance)[None, :]
        )
        np.fill_diagonal(cost, np.inf)
        flat = int(np.argmin(cost))
        i, j = divmod(flat, len(weight))
        keep = np.ones(len(weight), dtype=bool)
        keep[[i, j]] = False
        weight = np.concatenate([weight[keep], [total[i, j]]])
        mean = np.concatenate([mean[keep], [merged_mean[i, j]]])
        variance = np.concatenate([variance[keep], [merged_variance[i, j]]])
    return weight, mean, variance


def sequential_product(mixtures, prior_precision, beam, strategy="drop"):
    """Product of the per-observation mixtures, pruned by ``strategy``.

    ``drop``  keep the ``beam`` heaviest components (what mixture_product does).
    ``merge`` collapse down to ``beam`` components by Runnalls merging.

    As in ``mixture_product.beam_product``, the ``prior^(1-n)`` division is
    applied once at the end: a partial product of normalized densities always
    has positive precision, so pruning and merging are well defined at every
    step, which is not true once the division is folded in.
    """
    naturals = [to_natural(*mixture) for mixture in mixtures]
    n = len(naturals)
    precision, precision_mean, log_constant = np.zeros(1), np.zeros(1), np.zeros(1)
    for natural in naturals:
        precision = (precision[:, None] + natural[0][None, :]).reshape(-1)
        precision_mean = (precision_mean[:, None] + natural[1][None, :]).reshape(-1)
        log_constant = (log_constant[:, None] + natural[2][None, :]).reshape(-1)
        if len(precision) > beam:
            if strategy == "drop":
                order = np.argsort(
                    log_mass(precision, precision_mean, log_constant))[-beam:]
                precision, precision_mean, log_constant = (
                    precision[order], precision_mean[order], log_constant[order]
                )
            else:
                precision, precision_mean, log_constant = from_moments(
                    *merge_to(*to_moments(precision, precision_mean,
                                          log_constant), beam)
                )
    precision = precision - (n - 1) * prior_precision
    return (precision, precision_mean, log_constant,
            int((precision <= 0.0).sum()))


# ---------------------------------------------------------------------------
# 3. Choosing K without knowing it
# ---------------------------------------------------------------------------

def aic(draws, weights, means, variances):
    parameters = 3 * len(weights) - 1
    penalty_difference = 2.0 * parameters - parameters * np.log(len(draws))
    return bic(draws, weights, means, variances) + penalty_difference


def holdout_log_likelihood(draws, k, seed):
    """Fit on half the draws, score the other half. No penalty term needed."""
    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(draws))
    split = len(draws) // 2
    train, test = draws[order[:split]], draws[order[split:]]
    weights, means, variances = fit_mixture(train, k, seed=seed)
    log_component = (
        np.log(np.maximum(weights, 1e-300))[None, :]
        - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
        - 0.5 * (test[:, None] - means[None, :]) ** 2 / variances[None, :]
    )
    peak = log_component.max(axis=1, keepdims=True)
    return float(np.sum(peak[:, 0]
                        + np.log(np.exp(log_component - peak).sum(axis=1))))


def stability_choice(composed, grid, reference_sd, tolerance):
    """Smallest K whose composed posterior stops moving when K is incremented.

    Reference-free, and the only selector here that grades the object the method
    actually returns rather than the per-observation fit.
    """
    ks = sorted(composed)
    for left, right in zip(ks, ks[1:]):
        if w1_grid(grid, composed[left], composed[right]) / reference_sd < tolerance:
            return left
    return ks[-1]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load(dataset):
    with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
        return (stored["grid"], stored["joint_reference"], stored["single_draws"])


def run(dataset, arguments):
    grid, reference, draws = load(dataset)
    reference_sd = moments(grid, reference)["sd"]
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    n = draws.shape[0]

    def score(curve):
        return w1_grid(grid, curve, reference) / reference_sd

    rows, fits, fit_seconds = [], {}, {}
    for k in arguments.components:
        started = time.perf_counter()
        fits[k] = [fit_mixture(draws[j], k, seed=arguments.seed + j)
                   for j in range(n)]
        fit_seconds[k] = time.perf_counter() - started

    # --- exact enumeration, the target both strategies are chasing -----------
    exact = {}
    for k in arguments.components:
        combinations = k ** n
        if combinations > arguments.max_exact:
            continue
        started = time.perf_counter()
        components = exact_product(fits[k], prior_precision)
        elapsed = time.perf_counter() - started
        exact[k] = density_on_grid(grid, *components)
        rows.append({"dataset": dataset, "k": k, "strategy": "exact",
                     "beam": combinations, "w1_over_sigma": score(exact[k]),
                     "fit_seconds": fit_seconds[k], "product_seconds": elapsed})

    # --- drop against merge, at low beams ------------------------------------
    for k in arguments.components:
        for beam in arguments.beam:
            for strategy in ("drop", "merge"):
                started = time.perf_counter()
                precision, precision_mean, log_constant, improper = (
                    sequential_product(fits[k], prior_precision, beam, strategy)
                )
                elapsed = time.perf_counter() - started
                curve = density_on_grid(grid, precision, precision_mean,
                                        log_constant)
                rows.append({
                    "dataset": dataset, "k": k, "strategy": strategy,
                    "beam": beam, "w1_over_sigma": score(curve),
                    "fit_seconds": fit_seconds[k], "product_seconds": elapsed,
                    "improper": improper,
                })

    # --- K selection ---------------------------------------------------------
    composed = {}
    for k in arguments.components:
        if k in exact:
            composed[k] = exact[k]
        else:
            composed[k] = density_on_grid(*(
                (grid,) + sequential_product(fits[k], prior_precision,
                                             arguments.selection_beam,
                                             "merge")[:3]
            ))
    oracle = min(composed, key=lambda k: score(composed[k]))
    selectors = {
        "oracle": oracle,
        "bic": min(arguments.components,
                   key=lambda k: float(np.mean([bic(draws[j], *fits[k][j])
                                                for j in range(n)]))),
        "aic": min(arguments.components,
                   key=lambda k: float(np.mean([aic(draws[j], *fits[k][j])
                                                for j in range(n)]))),
        "holdout": max(arguments.components,
                       key=lambda k: float(np.mean([
                           holdout_log_likelihood(draws[j], k,
                                                  arguments.seed + j)
                           for j in range(n)]))),
        "stability": stability_choice(composed, grid, reference_sd,
                                      arguments.stability_tolerance),
    }
    selection = {name: {"k": int(k), "w1_over_sigma": score(composed[k])}
                 for name, k in selectors.items()}
    return rows, selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--beam", type=int, nargs="+", default=[8, 16, 32, 64, 256])
    parser.add_argument("--selection-beam", type=int, default=64)
    parser.add_argument("--stability-tolerance", type=float, default=0.05,
                        help="composed W1/sigma below which K is called settled")
    parser.add_argument("--max-exact", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=["all", "pruning", "selection"],
                        default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    datasets = arguments.datasets or sorted(
        path.name for path in ATTRIBUTION.iterdir()
        if (path / "curves.npz").exists()
    )
    arguments.output.mkdir(parents=True, exist_ok=True)

    everything = {}
    for dataset in datasets:
        rows, selection = run(dataset, arguments)
        everything[dataset] = {"rows": rows, "selection": selection}

        if arguments.task in ("all", "pruning"):
            print(f"\n=== {dataset}: drop vs merge (W1/sigma) ===")
            beams = arguments.beam
            print(f"{'K':>3} {'strategy':<8}" +
                  "".join(f"{'beam ' + str(b):>12}" for b in beams) +
                  f"{'exact':>12}")
            for k in arguments.components:
                for strategy in ("drop", "merge"):
                    cells = []
                    for beam in beams:
                        hit = [r for r in rows if r["k"] == k
                               and r["strategy"] == strategy and r["beam"] == beam]
                        cells.append(f"{hit[0]['w1_over_sigma']:>12.3f}" if hit
                                     else f"{'-':>12}")
                    got = [r for r in rows if r["k"] == k and r["strategy"] == "exact"]
                    tail = f"{got[0]['w1_over_sigma']:>12.3f}" if got else f"{'n/a':>12}"
                    print(f"{k:>3} {strategy:<8}" + "".join(cells) + tail)

        if arguments.task in ("all", "selection"):
            print(f"--- {dataset}: choosing K ---")
            for name, entry in selection.items():
                mark = " <-- best" if name == "oracle" else ""
                print(f"    {name:<10} K={entry['k']}  "
                      f"W1/sigma={entry['w1_over_sigma']:.3f}{mark}")

    path = arguments.output / "mixture_refinements.json"
    path.write_text(json.dumps(everything, indent=1, default=float))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
