#!/usr/bin/env python3
"""Why likelihood criteria cannot choose K, and what replaces them.

The identity that settles it
----------------------------
Composition is a product, so in log space the errors simply **add**::

    log q_hat(a) - log q(a) = sum_j [ log p_hat_j(a) - log p_j(a) ]

The composed error is therefore controlled by the *sup-norm* of the
per-observation log-density error, multiplied by ``n`` -- and amplified further
by the ``prior^(1-n)`` division, which contributes ``-(n-1) log p_0(a)``, a term
growing quadratically in ``a``.  At ``n = 16`` and ``|a| = 1.5`` with a
``N(0, 0.5^2)`` prior that is a lift of ``exp(+70)`` applied to whatever the fits
claim in the tail.

Now compare what BIC, AIC and held-out likelihood actually measure::

    E_{p_j}[ log p_hat_j ]

an average **weighted by p_j itself**.  By construction that cannot see a region
where ``p_j`` is small and ``p_hat_j`` is not -- the weight is the thing that
vanishes there.  So the three selectors that failed did not fail by accident:
they estimate the wrong norm.  An L1/sup criterion in log space is required, and
likelihood is an L1 criterion in *density*, weighted the opposite way.

The classical statistic with exactly the right weighting is **Anderson-Darling**,
whose weight ``1 / (F (1-F))`` diverges at both tails; Kolmogorov-Smirnov is the
unweighted version and should therefore behave like the likelihood criteria.
Both are computable from the draws alone, with no reference.

This grades AD and KS as selectors of K against the oracle, alongside BIC.
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
from scipy.special import ndtr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import hierarchical as H  # noqa: E402
from composition_attribution import moments, w1_grid  # noqa: E402
from mixture_product import bic, density_on_grid, fit_mixture  # noqa: E402
from mixture_refinements import sequential_product  # noqa: E402

ATTRIBUTION = ROOT / "artifacts" / "composition_attribution"
OUTPUT = ROOT / "artifacts" / "mixture_product"
DATASETS = ["selected_n8_seed99", "selected_n8_seed101", "selected_n16_seed99",
            "prior_n8_seed3", "exact_zero_n8_seed99"]


def mixture_cdf(draws, weights, means, variances):
    return (weights[None, :] * ndtr(
        (draws[:, None] - means[None, :]) / np.sqrt(variances)[None, :]
    )).sum(axis=1)


def anderson_darling(draws, weights, means, variances):
    """A^2 with its 1/(F(1-F)) tail weighting -- the norm composition needs."""
    values = np.sort(mixture_cdf(np.sort(draws), weights, means, variances))
    values = np.clip(values, 1e-12, 1.0 - 1e-12)
    count = len(values)
    index = np.arange(1, count + 1)
    return float(-count - np.mean(
        (2 * index - 1) * (np.log(values) + np.log1p(-values[::-1]))
    ))


def kolmogorov_smirnov(draws, weights, means, variances):
    """The unweighted counterpart -- the control."""
    values = np.sort(mixture_cdf(np.sort(draws), weights, means, variances))
    count = len(values)
    empirical = np.arange(1, count + 1) / count
    return float(np.max(np.abs(values - empirical)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--em-draws", type=int, default=5000)
    parser.add_argument("--em-restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    everything, picks = {}, {"oracle": [], "AD": [], "KS": [], "BIC": []}

    for dataset in DATASETS:
        with np.load(ATTRIBUTION / dataset / "curves.npz") as stored:
            grid = stored["grid"]
            reference = stored["joint_reference"]
            draws = stored["single_draws"]
        reference_sd = moments(grid, reference)["sd"]
        n = draws.shape[0]
        subset = draws[:, :arguments.em_draws]

        scores, statistics = {}, {"AD": {}, "KS": {}, "BIC": {}}
        for k in arguments.components:
            fits = [fit_mixture(subset[j], k, restarts=arguments.em_restarts,
                                seed=arguments.seed + j) for j in range(n)]
            curve = density_on_grid(*(
                (grid,) + sequential_product(fits, prior_precision,
                                             arguments.beam, "merge")[:3]
            ))
            scores[k] = w1_grid(grid, curve, reference) / reference_sd
            statistics["AD"][k] = float(np.mean(
                [anderson_darling(subset[j], *fits[j]) for j in range(n)]))
            statistics["KS"][k] = float(np.mean(
                [kolmogorov_smirnov(subset[j], *fits[j]) for j in range(n)]))
            statistics["BIC"][k] = float(np.mean(
                [bic(subset[j], *fits[j]) for j in range(n)]))

        chosen = {name: min(values, key=lambda k: values[k])
                  for name, values in statistics.items()}
        oracle = min(scores, key=lambda k: scores[k])
        everything[dataset] = {"scores": scores, "statistics": statistics,
                               "chosen": chosen, "oracle": oracle}
        picks["oracle"].append(scores[oracle])
        for name in ("AD", "KS", "BIC"):
            picks[name].append(scores[chosen[name]])

        print(f"\n=== {dataset} (n = {n}) ===")
        print(f"{'K':>3}{'W1/sigma':>11}{'A^2':>12}{'KS':>10}{'BIC':>14}")
        for k in arguments.components:
            mark = "  <-- oracle" if k == oracle else ""
            print(f"{k:>3}{scores[k]:>11.3f}{statistics['AD'][k]:>12.2f}"
                  f"{statistics['KS'][k]:>10.4f}{statistics['BIC'][k]:>14.1f}{mark}")
        print(f"    chosen: AD K={chosen['AD']}  KS K={chosen['KS']}  "
              f"BIC K={chosen['BIC']}   (oracle K={oracle})")

    print("\nW1/sigma delivered by each selector")
    for name in ("oracle", "AD", "KS", "BIC"):
        values = picks[name]
        print(f"  {name:<8} mean {np.mean(values):.3f}  worst {max(values):.3f}  "
              f"per-dataset " + " ".join(f"{v:.3f}" for v in values))

    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / "tail_weighted_selection.json"
    path.write_text(json.dumps(everything, indent=1, default=float))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
