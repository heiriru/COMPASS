#!/usr/bin/env python3
"""How much of the composed error survives a *perfect* score network.

The sweep measures rules that read their per-observation scores from a trained
network, so a failure there has two possible causes and the sweep cannot
separate them: the network's score error, amplified by the composition, or the
composition itself. This script removes the first cause entirely.

Every composition rule in COMPASS builds the shared score as

    s(g) = sum_i  d/dg log p(g | x_i)   -   (n - 1) d/dg log p_prior(g)

and the sampler API accepts only a **Gaussian** prior, so on this problem it is
handed the moment match ``N(0, sqrt(mu_0^2 + tau_g^2)) = N(0, 2.53)`` in place of
a prior whose components have sd 0.4. Here the per-observation posteriors are
the **exact** ones -- closed form, no network anywhere -- and the composed
density is integrated from that score on a grid. So the only error left is the
prior term.

Three columns are reported per (regime, n):

``exact``
    the same composition with the exact mixture prior. It reproduces the closed
    form to grid accuracy, which is the identity check on this whole file: the
    factorisation is exact, so anything else in the table is misspecification
    and not the rule's algebra.

``gaussian prior``
    the moment-matched Gaussian in the ``(n-1)`` term. **A perfect network,
    composed with the prior the API forces.**

``+ 2% score error``
    the exact per-observation scores scaled by 1.02 -- a 2% multiplicative error,
    far smaller than any network's -- to show how the ``(n-1)`` cancellation
    converts a small relative score error into an O(1) error in the answer.

Usage:
    python prior_ablation.py
    python prior_ablation.py --observations 1 2 4 8 16 32 --scale 1.05
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import datasets  # noqa: E402
import metrics  # noqa: E402
from model import DEFAULT  # noqa: E402

ROOT = Path(__file__).resolve().parent


def single_observation_score(mixture, grid, value):
    """``d/dg log p(g | x_i)`` in closed form, for one observation."""
    weights, means, variance = mixture.global_posterior(np.array([value]))
    log_terms = np.stack([
        np.log(weight) - 0.5 * (grid - mean) ** 2 / variance
        for weight, mean in zip(weights, means)
    ])
    log_terms -= log_terms.max(axis=0, keepdims=True)
    responsibility = np.exp(log_terms)
    responsibility /= responsibility.sum(axis=0, keepdims=True)
    scores = np.stack([-(grid - mean) / variance for mean in means])
    return (responsibility * scores).sum(axis=0)


def gaussian_prior_score(mixture, grid):
    """The score of the moment match the sampler API is handed."""
    return -grid / mixture.global_prior_std ** 2


def composed_density(grid, score):
    """The normalised density whose log-derivative is ``score``."""
    log_density = np.concatenate([[0.0], np.cumsum(
        0.5 * (score[1:] + score[:-1]) * np.diff(grid))])
    log_density -= log_density.max()
    density = np.exp(log_density)
    return density / np.trapezoid(density, grid)


def compose(mixture, x, grid, prior="exact", scale=1.0):
    """The composed shared density from exact per-observation scores."""
    total = scale * sum(single_observation_score(mixture, grid, value)
                        for value in x)
    if prior == "exact":
        prior_score = mixture.diffused_prior_score(grid, 0.0)
    elif prior == "gaussian":
        prior_score = gaussian_prior_score(mixture, grid)
    else:
        raise ValueError(f"unknown prior {prior!r}")
    return composed_density(grid, total - (len(x) - 1) * prior_score)


def wasserstein_to_grid(mixture, x, grid, density):
    """W1 between a tabulated density and the exact posterior, in exact sds."""
    probabilities = np.linspace(0.0005, 0.9995, 2000)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cumulative /= cumulative[-1]
    keep = np.concatenate([[True], np.diff(cumulative) > 0])
    got = np.interp(probabilities, cumulative[keep], grid[keep])
    want = mixture.global_quantiles(probabilities, x)
    return float(np.abs(got - want).mean()) / mixture.global_moments(x)[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--observations", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32])
    result.add_argument("--regimes", nargs="+", default=list(datasets.REGIMES))
    result.add_argument("--branch-mass", type=float, default=0.75)
    result.add_argument("--scale", type=float, default=1.02,
                        help="Multiplicative error applied to every exact "
                             "per-observation score in the third column.")
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--replicates", type=int, default=3,
                        help="Must match the sweep's: the datasets are rebuilt "
                             "from the same seeds and the median is taken over "
                             "the same replicates, or the two columns would be "
                             "medians of different data. Under 'natural' that "
                             "is the difference between agreement and nonsense, "
                             "since each replicate there draws its own g.")
    result.add_argument("--points", type=int, default=200_001)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    mixture = DEFAULT
    grid = np.linspace(-8.0, 8.0, int(arguments.points))
    rows = []
    print(f"W1 / exact sd, from EXACT per-observation posteriors "
          f"(no network anywhere)\n")
    header = (f"{'regime':<15}{'n':>4}{'exact prior':>14}"
              f"{'gaussian prior':>16}{f'+{arguments.scale:.2f}x score':>16}")
    print(header)
    print("-" * len(header))
    for regime in arguments.regimes:
        for n in arguments.observations:
            entry = {"regime": regime, "observations": int(n)}
            per_replicate = {"exact": [], "gaussian": [], "scaled": []}
            for replicate in range(int(arguments.replicates)):
                rng = np.random.default_rng(
                    [arguments.seed, datasets.REGIMES.index(regime), n,
                     replicate]
                )
                x = datasets.make_dataset(regime, n, rng, mixture=mixture,
                                          branch_mass=arguments.branch_mass)["x"]
                for label, kwargs in (
                    ("exact", {"prior": "exact"}),
                    ("gaussian", {"prior": "gaussian"}),
                    ("scaled", {"prior": "exact", "scale": arguments.scale}),
                ):
                    per_replicate[label].append(wasserstein_to_grid(
                        mixture, x, grid, compose(mixture, x, grid, **kwargs)
                    ))
            entry.update({label: float(np.median(values))
                          for label, values in per_replicate.items()})
            rows.append(entry)
            print(f"{regime:<15}{n:>4}{entry['exact']:>14.4f}"
                  f"{entry['gaussian']:>16.4f}{entry['scaled']:>16.4f}")
    output = Path(arguments.artifacts) / "scaling"
    output.mkdir(parents=True, exist_ok=True)
    (output / "prior_ablation.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {output / 'prior_ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
