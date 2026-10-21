#!/usr/bin/env python3
"""Can a Gaussian-*mixture* per-observation summary replace the grid product?

The measured defect in every rule in ``MultiObsSampler`` is that it summarises
each observation's shared marginal as **one** Gaussian, and one Gaussian cannot
carry unequal branch weights (``docs/nonsymmetric_composition.md``).  The exact
1-D density product fixes that but needs a grid, so it binds at ~3 shared
dimensions and cannot reach ``Partial_Pooling``.

A mixture summary removes both limits in principle, because mixtures are closed
under the two operations composition needs:

* **Products.** ``prod_j sum_k w_jk N(mu_jk, s_jk^2)`` is again a Gaussian
  mixture, in closed form, in any dimension -- and the shared prior here is
  ``N(0, 0.5^2)``, so the ``prior^(1-n)`` division is analytic too.
* **Diffusion.** A Gaussian mixture convolved with ``N(0, sigma_t^2)`` is a
  Gaussian mixture with ``Sigma + sigma_t^2 I``, so the composed marginal can be
  diffused exactly and fed back as a shared score at every ``t``.

``K = 1`` reproduces ``gauss_hierarchical`` exactly, so this is a strict
generalization of what the library already does, with the component count acting
as its own dispatcher.

This script tests the idea on saved arrays only -- no model, no sampler, no GPU.
It reuses the per-observation draws and the ABC reference that
``composition_attribution.py`` already wrote, so the mixture product is graded
against exactly the same reference as every rule in that table.

Two failure modes are separated deliberately:

1. **Is the summary adequate?**  ``--mode exact`` enumerates every one of the
   ``K^n`` product components with no pruning at all.  Any gap to the grid
   product is then the mixture representation and nothing else.
2. **Does pruning survive?**  ``--mode beam`` keeps only the ``M`` heaviest
   components after each factor, which is what a dimension-free implementation
   would have to do.  The gap to ``exact`` is the pruning error.

Usage:

    python mixture_product.py                      # every saved dataset
    python mixture_product.py --datasets selected_n8_seed99 --components 1 2 3 4
    python mixture_product.py --mode beam --beam 64 256 1024
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import hierarchical as H  # noqa: E402
from composition_attribution import (  # noqa: E402
    compare, grid_product, moments, prior_density,
)

ATTRIBUTION = ROOT / "artifacts" / "composition_attribution"
OUTPUT = ROOT / "artifacts" / "mixture_product"
RULES = ("gauss_jacobian", "gauss_hierarchical", "langevin_fnpse")
# A variance floor in units of the draw spread.  Without one, EM on 20k draws
# from a sharply peaked branch will happily collapse a component onto a handful
# of points and report an enormous likelihood.
VARIANCE_FLOOR_FRACTION = 1e-3


# ---------------------------------------------------------------------------
# 1-D EM, because the environment has no sklearn
# ---------------------------------------------------------------------------

def fit_mixture(draws, k, restarts=8, iterations=400, tolerance=1e-9, seed=0,
                variance_ceiling=None):
    """EM for a 1-D ``k``-component Gaussian mixture, best of ``restarts``.

    Returns ``(weights, means, variances)``.  ``k = 1`` is the moment match, so
    it is computed in closed form rather than iterated.
    """
    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    if k == 1:
        variance = max(draws.var(), 1e-12)
        if variance_ceiling is not None:
            variance = min(variance, float(variance_ceiling))
        return (np.ones(1), np.array([draws.mean()]), np.array([variance]))

    floor = VARIANCE_FLOOR_FRACTION * draws.var()
    best = None
    for restart in range(restarts):
        rng = np.random.default_rng(seed + 977 * restart)
        means = rng.choice(draws, size=k, replace=False).astype(np.float64)
        variances = np.full(k, draws.var())
        weights = np.full(k, 1.0 / k)
        previous = -np.inf
        for _ in range(iterations):
            # Responsibilities, in log space so a far-away component underflows
            # to zero rather than to a NaN.
            log_component = (
                np.log(np.maximum(weights, 1e-300))[None, :]
                - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
                - 0.5 * (draws[:, None] - means[None, :]) ** 2 / variances[None, :]
            )
            peak = log_component.max(axis=1, keepdims=True)
            scaled = np.exp(log_component - peak)
            total = scaled.sum(axis=1, keepdims=True)
            log_likelihood = float(np.sum(peak[:, 0] + np.log(total[:, 0])))
            responsibility = scaled / total

            mass = responsibility.sum(axis=0)
            weights = mass / len(draws)
            means = (responsibility * draws[:, None]).sum(axis=0) / np.maximum(mass, 1e-300)
            variances = np.maximum(
                (responsibility * (draws[:, None] - means[None, :]) ** 2).sum(axis=0)
                / np.maximum(mass, 1e-300),
                floor,
            )
            if variance_ceiling is not None:
                # A posterior component cannot be wider than the prior component
                # it came from: multiplying a prior by a likelihood only ever
                # narrows. Without this the fitted tails are far heavier than the
                # prior's, and ``prior^(1-n)`` amplifies exactly that excess.
                variances = np.minimum(variances, float(variance_ceiling))
            if log_likelihood - previous < tolerance * abs(log_likelihood):
                break
            previous = log_likelihood
        if best is None or log_likelihood > best[0]:
            best = (log_likelihood, weights.copy(), means.copy(), variances.copy())
    return best[1], best[2], best[3]


def bic(draws, weights, means, variances):
    """Bayesian information criterion, the component-count selector."""
    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    log_component = (
        np.log(np.maximum(weights, 1e-300))[None, :]
        - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
        - 0.5 * (draws[:, None] - means[None, :]) ** 2 / variances[None, :]
    )
    peak = log_component.max(axis=1, keepdims=True)
    log_likelihood = float(np.sum(
        peak[:, 0] + np.log(np.exp(log_component - peak).sum(axis=1))
    ))
    parameters = 3 * len(weights) - 1
    return parameters * np.log(len(draws)) - 2.0 * log_likelihood


# ---------------------------------------------------------------------------
# The product, in natural parameters
# ---------------------------------------------------------------------------
# A component is (precision, precision_mean, log_constant), standing for
#     exp(-0.5 * precision * a^2 + precision_mean * a + log_constant),
# which is the form that *adds* under multiplication.  Working here rather than
# in (weight, mean, variance) is what keeps the ``prior^(1-n)`` division exact:
# it is one more factor with a **negative** precision, added like any other.

def to_natural(weights, means, variances):
    precision = 1.0 / variances
    return (precision, means * precision,
            np.log(np.maximum(weights, 1e-300))
            - 0.5 * np.log(2.0 * np.pi * variances)
            - 0.5 * means**2 * precision)


def log_mass(precision, precision_mean, log_constant):
    """Log integral of a component; ``-inf`` where it is improper."""
    proper = precision > 0.0
    result = np.full(precision.shape, -np.inf)
    result[proper] = (
        log_constant[proper]
        + 0.5 * precision_mean[proper] ** 2 / precision[proper]
        + 0.5 * np.log(2.0 * np.pi / precision[proper])
    )
    return result


def exact_product(mixtures, prior_precision):
    """Every one of the ``K^n`` components, with no pruning whatsoever."""
    naturals = [to_natural(*mixture) for mixture in mixtures]
    n = len(naturals)
    counts = [len(natural[0]) for natural in naturals]
    combinations = np.array(list(itertools.product(*[range(c) for c in counts])))
    precision = sum(naturals[j][0][combinations[:, j]] for j in range(n))
    precision_mean = sum(naturals[j][1][combinations[:, j]] for j in range(n))
    log_constant = sum(naturals[j][2][combinations[:, j]] for j in range(n))
    # The prior^(1-n) division: one factor, negative precision, zero mean.
    precision = precision - (n - 1) * prior_precision
    return precision, precision_mean, log_constant


def beam_product(mixtures, prior_precision, beam):
    """Sequential product keeping the ``beam`` heaviest components each step.

    The ``prior^(1-n)`` division is applied **once, at the end**, not spread
    across the factors.  Spreading it was the first thing tried and it is wrong:
    the per-observation posteriors here are barely narrower than the
    ``N(0, 0.5^2)`` prior, so subtracting even a fraction of ``(n-1)`` prior
    precisions from a single factor drives every component improper and the
    product is empty before it starts.  A partial product of *normalized*
    densities, by contrast, always has positive precision, so pruning on its
    component masses is well defined at every step.

    Returns the surviving components plus how many went improper after the
    division -- the quantity that decides whether this is safe to sample from.
    """
    naturals = [to_natural(*mixture) for mixture in mixtures]
    n = len(naturals)
    precision = np.zeros(1)
    precision_mean = np.zeros(1)
    log_constant = np.zeros(1)
    for natural in naturals:
        precision = (precision[:, None] + natural[0][None, :]).reshape(-1)
        precision_mean = (precision_mean[:, None] + natural[1][None, :]).reshape(-1)
        log_constant = (log_constant[:, None] + natural[2][None, :]).reshape(-1)
        if len(precision) > beam:
            order = np.argsort(
                log_mass(precision, precision_mean, log_constant)
            )[-beam:]
            precision, precision_mean, log_constant = (
                precision[order], precision_mean[order], log_constant[order]
            )
    precision = precision - (n - 1) * prior_precision
    return (precision, precision_mean, log_constant,
            int((precision <= 0.0).sum()))


def density_on_grid(grid, precision, precision_mean, log_constant):
    """Tabulate the component mixture.  Used for **grading only**.

    The method itself never needs a grid -- the composed object is the component
    list.  The reference this is graded against is a grid density, so the
    comparison has to happen somewhere, and it happens here.
    """
    grid = np.asarray(grid, dtype=np.float64)
    log_density = (
        -0.5 * precision[None, :] * grid[:, None] ** 2
        + precision_mean[None, :] * grid[:, None]
        + log_constant[None, :]
    )
    peak = log_density.max()
    density = np.exp(log_density - peak).sum(axis=1)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("mixture product did not normalise on the grid")
    return density / area


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------

def run_dataset(name, arguments):
    archive = ATTRIBUTION / name / "curves.npz"
    if not archive.exists():
        raise FileNotFoundError(f"no saved curves at {archive}")
    with np.load(archive) as stored:
        grid = stored["grid"]
        reference = stored["joint_reference"]
        draws = stored["single_draws"]
        per_observation_network = stored["per_observation_network"]
        rule_curves = {rule: stored[f"curve_{rule}"] for rule in RULES
                       if f"curve_{rule}" in stored}
        network_product = stored["curve_network_product"]
    reference_moments = moments(grid, reference)
    prior = prior_density(grid)
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    n = draws.shape[0]

    rows = []
    # The two things the mixture has to beat, on this dataset's own reference.
    rows.append({"method": "grid_product", "components": None, "beam": None,
                 **compare(grid, grid_product(grid, per_observation_network, prior),
                           reference, reference_moments)})
    rows.append({"method": "network_product_saved", "components": None,
                 "beam": None,
                 **compare(grid, network_product, reference, reference_moments)})
    for rule, curve in rule_curves.items():
        rows.append({"method": rule, "components": None, "beam": None,
                     **compare(grid, curve, reference, reference_moments)})

    # Component counts, and what BIC would have picked per observation.
    selected = []
    for k in arguments.components:
        mixtures = [fit_mixture(draws[j], k, seed=arguments.seed + j)
                    for j in range(n)]
        scores = [bic(draws[j], *mixtures[j]) for j in range(n)]
        selected.append((k, mixtures, float(np.mean(scores)), scores))

        combinations = int(np.prod([len(m[0]) for m in mixtures]))
        if arguments.mode in ("exact", "both") and combinations <= arguments.max_exact:
            precision, precision_mean, log_constant = exact_product(
                mixtures, prior_precision
            )
            improper = int((precision <= 0.0).sum())
            rows.append({
                "method": f"mixture_exact_k{k}", "components": k, "beam": None,
                "product_components": combinations,
                "improper_components": improper,
                **compare(grid, density_on_grid(
                    grid, precision, precision_mean, log_constant),
                    reference, reference_moments),
            })
        if arguments.mode in ("beam", "both"):
            for beam in arguments.beam:
                precision, precision_mean, log_constant, dropped = beam_product(
                    mixtures, prior_precision, beam
                )
                rows.append({
                    "method": f"mixture_beam{beam}_k{k}", "components": k,
                    "beam": beam, "product_components": len(precision),
                    "improper_components": dropped,
                    **compare(grid, density_on_grid(
                        grid, precision, precision_mean, log_constant),
                        reference, reference_moments),
                })

    bic_pick = min(selected, key=lambda entry: entry[2])[0]
    return rows, {"observations": n, "bic_component_choice": bic_pick,
                  "bic_mean_by_k": {str(k): s for k, _, s, _ in selected},
                  "reference": reference_moments}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subdirectories of artifacts/composition_attribution")
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--mode", choices=["exact", "beam", "both"], default="both")
    parser.add_argument("--beam", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--max-exact", type=int, default=300_000,
                        help="skip exact enumeration past this many components")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    datasets = arguments.datasets or sorted(
        path.name for path in ATTRIBUTION.iterdir()
        if (path / "curves.npz").exists()
    )
    arguments.output.mkdir(parents=True, exist_ok=True)

    everything = {}
    for name in datasets:
        rows, summary = run_dataset(name, arguments)
        everything[name] = {"rows": rows, "summary": summary}
        print(f"\n=== {name}  (n={summary['observations']}, "
              f"BIC picks k={summary['bic_component_choice']}) ===")
        print(f"{'method':<26}{'W1/sig':>8}{'mean/sig':>10}{'width':>8}"
              f"{'mass>0':>8}{'dip':>8}{'comps':>9}{'improper':>9}")
        for row in rows:
            print(f"{row['method']:<26}{row['w1_over_sigma']:>8.3f}"
                  f"{row['mean_error_sigma']:>10.3f}{row['width_ratio']:>8.3f}"
                  f"{row['positive_mass']:>8.3f}{row['dip_ratio']:>8.3f}"
                  f"{row.get('product_components') or 0:>9d}"
                  f"{row.get('improper_components') or 0:>9d}")

    path = arguments.output / "mixture_product.json"
    path.write_text(json.dumps(everything, indent=1, default=float))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
