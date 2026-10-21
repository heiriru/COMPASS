#!/usr/bin/env python3
"""Inverse kinematics: grade the composition rule with the network removed.

Every inverse-kinematics number measured so far mixes three error sources -- the
trained score, the diffusion discretisation, and the composition rule -- and
only the third is what a new rule changes. On the Gaussian-mixtures benchmark
those can be separated because the single-observation posterior is closed form.
Here it is not closed form, but it does not need a network either: the forward
map is deterministic, so ``p(x_j | a)`` is the pushforward of the local prior
through ``f(a, .)`` and ``hierarchical.reference_global`` estimates it by Monte
Carlo with a Gaussian kernel. Called with a *single* observation it gives
``p(a | x_j)`` on a grid, to the same ABC accuracy as the tall reference that
grades everything else.

So: build the ``n`` single-observation references, draw from each by inverting
its CDF, hand those draws to the composition rules, and grade the result against
the tall reference built the same way. Errors that survive belong to the rule.

Both objects share the ABC kernel's smoothing bias, which is the point -- the
rules are being compared against a reference that makes the same approximation
they are fed, so a rule cannot be penalised for the referee's bias.

Usage:
    python ik_analytic_composition.py --observations 8 32 100
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
import simulator as sim  # noqa: E402
import tilt_fit  # noqa: E402
from bimodality_study import selected_dataset  # noqa: E402

FLOOR = 1e-300


def _normalise(grid, density):
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("density did not normalise")
    return density / area


def inverse_cdf_sample(grid, density, count, rng):
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cumulative /= cumulative[-1]
    keep = np.concatenate([[True], np.diff(cumulative) > 0])
    return np.interp(rng.random(int(count)), cumulative[keep], grid[keep])


def w1_over_sigma(grid, density, reference):
    step = np.diff(grid)
    a = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * step)])
    b = np.concatenate([[0.0], np.cumsum(0.5 * (reference[1:] + reference[:-1]) * step)])
    a /= a[-1]
    b /= b[-1]
    mean = float(np.trapezoid(grid * reference, grid))
    second = float(np.trapezoid(grid ** 2 * reference, grid))
    sigma = float(np.sqrt(max(second - mean ** 2, 1e-12)))
    return float(np.trapezoid(np.abs(a - b), grid)) / sigma


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

def arm_exact_product(draws, grid, log_prior, factors, **_):
    """The identity itself, using the ABC factors directly -- the rule's floor.

    No estimation at all: the per-observation densities are the ones the
    reference is built from, multiplied on the grid with the exact prior power.
    Anything a sample-based arm loses against this is its estimator's cost.
    """
    n = len(factors)
    value = np.log(np.maximum(factors, FLOOR)).sum(axis=0) - (n - 1) * log_prior
    value -= value.max()
    return _normalise(grid, np.exp(value)), {}


def arm_mixture(draws, grid, log_prior, factors, k_max=5, seed=0, bag=0, **_):
    from mixture_inference import infer as mixture_infer

    density, diagnostics = mixture_infer(
        draws, grid, float(H.GLOBAL_PRIOR_STD[0]), k_max=k_max, beam=64,
        restarts=1, seed=seed, device=None, bag=bag)
    return density, {"K": [int(v) for v in diagnostics["chosen_k"]]}


def arm_tilt(draws, grid, log_prior, factors, degrees=(2, 4, 6, 8, 10, 12),
             bag=0, seed=0, **_):
    thetas, chosen, centre, scale, _ = tilt_fit.fit_all(
        draws, grid, log_prior, degrees=tuple(degrees))
    if bag:
        density = tilt_fit.compose_bagged(thetas, grid, log_prior, centre,
                                          scale, replicates=bag, seed=seed)
    else:
        density = tilt_fit.compose(thetas, grid, log_prior, centre, scale)
    return density, {"degree": [int(v) for v in chosen]}


ARMS = {
    "exact_product": arm_exact_product,
    "mixture": arm_mixture,
    "mixture_bagged": lambda *a, **k: arm_mixture(*a, **{**k, "bag": 64}),
    "tilt": arm_tilt,
    "tilt_bagged": lambda *a, **k: arm_tilt(*a, **{**k, "bag": 64}),
}
ORDER = list(ARMS)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--observations", type=int, nargs="+",
                        default=[8, 16, 32, 64, 100])
    result.add_argument("--arms", nargs="+", default=None)
    result.add_argument("--seed", type=int, default=99)
    result.add_argument("--true-angle", type=float, default=0.6)
    result.add_argument("--select-x1", type=float, default=0.15)
    result.add_argument("--single-draws", type=int, default=20_000)
    result.add_argument("--reference-draws", type=int, default=60_000)
    result.add_argument("--grid-points", type=int, default=1_601)
    result.add_argument("--replicates", type=int, default=3,
                        help="Resampling replicates: the ABC factors are fixed, "
                             "so this varies only the draws taken from them, "
                             "which is exactly the estimator's own noise.")
    result.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "mixture_inference"
                        / "ik_analytic")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    arms = list(arguments.arms or ORDER)
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")

    span = 4.0 * float(H.GLOBAL_PRIOR_STD[0])
    grid = np.linspace(-span, span, int(arguments.grid_points))
    log_prior = -0.5 * (grid / float(H.GLOBAL_PRIOR_STD[0])) ** 2
    log_prior -= np.log(np.trapezoid(np.exp(log_prior), grid))

    arguments.output.mkdir(parents=True, exist_ok=True)
    rows, stored = [], {"grid": grid}
    for n in arguments.observations:
        rng = sim.generator("bimodal-dataset", arguments.seed)
        x = np.asarray(selected_dataset(n, arguments.true_angle,
                                        arguments.select_x1, rng),
                       dtype=np.float64)
        started = time.perf_counter()
        _, reference = H.reference_global(x, grid=grid,
                                          draws=arguments.reference_draws)
        factors = np.asarray([
            _normalise(grid, H.reference_global(
                x[index:index + 1], grid=grid,
                draws=arguments.reference_draws)[1])
            for index in range(n)
        ])
        print(f"n={n}: {n + 1} ABC references in "
              f"{time.perf_counter() - started:.0f}s", flush=True)
        stored[f"reference_n{n}"] = reference
        stored[f"x_n{n}"] = x

        for replicate in range(int(arguments.replicates)):
            generator = np.random.default_rng([arguments.seed, n, replicate])
            draws = np.asarray([
                inverse_cdf_sample(grid, factor, arguments.single_draws,
                                   generator)
                for factor in factors
            ])
            for arm in arms:
                begin = time.perf_counter()
                density, detail = ARMS[arm](
                    draws, grid, log_prior, factors,
                    seed=arguments.seed + replicate)
                seconds = time.perf_counter() - begin
                score = w1_over_sigma(grid, density, reference)
                rows.append({"observations": int(n), "replicate": replicate,
                             "arm": arm, "w1_over_sigma": score,
                             "seconds": seconds, **detail})
                if replicate == 0:
                    stored[f"{arm}_n{n}"] = density
        print("   " + "  ".join(
            f"{arm}:{np.median([r['w1_over_sigma'] for r in rows if r['arm']==arm and r['observations']==n]):.3f}"
            for arm in arms), flush=True)

    (arguments.output / "metrics.json").write_text(
        json.dumps(rows, indent=1, default=float))
    np.savez(arguments.output / "curves.npz", **stored)
    print(f"\nWrote {arguments.output}")

    print("\nmedian W1/sigma, composition rule only (ABC factors, no network)")
    print(f"{'n':>5s}  " + "  ".join(f"{a[:16]:>16s}" for a in arms))
    for n in arguments.observations:
        cells = []
        for arm in arms:
            values = [r["w1_over_sigma"] for r in rows
                      if r["arm"] == arm and r["observations"] == n]
            cells.append(f"{np.median(values):16.3f}" if values
                         else f"{'-':>16s}")
        print(f"{n:5d}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
