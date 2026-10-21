#!/usr/bin/env python3
"""Composition rules graded with the score removed entirely.

Every arm in ``run_scaling.py`` feeds on draws from ``p(g | x_j)`` produced by a
sampler, so its error is a sum of three things: the network, the sampler's
discretisation, and the composition rule. ``--oracle`` removes the first. This
script removes the second as well: the single-observation posterior of this
benchmark is a two-component Gaussian mixture in closed form, so its draws can
be taken *exactly*, with no diffusion in the loop at all.

What is left is the rule, measured against ``model.global_density``. That makes
this the right place to iterate: a full sweep to n = 128 runs in seconds on CPU,
and any improvement found here is an improvement to the composition and nothing
else.

Usage:
    python analytic_composition.py --observations 2 4 8 16 32 64 100 128
"""

import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "3")
try:
    os.sched_setaffinity(0, list(sorted(os.sched_getaffinity(0)))[:3])
except (AttributeError, OSError):
    pass

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# The affinity cap above hands this process three logical CPUs; torch still
# defaults its pool to the *host* count, and 36 threads fighting over 3 cores
# made the batched EM 24x slower than it is.
torch.set_num_threads(3)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "inverse_kinematics" / "experiments"))

import datasets  # noqa: E402
import model as M  # noqa: E402
import local_refine  # noqa: E402

FLOOR = 1e-300
EM_DEVICE = "cpu"


def _normalise(grid, density):
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("density did not normalise")
    return density / area


def exact_single_draws(mixture, x, count, rng):
    """``(n, count)`` exact draws from every ``p(g | x_j)``.

    Each single-observation posterior is a two-component Gaussian mixture whose
    weights, means and shared variance ``model`` gives in closed form, so this
    is sampling, not approximation.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    variance = mixture.posterior_variance(1)
    out = np.empty((len(x), count), dtype=np.float64)
    for index, value in enumerate(x):
        weights, means, _ = mixture.global_posterior(np.array([value]))
        pick = rng.random(count) < weights[1]
        out[index] = np.where(pick, means[1], means[0]) + \
            np.sqrt(variance) * rng.standard_normal(count)
    return out


def w1(grid, density, mixture, x):
    """``W1`` between a tabulated density and the exact posterior, in its sd."""
    reference = mixture.global_density(grid, x)
    step = np.diff(grid)
    cdf_a = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * step)])
    cdf_b = np.concatenate([[0.0], np.cumsum(0.5 * (reference[1:] + reference[:-1]) * step)])
    cdf_a /= cdf_a[-1]
    cdf_b /= cdf_b[-1]
    distance = float(np.trapezoid(np.abs(cdf_a - cdf_b), grid))
    _, sd = mixture.global_moments(x)
    return distance / sd


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

def arm_kde(draws, grid, prior, mixture, **_):
    from scipy.stats import gaussian_kde
    densities = np.asarray([_normalise(grid, gaussian_kde(row)(grid))
                            for row in draws])
    n = len(densities)
    log_density = (np.log(np.maximum(densities, FLOOR)).sum(axis=0)
                   - (n - 1) * np.log(np.maximum(prior, FLOOR)))
    log_density -= log_density.max()
    return _normalise(grid, np.exp(log_density)), densities, {}


_CACHE = {}


def _fit_mixtures(draws, k_max, seed, prior_at=None, weight_clip=100.0):
    """EM fits per observation, memoised: every arm asks for the same ones."""
    key = (id(draws), int(k_max), int(seed), prior_at is not None)
    if key in _CACHE:
        return _CACHE[key]
    value = _fit_mixtures_uncached(draws, k_max, seed, prior_at, weight_clip)
    _CACHE[key] = value
    return value


def _fit_mixtures_uncached(draws, k_max, seed, prior_at=None, weight_clip=100.0):
    """Batched EM for every observation at once (``batched_em``), K by BIC.

    The likelihood form is handled the same way it is in ``methods.py``: draws
    from ``p(g | x_j)`` resampled with probability proportional to ``1 / p(g)``
    are draws from a measure proportional to ``L_j``, so the identical fitter
    applies to the reweighted sample and estimates the likelihood directly.
    """
    from batched_em import fit_all

    sample = np.asarray(draws, dtype=np.float64)
    if prior_at is not None:
        rows = []
        for index, row in enumerate(sample):
            weight = 1.0 / np.maximum(prior_at(row), FLOOR)
            weight = np.minimum(weight, weight_clip * np.median(weight))
            weight = weight / weight.sum()
            rows.append(np.random.default_rng(seed + index).choice(
                row, size=len(row), replace=True, p=weight))
        sample = np.asarray(rows)
    return fit_all(sample, k_max=k_max, seed=seed, device=EM_DEVICE)


_TABLE = {}


def _tabulate(fits, grid):
    key = (id(fits), id(grid))
    if key in _TABLE:
        return _TABLE[key]
    _TABLE[key] = _tabulate_uncached(fits, grid)
    return _TABLE[key]


def _tabulate_uncached(fits, grid):
    return np.asarray([
        _normalise(grid, (
            w[None, :] / np.sqrt(2.0 * np.pi * v[None, :])
            * np.exp(-0.5 * (grid[:, None] - m[None, :]) ** 2 / v[None, :])
        ).sum(axis=1))
        for w, m, v in fits
    ])


def arm_mixture(draws, grid, prior, mixture, k_max=5, seed=0, **_):
    fits, chosen = _fit_mixtures(draws, k_max, seed)
    densities = _tabulate(fits, grid)
    n = len(densities)
    log_density = (np.log(np.maximum(densities, FLOOR)).sum(axis=0)
                   - (n - 1) * np.log(np.maximum(prior, FLOOR)))
    log_density -= log_density.max()
    return _normalise(grid, np.exp(log_density)), densities, {"K": chosen}


def arm_likelihood(draws, grid, prior, mixture, k_max=5, seed=0, **_):
    fits, chosen = _fit_mixtures(
        draws, k_max, seed,
        prior_at=lambda values: mixture.prior_density(values))
    densities = _tabulate(fits, grid)
    log_density = (np.log(np.maximum(prior, FLOOR))
                   + np.log(np.maximum(densities, FLOOR)).sum(axis=0))
    log_density -= log_density.max()
    return _normalise(grid, np.exp(log_density)), densities, {"K": chosen}


def arm_likelihood_bagged(draws, grid, prior, mixture, k_max=5, seed=0,
                          bag=64, **_):
    _, densities, detail = arm_likelihood(draws, grid, prior, mixture,
                                          k_max=k_max, seed=seed)
    log_prior = np.log(np.maximum(prior, FLOOR))
    log_factors = np.log(np.maximum(densities, FLOOR))
    rng = np.random.default_rng(seed)
    total = np.zeros(len(grid))
    for _ in range(bag):
        picked = rng.integers(0, len(log_factors), size=len(log_factors))
        value = log_prior + log_factors[picked].sum(axis=0)
        value -= value.max()
        total += _normalise(grid, np.exp(value))
    return _normalise(grid, total / bag), densities, detail


def _locator(draws, grid, prior, mixture, k_max, seed):
    """The first pass the refinement localises on: the likelihood form."""
    density, densities, detail = arm_likelihood(draws, grid, prior, mixture,
                                                k_max=k_max, seed=seed)
    return density, densities, detail


def arm_local(draws, grid, prior, mixture, k_max=5, seed=0, bag=0,
              level="mass", inflate=4.0, **_):
    pass1, densities, detail = _locator(draws, grid, prior, mixture, k_max, seed)
    log_prior = np.log(np.maximum(prior, FLOOR))
    density, info = local_refine.refine(
        draws, grid, log_prior, pass1, inflate=inflate, level=level,
        bag=bag, seed=seed)
    detail = dict(detail)
    detail["modes"] = info["modes"]
    return density, densities, detail


def arm_tilt(draws, grid, prior, mixture, degrees=(2,), bag=0, seed=0, **_):
    """The exponential-tilt composition: fit each likelihood, add the tilts."""
    import tilt_fit

    log_prior = np.log(np.maximum(prior, FLOOR))
    thetas, chosen, centre, scale, _ = tilt_fit.fit_all(
        draws, grid, log_prior, degrees=degrees)
    if bag:
        density = tilt_fit.compose_bagged(thetas, grid, log_prior, centre, scale,
                                          replicates=bag, seed=seed)
    else:
        density = tilt_fit.compose(thetas, grid, log_prior, centre, scale)
    return density, np.empty((0, len(grid))), {"degree": chosen}


ARMS = {
    "kde_product": arm_kde,
    "mixture_product": arm_mixture,
    "mixture_likelihood": arm_likelihood,
    "mixture_likelihood_bagged": arm_likelihood_bagged,
    "tilt2": lambda *a, **k: arm_tilt(*a, **{**{key: v for key, v in k.items()
                                                if key != "inflate"},
                                            "degrees": (2,)}),
    "tilt2_bagged": lambda *a, **k: arm_tilt(*a, **{**{key: v for key, v in k.items()
                                                      if key != "inflate"},
                                                  "degrees": (2,), "bag": 64}),
    "tilt_bic": lambda *a, **k: arm_tilt(*a, **{**{key: v for key, v in k.items()
                                                  if key != "inflate"},
                                                "degrees": (2, 4, 6)}),
    "tilt_bic_bagged": lambda *a, **k: arm_tilt(*a, **{**{key: v for key, v in k.items()
                                                         if key != "inflate"},
                                                       "degrees": (2, 4, 6),
                                                       "bag": 64}),
    "local_refine": arm_local,
    "local_refine_bagged": lambda *a, **k: arm_local(*a, **{**k, "bag": 64}),
    "local_refine_peak": lambda *a, **k: arm_local(*a, **{**k, "level": "peak"}),
    "local_refine_peak_bagged": lambda *a, **k: arm_local(
        *a, **{**k, "level": "peak", "bag": 64}),
}
ORDER = list(ARMS)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--observations", type=int, nargs="+",
                        default=[2, 4, 8, 16, 32, 64, 100])
    result.add_argument("--regimes", nargs="+", default=["asymmetric"])
    result.add_argument("--arms", nargs="+", default=None)
    result.add_argument("--replicates", type=int, default=5)
    result.add_argument("--single-draws", type=int, default=20_000)
    result.add_argument("--grid-points", type=int, default=6_001)
    result.add_argument("--k-max", type=int, default=5)
    result.add_argument("--inflate", type=float, default=4.0)
    result.add_argument("--branch-mass", type=float, default=0.75)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--em-device", default="cpu")
    result.add_argument("--tag", default="analytic")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    global EM_DEVICE
    EM_DEVICE = arguments.em_device
    mixture = M.DEFAULT
    arms = list(arguments.arms or ORDER)
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")

    rows, stored = [], {}
    for regime in arguments.regimes:
        for n in arguments.observations:
            for replicate in range(arguments.replicates):
                rng = np.random.default_rng(
                    [arguments.seed, datasets.REGIMES.index(regime), n, replicate])
                data = datasets.make_dataset(regime, n, rng, mixture=mixture,
                                             branch_mass=arguments.branch_mass)
                x = np.asarray(data["x"], dtype=np.float64).ravel()
                grid = mixture.default_grid(x, points=arguments.grid_points)
                prior = _normalise(grid, mixture.prior_density(grid))
                draws = exact_single_draws(
                    mixture, x, arguments.single_draws,
                    np.random.default_rng([arguments.seed, n, replicate, 7]))
                _CACHE.clear(); _TABLE.clear()
                for arm in arms:
                    started = time.perf_counter()
                    density, _, detail = ARMS[arm](
                        draws, grid, prior, mixture,
                        k_max=arguments.k_max, seed=arguments.seed + replicate,
                        inflate=arguments.inflate)
                    seconds = time.perf_counter() - started
                    score = w1(grid, density, mixture, x)
                    rows.append({"regime": regime, "observations": int(n),
                                 "replicate": replicate, "arm": arm,
                                 "w1_over_sd": score, "seconds": seconds,
                                 "modes": len(detail.get("modes", [])) or None})
                    if replicate == 0:
                        stored[f"{regime}|{n}|{arm}"] = density
                        stored[f"{regime}|{n}|grid"] = grid
                        stored[f"{regime}|{n}|reference"] = \
                            mixture.global_density(grid, x)
                        stored[f"{regime}|{n}|x"] = x
                print(f"n={n:4d} {regime:13s} " + "  ".join(
                    f"{arm}:{np.median([r['w1_over_sd'] for r in rows if r['arm']==arm and r['observations']==n and r['regime']==regime]):.3f}"
                    for arm in arms), flush=True)

    out = ROOT / "artifacts" / "analytic" / arguments.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("w") as handle:
        json.dump(rows, handle, indent=1)
    np.savez(out / "densities.npz", **stored)
    print(f"\nWrote {out}", flush=True)

    print("\nmedian W1/sd")
    header = f"{'n':>5s}  " + "  ".join(f"{a[:22]:>22s}" for a in arms)
    print(header)
    for regime in arguments.regimes:
        for n in arguments.observations:
            cells = []
            for arm in arms:
                values = [r["w1_over_sd"] for r in rows
                          if r["arm"] == arm and r["observations"] == n
                          and r["regime"] == regime]
                cells.append(f"{np.median(values):22.3f}")
            print(f"{n:5d}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
