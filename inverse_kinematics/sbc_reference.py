#!/usr/bin/env python3
"""Simulation-based calibration of the ABC reference posterior.

The reference in :func:`hierarchical.reference_global` is what every composition
rule is graded against, so before reading any gap between a rule and the
reference as the *rule's* error, the reference has to be shown correct. SBC is
the end-to-end test for that and the only one that does not grade a formula
against another formula:

    draw a_true ~ prior, simulate a dataset from it, evaluate the reference
    posterior CDF at a_true

If the reference is the posterior of this simulator, those ranks are Uniform(0,1)
over replicates -- exactly, at any number of observations. Systematic departures
name their own failure: ranks piling up in the middle means the reference is too
wide, piling at the tails means too narrow, and a shifted mean rank means biased.

Usage:
    python inverse_kinematics/sbc_reference.py --replicates 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402

ROOT = Path(__file__).resolve().parent


def rank_of(truth, grid, density):
    """Posterior CDF evaluated at ``truth``: P(a < a_true | x)."""
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    )])
    cumulative /= cumulative[-1]
    return float(np.interp(truth, grid, cumulative))


def kolmogorov_smirnov(ranks):
    """One-sample KS distance of ``ranks`` from Uniform(0, 1)."""
    ranks = np.sort(np.asarray(ranks, dtype=np.float64))
    n = len(ranks)
    steps = (np.arange(1, n + 1)) / n
    return float(np.max(np.abs(
        np.concatenate([steps - ranks, ranks - (steps - 1.0 / n)])
    )))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--replicates", type=int, default=200)
    result.add_argument("--observations", type=int, default=8)
    result.add_argument("--draws", type=int, default=20_000)
    result.add_argument("--bandwidth", type=float, default=0.05)
    result.add_argument("--grid-points", type=int, default=401)
    result.add_argument("--seed", type=int, default=101)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    rng = sim.generator("sbc", arguments.seed)
    span = 4.0 * H.GLOBAL_PRIOR_STD[0]
    grid = np.linspace(-span, span, arguments.grid_points)

    ranks = []
    started = time.perf_counter()
    for replicate in range(arguments.replicates):
        truth, _, x = H.simulate_dataset(arguments.observations, rng)
        _, density = H.reference_global(
            x, grid=grid, draws=arguments.draws,
            bandwidth=arguments.bandwidth, seed=arguments.seed + replicate,
        )
        ranks.append(rank_of(truth, grid, density))
        if (replicate + 1) % 25 == 0:
            elapsed = time.perf_counter() - started
            print(f"  {replicate + 1}/{arguments.replicates} "
                  f"({elapsed:.0f}s, mean rank {np.mean(ranks):.4f})", flush=True)

    ranks = np.asarray(ranks)
    ks = kolmogorov_smirnov(ranks)
    # The 95% KS critical value for a fully specified uniform.
    critical = 1.358 / np.sqrt(len(ranks))
    verdict = "consistent with Uniform(0,1)" if ks < critical else "REJECTS uniformity"

    print(f"\n{len(ranks)} replicates, {arguments.observations} observations each")
    print(f"  mean rank    {ranks.mean():.4f}   (0.5 if calibrated)")
    print(f"  KS distance  {ks:.4f}   (95% critical {critical:.4f})  -> {verdict}")
    print(f"  deciles      {np.histogram(ranks, bins=10, range=(0, 1))[0]}")

    output = arguments.artifacts / "inference" / "sbc_reference.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "replicates": int(len(ranks)),
        "observations": int(arguments.observations),
        "draws": int(arguments.draws),
        "bandwidth": float(arguments.bandwidth),
        "mean_rank": float(ranks.mean()),
        "ks_distance": ks,
        "ks_critical_95": float(critical),
        "uniform": bool(ks < critical),
        "deciles": np.histogram(ranks, bins=10, range=(0, 1))[0].tolist(),
        "ranks": ranks.tolist(),
    }, indent=2))
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
