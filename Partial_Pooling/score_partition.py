#!/usr/bin/env python3
"""Partition sensitivity: does the answer depend on which subjects share a group?

The exact Bayesian posterior cannot. `p(g | x_1..x_100)` is what it is however
the 100 subjects are bundled into composed factors, so **any** dependence here
is an approximation artifact that the grouped representation introduces and the
flat arms cannot have. That is why the plan gates every quotable number on this
measurement rather than taking it on faith.

Two pre-declared tolerances (plan section 3.5):

* across-run sd of each global posterior median, as a fraction of the *within*-run
  posterior sd -- but **measured against the Monte-Carlo floor, not against
  zero**. The plan's literal "under 10%" is not achievable and was mis-specified:
  each run draws only 100 posterior samples, and the median of 100 draws from a
  roughly normal posterior already has sd ~ `sigma * sqrt(pi/200)` = **12.5% of
  the posterior sd** from resampling alone. A criterion below the noise floor
  would fail every arm, including one with no partition dependence whatsoever.
  What is actually diagnostic is whether the across-grouping spread **exceeds**
  what independently redrawing the same posterior would produce, so the floor is
  estimated by bootstrapping each run's own draws;
* the `log_sigma_nu` median-on-truth slope must vary by **less than 0.05**.

Run:
    python Partial_Pooling/score_partition.py \\
        --contiguous ladder_group/gK20/artifacts_churn_eta2 \\
        --seeded 'ladder_group/gK20_part*/artifacts_churn_eta2' \\
        --datasets 30
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from schema import GLOBAL_NAMES  # noqa: E402
from score_bayesflow_metrics import ROOT, load_run  # noqa: E402
from score_shrinkage import regress  # noqa: E402

METHOD = "dpm2_gauss_hierarchical"
# Ratio of the observed across-grouping spread to the bootstrap Monte-Carlo
# floor. 1.0 means the groupings differ no more than independent redraws of the
# same posterior would; the tolerance allows a 50% excess over that.
EXCESS_TOLERANCE = 1.5
SLOPE_TOLERANCE = 0.05        # spread of the log_sigma_nu slope
BOOTSTRAP = 200

# The across-grouping sd is itself estimated from only a handful of runs, and a
# sample sd from n observations has relative standard error ~ 1/sqrt(2(n-1)):
# 32% at n=6, 25% at n=9. Quoting a bare excess ratio against a threshold reads
# far more precision into it than a study this size can carry, so every ratio is
# reported with an interval and the verdict is stated in terms of what the
# interval actually excludes.
def excess_interval(ratio, runs):
    relative = 1.0 / np.sqrt(2.0 * (runs - 1))
    return ratio * (1 - relative), ratio * (1 + relative)


def bootstrap_floor(estimates, rounds=BOOTSTRAP, seed=0):
    """Across-replicate sd of the median from resampling alone.

    The empirical version of `sigma * sqrt(pi/2n)`, which avoids assuming the
    posterior is normal. This is the spread two groupings would show even if the
    grouping changed nothing at all, so it is the baseline the observed spread
    has to be judged against.
    """
    generator = np.random.default_rng(seed)
    datasets, draws, params = estimates.shape
    medians = np.empty((rounds, datasets, params))
    for index in range(rounds):
        pick = generator.integers(0, draws, size=draws)
        medians[index] = np.median(estimates[:, pick, :], axis=1)
    return medians.std(axis=0, ddof=1)               # (datasets, params)


def collect(root, datasets):
    """Per-dataset posterior medians and within-run sds for one grouping."""
    payload = load_run(root, METHOD, limit=datasets)
    estimates = payload["global_estimates"]          # (datasets, draws, params)
    return {
        "floor": bootstrap_floor(estimates),
        "root": payload["run"],
        "datasets": payload["datasets"],
        "median": np.median(estimates, axis=1),      # (datasets, params)
        "sd": np.std(estimates, axis=1),             # (datasets, params)
        "slopes": np.array([r["slope"] for r in
                            regress(estimates, payload["global_targets"])]),
        "r": np.array([r["r"] for r in
                       regress(estimates, payload["global_targets"])]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Partition sensitivity")
    parser.add_argument("--contiguous", required=True)
    parser.add_argument("--seeded", required=True,
                        help="glob over the seeded run roots")
    parser.add_argument("--datasets", type=int, default=30)
    parser.add_argument(
        "--min-datasets", type=int, default=10,
        help="Ignore a grouping with fewer than this many finished datasets; "
             "a handful gives a degenerate slope and a meaningless spread.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    # load_run resolves a relative root against the benchmark ROOT, so the glob
    # has to as well -- resolving it against the process CWD instead silently
    # matches nothing and the study looks like it has one grouping.
    pattern = args.seeded
    if not Path(pattern).is_absolute():
        pattern = str(ROOT / pattern)
    roots = [args.contiguous] + sorted(glob.glob(pattern))
    # A run still in progress has fewer datasets. Truncating every run to the
    # common count is what makes them comparable at all -- the same reason
    # load_run takes a limit -- and the dataset files are index-named, so the
    # common prefix is the *same* test datasets in every grouping.
    counts = {}
    for root in roots:
        try:
            counts[root] = collect(root, args.datasets)["datasets"]
        except SystemExit:
            pass
    usable = {r: n for r, n in counts.items() if n >= args.min_datasets}
    if len(usable) < 3:
        raise SystemExit(
            f"need at least 3 groupings with >= {args.min_datasets} datasets; "
            f"have {len(usable)} (counts: {counts})"
        )
    common = min(usable.values())
    runs = [collect(root, common) for root in sorted(usable)]
    print(f"{len(runs)} groupings, truncated to the common {common} datasets "
          f"(1 contiguous + {len(runs)-1} seeded)\n")

    medians = np.stack([run["median"] for run in runs])     # (runs, datasets, params)
    within = np.stack([run["sd"] for run in runs])
    slopes = np.stack([run["slopes"] for run in runs])      # (runs, params)
    correlations = np.stack([run["r"] for run in runs])

    # Across-run spread of the median, per dataset and parameter, expressed in
    # units of that run's own posterior sd -- the comparison the tolerance is in.
    across = medians.std(axis=0, ddof=1)                    # (datasets, params)
    floor = np.stack([run["floor"] for run in runs]).mean(axis=0)
    ratio = across / within.mean(axis=0)
    excess = across / floor
    print(f"{'parameter':>22} {'med sd/post sd':>15} {'MC floor':>9} "
          f"{'excess':>7} {'~68% CI':>13} {'slope spread':>13} {'r spread':>10}")
    record, worst_ratio, worst_slope = {}, 0.0, 0.0
    for index, name in enumerate(GLOBAL_NAMES):
        mean_ratio = float(ratio[:, index].mean())
        mean_floor = float((floor[:, index] / within.mean(axis=0)[:, index]).mean())
        mean_excess = float(excess[:, index].mean())
        slope_spread = float(slopes[:, index].max() - slopes[:, index].min())
        r_spread = float(correlations[:, index].max() - correlations[:, index].min())
        worst_ratio = max(worst_ratio, mean_excess)
        if name == "log_sigma_nu":
            worst_slope = slope_spread
        flag = ""
        if mean_excess > EXCESS_TOLERANCE:
            flag += "  EXCESS TOL"
        if name == "log_sigma_nu" and slope_spread > SLOPE_TOLERANCE:
            flag += "  SLOPE TOL"
        record[name] = {"median_ratio": mean_ratio, "mc_floor": mean_floor,
                        "excess": mean_excess, "slope_spread": slope_spread,
                        "r_spread": r_spread,
                        "slopes": slopes[:, index].tolist(),
                        "r": correlations[:, index].tolist()}
        low, high = excess_interval(mean_excess, len(runs))
        record[name]["excess_interval"] = [low, high]
        print(f"{name:>22} {mean_ratio:>15.4f} {mean_floor:>9.4f} "
              f"{mean_excess:>7.2f} {f'[{low:.2f},{high:.2f}]':>13} "
              f"{slope_spread:>13.4f} {r_spread:>10.4f}{flag}")

    mean_r = correlations.mean(axis=1)
    print(f"\nmean r across groupings: {mean_r.min():.4f} - {mean_r.max():.4f} "
          f"(spread {mean_r.max()-mean_r.min():.4f})")
    print(f"\nTolerances (plan 3.5, with the median criterion restated against "
          f"the Monte-Carlo floor):")
    low, high = excess_interval(worst_ratio, len(runs))
    print(f"  across-grouping spread / MC floor < {EXCESS_TOLERANCE}: "
          f"worst is {worst_ratio:.2f} [{low:.2f}, {high:.2f}] -> "
          f"{'PASS' if worst_ratio <= EXCESS_TOLERANCE else 'FAIL'}")
    if low <= EXCESS_TOLERANCE <= high:
        print(f"    ...but the interval straddles {EXCESS_TOLERANCE}: with "
              f"{len(runs)} groupings this study cannot resolve whether the "
              f"threshold is exceeded. What it does establish is the interval's "
              f"lower end, {low:.2f}.")
    print(f"  log_sigma_nu slope spread < {SLOPE_TOLERANCE}: "
          f"{worst_slope:.4f} -> {'PASS' if worst_slope <= SLOPE_TOLERANCE else 'FAIL'}")
    if worst_ratio <= EXCESS_TOLERANCE and worst_slope <= SLOPE_TOLERANCE:
        print("\nThe grouping does not materially change the answer, so the "
              "headline numbers are not an artifact of how subjects were bundled.")
    else:
        print("\nPartition dependence exceeds the declared tolerance. Report it, "
              "and do not quote a headline without it.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"runs": [r["root"] for r in runs], "per_parameter": record,
             "mean_r": mean_r.tolist(),
             "tolerances": {"excess_over_mc_floor": EXCESS_TOLERANCE,
                            "slope_spread": SLOPE_TOLERANCE}},
            indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
