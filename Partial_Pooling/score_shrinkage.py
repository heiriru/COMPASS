#!/usr/bin/env python3
"""Median-on-truth regression: the shrinkage diagnostic, per global parameter.

`score_bayesflow_metrics.py` reports the paper's metrics -- NRMSE, calibration
error, coverage, width, median correlation. It does not report the quantity the
Group experiment is actually about.

`docs/followup_results.md` localised the whole remaining gap to the paper as
shrinkage of the group *scales*: regressing each parameter's posterior median on
its truth across datasets gives a slope of **0.339** for `log_sigma_nu`, which
mechanically caps that parameter's correlation near 0.65 -- the 0.653 measured.
The group means are fine (slopes 0.78-1.06). So the mechanism criterion for a
grouped arm is that this slope rises substantially from 0.339, and mean `r` is
secondary: if `r` improves while the slope does not, the mechanism claim is
wrong and has to be reported as wrong.

Columns, matching the table in `docs/followup_results.md`:

* ``slope``      -- b in ``median ~ a + b * truth``. 1.0 is no shrinkage.
* ``shift``      -- mean(median - truth) in units of the truth's own sd.
* ``shift_se``   -- |shift| / its standard error, so >2 is a real offset.
* ``err_ratio``  -- sd(median - truth) / sd(truth).
* ``r``          -- Pearson correlation of median and truth (the paper's metric).

Run:
    python Partial_Pooling/score_shrinkage.py \\
        --arm mask6mix:dpm2_gauss_hierarchical:ladder_followup/mask6mix \\
        --arm gK20:dpm2_gauss_hierarchical:ladder_group/gK20/artifacts_churn_eta2 \\
        --datasets 30 --output Partial_Pooling/artifacts_ladder_report/shrinkage.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from score_bayesflow_metrics import load_run, parse_arm  # noqa: E402
from schema import GLOBAL_NAMES, LOCAL_NAMES  # noqa: E402

# The published diagnosis (docs/followup_results.md, "Where the remaining bias
# actually is"). These are mask6mix under **gauss_hierarchical only**, so they
# are the right reference for a gauss_hierarchical arm and the wrong one for any
# other rule: mask6mix's own log_sigma_log_alpha slope is 0.532 under
# gauss_hierarchical but 0.222 under F-NPSE, so scoring an F-NPSE arm against
# 0.532 reads as a regression when it is an improvement on its own baseline.
# The column is labelled "hier-ref" for that reason, and when two arms are given
# the same-method delta is printed underneath, which is the comparison that
# actually means something.
REFERENCE_SLOPES = {
    "mu_nu": 0.782, "mu_log_alpha": 1.057, "mu_log_t0": 1.040,
    "log_sigma_nu": 0.339, "log_sigma_log_alpha": 0.532,
    "log_sigma_log_t0": 0.897, "beta_raw": 0.948,
}
# The plan's pre-declared mechanism criterion, and the two halves are different
# in kind: log_sigma_nu must *rise* substantially from its 0.339, while
# log_sigma_log_alpha must merely not *fall* from its 0.532. Labelling both as
# "target" would read as a failure whenever the second one merely held.
CRITERIA = {
    "log_sigma_nu": ("rise to", 0.55),
    "log_sigma_log_alpha": ("hold at", 0.532),
}
FLOOR_TOLERANCE = 0.005      # a hold is a hold; do not fail it on rounding


def regress(estimates, targets):
    """One row per parameter: slope, offset and spread of median against truth.

    ``estimates`` is ``(datasets, draws, parameters)``; the median is taken over
    draws, so this is blind to posterior width by construction -- which is the
    point, since NRMSE already charges for width and would confound the two.
    """
    medians = np.median(estimates, axis=1)
    rows = []
    for index in range(medians.shape[1]):
        median, truth = medians[:, index], targets[:, index]
        truth_sd = float(np.std(truth))
        # Ordinary least squares of median on truth.
        centered = truth - truth.mean()
        denominator = float(np.sum(centered**2))
        slope = float(np.sum(centered * (median - median.mean())) / denominator)
        residual = median - truth
        shift = float(residual.mean() / truth_sd) if truth_sd else float("nan")
        standard_error = float(
            np.std(residual, ddof=1) / np.sqrt(len(residual)) / truth_sd
        ) if truth_sd else float("nan")
        rows.append({
            "slope": slope,
            "shift": shift,
            "shift_se": abs(shift) / standard_error if standard_error else float("nan"),
            "err_ratio": float(np.std(residual) / truth_sd) if truth_sd else float("nan"),
            "r": float(np.corrcoef(median, truth)[0, 1]),
        })
    return rows


def score_arm(arm, datasets=None):
    payload = load_run(arm["root"], arm["method"], limit=datasets)
    result = {"run": payload["run"], "datasets": payload["datasets"]}
    for scope, names in (("global", GLOBAL_NAMES), ("local", LOCAL_NAMES)):
        rows = regress(payload[f"{scope}_estimates"], payload[f"{scope}_targets"])
        result[scope] = dict(zip(names, rows))
        result[f"{scope}_mean_slope"] = float(np.mean([r["slope"] for r in rows]))
        result[f"{scope}_mean_r"] = float(np.mean([r["r"] for r in rows]))
    return result


def report(label, result):
    print(f"\n=== {label}  ({result['datasets']} datasets)")
    print(f"{'parameter':>22} {'slope':>7} {'hier-ref':>8} {'shift':>7} "
          f"{'|s|/se':>7} {'sd(e)/sd':>9} {'r':>7}")
    for name, row in result["global"].items():
        was = REFERENCE_SLOPES.get(name)
        flag = ""
        if name in CRITERIA:
            verb, threshold = CRITERIA[name]
            bound = threshold - (FLOOR_TOLERANCE if verb == "hold at" else 0.0)
            flag = "  <-- must %s %.3f: %s" % (
                verb, threshold, "MET" if row["slope"] >= bound else "NOT MET",
            )
        print(f"{name:>22} {row['slope']:>7.3f} {'':>8} if False else "
              f"{was if was is None else f'{was:8.3f}'} "
              f"{row['shift']:>7.3f} {row['shift_se']:>7.1f} {row['err_ratio']:>9.3f} "
              f"{row['r']:>7.3f}{flag}".replace(" if False else ", ""))
    print(f"{'mean over 7 globals':>22} {result['global_mean_slope']:>7.3f} "
          f"{'':>7} {'':>7} {'':>7} {'':>9} {result['global_mean_r']:>7.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Median-on-truth shrinkage")
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--datasets", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    record = {}
    for arm in args.arm:
        result = score_arm(arm, args.datasets)
        record[arm["label"]] = {**result, "method": arm["method"]}
        report(f"{arm['label']} / {arm['method']}", result)

    # Same-method deltas. The hier-ref column above is a fixed published
    # baseline and is only the right comparison for a gauss_hierarchical arm;
    # this is the one that is always right, provided the arms share a method.
    if len(args.arm) == 2 and args.arm[0]["method"] == args.arm[1]["method"]:
        base, test = (record[a["label"]] for a in args.arm)
        print(f"\n=== same-method delta ({args.arm[1]['label']} - "
              f"{args.arm[0]['label']}, {args.arm[0]['method']})")
        print(f"{'parameter':>22} {'d slope':>9} {'d r':>9}")
        for name in base["global"]:
            ds = test["global"][name]["slope"] - base["global"][name]["slope"]
            dr = test["global"][name]["r"] - base["global"][name]["r"]
            mark = "  worse" if dr < -0.02 else ("  better" if dr > 0.02 else "")
            print(f"{name:>22} {ds:>+9.3f} {dr:>+9.3f}{mark}")
        print(f"{'mean':>22} "
              f"{test['global_mean_slope'] - base['global_mean_slope']:>+9.3f} "
              f"{test['global_mean_r'] - base['global_mean_r']:>+9.3f}")

    print(
        "\nThe mechanism criterion is the log_sigma_nu slope, not mean r. If r "
        "improves while the slope does not, the hypothesis is wrong and must be "
        "reported as wrong (plan section 3.6)."
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
