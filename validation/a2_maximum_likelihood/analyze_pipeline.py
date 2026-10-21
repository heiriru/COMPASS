"""Aggregate pipeline_selection shards: rates vs predictions, per-dataset agreement.

    .COMPASS/bin/python -m validation.a2_maximum_likelihood.analyze_pipeline
"""

import validation.common.runtime  # noqa: F401

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import chi2, ncx2

from compass import information_criteria as ic
from validation.common.gaussian_oracles import wilson_interval

HERE = os.path.dirname(__file__)
CRITERIA = ("aic", "aicc", "bic")
SHRINK_FACTOR = 0.75     # 1 - (1 - c)^2 with c = 1/2


def penalty_gap(criterion, n):
    import torch
    zeros = torch.zeros(n)
    results, _ = ic.compare_criteria([zeros, zeros], criterion, [1, 0], [1, 1], 2)
    return (results[0]["total"] - results[1]["total"]).item()


def predicted_rate(gap, n, delta, factor=1.0):
    lam = n * delta ** 2
    threshold = gap / factor
    return float(ncx2.sf(threshold, 1, lam)) if lam > 0 else float(chi2.sf(threshold, 1))


def main():
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(HERE, "results", "pipeline", "*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                groups[r["n"], r["delta"]].append(r)

    lines = []
    summary = {}
    for (n, delta), recs in sorted(groups.items()):
        R = len(recs)
        lr_e = np.array([r["lr_exact"] for r in recs])
        lr_m = np.array([r["lr_mle"] for r in recs])
        lr_p = np.array([r["lr_map"] for r in recs])
        g_err = np.array([r["g_mle"] - r["g_exact_mle"] for r in recs])
        post_sd_g = math.sqrt(0.5) / math.sqrt(n)   # composed posterior sd of g (c = 1/2)
        lines.append(f"\n### N = {n}, delta = {delta:g}  ({R} datasets)\n")
        lines.append("| criterion | arm | rate M1 | 99% CI | predicted | agreement with exact |")
        lines.append("|---|---|---|---|---|---|")
        block = {}
        for c in CRITERIA:
            gap = penalty_gap(c, n)
            for arm, factor in (("exact", 1.0), ("mle", 1.0), ("map", SHRINK_FACTOR)):
                k = sum(r[f"{c}_{arm}"] for r in recs)
                lo, hi = wilson_interval(k, R)
                pred = predicted_rate(gap, n, delta, factor)
                agree = np.mean([r[f"{c}_{arm}"] == r[f"{c}_exact"] for r in recs])
                ok = "yes" if lo <= pred <= hi else "NO"
                lines.append(f"| {c} | {arm} | {k / R:.3f} | [{lo:.3f}, {hi:.3f}] | "
                             f"{pred:.3f} ({ok}) | {agree:.3f} |")
                block[f"{c}_{arm}"] = dict(rate=k / R, ci99=(lo, hi), predicted=pred,
                                           agreement=float(agree))
        d_mle, d_map = lr_m - lr_e, lr_p - lr_e
        ratio = np.median(lr_p[lr_e > 1] / lr_e[lr_e > 1]) if np.any(lr_e > 1) else float("nan")
        lines.append("")
        lines.append(f"- LR error, MLE plug-in: mean {d_mle.mean():+.4f}, sd {d_mle.std():.4f}, "
                     f"max |.| {np.abs(d_mle).max():.4f}")
        lines.append(f"- LR error, MAP plug-in: mean {d_map.mean():+.4f}, sd {d_map.std():.4f}; "
                     f"median LR_MAP / LR_exact (LR_exact > 1) = {ratio:.3f} "
                     f"(predicted {SHRINK_FACTOR})")
        lines.append(f"- g MLE error: mean {g_err.mean():+.2e}, max |.| {np.abs(g_err).max():.2e} "
                     f"({np.abs(g_err).max() / post_sd_g:.3f} posterior sd)")
        lines.append(f"- M1 log-lik error at MLE vs exact (smoothed): mean "
                     f"{np.mean([r['ll_error_m1_mle'] for r in recs]):+.4f} nats/dataset; "
                     f"M0: {np.mean([r['ll_error_m0_mle'] for r in recs]):+.4f}")
        lines.append(f"- datasets where M1's MLE touched the box: "
                     f"{sum(r['m1_at_bound'] > 0 for r in recs)}; M0: "
                     f"{sum(r['m0_at_bound'] > 0 for r in recs)}; "
                     f"rows falling back to the MAP (M1): "
                     f"{sum(r['m1_rows_using_map'] for r in recs)}")
        lines.append(f"- seconds per dataset (both models): {np.mean([r['seconds'] for r in recs]):.1f}")
        block.update(lr_error_mle_mean=float(d_mle.mean()), lr_error_mle_sd=float(d_mle.std()),
                     lr_error_map_mean=float(d_map.mean()), lr_ratio_map=float(ratio))
        summary[f"n{n}_delta{delta:g}"] = block

    text = "\n".join(lines)
    with open(os.path.join(HERE, "results", "pipeline_summary.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(HERE, "results", "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
