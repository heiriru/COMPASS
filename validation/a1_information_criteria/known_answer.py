"""E-A1.2 / E-A1.3 / E-A1.4: selection rates of the criteria against closed-form predictions.

Level (a) of the validation ladder: exact likelihoods, exact MLE/MAP, and the
*library* criterion code (compass.information_criteria) -- no network, no sampler.

Nested hierarchical pair (TODO.md E-A1.2/E-A1.3), rows x_i = (x1_i, x2_i):
    M1: x1 = l_i + nu e,  x2 = g + nu e      params: global g, local l_i
    M0: x1 = l_i + nu e,  x2 = nu e          params: local l_i
With exact MLEs the likelihood-ratio statistic LR = N xbar2^2 / nu^2 is exactly
chi^2_1 under g = 0 (noncentral with lambda = N delta^2 / nu^2 under g = delta),
so the rate at which a criterion picks M1 is P(LR > penalty(M1) - penalty(M0)).
With a MAP plug-in that shrinks g by c = (N/nu^2) / (N/nu^2 + 1/s_g^2) the
statistic is (1 - (1 - c)^2) LR and the rate drops accordingly.

Flat pair (E-A1.4): every row has its own parameters,
    M1f: x = (a_i + nu e, b_i + nu e),  M0f: x = (a_i + nu e, nu e),
and LR = sum_i x2_i^2 / nu^2 ~ chi^2_N.

Run from the repository root:
    .COMPASS/bin/python -m validation.a1_information_criteria.known_answer
"""

import validation.common.runtime  # noqa: F401  (CPU cap before numeric imports)

import json
import math
import os

import numpy as np
import torch
from scipy.stats import chi2, ncx2

from compass import information_criteria as ic
from validation.common.gaussian_oracles import wilson_interval

OUT = os.path.join(os.path.dirname(__file__), "results")
NU = 1.0
N_PER_OBS = 2          # scalars per row
REPLICATES = 10000
NS = (4, 16, 64, 256)


def gauss_logpdf(x, mean):
    return -0.5 * math.log(2 * math.pi * NU ** 2) - 0.5 * (x - mean) ** 2 / NU ** 2


def legacy_total_penalty(criterion, k, n_rows):
    """The historical ModelTransfuser._aicc/_bic, per row with the full latent k, summed."""
    if criterion == "bic":
        return n_rows * k * math.log(n_rows)
    if criterion == "aic_plain":
        return n_rows * 2 * k
    denom = n_rows - k - 1
    per_row = 2 * k + ((2 * k ** 2 + 2 * k) / denom if denom > 0 else 0.0)
    return n_rows * per_row


def counted_totals(ll1, ll0, criterion, k1, k0):
    """Counted criteria of (M1, M0), decided jointly as in ModelTransfuser.compare."""
    results, _ = ic.compare_criteria([torch.as_tensor(ll1), torch.as_tensor(ll0)],
                                     criterion, [k1[0], k0[0]], [k1[1], k0[1]], N_PER_OBS)
    return results[0]["total"].item(), results[1]["total"].item()


def counted_penalty_gap(criterion, n, k1, k0):
    zeros = np.zeros(n)
    ic1, ic0 = counted_totals(zeros, zeros, criterion, k1, k0)
    return ic1 - ic0


def hierarchical_rows(rng, n, delta):
    x1 = rng.normal(0.0, NU, n)
    x2 = rng.normal(delta, NU, n)
    return x1, x2


def run_hierarchical(n, delta, shrink, rng):
    """Selection counts for M1 under every criterion/convention, plus predictions."""
    criteria = {"aic": ("aic", "aic_plain"), "aicc": ("aicc", "aicc"), "bic": ("bic", "bic")}
    counts = {(conv, name): 0 for conv in ("counted", "legacy") for name in criteria}
    for _ in range(REPLICATES):
        x1, x2 = hierarchical_rows(rng, n, delta)
        g_hat = shrink * x2.mean()            # shrink = 1 is the MLE
        # Locals are fitted identically in both models; with the MLE l_i = x1_i.
        ll_local = gauss_logpdf(x1, x1)
        ll1 = ll_local + gauss_logpdf(x2, g_hat)
        ll0 = ll_local + gauss_logpdf(x2, 0.0)
        for name, (counted_name, legacy_name) in criteria.items():
            ic1, ic0 = counted_totals(ll1, ll0, counted_name, (1, 1), (0, 1))
            counts["counted", name] += ic1 < ic0
            leg1 = -2 * ll1.sum() + legacy_total_penalty(legacy_name, 2, n)
            leg0 = -2 * ll0.sum() + legacy_total_penalty(legacy_name, 1, n)
            counts["legacy", name] += leg1 < leg0

    rows = []
    factor = 1 - (1 - shrink) ** 2
    lam = n * delta ** 2 / NU ** 2
    for (conv, name), count in counts.items():
        if conv == "counted":
            pen1, pen0 = counted_penalty_gap(criteria[name][0], n, (1, 1), (0, 1)), 0.0
        else:
            pen1 = legacy_total_penalty(criteria[name][1], 2, n)
            pen0 = legacy_total_penalty(criteria[name][1], 1, n)
        threshold = (pen1 - pen0) / factor
        predicted = float(ncx2.sf(threshold, 1, lam)) if lam > 0 else float(chi2.sf(threshold, 1))
        lo, hi = wilson_interval(count, REPLICATES)
        rows.append(dict(experiment="hierarchical", n=n, delta=delta, shrink=shrink,
                         convention=conv, criterion=name, penalty_gap=pen1 - pen0,
                         rate=int(count) / REPLICATES, ci99=(lo, hi), predicted=predicted,
                         consistent=bool(lo <= predicted <= hi)))
    return rows


def run_flat(n, rng):
    criteria = {"aic": ("aic", "aic_plain"), "aicc": ("aicc", "aicc"), "bic": ("bic", "bic")}
    counts = {(conv, name): 0 for conv in ("counted", "legacy") for name in criteria}
    for _ in range(REPLICATES):
        x1 = rng.normal(0.0, NU, n)
        x2 = rng.normal(0.0, NU, n)
        ll1 = gauss_logpdf(x1, x1) + gauss_logpdf(x2, x2)
        ll0 = gauss_logpdf(x1, x1) + gauss_logpdf(x2, 0.0)
        for name, (counted_name, legacy_name) in criteria.items():
            ic1, ic0 = counted_totals(ll1, ll0, counted_name, (0, 2), (0, 1))
            counts["counted", name] += ic1 < ic0
            leg1 = -2 * ll1.sum() + legacy_total_penalty(legacy_name, 2, n)
            leg0 = -2 * ll0.sum() + legacy_total_penalty(legacy_name, 1, n)
            counts["legacy", name] += leg1 < leg0
    rows = []
    for (conv, name), count in counts.items():
        if conv == "counted":
            pen1, pen0 = counted_penalty_gap(criteria[name][0], n, (0, 2), (0, 1)), 0.0
        else:
            pen1 = legacy_total_penalty(criteria[name][1], 2, n)
            pen0 = legacy_total_penalty(criteria[name][1], 1, n)
        predicted = float(chi2.sf(pen1 - pen0, n))
        lo, hi = wilson_interval(count, REPLICATES)
        rows.append(dict(experiment="flat", n=n, delta=0.0, shrink=1.0, convention=conv,
                         criterion=name, penalty_gap=pen1 - pen0, rate=int(count) / REPLICATES,
                         ci99=(lo, hi), predicted=predicted,
                         consistent=bool(lo <= predicted <= hi)))
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    suffix = "" if args.seed == 20261013 else f"_seed{args.seed}"
    rows = []
    for n in NS:
        rows += run_hierarchical(n, 0.0, 1.0, rng)              # E-A1.2, exact MLE
        rows += run_hierarchical(n, 0.0, 0.5, rng)              # MAP plug-in, c = 0.5
        rows += run_hierarchical(n, 2.0 / math.sqrt(n), 1.0, rng)  # E-A1.3 power, lambda = 4
        rows += run_flat(n, rng)                                 # E-A1.4
    with open(os.path.join(OUT, f"known_answer{suffix}.json"), "w") as f:
        json.dump(rows, f, indent=1)

    lines = ["| experiment | N | delta | plug-in | convention | criterion | penalty gap | "
             "rate M1 | 99% CI | predicted | ok |", "|" + "---|" * 11]
    for r in rows:
        plug = "MLE" if r["shrink"] == 1.0 else f"MAP c={r['shrink']}"
        lines.append(f"| {r['experiment']} | {r['n']} | {r['delta']:.3f} | {plug} | "
                     f"{r['convention']} | {r['criterion']} | {r['penalty_gap']:.2f} | "
                     f"{r['rate']:.4f} | [{r['ci99'][0]:.4f}, {r['ci99'][1]:.4f}] | "
                     f"{r['predicted']:.4f} | {'yes' if r['consistent'] else 'NO'} |")
    table = "\n".join(lines)
    with open(os.path.join(OUT, f"known_answer{suffix}.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    bad = [r for r in rows if not r["consistent"]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} rates inside the 99% interval of the prediction.")


if __name__ == "__main__":
    main()
