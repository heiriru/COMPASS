#!/usr/bin/env python3
"""Score the run_pd.py variants against the stored runs and the exact posteriors.

For every (experiment, arm, n) of compare_noproj.py: W1 to the exact posterior in exact sd for
the stored run (= row projection, as published) and each variant, and |P(global > 0) - exact|;
the share of samples whose raw Lambda_S = Lambda_prior + sum_j I_j was not PD (from each
variant's own diagnostics), the composed-repair rate and the clamp activity. Partial pooling as
in compare_noproj.pp_table. The "default" variant must reproduce the stored draws bit for bit.

    python compare_pd.py [--variants ...] -> artifacts/noproj/comparison_pd.{txt,json},
                                             figures/17_pd_strategies.png
"""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "3")
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

import compare_noproj as cn  # noqa: E402

VARIANTS = ["default", "lams", "lamsrow", "gloeckler", "consistent", "clampmetric"]
TOL_REL, TOL_ABS = 0.15, 0.02   # "same" band, as in the earlier safeguard comparisons


def diagnostics():
    """label -> (correction, n) -> mean diagnostics over that job's sample() calls."""
    table = defaultdict(lambda: defaultdict(list))
    for path in (cn.OUT / "diagnostics").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            d = json.loads(line)
            if "lambda_s_checks" not in d:
                continue
            key = (d["label"], d["correction"], d["observations"])
            evaluations = max(d["score_evaluations"], d["composed_precision_evaluations"], 1)
            table[key]["nonpd"].append(d["lambda_s_nonpd_fraction"])
            table[key]["nonpd_calls"].append(d["lambda_s_nonpd_call_fraction"])
            table[key]["repair"].append(d["repair_count"] / evaluations)
            table[key]["clamp"].append(d["clamp_active_fraction"])
            table[key]["neg"].append(d["negative_information_fraction"])
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in table.items()}


def collect(variants):
    """{(experiment, arm, n): {"ref", "stored", variant: draws, "job": {variant: label}}}."""
    rows = {}
    for v in variants:
        cn.V = v
        for experiment, arm, job, correction, n, ref, on, off in cn.rows():
            key = (experiment, arm, n)
            row = rows.setdefault(key, {"ref": ref, "stored": on, "correction": correction, "jobs": {}})
            row[v] = off
            row["jobs"][v] = job
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    variants = parser.parse_args().variants
    diag = diagnostics()
    rows = collect(variants)
    records, lines = [], []
    head = f"{'experiment':22s} {'arm':32s} {'n':>4s}  W1/sd: {'stored':>6s} " + " ".join(f"{v[:9]:>9s}" for v in variants)
    lines.append(head + "   nonPD(raw Lambda_S) per variant")
    tally = {v: [0, 0, 0] for v in variants}   # better, same, worse vs stored
    reproduce = []
    for (experiment, arm, n), row in sorted(rows.items()):
        ref = row["ref"]
        w = lambda a: float(wasserstein_distance(a, ref) / ref.std())  # noqa: E731
        rec = dict(experiment=experiment, arm=arm, n=n, p_exact=float(np.mean(ref > 0)),
                   w1={"stored": w(row["stored"])}, p_err={"stored": abs(float(np.mean(row["stored"] > 0)) - float(np.mean(ref > 0)))},
                   sd={"stored": float(row["stored"].std() / ref.std())}, diag={})
        for v in variants:
            if v not in row:
                continue
            rec["w1"][v] = w(row[v])
            rec["p_err"][v] = abs(float(np.mean(row[v] > 0)) - rec["p_exact"])
            rec["sd"][v] = float(row[v].std() / ref.std())
            rec["diag"][v] = diag.get((row["jobs"][v], row["correction"], n), {})
            if v == "default":
                same_shape = row[v].shape == row["stored"].shape
                reproduce.append((experiment, arm, n, float(np.abs(row[v] - row["stored"]).max()) if same_shape else None))
            a, b = rec["w1"]["stored"], rec["w1"][v]
            band = TOL_REL * a + TOL_ABS
            tally[v][0 if b < a - band else (2 if b > a + band else 1)] += 1
        records.append(rec)
        cells = " ".join(f"{rec['w1'].get(v, float('nan')):9.3f}" for v in variants)
        nonpd = " ".join(f"{rec['diag'].get(v, {}).get('nonpd', float('nan')):5.2f}" for v in variants)
        lines.append(f"{experiment:22s} {arm:32s} {n:4d}  {'':6s} {rec['w1']['stored']:6.3f} {cells}   {nonpd}")
    lines.append("")
    lines.append(f"vs stored (better / same / worse, same = within {TOL_REL:.0%} + {TOL_ABS}):")
    for v in variants:
        lines.append(f"  {v:12s} {tally[v][0]:3d} / {tally[v][1]:3d} / {tally[v][2]:3d}")
    lines.append("")
    lines.append("default variant vs stored draws (max |diff|; must be 0):")
    for experiment, arm, n, diff in reproduce:
        lines.append(f"  {experiment:22s} {arm:32s} {n:4d}  {diff}")
    pp = {}
    for v in variants:
        cn.V = v
        try:
            pp[v] = cn.pp_table()
        except Exception as error:  # the PP outputs may not exist yet
            pp[v] = []
            lines.append(f"  (PP {v}: {error})")
    lines.append("")
    lines.append("partial pooling C2 (mean r, global RMSE, collapses):")
    for v, table in pp.items():
        for t in table:
            if t["arm"].startswith("with") and v != variants[0]:
                continue
            d = diag.get((f"{v}_pp_" + ("hierarchical" if t["net"] == "baseline" else "hierarchical_Rg"),
                          cn.JAC, 100), {})
            extra = f"  nonPD {d['nonpd']:.3f} repair {d['repair']:.3f} clamp {d['clamp']:.4f}" if d and not t["arm"].startswith("with") else ""
            lines.append(f"  {t['net']:8s} {t['arm']:22s} mean r {t['mean_r']:.3f}  global RMSE {t['global_rmse']:.3f}  "
                         f"local {t['local_rmse']:.3f}  collapses {t['collapses']}  ({t['datasets']} datasets){extra}")
    text = "\n".join(lines)
    print(text)
    (cn.OUT / "comparison_pd.txt").write_text(text + "\n")
    (cn.OUT / "comparison_pd.json").write_text(json.dumps({"variants": variants, "rows": records, "pp": pp,
                                                            "tally": tally, "reproduce": reproduce}, indent=2) + "\n")
    figure(records, variants)


def figure(records, variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [v for v in variants if v != "default"]
    if not records or not shown:
        return
    ink, muted = "#1f1f1e", "#6b6a63"
    fig, axes = plt.subplots(1, len(shown), figsize=(3.3 * len(shown), 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    groups = sorted({r["experiment"] for r in records})
    markers = dict(zip(groups, "osD^v<>phX*P8"))
    top = max(max(r["w1"].values()) for r in records) * 1.05
    for ax, v in zip(axes, shown):
        ax.plot([0, top], [0, top], color=muted, lw=0.8, ls=":")
        for g in groups:
            pts = [(r["w1"]["stored"], r["w1"][v]) for r in records if r["experiment"] == g and v in r["w1"]]
            if pts:
                a, b = zip(*pts)
                ax.scatter(a, b, s=18, marker=markers[g], color="#eda100" if "toy" in g else "#2a78d6",
                           edgecolor=ink, linewidth=0.3, label=g)
        ax.set_title(v, color=ink, fontsize=10)
        ax.set_xlabel("stored (row projection)", color=ink, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("variant: W1 to exact / exact sd", color=ink, fontsize=8)
    axes[-1].legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.suptitle("below the diagonal: the variant is closer to the exact posterior", color=ink, fontsize=9)
    fig.tight_layout()
    path = cn.ART / "figures" / "17_pd_strategies.png"
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
