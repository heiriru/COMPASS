"""C9: can network seeds alone produce a model-comparison verdict?

    .COMPASS/bin/python -m validation.c9_network_seeds.analyze_swap

Uses the B5/B6 cross-evaluations (no new network evaluation): for every trained
network (3 families x 3 budgets x 3 seeds), the PF-ODE log p(x_i | theta_i) at the true
theta on the 512 test rows of each family (isotropic endpoint, 100 steps).

E-C9.3 swap test. Two seeds of the same family are identical models with different
names. Datasets = disjoint chunks of N rows. With equal parameter counts the Akaike
weight of seed a is sigmoid(sum_i [log p_a - log p_b]); the right answer is 0.5.
Measured: spread of the dataset log-likelihood difference, the fraction of datasets
with a confident (> 0.9 or < 0.1) verdict, and how both scale with N. Independent
per-row noise grows like sqrt(N); a per-seed systematic offset grows like N and makes
identical models look decisively different at large N.

E-C9.1 seed spread vs model gap. gauss vs student models, with all 3 x 3 seed pairs:
the seed spread of the dataset Delta logL against the exact Delta. The smallest
resolvable gap is 3x the seed sd.

Figures and tables go to results/.
"""

import validation.common.runtime  # noqa: F401

import glob
import itertools
import os

import numpy as np
import torch

from validation.b5_b6_pfode_accuracy.families import Family
from validation.common import plotting as P

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "b5_b6_pfode_accuracy", "results", "raw")
FIG = os.path.join(HERE, "results", "figures")
FAMS = ("gauss", "student", "bimodal")
BUDGETS = (4096, 16384, 65536)
NS = (4, 8, 16, 32, 64)        # N = 256 would leave only 2 datasets per seed pair
SIG = float(np.sqrt((25.0 ** (2e-3) - 1) / (2 * np.log(25.0))))
COL = {"gauss": P.NEW, "student": P.LEGACY, "bimodal": P.EXACT}


def load_cross(endpoint="isotropic", key="cross"):
    """cross[(model_family, budget, seed)][data_family] = per-row log p (512,)."""
    out, xs = {}, {}
    for fam in FAMS:
        for path in glob.glob(os.path.join(RAW, f"{fam}_nu0.5", "n*_s?.pt")):
            name = os.path.basename(path)[1:-3]
            budget, seed = map(int, name.split("_s"))
            o = torch.load(path, weights_only=False)
            if key not in o:
                continue
            out[(fam, budget, seed)] = {d: o[key][d][endpoint].numpy() for d in FAMS}
            for d in FAMS:
                xs.setdefault(d, o[key][d]["x"])
                assert torch.equal(xs[d], o[key][d]["x"]), "test rows differ"
    return out


def chunk_sums(v, n):
    m = len(v) // n * n
    return v[:m].reshape(-1, n).sum(1)


def swap_table(cross):
    rows = []
    for fam in FAMS:
        for budget in BUDGETS:
            for data in FAMS:
                for n in NS:
                    diffs = []
                    for a, b in itertools.combinations(range(3), 2):
                        ka, kb = (fam, budget, a), (fam, budget, b)
                        if ka in cross and kb in cross:
                            diffs.append(chunk_sums(cross[ka][data] - cross[kb][data], n))
                    if not diffs:
                        continue
                    d = np.concatenate(diffs)
                    w = 1 / (1 + np.exp(-np.clip(d, -50, 50)))
                    per_row = np.concatenate([cross[(fam, budget, a)][data]
                                              - cross[(fam, budget, b)][data]
                                              for a, b in itertools.combinations(range(3), 2)])
                    rows.append(dict(model=fam, budget=budget, data=data, n=n,
                                     sd=float(d.std()), mean_abs_offset=float(abs(per_row.mean()) * n),
                                     confident=float(np.mean((w > 0.9) | (w < 0.1))),
                                     row_sd=float(per_row.std()), row_mean=float(per_row.mean())))
    return rows


def gap_table(cross):
    """gauss vs student: seed spread of dataset Delta logL vs exact Delta (true theta)."""
    rows = []
    for data in ("gauss", "student"):
        fam = Family(data, 0.5)
        o = torch.load(glob.glob(os.path.join(RAW, f"{data}_nu0.5", "n65536_s0.pt"))[0],
                       weights_only=False)
        th, x = o["cross"][data]["theta"], o["cross"][data]["x"]
        exact = (Family("gauss", 0.5).log_lik(th, x, SIG)
                 - Family("student", 0.5).log_lik(th, x, SIG)).numpy()
        for budget in BUDGETS:
            for n in NS:
                ex = chunk_sums(exact, n)
                nets = np.stack([chunk_sums(cross[("gauss", budget, a)][data]
                                            - cross[("student", budget, b)][data], n)
                                 for a in range(3) for b in range(3)])     # (9, datasets)
                seed_sd = nets.std(0, ddof=1).mean()
                err = nets.mean(0) - ex
                decided = np.abs(ex) > 3 * seed_sd
                agree = np.mean(np.sign(nets) == np.sign(ex)[None, :])
                rows.append(dict(data=data, budget=budget, n=n, seed_sd=float(seed_sd),
                                 exact_abs_median=float(np.median(np.abs(ex))),
                                 bias=float(err.mean()), resolvable=3 * float(seed_sd),
                                 frac_resolved=float(decided.mean()),
                                 sign_agreement=float(agree)))
    return rows


def checkpoint_table(final, best):
    """E-C9.2: final-epoch vs best-validation checkpoint, own-family data, true theta."""
    rows = []
    for fam in FAMS:
        th = x = None
        o = torch.load(os.path.join(RAW, f"{fam}_nu0.5", "n65536_s0.pt"), weights_only=False)
        th, x = o["cross"][fam]["theta"], o["cross"][fam]["x"]
        exact = Family(fam, 0.5).log_lik(th, x, SIG).numpy()
        for budget in BUDGETS:
            for label, src in (("final epoch", final), ("best validation", best)):
                runs = [src[(fam, budget, s)][fam] for s in range(3) if (fam, budget, s) in src]
                if len(runs) < 3:
                    continue
                rm = [float(np.sqrt(np.mean((r - exact) ** 2))) for r in runs]
                swap = np.concatenate([runs[a] - runs[b] for a, b in
                                       itertools.combinations(range(3), 2)])
                rows.append(dict(family=fam, budget=budget, rule=label, rmse=np.mean(rm),
                                 rmse_sd=np.std(rm), swap_sd=float(swap.std())))
    return rows


def figures(swap, gap):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for fam in FAMS:
        for data, ls in ((fam, "-"), ("other", "--")):
            pick = [r for r in swap if r["model"] == fam and r["budget"] == max(BUDGETS)
                    and ((r["data"] == fam) if data == fam else (r["data"] != fam))]
            if not pick:
                continue
            ns = sorted(set(r["n"] for r in pick))
            sd = [np.mean([r["sd"] for r in pick if r["n"] == n]) for n in ns]
            conf = [np.mean([r["confident"] for r in pick if r["n"] == n]) for n in ns]
            lab = f"{fam} nets, " + ("own data" if data == fam else "other families' data")
            axes[0].plot(ns, sd, "o" + ls, color=COL[fam], label=lab)
            axes[1].plot(ns, conf, "o" + ls, color=COL[fam], label=lab)
    ref = np.array(NS, dtype=float)
    base = [r for r in swap if r["model"] == "gauss" and r["budget"] == max(BUDGETS)
            and r["data"] == "gauss" and r["n"] == NS[0]][0]["sd"]
    axes[0].plot(ref, base * np.sqrt(ref / NS[0]), color=P.INK2, lw=1, ls=":",
                 label="√N growth (independent rows)")
    axes[0].plot(ref, base * ref / NS[0], color=P.INK2, lw=1, ls="-.",
                 label="N growth (systematic offset)")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("rows per dataset N"); axes[0].set_ylabel("sd of Δ log L between seeds (nats)")
    axes[0].set_title("(a) identical models, different seeds (2^16)", loc="left", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log", base=2); axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("rows per dataset N")
    axes[1].set_ylabel("fraction of datasets with weight > 0.9 or < 0.1")
    axes[1].set_title("(b) spurious confident verdicts (right answer: 0)", loc="left", fontsize=10)
    for data, c in (("gauss", P.NEW), ("student", P.LEGACY)):
        for budget, mk in zip(BUDGETS, ("s", "D", "o")):
            pick = [r for r in gap if r["data"] == data and r["budget"] == budget]
            axes[2].plot([r["n"] for r in pick], [r["resolvable"] for r in pick], mk + "-",
                         color=c, alpha=0.4 + 0.3 * BUDGETS.index(budget),
                         label=f"{data} data, 2^{int(np.log2(budget))}")
        pick = [r for r in gap if r["data"] == data and r["budget"] == max(BUDGETS)]
        axes[2].plot([r["n"] for r in pick], [r["exact_abs_median"] for r in pick], "*:",
                     color=c, ms=10, label=f"{data} data: median |exact Δ|")
    axes[2].set_xscale("log", base=2); axes[2].set_yscale("log")
    axes[2].set_xlabel("rows per dataset N")
    axes[2].set_ylabel("smallest resolvable Δ log L = 3 × seed sd (nats)")
    axes[2].set_title("(c) gauss vs student: seed noise vs real gap", loc="left", fontsize=10)
    axes[2].legend(fontsize=6.5, ncol=2)
    return P.save(fig, FIG, "fig_swap_and_gap")


def main():
    P.style()
    cross = load_cross()
    swap, gap = swap_table(cross), gap_table(cross)
    ckpt = checkpoint_table(cross, load_cross(key="cross_best"))
    lines = ["## E-C9.3 swap test (identical models, different seeds; true θ)\n",
             "| model | budget | data | N | sd of dataset Δ | per-row Δ mean (offset) | "
             "per-row Δ sd | confident verdicts |", "|" + "---|" * 8]
    for r in swap:
        if r["n"] in (16, 64):
            lines.append(f"| {r['model']} | {r['budget']} | {r['data']} | {r['n']} | "
                         f"{r['sd']:.2f} | {r['row_mean']:+.3f} | {r['row_sd']:.3f} | "
                         f"{100 * r['confident']:.0f}% |")
    lines += ["\n## E-C9.1 gauss vs student: seed spread vs real gap\n",
              "| data | budget | N | seed sd | resolvable (3 sd) | median exact |Δ| | "
              "datasets resolved | sign agreement (all seed pairs) | mean bias |",
              "|" + "---|" * 9]
    for r in gap:
        lines.append(f"| {r['data']} | {r['budget']} | {r['n']} | {r['seed_sd']:.2f} | "
                     f"{r['resolvable']:.2f} | {r['exact_abs_median']:.2f} | "
                     f"{100 * r['frac_resolved']:.0f}% | {100 * r['sign_agreement']:.1f}% | "
                     f"{r['bias']:+.2f} |")
    lines += ["\n## E-C9.2 checkpoint rule (own-family data, true θ)\n",
              "| family | budget | rule | log p RMSE (mean ± sd over seeds) | seed-swap sd/row |",
              "|" + "---|" * 5]
    for r in ckpt:
        lines.append(f"| {r['family']} | {r['budget']} | {r['rule']} | {r['rmse']:.3f} ± "
                     f"{r['rmse_sd']:.3f} | {r['swap_sd']:.3f} |")
    text = "\n".join(lines)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "swap_and_gap.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(figures(swap, gap))


if __name__ == "__main__":
    main()
