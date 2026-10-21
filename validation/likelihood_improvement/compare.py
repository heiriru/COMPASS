"""Baseline vs log-lambda time embedding: likelihood accuracy and posterior side effects.

    .COMPASS/bin/python -m validation.likelihood_improvement.compare

Reads the B6 raw results (``n65536_s<seed>[_loglam].pt``) and the posterior checks
(``results/posterior``). No network evaluation. Writes results/summary.md and figures.
"""

import validation.common.runtime  # noqa: F401

import glob
import os

import numpy as np
import torch

from validation.b5_b6_pfode_accuracy.evaluate import LAMBDA_C
from validation.common import plotting as P

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "b5_b6_pfode_accuracy", "results", "raw")
POST = os.path.join(HERE, "results", "posterior")
FIG = os.path.join(HERE, "results", "figures")
FAMS = ("gauss", "student", "bimodal")
ARMS = (("", "baseline (Bernoulli masks, raw t)", P.LEGACY),
        ("_loglam", "log-λ time input", P.FOURTH),
        ("_menu", "inference mask menu", P.EXACT),
        ("_edm", "EDM high-noise loss weight", "#4a3aa7"),
        ("_menu_edm", "menu + EDM", P.NEW),
        ("_menu_edm_ml", "menu + EDM + ML fine-tune (one network)", "#008300"),
        ("_menu_edm_h256", "menu + EDM, width 256", "#e34948"),
        ("_menu_edm_d8", "menu + EDM, depth 8", "#e87ba4"),
        ("_menu_edm_p100e1500", "menu + EDM, longer training", "#52514e"),
        ("_menu_edm@262144", "menu + EDM, 2^18 sims", "#1c5cab"))


def err(o, key):
    s = o["settings"][key]
    return (s["logp"] - o["exact"][s["eps"]]).numpy()


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def load(fam, suffix):
    budget, suffix = (int(suffix.split("@")[1]), suffix.split("@")[0]) if "@" in suffix else (65536, suffix)
    out = {}
    for seed in range(3):
        p = os.path.join(RAW, f"{fam}_nu0.5", f"n{budget}_s{seed}{suffix}.pt")
        if os.path.exists(p):
            out[seed] = torch.load(p, weights_only=False)
    return out


def load_post(fam, suffix):
    budget, suffix = (int(suffix.split("@")[1]), suffix.split("@")[0]) if "@" in suffix else (65536, suffix)
    out = {}
    for seed in range(3):
        p = os.path.join(POST, f"{fam}_s{seed}{suffix}" + ("" if budget == 65536 else f"_n{budget}") + ".pt")
        if os.path.exists(p):
            out[seed] = torch.load(p, weights_only=False)
    return out


def post_metrics(r):
    cat = lambda k: torch.cat([r[f"{k}_0"], r[f"{k}_1"]]).double()
    return {"mean_err": float(cat("mean_err_sd").abs().mean()),
            "sd_ratio_dev": float((cat("sd_ratio") - 1).abs().mean()),
            "w1": float(cat("w1_sd").mean()), "w1_floor": float(cat("floor_w1_sd").mean()),
            "density_rmse": rmse(r["density_err"].numpy())}


def seed_swap_sd(runs):
    """sd of per-row log p differences between seeds on own data (C9 swap test)."""
    keys = sorted(runs)
    diffs = [runs[a]["settings"]["steps_100"]["logp"].numpy()
             - runs[b]["settings"]["steps_100"]["logp"].numpy()
             for i, a in enumerate(keys) for b in keys[i + 1:]]
    return float(np.std(np.concatenate(diffs))) if diffs else float("nan")


def main():
    P.style()
    import matplotlib.pyplot as plt
    lines = ["| family | arm | seeds | log p RMSE (mean ± sd) | bias | seed-swap sd/row | "
             "posterior mean err (sd) | posterior sd-ratio dev | W1 (sd) [floor] | "
             "posterior density RMSE |", "|" + "---|" * 10]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for fi, fam in enumerate(FAMS):
        for suffix, label, color in ARMS:
            runs = load(fam, suffix)
            posts = load_post(fam, suffix)
            if not runs:
                continue
            r = [rmse(err(o, "steps_100")) for o in runs.values()]
            b = np.mean([err(o, "steps_100").mean() for o in runs.values()])
            pm = [post_metrics(p) for p in posts.values()]
            pmv = (lambda k: np.mean([m[k] for m in pm]) if pm else float("nan"))
            lines.append(
                f"| {fam} | {label} | {len(runs)} | {np.mean(r):.3f} ± {np.std(r):.3f} | "
                f"{b:+.3f} | {seed_swap_sd(runs):.3f} | {pmv('mean_err'):.3f} | "
                f"{pmv('sd_ratio_dev'):.3f} | {pmv('w1'):.3f} [{pmv('w1_floor'):.3f}] | "
                f"{pmv('density_rmse'):.3f} |")
            lam = [0.0317 if lc == 0 else (9.8 if lc > 100 else lc) for lc in LAMBDA_C]
            curves = np.array([[rmse(err(o, f"hybrid_{lc:g}")) for lc in LAMBDA_C]
                               for o in runs.values()])
            ax = axes[fi]
            ax.plot(lam, curves.mean(0), "o-", color=color, label=label)
            ax.fill_between(lam, curves.min(0), curves.max(0), color=color, alpha=0.15)
        axes[fi].set_xscale("log"); axes[fi].set_yscale("log")
        axes[fi].set_title(f"({'abc'[fi]}) {fam}, 2^16 sims", loc="left")
        axes[fi].set_xlabel("λ_c: trained network only above λ_c")
        axes[fi].legend(fontsize=8)
    axes[0].set_ylabel("RMSE of log p per row (nats)")
    os.makedirs(FIG, exist_ok=True)
    path = P.save(fig, FIG, "fig_bands_all_arms")

    # Posterior side effects: per-network W1 and density RMSE, both arms.
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.0))
    for fi, fam in enumerate(FAMS):
        for ai, (suffix, label, color) in enumerate(ARMS):
            pm = [post_metrics(p) for p in load_post(fam, suffix).values()]
            if not pm:
                continue
            xs = fi + (ai - 4.5) * 0.08
            ax2[0].scatter([xs] * len(pm), [m["w1"] for m in pm], color=color, s=30,
                           label=label if fi == 0 else None)
            ax2[0].scatter([xs], [np.mean([m["w1_floor"] for m in pm])], marker="_", s=300,
                           color=P.INK2, label="exact-draw floor" if fi == 0 and ai == 0 else None)
            ax2[1].scatter([xs] * len(pm), [m["density_rmse"] for m in pm], color=color, s=30,
                           label=label if fi == 0 else None)
    for a, yl, t in ((ax2[0], "posterior marginal W1 (posterior sds)", "(a) posterior samples"),
                     (ax2[1], "PF-ODE posterior log-density RMSE (nats)", "(b) posterior density")):
        a.set_xticks(range(3)); a.set_xticklabels(FAMS); a.set_ylabel(yl)
        a.set_title(t, loc="left"); a.legend(fontsize=8)
    path2 = P.save(fig2, FIG, "fig_posterior_side_effects")
    text = "\n".join(lines)
    with open(os.path.join(HERE, "results", "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(path, path2, sep="\n")


if __name__ == "__main__":
    main()
