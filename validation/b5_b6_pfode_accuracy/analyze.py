"""B5/B6 analysis and figures from results/raw (no network evaluation).

    .COMPASS/bin/python -m validation.b5_b6_pfode_accuracy.analyze

Error of a setting = PF-ODE log p(x | theta) - exact log p smoothed by sigma(eps), per
test point (3 observed dims). Tables -> results/summary.md, figures -> results/figures/.
"""

import validation.common.runtime  # noqa: F401

import glob
import os
from collections import defaultdict

import numpy as np
import torch

from validation.b5_b6_pfode_accuracy.evaluate import test_set
from validation.b5_b6_pfode_accuracy.families import Family
from validation.common import plotting as P

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "results", "raw")
FIG = os.path.join(HERE, "results", "figures")
FAM_COLORS = {"gauss": P.NEW, "student": P.LEGACY, "bimodal": P.EXACT}
BUDGETS = (4096, 16384, 65536)


def load():
    nets, oracles = defaultdict(dict), {}
    for path in glob.glob(os.path.join(RAW, "*", "*.pt")):
        tag = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)[:-3]
        obj = torch.load(path, weights_only=False)
        if name == "oracle":
            oracles[tag] = obj
        else:
            budget, seed = name[1:].split("_s")
            nets[tag][(int(budget), int(seed))] = obj
    return nets, oracles


def err(obj, key):
    s = obj["settings"][key]
    return (s["logp"] - obj["exact"][s["eps"]]).numpy()


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def table(nets, oracles):
    lines = ["| family | budget | seeds | bias (nats/row) | RMSE (isotropic endpoint, default) "
             "| RMSE fitted-Gaussian endpoint | RMSE with exact endpoint only | "
             "oracle floor RMSE |", "|" + "---|" * 8]
    for tag in sorted(nets):
        floor = rmse(err(oracles[tag], "steps_100")) if tag in oracles else float("nan")
        for b in BUDGETS:
            runs = [o for (bb, _), o in nets[tag].items() if bb == b]
            if not runs:
                continue
            bias = np.mean([err(o, "steps_100").mean() for o in runs])
            r = [rmse(err(o, "steps_100")) for o in runs]
            ri = [rmse(err(o, "endpoint_gaussian")) for o in runs]
            ro = [rmse(err(o, "oracle_endpoint_isotropic")) for o in runs
                  if "oracle_endpoint_isotropic" in o["settings"]]
            lines.append(f"| {tag} | {b} | {len(runs)} | {bias:+.3f} | {np.mean(r):.3f} ± "
                         f"{np.std(r):.3f} | {np.mean(ri):.3f} | "
                         f"{np.mean(ro) if ro else float('nan'):.3f} | {floor:.4f} |")
    return "\n".join(lines)


def fig_budget(nets, oracles):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for tag in sorted(t for t in nets if t.endswith("nu0.5")):
        fam = tag.split("_")[0]
        for ax, key, lab in ((axes[0], "steps_100", "isotropic endpoint (default)"),
                             (axes[1], "endpoint_gaussian", "fitted-Gaussian endpoint")):
            xs, ys = [], []
            for (b, s), o in nets[tag].items():
                xs.append(b), ys.append(rmse(err(o, key)))
            ax.scatter(xs, ys, color=FAM_COLORS[fam], s=22, alpha=0.7)
            means = [np.mean([y for x_, y in zip(xs, ys) if x_ == b]) for b in BUDGETS
                     if b in xs]
            ax.plot([b for b in BUDGETS if b in xs], means, "-", color=FAM_COLORS[fam], label=fam)
            if tag in oracles:
                ax.axhline(rmse(err(oracles[tag], key)),
                           color=FAM_COLORS[fam], ls=":", lw=1.2)
            ax.set_title(f"({'ab'[key != 'steps_100']}) {lab}", loc="left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("training simulations")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("RMSE of log p(x | θ) per row (nats)")
    fig.text(0.5, -0.03, "points: individual seeds; lines: seed mean; dotted: exact-score "
             "oracle through the same PF-ODE (solver floor)", ha="center", color=P.INK2,
             fontsize=8.5)
    return P.save(fig, FIG, "fig_error_vs_budget")


def fig_bands(nets):
    import matplotlib.pyplot as plt
    from validation.b5_b6_pfode_accuracy.evaluate import LAMBDA_C
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    lam_plot = [0.0317 if lc == 0 else (9.8 if lc > 100 else lc) for lc in LAMBDA_C]
    for tag in sorted(t for t in nets if t.endswith("nu0.5")):
        fam = tag.split("_")[0]
        runs = [o for (b, _), o in nets[tag].items() if b == max(BUDGETS)]
        if not runs:
            continue
        curves = np.array([[rmse(err(o, f"hybrid_{lc:g}")) for lc in LAMBDA_C] for o in runs])
        ax.plot(lam_plot, curves.mean(0), "o-", color=FAM_COLORS[fam], label=fam)
        ax.fill_between(lam_plot, curves.min(0), curves.max(0), color=FAM_COLORS[fam], alpha=0.15)
        if "oracle_endpoint_isotropic" in runs[0]["settings"]:
            oe = np.mean([rmse(err(o, "oracle_endpoint_isotropic")) for o in runs])
            ax.scatter([0.0317], [oe], marker="*", s=140, color=FAM_COLORS[fam],
                       edgecolors=P.INK, linewidths=0.6, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("λ_c: trained network used only for noise λ > λ_c (oracle below)")
    ax.set_ylabel("RMSE of log p per row (nats)")
    ax.set_title("Which noise band carries the network's density error? (2^16 sims)",
                 loc="left", fontsize=10)
    ax.legend(fontsize=9, title="band: seed min–max", title_fontsize=8)
    ax.text(0.035, ax.get_ylim()[0] * 1.3, "★ trained everywhere except the t = 1 endpoint",
            fontsize=8, color=P.INK2)
    return P.save(fig, FIG, "fig_band_attribution")


def fig_solver(nets, oracles):
    import matplotlib.pyplot as plt
    from validation.b5_b6_pfode_accuracy.evaluate import STEPS, PROBES, EPS
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for tag in sorted(t for t in nets if t.endswith("nu0.5")):
        fam = tag.split("_")[0]
        runs = [o for (b, _), o in nets[tag].items() if b == max(BUDGETS)]
        if not runs:
            continue
        c = FAM_COLORS[fam]
        y = [np.mean([rmse(err(o, f"steps_{T}")) for o in runs]) for T in STEPS]
        axes[0].plot(STEPS, y, "o-", color=c, label=f"{fam} network")
        if tag in oracles:
            axes[0].plot(STEPS, [rmse(err(oracles[tag], f"steps_{T}")) for T in STEPS], "o:",
                         color=c, alpha=0.6, label=f"{fam} oracle")
        y = [np.mean([rmse(err(o, f"hutch_{k}")) for o in runs]) for k in PROBES]
        axes[1].plot(PROBES, y, "o-", color=c, label=fam)
        axes[1].axhline(np.mean([rmse(err(o, "steps_100")) for o in runs]), color=c, ls="--",
                        lw=1)
        y = [np.mean([np.mean(err(o, f"eps_{e:g}")) for o in runs]) for e in EPS]
        axes[2].plot(EPS, y, "o-", color=c, label=f"{fam} network")
        if tag in oracles:
            axes[2].plot(EPS, [np.mean(err(oracles[tag], f"eps_{e:g}")) for e in EPS], "o:",
                         color=c, alpha=0.6)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("PF-ODE steps"); axes[0].set_ylabel("RMSE per row (nats)")
    axes[0].set_title("(a) solver steps", loc="left")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Hutchinson probes (dashed: exact divergence)")
    axes[1].set_title("(b) divergence estimator", loc="left")
    axes[2].set_xscale("log"); axes[2].axhline(0, color=P.INK2, lw=1)
    axes[2].axvline(1e-3, color=P.GRID, lw=6, zorder=0)
    axes[2].set_xlabel("eps (grey: trained floor; 3e-4 extrapolates)")
    axes[2].set_ylabel("mean error vs exact smoothed (nats)")
    axes[2].set_title("(c) end time eps (dotted: oracle)", loc="left")
    for ax in axes:
        ax.legend(fontsize=7.5)
    return P.save(fig, FIG, "fig_solver_factors")


def fig_scatter(nets):
    import matplotlib.pyplot as plt
    tags = [t for t in sorted(nets) if t.endswith("nu0.5")]
    fig, axes = plt.subplots(1, len(tags), figsize=(4.4 * len(tags), 4.2))
    for ax, tag in zip(np.atleast_1d(axes), tags):
        fam = tag.split("_")[0]
        o = nets[tag].get((max(BUDGETS), 0))
        if o is None:
            continue
        ex = o["exact"][1e-3].numpy()
        for key, c, lab in (("endpoint_gaussian", P.LEGACY, "fitted-Gaussian endpoint"),
                            ("steps_100", P.NEW, "isotropic endpoint (default)")):
            ax.scatter(ex, o["settings"][key]["logp"].numpy(), s=8, alpha=0.5, color=c,
                       label=lab, edgecolors="none")
        lo, hi = ex.min(), ex.max()
        ax.plot([lo, hi], [lo, hi], color=P.INK2, lw=1)
        ax.set_xlabel("exact log p(x | θ)")
        ax.set_title(f"{fam} (2^16 sims, seed 0)", loc="left")
    np.atleast_1d(axes)[0].set_ylabel("PF-ODE log p(x | θ)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    return P.save(fig, FIG, "fig_scatter_per_row")


def model_difference(nets):
    """B6.3: error of per-row log p_i - log p_j against the exact difference."""
    import matplotlib.pyplot as plt
    pairs = [("gauss", "student"), ("gauss", "bimodal")]
    lines = ["| pair | data | budget | endpoint | exact Δ sd/row | Δ error mean | Δ error sd | "
             "N=16 sum: sign agreement |", "|" + "---|" * 8]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sm = None
    for ax, (a, b) in zip(axes, pairs):
        ta, tb = f"{a}_nu0.5", f"{b}_nu0.5"
        for data_fam in (a, b):
            th, x = test_set(Family(data_fam, 0.5))
            if sm is None:      # sigma(eps = 1e-3) of the VESDE with sigma = 25
                sm = float(np.sqrt((25.0 ** (2e-3) - 1) / (2 * np.log(25.0))))
            ex = (Family(a, 0.5).log_lik(th, x, sm) - Family(b, 0.5).log_lik(th, x, sm)).numpy()
            # Grade only against the very rows each network saw (stored with the result).
            for o in list(nets[ta].values()) + list(nets[tb].values()):
                if "cross" in o:
                    stored = o["cross"][data_fam].get("x")
                    if stored is None or not torch.equal(stored, x):
                        raise RuntimeError(f"cross-evaluation rows differ from the test set "
                                           f"({data_fam}); re-run evaluate --cross-only")
            for budget in BUDGETS:
                for ep in ("gaussian", "isotropic"):
                    errs, agree = [], []
                    for seed in range(3):
                        oa, ob = nets[ta].get((budget, seed)), nets[tb].get((budget, seed))
                        if oa is None or ob is None or "cross" not in oa:
                            continue
                        net = (oa["cross"][data_fam][ep] - ob["cross"][data_fam][ep]).numpy()
                        errs.append(net - ex)
                        S = len(ex) // 16 * 16
                        agree.append(np.mean(np.sign(net[:S].reshape(-1, 16).sum(1))
                                             == np.sign(ex[:S].reshape(-1, 16).sum(1))))
                        if budget == max(BUDGETS) and ep == "isotropic" and seed == 0:
                            ax.scatter(ex, net, s=8, alpha=0.5, edgecolors="none",
                                       color=FAM_COLORS[data_fam], label=f"data from {data_fam}")
                    if errs:
                        e = np.concatenate(errs)
                        lines.append(f"| {a} vs {b} | {data_fam} | {budget} | {ep} | {ex.std():.3f} "
                                     f"| {e.mean():+.3f} | {e.std():.3f} | {np.mean(agree):.3f} |")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        lim = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.plot(lim, lim, color=P.INK2, lw=1, label="y = x")
        ax.set_xlabel(f"exact log p_{a} − log p_{b} per row")
        ax.set_title(f"({'ab'[pairs.index((a, b))]}) {a} vs {b} model (2^16, seed 0, "
                     "isotropic endpoint)", loc="left", fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("network difference per row (symlog)")
    path = P.save(fig, FIG, "fig_model_difference")
    return "\n".join(lines), path


def fig_noise_sweep(nets):
    """E-B5.3: smoothing bias and network error vs nu / sigma(eps)."""
    import matplotlib.pyplot as plt
    sig = float(np.sqrt((25.0 ** (2e-3) - 1) / (2 * np.log(25.0))))
    rows = []
    for tag in sorted(nets):
        if not tag.startswith("gauss_"):
            continue
        nu = float(tag.split("nu")[1])
        o = nets[tag].get((max(BUDGETS), 0))
        if o is None:
            continue
        bias = float((o["exact"][1e-3] - o["exact_unsmoothed"]).mean())
        net = float(np.mean(o["settings"]["steps_100"]["logp"].numpy()
                            - o["exact_unsmoothed"].numpy()))
        rows.append((nu, bias, rmse(err(o, "steps_100")), net))
    rows.sort()
    nu = np.array([r[0] for r in rows])
    grid = np.logspace(np.log10(nu.min() * 0.7), np.log10(nu.max() * 1.4), 100)
    a = sig ** 2 / grid ** 2
    pred = 3 * (-0.5 * np.log1p(a) + 0.5 * a / (1 + a))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(grid / sig, pred, color=P.EXACT, lw=1.5, alpha=0.6,
            label="predicted smoothing bias, 3 dims")
    ax.plot(nu / sig, [r[1] for r in rows], "o", color=P.EXACT,
            label="measured smoothing bias (exact smoothed − true)")
    ax.plot(nu / sig, [r[2] for r in rows], "s-", color=P.NEW,
            label="network RMSE vs smoothed target")
    ax.plot(nu / sig, [r[3] for r in rows], "D--", color=P.LEGACY,
            label="network mean error vs the *true* density")
    ax.axhline(0, color=P.INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("ν / σ(eps)   (noise width relative to PF-ODE smoothing)")
    ax.set_ylabel("nats per row")
    ax.set_title("Gaussian likelihood, 2^16 sims: smoothing vs network error", loc="left",
                 fontsize=10)
    ax.legend(fontsize=7.5)
    return P.save(fig, FIG, "fig_noise_sweep")


def mle_check(nets):
    lines = ["| family | budget | iters | PF value − exact at returned θ (overestimate) | "
             "exact at returned θ − exact max |", "|" + "---|" * 5]
    for tag in sorted(t for t in nets if t.endswith("nu0.5")):
        for b in BUDGETS:
            runs = [o for (bb, _), o in nets[tag].items() if bb == b and "mle" in o]
            if not runs:
                continue
            for it in (5, 20, 100):
                over = np.mean([(o["mle"][f"iters_{it}"]["pf_at_mle"]
                                 - o["mle"][f"iters_{it}"]["exact_at_mle"]).mean() for o in runs])
                gap = np.mean([(o["mle"][f"iters_{it}"]["exact_at_mle"]
                                - o["mle"]["exact_max"]).mean() for o in runs])
                lines.append(f"| {tag} | {b} | {it} | {over:+.3f} | {gap:+.3f} |")
    return "\n".join(lines)


def main():
    P.style()
    nets, oracles = load()
    parts = ["## Error per family and budget (100 steps, exact divergence, eps 1e-3)\n",
             table(nets, oracles)]
    figs = [fig_budget(nets, oracles), fig_bands(nets), fig_solver(nets, oracles),
            fig_scatter(nets)]
    md, path = model_difference(nets)
    figs += [path, fig_noise_sweep(nets)]
    parts += ["\n## B6.3 model differences\n", md, "\n## A2 'too good MLE' check\n",
              mle_check(nets)]
    text = "\n".join(parts)
    with open(os.path.join(HERE, "results", "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    print("\n".join(figs))


if __name__ == "__main__":
    main()
