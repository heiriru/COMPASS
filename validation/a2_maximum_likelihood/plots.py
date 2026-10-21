"""Figures for A2 from the saved results (no recomputation).

    .COMPASS/bin/python -m validation.a2_maximum_likelihood.plots

fig_plugin_error: AIC error of the plug-in estimate against prior scale s/nu (32 rows),
    MAP vs PF-ODE MLE, log scale (exactness.json).
fig_pipeline_lr: per dataset, the pipeline's likelihood-ratio statistic against the
    exact one, for the MLE and MAP plug-ins, with y = x and the predicted MAP line
    y = 0.75 x; the AIC decision boundary LR = 2 is marked. Null and effect panels.
fig_nonconcave_shortfall: distribution over rows of the log-likelihood shortfall
    below the global maximum for each start strategy (nonconcave.py is re-run, ~5 s).
"""

import validation.common.runtime  # noqa: F401

import glob
import json
import os

import numpy as np

from validation.common import plotting as P

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "results", "figures")


def plugin_error():
    import matplotlib.pyplot as plt
    with open(os.path.join(HERE, "results", "exactness.json")) as f:
        res = json.load(f)
    s = np.array([r["s_over_nu"] for r in res])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(s, [r["map_aic_error"] for r in res], "o-", color=P.LEGACY, label="MAP plug-in")
    ax.plot(s, [max(r["unbounded"]["aic_error"], 1e-8) for r in res], "o-", color=P.NEW,
            label="PF-ODE MLE")
    ax.plot(s, [max(r["default_box"]["aic_error"], 1e-8) for r in res], "o--", color=P.NEW,
            alpha=0.55, label="PF-ODE MLE, default box")
    ax.axhline(10, color=P.INK2, lw=1, ls=":")
    ax.text(s[-1], 12, "ΔAIC = 10 (\"decisive\")", ha="right", color=P.INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prior scale s / ν   (small = informative prior)")
    ax.set_ylabel("AIC error vs exact, 32 rows")
    ax.set_title("Plug-in error of the fit term", loc="left")
    ax.legend(fontsize=9)
    print(P.save(fig, FIG, "fig_plugin_error"))


def convergence():
    """Objective gap to its converged value vs evaluation, per prior scale (exactness.json)."""
    import matplotlib.pyplot as plt
    with open(os.path.join(HERE, "results", "exactness.json")) as f:
        res = json.load(f)
    colors = [P.LEGACY, P.FOURTH, P.EXACT, P.NEW]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, label in zip(axes, ("unbounded", "default_box")):
        for r, color in zip(res, colors):
            m = r[label]
            trace = np.array(m["trace_total_loglik"])
            gap = np.maximum(trace.max() - trace, 1e-7)
            stop = "converged" if m["messages"][0].startswith("CONVERGENCE") else "line search stopped"
            ax.plot(np.arange(1, len(trace) + 1), gap, "o-", color=color, ms=4,
                    label=f"s/ν = {r['s_over_nu']:g}: {stop}; exact deficit "
                          f"{m['exact_deficit_per_row_max']:.0e} nats/row")
        ax.set_yscale("log")
        ax.set_xlabel("objective evaluation (PF-ODE solve with gradient)")
        ax.set_title("(a) unbounded" if label == "unbounded"
                     else "(b) default box (posterior range ± 3 sd)", loc="left")
        ax.legend(fontsize=7.5, loc="lower left" if label == "unbounded" else "upper right")
    axes[0].set_ylabel("summed log-lik below its final value (nats, 32 rows)")
    fig.text(0.5, -0.03, "Start (evaluation 1) = MAP. Points at the floor (1e-7) are at the "
             "final value. 'exact deficit' = max over rows of exact max logL − exact logL at "
             "the returned θ.", ha="center", color=P.INK2, fontsize=8.5)
    print(P.save(fig, FIG, "fig_mle_convergence"))


def pipeline_lr():
    import matplotlib.pyplot as plt
    recs = []
    for path in glob.glob(os.path.join(HERE, "results", "pipeline", "*.jsonl")):
        with open(path) as f:
            recs += [json.loads(line) for line in f]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, delta, title in ((axes[0], 0.0, "(a) no global effect (g = 0), 400 datasets"),
                             (axes[1], 0.5, "(b) real effect (λ = 4), 200 datasets")):
        sub = [r for r in recs if r["delta"] == delta]
        e = np.array([r["lr_exact"] for r in sub])
        top = max(e.max(), 1.0) * 1.05
        grid = np.linspace(0, top, 2)
        ax.plot(grid, grid, color=P.INK2, lw=1, label="y = x")
        ax.plot(grid, 0.75 * grid, color=P.LEGACY, lw=1, ls=":", label="predicted MAP, y = 0.75 x")
        ax.axvline(2, color=P.GRID, lw=6, zorder=0)
        ax.axhline(2, color=P.GRID, lw=6, zorder=0)
        ax.scatter(e, [r["lr_map"] for r in sub], s=14, color=P.LEGACY, alpha=0.7,
                   label="pipeline, MAP plug-in", edgecolors="none")
        ax.scatter(e, [r["lr_mle"] for r in sub], s=14, color=P.NEW, alpha=0.8,
                   label="pipeline, PF-ODE MLE", edgecolors="none")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("exact LR statistic  2(ℓ₁ − ℓ₀)")
        ax.set_title(title, loc="left")
        ax.text(2.1, top * 0.96, "AIC picks M1 →", color=P.INK2, fontsize=8, va="top")
    axes[0].set_ylabel("LR statistic from compare()")
    axes[0].legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.93))
    print(P.save(fig, FIG, "fig_pipeline_lr"))


def nonconcave_shortfall():
    """Re-run the four arms keeping per-row shortfalls (cheap analytic model)."""
    import matplotlib.pyplot as plt
    import torch
    from compass.likelihood_maximization import maximize_likelihood
    from validation.a2_maximum_likelihood import nonconcave as nc

    g = torch.Generator().manual_seed(20261013)
    R = nc.R
    theta_true = (1.0 + 0.5 * torch.rand(R, generator=g)) * torch.where(
        torch.rand(R, generator=g) < 0.5, -1.0, 1.0)
    x1 = theta_true ** 2 + nc.NU1 * torch.randn(R, generator=g)
    x2 = nc.A * theta_true + nc.NU2 * torch.randn(R, generator=g)
    model, mask = nc.BimodalLikelihood(), torch.tensor([1.0, 0.0, 0.0])
    grid = torch.linspace(-4, 4, 80001)
    ll_grid = model.log_prob(torch.stack([grid.expand(R, -1).reshape(-1),
                                          x1.repeat_interleave(len(grid)),
                                          x2.repeat_interleave(len(grid))], 1), mask
                             ).reshape(R, -1)
    theta_global, ll_global = grid[ll_grid.argmax(1)], ll_grid.max(1).values
    post = torch.softmax(ll_grid - 0.5 * grid ** 2, 1)
    wrong = -torch.sign(theta_global) * x1.clamp_min(0).sqrt()
    draws = [grid[torch.multinomial(post, 1, generator=g).squeeze(1)] for _ in range(4)]
    draws32 = [grid[torch.multinomial(post, 1, generator=g).squeeze(1)] for _ in range(32)]

    def run(first, rest, keep=None):
        data = torch.stack([first, x1, x2], 1)
        starts = torch.stack([torch.stack([s, x1, x2], 1) for s in rest]) if rest else None
        res = maximize_likelihood(model, data, mask, starts=starts, max_iterations=100,
                                  keep_best_starts=keep)
        return (ll_global - res.log_likelihood.double()).clamp_min(0).numpy()

    arms = [("1 start, wrong basin", run(wrong, []), P.LEGACY),
            ("4 posterior draws", run(draws[0], draws[1:]), P.FOURTH),
            ("wrong + best 3 of 32 screened\n(compare() default)", run(wrong, draws32, keep=3), P.NEW)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.concatenate([[0], np.logspace(-6, 1, 36)])
    for label, gap, color in arms:
        found = np.mean(gap < 1e-3)
        ax.hist(np.clip(gap, 1e-6, None), bins=bins, histtype="step", lw=2, color=color,
                label=f"{label}: {100 * found:.1f}% at global max")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1500)
    ax.axvline(1e-3, color=P.INK2, lw=1, ls=":")
    ax.text(1.3e-6, 600, "≤ 1e-6\n(converged)", fontsize=8, color=P.INK2)
    ax.text(1.2e-3, 600, "global-max\nthreshold", fontsize=8, color=P.INK2)
    ax.set_xlabel("log-likelihood shortfall below the global maximum (nats, per row)")
    ax.set_ylabel("rows (of 400, log scale)")
    ax.set_title("Two unequal likelihood maxima: does the MLE find the global one?",
                 loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.1, 0.5))
    print(P.save(fig, FIG, "fig_nonconcave_shortfall"))


def main():
    P.style()
    plugin_error()
    convergence()
    pipeline_lr()
    nonconcave_shortfall()


if __name__ == "__main__":
    main()
