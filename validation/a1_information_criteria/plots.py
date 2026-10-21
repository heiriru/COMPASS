"""Figures for A1 from results/known_answer.json (no recomputation).

    .COMPASS/bin/python -m validation.a1_information_criteria.plots

fig_selection_rates: rate at which each criterion picks the larger model against N,
measured (points, 99% Wilson bars) vs the chi^2 prediction (lines), in three panels:
    (a) superfluous global, null      -> AIC should sit flat at 0.157
    (b) real global effect, lambda=4  -> power; legacy collapses to 0
    (c) superfluous local, flat       -> counted BIC is inconsistent (-> 1)
"""

import validation.common.runtime  # noqa: F401

import json
import os

import numpy as np

from validation.common import plotting as P

HERE = os.path.dirname(__file__)


def series(rows, experiment, convention, criterion, null=True, shrink=1.0):
    pick = [r for r in rows if r["experiment"] == experiment and r["convention"] == convention
            and r["criterion"] == criterion and r["shrink"] == shrink
            and ((r["delta"] == 0) == null)]
    pick.sort(key=lambda r: r["n"])
    n = np.array([r["n"] for r in pick])
    rate = np.array([r["rate"] for r in pick])
    lo = np.array([r["ci99"][0] for r in pick])
    hi = np.array([r["ci99"][1] for r in pick])
    pred = np.array([r["predicted"] for r in pick])
    return n, rate, lo, hi, pred


def panel(ax, rows, experiment, null, lines, title, ylabel=True):
    for convention, criterion, color, label, marker in lines:
        n, rate, lo, hi, pred = series(rows, experiment, convention, criterion, null)
        ax.plot(n, pred, color=color, alpha=0.45, lw=2)
        ax.errorbar(n, rate, yerr=[rate - lo, hi - rate], fmt=marker, color=color,
                    ms=7, capsize=3, lw=1.2, label=label, mec=P.SURFACE, mew=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 64, 256])
    ax.set_xticklabels(["4", "16", "64", "256"])
    ax.set_xlabel("rows N")
    if ylabel:
        ax.set_ylabel("P(select larger model)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, loc="left")


def main():
    P.style()
    with open(os.path.join(HERE, "results", "known_answer.json")) as f:
        rows = json.load(f)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    lines = [("counted", "aic", P.NEW, "counted AIC", "o"),
             ("counted", "bic", P.EXACT, "counted BIC", "s"),
             ("legacy", "aicc", P.LEGACY, "legacy (default, AICc)", "D")]
    panel(axes[0], rows, "hierarchical", True, lines,
          "(a) superfluous global (null)\nAIC should sit at 0.157")
    axes[0].axhline(0.1573, color=P.INK2, lw=1, ls=":")
    panel(axes[1], rows, "hierarchical", False, lines,
          "(b) real global effect, λ = 4\npower — higher is better", ylabel=False)
    panel(axes[2], rows, "flat", True,
          [("counted", "aic", P.NEW, "counted AIC", "o"),
           ("counted", "bic", P.EXACT, "counted BIC (n/row = 2)", "s"),
           ("legacy", "bic", P.LEGACY, "legacy BIC (ln N per row)", "D")],
          "(c) superfluous local per row (flat)\nshould go to 0", ylabel=False)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[2].legend(loc="center right", fontsize=9)
    fig.text(0.5, -0.02, "points: measured over 10,000 datasets (99% Wilson bars); "
             "faint lines: χ² prediction", ha="center", color=P.INK2, fontsize=9)
    print(P.save(fig, os.path.join(HERE, "results", "figures"), "fig_selection_rates"))


if __name__ == "__main__":
    main()
