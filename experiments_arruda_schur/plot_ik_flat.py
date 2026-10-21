#!/usr/bin/env python3
"""Figures for ``ik_flat.py`` (observations at the shoulder height), from its saved runs.

summary.png   branch mass P(a1>0) and W1/sd against n, exact vs both rules, per variant.
branches.png  each branch's shape on |a1|, normalised within the branch so mass and shape
              are separated; rows = variant x branch, columns = n.
height.png    h global: exact marginal of h vs each rule, and the rules' h given the branch.

    python plot_ik_flat.py
"""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "3")
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

ROOT = Path(__file__).resolve().parent / "artifacts" / "ik_flat"
NS = (8, 32, 128)
VARIANTS = (("fixed", "h known (h = 0, not a parameter)"), ("global", "h global (prior sd 0.05)"))
METHODS = (("gauss_jacobian_FD", "hier. attn + gauss_jacobian_FD", "#eda100"),
           ("arruda_direct", "hier. attn + Arruda (plain sum)", "#e87ba4"))
INK, MUTED, REF = "#1f1f1e", "#6b6a63", "#d9d8d2"
EXACT = np.load(ROOT / "exact.npz")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def theta(variant, method, n):
    path = ROOT / "runs" / variant / f"{method}_n{n}.npz"
    return np.load(path)["theta"] if path.exists() else None


def exact(variant, n, key=""):
    grid, density = EXACT[f"{variant}_{key or ''}grid_n{n}"], EXACT[f"{variant}_{key + "_" if key else ''}n{n}"]
    return grid, density / np.trapezoid(density, grid)


def draws(grid, density, count=200_000):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return np.interp(np.random.default_rng(0).random(count), cdf / cdf[-1], grid)


def summary():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), squeeze=False)
    for j, (variant, title) in enumerate(VARIANTS):
        top, bottom = axes[0, j], axes[1, j]
        style(top), style(bottom)
        mass = []
        for n in NS:
            g, d = exact(variant, n)
            mass.append(np.trapezoid(d * (g > 0), g))
        top.plot(NS, mass, "o-", color=INK, lw=2, ms=8, label="exact")
        for method, label, colour in METHODS:
            p, w = [], []
            for n in NS:
                t = theta(variant, method, n)
                a1 = t[:, 0, 0]
                p.append(np.mean(a1 > 0))
                ref = draws(*exact(variant, n))
                w.append(wasserstein_distance(a1, ref) / ref.std())
            top.plot(NS, p, "o-", color=colour, lw=2, ms=8, label=label)
            bottom.plot(NS, w, "o-", color=colour, lw=2, ms=8, label=label)
            top.annotate(f"{p[-1]:.2f}", (NS[-1], p[-1]), xytext=(6, 0), textcoords="offset points",
                         fontsize=7, color=INK, va="center")
        top.annotate(f"{mass[-1]:.2f}", (NS[-1], mass[-1]), xytext=(6, 0), textcoords="offset points",
                     fontsize=7, color=INK, va="center")
        top.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        top.set_ylim(0, 1)
        top.set_title(title, fontsize=9, color=INK)
        top.set_ylabel("P(a1 > 0)  (branch mass)", fontsize=8, color=INK)
        bottom.set_ylabel("W1 / sd(exact)  on a1", fontsize=8, color=INK)
        bottom.set_ylim(bottom=0)
        for ax in (top, bottom):
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS, [str(n) for n in NS])
            ax.set_xlabel("n observations", fontsize=8, color=INK)
            ax.grid(axis="y", color=REF, lw=0.6)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("IK, observations at the shoulder height (|x1 - h| < 0.001), true a1 = 0.6\n"
                 "top: how the mass splits between the ±a1 modes;  bottom: overall distance to exact",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "summary.png", dpi=150, bbox_inches="tight")


def branches():
    rows = [(v, t, s) for v, t in VARIANTS for s in (1, -1)]
    fig, axes = plt.subplots(len(rows), len(NS), figsize=(3.6 * len(NS), 1.9 * len(rows)), squeeze=False)
    bins = np.linspace(0.2, 1.2, 81)
    for i, (variant, title, sign) in enumerate(rows):
        for j, n in enumerate(NS):
            ax = axes[i, j]
            style(ax)
            g, d = exact(variant, n)
            keep = sign * g > 0
            ga, da = np.abs(g[keep]), d[keep]
            order = np.argsort(ga)
            ga, da = ga[order], da[order]
            mass = np.trapezoid(da, ga)
            da = da / mass
            mu = np.trapezoid(ga * da, ga)
            sd = np.sqrt(np.trapezoid((ga - mu) ** 2 * da, ga))
            top = 0
            for method, label, colour in METHODS:
                a1 = theta(variant, method, n)[:, 0, 0]
                branch = np.abs(a1[sign * a1 > 0])
                if len(branch) > 20:
                    h, _, _ = ax.hist(branch, bins=bins, density=True, histtype="step", color=colour, lw=2)
                    top = max(top, h.max())
            cap = 1.3 * top if top else da.max()
            ax.fill_between(ga, np.minimum(da, cap), color=REF, lw=0)
            ax.plot(ga, np.minimum(da, cap), color=INK, lw=1)
            if da.max() > cap:
                ax.annotate("exact peak\noff-scale", (mu, cap), xytext=(8, -4), textcoords="offset points",
                            fontsize=6.5, color=MUTED, va="top")
            ax.set_ylim(0, cap * 1.05)
            ax.set_xlim(bins[0], bins[-1])
            ax.set_yticks([])
            ax.set_title(f"exact: mass {mass:.2f}, |a1| {mu:.3f} ± {sd:.4f}", fontsize=7, color=INK)
            if j == 0:
                side = "a1 > 0" if sign > 0 else "a1 < 0"
                ax.set_ylabel(f"{title.split(' (')[0]}\nbranch {side}", fontsize=7.5, color=INK,
                              rotation=0, ha="right", va="center")
            if i == 0:
                ax.annotate(f"n = {n}", (0.5, 1.3), xycoords="axes fraction", ha="center", fontsize=10,
                            color=INK, weight="bold")
            if i == len(rows) - 1:
                ax.set_xlabel("|a1| within the branch", color=INK)
    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for _, l, c in METHODS]
    handles.insert(0, plt.Rectangle((0, 0), 1, 1, color=REF, label="exact"))
    fig.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=False, ncol=3,
               bbox_to_anchor=(0.98, 1.02))
    fig.suptitle("Shape and position of each mode, normalised within the branch (mass taken out)\n",
                 fontsize=10, color=INK, x=0.35, y=1.03)
    fig.tight_layout()
    fig.savefig(ROOT / "branches.png", dpi=150, bbox_inches="tight")


def height():
    fig, axes = plt.subplots(2, len(NS), figsize=(3.6 * len(NS), 4.6), squeeze=False)
    bins = np.linspace(-0.6, 0.6, 81)
    for j, n in enumerate(NS):
        top, bottom = axes[0, j], axes[1, j]
        style(top), style(bottom)
        gh, dh = exact("global", n, "h")
        ref = draws(gh, dh)  # rebinned: the grid under-resolves the sharp likelihood
        top.hist(ref, bins=bins, density=True, color=REF, lw=0)
        top.hist(ref, bins=bins, density=True, histtype="step", color=INK, lw=1)
        lo, hi = np.quantile(ref, [0.16, 0.84])
        top.plot(gh, np.exp(-0.5 * (gh / 0.05) ** 2) / (0.05 * np.sqrt(2 * np.pi)), color=MUTED, lw=0.8,
                 ls=":")
        positions = np.arange(len(METHODS))
        for k, (method, label, colour) in enumerate(METHODS):
            t = theta("global", method, n)
            h, a1 = t[:, 0, 1], t[:, 0, 0]
            top.hist(h, bins=bins, density=True, histtype="step", color=colour, lw=2)
            for s, marker, offset in ((1, "^", -0.12), (-1, "v", 0.12)):
                sel = h[s * a1 > 0]
                if len(sel) > 20:
                    bottom.errorbar(k + offset, sel.mean(), yerr=sel.std(), fmt=marker, ms=8, color=colour,
                                    ecolor=colour, lw=1.5)
        bottom.axhspan(lo, hi, color=REF, lw=0)
        bottom.axhline(0, color=INK, lw=0.8)
        bottom.set_xticks(positions, ["gauss_jacobian_FD", "Arruda"], fontsize=7)
        bottom.set_xlim(-0.6, len(METHODS) - 0.4)
        bottom.set_ylim(-0.6, 0.6)
        top.set_xlim(bins[0], bins[-1])
        top.set_yticks([])
        top.set_xlabel("h", color=INK, fontsize=8)
        top.set_title(f"n = {n}", fontsize=10, color=INK, weight="bold")
        if j == 0:
            top.set_ylabel("marginal of h", fontsize=8, color=INK)
            bottom.set_ylabel("h given the branch\n(mean ± sd)", fontsize=8, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REF, label="exact"),
               plt.Line2D([], [], color=MUTED, lw=0.8, ls=":", label="prior N(0, 0.05²)")]
    handles += [plt.Line2D([], [], color=c, lw=2, label=l) for _, l, c in METHODS]
    handles += [plt.Line2D([], [], color=MUTED, marker="^", ls="", label="draws with a1 > 0"),
                plt.Line2D([], [], color=MUTED, marker="v", ls="", label="draws with a1 < 0")]
    fig.legend(handles=handles, loc="lower center", fontsize=7.5, frameon=False, ncol=3,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("h global: the shoulder height the samplers infer (true h = 0)\nbottom: grey band = exact 16-84% range of h (all draws)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "height.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    summary()
    branches()
    height()
    print("wrote", ROOT / "summary.png", ROOT / "branches.png", ROOT / "height.png")
