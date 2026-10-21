#!/usr/bin/env python3
"""Posterior plots for the folded hierarchy: every arm against Gibbs MCMC.

fig1_shared.png  p(g | x_1..n): rows = arms, columns = n. Grey = MCMC histogram,
                 black = closed form, colour = the arm. Title: W1/sd vs MCMC, P(g>0).
Pass --model=square for step A, --model=lean for step B (artifacts/<model>/figures/).
Pass "oracle" as the first argument for fig1_shared_oracle.png: the same arms
with the exact row score in place of the network (run.py --mode oracle_*).
fig2_locals.png  n = 16: joint (g, l_1) over MCMC, and every local's posterior
                 mean +- sd against MCMC's.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

import experiments  # noqa: E402

MODEL_NAME = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model=")), "abs")
EXPERIMENT = experiments.get(MODEL_NAME)
DEFAULT = EXPERIMENT["model"]
RUNS = EXPERIMENT["root"] / "runs"
OUT = EXPERIMENT["root"] / "figures"
INK, MUTED, GRID, REF = "#1f1f1e", "#6b6a63", "#e6e5df", "#bdbcb4"
# (rule, network-mode prefix, label, colour) -- fixed categorical order
RULES = [
    ("langevin_fnpse", "flat", "Langevin + F-NPSE", "#2a78d6"),
    ("gauss_hierarchical", "flat", "DPM2 + gauss_hierarchical", "#eb6834"),
    ("gauss_jacobian_FD", "flat", "DPM2 + gauss_jacobian_FD", "#1baf7a"),
    ("gauss_jacobian_FD", "hier", "hier. attn + gauss_jacobian_FD", "#eda100"),
    ("arruda_direct", "hier", "hier. attn + Arruda (plain sum)", "#e87ba4"),
]
# Score sources. The oracle replaces the production network by the exact *joint*
# row score and the hierarchical network by the exact hierarchical-form score.
SOURCES = {
    "network": ({"flat": "none", "hier": "hierarchical"}, "trained networks", ""),
    "oracle": ({"flat": "oracle_joint", "hier": "oracle_hierarchical"},
               "EXACT row scores (no network)", "_oracle"),
    "ladder": ({}, "low-noise ablations of hier. attn + gauss_jacobian_FD (precondition.py)",
               "_ladder"),
    "residual": ({}, "hier. attn nets, stock vs residual output (R)", "_residual"),
}
# fig1_shared_residual.png: stock hierarchical net vs hierarchical_R, two rules each
RESIDUAL = [
    ("hierarchical_gauss_jacobian_FD", "stock + gauss_jacobian_FD", "#eda100"),
    ("hierarchical_arruda_direct", "stock + Arruda (plain sum)", "#e87ba4"),
    ("hierarchical_R_gauss_jacobian_FD", "R + gauss_jacobian_FD", "#2a78d6"),
    ("hierarchical_R_arruda_direct", "R + Arruda (plain sum)", "#1baf7a"),
]
# fig1_shared_ladder.png: run.py --mode hierarchical_<tag>; one seed each
LADDER = [
    ("oracle_hierarchical_gauss_jacobian_FD", "exact score (rule's own error)", "#6b6a63"),
    ("hierarchical_gauss_jacobian_FD", "stock net", "#eda100"),
    ("hierarchical_T_gauss_jacobian_FD", "T: log-λ time input", "#2a78d6"),
    ("hierarchical_T_e4_gauss_jacobian_FD", "T, sampled to eps 1e-4", "#7b8fd6"),
    ("hierarchical_TF_gauss_jacobian_FD", "TF: + floor eps 1e-4 (train+sample)", "#1baf7a"),
    ("hierarchical_R_gauss_jacobian_FD", "R: residual output only", "#e87ba4"),
    ("hierarchical_TFR_gauss_jacobian_FD", "TFR: all three", "#eb6834"),
]


def arms(source):
    if source == "ladder":
        return LADDER
    if source == "residual":
        return RESIDUAL
    prefixes, _, _ = SOURCES[source]
    labels = {"network": {}, "oracle": {"flat": "exact joint", "hier": "exact hier."}}[source]
    result = []
    for rule, kind, label, colour in RULES:
        if kind in labels:
            label = label.replace("hier. attn", labels[kind]) if kind == "hier" else f"{labels[kind]} + {label}"
        result.append((f"{prefixes[kind]}_{rule}", label, colour))
    return result
NS = [4, 16, 64, 128]


def load(name):
    path = RUNS / f"{name}.npz"
    return dict(np.load(path)) if path.exists() else None


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def shared_figure(arms, source="network"):
    fig, axes = plt.subplots(len(arms), len(NS), figsize=(3.4 * len(NS), 1.9 * len(arms)),
                             squeeze=False, sharex="col")
    for j, n in enumerate(NS):
        ref = load(f"mcmc_n{n}")
        half = DEFAULT.plot_half_width(ref["x"])
        grid = np.linspace(-half, half, 1200)
        bins = np.linspace(-half, half, 90)
        exact = DEFAULT.global_density(grid, ref["x"])
        sd = ref["g"].std()
        for i, (name, label, colour) in enumerate(arms):
            ax = axes[i, j]
            style(ax)
            ax.hist(ref["g"], bins=bins, density=True, color=REF, alpha=0.55, lw=0)
            ax.plot(grid, exact, color=INK, lw=1.0)
            run = load(f"{name}_n{n}")
            if run is None:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color=MUTED)
            else:
                g = run["g"]
                ax.hist(g, bins=bins, density=True, histtype="step", color=colour, lw=2)
                outside = np.mean(np.abs(g) > half)
                w1 = wasserstein_distance(g, ref["g"]) / sd
                ax.set_title(f"W1/sd {w1:.2f}   P(g>0) {np.mean(g > 0):.2f}"
                             + (f"   {outside:.0%} off-axis" if outside > 0.005 else ""),
                             fontsize=7.5, color=INK)
            ax.set_yticks([])
            ax.set_xlim(-half, half)
            if j == 0:
                ax.set_ylabel(label, fontsize=7.5, color=INK, rotation=0, ha="right", va="center")
            if i == 0:
                exact_mass = (f"   (exact P(g>0) {DEFAULT.positive_mass(ref['x']):.2f})"
                         if hasattr(DEFAULT, "positive_mass") else "")
                ax.annotate(f"n = {n}{exact_mass}", (0.5, 1.35), xycoords="axes fraction", ha="center",
                            fontsize=10, color=INK, weight="bold")
            if i == len(arms) - 1:
                ax.set_xlabel("g", color=INK)
    fig.suptitle(f"{DEFAULT.label}, g_true = {EXPERIMENT['g_true']:g}:  "
                 "shared posterior p(g | x₁..xₙ)\ngrey = Gibbs MCMC, black = closed form, "
                 f"colour = composition arm on {SOURCES[source][1]}", fontsize=10, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / f"fig1_shared{SOURCES[source][2]}.png", dpi=150, bbox_inches="tight")


def locals_figure(arms, n=16):
    ref = load(f"mcmc_n{n}")
    fig, axes = plt.subplots(2, len(arms), figsize=(3.0 * len(arms), 6.0), squeeze=False)
    rm, rs = ref["l"].mean(0), ref["l"].std(0)
    order = np.argsort(rm)
    pick = np.random.default_rng(0).choice(len(ref["g"]), 1500, replace=False)
    for k, (name, label, colour) in enumerate(arms):
        top, bottom = axes[0, k], axes[1, k]
        style(top), style(bottom)
        top.scatter(ref["g"][pick], ref["l"][pick, 0], s=3, color=REF, lw=0)
        bottom.errorbar(np.arange(n), rm[order], yerr=rs[order], fmt="o", ms=3, color=REF,
                        ecolor=REF, lw=3, label="MCMC")
        run = load(f"{name}_n{n}")
        top.set_title(label, fontsize=8, color=INK)
        if run is not None:
            sub = np.random.default_rng(1).choice(len(run["g"]), min(1500, len(run["g"])), replace=False)
            top.scatter(run["g"][sub], run["l"][sub, 0], s=3, color=colour, lw=0, alpha=0.6)
            mm, ms = run["l"].mean(0), run["l"].std(0)
            bottom.errorbar(np.arange(n) + 0.25, mm[order], yerr=ms[order], fmt="o", ms=3,
                            color=colour, ecolor=colour, lw=1, label="arm")
            err = np.mean(np.abs(mm - rm) / rs)
            bottom.set_title(f"mean |Δμ|/sd {err:.2f}   sd ratio {np.mean(ms / rs):.2f}",
                             fontsize=7.5, color=INK)
        top.set_xlim(-2.5, 2.5)
        top.set_ylim(rm[0] - 4 * rs[0], rm[0] + 4 * rs[0])
        top.set_xlabel("g", color=INK)
        bottom.set_xlabel("local index (sorted by MCMC mean)", fontsize=7.5, color=INK)
        if k == 0:
            top.set_ylabel("l₁", color=INK)
            bottom.set_ylabel("l_j  (mean ± sd)", color=INK)
            bottom.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{DEFAULT.label}, n = {n}: joint (g, l₁) and all local marginals "
                 "(grey = Gibbs MCMC)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_locals.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    source = positional[0] if positional else "network"
    shared_figure(arms(source), source)
    if source == "network":
        locals_figure(arms(source))
    print("wrote", OUT)
