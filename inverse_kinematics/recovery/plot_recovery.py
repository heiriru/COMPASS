#!/usr/bin/env python3
"""Recovery plots and metrics: MCMC reference vs the network, n=1 and hierarchical.

Reads ``artifacts/<setting>/{data,reference,network_*}.npz`` and writes
``artifacts/figures/*.png`` and ``artifacts/metrics.json``. CPU only.

Usage (from COMPASS/): .COMPASS/bin/python inverse_kinematics/recovery/plot_recovery.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import kstest, wasserstein_distance  # noqa: E402

import datasets  # noqa: E402
from layouts import LAYOUTS, NODE_NAMES, PRIOR_STD  # noqa: E402

FIGURES = datasets.ARTIFACTS / "figures"
NS = (2, 8, 32, 64, 128)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"angle_1": "angle_1", "height_arm": "height", "angle_2": "angle_2", "angle_3": "angle_3"}
SINGLE_SOURCES = (("reference", "MCMC reference", BLUE),
                  ("network_angle", "network, angle_1 global", ORANGE),
                  ("network_angle_height", "network, angle_1+height global", AQUA))
HIER_SOURCES = (("reference", "MCMC reference", BLUE),
                ("network_gauss_hierarchical", "gauss_hierarchical", ORANGE),
                ("network_gauss_jacobian_FD", "gauss_jacobian_FD", AQUA))
# Ablation arms (only where run): metrics and the against-reference figure, not the recovery grid.
EXTRA_SOURCES = (("network_gauss_hierarchical_eps1e-05", "gauss_hierarchical, eps 1e-5", "#eda100"),)

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def load(folder, name):
    path = folder / f"{name}.npz"
    return np.load(path)["theta"].astype(np.float64) if path.exists() else None


def columns(truth, draws, param, n_global):
    """Flatten to (items, draws) for one parameter: globals once per dataset, locals per observation."""
    if param < n_global:
        return truth[:, 0, param], draws[:, :, 0, param]
    return truth[:, :, param].reshape(-1), draws[:, :, :, param].transpose(0, 2, 1).reshape(-1, draws.shape[1])


def recovery_metrics(truth, draws, prior_sd, sbc_items=None):
    """``sbc_items``: indices used for the rank test. Locals of one dataset share the global's
    posterior error, so their ranks are correlated; the KS test then needs one item per dataset."""
    mean = draws.mean(1)
    sd = draws.std(1)
    rank = (draws < truth[:, None]).mean(1)
    rank = rank if sbc_items is None else rank[sbc_items]
    out = {"rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
           "r2": float(1.0 - np.mean((mean - truth) ** 2) / np.var(truth)),
           "mean_posterior_sd": float(sd.mean()),
           "contraction": float(1.0 - np.mean(sd ** 2) / prior_sd ** 2),
           "sbc_ks_pvalue": float(kstest(rank, "uniform").pvalue)}
    for mass in (0.5, 0.9):
        lo, hi = np.quantile(draws, [(1 - mass) / 2, (1 + mass) / 2], axis=1)
        out[f"coverage_{int(mass * 100)}"] = float(np.mean((lo <= truth) & (truth <= hi)))
    return out


def versus_reference(draws, reference):
    """W1 of the network to the reference's first half of chains, against the floor set by the
    reference's second half (independent chains, same size); all divided by the reference sd."""
    half = reference.shape[1] // 2
    ref_a, ref_b = reference[:, :half], reference[:, half:]
    ref_sd = np.maximum(reference.std(1), 1e-12)
    rng = np.random.default_rng(0)
    sub = draws[:, rng.choice(draws.shape[1], size=min(half, draws.shape[1]), replace=False)]
    w1 = np.array([wasserstein_distance(d, r) for d, r in zip(sub, ref_a)]) / ref_sd
    floor = np.array([wasserstein_distance(b, a) for b, a in zip(ref_b, ref_a)]) / ref_sd
    return {"median_w1_over_ref_sd": float(np.median(w1)), "mean_w1_over_ref_sd": float(w1.mean()),
            "p90_w1_over_ref_sd": float(np.quantile(w1, 0.9)),
            "floor_median_w1_over_ref_sd": float(np.median(floor)), "floor_p90_w1_over_ref_sd": float(np.quantile(floor, 0.9)),
            "median_abs_mean_error_over_ref_sd": float(np.median(np.abs(draws.mean(1) - reference.mean(1)) / ref_sd)),
            "median_sd_ratio": float(np.median(draws.std(1) / ref_sd))}, w1


def recovery_panel(ax, truth, draws, colour, max_bars=150):
    med = np.median(draws, 1)
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=1)
    pick = np.linspace(0, len(truth) - 1, min(max_bars, len(truth))).astype(int)
    ax.vlines(truth[pick], lo[pick], hi[pick], color=colour, alpha=0.25, lw=0.8)
    ax.scatter(truth, med, s=6, color=colour, lw=0)
    span = [min(truth.min(), lo.min()), max(truth.max(), hi.max())]
    ax.plot(span, span, color=MUTED, lw=0.8, ls="--")
    m = recovery_metrics(truth, draws, 1.0)
    ax.text(0.03, 0.97, f"R² {m['r2']:.2f}\n90% cov {m['coverage_90']:.2f}", transform=ax.transAxes, va="top",
            fontsize=7, color=INK)


def sbc_panel(ax, truth, draws, colour, label):
    rank = np.sort((draws < truth[:, None]).mean(1))
    grid = np.linspace(0, 1, 201)
    ecdf = np.searchsorted(rank, grid, side="right") / len(rank)
    band = 1.36 / np.sqrt(len(rank))                    # 95% KS band (simultaneous)
    ax.fill_between(grid, -band, band, color=GRID, lw=0)
    ax.plot(grid, ecdf - grid, color=colour, lw=1.4, label=label)


def single(metrics):
    folder = datasets.folder("single")
    data = np.load(folder / "data.npz")
    truth = data["theta"]
    sources = [(k, lab, c, load(folder, k)) for k, lab, c in SINGLE_SOURCES]
    sources = [s for s in sources if s[3] is not None]
    reference = sources[0][3]
    fig, axes = plt.subplots(len(sources), 4, figsize=(10, 2.3 * len(sources)), constrained_layout=True,
                             squeeze=False, sharex="col", sharey="col")
    result = {}
    for r, (key, label, colour, draws) in enumerate(sources):
        result[key] = {}
        for p, name in enumerate(NODE_NAMES):
            t, d = columns(truth, draws, p, 4)
            recovery_panel(axes[r, p], t, d, colour)
            entry = recovery_metrics(t, d, PRIOR_STD[p])
            if key != "reference":
                entry["vs_reference"] = versus_reference(d, columns(truth, reference, p, 4)[1])[0]
            result[key][name] = entry
            if r == 0:
                axes[r, p].set_title(LABELS[name], fontsize=9)
            if r == len(sources) - 1:
                axes[r, p].set_xlabel("true value")
        axes[r, 0].set_ylabel(f"{label}\nposterior median (90% CI)")
    fig.suptitle(f"n = 1, all four parameters local: recovery over {len(truth)} simulated datasets", fontsize=10)
    fig.savefig(FIGURES / "single_recovery.png", dpi=170)
    plt.close(fig)

    # network vs reference: posterior means, per-dataset W1, SBC
    fig, axes = plt.subplots(3, 4, figsize=(10, 7), constrained_layout=True)
    for p, name in enumerate(NODE_NAMES):
        _, ref = columns(truth, reference, p, 4)
        for key, label, colour, draws in sources[1:]:
            _, d = columns(truth, draws, p, 4)
            axes[0, p].scatter(ref.mean(1), d.mean(1), s=5, color=colour, lw=0, label=label)
            _, w1 = versus_reference(d, ref)
            axes[1, p].hist(np.minimum(w1, 2.0), bins=np.linspace(0, 2, 41), histtype="step", color=colour, lw=1.4)
        lim = [ref.mean(1).min(), ref.mean(1).max()]
        axes[0, p].plot(lim, lim, color=MUTED, lw=0.8, ls="--")
        axes[0, p].set_title(LABELS[name], fontsize=9)
        axes[0, p].set_xlabel("reference posterior mean")
        axes[1, p].set_xlabel("W1(network, reference) / reference sd")
        for key, label, colour, draws in sources:
            t, d = columns(truth, draws, p, 4)
            sbc_panel(axes[2, p], t, d, colour, label)
        axes[2, p].set_xlabel("fractional rank of truth")
    axes[0, 0].set_ylabel("network posterior mean")
    axes[1, 0].set_ylabel("datasets")
    axes[2, 0].set_ylabel("ECDF - uniform (grey: 95% band)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right", markerscale=2)
    axes[2, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("n = 1: network posterior against the MCMC reference (W1 clipped at 2)", fontsize=10)
    fig.savefig(FIGURES / "single_vs_reference.png", dpi=170)
    plt.close(fig)
    metrics["single"] = result


def hierarchical(layout_name, metrics):
    layout = LAYOUTS[layout_name]
    g = layout.n_global
    present = [n for n in NS if (datasets.folder("hier", layout_name, n) / "reference.npz").exists()]
    if not present:
        return
    cells = {n: {} for n in present}
    for n in present:
        folder = datasets.folder("hier", layout_name, n)
        truth = np.load(folder / "data.npz")["theta"]
        for key, label, colour in HIER_SOURCES + EXTRA_SOURCES:
            draws = load(folder, key)
            if draws is not None:
                cells[n][key] = (truth, draws)
    # global recovery: rows n, columns (global param x source)
    cols = [(p, s) for p in range(g) for s in HIER_SOURCES]
    fig, axes = plt.subplots(len(present), len(cols), figsize=(2.3 * len(cols), 2.2 * len(present)),
                             constrained_layout=True, squeeze=False, sharex="col", sharey="col")
    result = {}
    for r, n in enumerate(present):
        result[f"n{n}"] = {}
        ref_truth, reference = cells[n]["reference"]
        for c, (p, (key, label, colour)) in enumerate(cols):
            ax = axes[r, c]
            if key not in cells[n]:
                ax.set_axis_off()
                continue
            truth, draws = cells[n][key]
            t, d = columns(truth, draws, p, g)
            recovery_panel(ax, t, d, colour)
            if r == 0:
                ax.set_title(f"{LABELS[NODE_NAMES[p]]}: {label}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"n = {n}\nposterior median (90% CI)")
            if r == len(present) - 1:
                ax.set_xlabel("true value")
        for key, label, colour in HIER_SOURCES + EXTRA_SOURCES:
            if key not in cells[n]:
                continue
            truth, draws = cells[n][key]
            result[f"n{n}"][key] = {}
            for p, name in enumerate(NODE_NAMES):
                t, d = columns(truth, draws, p, g)
                first = None if p < g else np.arange(0, len(t), truth.shape[1])
                entry = recovery_metrics(t, d, PRIOR_STD[p], first)
                if key != "reference":
                    entry["vs_reference"] = versus_reference(d, columns(truth, reference, p, g)[1])[0]
                result[f"n{n}"][key][name] = entry
    fig.suptitle(f"Hierarchical recovery, {' + '.join(layout.global_names)} global: "
                 f"global parameters over {len(ref_truth)} datasets per n", fontsize=10)
    fig.savefig(FIGURES / f"hier_{layout_name}_globals.png", dpi=170)
    plt.close(fig)

    # locals: network vs reference posterior medians, pooled over datasets x observations
    locals_ = list(range(g, 4))
    fig, axes = plt.subplots(len(present), len(locals_), figsize=(2.6 * len(locals_), 2.3 * len(present)),
                             constrained_layout=True, squeeze=False)
    for r, n in enumerate(present):
        truth, reference = cells[n]["reference"]
        for c, p in enumerate(locals_):
            ax = axes[r, c]
            _, ref = columns(truth, reference, p, g)
            pick = np.random.default_rng(0).choice(len(ref), size=min(800, len(ref)), replace=False)
            for key, label, colour in HIER_SOURCES[1:]:
                if key in cells[n]:
                    _, d = columns(truth, cells[n][key][1], p, g)
                    ax.scatter(np.median(ref[pick], 1), np.median(d[pick], 1), s=4, lw=0, color=colour,
                               alpha=0.7, label=label)
            lim = [np.median(ref, 1).min(), np.median(ref, 1).max()]
            ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
            if r == 0:
                ax.set_title(LABELS[NODE_NAMES[p]], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"n = {n}\nnetwork posterior median")
            if r == len(present) - 1:
                ax.set_xlabel("reference posterior median")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left", markerscale=2)
    fig.suptitle(f"Local parameters, {' + '.join(layout.global_names)} global "
                 "(800 random dataset-observation pairs per panel)", fontsize=10)
    fig.savefig(FIGURES / f"hier_{layout_name}_locals.png", dpi=170)
    plt.close(fig)

    # globals against the reference: standardised mean error and sd ratio, per dataset
    fig, axes = plt.subplots(len(present), 2 * g, figsize=(3.2 * 2 * g, 2.1 * len(present)),
                             constrained_layout=True, squeeze=False)
    for r, n in enumerate(present):
        truth, reference = cells[n]["reference"]
        for p in range(g):
            _, ref = columns(truth, reference, p, g)
            ref_mean, ref_sd = ref.mean(1), ref.std(1)
            for key, label, colour in HIER_SOURCES[1:] + EXTRA_SOURCES:
                if key not in cells[n]:
                    continue
                _, d = columns(truth, cells[n][key][1], p, g)
                z = np.clip((d.mean(1) - ref_mean) / ref_sd, -8, 8)
                ratio = np.clip(d.std(1) / ref_sd, 1 / 64, 64)
                axes[r, 2 * p].hist(z, bins=np.linspace(-8, 8, 49), histtype="step", color=colour, lw=1.4,
                                    label=label)
                axes[r, 2 * p + 1].hist(np.log2(ratio), bins=np.linspace(-2, 6, 49), histtype="step",
                                        color=colour, lw=1.4)
            axes[r, 2 * p].axvline(0, color=MUTED, lw=0.8, ls="--")
            axes[r, 2 * p + 1].axvline(0, color=MUTED, lw=0.8, ls="--")
            axes[r, 2 * p].set_title(f"{LABELS[NODE_NAMES[p]]}, n = {n}: mean error", fontsize=8)
            axes[r, 2 * p + 1].set_title(f"{LABELS[NODE_NAMES[p]]}, n = {n}: width", fontsize=8)
        axes[r, 0].set_ylabel("datasets")
    for p in range(g):
        axes[-1, 2 * p].set_xlabel("(network mean - reference mean) / reference sd")
        axes[-1, 2 * p + 1].set_xlabel("log2(network sd / reference sd)")
    axes[-1, 0].legend(fontsize=7, frameon=False, loc="upper center")
    fig.suptitle(f"Global posteriors against the MCMC reference, {' + '.join(layout.global_names)} global "
                 "(clipped at the axis limits)", fontsize=10)
    fig.savefig(FIGURES / f"hier_{layout_name}_vs_reference.png", dpi=170)
    plt.close(fig)

    # SBC of the globals for every source
    fig, axes = plt.subplots(len(present), g, figsize=(3.4 * g, 2.2 * len(present)), constrained_layout=True,
                             squeeze=False)
    for r, n in enumerate(present):
        for p in range(g):
            for key, label, colour in HIER_SOURCES:
                if key in cells[n]:
                    truth, draws = cells[n][key]
                    t, d = columns(truth, draws, p, g)
                    sbc_panel(axes[r, p], t, d, colour, label)
            axes[r, p].set_title(f"{LABELS[NODE_NAMES[p]]}, n = {n}", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("fractional rank of truth")
    fig.suptitle("SBC of the global parameters (ECDF - uniform; grey = 95% band)", fontsize=10)
    fig.savefig(FIGURES / f"hier_{layout_name}_sbc.png", dpi=170)
    plt.close(fig)
    metrics[f"hier_{layout_name}"] = result


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if (datasets.folder("single") / "reference.npz").exists():
        single(metrics)
    for layout_name in LAYOUTS:
        hierarchical(layout_name, metrics)
    (datasets.ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"wrote {FIGURES}")


if __name__ == "__main__":
    main()
