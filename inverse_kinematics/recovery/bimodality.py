#!/usr/bin/env python3
"""Why is the IK angle_1 posterior (almost) never bimodal, and where do the modes sit when it is?

Everything here uses the exact, noise-free posterior of ``ik_exact`` (no ABC kernel).

1. **Natural data** (the recovery sets): how often the exact angle_1 posterior has two
   modes, and how much the height prior charges the mirror branch. The arm's only exact
   symmetry, ``(a1, a2, a3, h) -> (-a1, -a2, -a3, 2 x1 - h)``, leaves every observation's
   likelihood unchanged; only the height prior tells a posterior point from its mirror
   twin, by the exact log-density gap ``sum_j -2 x1_j (x1_j - h_j) / sigma_h^2``.
2. **Constructed bimodal data** (true angle_1 = 0.6, nested n = 1..128):
   * ``rail``: rows kept only if the tip is within 0.05 of the shoulder height,
     ``|x1 - h| < 0.05`` (the rail-height set of ``Arruda/artifacts/ik/rail_height``);
   * ``x1sel``: rows kept only if ``|x1| < 0.15`` (``bimodality_study.selected_dataset``).
   For each: the exact posterior of the standard model (what every reference and every
   network computes) and the exact *selection-aware* posterior, which divides each row's
   likelihood by ``P(selected | a1)`` (computed exactly, no Monte Carlo).

Usage (from COMPASS/): .COMPASS/bin/python inverse_kinematics/recovery/bimodality.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import datasets  # noqa: E402
import ik_exact as E  # noqa: E402
import simulator as sim  # noqa: E402
from bimodality_study import selected_dataset  # noqa: E402

OUT = datasets.ARTIFACTS / "bimodality"
RAIL = ROOT / "Arruda" / "artifacts" / "ik" / "rail_height"
COMPASS_RAIL = ROOT / "Arruda" / "artifacts" / "ik" / "rail_a0.6_compass"
TRUE = 0.6
NS = (1, 2, 4, 8, 16, 32, 64, 128)
GRID = np.linspace(-2.0, 2.0, 2001)
BLUE, ORANGE, AQUA, INK, MUTED, GRIDC = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


# ---------------------------------------------------------------------------
# shape statistics of a density on a grid
# ---------------------------------------------------------------------------

def shape(grid, density, min_mass=0.05):
    """Peaks carrying >= min_mass (split at the valleys), dip ratio, P(a1 > 0), E|a1|."""
    p = density / np.trapezoid(density, grid)
    interior = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    peaks = [i for i in interior if p[i] > 1e-3 * p.max()]
    # merge peaks separated by shallow valleys into basins, then keep basins with enough mass
    basins = []
    for i in peaks:
        if basins:
            j = basins[-1]
            valley = p[j:i + 1].min()
            if valley > 0.8 * min(p[i], p[j]):
                if p[i] > p[j]:
                    basins[-1] = i
                continue
        basins.append(i)
    masses = []
    cuts = [0] + [basins[k] + int(np.argmin(p[basins[k]:basins[k + 1] + 1])) for k in range(len(basins) - 1)] + [len(p)]
    for k in range(len(basins)):
        masses.append(float(np.trapezoid(p[cuts[k]:cuts[k + 1]], grid[cuts[k]:cuts[k + 1]])))
    kept = [b for b, m in zip(basins, masses) if m >= min_mass]
    dip = 1.0
    if len(kept) >= 2:
        a, b = sorted(sorted(kept, key=lambda i: -p[i])[:2])
        dip = float(p[a:b + 1].min() / min(p[a], p[b]))
    return {"peaks": [round(float(grid[i]), 3) for i in kept], "dip_ratio": dip,
            "bimodal": len(kept) >= 2 and dip < 0.8,
            "p_positive": float(np.trapezoid(np.where(grid > 0, p, 0), grid)),
            "mean_abs": float(np.trapezoid(np.abs(grid) * p, grid)),
            "mode": float(grid[np.argmax(p)])}


def kde_density(draws, grid):
    return gaussian_kde(draws)(grid)


# ---------------------------------------------------------------------------
# 1. natural data
# ---------------------------------------------------------------------------

def mirror_gap(x, h):
    """log p(mirror twin) - log p(point), per draw; x (n, 2), h (draws, n)."""
    return (-2.0 * x[None, :, 0] * (x[None, :, 0] - h) / E.SD_H ** 2).sum(1)


def natural():
    rows = []
    settings = [("single", None, 1)] + [("hier", l, n) for l in ("angle", "angle_height") for n in (2, 8, 32, 64, 128)]
    for kind, layout, n in settings:
        folder = datasets.folder(kind, layout, n)
        if not (folder / "reference.npz").exists():
            continue
        x = np.load(folder / "data.npz")["x"]
        theta = np.load(folder / "reference.npz")["theta"].astype(np.float64)
        truth = np.load(folder / "data.npz")["theta"][:, 0, 0]
        bimodal, gaps, flips, peaks = [], [], [], []
        for i in range(len(x)):
            # KDE of the reference draws (validated against the exact grid in check_reference.py)
            grid = np.linspace(theta[i, :, 0, 0].min() - 0.05, theta[i, :, 0, 0].max() + 0.05, 801)
            dens = kde_density(theta[i, :, 0, 0], grid)
            if layout != "angle_height":
                gaps.append(float(np.median(mirror_gap(x[i], theta[i, :, :, 1]))))
            s = shape(grid, dens)
            bimodal.append(s["bimodal"])
            peaks.append(s["peaks"])
            # does the posterior put its mode on the wrong sign? (only meaningful when |truth| is resolved)
            sd = theta[i, :, 0, 0].std()
            if abs(truth[i]) > 2 * sd:
                flips.append(np.sign(s["mode"]) != np.sign(truth[i]))
        row = {"setting": folder.name, "datasets": len(x), "bimodal_fraction": float(np.mean(bimodal)),
               "bimodal_examples": [p for p, b in zip(peaks, bimodal) if b][:5],
               "mode_on_wrong_side_fraction": float(np.mean(flips)) if flips else None,
               "median_mirror_gap_nats": float(np.median(gaps)) if gaps else None,
               "p10_mirror_gap_nats": float(np.quantile(gaps, 0.9)) if gaps else None}
        rows.append(row)
        print(f"{folder.name:18s} bimodal {row['bimodal_fraction']:.2f}  mirror gap median "
              f"{row['median_mirror_gap_nats'] if gaps else float('nan'):8.1f} nats  "
              f"wrong-side mode {row['mode_on_wrong_side_fraction']}", flush=True)
    # the per-observation mirror cost under the prior predictive, at the true h
    rng = sim.generator("mirror-cost")
    th = rng.normal(0.0, E.PRIOR_SD, (1_000_000, 4))
    import hierarchical as H
    x1 = H.forward(th)[:, 0]
    per_obs = -2.0 * x1 * (x1 - th[:, 1]) / E.SD_H ** 2
    prior_cost = {"mean": float(per_obs.mean()), "median": float(np.median(per_obs)),
                  "fraction_below_1_nat": float(np.mean(per_obs > -1.0)),
                  "fraction_mirror_favoured": float(np.mean(per_obs > 0))}
    print("per-observation mirror gap under the prior predictive:", prior_cost)
    return rows, per_obs, prior_cost


# ---------------------------------------------------------------------------
# 2. constructed bimodal data
# ---------------------------------------------------------------------------

def p_rail_selected(a1, tol, a2_nodes=4001):
    """P(|S(a1, a2, a3)| < tol | a1), S = L1 sin a1 + L2 sin(a1+a2) + L3 sin(a1+a2+a3), exactly:
    for fixed a2 the condition is sin(phi3) in an interval, i.e. phi3 in two arcsin intervals."""
    a2 = np.linspace(-6 * E.SD_A2, 6 * E.SD_A2, a2_nodes)
    w2 = np.exp(-0.5 * (a2 / E.SD_A2) ** 2)
    w2 /= w2.sum()
    out = []
    for a in np.atleast_1d(a1):
        c = -(E.L1 * np.sin(a) + E.L2 * np.sin(a + a2)) / E.L3
        lo, hi = np.clip(c - tol / E.L3, -1, 1), np.clip(c + tol / E.L3, -1, 1)
        total = np.zeros_like(a2)
        for k in (-1, 0, 1):
            for left, right in ((np.arcsin(lo), np.arcsin(hi)), (np.pi - np.arcsin(hi), np.pi - np.arcsin(lo))):
                shift = 2 * np.pi * k - a - a2
                total += ndtr((right + shift) / E.SD_A3) - ndtr((left + shift) / E.SD_A3)
        out.append(float((w2 * total).sum()))
    return np.array(out)


def p_x1_selected(a1, eps, nodes=120):
    """P(|x1| < eps | a1) with x1 = h + S(a1, a2, a3), by 2-D Gauss-Hermite over (a2, a3)."""
    z, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    A2, A3 = np.meshgrid(z * E.SD_A2, z * E.SD_A3, indexing="ij")
    W = np.outer(w, w)
    out = []
    for a in np.atleast_1d(a1):
        S = E.L1 * np.sin(a) + E.L2 * np.sin(a + A2) + E.L3 * np.sin(a + A2 + A3)
        out.append(float((W * (ndtr((eps - S) / E.SD_H) - ndtr((-eps - S) / E.SD_H))).sum()))
    return np.array(out)


def log_lik_rail_window(a1, x1, x2, tol, cells=8192):
    """log int N(h) p(x | a1, h) 1{|x1 - h| < tol} dh, midpoint rule in the regularised variable."""
    d, lo, hi, reach = E.height_support(a1, x1, x2)
    t = (np.arange(cells // 2) + 0.5) * np.pi / (cells // 2)
    u = lo + (hi - lo) * 0.5 * (1 - np.cos(t))
    jac = (hi - lo) * 0.5 * np.sin(t) * (np.pi / (cells // 2))
    total = 0.0
    for side in (1.0, -1.0):
        h = d - side * u
        inside = np.abs(x1 - h) < tol
        if inside.any():
            ll = E.log_normal(h[inside], E.SD_H) + E.log_lik_a1h(a1, h[inside], x1, x2, wraps=(0,))
            total += np.sum(np.exp(ll) * jac[inside])
    return np.log(total) if (reach and total > 0) else -np.inf


def constructed():
    datasets_ = {
        "rail": (np.load(RAIL / "data_a0.6_tol0.05.npy"), "rows with |x1 - h| < 0.05"),
        "x1sel": (selected_dataset(128, TRUE, 0.15, sim.generator("bimodality-x1sel")), "rows with |x1| < 0.15"),
    }
    results, curves = {}, {}
    for name, (x, label) in datasets_.items():
        x = x[:128]
        # per-row log-likelihood tables on the grid
        standard = np.stack([E.log_marginal_a1(GRID, x[j, 0], x[j, 1], nodes=64, wraps=(0,)) for j in range(len(x))], 1)
        if name == "rail":
            coarse = np.linspace(-2, 2, 401)
            log_sel = np.interp(GRID, coarse, np.log(p_rail_selected(coarse, 0.05)))
            window = np.stack([[log_lik_rail_window(a, x[j, 0], x[j, 1], 0.05) for a in GRID[::5]]
                               for j in range(len(x))], 1)
            numer = np.stack([np.interp(GRID, GRID[::5], window[:, j]) for j in range(len(x))], 1)
            aware = numer - log_sel[:, None]
        else:
            coarse = np.linspace(-2, 2, 401)
            log_sel = np.interp(GRID, coarse, np.log(p_x1_selected(coarse, 0.15)))
            aware = standard - log_sel[:, None]
        prior = E.log_normal(GRID, E.SD_A1)
        results[name] = {"label": label, "x1_abs_mean": float(np.abs(x[:, 0]).mean()), "rows": {}}
        curves[name] = {}
        for n in NS:
            entry = {}
            for kind, table in (("standard", standard), ("selection_aware", aware)):
                lp = prior + table[:, :n].sum(1)
                p = np.exp(lp - lp.max())
                p /= np.trapezoid(p, GRID)
                curves[name][(kind, n)] = p
                entry[kind] = shape(GRID, p)
            results[name]["rows"][n] = entry
            print(f"{name:6s} n={n:3d}  standard: peaks {entry['standard']['peaks']} P(a1>0) "
                  f"{entry['standard']['p_positive']:.2f} E|a1| {entry['standard']['mean_abs']:.2f} | "
                  f"selection-aware: peaks {entry['selection_aware']['peaks']} P(a1>0) "
                  f"{entry['selection_aware']['p_positive']:.2f} E|a1| {entry['selection_aware']['mean_abs']:.2f}", flush=True)
    return results, curves


def figures(curves, rows, per_obs):
    OUT.mkdir(parents=True, exist_ok=True)
    abc = np.load(RAIL / "reference_a0.6_tol0.05.npz")
    abc_small = np.load(RAIL / "reference_a0.6_tol0.05_h0.0125.npz")
    fig, axes = plt.subplots(2, len(NS), figsize=(2.1 * len(NS), 4.6), constrained_layout=True, sharex=True)
    for r, name in enumerate(("rail", "x1sel")):
        for c, n in enumerate(NS):
            ax = axes[r, c]
            ax.plot(GRID, curves[name][("standard", n)], color=INK, lw=1.3, label="exact, standard model")
            ax.plot(GRID, curves[name][("selection_aware", n)], color=BLUE, lw=1.3, label="exact, selection-aware")
            if name == "rail":
                if f"n{n}" in abc:
                    ax.plot(abc["grid"], abc[f"n{n}"], color=MUTED, lw=1, ls="--", label="old ABC reference, h = 0.05")
                if f"n{n}" in abc_small:
                    ax.plot(abc_small["grid"], abc_small[f"n{n}"], color=MUTED, lw=1, ls=":",
                            label="old ABC reference, h = 0.0125")
                path = COMPASS_RAIL / f"samples_gauss_hierarchical_n{n}.npy"
                if path.exists():
                    s = np.load(path)
                    ax.hist(s, bins=np.linspace(-2, 2, 81), density=True, histtype="step", color=ORANGE, lw=1.2,
                            label="COMPASS gauss_hierarchical")
            for v in (-TRUE, TRUE):
                ax.axvline(v, color=MUTED, lw=0.7, ls=(0, (2, 2)))
            ax.set_yticks([])
            ax.set_xlim(-1.5, 1.5)
            if r == 0:
                ax.set_title(f"n = {n}", fontsize=9)
            if r == 1:
                ax.set_xlabel("angle_1")
        axes[r, 0].set_ylabel({"rail": "rail-height data\n|x1 - h| < 0.05", "x1sel": "near-rail tip\n|x1| < 0.15"}[name])
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Constructed bimodal data, true angle_1 = 0.6 (dotted: +-0.6)", fontsize=10)
    fig.savefig(OUT / "constructed_posteriors.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), constrained_layout=True)
    axes[0].hist(np.clip(per_obs, -60, 5), bins=np.linspace(-60, 5, 66), color=BLUE)
    axes[0].set_xlabel("log p(mirror twin) - log p(truth), one observation (nats; clipped at -60)")
    axes[0].set_ylabel("prior-predictive draws")
    angle_rows = [r for r in rows if r["median_mirror_gap_nats"] is not None]
    ns = [1 if r["setting"] == "single" else int(r["setting"].split("_n")[-1]) for r in angle_rows]
    axes[1].plot(ns, [-r["median_mirror_gap_nats"] for r in angle_rows], "o-", color=BLUE)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("observations n")
    axes[1].set_ylabel("median mirror penalty (nats)")
    axes[1].set_title("exact posterior, angle_1 global", fontsize=9)
    fig.savefig(OUT / "mirror_penalty.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_obs, prior_cost = natural()
    results, curves = constructed()
    figures(curves, rows, per_obs)
    (OUT / "bimodality.json").write_text(json.dumps({"natural": rows, "prior_predictive_mirror_gap": prior_cost,
                                                     "constructed": results}, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
