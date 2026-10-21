#!/usr/bin/env python3
"""Coverage, calibration and recovery of the wide-prior Rg network (set_training_prior
composition, hierarchical attention + gauss_jacobian_FD) as the number of composed subjects
grows: n = 1, 4, 25, 100 (the first n subjects of the same 30 test datasets).

    python plot_calibration_nsweep.py -> artifacts/figures/19_wide_nsweep_calibration.png,
                                          20_wide_nsweep_recovery.png
    python plot_calibration_nsweep.py --run hierarchical_Rg_wide1.5_n65536 --prefix 21_budget65k \
        --label "65,536-sim budget"     (another network's n-sweep, its own file names)
"""
import argparse
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "3")
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from plot_calibration import GLOBALS, LEVELS, LOCALS, coverage, rank, style  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE / "artifacts" / "pp" / "hierarchical_Rg_wide1.5" / "recovery" / "rerun"
NAMES = {"calibration": "19_wide_nsweep_calibration", "recovery": "20_wide_nsweep_recovery"}
LABEL = "Wide-prior Rg"
NS = (1, 4, 25, 100)
# one hue, light -> dark with n (sequential: the encoding is magnitude)
COLORS = dict(zip(NS, ("#a9cbee", "#5f9fe0", "#2a6fc0", "#0d3a73")))
INK, MUTED, BAND = "#1f1f1e", "#6b6a63", "#e6e5df"


def load(n):
    ranks = {p: [] for p in GLOBALS + LOCALS}
    rec = {"truth": [], "median": [], "lo": [], "hi": [], "ltruth": [], "lmedian": []}
    for path in sorted((ROOT / f"gauss_jacobian_FD-s{n}-t50").glob("dataset-*.pt")):
        o = torch.load(path, map_location="cpu", weights_only=False)
        g, l = o["posterior"]["globals"].numpy(), o["posterior"]["locals"].numpy()
        tg, tl = o["truth_globals"].numpy(), o["truth_locals"].numpy()
        for k, p in enumerate(GLOBALS):
            ranks[p].append(rank(g[:, k], tg[k]))
        for k, p in enumerate(LOCALS):
            ranks[p].extend(rank(l[:, j, k], tl[j, k]) for j in range(tl.shape[0]))
        rec["truth"].append(tg)
        rec["median"].append(np.median(g, 0))
        rec["lo"].append(np.quantile(g, 0.05, 0))
        rec["hi"].append(np.quantile(g, 0.95, 0))
        rec["ltruth"].append(tl)
        rec["lmedian"].append(np.median(l, 0))
    out = {k: np.concatenate(v) if k in ("ltruth", "lmedian") else np.array(v) for k, v in rec.items()}
    return {p: np.array(v) for p, v in ranks.items()}, out


def calibration(data):
    params = GLOBALS + LOCALS
    fig, axes = plt.subplots(2, len(params), figsize=(2.05 * len(params), 4.7), sharey="row")
    sets = len(data[NS[0]][0][GLOBALS[0]])
    for j, p in enumerate(params):
        top, bottom = axes[0, j], axes[1, j]
        half = 1.96 * np.sqrt(LEVELS * (1 - LEVELS) / sets)  # band for 30 truths (the globals, and locals at n=1)
        top.fill_between(LEVELS, LEVELS - half, LEVELS + half, color=BAND, lw=0)
        top.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
        bottom.fill_between(LEVELS, -half, half, color=BAND, lw=0)
        bottom.axhline(0, color=MUTED, lw=0.7, ls=":")
        notes = []
        for n in NS:
            u = data[n][0][p]
            top.plot(LEVELS, coverage(u), color=COLORS[n], lw=1.6)
            bottom.plot(LEVELS, np.array([np.mean(u <= z) for z in LEVELS]) - LEVELS, color=COLORS[n], lw=1.6)
            notes.append(f"{np.mean(np.abs(u - 0.5) <= 0.45):.2f}")
        top.set_title(f"{p}\n90% cov. {' '.join(notes)}", fontsize=6.8, color=INK)
        for ax in (top, bottom):
            ax.set_xticks([0, 0.5, 1])
            style(ax)
        top.set_xlabel("nominal level", fontsize=6.5, color=MUTED)
        bottom.set_xlabel("fractional rank", fontsize=6.5, color=MUTED)
    axes[0, 0].set_ylabel("empirical coverage", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("ECDF(rank) − rank", fontsize=7.5, color=INK)
    handles = [plt.Line2D([], [], color=COLORS[n], lw=1.6, label=f"n = {n} subjects") for n in NS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND, label=f"95% band if calibrated, {sets} truths"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"{LABEL}, hierarchical attention + gauss_jacobian_FD: calibration vs number of composed "
                 f"subjects ({sets} datasets; titles: 90% coverage for n = {', '.join(map(str, NS))}; locals pool "
                 "n subjects per dataset)", fontsize=9, color=INK, y=1.12)
    fig.tight_layout()
    out = HERE / "artifacts" / "figures" / f"{NAMES['calibration']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def recovery(data):
    params = GLOBALS + LOCALS
    fig, axes = plt.subplots(len(NS), len(params), figsize=(2.05 * len(params), 2.0 * len(NS)))
    for k, p in enumerate(params):
        local = k >= len(GLOBALS)
        key_t, key_m = ("ltruth", "lmedian") if local else ("truth", "median")
        kk = k - len(GLOBALS) if local else k
        values = np.concatenate([np.concatenate([data[n][1][key_t][:, kk], data[n][1][key_m][:, kk]]) for n in NS])
        lo, hi = np.quantile(values, [0.002, 0.998]) if local else (values.min(), values.max())
        pad = 0.05 * (hi - lo)
        for i, n in enumerate(NS):
            ax = axes[i, k]
            rec = data[n][1]
            t, m = rec[key_t][:, kk], rec[key_m][:, kk]
            if local:
                ax.scatter(t, m, s=4 if n < 25 else 1.5, color=COLORS[n], alpha=0.8 if n < 25 else 0.35, lw=0)
            else:
                ax.errorbar(t, m, yerr=[m - rec["lo"][:, kk], rec["hi"][:, kk] - m], fmt="o", ms=2.8,
                            color=COLORS[n], ecolor=COLORS[n], elinewidth=0.6, capsize=0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.7, ls=":")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            r = np.corrcoef(t, m)[0, 1]
            rmse = np.sqrt(np.mean((t - m) ** 2))
            ax.set_title((f"{p}\n" if i == 0 else "") + f"r {r:.2f}  RMSE {rmse:.2f}", fontsize=6.8, color=INK)
            style(ax)
            if k == 0:
                ax.set_ylabel(f"n = {n}\nposterior median", fontsize=7.5, color=INK)
            if i == len(NS) - 1:
                ax.set_xlabel("truth", fontsize=6.5, color=MUTED)
    fig.suptitle(f"{LABEL}: recovery vs number of composed subjects (globals: median and 90% interval; "
                 "locals: median of each subject)", fontsize=9, color=INK, y=1.01)
    fig.tight_layout()
    out = HERE / "artifacts" / "figures" / f"{NAMES['recovery']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def main():
    global ROOT, LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="hierarchical_Rg_wide1.5", help="folder under artifacts/pp")
    parser.add_argument("--prefix", default="", help="output names <prefix>_nsweep_{calibration,recovery}.png")
    parser.add_argument("--label", default="", help="appended to the figure titles")
    arguments = parser.parse_args()
    ROOT = HERE / "artifacts" / "pp" / arguments.run / "recovery" / "rerun"
    if arguments.prefix:
        NAMES.update({k: f"{arguments.prefix}_nsweep_{k}" for k in NAMES})
    if arguments.label:
        LABEL = f"{LABEL} ({arguments.label})"
    data = {n: load(n) for n in NS}
    missing = [n for n in NS if len(data[n][0][GLOBALS[0]]) == 0]
    if missing:
        raise SystemExit(f"no runs yet for n = {missing}")
    calibration(data)
    recovery(data)


if __name__ == "__main__":
    main()
