"""Figures from the saved summaries (analyze.py); no recomputation.

    .COMPASS/bin/python -m validation.a4_waic.plots --family offset --compass-tag oracle_d1000
"""

import validation.common.runtime  # noqa: F401

import argparse
import json
import os

import numpy as np

from validation.a4_waic import common
from validation.common import plotting
from validation.common.plotting import EXACT, FOURTH, INK2, LEGACY, NEW, plt

FIG = os.path.join(common.RESULTS, "figures")
LABELS = {"compass_waic": ("COMPASS WAIC weight", NEW),
          "compass_pbma_plus": ("COMPASS pseudo-BMA+", FOURTH),
          "quad_waic": ("exact WAIC weight", EXACT), "mcmc_waic": ("MCMC WAIC weight", EXACT),
          "ns_waic": ("NS WAIC weight", INK2),
          "quad_evidence": ("exact evidence P(M|x)", LEGACY),
          "ns_evidence": ("NS evidence P(M|x)", LEGACY)}


def waic_scatter(s, name):
    rows = [r for r in s["datasets"] if "compass" in r and s["ref_key"] in r]
    names, ref = s["models"], s["ref_key"]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]][:3]
    fig, axes = plt.subplots(1, len(names) + len(pairs), figsize=(3.4 * (len(names) + len(pairs)), 3.4))
    for ax, n in zip(axes, names):
        x = np.array([r[ref][n] for r in rows]); y = np.array([r["compass"][n] for r in rows])
        se = np.array([r["compass_mc_se"][n] for r in rows])
        ax.errorbar(x, y - x, yerr=se, fmt="o", ms=3, color=NEW, alpha=0.6, elinewidth=0.8)
        ax.axhline(0, color=EXACT)
        if np.ptp(y - x) > 100:   # misspecified-model outliers: keep the bulk readable
            ax.set_yscale("symlog", linthresh=10)
        e = s["waic_error"][n]
        ax.set_title(f"{n}: bias {e['bias']:+.2f}, sd {e['sd']:.2f}")
        ax.set_xlabel(f"{ref} WAIC"); ax.set_ylabel("COMPASS − reference")
    for ax, (a, b) in zip(axes[len(names):], pairs):
        x = np.array([r[ref][a] - r[ref][b] for r in rows])
        y = np.array([r["compass"][a] - r["compass"][b] for r in rows])
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color=EXACT); ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
        ax.scatter(x, y, s=8, color=NEW, alpha=0.6)
        if max(np.ptp(x), np.ptp(y)) > 100:
            ax.set_xscale("symlog", linthresh=10); ax.set_yscale("symlog", linthresh=10)
        d = s["delta_error"][f"{a}-{b}"]
        ax.set_title(f"ΔWAIC {a}−{b}: sign agree {d['sign_agreement']:.1%}")
        ax.set_xlabel(f"{ref} ΔWAIC"); ax.set_ylabel("COMPASS ΔWAIC")
    return plotting.save(fig, FIG, f"fig_waic_error_{name}")


def reliability_plot(s, name):
    labels = [k for k in LABELS if k in s["calibration"]]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, labels):
        cal = s["calibration"][k]; title, color = LABELS[k]
        b = cal["bins"]
        p = [x["mean_prob"] for x in b]; f = [x["freq"] for x in b]
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        ax.errorbar(p, f, yerr=[[fi - x["lo"] for fi, x in zip(f, b)], [x["hi"] - fi for fi, x in zip(f, b)]],
                    fmt="o-", color=color, ms=4)
        for pi, fi, x in zip(p, f, b):
            ax.annotate(str(x["n"]), (pi, fi), fontsize=6, color=INK2, xytext=(3, -8), textcoords="offset points")
        ax.set_title(f"{title}\nECE {cal['ece']:.3f} (null95 {cal['ece_null_q95']:.3f})", fontsize=9)
        ax.set_xlabel("reported probability"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("frequency of true model")
    return plotting.save(fig, FIG, f"fig_reliability_{name}")


def confusion_plot(s, name):
    labels = [k for k in LABELS if k in s["recovery"]]
    names = s["models"]
    fig, axes = plt.subplots(1, len(labels), figsize=(2.6 * len(labels) + 0.5, 2.8))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, labels):
        m = np.array(s["recovery"][k]["confusion"], float)
        frac = m / m.sum(1, keepdims=True).clip(1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center",
                        color="white" if frac[i, j] > 0.6 else "black", fontsize=8)
        ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
        ax.set_title(f"{LABELS[k][0]}\nacc {s['recovery'][k]['accuracy']:.3f}", fontsize=8)
        ax.set_xlabel("selected"); ax.grid(False)
    axes[0].set_ylabel("true")
    return plotting.save(fig, FIG, f"fig_confusion_{name}")


def reference_plot(s, name):
    names, ref = s["models"], s["ref_key"]
    others = [k for k in ("mcmc", "ns") if k != ref and any(k in r for r in s["datasets"])]
    has_z = any("ns_logz" in r and "quad_logz" in r for r in s["datasets"])
    n = len(others) + int(has_z)
    if n == 0:
        return None
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, others):
        rows = [r for r in s["datasets"] if k in r and ref in r]
        for n_, c in zip(names, (NEW, LEGACY, EXACT, FOURTH)):
            x = np.array([r[ref][n_] for r in rows]); y = np.array([r[k][n_] for r in rows])
            ax.scatter(x, y - x, s=8, color=c, alpha=0.7, label=n_)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{k} WAIC − {ref} WAIC"); ax.set_xlabel(f"{ref} WAIC"); ax.legend(fontsize=7)
    if has_z:
        ax = axes[-1]
        rows = [r for r in s["datasets"] if "ns_logz" in r and "quad_logz" in r]
        z = np.concatenate([[(r["ns_logz"][m] - r["quad_logz"][m]) / r["ns_logz_err"][m] for r in rows] for m in names])
        ax.hist(z, bins=20, density=True, color=NEW, alpha=0.7)
        g = np.linspace(-4, 4, 200); ax.plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), color=EXACT)
        ax.set_title(f"NS log Z z-score (mean {z.mean():+.2f}, sd {z.std():.2f})"); ax.set_xlabel("z")
    return plotting.save(fig, FIG, f"fig_references_{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--compass-tag", required=True)
    args = parser.parse_args()
    plotting.style()
    name = f"{args.family}_{args.compass_tag}"
    with open(os.path.join(common.RESULTS, f"summary_{name}.json")) as f:
        s = json.load(f)
    for fn in (waic_scatter, reliability_plot, confusion_plot, reference_plot):
        try:
            print(fn(s, name))
        except (ValueError, KeyError, IndexError) as err:
            print(f"{fn.__name__}: skipped ({err})")


if __name__ == "__main__":
    main()
