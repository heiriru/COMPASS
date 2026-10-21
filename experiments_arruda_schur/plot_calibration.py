#!/usr/bin/env python3
"""Coverage and calibration of the partial-pooling C2 posteriors: Rg network trained on the
1.5x-wide global prior (set_training_prior composition), next to plain Rg as the reference.

Per parameter, from the fractional rank u = P_draws(theta < truth) of each truth:
  - coverage: share of truths inside the central credible interval, against its nominal level
  - SBC ECDF difference: ECDF(u) - u (0 everywhere = calibrated; below 0 in the middle and
    above at the ends = too narrow; a one-signed hump = biased)
  - recovery: posterior median and 90% interval against the truth (globals)
Bands: pointwise 95% binomial for the number of truths. Locals pool 30 datasets x 100 subjects;
subjects of one dataset share the globals' error, so their band is optimistic.

    python plot_calibration.py   -> artifacts/figures/17_wide_calibration.png, 18_wide_recovery.png
"""
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

HERE = Path(__file__).resolve().parent
PP = HERE / "artifacts" / "pp"
ARMS = [("Rg, training prior 1.5x wide", PP / "hierarchical_Rg_wide1.5/recovery/rerun/gauss_jacobian_FD-s100-t50",
         "#eda100", 2.0),
        ("Rg (reference)", PP / "hierarchical_Rg/recovery/rerun/gauss_jacobian_FD-s100-t50", "#8c8b85", 1.2)]
GLOBALS = ("mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu", "log_sigma_log_alpha",
           "log_sigma_log_t0", "beta_raw")
LOCALS = ("nu (local)", "log_alpha (local)", "log_t0 (local)")
INK, MUTED, BAND = "#1f1f1e", "#6b6a63", "#e6e5df"
LEVELS = np.linspace(0, 1, 101)
RNG = np.random.default_rng(0)


def load(directory):
    """Fractional ranks per parameter, plus medians / intervals / truths of the globals."""
    ranks = {p: [] for p in GLOBALS + LOCALS}
    rec = {"truth": [], "median": [], "lo": [], "hi": []}
    for path in sorted(directory.glob("dataset-*.pt")):
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
    return {p: np.array(v) for p, v in ranks.items()}, {k: np.array(v) for k, v in rec.items()}


def rank(draws, truth):
    below, ties = np.sum(draws < truth), np.sum(draws == truth)
    return (below + RNG.uniform() * (ties + 1)) / (len(draws) + 1)


def coverage(u):
    return np.array([np.mean(np.abs(u - 0.5) <= a / 2) for a in LEVELS])


def band(n, center):
    half = 1.96 * np.sqrt(np.clip(center * (1 - center), 1e-9, None) / n)
    return center - half, center + half


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6.5, colors=MUTED)


def calibration_figure(data):
    params = GLOBALS + LOCALS
    fig, axes = plt.subplots(2, len(params), figsize=(2.05 * len(params), 4.6), sharey="row")
    for j, p in enumerate(params):
        n = len(data[0][1][0][p])
        top, bottom = axes[0, j], axes[1, j]
        lo, hi = band(n, LEVELS)
        top.fill_between(LEVELS, lo, hi, color=BAND, lw=0)
        top.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
        bottom.fill_between(LEVELS, *(b - LEVELS for b in band(n, LEVELS)), color=BAND, lw=0)
        bottom.axhline(0, color=MUTED, lw=0.7, ls=":")
        notes = []
        for label, (ranks, _), color, width in data:
            u = ranks[p]
            c = coverage(u)
            top.plot(LEVELS, c, color=color, lw=width)
            ecdf = np.array([np.mean(u <= z) for z in LEVELS])
            bottom.plot(LEVELS, ecdf - LEVELS, color=color, lw=width)
            notes.append(f"{np.mean(np.abs(u - 0.5) <= 0.45):.2f}")
        top.set_title(f"{p}\n90% cov. {' / '.join(notes)}", fontsize=7, color=INK)
        top.set_xticks([0, 0.5, 1])
        bottom.set_xticks([0, 0.5, 1])
        bottom.set_xlabel("fractional rank", fontsize=6.5, color=MUTED)
        top.set_xlabel("nominal level", fontsize=6.5, color=MUTED)
        style(top)
        style(bottom)
    axes[0, 0].set_ylabel("empirical coverage", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("ECDF(rank) − rank", fontsize=7.5, color=INK)
    handles = [plt.Line2D([], [], color=c, lw=w, label=l) for l, _, c, w in data]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND, label="95% band if calibrated"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    n_sets = len(data[0][1][0][GLOBALS[0]])
    fig.suptitle(f"Partial pooling C2, hierarchical attention + gauss_jacobian_FD: coverage (top) and SBC "
                 f"rank ECDF (bottom); {n_sets} datasets, locals pooled over 100 subjects each "
                 f"(titles: 90% coverage, wide / reference)", fontsize=9, color=INK, y=1.12)
    fig.tight_layout()
    out = HERE / "artifacts" / "figures" / "17_wide_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def recovery_figure(data):
    fig, axes = plt.subplots(1, len(GLOBALS), figsize=(2.1 * len(GLOBALS), 2.6))
    for k, (p, ax) in enumerate(zip(GLOBALS, axes)):
        for label, (_, rec), color, width in data[::-1]:
            t, m = rec["truth"][:, k], rec["median"][:, k]
            ax.errorbar(t, m, yerr=[m - rec["lo"][:, k], rec["hi"][:, k] - m], fmt="o", ms=3 if width > 1.5 else 2.4,
                        color=color, ecolor=color, elinewidth=0.6, alpha=0.9 if width > 1.5 else 0.55, capsize=0)
        allv = np.concatenate([d[1][1]["truth"][:, k] for d in data] + [d[1][1]["median"][:, k] for d in data])
        lo, hi = allv.min(), allv.max()
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, ls=":")
        r = [np.corrcoef(d[1][1]["truth"][:, k], d[1][1]["median"][:, k])[0, 1] for d in data]
        ax.set_title(f"{p}\nr {r[0]:.2f} / {r[1]:.2f}", fontsize=7, color=INK)
        ax.set_xlabel("truth", fontsize=6.5, color=MUTED)
        style(ax)
    axes[0].set_ylabel("posterior median, 90% interval", fontsize=7, color=INK)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0, label=l) for l, _, c, w in data]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    out = HERE / "artifacts" / "figures" / "18_wide_recovery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def main():
    data = [(label, load(directory), color, width) for label, directory, color, width in ARMS]
    calibration_figure(data)
    recovery_figure(data)


if __name__ == "__main__":
    main()
