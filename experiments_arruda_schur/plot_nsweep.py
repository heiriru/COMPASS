"""Recovery vs number of observations: `hierarchical` attention + gauss_jacobian_FD, churn 2 and 4.

    python plot_nsweep.py

Reads ``artifacts/pp/hierarchical/recovery/rerun/<method>-s<n>-t50`` for n in
NS (n = subjects composed per dataset), and writes to ``artifacts/figures``:

- ``12_nsweep_recovery_<method>.png``: posterior median vs truth per global
  (rows n, columns globals), 5-95 % interval as a bar, r in each panel;
- ``13_nsweep_metrics.png``: global/local std-RMSE, mean r, per-global r and
  95 % coverage against n for both churn settings, C2 `none` at n = 100 for
  reference.
"""
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path[:0] = [str(common.COMPASS)]
import Partial_Pooling.runtime as rt  # noqa: E402

rt.configure_cpu_limit(3)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.model_pipeline import load_normalizers  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402
from Partial_Pooling.schema import GLOBAL_NAMES  # noqa: E402

NS = (10, 25, 50, 100)
ARMS = {"gauss_jacobian_FD": ("churn 2", "#2a78b5"),
        "gauss_jacobian_FD_churn4": ("churn 4", "#d0612b")}
REFERENCE = ("C2 none / gauss_jacobian_FD", "none", "gauss_jacobian_FD", "#6b6a63")
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e2da"
OUT = HERE / "artifacts" / "figures"
plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "legend.frameon": False, "figure.dpi": 150,
})


def load(mode, method, n, scale):
    files = sorted(glob.glob(str(HERE / "artifacts/pp" / mode / "recovery/rerun"
                                 / f"{method}-s{n}-t50" / "dataset-*.pt")))
    if not files:
        return None
    med, lo, hi, truth, g_err, l_err = [], [], [], [], [], []
    for f in files:
        o = torch.load(f, map_location="cpu", weights_only=False)
        g = o["posterior"]["globals"]
        m = g.median(0).values
        med.append(m.numpy())
        lo.append(g.quantile(0.05, 0).numpy())
        hi.append(g.quantile(0.95, 0).numpy())
        truth.append(o["truth_globals"].numpy())
        g_err.append(float((((m - o["truth_globals"]) / scale[:7]) ** 2).mean().sqrt()))
        loc = o["posterior"]["locals"].median(0).values
        l_err.append(float((((loc - o["truth_locals"]) / scale[7:]) ** 2).mean().sqrt()))
    med, lo, hi, truth = map(np.array, (med, lo, hi, truth))
    r = np.array([np.corrcoef(med[:, k], truth[:, k])[0, 1] for k in range(7)])
    return dict(med=med, lo=lo, hi=hi, truth=truth, r=r, n_sets=len(files),
                g=np.mean(g_err), l=np.mean(l_err),
                cover90=((truth >= lo) & (truth <= hi)).mean(0))


def recovery_figure(method, label, colour, runs):
    rows = [n for n in NS if runs.get(n)]
    fig, axes = plt.subplots(len(rows), 7, figsize=(15, 2.15 * len(rows)), squeeze=False)
    for i, n in enumerate(rows):
        run = runs[n]
        for k in range(7):
            ax = axes[i, k]
            t, m = run["truth"][:, k], run["med"][:, k]
            ax.vlines(t, run["lo"][:, k], run["hi"][:, k], color=colour, alpha=0.35, lw=1)
            ax.scatter(t, m, s=10, color=colour, edgecolor="white", linewidth=0.4, zorder=3)
            span = np.array([min(t.min(), run["lo"][:, k].min()), max(t.max(), run["hi"][:, k].max())])
            ax.plot(span, span, color=MUTED, lw=0.8, ls="--", zorder=1)
            ax.text(0.04, 0.93, f"r = {run['r'][k]:.2f}", transform=ax.transAxes, va="top", fontsize=7.5)
            ax.grid(color=GRID, lw=0.5)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(GLOBAL_NAMES[k], fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("truth", fontsize=7.5)
            if k == 0:
                ax.set_ylabel(f"n = {n}\nposterior median", fontsize=8)
    fig.suptitle(f"Recovery of the globals, `hierarchical` attention + gauss_jacobian_FD, {label} "
                 f"({runs[rows[0]]['n_sets']} datasets; bars = 5–95 % posterior interval)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / f"12_nsweep_recovery_{method}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def metrics_figure(all_runs, reference):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    panels = [("g", "global std-RMSE (lower better)"), ("l", "local std-RMSE (lower better)"),
              ("mean_r", "mean r over globals (higher better)"), ("cover", "90 % coverage of globals")]
    for ax, (key, title) in zip(axes[0], panels):
        for method, (label, colour) in ARMS.items():
            ns = [n for n in NS if all_runs[method].get(n)]
            vals = [_metric(all_runs[method][n], key) for n in ns]
            ax.plot(ns, vals, "-o", color=colour, lw=2, ms=6, label=label)
            ax.text(ns[-1], vals[-1], f"  {vals[-1]:.3f}", va="center", fontsize=7.5, color=INK)
        if reference:
            ax.plot([100], [_metric(reference, key)], "D", color=REFERENCE[3], ms=6, label=REFERENCE[0])
        if key == "mean_r":
            ax.axhline(0.870, color=MUTED, lw=0.8, ls=":")
            ax.text(10, 0.871, "paper 0.870", fontsize=7, color=MUTED, va="bottom")
        if key == "cover":
            ax.axhline(0.90, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.minorticks_off()
        ax.set_xlim(8, 125)
        ax.set_xlabel("observations n (subjects per dataset)")
        ax.set_title(title, fontsize=9)
        ax.grid(color=GRID, lw=0.5)
    axes[0, 0].legend(fontsize=7.5)
    # per-global r, one panel per churn plus the difference
    for ax, method in zip(axes[1, :2], ARMS):
        label, _ = ARMS[method]
        ns = [n for n in NS if all_runs[method].get(n)]
        cmap = plt.get_cmap("tab10")
        for k in range(7):
            ax.plot(ns, [all_runs[method][n]["r"][k] for n in ns], "-o", ms=4, lw=1.5,
                    color=cmap(k), label=GLOBAL_NAMES[k])
        ax.set_xscale("log")
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.minorticks_off()
        ax.set_xlim(8, 125)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"per-global r, {label}", fontsize=9)
        ax.set_xlabel("observations n")
        ax.grid(color=GRID, lw=0.5)
    axes[1, 0].legend(fontsize=6.5, ncol=2, loc="lower right")
    # per-global posterior width (mean 5-95 % width / prior-ish scale) vs n
    for ax, method in zip(axes[1, 2:], ARMS):
        label, _ = ARMS[method]
        ns = [n for n in NS if all_runs[method].get(n)]
        cmap = plt.get_cmap("tab10")
        for k in range(7):
            ax.plot(ns, [all_runs[method][n]["cover90"][k] for n in ns], "-o", ms=4, lw=1.5, color=cmap(k))
        ax.axhline(0.90, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.minorticks_off()
        ax.set_xlim(8, 125)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"per-global 90 % coverage, {label}", fontsize=9)
        ax.set_xlabel("observations n")
        ax.grid(color=GRID, lw=0.5)
    fig.suptitle("`hierarchical` attention (g→g, l→l, l→g; g never → l) + gauss_jacobian_FD: "
                 "how recovery evolves with the number of observations", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "13_nsweep_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _metric(run, key):
    if key == "mean_r":
        return float(run["r"].mean())
    if key == "cover":
        return float(run["cover90"].mean())
    return float(run[key])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scale = load_normalizers(get_config("small"), BenchmarkPaths.default())["sde_joint_theta"].scale
    all_runs = {m: {n: load("hierarchical", m, n, scale) for n in NS} for m in ARMS}
    reference = load(REFERENCE[1], REFERENCE[2], 100, scale)
    print(f"{'rule':26s} {'n':>4s} {'sets':>4s} {'glob':>6s} {'loc':>6s} {'mean r':>6s} {'cov90':>6s}  per-global r")
    for method, runs in all_runs.items():
        for n, run in runs.items():
            if run:
                print(f"{method:26s} {n:4d} {run['n_sets']:4d} {run['g']:6.3f} {run['l']:6.3f} "
                      f"{run['r'].mean():6.3f} {run['cover90'].mean():6.3f}  "
                      + " ".join(f"{x:.2f}" for x in run["r"]))
    for method, (label, colour) in ARMS.items():
        print(recovery_figure(method, label, colour, all_runs[method]))
    print(metrics_figure(all_runs, reference))


if __name__ == "__main__":
    main()
