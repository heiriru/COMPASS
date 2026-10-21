#!/usr/bin/env python3
"""One figure: hierarchical attention + gauss_jacobian_FD on every bimodal experiment with saved
draws, against the exact posterior of the global, at a small, middle and large n.

Reads only saved runs (nothing is sampled):
  abs / square / lean / curved toys  abs_hierarchy/artifacts[/<model>]/runs  (Gibbs MCMC reference)
  Gaussian mixture                   artifacts/gm/hierarchical/scaling/posteriors/draws.npz
  IK |x1|                            artifacts/ik_absx1  (exact grid)
  IK reach-only (sym, shift)         artifacts/ik_reach  (exact grid)

    python plot_hier_jac_overview.py      -> artifacts/figures/14_hier_jac_overview.png
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
from scipy.stats import wasserstein_distance  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TOYS = HERE.parent / "abs_hierarchy" / "artifacts"
INK, MUTED, REF, ARM = "#1f1f1e", "#6b6a63", "#d9d8d2", "#eda100"
RNG = np.random.default_rng(0)


def from_grid(grid, density, count=100_000):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return np.interp(RNG.random(count), cdf / cdf[-1], grid)


def toy(root, label):
    def load(n):
        ref = np.load(root / "runs" / f"mcmc_n{n}.npz")["g"]
        arm = np.load(root / "runs" / f"hierarchical_gauss_jacobian_FD_n{n}.npz")["g"]
        return ref, arm
    return label, "g", [(n, *load(n)) for n in (4, 16, 128)]


def gm(regime):
    z = np.load(ART / "gm" / "hierarchical" / "scaling" / "posteriors" / "draws.npz")
    return (f"Gaussian mixture, {regime}", "g",
            [(n, z[f"{regime}_n{n}_exact_mc"], z[f"{regime}_n{n}_gauss_jacobian_FD"]) for n in (8, 16, 64)])


def absx1():
    e = np.load(ART.parent.parent / "Arruda" / "artifacts" / "ik" / "exact" / "absx1_a0.6.npz")
    rows = []
    for n in (8, 32, 128):
        arm = np.load(ART / "ik_absx1" / "runs" / "hierarchical" / f"gauss_jacobian_FD_n{n}.npz")["theta"][:, 0, 0]
        rows.append((n, from_grid(e["grid"], e[f"n{n}"]), arm))
    return "IK, |x1| observed", "angle_1", rows


def reach(variant, label):
    e = np.load(ART / "ik_reach" / "exact.npz")
    rows = []
    for n in (8, 32, 128):
        arm = np.load(ART / "ik_reach" / "runs" / variant / f"gauss_jacobian_FD_n{n}.npz")["theta"][:, 0, 0]
        rows.append((n, from_grid(e["grid"], e[f"{variant}_n{n}"]), arm))
    return label, "angle_1", rows


EXPERIMENTS = [
    toy(TOYS, "toy l~N(|g|), exact 50/50"),
    toy(TOYS / "square", "toy l~N(g²), exact 50/50"),
    toy(TOYS / "lean", "toy l~N(|g|+0.2g), exact ~74/26"),
    toy(TOYS / "curved", "toy l~N(g²+0.4g), exact ~62/38"),
    gm("symmetric"),
    gm("asymmetric"),
    absx1(),
    reach("sym", "IK, reach only, exact 50/50"),
    reach("shift", "IK, reach only, exact ~67/33"),
]


def main():
    fig, axes = plt.subplots(len(EXPERIMENTS), 3, figsize=(11, 1.75 * len(EXPERIMENTS)), squeeze=False)
    for i, (label, var, rows) in enumerate(EXPERIMENTS):
        for j, (n, ref, arm) in enumerate(rows):
            ax = axes[i, j]
            lo, hi = np.quantile(np.concatenate([ref, arm]), [0.002, 0.998])
            pad = 0.08 * (hi - lo)
            bins = np.linspace(lo - pad, hi + pad, 90)
            ax.hist(ref, bins=bins, density=True, color=REF, lw=0)
            ax.hist(ref, bins=bins, density=True, histtype="step", color=INK, lw=0.9)
            ax.hist(arm, bins=bins, density=True, histtype="step", color=ARM, lw=2)
            ax.axvline(0, color=MUTED, lw=0.6, ls=":")
            w1 = wasserstein_distance(arm, ref) / ref.std()
            ax.set_title(f"n={n}   P({var}>0) exact {np.mean(ref > 0):.2f} / method {np.mean(arm > 0):.2f}"
                         f"   W1/sd {w1:.2f}", fontsize=7, color=INK)
            ax.set_yticks([])
            ax.tick_params(labelsize=6.5, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel(label, fontsize=7.5, color=INK, rotation=0, ha="right", va="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=REF, label="exact posterior of the global"),
               plt.Line2D([], [], color=ARM, lw=2, label="hierarchical attention + gauss_jacobian_FD")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.6, 1.005))
    fig.suptitle("Hierarchical attention + gauss_jacobian_FD on every bimodal experiment (small / middle / large n)",
                 fontsize=10, color=INK, y=1.025)
    fig.tight_layout()
    out = ART / "figures" / "14_hier_jac_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()


# -- with vs without the residual output, where a residual twin of the same net exists ---------
def residual_pairs():
    """(label, var, [(n, exact draws, stock draws, residual draws)]) for lean, curved, IK |x1|."""
    out = []
    for name, label in (("lean", "toy l~N(|g|+0.2g)"), ("curved", "toy l~N(g²+0.4g)")):
        runs = TOYS / name / "runs"
        out.append((label, "g", [(n, np.load(runs / f"mcmc_n{n}.npz")["g"],
                                  np.load(runs / f"hierarchical_gauss_jacobian_FD_n{n}.npz")["g"],
                                  np.load(runs / f"hierarchical_R_gauss_jacobian_FD_n{n}.npz")["g"])
                                 for n in (4, 16, 128)]))
    e = np.load(ART.parent.parent / "Arruda" / "artifacts" / "ik" / "exact" / "absx1_a0.6.npz")
    runs = ART / "ik_absx1" / "runs"
    out.append(("IK, |x1| observed", "angle_1", [
        (n, from_grid(e["grid"], e[f"n{n}"]),
         np.load(runs / "hierarchical" / f"gauss_jacobian_FD_n{n}.npz")["theta"][:, 0, 0],
         np.load(runs / "hierarchical_R" / f"gauss_jacobian_FD_n{n}.npz")["theta"][:, 0, 0]) for n in (8, 32, 128)]))
    return out


def residual_figure():
    rows = residual_pairs()
    resid = "#2a78d6"
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 1.9 * len(rows)), squeeze=False)
    for i, (label, var, cells) in enumerate(rows):
        for j, (n, ref, stock, res) in enumerate(cells):
            ax = axes[i, j]
            lo, hi = np.quantile(np.concatenate([ref, stock, res]), [0.002, 0.998])
            bins = np.linspace(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo), 90)
            ax.hist(ref, bins=bins, density=True, color=REF, lw=0)
            ax.hist(ref, bins=bins, density=True, histtype="step", color=INK, lw=0.9)
            ax.hist(stock, bins=bins, density=True, histtype="step", color=ARM, lw=2)
            ax.hist(res, bins=bins, density=True, histtype="step", color=resid, lw=1.6, ls="--")
            w = lambda a: wasserstein_distance(a, ref) / ref.std()  # noqa: E731
            ax.set_title(f"n={n}  P({var}>0) {np.mean(ref > 0):.2f} / {np.mean(stock > 0):.2f} / {np.mean(res > 0):.2f}"
                         f"   W1/sd {w(stock):.2f} / {w(res):.2f}", fontsize=6.8, color=INK)
            ax.set_yticks([])
            ax.tick_params(labelsize=6.5, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel(label, fontsize=7.5, color=INK, rotation=0, ha="right", va="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=REF, label="exact"),
               plt.Line2D([], [], color=ARM, lw=2, label="without residual (stock)"),
               plt.Line2D([], [], color=resid, lw=1.6, ls="--", label="with residual output")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.6, 1.02))
    fig.suptitle("Hierarchical attention + gauss_jacobian_FD, with vs without the residual output "
                 "(titles: exact / stock / residual)", fontsize=10, color=INK, y=1.06)
    fig.tight_layout()
    out = ART / "figures" / "15_residual_vs_stock.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    residual_figure()
