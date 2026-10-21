#!/usr/bin/env python3
"""Constructed bimodal IK data (true angle_1 = 0.6): the MCMC posterior against four compositional methods.

Rows: the rail-height set (|x1 - h| < 0.05) and the near-rail-tip set (|x1| < 0.15).

MCMC: the collapsed exact sampler (ik_mcmc.sample_angle_collapsed), standard model, one run per n
on the nested first-n rows. Methods: Arruda et al. adaptive compositional diffusion in BayesFlow
(paper setting, bridge d = 1; Arruda/artifacts/ik/rail_a0.6), COMPASS gauss_hierarchical, Langevin
F-NPSE (Arruda/artifacts/ik/rail_a0.6_compass) and gauss_jacobian_FD (run_rail_fd.py).

Usage (from COMPASS/): .COMPASS/bin/python inverse_kinematics/recovery/plot_rail_methods.py
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
from scipy.stats import wasserstein_distance  # noqa: E402

import ik_exact as E  # noqa: E402
import ik_mcmc as M  # noqa: E402

ROOT = HERE.parents[1]
IK = ROOT / "Arruda" / "artifacts" / "ik"
OUT = HERE / "artifacts" / "bimodality"
NS = (1, 2, 4, 8, 16, 32, 64, 128)
DATASETS = {   # name: (data file, Arruda folder, COMPASS folder, row label)
    "rail": (IK / "rail_height" / "data_a0.6_tol0.05.npy", "rail_a0.6", "rail_a0.6_compass",
             "rail-height rows\n|x1 - h| < 0.05"),
    "x1sel": (IK / "x1sel" / "data_a0.6_x1sel0.15.npy", "x1sel_a0.6", "x1sel_a0.6_compass",
              "near-rail tip\n|x1| < 0.15"),
}
METHODS = (("Arruda et al. (BayesFlow, paper setting)", lambda a, c, n: IK / a / f"samples_n{n}_bridge1.npy", "#2a78d6"),
           ("COMPASS gauss_hierarchical", lambda a, c, n: IK / c / f"samples_gauss_hierarchical_n{n}.npy", "#eb6834"),
           ("COMPASS gauss_jacobian_FD", lambda a, c, n: IK / c / f"samples_gauss_jacobian_FD_n{n}.npy", "#1baf7a"),
           ("COMPASS Langevin + F-NPSE", lambda a, c, n: IK / c / f"samples_langevin_fnpse_n{n}.npy", "#4a3aa7"))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})


def mcmc(name, x, n, rng):
    path = OUT / f"{name}_mcmc_n{n}.npz"
    if path.exists():
        return dict(np.load(path))
    out = M.sample_angle_collapsed(x[None, :n], rng, chains=8, warmup=1000, draws=1000, thin=4, nodes=64)
    a = out["a1"][0]
    grid, dens = E.grid_posterior_a1(x[:n], nodes=64)
    cdf = np.cumsum(dens) / dens.sum()
    ks = float(np.abs(np.searchsorted(np.sort(a.ravel()), grid, side="right") / a.size - cdf).max())
    result = {"draws": a.ravel(), "rhat": M.rhat(a), "ess": M.ess_bulk(a), "ks_vs_grid": ks}
    np.savez(path, **result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260930)
    fig, axes = plt.subplots(len(DATASETS), len(NS), figsize=(2.2 * len(NS), 2.3 * len(DATASETS)),
                             constrained_layout=True, sharex=True, squeeze=False)
    bins = np.linspace(-1.5, 1.5, 121)
    table = {}
    for r, (name, (data, arruda, compass, row_label)) in enumerate(DATASETS.items()):
        x = np.load(data)
        table[name] = {}
        for c, n in enumerate(NS):
            ref = mcmc(name, x, n, rng)
            draws = ref["draws"]
            ax = axes[r, c]
            ax.hist(draws, bins=bins, density=True, color="#d6d5d0", lw=0, label="MCMC posterior (exact)")
            top = np.histogram(draws, bins=bins, density=True)[0].max()
            row = {"mcmc": {"rhat": float(ref["rhat"]), "ess": float(ref["ess"]),
                            "ks_vs_grid": float(ref["ks_vs_grid"]), "p_positive": float((draws > 0).mean())}}
            for label, path, colour in METHODS:
                if not path(arruda, compass, n).exists():
                    continue
                s = np.load(path(arruda, compass, n)).ravel()
                ax.hist(np.clip(s, -1.5, 1.5), bins=bins, density=True, histtype="step", color=colour, lw=1.3,
                        label=label)
                row[label] = {"w1_over_mcmc_sd": float(wasserstein_distance(s, draws) / draws.std()),
                              "p_positive": float((s > 0).mean())}
            for v in (-0.6, 0.6):
                ax.axvline(v, color="#52514e", lw=0.7, ls=(0, (2, 2)))
            ax.set_ylim(0, 1.3 * top)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"n = {n}", fontsize=9)
            if r == len(DATASETS) - 1:
                ax.set_xlabel("angle_1")
            table[name][n] = row
            print(f"{name:5s} n={n:3d} MCMC R-hat {row['mcmc']['rhat']:.3f} ESS {row['mcmc']['ess']:.0f} "
                  f"KS vs grid {row['mcmc']['ks_vs_grid']:.3f} P(a1>0) {row['mcmc']['p_positive']:.2f} | "
                  + "  ".join(f"{k.split('(')[0].replace('COMPASS ', '').strip()}: W1/sd {v['w1_over_mcmc_sd']:.2f} "
                              f"P+ {v['p_positive']:.2f}" for k, v in row.items() if k != "mcmc"), flush=True)
        axes[r, 0].set_ylabel(row_label)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Constructed bimodal data, true angle_1 = 0.6 (dotted: ±0.6); histograms clipped at ±1.5",
                 fontsize=10)
    fig.savefig(OUT / "constructed_methods_vs_mcmc.png", dpi=170, bbox_inches="tight")
    (OUT / "constructed_methods_vs_mcmc.json").write_text(json.dumps(table, indent=1))
    print("wrote", OUT / "constructed_methods_vs_mcmc.png")


if __name__ == "__main__":
    main()
