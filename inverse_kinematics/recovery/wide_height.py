#!/usr/bin/env python3
"""Wide height prior (sigma_h = 10), true angle_1 = 0.25, natural (unfiltered) data.

The target where the exact angle_1 posterior is bimodal at +-0.25 at every n up to 128
(bimodality_options.py): the mirror twin costs ``2 x1 (x1 - h) / sigma_h^2`` per observation,
so widening sigma_h 40x keeps both modes. Every stage must run with ``IK_SD_H=10`` in the
environment so the simulator, the exact likelihood and every sampler's prior agree; the
COMPASS network also needs ``IK_VESDE_SIGMA=60`` (x1 spans about +-30).

    IK_SD_H=10 python wide_height.py data         # COUNT datasets of 128 rows, reference grids
    IK_SD_H=10 python wide_height.py mcmc         # collapsed MCMC per dataset and n, R-hat, KS vs grid
    IK_SD_H=10 IK_VESDE_SIGMA=60 python wide_height.py train
    IK_SD_H=10 IK_VESDE_SIGMA=60 python wide_height.py network   # gauss_hierarchical, gauss_jacobian_FD
    IK_SD_H=10 python wide_height.py plot

Arruda and Langevin F-NPSE run from run_wide_height.sh on the same files. Everything is
resumable: finished files are skipped.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IK = HERE.parent
sys.path[:0] = [str(HERE), str(IK)]

SD_H = 10.0
if float(os.environ.get("IK_SD_H") or 0) != SD_H:
    raise SystemExit(f"run with IK_SD_H={SD_H:g} so every prior agrees")

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import json  # noqa: E402

import numpy as np  # noqa: E402

import hierarchical as H  # noqa: E402
import ik_exact as E  # noqa: E402
import simulator as sim  # noqa: E402

TRUTH = 0.25
COUNT = 10
ROWS = 128
NS = (1, 2, 8, 32, 128)
# Fine: at n = 128 each mode is a few 1e-3 wide; a 1401-point grid failed the KS check against MCMC.
GRID = np.linspace(-2.0, 2.0, 40001)
OUT = HERE / "artifacts" / "wide_height"
ARRUDA = IK.parent / "Arruda" / "artifacts" / "ik" / "wide_sdh10"
CHECKPOINT = IK / "artifacts" / "models" / "legacy_h16d2_n200000_six_weighted_log_sigma_sdh10" / "Model_checkpoint.pt"
METHODS = {  # label, file pattern (dataset d, n), colour
    "MCMC": (lambda d, n: OUT / f"d{d}" / f"mcmc_n{n}.npz", "#52514e"),
    "Arruda (BayesFlow)": (lambda d, n: ARRUDA / f"d{d}" / f"samples_n{n}_bridge1.npy", "#d1495b"),
    "gauss_hierarchical": (lambda d, n: OUT / f"d{d}" / f"gauss_hierarchical_n{n}.npy", "#2a78d6"),
    "gauss_jacobian_FD": (lambda d, n: OUT / f"d{d}" / f"gauss_jacobian_FD_n{n}.npy", "#e3a008"),
    "Langevin F-NPSE": (lambda d, n: OUT / f"d{d}" / "langevin" / f"samples_langevin_fnpse_n{n}.npy", "#2a9d8f"),
}


def data():
    """``COUNT`` natural datasets (a1 = TRUTH, locals from the wide prior) and exact grid posteriors."""
    rng = sim.generator("wide-height-data")
    for d in range(COUNT):
        folder = OUT / f"d{d}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "reference_exact.npz").exists():
            continue
        locals_ = rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, (ROWS, 3))
        if (folder / "data.npy").exists():
            x = np.load(folder / "data.npy").astype(np.float64)
        else:
            x = H.forward(np.column_stack([np.full(ROWS, TRUTH), locals_]))
            np.save(folder / "data.npy", x.astype(np.float32))
        table = np.stack([E.log_marginal_a1(GRID, x[j, 0], x[j, 1], nodes=32, wraps=(0,)) for j in range(ROWS)], 1)
        curves = {}
        for n in (1, 2, 4, 8, 16, 32, 64, 128):      # Arruda's runner asks for the whole ladder
            lp = E.log_normal(GRID, E.SD_A1) + table[:, :n].sum(1)
            p = np.exp(lp - lp.max())
            curves[f"n{n}"] = p / np.trapezoid(p, GRID)
        np.savez(folder / "reference_exact.npz", grid=GRID, **curves)
        print(f"dataset {d}: n=128 P(a1>0) {np.trapezoid(curves['n128'] * (GRID > 0), GRID):.2f}", flush=True)


def mcmc():
    """Collapsed MCMC (the production angle-layout sampler) per dataset and n; R-hat and KS vs the grid."""
    import ik_mcmc as M
    from scipy.stats import kstest

    report = {}
    for d in range(COUNT):
        folder = OUT / f"d{d}"
        x = np.load(folder / "data.npy").astype(np.float64)
        ref = np.load(folder / "reference_exact.npz")
        for n in NS:
            path = folder / f"mcmc_n{n}.npz"
            if not path.exists():
                rng = sim.generator(f"wide-height-mcmc-{d}-{n}")
                res = M.sample_angle_collapsed(x[None, :n], rng, chains=8, warmup=1000, draws=1000, thin=2)
                np.savez(path, a1=res["a1"][0])
            chains = np.load(path)["a1"]
            cdf = np.cumsum(ref[f"n{n}"]) * (GRID[1] - GRID[0])
            ks = kstest(chains.ravel(), lambda v: np.interp(v, GRID, cdf / cdf[-1]))
            ess = float(M.ess_bulk(chains))
            report[f"d{d}_n{n}"] = {"rhat": float(M.rhat(chains)), "ess": ess, "ks": float(ks.statistic),
                                   "ks_floor": float(1.36 / np.sqrt(ess)), "p_positive": float((chains > 0).mean())}
            r = report[f"d{d}_n{n}"]
            print(f"d{d} n={n:3d}: R-hat {r['rhat']:.3f} ESS {ess:6.0f} KS {r['ks']:.3f} (floor {r['ks_floor']:.3f})"
                  f" P+ {r['p_positive']:.2f}", flush=True)
    (OUT / "mcmc_checks.json").write_text(json.dumps(report, indent=1))


def train():
    if float(os.environ.get("IK_VESDE_SIGMA") or 0) < 30:
        raise SystemExit("set IK_VESDE_SIGMA=60: the stock sigma=8 ceiling is below the x1 range")
    import train_model as trainer

    raise SystemExit(trainer.main(["--arch", "legacy", "--config", "h16d2", "--simulations", "200000",
                                   "--mask-menu", "six_weighted", "--time-sampling", "log_sigma",
                                   "--seed", "7", "--output-suffix", "_sdh10"]))


def network():
    import run_network as RN  # CPU cap + autocvd
    from layouts import LAYOUTS, PRIOR_STD

    assert PRIOR_STD[1] == SD_H
    layout = LAYOUTS["angle"]
    model = RN.trainer.load_to(CHECKPOINT, "cuda")
    for method in ("gauss_hierarchical", "gauss_jacobian_FD"):
        for d in range(COUNT):
            x = np.load(OUT / f"d{d}" / "data.npy")
            for n in NS:
                path = OUT / f"d{d}" / f"{method}_n{n}.npy"
                if path.exists():
                    continue
                theta = RN.composed(model, layout, x[:n], method, 4000, 2026 + n)
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, theta[:, 0, 0].astype(np.float32))
                tmp.rename(path)
                print(f"{method} d{d} n={n}: mean {theta[:, 0, 0].mean():+.3f} sd {theta[:, 0, 0].std():.3f}",
                      flush=True)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wasserstein_distance

    from bimodality import shape

    summary = {}
    show = int(os.environ.get("SHOW") or 0)
    fig, axes = plt.subplots(len(METHODS), len(NS), figsize=(2.5 * len(NS), 1.9 * len(METHODS)),
                             constrained_layout=True, sharex=True, squeeze=False)
    for r, (label, (pattern, colour)) in enumerate(METHODS.items()):
        for c, n in enumerate(NS):
            ref = np.load(OUT / f"d{show}" / "reference_exact.npz")[f"n{n}"]
            ax = axes[r, c]
            ax.fill_between(GRID, ref, color="#8a8984", alpha=0.35, lw=0, label="exact (MCMC-checked)")
            path = pattern(show, n)
            if path.exists():
                draws = np.load(path)["a1"] if path.suffix == ".npz" else np.load(path)
                ax.hist(draws.ravel(), bins=np.linspace(-1.2, 1.2, 97), density=True, histtype="step",
                        color=colour, lw=1.4, label=label)
            for v in (-TRUTH, TRUTH):
                ax.axvline(v, color="#52514e", lw=0.7, ls=(0, (2, 2)))
            ax.set_xlim(-1.2, 1.2)
            ax.set_yticks([])
            ax.spines[["top", "right", "left"]].set_visible(False)
            if r == 0:
                ax.set_title(f"n = {n}", fontsize=9)
        axes[r, 0].set_ylabel(label, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("angle_1")
    fig.suptitle(f"sigma_h = {SD_H:g}, true angle_1 = {TRUTH} (dotted +-{TRUTH}), dataset {show}: grey = exact posterior",
                 fontsize=10)
    fig.savefig(OUT / f"wide_height_methods_d{show}.png", dpi=170)

    for label, (pattern, _) in METHODS.items():
        for n in NS:
            rows = []
            for d in range(COUNT):
                path = pattern(d, n)
                if not path.exists():
                    continue
                draws = (np.load(path)["a1"] if path.suffix == ".npz" else np.load(path)).ravel()
                ref = np.load(OUT / f"d{d}" / "reference_exact.npz")[f"n{n}"]
                ref_draws = np.random.default_rng(d).choice(GRID, 20_000, p=ref / ref.sum())
                hist, edges = np.histogram(draws, bins=np.linspace(-2, 2, 801), density=True)
                s = shape(0.5 * (edges[1:] + edges[:-1]), hist + 1e-12)
                rows.append({"w1_over_sd": wasserstein_distance(draws, ref_draws) / ref_draws.std(),
                             "p_positive": float((draws > 0).mean()),
                             "p_positive_exact": float((ref_draws > 0).mean()),
                             "both_modes": any(abs(p - TRUTH) < 0.1 for p in s["peaks"])
                             and any(abs(p + TRUTH) < 0.1 for p in s["peaks"])})
            if rows:
                summary[f"{label} n={n}"] = {
                    "datasets": len(rows),
                    "median_w1_over_sd": float(np.median([q["w1_over_sd"] for q in rows])),
                    "median_abs_p_positive_error": float(np.median([abs(q["p_positive"] - q["p_positive_exact"])
                                                                    for q in rows])),
                    "both_modes_fraction": float(np.mean([q["both_modes"] for q in rows]))}
                s = summary[f"{label} n={n}"]
                print(f"{label:22s} n={n:3d} ({s['datasets']:2d} sets): W1/sd {s['median_w1_over_sd']:.3f}  "
                      f"|P+ err| {s['median_abs_p_positive_error']:.3f}  both modes {s['both_modes_fraction']:.2f}")
    (OUT / "wide_height_summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", OUT / "wide_height_methods.png")


if __name__ == "__main__":
    {"data": data, "mcmc": mcmc, "train": train, "network": network, "plot": plot}[sys.argv[1]]()
