#!/usr/bin/env python3
"""COMPASS composition rules vs BayesFlow's Arruda sampler on the hierarchical GM problem.

Same model as `Arruda/artifacts/gm/hier_g1.2` (mu_0 = 1.5, tau_g = 0.4, tau_l = 0.7, sigma_x = 6),
same nested datasets (one true g = 1.2, first n of one 128-long sequence), same exact posterior.

Stages:
  train   the stock GM COMPASS recipe (tokenized h32d3wide, 200k simulations, six_weighted masks,
          log_sigma time sampling) on this model, into its own tagged checkpoint directory
  sample  langevin_fnpse, gauss_hierarchical, gauss_jacobian (+ gauss_jacobian with the exact
          diffused mixture prior) via Gaussian_mixtures_compositional_score_modelling/methods.py
  plot    small multiples: one row per method, one column per n, exact posterior and true g in each

Run from COMPASS: python Arruda/compare_compass_gm.py --stage train|sample|plot
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "Gaussian_mixtures_compositional_score_modelling"
sys.path[:0] = [str(ROOT / "src"), str(BENCHMARK), str(ROOT)]

from runtime import configure_runtime  # noqa: E402  (3-thread CPU cap, as every GM script)

MU0, SIGMA_X, G_TRUE, SEED, MAX_N = 1.5, 6.0, 1.2, 20260923, 128
NS = (1, 2, 4, 8, 16, 32, 64, 128)
MODEL_DIR = BENCHMARK / "artifacts" / "models" / "tokenized_h32d3wide_n200000_six_weighted_log_sigma_mu1.5_sx6"
OUT = ROOT / "Arruda" / "artifacts" / "gm" / "hier_g1.2_compass"
ARRUDA = ROOT / "Arruda" / "artifacts" / "gm" / "hier_g1.2"
COMPASS_METHODS = ("langevin_fnpse", "gauss_hierarchical", "gauss_jacobian", "gauss_jacobian_mixture_prior")


def mixture():
    import model as M
    return M.Mixture(mu_0=MU0, sigma_x=SIGMA_X, weights=(0.5, 0.5))


def dataset(n):
    """Identical to run_arruda.gm_tall_problem(..., "hierarchical", 1.2): l_i ~ N(g, tau_l^2), x_i ~ N(l_i, sigma_x^2)."""
    import numpy as np
    mix = mixture()
    rng = np.random.default_rng(SEED)
    local = G_TRUE + mix.tau_l * rng.standard_normal(MAX_N)
    return (local + mix.sigma_x * rng.standard_normal(MAX_N))[:n]


def train(arguments):
    import train as T
    network, seconds, parameters = T.train_or_load(MODEL_DIR, "h32d3wide", "tokenized", 200_000, arguments.device,
                                                   seed=7, mixture=mixture())
    if parameters is not None:
        (MODEL_DIR / "training.json").write_text(json.dumps(
            {"mu_0": MU0, "sigma_x": SIGMA_X, "seconds": seconds, "parameters": int(parameters)}, indent=2) + "\n")
    print(f"trained in {seconds:.0f}s -> {MODEL_DIR}", flush=True)


def sample(arguments):
    import numpy as np
    import torch
    import methods
    import train as T
    network = T.load_to(MODEL_DIR / "Model_checkpoint.pt", arguments.device)
    run_args = argparse.Namespace(pilot_draws=1_000, jacobian_refresh=5, churn_eta=4.0, denoise_clamp=5.0,
                                  seed=2026, draws=arguments.draws)
    OUT.mkdir(parents=True, exist_ok=True)
    timings = {}
    for method in arguments.methods:
        for n in arguments.ns:
            path = OUT / f"samples_{method}_n{n}.npy"
            if path.exists():
                continue
            # Chunk the draws so draws x n stays within what an 11 GB card holds for the Jacobian rules
            # (4,000 x 16 fits, 4,000 x 32 does not); each chunk gets its own seed.
            per_call = max(1, min(arguments.draws, arguments.pair_budget // n))
            parts, seconds = [], 0.0
            for k, start in enumerate(range(0, arguments.draws, per_call)):
                chunk_args = argparse.Namespace(**{**vars(run_args), "draws": min(per_call, arguments.draws - start),
                                                   "seed": run_args.seed + k})
                samples, took = methods.draw_score_rule(network, dataset(n), method, chunk_args, arguments.device,
                                                        mixture=mixture())
                parts.append(samples[0, :, 0])                               # the shared coordinate (node 0 = g)
                seconds += took
                torch.cuda.empty_cache()
            g = np.concatenate(parts)
            np.save(path, g)
            timings[f"{method}_n{n}"] = seconds
            print(f"{method:30s} n={n:3d}: {seconds:7.1f}s  mean {g.mean():+.3f} sd {g.std():.3f}", flush=True)
            # One file per method, so methods sampled by parallel processes never overwrite each other.
            (OUT / f"timings_{method}.json").write_text(json.dumps(
                {k: v for k, v in timings.items() if k.startswith(method + "_n")}, indent=2) + "\n")


def plot(arguments):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import wasserstein_distance
    mix = mixture()
    reference = np.load(ARRUDA / "reference.npz")
    grid = reference["grid"]
    arruda = {row["method"]: row for row in json.loads((ARRUDA / "results.json").read_text())}
    rows = [("Arruda, tutorial (d=1)", lambda n: ARRUDA / f"samples_n{n}_bridge1.npy", "#e76f51"),
            ("Arruda, damped (d=0.1)", lambda n: ARRUDA / f"samples_n{n}_bridge0.1.npy", "#f4a261"),
            ("Arruda, diffused prior", lambda n: ARRUDA / f"samples_n{n}_bridge1_diffused.npy", "#3a86ff"),
            ("COMPASS Langevin + F-NPSE", lambda n: OUT / f"samples_langevin_fnpse_n{n}.npy", "#2a9d8f"),
            ("COMPASS gauss_hierarchical", lambda n: OUT / f"samples_gauss_hierarchical_n{n}.npy", "#9d4edd"),
            ("COMPASS gauss_jacobian", lambda n: OUT / f"samples_gauss_jacobian_n{n}.npy", "#5e60ce"),
            ("COMPASS gauss_jacobian, mixture prior", lambda n: OUT / f"samples_gauss_jacobian_mixture_prior_n{n}.npy", "#b5179e")]
    rows = [r for r in rows if all(r[1](n).exists() for n in NS)]
    fig, axes = plt.subplots(len(rows), len(NS), figsize=(2.55 * len(NS), 1.75 * len(rows)), sharex=True,
                             constrained_layout=True, squeeze=False)
    bins = np.linspace(grid[0], grid[-1], 81)
    table = {}
    for c, n in enumerate(NS):
        exact = reference[f"n{n}"]
        ref_draws = np.random.default_rng(SEED + 10 * n).choice(grid, size=50_000, p=exact / exact.sum())
        top = exact.max()
        for r, (label, path, colour) in enumerate(rows):
            ax = axes[r, c]
            values = np.load(path(n))
            w1 = wasserstein_distance(values, ref_draws) / ref_draws.std()
            table.setdefault(label, {})[n] = round(float(w1), 3)
            ax.fill_between(grid, exact, color="0.85", lw=0)
            ax.plot(grid, exact, color="black", lw=1.2)
            ax.hist(values, bins=bins, weights=np.full(len(values), 1 / (len(values) * (bins[1] - bins[0]))),
                    histtype="step", color=colour, lw=1.4)
            ax.axvline(G_TRUE, color="0.3", ls="--", lw=1)
            ax.set_ylim(0, 1.6 * top)
            ax.set_yticks([])
            off = np.mean(np.abs(values) > grid[-1])
            ax.text(0.03, 0.93, f"W1/sd {w1:.2f}" + (f"\n{off:.0%} off" if off > 0.005 else ""), transform=ax.transAxes,
                    fontsize=7, va="top", color=colour)
            if r == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", color=colour)
    for ax in axes[-1]:
        ax.set_xlabel("g", fontsize=9)
    fig.suptitle(f"GM hierarchical model (mu_0 = {MU0:g}, sigma_x = {SIGMA_X:g}), data from one true g = {G_TRUE:g} (dashed); "
                 "grey = exact posterior", fontsize=11)
    fig.savefig(OUT / "posterior_comparison.png", dpi=170, bbox_inches="tight")
    (OUT / "w1_over_sd.json").write_text(json.dumps(table, indent=2) + "\n")
    for label, row in table.items():
        print(f"{label:40s} " + "  ".join(f"n={n}:{v:.2f}" for n, v in row.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "sample", "plot"), required=True)
    parser.add_argument("--methods", nargs="+", default=list(COMPASS_METHODS))
    parser.add_argument("--ns", type=int, nargs="+", default=list(NS))
    parser.add_argument("--draws", type=int, default=4_000)
    parser.add_argument("--pair-budget", type=int, default=64_000, help="max draws x observations per sampler call")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.stage != "plot":
        configure_runtime(gpu=arguments.device.startswith("cuda"))
    {"train": train, "sample": sample, "plot": plot}[arguments.stage](arguments)


if __name__ == "__main__":
    main()
