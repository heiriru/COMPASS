#!/usr/bin/env python3
"""COMPASS composition rules vs BayesFlow's Arruda sampler on the hierarchical IK problem.

Same data as `Arruda/artifacts/ik/hier_a0.6` (run_arruda.py --ik-regime hierarchical): one true
angle_1 = 0.6, every observation's (height, angle_2, angle_3) drawn from the prior, no selection,
nested first-n of one 128-long sequence. Same ABC reference (that run's reference.npz).

COMPASS arms use the stock IK checkpoint and the settings of ModeTree/experiments/experiment_ik.py
(`run_baseline`: churned DPM2 at eta = 2, Jacobian refresh 5, 100 timesteps), plus
gauss_hierarchical with a 1,000-draw pilot. The IK prior is Gaussian, so the GM "mixture prior"
variant would be identical to gauss_jacobian and is not run.

Run from COMPASS: python Arruda/compare_compass_ik.py --stage sample|plot
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ModeTree" / "experiments"), str(ROOT / "ModeTree"), str(ROOT / "src"),
                str(ROOT / "inverse_kinematics")]

TRUE_ANGLE, SEED, MAX_N = 0.6, 20260923, 128
NS = (1, 2, 4, 8, 16, 32, 64, 128)
OUT = ROOT / "Arruda" / "artifacts" / "ik" / "hier_a0.6_compass"
ARRUDA = ROOT / "Arruda" / "artifacts" / "ik" / "hier_a0.6"
COMPASS_METHODS = ("langevin_fnpse", "gauss_hierarchical", "gauss_jacobian")


REFERENCE = None          # --reference: graded against this instead of ARRUDA/reference.npz
DATA_FILE = None          # --data: a saved (128, 2) dataset instead of the built-in hierarchical draw


def dataset(n):
    """Identical to run_arruda.ik_tall_problem(n, SEED, "hierarchical"), or the first n rows of --data."""
    import numpy as np
    if DATA_FILE is not None:
        return np.load(DATA_FILE)[:n]
    import hierarchical as H
    rng = np.random.default_rng(SEED)
    locals_ = rng.normal(loc=H.LOCAL_PRIOR_MEAN, scale=H.LOCAL_PRIOR_STD, size=(MAX_N, 3))
    theta = np.concatenate([np.full((MAX_N, 1), TRUE_ANGLE), locals_], axis=1)
    return H.forward(theta)[:n]


def sample(arguments):
    import numpy as np
    import torch
    import experiment_ik as E   # applies the 3-thread CPU cap on import
    import os
    if "CUDA_VISIBLE_DEVICES" not in os.environ:     # an explicit card from the caller wins
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    E.BASELINES["gauss_hierarchical"] = {
        "label": "DPM2 + gauss_hierarchical + churn", "churn": True,
        "kwargs": {"method": "dpm", "order": 2, "correction": "gauss_hierarchical", "corrector_steps": 0,
                   "final_corrector_steps": 0, "terminal_corrector_steps": 0,
                   "precision_est_samples": 1_000, "precision_est_timesteps": E.TIMESTEPS},
    }
    model = E.trainer.load_to(arguments.checkpoint or E.CHECKPOINT, "cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for method in arguments.methods:
        timings = {}
        for n in arguments.ns:
            path = OUT / f"samples_{method}_n{n}.npy"
            if path.exists():
                continue
            # Chunk draws so draws x n stays within an 11 GB card for the Jacobian rule; own seed per chunk.
            per_call = max(1, min(arguments.draws, arguments.pair_budget // n))
            parts, seconds = [], 0.0
            for k, start in enumerate(range(0, arguments.draws, per_call)):
                g, took = E.run_baseline(model, dataset(n), method, min(per_call, arguments.draws - start),
                                         2026 + k, "cuda", churn_eta=2.0)
                parts.append(np.asarray(g)); seconds += took
                torch.cuda.empty_cache()
            g = np.concatenate(parts)
            np.save(path, g)
            timings[f"{method}_n{n}"] = seconds
            print(f"{method:20s} n={n:3d}: {seconds:7.1f}s  mean {g.mean():+.3f} sd {g.std():.3f}", flush=True)
            (OUT / f"timings_{method}.json").write_text(json.dumps(timings, indent=2) + "\n")


def plot(arguments):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import wasserstein_distance
    reference = np.load(REFERENCE or ARRUDA / "reference.npz")
    grid = reference["grid"]
    rows = [("Arruda, tutorial (d=1)", lambda n: ARRUDA / f"samples_n{n}_bridge1.npy", "#e76f51"),
            ("Arruda, damped (d=0.1)", lambda n: ARRUDA / f"samples_n{n}_bridge0.1.npy", "#f4a261"),
            ("Arruda, diffused prior", lambda n: ARRUDA / f"samples_n{n}_bridge1_diffused.npy", "#3a86ff"),
            ("COMPASS Langevin + F-NPSE", lambda n: OUT / f"samples_langevin_fnpse_n{n}.npy", "#2a9d8f"),
            ("COMPASS gauss_hierarchical", lambda n: OUT / f"samples_gauss_hierarchical_n{n}.npy", "#9d4edd"),
            ("COMPASS gauss_jacobian", lambda n: OUT / f"samples_gauss_jacobian_n{n}.npy", "#5e60ce"),
            ("COMPASS gauss_jacobian_FD", lambda n: OUT / f"samples_gauss_jacobian_FD_n{n}.npy", "#3c096c"),
            ("COMPASS mixture (K=3)", lambda n: OUT / f"samples_mixture_n{n}.npy", "#d62828")]
    rows = [r for r in rows if all(r[1](n).exists() for n in NS)]
    fig, axes = plt.subplots(len(rows), len(NS), figsize=(2.55 * len(NS), 1.75 * len(rows)), sharex=True,
                             constrained_layout=True, squeeze=False)
    bins = np.linspace(grid[0], grid[-1], 321)
    table = {}
    for c, n in enumerate(NS):
        exact = reference[f"n{n}"]
        # Inverse CDF with interpolation, not grid.choice: at n = 128 the grid step is a sizeable share of the sd.
        cdf = np.concatenate([[0], np.cumsum(0.5 * (exact[1:] + exact[:-1]) * np.diff(grid))])
        ref_draws = np.interp(np.random.default_rng(SEED + 10 * n).random(50_000), cdf / cdf[-1], grid)
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
            ax.axvline(TRUE_ANGLE, color="0.3", ls="--", lw=1)
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
        ax.set_xlabel("angle_1", fontsize=9)
    fig.suptitle(f"IK hierarchical model, data from one true angle_1 = {TRUE_ANGLE:g} (dashed); "
                 f"grey = {arguments.reference_label}", fontsize=11)
    fig.savefig(OUT / f"posterior_comparison{arguments.suffix}.png", dpi=170, bbox_inches="tight")
    (OUT / f"w1_over_sd{arguments.suffix}.json").write_text(json.dumps(table, indent=2) + "\n")
    for label, row in table.items():
        print(f"{label:30s} " + "  ".join(f"n={n}:{v:.2f}" for n, v in row.items()))


def main():
    global DATA_FILE, OUT, ARRUDA, TRUE_ANGLE, REFERENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sample", "plot"), required=True)
    parser.add_argument("--methods", nargs="+", default=list(COMPASS_METHODS))
    parser.add_argument("--ns", type=int, nargs="+", default=list(NS))
    parser.add_argument("--draws", type=int, default=4_000)
    parser.add_argument("--pair-budget", type=int, default=32_000, help="max draws x observations per sampler call")
    parser.add_argument("--data", default="", help="(128, 2) .npy of observations (default: the built-in hierarchical draw)")
    parser.add_argument("--true-angle", type=float, default=TRUE_ANGLE)
    parser.add_argument("--out", default="", help="output folder (default hier_a0.6_compass)")
    parser.add_argument("--arruda", default="", help="folder of the matching Arruda run (default hier_a0.6)")
    parser.add_argument("--reference", default="", help="reference .npz (grid, n1..n128); default ARRUDA/reference.npz")
    parser.add_argument("--reference-label", default="ABC reference posterior")
    parser.add_argument("--suffix", default="", help="appended to the figure and W1 file names")
    parser.add_argument("--checkpoint", default="", help="network checkpoint (default: the stock IK one)")
    arguments = parser.parse_args()
    DATA_FILE = Path(arguments.data) if arguments.data else None
    OUT = Path(arguments.out) if arguments.out else OUT
    ARRUDA = Path(arguments.arruda) if arguments.arruda else ARRUDA
    TRUE_ANGLE = arguments.true_angle
    REFERENCE = Path(arguments.reference) if arguments.reference else None
    {"sample": sample, "plot": plot}[arguments.stage](arguments)


if __name__ == "__main__":
    main()
