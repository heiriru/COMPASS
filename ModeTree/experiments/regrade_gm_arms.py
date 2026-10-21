#!/usr/bin/env python
"""Re-grade every trained GM arm over several datasets per n.

`exp2_gm_train.py` grades each `n` on a single dataset, which is enough to see a
5x effect and not enough to rank arms that differ by tens of percent. This
loads the trained checkpoints -- no retraining -- and grades each `(arm, n,
lambda)` cell on `--replicates` independent datasets, reporting the median and
the spread, so a ranking can be read off without a single lucky draw setting it.

Run:  python ModeTree/experiments/regrade_gm_arms.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BENCHMARK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import datasets    # noqa: E402
import model as M  # noqa: E402

from exp1_gm_residual_error import (  # noqa: E402
    SEED, TRAINED_LAMBDA_FLOOR, integrate, mixture_prior, roots_of,
    slice_state, sources, w1,
)
from exp2_gm_train import ARMS, MODELS, OUT, build_arm  # noqa: E402
from modetree.compass_adapter import CompassRowSource   # noqa: E402

LAMBDAS = (0.2, 0.1, 0.05, 0.03167, 0.02)
N_LADDER = (1, 2, 4, 8, 16, 32, 64, 128)


def load(name, mixture, simulations=200_000, tag="", seed=7):
    directory = MODELS / f"arm{name}_h32d3wide_n{simulations}{tag}"
    checkpoint = directory / "Model_checkpoint.pt"
    if not checkpoint.exists():
        return None
    network = build_arm(name, mixture, seed=seed)
    backbone = network.model
    while hasattr(backbone, "backbone"):
        backbone = backbone.backbone
    backbone.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model_state_dict"])
    return network


def cell(network, mixture, prior, grid, x, lam):
    column = x.reshape(-1, 1)
    source = CompassRowSource(network, column, M.HIERARCHY, M.LOCAL_INDICES,
                              M.OBSERVED_INDICES, prior)
    state = slice_state(mixture, grid, x)
    s_learned = source.row_scores(state, lam).numpy()[..., 0]
    s_prior = mixture.diffused_prior_score(grid, lam ** 2)
    composed = (1 - len(x)) * s_prior + s_learned.sum(axis=1)
    density = integrate(grid, composed)
    clean = mixture.global_density(grid, x)
    reference = np.sort(mixture.branch_means(x))
    weights = mixture.branch_weights(x)
    sd = mixture.global_moments(x)[1]
    within = float(np.sqrt(mixture.posterior_variance(len(x))))
    modes = roots_of(grid, composed)
    middle = 0.5 * (reference[0] + reference[1])
    mass = float(np.trapezoid(density[grid > middle], grid[grid > middle]))
    mean = float(np.trapezoid(grid * density, grid))
    variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    distances = ([np.min(np.abs(np.array(modes) - m)) for m in reference]
                 if modes else None)
    tolerance = 0.5 * within + 0.25
    return {
        "w1_over_sd": w1(grid, density, clean) / sd,
        "weight_tv": 0.5 * float(abs(mass - weights[1])
                                 + abs((1 - mass) - weights[0])),
        "width_ratio": float(np.sqrt(max(variance, 0.0)) / sd),
        "mode_error_over_sd": (float(np.max(distances) / within)
                               if distances else None),
        "mode_recall": (float(np.mean([d < tolerance for d in distances]))
                        if distances else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=sorted(ARMS))
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--points", type=int, default=801)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--observations", type=int, nargs="+", default=N_LADDER)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    args = parser.parse_args()

    mixture = M.DEFAULT
    prior = mixture_prior(mixture)
    grid = np.linspace(-6.0, 6.0, args.points)
    keys = ("w1_over_sd", "weight_tv", "width_ratio", "mode_error_over_sd",
            "mode_recall")

    out = {}
    for name in args.arms:
        network = load(name, mixture, tag=args.tag, seed=args.seed)
        if network is None:
            print(f"arm {name}: no checkpoint, skipped", flush=True)
            continue
        rows = []
        for n in args.observations:
            for lam in args.lambdas:
                gathered = {key: [] for key in keys}
                for replicate in range(args.replicates):
                    rng = np.random.default_rng(
                        [SEED, datasets.REGIMES.index("asymmetric"), n,
                         replicate])
                    data = datasets.make_dataset(
                        "asymmetric", n, rng, mixture=mixture,
                        branch_mass=0.75, spread=1.0)
                    x = np.asarray(data["x"], dtype=np.float64).ravel()
                    result = cell(network, mixture, prior, grid, x, lam)
                    for key in keys:
                        if result[key] is not None:
                            gathered[key].append(result[key])
                row = {"arm": name, "n": int(n), "lambda": float(lam),
                       "replicates": args.replicates,
                       "extrapolated": bool(lam < TRAINED_LAMBDA_FLOOR)}
                for key in keys:
                    values = gathered[key]
                    row[key] = float(np.median(values)) if values else None
                    row[key + "_iqr"] = (
                        float(np.subtract(*np.percentile(values, [75, 25])))
                        if values else None)
                rows.append(row)
                print(f"  {name:6s} n={n:4d} lam={lam:7.5g}  "
                      f"W1/sd={row['w1_over_sd']:7.3f} "
                      f"(iqr {row['w1_over_sd_iqr']:.3f})  "
                      f"recall={row['mode_recall']:.2f} "
                      f"TV={row['weight_tv']:.3f} "
                      f"width={row['width_ratio']:.3f}", flush=True)
        out[name] = rows
        with open(OUT / f"regrade{args.tag}.json", "w") as handle:
            json.dump(out, handle, indent=1)
    print(f"\nwrote {OUT / f'regrade{args.tag}.json'}")


if __name__ == "__main__":
    main()
