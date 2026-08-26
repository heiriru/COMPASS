#!/usr/bin/env python3
"""How small a network, and how many simulations, does this hierarchy need?

Two questions that fail independently, so they get two metrics -- and both are
measured on **single-observation posteriors only**, with no composition anywhere
in the loop. That is deliberate: composition amplifies score error coherently
across observations, so grading capacity through the composed pipeline would
confound the network with the composition rule under test.

``w1_g``, ``w1_l``
    1-D Wasserstein distance between sampled and exact marginals of
    ``p(g, l | x_j)``, averaged over the 30 observations, in units of each
    posterior's own standard deviation. Distributional fidelity.

``score_rmse``
    Relative RMS error of the network's raw score output against the *exact*
    diffused single-observation score (the N = 1 case of
    ``hierarchy.diffused_score``), at states drawn from the exact diffused joint,
    across a ladder of noise levels spanning the sampler's schedule. This is the
    quantity every composition rule consumes, so it is the one that predicts
    whether the pipeline can work at all.

One (config, simulations) cell per process so the sweep can be spread over GPUs;
rows are appended as they finish and existing rows are skipped, so it resumes
after a kill.

Usage:
    CUDA_VISIBLE_DEVICES=2 python capacity.py --config h8d1 --train-samples 200000
    python capacity.py --list
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hierarchy  # noqa: E402
import train as recipe  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SWEEP_CSV = ARTIFACTS / "capacity.csv"

# The sampler's schedule for sigma = 8 runs lam in [0.032, 3.9]; this ladder
# spans it, with two rungs down where the hard wall dominates the score.
LAMBDAS = (0.05, 0.1, 0.3, 1.0, 3.0)

FIELDS = [
    "config", "train_samples", "hidden_size", "depth", "num_heads", "mlp_ratio",
    "parameters", "train_seconds", "w1_g", "w1_l", "score_rmse",
    *[f"score_rmse_lam{lam:g}" for lam in LAMBDAS],
]


def load_done(path):
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {(row["config"], int(row["train_samples"]))
                for row in csv.DictReader(handle)}


def append(path, row):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", choices=sorted(recipe.CONFIGS))
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--observations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--score-states", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Recompute a cell already present in the CSV.")
    arguments = parser.parse_args()

    if arguments.list or arguments.config is None:
        for name, kwargs in recipe.CONFIGS.items():
            print(f"{name:<8} {kwargs}")
        return

    cell = (arguments.config, int(arguments.train_samples))
    if cell in load_done(SWEEP_CSV) and not arguments.force:
        print(f"skip {cell}: already in {SWEEP_CSV}")
        return

    device = arguments.device
    directory = recipe.model_directory(ARTIFACTS, *cell)
    model, train_seconds = recipe.train_or_load(
        directory, arguments.config, arguments.train_samples, device,
        seed=arguments.train_seed, force=arguments.force_retrain, verbose=True,
    )
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"[{cell}] {parameters:,} parameters")

    _, _, x_observed = hierarchy.observations(arguments.observations, arguments.seed)

    print(f"[{cell}] single-observation distributional fidelity")
    w1_g, w1_l = recipe.posterior_fidelity(
        model, x_observed, arguments.eval_samples, arguments.timesteps, device
    )
    print(f"[{cell}] W1/sigma  g={w1_g:.4f}  l={w1_l:.4f}")

    print(f"[{cell}] single-observation score fidelity")
    scores = recipe.score_fidelity(
        model, x_observed, LAMBDAS, arguments.score_states, device
    )
    for lam in LAMBDAS:
        print(f"[{cell}]   lam={lam:<5g} relative score RMSE {scores[lam]:.4f}")
    print(f"[{cell}] mean relative score RMSE {scores['mean']:.4f}")

    append(SWEEP_CSV, {
        "config": arguments.config, "train_samples": arguments.train_samples,
        **recipe.CONFIGS[arguments.config], "parameters": parameters,
        "train_seconds": round(train_seconds, 1),
        "w1_g": round(w1_g, 5), "w1_l": round(w1_l, 5),
        "score_rmse": round(scores["mean"], 5),
        **{f"score_rmse_lam{lam:g}": round(scores[lam], 5) for lam in LAMBDAS},
    })
    print(f"[{cell}] wrote row to {SWEEP_CSV}")


if __name__ == "__main__":
    main()
