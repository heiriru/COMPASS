#!/usr/bin/env python3
"""How small a score network can learn the exponential hierarchy?

The default backbone for this problem is COMPASS's generic HQ transformer at
5.2M parameters, which is absurd for three scalar nodes. This sweep trains the
same recipe at shrinking widths and depths and measures two different things,
because they can fail independently:

**Network fidelity** (`w1_g_over_sigma`, `w1_l_over_sigma`) — single-observation
posteriors only. For each of the 30 observed `x_j`, draw `p(g, l | x_j)` with the
ordinary sampler (no composition, no correction) and compare each marginal to its
closed form by 1-D Wasserstein distance, in units of that posterior's own std.
This grades the network alone; nothing compositional can hide in it.

**Pipeline accuracy** (`global_map_error_sigma`, `local_map_*`) — the full
`new_method.py` run at n=30 with that checkpoint. This is what actually matters
for the figure, and it is strictly harder: composition amplifies score error
coherently across observations.

Both exact references come from `exponential_hierarchy.py`: for one observation,
`p(g | x) ∝ N(g; mu, sigma^2) e^{rate·g} (x - g) 1[g <= x]` (the N=1 case of the
tall posterior) and `p(l | x) = ∫ p(g | x) · Uniform(l; g, x) dg`.

One config per process so the sweep can be spread across GPUs; rows are appended
as they finish and existing rows are skipped, so it resumes after a kill.

Usage:
    CUDA_VISIBLE_DEVICES=1 python capacity_sweep.py --config h32d3
    python capacity_sweep.py --list
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exponential_hierarchy as expo  # noqa: E402
import new_method  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SWEEP_CSV = ARTIFACTS / "capacity_sweep.csv"

# Ordered large -> small. "h128d6" is the HQ backbone every other experiment in
# this directory uses, kept as the accuracy ceiling to measure the rest against.
CONFIGS = {
    "h128d6": {"hidden_size": 128, "depth": 6, "num_heads": 8, "mlp_ratio": 4},
    "h32d3": {"hidden_size": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 2},
    "h16d2": {"hidden_size": 16, "depth": 2, "num_heads": 2, "mlp_ratio": 1},
    "h16d1": {"hidden_size": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 1},
    "h8d1": {"hidden_size": 8, "depth": 1, "num_heads": 2, "mlp_ratio": 1},
    "h4d1": {"hidden_size": 4, "depth": 1, "num_heads": 1, "mlp_ratio": 1},
}

FIELDS = [
    "config", "hidden_size", "depth", "num_heads", "mlp_ratio", "parameters",
    "train_seconds", "w1_g_over_sigma", "w1_l_over_sigma",
    "global_map", "global_map_error_sigma", "global_width_ratio",
    "local_map_error_vs_posterior_mean_sigma",
    "local_map_error_vs_conditional_mode_sigma", "sample_runtime_seconds",
]


def model_directory(config):
    return ARTIFACTS / "models" / f"exponential_hierarchy_{config}"


# ---------------------------------------------------------------------------
# Exact single-observation references
# ---------------------------------------------------------------------------

def single_observation_reference(x, points=20001, span=8.0):
    """Grids and normalized weights for p(g | x) and p(l | x), one observation."""
    grid_g = np.linspace(x - span, x, points)
    log_density = expo.log_shared_posterior([x], grid_g)
    weights = np.exp(log_density - log_density.max())
    weights /= weights.sum()

    # p(l | x) = int p(g | x) Uniform(l; g, x) dg, i.e. for each l the shared
    # mass below it, each unit weighted by 1 / (x - g).
    grid_l = np.linspace(x - span, x, points)
    density = np.zeros_like(grid_l)
    gap = np.maximum(x - grid_g, 1e-12)
    contribution = weights / gap
    cumulative = np.cumsum(contribution)
    inside = grid_l < x
    indices = np.searchsorted(grid_g, grid_l[inside], side="right") - 1
    density[inside] = np.where(indices >= 0, cumulative[np.clip(indices, 0, None)], 0.0)
    total = density.sum()
    return grid_g, weights, grid_l, density / total if total > 0 else density


def wasserstein_1d(samples, grid, weights):
    """W1 between draws and a density tabulated on a grid."""
    samples = np.sort(np.asarray(samples, dtype=np.float64))
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    quantiles = (np.arange(len(samples)) + 0.5) / len(samples)
    positions = np.clip(np.searchsorted(cumulative, quantiles), 0, len(grid) - 1)
    return float(np.abs(samples - grid[positions]).mean())


def evaluate_network(model, x_observed, num_samples, timesteps, device,
                     chunk=5):
    """Mean W1 of the single-observation marginals, in units of their own std.

    Chunked over observations: the widest config puts observations x samples
    rows through the transformer at once, which does not fit an 11 GB card at
    the sweep's sample count. Chunking changes nothing statistically -- these
    are independent single-observation posteriors.
    """
    x = torch.as_tensor(np.asarray(x_observed, dtype=np.float32)).reshape(-1, 1)
    torch.manual_seed(1208)
    torch.cuda.manual_seed_all(1208)
    blocks = []
    for start in range(0, x.shape[0], chunk):
        blocks.append(model.sample(
            x=x[start:start + chunk], num_samples=num_samples,
            timesteps=timesteps, method="dpm", order=2,
            corrector_steps_interval=1, corrector_steps=10,
            final_corrector_steps=3, snr=0.2, device=device, verbose=False,
        ).detach().cpu().numpy())
    samples = np.concatenate(blocks, axis=0)

    errors_g, errors_l = [], []
    for index, value in enumerate(np.asarray(x_observed, dtype=np.float64)):
        grid_g, weights_g, grid_l, weights_l = single_observation_reference(value)
        mean_g = float((weights_g * grid_g).sum())
        std_g = float(max((weights_g * grid_g**2).sum() - mean_g**2, 0.0) ** 0.5)
        mean_l = float((weights_l * grid_l).sum())
        std_l = float(max((weights_l * grid_l**2).sum() - mean_l**2, 0.0) ** 0.5)
        errors_g.append(wasserstein_1d(samples[index, :, 0], grid_g, weights_g) / std_g)
        errors_l.append(wasserstein_1d(samples[index, :, 1], grid_l, weights_l) / std_l)
    return float(np.mean(errors_g)), float(np.mean(errors_l))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_done(path):
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {row["config"] for row in csv.DictReader(handle)}


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
    parser.add_argument("--config", choices=sorted(CONFIGS), required=False)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--observations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Recompute a config already present in the CSV.")
    arguments = parser.parse_args()

    if arguments.list or arguments.config is None:
        for name, kwargs in CONFIGS.items():
            print(f"{name:<8} {kwargs}")
        return

    if arguments.config in load_done(SWEEP_CSV) and not arguments.force:
        print(f"skip {arguments.config}: already in {SWEEP_CSV}")
        return

    kwargs = CONFIGS[arguments.config]
    directory = model_directory(arguments.config)
    device = arguments.device

    # Train with this config's architecture, everything else held fixed.
    original = dict(expo.MODEL_KWARGS)
    expo.MODEL_KWARGS = {"sde_type": "vesde", "sigma": 8.0, **kwargs}
    started = time.perf_counter()
    model = expo.train_or_load(
        directory, device, seed=arguments.train_seed,
        force=arguments.force_retrain, verbose=True,
    )
    train_seconds = time.perf_counter() - started
    expo.MODEL_KWARGS = original
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"[{arguments.config}] {parameters:,} parameters")

    global_truth, local_truth, x_observed = expo.observations(
        arguments.observations, arguments.seed
    )
    del global_truth, local_truth

    print(f"[{arguments.config}] single-observation fidelity")
    w1_g, w1_l = evaluate_network(
        model, x_observed, arguments.eval_samples, arguments.timesteps, device
    )
    print(f"[{arguments.config}] W1/sigma  g={w1_g:.4f}  l={w1_l:.4f}")

    print(f"[{arguments.config}] full pipeline at n={arguments.observations}")
    pipeline_arguments = SimpleNamespace(
        observations=arguments.observations, seed=arguments.seed,
        trained_device=device, device=device, model_dir=directory,
        train_seed=arguments.train_seed, force_retrain=False,
        quick_train=False, verbose=False, num_samples=arguments.num_samples,
        timesteps=arguments.timesteps, denoise_clamp=5.0, excursion_sigma=15.0,
        kde_bandwidth=None, map_timesteps=200, map_iterations=3, map_eps=1e-3,
    )
    problem = new_method.build_exponential_problem(
        arguments.observations, arguments.seed, device, pipeline_arguments
    )
    pipeline_keys = (
        "global_map", "global_map_error_sigma", "global_width_ratio",
        "local_map_error_vs_posterior_mean_sigma",
        "local_map_error_vs_conditional_mode_sigma", "sample_runtime_seconds",
    )
    try:
        _, metrics = new_method.run_problem(problem, pipeline_arguments)
        pipeline = {key: metrics[key] for key in pipeline_keys}
    except torch.OutOfMemoryError:
        # A real result, not a skipped cell: the composed sampler puts
        # observations x samples rows through the network in one call, so at
        # this width the run does not fit the GPU at all.
        print(f"[{arguments.config}] pipeline OOM at "
              f"{arguments.observations}x{arguments.num_samples} rows")
        pipeline = {key: "OOM" for key in pipeline_keys}

    append(SWEEP_CSV, {
        "config": arguments.config, **kwargs, "parameters": parameters,
        "train_seconds": train_seconds, "w1_g_over_sigma": w1_g,
        "w1_l_over_sigma": w1_l, **pipeline,
    })
    print(f"[{arguments.config}] wrote row to {SWEEP_CSV}")


if __name__ == "__main__":
    main()
