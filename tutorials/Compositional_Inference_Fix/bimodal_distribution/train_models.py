#!/usr/bin/env python3
"""Train (or reuse) one small score network per benchmark, then grade it.

Split out from ``run_experiment.py`` so training can be run once, on its own
GPU, and the inference sweep can then be rerun as often as the figures need
without ever touching the checkpoints. The grade printed at the end is the
relative RMS error of the learned row score against the closed-form one at the
noise levels the composed sampler queries -- the number that licenses reading a
downstream failure as a composition failure rather than a network failure.

Usage:
    python train_models.py                 # both benchmarks, reuse checkpoints
    python train_models.py --models sign --force
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
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["sign", "control"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--time-sampling", default=None,
                        choices=("mixture", "log_sigma", "uniform"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    if arguments.device is None:
        from autocvd import autocvd

        autocvd(num_gpus=1, interval=1)
        import torch

        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"

    import run_experiment as experiment
    import train as recipe

    config = arguments.config or recipe.DEFAULT_CONFIG
    samples = arguments.train_samples or recipe.TRAIN_SAMPLES
    time_sampling = arguments.time_sampling or recipe.DEFAULT_TIME_SAMPLING
    for name in arguments.models:
        module = experiment.MODELS[name]
        print(f"\n######## {module.TITLE} ########", flush=True)
        model, seconds = recipe.train_or_load(
            ROOT, module, config, device=arguments.device, force=arguments.force,
            verbose=True, train_samples=samples, time_sampling=time_sampling,
        )
        _, _, _, y = module.observations(
            experiment.OBSERVATIONS, experiment.DATA_SEEDS[name]
        )
        fidelity = recipe.score_fidelity(
            model, module, y, experiment.FIDELITY_LAMBDAS, device=arguments.device
        )
        print(f"  trained in {seconds:.0f}s; "
              f"{recipe.parameter_count(module, config):,} parameters")
        print("  learned row score, relative RMS vs the exact row score:")
        for key, value in fidelity.items():
            print(f"    {key if key == 'mean' else f'lam={key:g}':>10}: {value:.4f}")


if __name__ == "__main__":
    main()
