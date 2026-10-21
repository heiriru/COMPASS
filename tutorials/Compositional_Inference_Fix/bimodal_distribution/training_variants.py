#!/usr/bin/env python3
"""Two recipe questions, answered on the same footing: normalization and time sampling.

Both are changes that cost nothing at inference and add no parameters, so if
either works it is free accuracy. The main run's small network has a row-score
error of 0.166 on average and 0.44 at the smallest probed noise level, and the
question is which of these two knobs moves that.

The 2 x 2
---------
``baseline``              ``sign_model``, ``time_sampling="mixture"`` -- what the
                          main run used.
``log_sigma``             the same data, times drawn log-uniformly in the noise
                          scale instead of half uniformly in ``t``. The error is
                          concentrated at small ``lambda``, which is exactly
                          where the uniform-in-``t`` half of the mixture puts
                          almost none of its budget, so this is the targeted fix.
``normalized``            ``normalized_sign_model`` -- the identical problem with
                          every coordinate divided by ``sigma_g``, so the data
                          has unit scale and the sampler's default ``N(0, 1)``
                          prior is exactly right.
``normalized_log_sigma``  both.

Grading at corresponding noise levels
-------------------------------------
Relative RMS score error is scale free, but the *noise levels* are not: the
diffusion schedule is fixed while the standardized data is 1.5x smaller, so a
level ``lambda`` in the original problem is ``lambda / 1.5`` in the standardized
one. Grading both at the same numerical ``lambda`` would compare different
physical noise levels and flatter whichever one happened to be graded further
from its hard region. Every normalized variant is therefore graded at
``FIDELITY_LAMBDAS / SCALE``.

The end-to-end check
--------------------
Fidelity is the direct measurement, but the thing that matters is the posterior,
so each variant is also run through two sampler arms -- the best corrector-free
one and the stock corrector one -- and scored with ``W1 / sigma``, which is also
scale free and so directly comparable across variants.

Usage:
    python training_variants.py                  # train what is missing, grade all
    python training_variants.py --no-sampling    # fidelity only, much faster
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

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"

VARIANTS = {
    "baseline": {"normalized": False, "time_sampling": "mixture"},
    "log_sigma": {"normalized": False, "time_sampling": "log_sigma"},
    "normalized": {"normalized": True, "time_sampling": "mixture"},
    "normalized_log_sigma": {"normalized": True, "time_sampling": "log_sigma"},
}

# The two arms worth spending sampling time on: the best corrector-free setting
# found by efficiency_sweep.py, and the stock corrector setting, which the main
# run showed react to row-score error in opposite directions.
SAMPLER_ARMS = {
    "churn8_steps200": {
        "label": "churn 8, 200 steps", "churn": 8.0, "timesteps": 200,
        "sample_kwargs": {"method": "dpm", "order": 2,
                          "correction": "gauss_jacobian",
                          "corrector_steps": 0, "final_corrector_steps": 0,
                          "terminal_corrector_steps": 0},
    },
    "correctors10": {
        "label": "10 correctors, snr 0.2", "churn": None, "timesteps": 100,
        "sample_kwargs": {"method": "dpm", "order": 2,
                          "correction": "gauss_jacobian",
                          "corrector_steps_interval": 1, "corrector_steps": 10,
                          "final_corrector_steps": 3, "snr": 0.2},
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-samples", type=int, default=3_000)
    parser.add_argument("--no-sampling", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    if arguments.device is None:
        from autocvd import autocvd

        autocvd(num_gpus=1, interval=1)
        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"

    import normalized_sign_model
    import run_experiment as experiment
    import sign_model
    import train as recipe

    config = arguments.config or recipe.DEFAULT_CONFIG
    print(f"Device: {arguments.device}; config {config}")

    fidelity_rows, sampling_rows = [], []
    for name in arguments.variants:
        settings = VARIANTS[name]
        module = (normalized_sign_model if settings["normalized"]
                  else sign_model)
        scale = getattr(module, "SCALE", 1.0)
        lambdas = tuple(lam / scale for lam in experiment.FIDELITY_LAMBDAS)

        model, seconds = recipe.train_or_load(
            ROOT, module, config, device=arguments.device,
            force=arguments.force, verbose=True,
            time_sampling=settings["time_sampling"],
        )
        _, _, _, y = module.observations(experiment.OBSERVATIONS,
                                        experiment.DATA_SEEDS["sign"])
        fidelity = recipe.score_fidelity(
            model, module, y, lambdas, device=arguments.device
        )
        # Report against the *original* problem's noise levels, so the columns of
        # the table line up even though the standardized runs were graded at
        # lambda / scale.
        row = {"variant": name, "config": config,
               "normalized": settings["normalized"],
               "time_sampling": settings["time_sampling"],
               "train_seconds": round(seconds),
               **{f"rel_rmse_lam_{original:g}": fidelity[float(original / scale)]
                  for original in experiment.FIDELITY_LAMBDAS},
               "rel_rmse_mean": fidelity["mean"]}
        fidelity_rows.append(row)
        print(f"\n{name:<22} mean {fidelity['mean']:.4f}   "
              + "  ".join(f"lam={original:g}:"
                          f"{fidelity[float(original / scale)]:.3f}"
                          for original in experiment.FIDELITY_LAMBDAS),
              flush=True)

        if arguments.no_sampling:
            continue

        problem = experiment.build_problem(
            module, experiment.OBSERVATIONS, experiment.DATA_SEEDS["sign"],
            arguments.device,
        )
        for arm_name, arm in SAMPLER_ARMS.items():
            experiment.METHODS[arm_name] = {
                "label": arm["label"], "short": arm["label"],
                "churn": arm["churn"], "linestyle": "-",
                "sample_kwargs": arm["sample_kwargs"],
            }
            samples, sampler, runtime = experiment.draw(
                model, problem, arm_name, arguments.num_samples,
                arm["timesteps"], experiment.INFERENCE_SEED,
            )
            shared = samples[0, :, module.GLOBAL_INDEX].to(torch.float64).numpy()
            local = samples[:, :, module.LOCAL_INDEX].to(torch.float64).numpy()
            metrics = experiment.metrics(
                problem, shared, local, runtime,
                getattr(sampler, "score_network_calls", 0),
            )
            sampling_rows.append({"variant": name, "arm": arm_name, **metrics})
            print(f"  {arm['label']:<22} W1/s {metrics['shared_w1_over_sigma']:.4f}"
                  f"  gap {metrics['gap_draws']:.0f}"
                  f"  P+ {metrics['positive_mass']:.3f}"
                  f"  width/scale {metrics['mode_width'] * scale:.3f}"
                  f"  {runtime:.0f}s", flush=True)

    for rows, filename in ((fidelity_rows, "training_variants_fidelity.csv"),
                           (sampling_rows, "training_variants_sampling.csv")):
        if not rows:
            continue
        path = ARTIFACTS / filename
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
