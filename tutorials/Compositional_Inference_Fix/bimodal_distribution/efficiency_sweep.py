#!/usr/bin/env python3
"""What buys accuracy per unit of compute -- and does it buy it for both scores?

The main sweep left one arm best with the exact row score (``gauss_jacobian``
plus ten Langevin correctors) and a *different* arm best with a learned one
(``gauss_jacobian`` plus churn, corrector-free). That is an awkward place to
stop: it means the recommended configuration depends on something the user
cannot easily measure. This script looks for one setting that is good with both,
at a cost comparable to the cheap arms.

The hypothesis it tests
-----------------------
Correctors are unadjusted Langevin steps: ``x <- x + snr lam^2 s(x) + noise``,
with no Metropolis correction. Ten of them per level do not *nudge* the state,
they **equilibrate** it to the stationary law of whatever field they are given.
With an exact field that is the right law and the arm wins; with a slightly
over-confident learned field it is an over-concentrated law, which is exactly
the failure seen in the main run (mode s.d. 0.164-0.177 against an exact 0.199,
and no better with a nine-fold larger network).

If that reading is right, the fix is to spend the same compute on something that
does *not* equilibrate:

``steps200`` / ``steps400``   the corrector-free churned predictor with more
                              integrator steps. A predictor integrates a field
                              error once, however many steps it takes, so extra
                              steps should help the exact score (less
                              discretisation error) without opening the learned
                              score to the equilibration failure.
``correctors2``               two correctors instead of ten -- enough to nudge,
                              too few to equilibrate.
``snr005``                    ten correctors at a quarter of the step size:
                              the same question from the other direction.
``churn2``                    weaker churn, to separate "churn is leaking mass
                              into the valley" from "the predictor is fine and
                              churn strength is the wrong knob".

All arms use ``correction="gauss_jacobian"``, which the main run established as
the better of the two Gaussian rules at every noise level, so the only thing
varying here is how the sampler spends its score evaluations.

Usage:
    python efficiency_sweep.py                       # both score sources
    python efficiency_sweep.py --scores analytic     # seconds, no network
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"

CORRECTOR_FREE = {"corrector_steps": 0, "final_corrector_steps": 0,
                  "terminal_corrector_steps": 0}
JACOBIAN = {"method": "dpm", "order": 2, "correction": "gauss_jacobian"}

# ``timesteps`` is not a sample_kwarg -- it is a separate argument to
# MultiObsSampler.sample -- so each arm carries it alongside its kwargs.
ARMS = {
    "churn8_steps100": {
        "label": "churn 8, 100 steps (baseline)", "churn": 8.0, "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, **CORRECTOR_FREE},
    },
    "churn8_steps200": {
        "label": "churn 8, 200 steps", "churn": 8.0, "timesteps": 200,
        "sample_kwargs": {**JACOBIAN, **CORRECTOR_FREE},
    },
    "churn8_steps400": {
        "label": "churn 8, 400 steps", "churn": 8.0, "timesteps": 400,
        "sample_kwargs": {**JACOBIAN, **CORRECTOR_FREE},
    },
    "churn2_steps200": {
        "label": "churn 2, 200 steps", "churn": 2.0, "timesteps": 200,
        "sample_kwargs": {**JACOBIAN, **CORRECTOR_FREE},
    },
    "correctors2": {
        "label": "2 correctors, snr 0.2", "churn": None, "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, "corrector_steps_interval": 1,
                          "corrector_steps": 2, "final_corrector_steps": 3,
                          "snr": 0.2},
    },
    "correctors10": {
        "label": "10 correctors, snr 0.2 (baseline)", "churn": None,
        "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, "corrector_steps_interval": 1,
                          "corrector_steps": 10, "final_corrector_steps": 3,
                          "snr": 0.2},
    },
    "correctors10_snr005": {
        "label": "10 correctors, snr 0.05", "churn": None, "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, "corrector_steps_interval": 1,
                          "corrector_steps": 10, "final_corrector_steps": 3,
                          "snr": 0.05},
    },
    # The mixing-time arms. A Langevin corrector chain moves a distance that
    # scales with steps x snr, so ten steps at snr 0.2 and three at snr 0.7 cost
    # very different numbers of network evaluations for a similar amount of
    # relaxation. If accuracy tracks mixing time rather than evaluation count,
    # these are the cheap way to buy it -- and the point at which they stop
    # working is the stability limit of an unadjusted Langevin step.
    "correctors3_snr07": {
        "label": "3 correctors, snr 0.7", "churn": None, "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, "corrector_steps_interval": 1,
                          "corrector_steps": 3, "final_corrector_steps": 3,
                          "snr": 0.7},
    },
    "correctors4_snr05": {
        "label": "4 correctors, snr 0.5", "churn": None, "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, "corrector_steps_interval": 1,
                          "corrector_steps": 4, "final_corrector_steps": 3,
                          "snr": 0.5},
    },
    "correctors5_snr04": {
        "label": "5 correctors, snr 0.4", "churn": None, "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, "corrector_steps_interval": 1,
                          "corrector_steps": 5, "final_corrector_steps": 3,
                          "snr": 0.4},
    },
    "correctors10_snr04": {
        "label": "10 correctors, snr 0.4", "churn": None, "timesteps": 100,
        "sample_kwargs": {**JACOBIAN, "corrector_steps_interval": 1,
                          "corrector_steps": 10, "final_corrector_steps": 3,
                          "snr": 0.4},
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS))
    parser.add_argument("--scores", nargs="+", default=["analytic", "learned"])
    parser.add_argument("--num-samples", type=int, default=3_000)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    if arguments.device is None:
        try:
            from autocvd import autocvd

            autocvd(num_gpus=1, interval=1)
        except ImportError:
            pass
        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"

    import row_scores
    import run_experiment as experiment
    import sign_model
    import train as recipe
    from compass.SDE import VESDE

    module = sign_model
    problem = experiment.build_problem(
        module, experiment.OBSERVATIONS, experiment.DATA_SEEDS["sign"],
        arguments.device,
    )
    print(f"Device: {arguments.device}; exact P(g>0) "
          f"{problem['positive_mass']:.3f}, mode s.d. "
          f"{experiment.metrics(problem, problem['exact_shared'][:arguments.num_samples], problem['exact_local'][:arguments.num_samples].T, 0.0, 0)['mode_width']:.3f}")

    models = {}
    if "analytic" in arguments.scores:
        models["analytic"] = row_scores.AnalyticModel(
            VESDE(sigma=recipe.SDE_KWARGS["sigma"]), module,
            device=arguments.device, seed=experiment.INFERENCE_SEED,
        )
    for score in arguments.scores:
        if score in experiment.SCORE_CONFIG:
            models[score], _ = recipe.train_or_load(
                ROOT, module, experiment.SCORE_CONFIG[score],
                device=arguments.device, verbose=False,
            )

    rows = []
    for score in arguments.scores:
        for name in arguments.arms:
            arm = ARMS[name]
            # The sweep's arms are not in experiment.METHODS, so register each
            # one there for the duration of its run; experiment.draw reads the
            # churn strength and sample kwargs from that table.
            experiment.METHODS[name] = {"label": arm["label"],
                                        "short": arm["label"],
                                        "churn": arm["churn"],
                                        "linestyle": "-",
                                        "sample_kwargs": arm["sample_kwargs"]}
            print(f"  {score:<10}{arm['label']:<32}", end="", flush=True)
            samples, sampler, runtime = experiment.draw(
                models[score], problem, name, arguments.num_samples,
                arm["timesteps"], experiment.INFERENCE_SEED,
            )
            shared = samples[0, :, module.GLOBAL_INDEX].to(torch.float64).numpy()
            local = samples[:, :, module.LOCAL_INDEX].to(torch.float64).numpy()
            row = experiment.metrics(
                problem, shared, local, runtime,
                getattr(sampler, "score_network_calls", 0),
            )
            rows.append({"score": score, "arm": name, "label": arm["label"],
                         "timesteps": arm["timesteps"], **row})
            print(f"W1/s {row['shared_w1_over_sigma']:.4f}  "
                  f"gap {row['gap_draws']:.0f}  "
                  f"P+ {row['positive_mass']:.3f}  "
                  f"width {row['mode_width']:.3f}  "
                  f"calls {row['score_network_calls']}  "
                  f"{runtime:.0f}s")

    path = ARTIFACTS / f"efficiency_sweep_{'_'.join(arguments.scores)}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
