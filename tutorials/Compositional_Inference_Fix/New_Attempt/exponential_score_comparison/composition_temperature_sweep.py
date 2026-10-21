#!/usr/bin/env python3
"""Small exact-score regression for compositional likelihood temperature.

Runs the requested method combinations on the analytic Gaussian, Laplace and
exponential hierarchy problems. The DPM2 methods use stochastic-predictor churn
(eta=2); F-NPSE uses annealed Langevin. Exact row scores and exact tabulated
shared posteriors isolate composition/sampling error from network training.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from pathlib import Path

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(CPU_THREAD_LIMIT)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import compare  # noqa: E402
import symmetric_row_scores as row_scores  # noqa: E402
from damping_observation_sweep import shared_metrics  # noqa: E402
from stochastic_predictor_experiments import (  # noqa: E402
    churned_dpm_sampler,
    patched_dpm_sampler,
)
from symmetric_experiments import PROBLEMS, patched_problem_module  # noqa: E402
from compass.MultiObsSampler import MultiObsSampler  # noqa: E402
from compass.SDE import VESDE  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CSV_PATH = ARTIFACTS / "composition_temperature_sweep.csv"
JSON_PATH = ARTIFACTS / "composition_temperature_summary.json"

METHODS = {
    "hierarchical": {
        "label": "DPM2 + gauss_hierarchical + churn",
        "correction": "gauss_hierarchical",
        "family": "dpm",
    },
    "jacobian": {
        "label": "DPM2 + gauss_jacobian + churn",
        "correction": "gauss_jacobian",
        "family": "dpm",
    },
    "fnpe": {
        "label": "Langevin + F-NPSE",
        "correction": "fnpe",
        "family": "langevin",
    },
}


def write_rows(rows):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def one_cell(problem_name, method_name, count, beta, args):
    specification = PROBLEMS[problem_name]
    module = specification["module"]
    sde = VESDE(sigma=compare.recipe.SDE_KWARGS["sigma"])
    model = row_scores.AnalyticModel(
        sde, specification["row_score"], module, device=args.device,
        nodes=args.quadrature_nodes, seed=args.seed,
    )
    _, _, x = module.observations(count, args.seed)
    grid, density, _, exact_mean, exact_std = module.shared_reference(x)
    problem = {
        "data": torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1),
        "device": args.device,
    }
    method = METHODS[method_name]
    kwargs = {
        "correction": method["correction"],
        "composition_temperature": beta,
    }
    if method["family"] == "dpm":
        kwargs.update(
            method="dpm", order=2, corrector_steps=0,
            final_corrector_steps=0, terminal_corrector_steps=0,
        )
        if method_name == "hierarchical":
            kwargs.update(
                precision_est_samples=args.pilot_draws,
                precision_est_timesteps=args.timesteps,
            )
        sampler_context = patched_dpm_sampler(
            churned_dpm_sampler(args.churn_eta)
        )
    else:
        kwargs.update(
            method="langevin", corrector_steps=args.langevin_steps,
            final_corrector_steps=0, snr=args.snr,
        )
        sampler_context = contextlib.nullcontext()

    started = time.perf_counter()
    try:
        with patched_problem_module(module), sampler_context:
            samples, sampler, _ = compare.draw(
                model, problem, kwargs, args.num_samples, args.timesteps,
                args.seed, args.denoise_clamp,
            )
        shared = samples[0, :, module.HIERARCHY[0]].numpy()
        metrics = shared_metrics(
            shared, grid, density, exact_mean, exact_std,
            probabilities=min(2000, max(200, len(shared))),
        )
        status = "ok"
        calls = int(getattr(sampler, "score_network_calls", 0))
    except (RuntimeError, ValueError, AssertionError) as error:
        shared = np.array([])
        metrics = {
            "shared_w1_over_sigma": np.nan,
            "shared_mean_error_sigma": np.nan,
            "shared_width_ratio": np.nan,
            "sample_mean": np.nan,
            "sample_std": np.nan,
        }
        status = f"FAILED: {type(error).__name__}: {error}"[:220]
        calls = 0

    return {
        "problem": problem_name,
        "method": method_name,
        "label": method["label"],
        "observations": count,
        "beta": beta,
        "churn_eta": args.churn_eta if method["family"] == "dpm" else 0.0,
        "status": status,
        "exact_mean": float(exact_mean),
        "exact_std": float(exact_std),
        **metrics,
        "num_samples": int(shared.size),
        "score_calls": calls,
        "seconds": round(time.perf_counter() - started, 3),
    }


def summarize(rows):
    ok = [row for row in rows if row["status"] == "ok"]
    summary = {"rows": len(rows), "successful_rows": len(ok), "contraction": {}}
    for problem in sorted({row["problem"] for row in ok}):
        summary["contraction"][problem] = {}
        for method in METHODS:
            for beta in sorted({row["beta"] for row in ok}):
                selected = sorted(
                    (row for row in ok
                     if row["problem"] == problem
                     and row["method"] == method and row["beta"] == beta),
                    key=lambda row: row["observations"],
                )
                if not selected:
                    continue
                widths = [row["sample_std"] for row in selected]
                summary["contraction"][problem][f"{method}_beta{beta:g}"] = {
                    "observations": [row["observations"] for row in selected],
                    "sample_std": widths,
                    "strictly_decreasing": all(
                        later < earlier
                        for earlier, later in zip(widths, widths[1:])
                    ),
                    "mean_w1_over_sigma": float(np.mean(
                        [row["shared_w1_over_sigma"] for row in selected]
                    )),
                }
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--problems", nargs="+",
                        default=["gauss", "laplace", "exponential"])
    result.add_argument("--methods", nargs="+", default=list(METHODS))
    result.add_argument("--observations", nargs="+", type=int,
                        default=[1, 4, 16])
    result.add_argument("--temperatures", nargs="+", type=float,
                        default=[0.6, 1.0])
    result.add_argument("--num-samples", type=int, default=384)
    result.add_argument("--timesteps", type=int, default=40)
    result.add_argument("--quadrature-nodes", type=int, default=257)
    result.add_argument("--pilot-draws", type=int, default=256)
    result.add_argument("--langevin-steps", type=int, default=8)
    result.add_argument("--snr", type=float, default=0.2)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--seed", type=int, default=31)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.device.startswith("cuda"):
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
        args.device = "cuda"
    rows = []
    for problem in args.problems:
        for count in args.observations:
            for method in args.methods:
                for beta in args.temperatures:
                    row = one_cell(problem, method, count, beta, args)
                    rows.append(row)
                    print(
                        f"{problem:11s} n={count:2d} {method:12s} beta={beta:.2f} "
                        f"{row['status']:3s} W1/sd={row['shared_w1_over_sigma']:.3f} "
                        f"width={row['shared_width_ratio']:.3f} "
                        f"{row['seconds']:.1f}s",
                        flush=True,
                    )
                    write_rows(rows)
    summary = summarize(rows)
    JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {CSV_PATH}\nWrote {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
