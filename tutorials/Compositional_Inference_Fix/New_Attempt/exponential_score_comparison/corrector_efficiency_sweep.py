#!/usr/bin/env python3
"""Accuracy against wall clock for the knobs that make gauss_jacobian cheap.

``01_gauss_jacobian_predictor_only.png`` is cheap and wrong; the production
corrector setting is accurate and 5x the cost of Langevin+F-NPSE. Three knobs sit
between them, and none of them is the corrector count alone:

``corrector_steps``          how many Langevin steps per corrected level.
``corrector_steps_interval`` how often a level is corrected at all.
``jacobian_refresh``         how often the score Jacobian is recomputed. The
                             curvature varies far more slowly in ``t`` than the
                             score does, so reusing it across evaluations
                             removes most of the ``len(latent block)`` JVP
                             passes that make this rule expensive -- the only
                             part of its cost that F-NPSE does not also pay.

The reference points are the same problem's ``langevin_fnpe`` and predictor-only
arms, so the output is directly comparable to the corrector sweep in
``../../global_local_corrector_tradeoff``.

Usage:
    python corrector_efficiency_sweep.py --device cuda
    python corrector_efficiency_sweep.py --replot
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
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "corrector_efficiency"

BASE = {"method": "dpm", "order": 2, "correction": "gauss_jacobian", "snr": 0.2}

# (label, corrector_steps, corrector_steps_interval, jacobian_refresh)
GRID = [
    ("predictor only", 0, 1, 1),
    ("2 corr, every level", 2, 1, 1),
    ("2 corr, every level, refresh 5", 2, 1, 5),
    ("2 corr, every level, refresh 10", 2, 1, 10),
    ("2 corr, every 2nd level, refresh 5", 2, 2, 5),
    ("2 corr, every 5th level, refresh 5", 2, 5, 5),
    ("1 corr, every level, refresh 5", 1, 1, 5),
    ("10 corr, every level, refresh 10", 10, 1, 10),
]

REFERENCE = {
    "label": "Langevin + F-NPSE",
    "sample_kwargs": {"method": "langevin", "correction": "fnpe",
                      "corrector_steps": 10, "snr": 0.2},
}


def run(problem, arguments):
    rows = []
    for label, steps, interval, refresh in GRID:
        kwargs = dict(BASE)
        kwargs.update({
            "corrector_steps": steps, "corrector_steps_interval": interval,
            "final_corrector_steps": 3 if steps else 0,
            "terminal_corrector_steps": 0, "jacobian_refresh": refresh,
        })
        print(f"\n=== gauss_jacobian: {label} ===")
        samples, sampler, runtime = compare.draw(
            problem["model"], problem, kwargs, arguments.num_samples,
            arguments.timesteps, arguments.seed, arguments.denoise_clamp,
        )
        result = compare.kde_map_pipeline(
            problem["model"], problem, samples, arguments
        )
        metrics = compare.evaluate(problem, result, sampler, runtime, [])
        rows.append({
            "label": label, "rule": "gauss_jacobian", "corrector_steps": steps,
            "corrector_interval": interval, "jacobian_refresh": refresh,
            "seconds": runtime, "calls": int(sampler.score_network_calls),
            "w1_over_sigma": metrics["shared_w1_over_sigma"],
            "mean_error_sigma": metrics["shared_mean_error_sigma"],
            "width_ratio": metrics["shared_width_ratio"],
            "global_map_error_sigma": metrics["global_map_error_sigma"],
        })
        print(f"  {runtime:7.1f}s  {rows[-1]['calls']:5d} calls  "
              f"W1 {rows[-1]['w1_over_sigma']:.4f}")

    print(f"\n=== {REFERENCE['label']} ===")
    samples, sampler, runtime = compare.draw(
        problem["model"], problem, dict(REFERENCE["sample_kwargs"]),
        arguments.num_samples, arguments.timesteps, arguments.seed,
        arguments.denoise_clamp,
    )
    result = compare.kde_map_pipeline(problem["model"], problem, samples, arguments)
    metrics = compare.evaluate(problem, result, sampler, runtime, [])
    rows.append({
        "label": REFERENCE["label"], "rule": "fnpe", "corrector_steps": 10,
        "corrector_interval": 1, "jacobian_refresh": 0,
        "seconds": runtime, "calls": int(sampler.score_network_calls),
        "w1_over_sigma": metrics["shared_w1_over_sigma"],
        "mean_error_sigma": metrics["shared_mean_error_sigma"],
        "width_ratio": metrics["shared_width_ratio"],
        "global_map_error_sigma": metrics["global_map_error_sigma"],
    })
    print(f"  {runtime:7.1f}s  {rows[-1]['calls']:5d} calls  "
          f"W1 {rows[-1]['w1_over_sigma']:.4f}")
    return rows


def plot(rows, output):
    NAVY, TEAL, GOLD, MUTED = "#17223B", "#2A9D8F", "#FFB703", "#8A8F98"
    fig, axis = plt.subplots(figsize=(10.0, 6.4))
    reference = next(row for row in rows if row["rule"] == "fnpe")
    jacobian = [row for row in rows if row["rule"] == "gauss_jacobian"]

    axis.axvline(reference["seconds"], color=GOLD, ls="--", lw=1.6)
    axis.axhline(reference["w1_over_sigma"], color=GOLD, ls="--", lw=1.6)
    axis.plot(reference["seconds"], reference["w1_over_sigma"], "^", ms=11,
              color=GOLD, zorder=6)
    axis.annotate(reference["label"],
                  (reference["seconds"], reference["w1_over_sigma"]),
                  textcoords="offset points", xytext=(10, -14), fontsize=9,
                  color=NAVY, fontweight="bold")

    axis.plot([row["seconds"] for row in jacobian],
              [row["w1_over_sigma"] for row in jacobian], "o", ms=9,
              color=TEAL, zorder=5)
    # Several configurations land within a few seconds and a few percent of one
    # another, so labels are placed by rank rather than with a fixed offset.
    ordered = sorted(jacobian, key=lambda row: row["seconds"])
    for index, row in enumerate(ordered):
        right = index % 2 == 0
        axis.annotate(
            row["label"], (row["seconds"], row["w1_over_sigma"]),
            textcoords="offset points",
            xytext=(11, 5) if right else (-11, -13),
            ha="left" if right else "right", fontsize=8, color=NAVY,
        )

    axis.set(xscale="log", yscale="log", xlabel="sampling time (s)",
             ylabel="W1 of the shared marginal / posterior σ")
    axis.set_title(
        "Making gauss_jacobian cheap: correctors, their interval, and "
        "Jacobian refresh\n(learned score, exponential hierarchy; "
        "dashed lines are Langevin + F-NPSE)",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    axis.text(0.02, 0.95,
              "lower-left is better: cheaper and more accurate",
              transform=axis.transAxes, fontsize=8.5, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="h16d2")
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--observations", type=int, default=compare.OBSERVATIONS)
    parser.add_argument("--num-samples", type=int, default=compare.NUM_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=compare.TIMESTEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--excursion-sigma", type=float, default=15.0)
    parser.add_argument("--kde-bandwidth", default=None)
    parser.add_argument("--map-timesteps", type=int, default=200)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS,
                        help="Where sweep.csv and the figure are written; "
                             "checkpoints are always read from artifacts/.")
    parser.add_argument("--replot", action="store_true")
    arguments = parser.parse_args()

    compare.configure_style()
    output_root = arguments.output_dir
    (output_root / "corrector_efficiency").mkdir(parents=True, exist_ok=True)
    table = output_root / "corrector_efficiency" / "sweep.csv"

    if arguments.replot:
        with table.open(newline="") as handle:
            rows = [
                {key: (value if key in ("label", "rule") else float(value))
                 for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    else:
        problem = compare.build_problem(
            compare.recipe.model_directory(
                ARTIFACTS, arguments.config, arguments.train_samples
            ),
            arguments.config, arguments.train_samples, arguments.observations,
            arguments.seed, arguments.device,
        )
        rows = run(problem, arguments)
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {table}")

    plot(rows, output_root / "04_corrector_efficiency.png")


if __name__ == "__main__":
    main()
