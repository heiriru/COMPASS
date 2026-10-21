#!/usr/bin/env python3
"""Does Arruda-style score damping survive a growing observation count?

Motivation
----------
On the partial-pooling benchmark, moving from 20 to 100 subjects made the
*composed global* parameters worse, not better: ``beta_raw`` and ``mu_nu``
degraded at every DPM arm while the locals improved, and ``langevin_fnpse`` --
the one arm that never composes through ``_dpm_sampler`` -- was the only arm to
improve. That is the error-accumulation regime Arruda et al. (2026) describe,
and their remedy is a time-dependent damping factor ``d(t)`` on the composed
score.

Before touching the benchmark, grade the remedy where the answer is known. This
hierarchy is exactly solvable at every noise level, so the composed shared
posterior can be compared against the true one rather than against another
approximation, and the comparison can be repeated as the number of observations
grows -- which is the axis the benchmark failed on.

The three noise laws
--------------------
``hierarchy_gauss``   Normal: symmetric, no wall, and Gaussian, so every
                      composition rule is exact by construction. Damping can
                      only *hurt* here -- this is the control that says whether
                      the fix destroys a posterior that was already right.
``hierarchy_laplace`` Laplace: symmetric, no wall, non-Gaussian.
``hierarchy``         Exponential: skewed with a hard wall at ``min_j x_j``.

Scores are the exact per-row analytic scores (``symmetric_row_scores``), not a
trained network, so nothing here is contaminated by network error: what is
measured is the composition rule and the damping schedule alone.

The schedules
-------------
``MultiObsSampler`` already implements ``correction="damping"`` as
``d(t) * [(1-n)(1-t) prior_score + n/m sum_j score_j]`` with
``d(t) = at_data * exp(log(at_noise/at_data) * t)``, where ``t = 0`` is data and
``t = 1`` is noise. Its validation requires ``at_noise <= at_data``, i.e. it can
only damp harder toward the *noise* end. BayesFlow's ``compositional_bridge``
(``d0 = 0.1``, ``d1 = 1.0``, used by the paper's own case-study code) damps
harder toward the *data* end -- the opposite direction. Both are tested, the
reversed one by patching ``_damping_factor`` for the duration of a run, since
the built-in endpoints cannot express it.

    sum         no damping at all (prior-corrected tall sum), the baseline
    damp_off    correction="damping" with d = 1: isolates the (1-t) prior
                weighting from the damping factor itself
    damp_noise  d(0) = 1, d(1) = n^-0.5     COMPASS default direction
    damp_data   d(0) = 0.1, d(1) = 1        Arruda / BayesFlow direction
    damp_const  d = 1/n                     the literal "1/number of subjects"

Nothing in ``compass``, ``compare.py``, ``hierarchy*.py`` or
``symmetric_experiments.py`` is modified: the problem module is swapped into
``compare``'s namespace and the damping schedule onto ``MultiObsSampler`` for
the duration of one run, and both are restored afterwards.

Usage:
    python damping_observation_sweep.py
    python damping_observation_sweep.py --problem gauss --observations 2 10 50
    python damping_observation_sweep.py --replot
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
import contextlib  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[3]))

import numpy as np  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CSV_PATH = ARTIFACTS / "damping_observation_sweep.csv"

OBSERVATION_LADDER = (2, 5, 10, 20, 50, 100, 200)

DPM = {
    "method": "dpm", "order": 2,
    "corrector_steps": 0, "final_corrector_steps": 0,
    "terminal_corrector_steps": 0,
}
# The damping rule inherits F-NPSE's (1-t) prior weighting, and MultiObsSampler
# documents that family as consistent only with annealed Langevin sampling --
# which is also what Arruda et al. actually run. So every damping schedule is
# graded in both samplers: a failure under DPM alone would only restate the
# known predictor/target mismatch, not judge the damping.
LANGEVIN = {"method": "langevin", "corrector_steps": 10, "snr": 0.2}

# schedule: None uses the sampler's own endpoints; a callable takes the
# observation count and returns d(t).
VARIANTS = {
    "sum": {
        "label": "DPM2, no damping (prior-corrected sum)",
        "correction": "prior_corrected_sum", "base": DPM,
        "kwargs": {}, "schedule": None, "colour": "#8A8F98", "family": "dpm",
    },
    "damp_off": {
        "label": "DPM2, damping rule with d = 1 (isolates the (1-t) prior weight)",
        "correction": "damping", "base": DPM,
        "kwargs": {"damping_at_data": 1.0, "damping_at_noise": 1.0},
        "schedule": None, "colour": "#B8A1DC", "family": "dpm",
    },
    "damp_noise": {
        "label": "DPM2, d(0)=1 -> d(1)=n^-1/2  (COMPASS default)",
        "correction": "damping", "base": DPM,
        "kwargs": {"damping_at_data": 1.0, "damping_at_noise": None},
        "schedule": None, "colour": "#0F8B8D", "family": "dpm",
    },
    "damp_data": {
        "label": "DPM2, d(0)=0.1 -> d(1)=1  (Arruda / BayesFlow bridge)",
        "correction": "damping", "base": DPM,
        "kwargs": {"damping_at_data": 1.0, "damping_at_noise": 1.0},
        "schedule": lambda n: (lambda t: 0.1 ** (1.0 - t)),
        "colour": "#B0453A", "family": "dpm",
    },
    "damp_const": {
        "label": "DPM2, d = 1/n (constant)",
        "correction": "damping", "base": DPM,
        "kwargs": {"damping_at_data": 1.0, "damping_at_noise": 1.0},
        "schedule": lambda n: (lambda t: 1.0 / n),
        "colour": "#9C6B1F", "family": "dpm",
    },
    "lang_fnpe": {
        "label": "Langevin + F-NPSE, no damping (the reference)",
        "correction": "fnpe", "base": LANGEVIN,
        "kwargs": {}, "schedule": None, "colour": "#4A4A4A", "family": "langevin",
    },
    "lang_damp_noise": {
        "label": "Langevin, d(0)=1 -> d(1)=n^-1/2  (COMPASS default)",
        "correction": "damping", "base": LANGEVIN,
        "kwargs": {"damping_at_data": 1.0, "damping_at_noise": None},
        "schedule": None, "colour": "#0F8B8D", "family": "langevin",
    },
    "lang_damp_data": {
        "label": "Langevin, d(0)=0.1 -> d(1)=1  (Arruda / BayesFlow bridge)",
        "correction": "damping", "base": LANGEVIN,
        "kwargs": {"damping_at_data": 1.0, "damping_at_noise": 1.0},
        "schedule": lambda n: (lambda t: 0.1 ** (1.0 - t)),
        "colour": "#B0453A", "family": "langevin",
    },
    "lang_damp_const": {
        "label": "Langevin, d = 1/n (constant)",
        "correction": "damping", "base": LANGEVIN,
        "kwargs": {"damping_at_data": 1.0, "damping_at_noise": 1.0},
        "schedule": lambda n: (lambda t: 1.0 / n),
        "colour": "#9C6B1F", "family": "langevin",
    },
}

FIELDS = (
    "problem", "variant", "family", "observations", "correction", "status",
    "shared_w1_over_sigma", "shared_mean_error_sigma", "shared_width_ratio",
    "exact_mean", "exact_std", "sample_mean", "sample_std",
    "num_samples", "seconds",
)


@contextlib.contextmanager
def patched_damping(sampler_class, schedule):
    """Replace ``_damping_factor`` with an arbitrary d(t) for one run."""
    if schedule is None:
        yield
        return
    import torch

    original = sampler_class._damping_factor

    def factor(self, t):
        value = schedule(float(np.asarray(torch.as_tensor(t).detach().cpu()).reshape(-1)[0]))
        return torch.as_tensor(value, dtype=torch.as_tensor(t).dtype,
                               device=torch.as_tensor(t).device)

    sampler_class._damping_factor = factor
    try:
        yield
    finally:
        sampler_class._damping_factor = original


def shared_metrics(draws, grid, density, exact_mean, exact_std,
                   probabilities=4000):
    """W1 against the exact shared posterior, plus location and width errors.

    W1 is computed as the mean absolute difference of the two quantile
    functions, which is exactly the L1 distance between the CDFs, and is
    reported in units of the exact posterior's own standard deviation so the
    numbers stay comparable across observation counts (the posterior narrows
    like 1/n).
    """
    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]
    keep = np.concatenate(([True], np.diff(cdf) > 0))
    probability = np.linspace(0.5 / probabilities, 1.0 - 0.5 / probabilities,
                              probabilities)
    exact_quantiles = np.interp(probability, cdf[keep], grid[keep])
    sample_quantiles = np.quantile(draws, probability)
    w1 = float(np.mean(np.abs(exact_quantiles - sample_quantiles)))
    return {
        "shared_w1_over_sigma": w1 / exact_std,
        "shared_mean_error_sigma": float(draws.mean() - exact_mean) / exact_std,
        "shared_width_ratio": float(draws.std(ddof=1)) / exact_std,
        "sample_mean": float(draws.mean()),
        "sample_std": float(draws.std(ddof=1)),
    }


def run_cell(problem_name, specification, variant_name, entry, count, arguments):
    import torch
    import compare
    import symmetric_row_scores as row_scores
    from symmetric_experiments import patched_problem_module
    from compass.MultiObsSampler import MultiObsSampler
    from compass.SDE import VESDE

    module = specification["module"]
    sde = VESDE(sigma=compare.recipe.SDE_KWARGS["sigma"])
    model = row_scores.AnalyticModel(
        sde, specification["row_score"], module, device=arguments.device,
        nodes=arguments.quadrature_nodes, seed=arguments.seed,
    )
    _, _, x = module.observations(count, arguments.seed)
    grid, density, _, exact_mean, exact_std = module.shared_reference(x)

    problem = {
        "data": torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1),
        "device": arguments.device,
    }
    sample_kwargs = dict(entry["base"])
    sample_kwargs["correction"] = entry["correction"]
    sample_kwargs.update(entry["kwargs"])

    schedule = entry["schedule"](count) if entry["schedule"] else None
    row = {
        "problem": problem_name, "variant": variant_name,
        "family": entry["family"], "observations": count,
        "correction": entry["correction"],
        "exact_mean": float(exact_mean), "exact_std": float(exact_std),
    }
    started = time.perf_counter()
    try:
        with patched_problem_module(module), patched_damping(MultiObsSampler, schedule):
            samples, _, _ = compare.draw(
                model, problem, sample_kwargs, arguments.num_samples,
                arguments.timesteps, arguments.seed, arguments.denoise_clamp,
            )
    except (RuntimeError, ValueError, AssertionError) as error:
        # A rule that diverges is an outcome, not a crash: record it and let the
        # rest of the ladder run. NaNs and the sampler's own damping guard both
        # land here, and both mean "this schedule failed at this n".
        row.update({
            "status": f"FAILED: {type(error).__name__}: {error}".replace("\n", " ")[:220],
            "seconds": round(time.perf_counter() - started, 2),
            "num_samples": 0,
        })
        return row
    seconds = time.perf_counter() - started

    shared = samples[0, :, module.HIERARCHY[0]].numpy()
    row.update({
        "status": "ok", "num_samples": int(shared.size),
        "seconds": round(seconds, 2),
    })
    row.update(shared_metrics(shared, grid, density, exact_mean, exact_std))
    return row


def load_rows(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def plot_summary(rows, problems, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import compare

    compare.configure_style()
    panels = ("shared_w1_over_sigma", "shared_width_ratio")
    titles = ("Shared posterior error  W1 / sigma",
              "Shared posterior width ratio  (1.0 = exact)")
    figure, axes = plt.subplots(
        len(panels), len(problems), figsize=(4.6 * len(problems), 7.4),
        squeeze=False, constrained_layout=True,
    )
    for column, problem_name in enumerate(problems):
        for panel_index, (field, title) in enumerate(zip(panels, titles)):
            axis = axes[panel_index][column]
            for variant_name, entry in VARIANTS.items():
                selected = sorted(
                    (row for row in rows
                     if row["problem"] == problem_name
                     and row["variant"] == variant_name
                     and row.get("status") == "ok"),
                    key=lambda row: int(row["observations"]),
                )
                if not selected:
                    continue
                axis.plot(
                    [int(row["observations"]) for row in selected],
                    [float(row[field]) for row in selected],
                    marker="o" if entry["family"] == "dpm" else "s",
                    ms=4, lw=1.7, color=entry["colour"],
                    ls="-" if entry["family"] == "dpm" else "--",
                    label=entry["label"] if panel_index == 0 and column == 0 else None,
                )
            axis.set_xscale("log")
            if field == "shared_w1_over_sigma":
                axis.set_yscale("log")
            else:
                axis.axhline(1.0, color="black", lw=1, ls="--")
            axis.set_xlabel("observations n")
            if column == 0:
                axis.set_ylabel(title)
            if panel_index == 0:
                axis.set_title(problem_name, fontsize=11)
            axis.grid(alpha=0.25)
    # "outside lower center" reserves its own strip under the axes; a plain
    # "lower center" with a negative bbox lands on top of the bottom row's
    # x-axis labels under constrained_layout.
    figure.legend(loc="outside lower center", ncol=2, frameon=True)
    figure.suptitle(
        "Score damping vs observation count, exact per-row scores, "
        "corrector-free DPM2 (eta = 0)", fontsize=12,
    )
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", nargs="+",
                        default=["gauss", "laplace", "exponential"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--observations", nargs="+", type=int,
                        default=list(OBSERVATION_LADDER))
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--quadrature-nodes", type=int, default=769)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Recompute cells already present in the CSV.")
    arguments = parser.parse_args()

    if not arguments.replot and arguments.device.startswith("cuda"):
        # Shared host: wait for a genuinely idle GPU rather than crowding one.
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=5)
        arguments.device = "cuda"

    from symmetric_experiments import PROBLEMS

    rows = load_rows(arguments.csv)
    done = {(row["problem"], row["variant"], int(row["observations"]))
            for row in rows}

    if not arguments.replot:
        for problem_name in arguments.problem:
            specification = PROBLEMS[problem_name]
            for count in arguments.observations:
                for variant_name in arguments.variants:
                    key = (problem_name, variant_name, count)
                    if key in done and not arguments.force:
                        continue
                    entry = VARIANTS[variant_name]
                    row = run_cell(problem_name, specification, variant_name,
                                   entry, count, arguments)
                    rows = [existing for existing in rows
                            if (existing["problem"], existing["variant"],
                                int(existing["observations"])) != key]
                    rows.append(row)
                    write_rows(arguments.csv, rows)
                    if row["status"] == "ok":
                        message = (
                            f"W1/sigma {row['shared_w1_over_sigma']:8.4f}  "
                            f"width {row['shared_width_ratio']:6.3f}  "
                            f"mean err {row['shared_mean_error_sigma']:+7.3f}"
                        )
                    else:
                        message = row["status"]
                    print(
                        f"{problem_name:12s} n={count:4d} "
                        f"{variant_name:16s} {message}  {row['seconds']:.1f}s",
                        flush=True,
                    )

    rows = load_rows(arguments.csv)
    output = plot_summary(
        rows, [name for name in arguments.problem], ARTIFACTS / "08_damping_observation_sweep.png"
    )
    print(f"\nWrote {arguments.csv}\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
