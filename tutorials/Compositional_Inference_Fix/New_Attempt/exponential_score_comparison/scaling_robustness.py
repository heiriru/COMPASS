#!/usr/bin/env python3
"""Making the composed shared posterior *improve* with n instead of degrading.

The diagnosis this starts from
------------------------------
``damping_observation_sweep.py`` showed that every composition rule degrades as
the observation count grows, and that Arruda-style damping does not fix it. But
its numbers also say *how* the rules fail, and that is the useful part: from
n = 20 upward, ``W1/sigma`` and ``|mean error|/sigma`` are equal to two decimals
while the width ratio stays near 0.7-0.9. The composed posterior has very
nearly the right shape in the wrong place. It is a **location** failure, not a
shape failure, so the Gaussian-composition approximation is not the first thing
to blame.

The obvious reading of that signature is initialization and discretization: the
tall posterior narrows like ``1/n``, ``MultiObsSampler._hierarchy_scale``
shrinks the initial noise on the shared coordinates by ``1/n`` **only** for
``correction="fnpe"`` (Geffner et al. 2023's bridging reference
``N(0, sigma_max^2/n)``), and F-NPSE is the one rule that stays stable. So the
``gauss_*`` rules would be starting n times too wide and running out of steps.

**That reading is wrong, and the pilot at gauss n = 100 killed it outright:**

    sum, as shipped                W1/sigma  3.6258
    sum, shared initial scale n^-1           3.5805     no effect
    sum, 4x steps                            3.6261     no effect
    sum, 16x steps                           3.6261     no effect
    sum, churn eta = 4                       0.4478     8.1x better

Sixteen times the step budget moves the answer in the fourth decimal, and every
initial scale is within 1%. It is not discretization and not initialization:
the deterministic trajectory converges perfectly well to the wrong answer. What
fixes it is **stochasticity** -- churn restores both the location (mean error
-3.63 -> -0.45 sigma) and the width (0.773 -> 0.942).

That reframes the question for each sampler family, and the grid below is
targeted accordingly rather than spending cells re-measuring a null result.

DPM2 (``sum``, ``gauss_jacobian``, ``gauss_hierarchical``)
    Only churn moves the needle, so sweep ``eta`` in {2, 4, 8, 16} against n.
    The question is whether the optimal ``eta`` grows with n, and whether the
    churned error can be made *flat or decreasing* in n rather than merely
    smaller.

Langevin + F-NPSE
    Two separate levers, both real in the pilot. Its built-in ``1/n`` shared
    scale is **too aggressive** -- the tall posterior's standard deviation
    shrinks like ``n^-1/2``, not ``n^-1``, and replacing the exponent halved the
    error (1.0515 -> 0.5121) at n = 100. Independently, more mixing per level
    helped 2.8x (1.0515 -> 0.3746). So sweep the exponent and the corrector
    budget, and their combination.

Nothing in ``compass``, ``compare.py``, ``hierarchy*.py`` or
``damping_observation_sweep.py`` is modified: ``_hierarchy_scale`` and
``_dpm_sampler`` are patched for the duration of one run and restored.

Usage:
    python scaling_robustness.py --problem gauss --observations 20 100
    python scaling_robustness.py                       # the full grid
    python scaling_robustness.py --replot
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[3]))

import numpy as np  # noqa: E402

from damping_observation_sweep import (  # noqa: E402
    load_rows, shared_metrics, write_rows as _write_rows,
)

ARTIFACTS = ROOT / "artifacts"
CSV_PATH = ARTIFACTS / "scaling_robustness.csv"

DPM = {
    "method": "dpm", "order": 2,
    "corrector_steps": 0, "final_corrector_steps": 0,
    "terminal_corrector_steps": 0,
}
LANGEVIN = {"method": "langevin", "corrector_steps": 10, "snr": 0.2}

RULES = {
    "sum": {"base": DPM, "correction": "prior_corrected_sum",
            "label": "DPM2 + prior-corrected sum", "colour": "#8A8F98"},
    "jac": {"base": DPM, "correction": "gauss_jacobian",
            "label": "DPM2 + gauss_jacobian", "colour": "#0F8B8D"},
    "hier": {"base": DPM, "correction": "gauss_hierarchical",
             "label": "DPM2 + gauss_hierarchical", "colour": "#B0453A",
             "extra": {"precision_est_samples": 4096,
                       "precision_est_timesteps": 100}},
    "lang": {"base": LANGEVIN, "correction": "fnpe",
             "label": "Langevin + F-NPSE", "colour": "#4A4A4A"},
}

# init_scale: exponent p in scale = n**-p applied to the shared coordinates.
# step_factor: multiplier on the step budget. eta: churn. mix: corrector steps.
INTERVENTIONS = {
    "base":        {"label": "as shipped"},
    "init_sqrtn":  {"label": "shared scale n^-1/2", "init_scale": 0.5},
    "init_qn":     {"label": "shared scale n^-1/4", "init_scale": 0.25},
    "init_n":      {"label": "shared scale n^-1", "init_scale": 1.0},
    "steps4":      {"label": "4x steps", "step_factor": 4},
    "churn2":      {"label": "churn eta=2", "eta": 2.0},
    "churn4":      {"label": "churn eta=4", "eta": 4.0},
    "churn8":      {"label": "churn eta=8", "eta": 8.0},
    "churn16":     {"label": "churn eta=16", "eta": 16.0},
    "mix4":        {"label": "4x Langevin corrector steps", "mix": 4},
    "sqrtn_mix4":  {"label": "n^-1/2 + 4x corrector steps",
                    "init_scale": 0.5, "mix": 4},
}

# Which interventions make sense for which rule. Churn patches the DPM
# predictor, so it is inert for Langevin; `mix` is a Langevin-only knob.
APPLICABLE = {
    # steps/init are dropped for the DPM rules: the pilot showed 16x steps and
    # every initial scale change the answer in the 4th decimal at n = 100, so
    # spending the grid on them would only re-measure a null result.
    "sum":  ("base", "churn2", "churn4", "churn8", "churn16"),
    "jac":  ("base", "churn2", "churn4", "churn8", "churn16"),
    "hier": ("base", "churn2", "churn4", "churn8", "churn16"),
    "lang": ("base", "init_sqrtn", "init_qn", "steps4", "mix4", "sqrtn_mix4"),
}

FIELDS = (
    "problem", "rule", "intervention", "observations", "correction", "status",
    "shared_w1_over_sigma", "shared_mean_error_sigma", "shared_width_ratio",
    "exact_mean", "exact_std", "sample_mean", "sample_std",
    "timesteps", "num_samples", "seconds",
)


@contextlib.contextmanager
def patched_hierarchy_scale(sampler_class, exponent):
    """Give any composition rule F-NPSE's n-aware initial scale.

    ``_hierarchy_scale`` sets the per-feature scale for the initial noise and
    the Langevin step sizes. Upstream it returns ``1/n`` on the shared
    coordinates for ``fnpe`` and 1 everywhere else; here the same shrink is
    applied to whichever rule is running, with a tunable exponent.
    """
    if exponent is None:
        yield
        return
    import torch

    original = sampler_class._hierarchy_scale

    def scale(self, x):
        value = torch.ones(x.shape[2], device=x.device)
        value[self.hierarchy] = float(self.num_observations) ** -float(exponent)
        return value

    sampler_class._hierarchy_scale = scale
    try:
        yield
    finally:
        sampler_class._hierarchy_scale = original


def run_cell(problem_name, specification, rule_name, rule,
             intervention_name, intervention, count, arguments):
    import torch
    import compare
    import symmetric_row_scores as row_scores
    import stochastic_predictor_experiments as churn
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

    sample_kwargs = dict(rule["base"])
    sample_kwargs["correction"] = rule["correction"]
    sample_kwargs.update(rule.get("extra", {}))
    if "mix" in intervention:
        sample_kwargs["corrector_steps"] = (
            sample_kwargs.get("corrector_steps", 10) * intervention["mix"]
        )
    timesteps = arguments.timesteps * intervention.get("step_factor", 1)

    row = {
        "problem": problem_name, "rule": rule_name,
        "intervention": intervention_name, "observations": count,
        "correction": rule["correction"], "timesteps": timesteps,
        "exact_mean": float(exact_mean), "exact_std": float(exact_std),
    }
    started = time.perf_counter()
    try:
        stack = contextlib.ExitStack()
        with stack:
            stack.enter_context(patched_problem_module(module))
            stack.enter_context(patched_hierarchy_scale(
                MultiObsSampler, intervention.get("init_scale")))
            if intervention.get("eta"):
                stack.enter_context(churn.patched_dpm_sampler(
                    churn.churned_dpm_sampler(intervention["eta"])))
            samples, _, _ = compare.draw(
                model, problem, sample_kwargs, arguments.num_samples,
                timesteps, arguments.seed, arguments.denoise_clamp,
            )
    except (RuntimeError, ValueError, AssertionError) as error:
        row.update({
            "status": f"FAILED: {type(error).__name__}: {error}".replace("\n", " ")[:200],
            "seconds": round(time.perf_counter() - started, 2), "num_samples": 0,
        })
        return row
    seconds = time.perf_counter() - started

    shared = samples[0, :, module.HIERARCHY[0]].numpy()
    row.update({"status": "ok", "num_samples": int(shared.size),
                "seconds": round(seconds, 2)})
    row.update(shared_metrics(shared, grid, density, exact_mean, exact_std))
    return row


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def plot_summary(rows, problems, observations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import compare

    compare.configure_style()
    names = [name for name in INTERVENTIONS]
    figure, axes = plt.subplots(
        len(problems), len(RULES),
        figsize=(4.0 * len(RULES), 3.4 * len(problems)),
        squeeze=False, constrained_layout=True,
    )
    for row_index, problem_name in enumerate(problems):
        for column, (rule_name, rule) in enumerate(RULES.items()):
            axis = axes[row_index][column]
            for intervention_name in names:
                if intervention_name not in APPLICABLE[rule_name]:
                    continue
                selected = sorted(
                    (row for row in rows
                     if row["problem"] == problem_name
                     and row["rule"] == rule_name
                     and row["intervention"] == intervention_name
                     and row.get("status") == "ok"),
                    key=lambda row: int(row["observations"]),
                )
                if not selected:
                    continue
                axis.plot(
                    [int(row["observations"]) for row in selected],
                    [float(row["shared_w1_over_sigma"]) for row in selected],
                    marker="o", ms=4, lw=1.6,
                    ls="-" if intervention_name == "base" else "--",
                    color="black" if intervention_name == "base" else None,
                    label=(INTERVENTIONS[intervention_name]["label"]
                           if row_index == 0 else None),
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.set_xlabel("observations n")
            if column == 0:
                axis.set_ylabel(f"{problem_name}\nW1 / sigma")
            if row_index == 0:
                axis.set_title(rule["label"], fontsize=10)
            axis.legend(fontsize=6.5, loc="upper left")
    figure.suptitle(
        "Making the composed shared posterior robust to n "
        "(exact per-row scores; lower is better)", fontsize=12,
    )
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", nargs="+", default=["gauss", "exponential"])
    parser.add_argument("--rules", nargs="+", default=list(RULES))
    parser.add_argument("--interventions", nargs="+", default=list(INTERVENTIONS))
    parser.add_argument("--observations", nargs="+", type=int,
                        default=[10, 20, 50, 100, 200])
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--quadrature-nodes", type=int, default=769)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    if not arguments.replot and arguments.device.startswith("cuda"):
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=5)
        arguments.device = "cuda"

    from symmetric_experiments import PROBLEMS

    rows = load_rows(arguments.csv)
    done = {(row["problem"], row["rule"], row["intervention"],
             int(row["observations"])) for row in rows}

    if not arguments.replot:
        for problem_name in arguments.problem:
            specification = PROBLEMS[problem_name]
            for count in arguments.observations:
                for rule_name in arguments.rules:
                    for intervention_name in arguments.interventions:
                        if intervention_name not in APPLICABLE[rule_name]:
                            continue
                        key = (problem_name, rule_name, intervention_name, count)
                        if key in done and not arguments.force:
                            continue
                        row = run_cell(
                            problem_name, specification, rule_name,
                            RULES[rule_name], intervention_name,
                            INTERVENTIONS[intervention_name], count, arguments,
                        )
                        rows = [existing for existing in rows
                                if (existing["problem"], existing["rule"],
                                    existing["intervention"],
                                    int(existing["observations"])) != key]
                        rows.append(row)
                        write_rows(arguments.csv, rows)
                        if row["status"] == "ok":
                            message = (
                                f"W1/sigma {row['shared_w1_over_sigma']:8.4f}  "
                                f"width {row['shared_width_ratio']:6.3f}  "
                                f"mean err {row['shared_mean_error_sigma']:+8.3f}"
                            )
                        else:
                            message = row["status"]
                        print(
                            f"{problem_name:12s} n={count:4d} {rule_name:5s} "
                            f"{intervention_name:14s} {message}  "
                            f"{row['seconds']:.1f}s", flush=True,
                        )

    rows = load_rows(arguments.csv)
    output = plot_summary(rows, arguments.problem, arguments.observations,
                          ARTIFACTS / "09_scaling_robustness.png")
    print(f"\nWrote {arguments.csv}\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
