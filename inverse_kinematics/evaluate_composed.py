#!/usr/bin/env python3
"""Quantitative accuracy and calibration of the composed shared posterior.

A single dataset and an eye test establish nothing: one draw can look right for
the wrong reasons, and a bimodal figure is especially easy to over-read. This
script replaces that with two things a picture cannot give.

**Accuracy** -- per replicate, the composed draws against the SBC-validated ABC
reference: W1 in units of the reference's own sd, the error in each mode's
location, and the error in the mass split between branches. Reported as a median
and an interquartile range over replicates, because a mean over a handful of
datasets hides exactly the variability that made the single-dataset sweeps in
``bimodality_study.py`` unreadable.

**Calibration** -- simulation-based calibration of the *composed* posterior. Draw
``angle_1`` from its prior, simulate, infer, and record the rank of the truth
among the composed draws. If the whole pipeline (network + composition rule +
sampler) is a correct posterior sampler, those ranks are Uniform(0, 1). This is
the only measurement here that can fail, and it grades the method end to end
rather than one component.

Selection and calibration: why ``--select-x1`` must be off for SBC
------------------------------------------------------------------
Accuracy and calibration need *different* datasets here, and conflating them
produces a confidently wrong answer.

For the **accuracy** metrics the selection is harmless. Selection is a function
of ``x`` alone, and the sampler is asked to reproduce ``p(a | x)`` for the ``x``
it is given; the reference computes that same ``p(a | x)``. Grading composed
draws against it on selected data is fair.

For **SBC** it is not, and the reason is subtle enough that a first version of
this script got it wrong. SBC needs the truth to be a draw from the same joint
whose conditional the sampler targets. Generating a dataset by drawing
``a ~ prior`` and then *resampling locals until every* ``|x1| < eps`` samples

    p(a) p(x | a) 1[.] / Z(a),    Z(a) = P(|x1| < eps | a)

which is **not** the selected joint ``p(a, x) 1[.] / Z``: it is missing a factor
``Z(a)^n``. And ``Z(a)`` is far from flat -- measured on this problem it runs from
0.0028 to 0.1307 over ``a in [-1.5, 1.5]``, a factor of 47, so at ``n = 8`` the
missing weight spans twelve orders of magnitude. Under that scheme the ranks are
non-uniform *even for a perfect sampler*, and the KS test measures the generator,
not the method.

Sampling the selected joint properly would mean rejecting whole datasets, which
at ``eps = 0.15`` and ``n = 8`` has acceptance ``~1e-8``. So SBC is run **without
selection** (``--select-x1 0``), on plain prior-predictive datasets, where the
generating joint and the sampler's target agree. The posterior is then usually
unimodal, which is fine: calibration is a property of the method, not of the
dataset shape.

Usage:
    python inverse_kinematics/evaluate_composed.py --replicates 30
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402
from bimodality_study import bimodality, selected_dataset  # noqa: E402
from sbc_reference import kolmogorov_smirnov  # noqa: E402


def rank_of(truth, draws):
    """Posterior rank of the truth among composed draws: P(a < a_true | x)."""
    draws = np.asarray(draws, dtype=np.float64)
    return float(np.mean(draws < float(truth)))


def mode_errors(grid, reference_density, draws):
    """Mode-location and mass-split error of ``draws`` against the reference."""
    from scipy.stats import gaussian_kde

    density = gaussian_kde(np.asarray(draws, dtype=np.float64))(grid)
    density /= np.trapezoid(density, grid)
    got, want = bimodality(grid, density), bimodality(grid, reference_density)
    if len(got["modes"]) < 2 or len(want["modes"]) < 2:
        return {"mode_error": float("nan"), "mass_error": float("nan"),
                "dip_ratio": got["dip_ratio"], "reference_dip": want["dip_ratio"],
                "both_bimodal": False}
    return {
        "mode_error": float(np.max(np.abs(
            np.sort(got["modes"]) - np.sort(want["modes"])
        ))),
        "mass_error": float(abs(got["mode_mass_ratio"] - want["mode_mass_ratio"])),
        "dip_ratio": got["dip_ratio"], "reference_dip": want["dip_ratio"],
        "both_bimodal": True,
    }


def summarize(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(values):
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"),
                "n": 0}
    return {"median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)), "n": int(len(values))}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="h16d2")
    result.add_argument("--arch", default="legacy")
    result.add_argument("--menu", action="store_true", default=True)
    result.add_argument("--no-menu", dest="menu", action="store_false")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--replicates", type=int, default=30)
    result.add_argument("--observations", type=int, default=8)
    result.add_argument("--select-x1", type=float, default=0.15)
    result.add_argument("--draws", type=int, default=1_500)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--pilot-draws", type=int, default=1_000)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--composition-temperature", type=float, default=1.0)
    result.add_argument("--reference-draws", type=int, default=60_000)
    result.add_argument("--reference-bandwidth", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--methods", nargs="+", default=list(inference.METHODS))
    result.add_argument("--device", default="cuda")
    result.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    directory = trainer.model_directory(
        arguments.artifacts, arguments.config, arguments.arch,
        arguments.simulations, menu=arguments.menu,
    )
    model = trainer.load_to(directory / "Model_checkpoint.pt", arguments.device)
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"{arguments.arch}/{arguments.config}"
          f"{' +menu' if arguments.menu else ''}  {parameters:,} parameters")
    mode = (f"|x1| < {arguments.select_x1} (accuracy only; SBC INVALID here)"
            if arguments.select_x1 > 0 else
            "plain prior predictive (SBC valid)")
    print(f"{arguments.replicates} replicates x {arguments.observations} "
          f"observations, {mode}\n", flush=True)

    grid = np.linspace(-2.0, 2.0, 801)
    rows = {method: [] for method in arguments.methods}
    started = time.perf_counter()

    for replicate in range(arguments.replicates):
        rng = sim.generator(f"eval-{replicate}", arguments.seed)
        local_truth = None
        if arguments.select_x1 > 0:
            truth = float(rng.normal(0.0, H.GLOBAL_PRIOR_STD[0]))
            x = selected_dataset(arguments.observations, truth,
                                 arguments.select_x1, rng)
        else:
            # Plain prior predictive: the generating joint is the one the
            # sampler targets, so ranks are interpretable.
            truth, local_truth, x = H.simulate_dataset(
                arguments.observations, rng)
        _, density = H.reference_global(
            x, grid=grid, draws=arguments.reference_draws,
            bandwidth=arguments.reference_bandwidth, seed=arguments.seed + replicate,
        )
        _, reference_std, _ = H.grid_moments(grid, density)
        data = torch.as_tensor(x, dtype=torch.float32)

        for method in arguments.methods:
            samples, _, _ = inference.draw(
                model, data, method, arguments, arguments.device
            )
            shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
            row = {
                "truth": truth,
                "rank": rank_of(truth, shared),
                "w1_over_sigma": inference.wasserstein_to_grid(
                    shared, grid, density) / reference_std,
                **mode_errors(grid, density, shared),
            }
            # Local ranks: one per observation per local coordinate. They need
            # no reference -- each is the rank of a known truth among that
            # observation's own posterior draws -- so they are available
            # whenever the truth is a prior draw, and they are the only check
            # here on the local block, which every rule also corrects.
            if local_truth is not None:
                row["local_ranks"] = {
                    name: [rank_of(local_truth[j, k], samples[j, :, node])
                           for j in range(samples.shape[0])]
                    for k, (name, node) in enumerate(
                        zip(H.LOCAL_NAMES, H.LOCAL_INDICES))
                }
            rows[method].append(row)
        if (replicate + 1) % 5 == 0:
            print(f"  {replicate + 1}/{arguments.replicates} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    print(f"\n{'rule':>22}{'W1/sigma':>20}{'mode err':>18}{'mass err':>18}"
          f"{'cov50':>8}{'cov90':>8}{'KS':>8}{'uniform?':>10}")
    summary = {}
    critical = 1.358 / np.sqrt(arguments.replicates)
    for method in arguments.methods:
        data = rows[method]
        ranks = np.array([r["rank"] for r in data])
        w1 = summarize([r["w1_over_sigma"] for r in data])
        mode = summarize([r["mode_error"] for r in data])
        mass = summarize([r["mass_error"] for r in data])
        # Equal-tailed credible-interval coverage, read off the ranks.
        cov50 = float(np.mean(np.abs(ranks - 0.5) < 0.25))
        cov90 = float(np.mean(np.abs(ranks - 0.5) < 0.45))
        ks = kolmogorov_smirnov(ranks)
        local_ranks = {}
        for name in H.LOCAL_NAMES:
            values = [v for r in data for v in r.get("local_ranks", {}).get(name, [])]
            if values:
                values = np.asarray(values)
                # KS is evaluated against the number of *independent* datasets,
                # not the number of ranks: the locals inside one dataset share a
                # composed global draw, so they are not independent samples.
                critical_local = 1.358 / np.sqrt(len(data))
                local_ranks[name] = {
                    "mean_rank": float(values.mean()),
                    "coverage_50": float(np.mean(np.abs(values - 0.5) < 0.25)),
                    "coverage_90": float(np.mean(np.abs(values - 0.5) < 0.45)),
                    "ks_distance": kolmogorov_smirnov(values),
                    "ks_critical_95": float(critical_local),
                    "count": int(len(values)),
                    "ranks": values.tolist(),
                }
        summary[method] = {
            "local": local_ranks,
            "w1_over_sigma": w1, "mode_error": mode, "mass_error": mass,
            "coverage_50": cov50, "coverage_90": cov90,
            "ks_distance": ks, "ks_critical_95": float(critical),
            "uniform": bool(ks < critical), "mean_rank": float(ranks.mean()),
            "bimodal_fraction": float(np.mean([r["both_bimodal"] for r in data])),
            "ranks": ranks.tolist(),
        }
        print(f"{method:>22}"
              f"{w1['median']:>9.3f} [{w1['q25']:.2f},{w1['q75']:.2f}]"
              f"{mode['median']:>9.3f} [{mode['q25']:.2f},{mode['q75']:.2f}]"
              f"{mass['median']:>9.3f} [{mass['q25']:.2f},{mass['q75']:.2f}]"
              f"{cov50:>8.2f}{cov90:>8.2f}{ks:>8.3f}"
              f"{('yes' if ks < critical else 'NO'):>10}")
    print(f"\nNominal coverage 0.50 / 0.90; KS 95% critical {critical:.3f} "
          f"at {arguments.replicates} replicates.")

    output = arguments.artifacts / "bimodal" / "composed_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "arch": arguments.arch, "config": arguments.config,
        "menu": bool(arguments.menu), "parameters": int(parameters),
        "replicates": int(arguments.replicates),
        "observations": int(arguments.observations),
        "select_x1": arguments.select_x1, "draws": int(arguments.draws),
        "summary": summary,
    }, indent=2))
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
