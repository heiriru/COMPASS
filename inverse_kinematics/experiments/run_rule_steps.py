#!/usr/bin/env python3
"""Does a Gaussian composition rule's smearing go away with a finer solver?

`gauss_jacobian` and `gauss_hierarchical` return a broad, near-unimodal blob on
this dataset where the reference is sharply bimodal -- dip ratio 0.63-0.99
against the reference's 0.00, modes pulled in to +-0.15..0.35, and about half the
reference's width. Two explanations are possible and they have different fixes:

1. **Discretisation.** The composed score is integrated on a finite grid in
   log-sigma; too few steps and the sampler never resolves the two basins. This
   is fixed by spending more steps, and the earlier F-NPSE measurement showed
   that a real part of its inverse-kinematics error *was* discretisation
   (0.237 -> 0.146 at n=8 going from 100 to 200 steps).

2. **The rule.** GAUSS summarises each observation's contribution as a mean and
   a covariance. A two-component posterior has no faithful mean-plus-covariance
   summary, so the composed object is a single broad Gaussian-like blob no matter
   how finely the SDE is integrated, and the error is bounded below by the
   family, not the solver.

There is a third knob specific to `gauss_jacobian`, and it is the cheapest one
to suspect: the rule needs the Jacobian of the per-observation score, and
`--jacobian-refresh` recomputes it only every k-th step, reusing a stale one in
between. At the default k = 5 and 40 steps that is eight evaluations for forty
steps. If the smearing is a stale-Jacobian artifact, refreshing every step
(k = 1) must fix it -- and unlike the step count, k costs nothing in solver
resolution, so it is a clean test.

Sweeping separates the three: under (1) the dip ratio falls with steps; under
(3) it falls with refresh frequency at fixed steps; under (2) neither moves.
Saves the density for every setting so the plot shows the shape, not only the
metric.

Usage:
    python run_rule_steps.py --observations 8 32 --steps 40 --refreshes 1 2 5 10 40
    python run_rule_steps.py --observations 8 --steps 10 20 40 100 200 400
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import hierarchical as H  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
from bimodality_study import selected_dataset  # noqa: E402
from composition_attribution import (  # noqa: E402
    DEFAULT_CHECKPOINT, compare, kde_density, moments,
)

OUTPUT = ROOT / "artifacts" / "mixture_inference" / "rule_steps"


def shape(grid, density):
    """Modes, branch mass, sd and dip ratio -- W1 alone flatters a broad blob."""
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    density = density / np.trapezoid(density, grid)
    interior = np.arange(1, len(grid) - 1)
    hit = interior[(density[interior] >= density[interior - 1])
                   & (density[interior] >= density[interior + 1])
                   & (density[interior] > 0.05 * density.max())]
    mean = float(np.trapezoid(grid * density, grid))
    sd = float(np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid)))
    if len(hit) >= 2:
        low, high = hit[0], hit[-1]
        dip = float(density[low:high + 1].min()
                    / min(density[low], density[high]))
    else:
        dip = 1.0
    return {"modes": [float(grid[i]) for i in hit],
            "mass_positive": float(np.trapezoid(density[grid > 0],
                                                grid[grid > 0])),
            "sd": sd, "dip_ratio": dip}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--attribution", type=Path, default=None,
                        help="Directory holding curves.npz; defaults to the "
                             "selected_n<observations>_seed<seed> bundle.")
    result.add_argument("--methods", nargs="+",
                        default=["gauss_jacobian", "gauss_hierarchical"])
    result.add_argument("--observations", type=int, nargs="+", default=[8, 32])
    result.add_argument("--steps", type=int, nargs="+",
                        default=[10, 20, 40, 100, 200, 400])
    result.add_argument("--select-x1", type=float, default=0.15)
    result.add_argument("--true-angle", type=float, default=0.6)
    result.add_argument("--seed", type=int, default=99)
    result.add_argument("--draws", type=int, default=2000)
    result.add_argument("--pilot-draws", type=int, default=1024)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--refreshes", type=int, nargs="+", default=None,
                        help="Jacobian refresh intervals to sweep; 1 means "
                             "recompute every step. Defaults to just "
                             "--jacobian-refresh.")
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--composition-temperature", type=float, default=1.0)
    result.add_argument("--kde-bandwidth", type=float, default=None)
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path, default=OUTPUT)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly: shared host.")

    from compass import ScoreBasedInferenceModel as SBIm
    model = SBIm.load(str(arguments.checkpoint), device=arguments.device)

    arguments.output.mkdir(parents=True, exist_ok=True)
    rows, stored = [], {}
    for n in arguments.observations:
        directory = arguments.attribution or (
            ROOT / "artifacts" / "composition_attribution"
            / f"selected_n{n}_seed{arguments.seed}")
        with np.load(directory / "curves.npz") as bundle:
            grid = bundle["grid"]
            reference = bundle["joint_reference"]
            stored_x = bundle["observations"]
        rng = sim.generator("bimodal-dataset", arguments.seed)
        x = np.asarray(selected_dataset(n, arguments.true_angle,
                                        arguments.select_x1, rng),
                       dtype=np.float64)
        if not np.allclose(x, stored_x):
            raise SystemExit(f"dataset at n={n} does not match its reference")
        data = torch.as_tensor(x, dtype=torch.float32)
        reference_moments = moments(grid, reference)
        stored[f"grid_n{n}"] = grid
        stored[f"reference_n{n}"] = reference
        reference_shape = shape(grid, reference)
        print(f"\n### n = {n}   reference: modes "
              f"{', '.join(f'{v:+.3f}' for v in reference_shape['modes'])}  "
              f"mass>0 {reference_shape['mass_positive']:.3f}  "
              f"sd {reference_shape['sd']:.3f}  "
              f"dip {reference_shape['dip_ratio']:.3f}", flush=True)
        print(f"{'method':<20}{'steps':>7}{'refresh':>9}{'W1/sig':>9}{'sd':>7}"
              f"{'mass>0':>9}{'dip':>7}{'secs':>8}", flush=True)

        refreshes = arguments.refreshes or [arguments.jacobian_refresh]
        for method in arguments.methods:
            for steps in arguments.steps:
                for refresh in refreshes:
                    # gauss_hierarchical never forms a Jacobian, so sweeping the
                    # refresh interval for it would re-run an identical
                    # configuration and invite reading noise as an effect.
                    if refresh != refreshes[0] and method != "gauss_jacobian":
                        continue
                    inference.COMMON["timesteps"] = int(steps)
                    trial = argparse.Namespace(**vars(arguments))
                    trial.timesteps = int(steps)
                    trial.jacobian_refresh = int(refresh)
                    started = time.perf_counter()
                    torch.manual_seed(int(arguments.seed))
                    torch.cuda.manual_seed_all(int(arguments.seed))
                    samples, _, seconds = inference.draw(
                        model, data, method, trial, arguments.device)
                    elapsed = time.perf_counter() - started
                    shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
                    density = kde_density(shared, grid, arguments.kde_bandwidth)
                    measured = shape(grid, density)
                    graded = compare(grid, density, reference,
                                     reference_moments)
                    rows.append({"observations": int(n), "method": method,
                                 "steps": int(steps), "refresh": int(refresh),
                                 "seconds": elapsed, **graded, **measured})
                    stored[f"{method}_n{n}_s{steps}_r{refresh}"] = density
                    print(f"{method:<20}{steps:7d}{refresh:9d}"
                          f"{graded['w1_over_sigma']:9.3f}"
                          f"{measured['sd']:7.3f}"
                          f"{measured['mass_positive']:9.3f}"
                          f"{measured['dip_ratio']:7.3f}{elapsed:8.1f}",
                          flush=True)

    (arguments.output / "metrics.json").write_text(
        json.dumps(rows, indent=1, default=float))
    np.savez_compressed(arguments.output / "curves.npz", **stored)
    print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
