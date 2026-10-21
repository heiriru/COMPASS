#!/usr/bin/env python3
"""Grade the exact marginal product against the score-composition rules.

Reuses the ABC reference already computed by ``composition_attribution.py`` --
that reference is most of the cost, and recomputing it would only add Monte
Carlo noise between the two tables.

Two things are graded, because a composition rule has to deliver both:

* the **shared** posterior ``p(a | x_1..n)`` against the ABC reference, by
  W1/sigma, mean error, width and the mass on the positive branch;
* the **locals**, by posterior-predictive endpoint error -- the distance from a
  drawn arm's tip to the observation it was conditioned on. This is
  reference-free (the forward map is deterministic), and it is the check that
  stops a method from winning on the shared coordinate by returning locals that
  do not reproduce the data.

Usage:
    python inverse_kinematics/experiments/evaluate_marginal_product.py \
        --attribution inverse_kinematics/artifacts/composition_attribution/selected_n8_seed99
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402
import marginal_product as mp  # noqa: E402
from bimodality_study import bimodality, exact_zero_dataset, selected_dataset  # noqa: E402
from composition_attribution import (compare, kde_density, moments,  # noqa: E402
                                     prior_density)


def dataset_for(settings):
    rng = sim.generator("bimodal-dataset", int(settings["seed"]))
    count = int(settings["observations"])
    angle = float(settings["true_angle"])
    kind = settings.get("dataset", "selected")
    if kind == "exact_zero":
        return exact_zero_dataset(count, angle, rng)[0]
    if kind == "prior":
        return H.simulate_dataset(count, rng, angle_1=angle)[2]
    return selected_dataset(count, angle, float(settings["select_x1"]), rng)


def endpoint_error(samples, x):
    """Mean tip-to-observation distance over every draw and observation."""
    distances = [
        np.linalg.norm(
            sim.forward(H.to_simulator_order(samples[i, :, :4])) - x[i], axis=1
        )
        for i in range(samples.shape[0])
    ]
    stacked = np.concatenate(distances)
    return {"endpoint_mean": float(stacked.mean()),
            "endpoint_median": float(np.median(stacked))}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--attribution", type=Path, required=True,
                        help="An output directory of composition_attribution.py")
    result.add_argument("--checkpoint", type=Path, default=None)
    result.add_argument("--draws", type=int, default=4000)
    result.add_argument("--single-draws", type=int, default=20_000)
    result.add_argument("--pilot-draws", type=int, default=1024)
    result.add_argument("--timesteps", type=int, default=100)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--composition-temperature", type=float, default=1.0)
    result.add_argument("--kde-bandwidth", type=float, default=None)
    result.add_argument("--score-bandwidth", type=float, default=None)
    result.add_argument("--score-noise-scale", type=float, default=0.05)
    result.add_argument("--score-prior", default="analytic",
                        choices=("analytic", "network"))
    result.add_argument("--estimators", nargs="*", default=["density", "score"],
                        choices=("density", "score"),
                        help="Shared-marginal estimator: 'density' multiplies "
                             "per-observation KDEs, 'score' regresses the "
                             "network's joint score (Fisher's identity).")
    result.add_argument("--methods", nargs="*",
                        default=["gauss_jacobian", "gauss_hierarchical",
                                 "langevin_fnpse"])
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path, default=None)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly on this shared host.")

    settings = json.loads((arguments.attribution / "metrics.json").read_text())
    curves = np.load(arguments.attribution / "curves.npz")
    grid = curves["grid"]
    joint_reference = curves["joint_reference"]
    reference_moments = moments(grid, joint_reference)
    arguments.seed = int(settings["seed"])
    checkpoint = arguments.checkpoint or Path(settings["checkpoint"])

    x = dataset_for(settings)
    stored_x = curves["observations"]
    if not np.allclose(x, stored_x):
        raise SystemExit(
            "Rebuilt dataset does not match the one in curves.npz; the "
            "generator or its arguments have drifted."
        )
    data = torch.as_tensor(x, dtype=torch.float32)
    inference.COMMON["timesteps"] = arguments.timesteps
    model = trainer.load_to(checkpoint, arguments.device)

    print(f"{settings['observations']} observations, dataset "
          f"{settings.get('dataset', 'selected')}, seed {settings['seed']}")
    print(f"reference: mean {reference_moments['mean']:+.4f} "
          f"sd {reference_moments['sd']:.4f}  "
          f"mass>0 "
          f"{float(np.trapezoid(joint_reference[grid >= 0], grid[grid >= 0])):.3f}\n")

    rows = {}
    for method in arguments.methods:
        samples, _, seconds = inference.draw(
            model, data, method, arguments, arguments.device
        )
        shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
        rows[method] = {
            **compare(grid, kde_density(shared, grid, arguments.kde_bandwidth),
                      joint_reference, reference_moments),
            **endpoint_error(samples, x), "seconds": seconds,
        }

    curve = {}
    for estimator in arguments.estimators:
        started = time.perf_counter()
        samples, aux = mp.draw(model, data, arguments, arguments.device,
                               estimator=estimator)
        seconds = time.perf_counter() - started
        name = f"marginal_product_{estimator}"
        curve[name] = aux["density"]
        rows[name] = {
            **compare(grid, aux["density"], joint_reference, reference_moments),
            **endpoint_error(samples, x), "seconds": seconds,
        }

    print(f"{'method':>20}{'mean err/sig':>14}{'width':>8}{'W1/sig':>9}"
          f"{'mass>0':>9}{'dip':>7}{'endpoint':>10}{'secs':>8}")
    for name, row in rows.items():
        print(f"{name:>20}{row['mean_error_sigma']:>+14.3f}"
              f"{row['width_ratio']:>8.3f}{row['w1_over_sigma']:>9.3f}"
              f"{row['positive_mass']:>9.3f}{row['dip_ratio']:>7.3f}"
              f"{row['endpoint_median']:>10.4f}{row['seconds']:>8.1f}")

    output = arguments.output or (arguments.attribution / "marginal_product.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "attribution": str(arguments.attribution),
        "checkpoint": str(checkpoint),
        "settings": settings,
        "draws": int(arguments.draws),
        "single_draws": int(arguments.single_draws),
        "reference": reference_moments,
        "methods": rows,
    }, indent=2, sort_keys=True))
    np.savez_compressed(
        arguments.attribution / "marginal_product_curve.npz",
        grid=grid, density=curve.get("marginal_product_density",
                                     aux["density"]),
        per_observation=aux["per_observation"], shared_draws=aux["shared_draws"],
        **{f"curve_{k}": v for k, v in curve.items()},
    )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
