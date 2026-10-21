#!/usr/bin/env python3
"""Inverse kinematics: mixture composition with K inferred, against the rules.

One dataset, one checkpoint, one seed. Four methods -- the three composition
rules in ``MultiObsSampler`` and the Gaussian-mixture product -- each timed end
to end and graded against the same ABC reference.

**K is never supplied and never swept.** ``mixture_inference.choose_and_fit``
picks it per observation by BIC at inference time; the chosen values are
reported, not chosen by the caller.

Saves the global posterior density, the local posteriors, the endpoint errors
and the timings, so the figure script needs no model.
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
import marginal_product  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
from bimodality_study import selected_dataset  # noqa: E402
from composition_attribution import (  # noqa: E402
    DEFAULT_CHECKPOINT, compare, kde_density, moments,
)

OUTPUT = ROOT / "artifacts" / "mixture_inference"
RULES = ("gauss_hierarchical", "gauss_jacobian", "langevin_fnpse")


def endpoint_error(samples, x):
    """Mean tip-to-observation distance over every draw and observation.

    Reference-free: the forward map is deterministic, so this checks that the
    locals actually reproduce the data rather than merely looking plausible.
    ``H.to_simulator_order`` is required -- node order is not simulator order.
    """
    distances = [
        np.linalg.norm(
            sim.forward(H.to_simulator_order(samples[i, :, :4])) - x[i], axis=1
        )
        for i in range(samples.shape[0])
    ]
    stacked = np.concatenate(distances)
    return float(stacked.mean()), float(np.median(stacked))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--attribution", type=Path, default=ROOT /
                        "artifacts/composition_attribution/selected_n8_seed99")
    parser.add_argument("--observations", type=int, default=8)
    parser.add_argument("--select-x1", type=float, default=0.15)
    parser.add_argument("--true-angle", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--single-draws", type=int, default=5000)
    parser.add_argument("--pilot-draws", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--churn-eta", type=float, default=2.0)
    parser.add_argument("--jacobian-refresh", type=int, default=5)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--composition-temperature", type=float, default=1.0)
    parser.add_argument("--kde-bandwidth", type=float, default=None)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--em-restarts", type=int, default=1)
    parser.add_argument("--local-estimator", choices=["sampler", "resample"],
                        default="sampler",
                        help="'resample' replaces stage 3's sampler call with a "
                             "Nadaraya-Watson resampler over stage 1's draws.")
    parser.add_argument("--resample-bandwidth", type=float, default=0.0)
    parser.add_argument("--local-strata", type=int, default=0,
                        help="Stratify stage 3 over this many quantiles of the "
                             "composed posterior; 0 keeps the per-draw path.")
    parser.add_argument("--bag", type=int, default=0,
                        help="Bootstrap replicates over observations for the "
                             "shared marginal; 0 disables bagging.")
    parser.add_argument("--composed", nargs="+",
                        default=["mixture"],
                        help="Composed arms to run through "
                             "marginal_product.draw: any of mixture, tilt, "
                             "tilt_bagged.")
    parser.add_argument("--rules", nargs="+", default=None,
                        help="Composition rules to run as baselines; 'none' "
                             "runs only the composed arms. gauss_jacobian with "
                             "--jacobian-refresh 1 is the memory ceiling of "
                             "this script on a large trunk, so a capacity "
                             "comparison wants 'none'.")
    parser.add_argument("--local-draws", type=int, default=0,
                        help="Cap on the draws stage 3 evaluates the locals at, "
                             "independent of --draws. Stage 3 costs n*draws "
                             "network rows and the global posterior never uses "
                             "it, so this is the main time knob. 0 means equal "
                             "to --draws.")
    parser.add_argument("--tilt-uncertainty", action="store_true",
                        help="Marginalise over the sampling covariance of the "
                             "fitted tilt parameters instead of plugging in the "
                             "point estimate. Widens the composed posterior by "
                             "exactly the amount the fit is uncertain.")
    parser.add_argument("--stage1-churn", type=float, default=None,
                        help="Churn for the stage-1 single-observation sampler, "
                             "separately from --churn-eta (which then applies to "
                             "stage 3 only). Defaults to --churn-eta.")
    parser.add_argument("--rewired-config", default=None,
                        help="Set to the trunk config (e.g. h16d2, h64d4wide) "
                             "when the checkpoint came from train_te.py; those "
                             "are rewired legacy models that SBIm.load cannot "
                             "reconstruct on its own.")
    parser.add_argument("--stage1-chunk", type=int, default=16,
                        help="Observations per stage-1 sampler call. The "
                             "single-observation posteriors are independent, so "
                             "this only caps peak GPU memory; 0 means one call.")
    parser.add_argument("--tilt-localise", type=float, default=1.0,
                        help="Power on the first-pass density used as the "
                             "localisation weight in the tilt_local arms; "
                             "below 1 widens the window, above 1 narrows it.")
    parser.add_argument("--tilt-bag", type=int, default=64,
                        help="Bootstrap replicates for the tilt_bagged arm, "
                             "kept separate from --bag so the mixture arm it "
                             "is compared against is unaffected.")
    parser.add_argument("--tilt-degrees", default="2,4,6",
                        help="Candidate log-polynomial degrees for the tilt "
                             "arms; chosen per observation by BIC.")
    parser.add_argument("--em-device", default=None,
                        help="Device for batched EM; None uses the CPU loop.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly: shared host.")
    inference.COMMON["timesteps"] = arguments.timesteps

    from compass import ScoreBasedInferenceModel as SBIm

    if arguments.rewired_config:
        # A ``te_rewire`` checkpoint is not self-describing: a legacy payload
        # never records ``time_embedding_size``, so ``SBIm.load`` rebuilds at
        # 256 unconditionally and raises on a checkpoint trained at any other
        # width. ``load_rewired`` reads the width off the weights first.
        import te_rewire as TE
        import train_te

        model, embedding = TE.load_rewired(
            arguments.checkpoint,
            lambda te, device: train_te.build(te, device=device,
                                              config=arguments.rewired_config),
            device=arguments.device)
        print(f"rewired legacy/{arguments.rewired_config}, "
              f"time_embedding_size {embedding}", flush=True)
    else:
        model = SBIm.load(str(arguments.checkpoint), device=arguments.device)

    # The saved ABC reference, and the dataset that produced it, rebuilt
    # deterministically so the grading is against exactly the same truth.
    with np.load(arguments.attribution / "curves.npz") as stored:
        grid = stored["grid"]
        reference = stored["joint_reference"]
        stored_x = stored["observations"]
    rng = sim.generator("bimodal-dataset", arguments.seed)
    x = selected_dataset(arguments.observations, arguments.true_angle,
                         arguments.select_x1, rng)
    x = np.asarray(x, dtype=np.float64)
    if not np.allclose(x, stored_x):
        raise SystemExit("dataset does not match the saved reference")
    data = torch.as_tensor(x, dtype=torch.float32)
    reference_moments = moments(grid, reference)

    results, curves = {}, {}
    selected_rules = () if arguments.rules == ["none"] else tuple(
        arguments.rules or RULES)
    unknown = set(selected_rules) - set(RULES)
    if unknown:
        raise SystemExit(f"unknown rules: {sorted(unknown)}")
    for method in selected_rules:
        samples, _, seconds = inference.draw(model, data, method, arguments,
                                             arguments.device)
        shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
        density = kde_density(shared, grid, arguments.kde_bandwidth)
        mean_error, median_error = endpoint_error(samples, x)
        results[method] = {
            **compare(grid, density, reference, reference_moments),
            "seconds": seconds, "endpoint_mean": mean_error,
            "endpoint_median": median_error,
        }
        curves[method] = density
        results[method]["_samples"] = samples
        print(f"  {method:<22} {seconds:6.1f}s  W1/sig "
              f"{results[method]['w1_over_sigma']:.3f}  endpoint "
              f"{mean_error:.4f}")

    for estimator in arguments.composed:
        name = "mixture_product" if estimator == "mixture" else f"{estimator}_product"
        started = time.perf_counter()
        samples, auxiliary = marginal_product.draw(
            model, data, arguments, arguments.device, grid=grid,
            estimator=estimator,
        )
        seconds = time.perf_counter() - started
        density = auxiliary["density"]
        mean_error, median_error = endpoint_error(samples, x)
        results[name] = {
            **compare(grid, density, reference, reference_moments),
            "seconds": seconds, "endpoint_mean": mean_error,
            "endpoint_median": median_error,
            "diagnostics": auxiliary["mixture"], "_samples": samples,
        }
        curves[name] = density
        print(f"  {name:<22} {seconds:6.1f}s  W1/sig "
              f"{results[name]['w1_over_sigma']:.3f}  endpoint "
              f"{mean_error:.4f}")
        print(f"    per-observation model order: "
              f"{auxiliary['mixture']['chosen_k']}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output / f"inverse_kinematics_n{arguments.observations}.npz",
        grid=grid, reference=reference, observations=x,
        **{f"curve_{name}": curve for name, curve in curves.items()},
        **{f"samples_{name}": results[name]["_samples"] for name in results},
    )
    table = {name: {k: v for k, v in entry.items() if not k.startswith("_")}
             for name, entry in results.items()}
    (arguments.output / f"inverse_kinematics_n{arguments.observations}.json").write_text(
        json.dumps(table, indent=1, default=float))
    print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
