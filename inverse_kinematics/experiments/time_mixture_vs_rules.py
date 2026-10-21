#!/usr/bin/env python3
"""Wall-clock cost of the mixture product against the Gaussian rules.

What is actually being compared, and why the split matters
---------------------------------------------------------
``gauss_hierarchical`` runs one compositional reverse diffusion over all ``n``
observations *jointly*, preceded by a pilot pass that draws ``precision_est_samples``
per observation to estimate the covariance bank.  It returns shared and local
draws together.

The mixture product does not compose a score at all.  It runs

  A. one ordinary amortized sampler call, batched over all ``n`` observations,
     giving ``p(theta | x_j)`` -- the same object the pilot pass estimates
     moments of, so this is the fair counterpart of the pilot, not an extra;
  B. EM fits, the mixture product and pruning -- pure CPU, no network;
  C. one more sampler call for ``p(l_j | a*, x_j)`` to attach the locals.

A and C are ordinary batched sampler calls, so the honest comparison is
A + B + C against ``gauss_hierarchical``'s whole run, and separately B on its own,
since B is the only part this method adds to what the library already does.

Every arm is timed on the same device, checkpoint, observations and seed.
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
from composition_attribution import DEFAULT_CHECKPOINT  # noqa: E402
from mixture_product import fit_mixture  # noqa: E402
from mixture_refinements import sequential_product  # noqa: E402

OUTPUT = ROOT / "artifacts" / "mixture_product"


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_stage_b_only(arguments):
    """Stage B on saved draws: EM, product, pruning. No network anywhere."""
    archive = (ROOT / "artifacts" / "composition_attribution"
               / arguments.from_saved / "curves.npz")
    with np.load(archive) as stored:
        draws = stored["single_draws"]
    n = draws.shape[0]
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    results = {}
    print(f"{arguments.from_saved}: n={n}, {draws.shape[1]} draws per observation")
    print(f"{'stage':<34}{'seconds':>10}")
    for k in arguments.components:
        fit_timings = []
        for _ in range(arguments.repeats):
            started = time.perf_counter()
            fits = [fit_mixture(draws[j], k, seed=arguments.seed + j)
                    for j in range(n)]
            fit_timings.append(time.perf_counter() - started)
        fit_seconds = float(np.median(fit_timings))
        for beam in arguments.beam:
            product_timings = []
            for _ in range(arguments.repeats):
                started = time.perf_counter()
                sequential_product(fits, prior_precision, beam, "merge")
                product_timings.append(time.perf_counter() - started)
            product_seconds = float(np.median(product_timings))
            results[f"k{k}_beam{beam}_merge"] = {
                "em_seconds": fit_seconds,
                "product_seconds": product_seconds,
                "seconds": fit_seconds + product_seconds,
            }
            print(f"{'K=' + str(k) + ', beam ' + str(beam) + ' (merge)':<34}"
                  f"{fit_seconds + product_seconds:>10.3f}"
                  f"   (EM {fit_seconds:.3f} + product {product_seconds:.4f})")
    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / f"runtime_stage_b_{arguments.from_saved}.json"
    path.write_text(json.dumps({"observations": n, "results": results}, indent=1))
    print(f"\nWrote {path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--observations", type=int, default=8)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--single-draws", type=int, default=20_000)
    parser.add_argument("--pilot-draws", type=int, default=1024)
    parser.add_argument("--components", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--beam", type=int, nargs="+", default=[8, 32, 1024])
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--churn-eta", type=float, default=2.0)
    parser.add_argument("--jacobian-refresh", type=int, default=5)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--composition-temperature", type=float, default=1.0)
    parser.add_argument("--select-x1", type=float, default=0.15)
    parser.add_argument("--true-angle", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+",
                        default=["gauss_hierarchical", "gauss_jacobian",
                                 "langevin_fnpse"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-saved", default=None,
                        help="time stage B only, on draws saved by "
                             "composition_attribution.py. No model, no GPU -- "
                             "stage B is pure CPU, so this measures it exactly, "
                             "and it is the part the mixture *adds* to what "
                             "gauss_hierarchical already pays for.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)

    if arguments.from_saved:
        return time_stage_b_only(arguments)

    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "Set CUDA_VISIBLE_DEVICES explicitly: this host is shared."
        )
    inference.COMMON["timesteps"] = arguments.timesteps

    from compass import ScoreBasedInferenceModel as SBIm
    model = SBIm.load(str(arguments.checkpoint), device=arguments.device)

    rng = sim.generator("bimodal-dataset", arguments.seed)
    x, _ = selected_dataset(arguments.observations, arguments.true_angle,
                            arguments.select_x1, rng)
    data = torch.as_tensor(np.asarray(x), dtype=torch.float32)

    results = {}

    # --- the rules, end to end ----------------------------------------------
    for method in arguments.methods:
        timings = []
        for _ in range(arguments.repeats):
            _, _, seconds = inference.draw(model, data, method, arguments,
                                           arguments.device)
            timings.append(seconds)
        results[method] = {"seconds_median": float(np.median(timings)),
                           "seconds_all": timings}
        print(f"{method:<28} {np.median(timings):8.2f} s")

    # --- A: one batched amortized call over all n observations ---------------
    observations = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    timings = []
    for _ in range(arguments.repeats):
        torch.manual_seed(arguments.seed)
        synchronize(arguments.device)
        started = time.perf_counter()
        samples = model.sample(
            x=observations, num_samples=arguments.single_draws,
            timesteps=arguments.timesteps, method="dpm", order=2,
            churn_eta=arguments.churn_eta, corrector_steps=0,
            final_corrector_steps=0, device=arguments.device, verbose=False,
        )
        synchronize(arguments.device)
        timings.append(time.perf_counter() - started)
    draws = samples.detach().cpu().numpy()[:, :, H.HIERARCHY[0]].astype(np.float64)
    stage_a = float(np.median(timings))
    results["stage_a_single_observation_sampling"] = {
        "seconds_median": stage_a, "seconds_all": timings,
        "draws_per_observation": arguments.single_draws,
    }
    print(f"{'A: amortized p(theta|x_j)':<28} {stage_a:8.2f} s "
          f"({arguments.single_draws} draws x {arguments.observations} obs)")

    # --- B: the CPU-only part this method adds -------------------------------
    prior_precision = 1.0 / float(H.GLOBAL_PRIOR_STD[0]) ** 2
    for k in arguments.components:
        fit_timings = []
        for _ in range(arguments.repeats):
            started = time.perf_counter()
            fits = [fit_mixture(draws[j], k, seed=arguments.seed + j)
                    for j in range(arguments.observations)]
            fit_timings.append(time.perf_counter() - started)
        for beam in arguments.beam:
            product_timings = []
            for _ in range(arguments.repeats):
                started = time.perf_counter()
                sequential_product(fits, prior_precision, beam, "merge")
                product_timings.append(time.perf_counter() - started)
            key = f"stage_b_k{k}_beam{beam}_merge"
            results[key] = {
                "fit_seconds_median": float(np.median(fit_timings)),
                "product_seconds_median": float(np.median(product_timings)),
                "seconds_median": float(np.median(fit_timings)
                                        + np.median(product_timings)),
            }
            print(f"{'B: EM + product K=' + str(k) + ' beam ' + str(beam):<28} "
                  f"{np.median(fit_timings) + np.median(product_timings):8.2f} s "
                  f"(EM {np.median(fit_timings):.2f} + "
                  f"product {np.median(product_timings):.3f})")

    arguments.output.mkdir(parents=True, exist_ok=True)
    path = arguments.output / "runtime.json"
    path.write_text(json.dumps(
        {"results": results, "config": {
            k: str(v) for k, v in vars(arguments).items()}}, indent=1))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
