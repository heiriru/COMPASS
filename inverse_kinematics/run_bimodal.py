#!/usr/bin/env python3
"""Can compositional score modelling represent a bimodal shared posterior?

The arm's reflection ``(a1,a2,a3,h) -> (-a1,-a2,-a3, 2*x1-h)`` is an exact
involution of the likelihood whose only cost is ``2*x1*(h-x1)/sigma_h^2``. At
``x1 = 0`` that cost is identically zero, so both branches carry equal prior mass
and the posterior is exactly symmetric under negating every angle. The BayesFlow
notebook exploits this by hand -- its bimodal figure is taken at
``obs = [0, 1.5]``.

This script builds the **tall** version: ``n`` observations sharing one
``angle_1``, each selected to have ``|x1| < eps``, so the shared posterior is
bimodal at ``+-angle_1``. Then it asks two separate questions, which must not be
conflated:

1. **Can the network represent bimodality at all?** The single-observation
   posterior at ``obs = [0, 1.5]``, with no composition anywhere. If this is
   unimodal the composition experiment is meaningless, because there is no second
   mode for any rule to preserve.
2. **Does composition preserve it?** The three rules on the tall dataset, graded
   against the SBC-validated ABC reference. This is where the Gaussian rules are
   under real strain: ``gauss_hierarchical`` weights each observation by a single
   covariance, an approximation that a two-branch posterior violates about as
   badly as anything can.

**No retraining is needed or done.** The network is trained on single
``(theta, x)`` rows from the prior predictive; the selection that produces
bimodality is a property of the evaluation dataset, not of the training
distribution. Reusing the checkpoint is what keeps this a clean test of the
*inference*, and retraining on selected data would instead answer a different
question.

Usage:
    python inverse_kinematics/run_bimodal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

CPU_INFO = configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402
from bimodality_study import (bimodality, exact_zero_dataset,  # noqa: E402
                              selected_dataset)

ROOT = Path(__file__).resolve().parent
SINGLE_OBSERVATION = np.array([0.0, 1.5])


def kde_shape(draws, grid, bandwidth=None):
    """Mode structure of a set of draws, on the reference's own grid."""
    from scipy.stats import gaussian_kde

    density = gaussian_kde(np.asarray(draws, dtype=np.float64),
                           bw_method=bandwidth)(grid)
    density = density / np.trapezoid(density, grid)
    shape = bimodality(grid, density)
    draws = np.asarray(draws)
    shape["fraction_positive"] = float(np.mean(draws > 0))
    return density, shape


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="h16d2")
    result.add_argument("--arch", default="legacy")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--observations", type=int, default=8)
    result.add_argument("--true-angle", type=float, default=0.6)
    result.add_argument("--select-x1", type=float, default=0.15)
    result.add_argument("--exact-x1-zero", action="store_true",
                        help="Place every observation exactly at x1 = 0, where "
                             "the reflection costs no prior mass and the shared "
                             "posterior is provably symmetric.")
    result.add_argument("--draws", type=int, default=4_000)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--composition-temperature", type=float, default=1.0)
    result.add_argument("--arruda-damping-at-noise", type=float, default=None,
                        help="Noise-endpoint d(1) for Langevin F-NPSE + Arruda "
                             "damping; defaults to 1/N. d(0) is always one.")
    result.add_argument("--arruda-damping-schedule", default="exponential",
                        choices=("exponential", "linear", "cosine", "quadratic", "sqrt"),
                        help="Endpoint-preserving schedule for damped Langevin F-NPSE.")
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--pilot-draws", type=int, default=2_000)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--reference-draws", type=int, default=120_000)
    result.add_argument("--reference-bandwidth", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=99)
    result.add_argument("--methods", nargs="+", default=list(inference.METHODS))
    result.add_argument("--mask-menu", default="bernoulli",
                        choices=trainer.MASK_MENUS,
                        help="Training condition-mask recipe of the checkpoint "
                             "to load.")
    result.add_argument("--menu", dest="mask_menu", action="store_const",
                        const="posterior",
                        help="Deprecated alias for --mask-menu posterior.")
    result.add_argument("--time-sampling", default="log_sigma",
                        choices=("log_sigma", "mixture", "uniform"),
                        help="Diffusion-time recipe of the checkpoint to load.")
    result.add_argument("--model-suffix", default="",
                        help="Appended to the resolved model directory name; "
                             "matches train_model.py --output-suffix.")
    result.add_argument("--device", default="cuda")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts",
                        help="Root the CHECKPOINT is looked up under. Changing "
                             "this changes which model is loaded, not where "
                             "results go -- use --output for that.")
    # Without this, the only way to keep one run's results was to redirect
    # --artifacts, which also redirects the checkpoint lookup and so loads the
    # wrong model (or none). Every run therefore wrote to artifacts/bimodal and
    # silently overwrote its predecessor.
    result.add_argument("--output", type=Path, default=None,
                        help="Directory for this run's metrics.json and "
                             "posteriors.npz (default: <artifacts>/bimodal).")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    directory = trainer.model_directory(
        arguments.artifacts, arguments.config, arguments.arch,
        arguments.simulations, menu=arguments.mask_menu,
        time_sampling=arguments.time_sampling,
    )
    if arguments.model_suffix:
        directory = directory.with_name(directory.name + arguments.model_suffix)
    model = trainer.load_to(directory / "Model_checkpoint.pt", arguments.device)
    print(f"Loaded {arguments.arch}/{arguments.config} (no retraining: the "
          f"training distribution is unchanged)\n")

    grid = np.linspace(-2.0, 2.0, 801)

    # --- 1. can the network represent two modes at all? -----------------
    single = torch.as_tensor(SINGLE_OBSERVATION.reshape(1, 2), dtype=torch.float32)
    torch.manual_seed(arguments.seed)
    # Pure churn, and this line is the whole point of lifting churn into the
    # library. README.md's Part 3 measured that Langevin correctors *break the
    # reflection symmetry* here -- frac>0 falls 0.503 -> 0.385 -> 0.240 as
    # correctors go 0 -> 2 -> 10, relaxing the ensemble onto whichever branch it
    # is nearest, which is exactly a corrector's failure mode on a multimodal
    # target. Corrector-free sampling restores frac>0 = 0.503, the symmetry the
    # reflection demands. Until now the single-observation `Sampler` had no
    # churn path at all (churn was a monkeypatch over MultiObsSampler), so this
    # probe had to choose between correctors and an uninitialised
    # predictor-only run that returns approximately the prior. It no longer does.
    single_draws = model.sample(
        x=single, num_samples=arguments.draws, timesteps=100, method="dpm",
        order=2, churn_eta=arguments.churn_eta,
        corrector_steps=0, final_corrector_steps=0,
        device=arguments.device, verbose=False,
    ).detach().cpu().numpy()[0]
    single_density, single_shape = kde_shape(single_draws[:, 0], grid)
    print(f"Single observation at {SINGLE_OBSERVATION.tolist()} "
          "(no composition):")
    print(f"  angle_1 modes {single_shape['modes']}  dip "
          f"{single_shape['dip_ratio']:.3f}  frac>0 "
          f"{single_shape['fraction_positive']:.3f}")
    for index, name in enumerate(H.NODE_NAMES[:4]):
        column = single_draws[:, index]
        print(f"    {name:12} mean {column.mean():+.3f} sd {column.std():.3f} "
              f"frac>0 {np.mean(column > 0):.3f}")

    # --- 2. the tall, bimodal dataset -----------------------------------
    dataset_rng = sim.generator("bimodal-dataset", arguments.seed)
    if arguments.exact_x1_zero:
        x, heights = exact_zero_dataset(arguments.observations,
                                        arguments.true_angle, dataset_rng)
        print(f"x1 = 0 exactly for all {arguments.observations} observations, "
              f"with height fixed at its prior mode ({heights.max():+.3f}); "
              "the shared posterior is provably symmetric here")
    else:
        x = selected_dataset(arguments.observations, arguments.true_angle,
                             arguments.select_x1, dataset_rng)
    data = torch.as_tensor(x, dtype=torch.float32)
    started = time.perf_counter()
    _, density = H.reference_global(
        x, grid=grid, draws=arguments.reference_draws,
        bandwidth=arguments.reference_bandwidth, seed=arguments.seed,
        symmetrize=arguments.exact_x1_zero,
    )
    reference_shape = bimodality(grid, density)
    reference_mean, reference_std, _ = H.grid_moments(grid, density)
    selection = ("x1 = 0 exactly" if arguments.exact_x1_zero
                 else f"|x1| < {arguments.select_x1}")
    print(f"\n{arguments.observations} observations, {selection}, "
          f"true angle_1 = {arguments.true_angle}")
    print(f"Reference: modes {reference_shape['modes']}  dip "
          f"{reference_shape['dip_ratio']:.3f}  mass ratio "
          f"{reference_shape['mode_mass_ratio']:.3f}  sd {reference_std:.3f}  "
          f"({time.perf_counter() - started:.0f}s)")
    if reference_shape["dip_ratio"] > 0.5:
        print("  WARNING: the reference is not clearly bimodal for this draw; "
              "the composition test below is weak. Try another --seed.")

    results, metrics = {}, {}
    print(f"\n{'rule':>24}{'modes':>22}{'dip':>8}{'frac>0':>9}{'W1/sig':>9}")
    for method in arguments.methods:
        samples, sampler, runtime = inference.draw(
            model, data, method, arguments, arguments.device
        )
        shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
        _, shape = kde_shape(shared, grid)
        row = {
            "label": inference.METHODS[method]["label"],
            "colour": inference.METHODS[method]["colour"],
            "w1_over_sigma": inference.wasserstein_to_grid(
                shared, grid, density) / reference_std,
            "posterior_std": float(shared.std()),
            "seconds": runtime, **shape,
        }
        modes = ", ".join(f"{m:+.2f}" for m in shape["modes"])
        print(f"{method:>24}{modes:>22}{shape['dip_ratio']:>8.3f}"
              f"{shape['fraction_positive']:>9.3f}{row['w1_over_sigma']:>9.3f}")
        results[method] = samples
        metrics[method] = row

    output = arguments.output or (arguments.artifacts / "bimodal")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "posteriors.npz", grid=grid, density=density,
        observations=x, single_draws=single_draws,
        single_density=single_density, truth=arguments.true_angle,
        **{f"samples_{k}": v for k, v in results.items()},
    )
    (output / "metrics.json").write_text(json.dumps({
        "arch": arguments.arch, "config": arguments.config,
        "observations": int(arguments.observations),
        "true_angle": arguments.true_angle,
        "select_x1": (0.0 if arguments.exact_x1_zero else arguments.select_x1),
        "exact_x1_zero": bool(arguments.exact_x1_zero),
        "draws": int(arguments.draws),
        "churn_eta": arguments.churn_eta, "seed": int(arguments.seed),
        "arruda_damping_at_noise": arguments.arruda_damping_at_noise,
        "arruda_damping_schedule": arguments.arruda_damping_schedule,
        "single_observation": {
            "observables": SINGLE_OBSERVATION.tolist(), **single_shape,
        },
        "reference": {"mean": reference_mean, "std": reference_std,
                      **reference_shape},
        "methods": metrics,
    }, indent=2, sort_keys=True))
    print(f"\nWrote {output / 'posteriors.npz'}")
    print(f"Wrote {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
