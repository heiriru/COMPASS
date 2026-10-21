#!/usr/bin/env python3
"""Where does the non-symmetric composition error come from: network or rule?

The tall model here has one shared coordinate ``a = angle_1`` and independent
local latents per observation, so the shared marginal factorises *exactly*::

    p(a | x_1..x_n)  proportional to  p(a)^(1-n) * prod_i p(a | x_i)

``a`` is one-dimensional, so that product can be formed **numerically on a
grid**, with no sampler and no composition rule anywhere in it. That gives a
third curve alongside the ABC reference and the sampler output, and the three
together attribute the error:

* ``reference_product`` -- the same product built from *ABC* single-observation
  posteriors. This is an algebraic identity with the joint ABC reference, so it
  is a self-test of the grid/KDE pipeline, not evidence about anything. If it
  does not reproduce the joint reference, nothing below can be trusted.
* ``network_product`` -- the product of the **network's own** single-observation
  posteriors. This is what a perfect composition rule would return. Its distance
  from the reference is the part of the error the network is responsible for and
  that no sampler change can remove.
* the sampler rows -- what the composition rules actually return. Their distance
  from ``network_product`` is the part the rule is responsible for.

Why this is the right question for the non-symmetric case: at ``x1 = 0`` the
posterior is exactly symmetric about the prior mean, so any error that pulls the
shared posterior *toward the prior* is invisible in the mean and shows up only in
the shape. Selecting ``|x1| < eps`` with eps > 0 breaks that symmetry and lets
the same error appear as a mean shift. A method can therefore look fine at
``x1 = 0`` and badly biased just off it, without anything having changed.

Usage:
    python inverse_kinematics/experiments/composition_attribution.py \
        --observations 8 --select-x1 0.15
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import hierarchical as H  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402
from bimodality_study import bimodality, exact_zero_dataset, selected_dataset  # noqa: E402

DEFAULT_CHECKPOINT = (
    ROOT / "artifacts/models/"
    "legacy_h16d2_n10000_six_weighted_log_sigma_e10000/Model_checkpoint.pt"
)
# Small enough not to matter against a density that integrates to one, large
# enough that log() of it is finite; the product of n logs is taken on the
# support mask below, so this only guards the arithmetic.
FLOOR = 1e-300


def normalise(grid, density):
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("density did not normalise")
    return density / area


def from_log(grid, log_density):
    log_density = np.asarray(log_density, dtype=np.float64)
    log_density = log_density - log_density.max()
    return normalise(grid, np.exp(log_density))


def kde_density(draws, grid, bandwidth=None):
    draws = np.asarray(draws, dtype=np.float64)
    density = gaussian_kde(draws, bw_method=bandwidth)(grid)
    return normalise(grid, density)


def moments(grid, density):
    mean, sd, mode = H.grid_moments(grid, density)
    return {"mean": mean, "sd": sd, "mode": mode}


def w1_grid(grid, left, right):
    """W1 between two densities tabulated on a shared grid."""
    def cdf(density):
        c = np.concatenate([[0.0], np.cumsum(
            0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
        return c / c[-1]
    return float(np.trapezoid(np.abs(cdf(left) - cdf(right)), grid))


def compare(grid, density, reference, reference_moments):
    stats = moments(grid, density)
    shape = bimodality(grid, density)
    return {
        **stats,
        "mean_error_sigma": (stats["mean"] - reference_moments["mean"])
                            / reference_moments["sd"],
        "width_ratio": stats["sd"] / reference_moments["sd"],
        "w1_over_sigma": w1_grid(grid, density, reference)
                         / reference_moments["sd"],
        "dip_ratio": shape["dip_ratio"],
        "mode_mass_ratio": shape["mode_mass_ratio"],
        "positive_mass": float(np.trapezoid(density[grid >= 0], grid[grid >= 0])),
    }


def prior_density(grid):
    scale = float(H.GLOBAL_PRIOR_STD[0])
    return normalise(grid, np.exp(-0.5 * (grid / scale) ** 2))


def grid_product(grid, per_observation, prior):
    """``prod_i p_i / prior^(n-1)``, formed in log space on the shared grid."""
    stack = np.asarray(per_observation, dtype=np.float64)
    n = stack.shape[0]
    log_terms = np.log(np.maximum(stack, FLOOR)).sum(axis=0)
    log_prior = np.log(np.maximum(prior, FLOOR))
    return from_log(grid, log_terms - (n - 1) * log_prior)


def single_observation_posteriors(model, x, grid, draws, bandwidth,
                                  churn_eta, seed, device, timesteps):
    """The network's own ``p(a | x_i)``, one ordinary non-compositional sample."""
    densities, all_draws = [], []
    for index, row in enumerate(x):
        torch.manual_seed(seed + index)
        sample = model.sample(
            x=torch.as_tensor(row[None, :], dtype=torch.float32),
            num_samples=draws, timesteps=timesteps, method="dpm", order=2,
            churn_eta=churn_eta, corrector_steps=0, final_corrector_steps=0,
            device=device, verbose=False,
        ).detach().cpu().numpy()[0, :, H.HIERARCHY[0]].astype(np.float64)
        all_draws.append(sample)
        densities.append(kde_density(sample, grid, bandwidth))
    return np.asarray(densities), np.asarray(all_draws)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--observations", type=int, default=8)
    result.add_argument("--select-x1", type=float, default=0.15)
    result.add_argument("--dataset", default="selected",
                        choices=("selected", "exact_zero", "prior"),
                        help="'selected' keeps |x1| < eps (bimodal, mildly "
                             "asymmetric); 'exact_zero' is the provably "
                             "symmetric control; 'prior' is an ordinary tall "
                             "dataset from the prior predictive -- the generic "
                             "non-symmetric case.")
    result.add_argument("--exact-x1-zero", action="store_true",
                        help="Deprecated alias for --dataset exact_zero.")
    result.add_argument("--true-angle", type=float, default=0.6)
    result.add_argument("--seed", type=int, default=99)
    result.add_argument("--draws", type=int, default=4000)
    result.add_argument("--single-draws", type=int, default=20_000)
    result.add_argument("--pilot-draws", type=int, default=1024)
    result.add_argument("--reference-draws", type=int, default=200_000)
    result.add_argument("--single-reference-draws", type=int, default=100_000,
                        help="Draws for the per-observation ABC references.")
    result.add_argument("--reference-bandwidth", type=float, default=0.05)
    result.add_argument("--kde-bandwidth", type=float, default=None,
                        help="Scott's rule if unset; --bandwidth-sweep varies it.")
    result.add_argument("--bandwidth-sweep", type=float, nargs="*",
                        default=[0.07, 0.10, 0.145, 0.20],
                        help="Rebuild network_product at each of these KDE "
                             "bandwidths. The product of n densities amplifies "
                             "tail error, so a conclusion that moves with the "
                             "bandwidth is an artifact of the estimator.")
    result.add_argument("--timesteps", type=int, default=100)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--composition-temperature", type=float, default=1.0)
    result.add_argument("--methods", nargs="*",
                        default=["gauss_jacobian", "gauss_hierarchical",
                                 "langevin_fnpse"])
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/composition_attribution")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "Set CUDA_VISIBLE_DEVICES explicitly: this host is shared and "
            "autocvd would block until a GPU is idle."
        )
    inference.COMMON["timesteps"] = arguments.timesteps
    span = 4.0 * H.GLOBAL_PRIOR_STD[0]
    grid = np.linspace(-span, span, 801)
    prior = prior_density(grid)

    if arguments.exact_x1_zero:
        arguments.dataset = "exact_zero"
    dataset_rng = sim.generator("bimodal-dataset", arguments.seed)
    if arguments.dataset == "exact_zero":
        x, _ = exact_zero_dataset(arguments.observations, arguments.true_angle,
                                  dataset_rng)
    elif arguments.dataset == "prior":
        # An ordinary tall dataset: one shared angle_1, locals straight from
        # their prior, no selection on x at all.
        _, _, x = H.simulate_dataset(arguments.observations, dataset_rng,
                                     angle_1=arguments.true_angle)
    else:
        x = selected_dataset(arguments.observations, arguments.true_angle,
                             arguments.select_x1, dataset_rng)
    data = torch.as_tensor(x, dtype=torch.float32)
    selection = {"exact_zero": "x1 = 0 exactly",
                 "prior": "no selection (prior predictive)",
                 "selected": f"|x1| < {arguments.select_x1}"}[arguments.dataset]
    print(f"{arguments.observations} observations, {selection}, "
          f"true angle_1 = {arguments.true_angle}")
    print(f"checkpoint {arguments.checkpoint.parent.name}\n")

    # --- 1. references --------------------------------------------------
    started = time.perf_counter()
    _, joint_reference = H.reference_global(
        x, grid=grid, draws=arguments.reference_draws,
        bandwidth=arguments.reference_bandwidth, seed=arguments.seed,
        symmetrize=(arguments.dataset == "exact_zero"),
    )
    reference_moments = moments(grid, joint_reference)
    print(f"joint ABC reference: mean {reference_moments['mean']:+.4f} "
          f"sd {reference_moments['sd']:.4f} mode {reference_moments['mode']:+.3f}"
          f"  ({time.perf_counter() - started:.0f}s)")

    per_observation_reference = np.asarray([
        H.reference_global(
            row[None, :], grid=grid, draws=arguments.single_reference_draws,
            bandwidth=arguments.reference_bandwidth, seed=arguments.seed,
        )[1] for row in x
    ])

    # --- 2. the network's own single-observation posteriors --------------
    model = trainer.load_to(arguments.checkpoint, arguments.device)
    per_observation_network, single_draws = single_observation_posteriors(
        model, x, grid, arguments.single_draws, arguments.kde_bandwidth,
        arguments.churn_eta, arguments.seed, arguments.device,
        arguments.timesteps,
    )

    rows = {
        "reference_product": grid_product(grid, per_observation_reference, prior),
        "network_product": grid_product(grid, per_observation_network, prior),
    }

    # --- 3. what the rules actually return -------------------------------
    sampler_draws = {}
    for method in arguments.methods:
        samples, _, seconds = inference.draw(
            model, data, method, arguments, arguments.device
        )
        shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
        sampler_draws[method] = shared
        rows[method] = kde_density(shared, grid, arguments.kde_bandwidth)
        print(f"  {method} sampled in {seconds:.0f}s")

    # --- 4. the table ----------------------------------------------------
    summary = {name: compare(grid, density, joint_reference, reference_moments)
               for name, density in rows.items()}
    print(f"\n{'curve':>22}{'mean':>9}{'sd':>8}{'mean err/sig':>14}"
          f"{'width':>8}{'W1/sig':>9}{'mass>0':>9}")
    print(f"{'joint ABC reference':>22}{reference_moments['mean']:>+9.4f}"
          f"{reference_moments['sd']:>8.4f}{0.0:>14.3f}{1.0:>8.3f}{0.0:>9.3f}"
          f"{float(np.trapezoid(joint_reference[grid >= 0], grid[grid >= 0])):>9.3f}")
    for name, row in summary.items():
        print(f"{name:>22}{row['mean']:>+9.4f}{row['sd']:>8.4f}"
              f"{row['mean_error_sigma']:>+14.3f}{row['width_ratio']:>8.3f}"
              f"{row['w1_over_sigma']:>9.3f}{row['positive_mass']:>9.3f}")

    # --- 4b. is network_product an artifact of the density estimate? ------
    # Each single-observation density is a KDE, and the product of n of them
    # raises any smoothing bias to the n-th power. If the verdict depends on the
    # bandwidth it is not a verdict.
    bandwidth_rows = []
    if arguments.bandwidth_sweep:
        print(f"\n{'KDE bandwidth':>15}{'mean err/sig':>14}{'width':>8}"
              f"{'W1/sig':>9}{'mass>0':>9}")
        for bandwidth in arguments.bandwidth_sweep:
            densities = np.asarray([kde_density(draw, grid, bandwidth)
                                    for draw in single_draws])
            product = grid_product(grid, densities, prior)
            row = {"bandwidth": bandwidth,
                   **compare(grid, product, joint_reference, reference_moments)}
            bandwidth_rows.append(row)
            print(f"{bandwidth:>15.3f}{row['mean_error_sigma']:>+14.3f}"
                  f"{row['width_ratio']:>8.3f}{row['w1_over_sigma']:>9.3f}"
                  f"{row['positive_mass']:>9.3f}")

    # --- 5. per-observation attribution ----------------------------------
    per_observation = []
    print(f"\n{'obs':>5}{'x1':>8}{'x2':>8}{'ref mean':>10}{'net mean':>10}"
          f"{'err/sig':>9}{'width':>8}{'W1/sig':>9}")
    for index in range(len(x)):
        reference_stats = moments(grid, per_observation_reference[index])
        network_stats = moments(grid, per_observation_network[index])
        row = {
            "observation": index,
            "x": x[index].tolist(),
            "reference": reference_stats,
            "network": network_stats,
            "mean_error_sigma": (network_stats["mean"] - reference_stats["mean"])
                                / reference_stats["sd"],
            "width_ratio": network_stats["sd"] / reference_stats["sd"],
            "w1_over_sigma": w1_grid(grid, per_observation_network[index],
                                     per_observation_reference[index])
                             / reference_stats["sd"],
        }
        per_observation.append(row)
        print(f"{index:>5}{x[index][0]:>+8.3f}{x[index][1]:>+8.3f}"
              f"{reference_stats['mean']:>+10.4f}{network_stats['mean']:>+10.4f}"
              f"{row['mean_error_sigma']:>+9.3f}{row['width_ratio']:>8.3f}"
              f"{row['w1_over_sigma']:>9.3f}")
    mean_bias = float(np.mean([r["mean_error_sigma"] for r in per_observation]))
    print(f"\nmean per-observation bias: {mean_bias:+.3f} single-obs sd")

    arguments.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output / "curves.npz", grid=grid, prior=prior,
        joint_reference=joint_reference,
        per_observation_reference=per_observation_reference,
        per_observation_network=per_observation_network,
        observations=x, single_draws=single_draws,
        **{f"curve_{k}": v for k, v in rows.items()},
        **{f"draws_{k}": v for k, v in sampler_draws.items()},
    )
    (arguments.output / "metrics.json").write_text(json.dumps({
        "checkpoint": str(arguments.checkpoint),
        "observations": int(arguments.observations),
        "dataset": arguments.dataset,
        "select_x1": (None if arguments.dataset == "prior" else
                      (0.0 if arguments.dataset == "exact_zero"
                       else arguments.select_x1)),
        "exact_x1_zero": arguments.dataset == "exact_zero",
        "true_angle": arguments.true_angle,
        "seed": int(arguments.seed),
        "draws": int(arguments.draws),
        "single_draws": int(arguments.single_draws),
        "reference_draws": int(arguments.reference_draws),
        "single_reference_draws": int(arguments.single_reference_draws),
        "reference_bandwidth": arguments.reference_bandwidth,
        "kde_bandwidth": arguments.kde_bandwidth,
        "churn_eta": arguments.churn_eta,
        "composition_temperature": arguments.composition_temperature,
        "reference": reference_moments,
        "curves": summary,
        "per_observation": per_observation,
        "mean_per_observation_bias_sigma": mean_bias,
        "bandwidth_sweep": bandwidth_rows,
    }, indent=2, sort_keys=True))
    print(f"\nWrote {arguments.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
