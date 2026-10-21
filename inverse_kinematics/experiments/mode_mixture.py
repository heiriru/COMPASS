#!/usr/bin/env python3
"""Mode-mixture composition on the arm. See ``docs/mode_mixture.md``.

The Gaussian composition rules summarise each observation by a mean and a
covariance, and unequal branch weights are exactly what that cannot carry, so on
the arm they return a near-symmetric shared posterior (mass 0.467 against a
reference 0.243). Gaussians are not closed under the operations composition
needs; Gaussian *mixtures* are. This replaces the one-Gaussian summary of
``p(theta | x_i)`` by a K-component mixture, with K discovered by continuation
down the noise ladder, and composes those.

The machinery lives in :mod:`mode_mixture_core`, which the Gaussian-mixture
benchmark runs too; this file is the arm's dataset, referee and controls.

Controls, all reached from here:

* ``--force-k1`` -- the same pipeline with one component per observation, which
  is a Laplace version of the existing rules and should reproduce their failure;
* ``--component-source em`` -- the same discovered K and the same composition,
  but the components refit to the network's own draws by maximum likelihood, so
  a failure that survives it is not the Laplace approximation's fault;
* ``--sigma-branch`` -- weights read off higher up the ladder, where the path
  between modes is short.

Usage:
    python inverse_kinematics/experiments/mode_mixture.py \
        --observations 8 --select-x1 0.15 --seed 99 --component-source em
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
import mode_mixture_core as core  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402
from bimodality_study import exact_zero_dataset, selected_dataset  # noqa: E402
from composition_attribution import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    compare,
    grid_product,
    kde_density,
    moments,
    prior_density,
    single_observation_posteriors,
    w1_grid,
)

PROBLEM = core.Problem(
    latent=len(H.LATENT_INDICES),
    shared=H.HIERARCHY[0],
    condition_mask=H.CONDITION_MASK,
    prior_mean=H.PRIOR_MEAN,
    prior_std=H.PRIOR_STD,
)


def settings_from(arguments):
    return core.Settings(
        sigma_top=arguments.sigma_top, sigma_weights=arguments.sigma_weights,
        ladder=arguments.ladder, newton_steps=arguments.newton_steps,
        refine_steps=arguments.refine_steps, trust_radius=arguments.trust_radius,
        curvature_floor=arguments.curvature_floor,
        split_scale=arguments.split_scale,
        split_directions=arguments.split_directions,
        dedup_tolerance=arguments.dedup_tolerance, max_modes=arguments.max_modes,
        top_starts=arguments.top_starts, level_starts=arguments.level_starts,
        path_nodes=arguments.path_nodes, path_bend=arguments.path_bend,
        deconvolve=arguments.deconvolve,
        deconvolve_floor=arguments.deconvolve_floor,
        min_weight=arguments.min_weight,
        curvature_source=arguments.curvature_source,
        force_k1=arguments.force_k1, keep_saddles=arguments.keep_saddles,
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--observations", type=int, default=8)
    result.add_argument("--select-x1", type=float, default=0.15)
    result.add_argument("--dataset", default="selected",
                        choices=("selected", "exact_zero", "prior"))
    result.add_argument("--true-angle", type=float, default=0.6)
    result.add_argument("--seed", type=int, default=99)

    result.add_argument("--sigma-top", type=float, default=1.0)
    result.add_argument("--sigma-weights", type=float, default=0.40,
                        help="Working noise scale. Weights are invariant along "
                             "the ladder, so this should be where the score is "
                             "calibrated, not at the bottom.")
    result.add_argument("--sigma-branch", type=float, default=0.0,
                        help="Read the branch weights at this noise scale "
                             "instead. 0 disables.")
    result.add_argument("--ladder", type=int, default=16)
    result.add_argument("--newton-steps", type=int, default=12)
    result.add_argument("--refine-steps", type=int, default=4,
                        help="Newton steps after the first level. Continuation "
                             "means every later level starts next to its "
                             "answer, so the first level's budget is waste "
                             "there.")
    result.add_argument("--trust-radius", type=float, default=2.0)
    result.add_argument("--curvature-floor", type=float, default=1e-2)
    result.add_argument("--split-scale", type=float, default=3.0)
    result.add_argument("--split-directions", type=int, default=2)
    result.add_argument("--dedup-tolerance", type=float, default=1.5)
    result.add_argument("--max-modes", type=int, default=8)
    result.add_argument("--top-starts", type=int, default=12)
    result.add_argument("--level-starts", type=int, default=4)
    result.add_argument("--validate-starts", type=int, default=64)
    result.add_argument("--path-nodes", type=int, default=64)
    result.add_argument("--path-bend", type=float, default=1.0)
    result.add_argument("--deconvolve", action="store_true", default=True)
    result.add_argument("--no-deconvolve", dest="deconvolve",
                        action="store_false")
    result.add_argument("--deconvolve-floor", type=float, default=0.1)
    result.add_argument("--min-weight", type=float, default=0.0,
                        help="Drop components below this share of their own "
                             "factor's largest. Off by default and dangerous: "
                             "composition is a product, so a factor that says "
                             "1e-4 somewhere is not interchangeable with one "
                             "that says 0 there. See mode_mixture_core.")
    result.add_argument("--force-k1", action="store_true")
    result.add_argument("--keep-saddles", action="store_true")
    result.add_argument("--component-source", default="laplace",
                        choices=("laplace", "em"))
    result.add_argument("--curvature-source", default="joint",
                        choices=("joint", "conditional"),
                        help="'conditional' assembles the Hessian from the two "
                             "role masks of SIX_CONDITIONALS instead of reading "
                             "it off the all-latent mask.")
    result.add_argument("--score-assembly", default="joint",
                        choices=("joint", "conditional"),
                        help="'conditional' assembles the score itself the same "
                             "way, so its critical points are the joint's "
                             "rather than a Cartesian product.")

    result.add_argument("--draws", type=int, default=4000)
    result.add_argument("--single-draws", type=int, default=20_000)
    result.add_argument("--pilot-draws", type=int, default=1024)
    result.add_argument("--reference-draws", type=int, default=200_000)
    result.add_argument("--single-reference-draws", type=int, default=100_000)
    result.add_argument("--reference-bandwidth", type=float, default=0.05)
    result.add_argument("--kde-bandwidth", type=float, default=None)
    result.add_argument("--timesteps", type=int, default=100)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--composition-temperature", type=float, default=1.0)
    result.add_argument("--methods", nargs="*",
                        default=["gauss_jacobian", "gauss_hierarchical"])
    result.add_argument("--skip-network-product", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/mode_mixture")
    return result


def build_dataset(arguments):
    generator = sim.generator("bimodal-dataset", arguments.seed)
    if arguments.dataset == "exact_zero":
        x, _ = exact_zero_dataset(arguments.observations, arguments.true_angle,
                                  generator)
    elif arguments.dataset == "prior":
        _, _, x = H.simulate_dataset(arguments.observations, generator,
                                     angle_1=arguments.true_angle)
    else:
        x = selected_dataset(arguments.observations, arguments.true_angle,
                             arguments.select_x1, generator)
    return x


def branch_weights_up_the_ladder(score, data, arguments, settings, mixtures):
    """Weights from a second pass whose ladder stops at ``--sigma-branch``.

    Branch mass is invariant along the ladder, so it can be read wherever the
    estimate is best conditioned rather than where the components are read.
    Just below a bifurcation the branches are still close together, so the path
    joining them is short, instead of spanning a deep valley at the bottom of
    the ladder. Matched to the working mixture by rank in the shared coordinate,
    and refused if the two passes disagree about K.
    """
    fields = dict(settings.__dict__)
    fields["sigma_weights"] = arguments.sigma_branch
    ladder, _ = core.build_mixtures(score, data, PROBLEM,
                                    core.Settings(**fields), arguments.seed)
    if any(a["K"] != b["K"] for a, b in zip(mixtures, ladder)):
        return False
    for target, source in zip(mixtures, ladder):
        rank = np.argsort(np.argsort(target["mean"]))
        target["weight_branch"] = source["weight"][np.argsort(source["mean"])][rank]
    return True


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "Set CUDA_VISIBLE_DEVICES explicitly: this host is shared and "
            "autocvd would block until a GPU is idle."
        )
    inference.COMMON["timesteps"] = arguments.timesteps
    torch.manual_seed(arguments.seed)
    settings = settings_from(arguments)

    span = 4.0 * H.GLOBAL_PRIOR_STD[0]
    grid = np.linspace(-span, span, 801)
    prior = prior_density(grid)
    prior_variance = float(H.GLOBAL_PRIOR_STD[0]) ** 2

    x = build_dataset(arguments)
    data = torch.as_tensor(x, dtype=torch.float32).to(arguments.device)
    print(f"{arguments.observations} observations, dataset {arguments.dataset}, "
          f"true angle_1 = {arguments.true_angle}, seed {arguments.seed}")
    print(f"checkpoint {arguments.checkpoint.parent.name}")
    print(f"ladder {arguments.sigma_top} -> {arguments.sigma_weights} in "
          f"{arguments.ladder} levels, components from "
          f"{arguments.component_source}, "
          f"{'K forced to 1' if arguments.force_k1 else 'K discovered'}\n")

    started = time.perf_counter()
    _, joint_reference = H.reference_global(
        x, grid=grid, draws=arguments.reference_draws,
        bandwidth=arguments.reference_bandwidth, seed=arguments.seed,
        symmetrize=(arguments.dataset == "exact_zero"),
    )
    reference_moments = moments(grid, joint_reference)
    reference_mass = float(np.trapezoid(joint_reference[grid >= 0],
                                        grid[grid >= 0]))
    print(f"joint ABC reference: mean {reference_moments['mean']:+.4f} "
          f"sd {reference_moments['sd']:.4f} mass>0 {reference_mass:.3f}"
          f"  ({time.perf_counter() - started:.0f}s)")

    model = trainer.load_to(arguments.checkpoint, arguments.device)
    joint_score = core.score_function(model, PROBLEM, arguments.device)
    score = (core.assembled_score_function(model, PROBLEM, arguments.device)
             if arguments.score_assembly == "conditional" else joint_score)
    # Weights stay on the all-latent mask: the assembly is exact only at
    # sigma = 0, so above it the assembled field is not a gradient and the path
    # integral stops being path-independent.
    weight_score = joint_score

    # The network's own p(a | x_i). The grid product of these is the target
    # every composition rule aims at, and each one individually is what that
    # observation's mixture is meant to be a K-component summary *of* -- a
    # cheaper thing to diagnose than the composed answer.
    per_observation_network = single_draws = None
    if not arguments.skip_network_product:
        per_observation_network, single_draws = single_observation_posteriors(
            model, x, grid, arguments.single_draws, arguments.kde_bandwidth,
            arguments.churn_eta, arguments.seed, arguments.device,
            arguments.timesteps,
        )

    started = time.perf_counter()
    mixtures, report = core.build_mixtures(score, data, PROBLEM, settings,
                                           arguments.seed,
                                           arguments.validate_starts,
                                           weight_score=weight_score)
    if arguments.component_source == "em":
        if single_draws is None:
            raise SystemExit("--component-source em needs the draws; drop "
                             "--skip-network-product")
        mixtures = [core.refit_components(m, single_draws[i])
                    for i, m in enumerate(mixtures)]
    if arguments.sigma_branch:
        branch_weights_up_the_ladder(score, data, arguments, settings, mixtures)
    mixture_seconds = time.perf_counter() - started

    print(f"\n{'obs':>4}{'x1':>8}{'x2':>8}{'K':>4}"
          f"{'mass>0 mix':>12}{'mass>0 lad':>12}{'mass>0 draws':>14}"
          f"{'mass>0 net':>12}{'W1 vs net':>11}{'path err':>10}")
    for index, mixture in enumerate(mixtures):
        drawn = network = fidelity = float("nan")
        if per_observation_network is not None:
            mixture["weight_empirical"] = core.empirical_weights(
                mixture, single_draws[index])
            drawn = core.positive_mass(mixture, "weight_empirical")
            network = float(np.trapezoid(
                per_observation_network[index][grid >= 0], grid[grid >= 0]))
            own = core.mixture_density(mixture, grid)
            own = own / np.trapezoid(own, grid)
            fidelity = w1_grid(grid, own, per_observation_network[index]) \
                / moments(grid, per_observation_network[index])["sd"]
            mixture["fidelity_w1"] = fidelity
        mixture["mixture_positive_mass"] = core.positive_mass(mixture)
        ladder_mass = (core.positive_mass(mixture, "weight_branch")
                       if "weight_branch" in mixture else float("nan"))
        mixture["ladder_positive_mass"] = ladder_mass
        print(f"{index:>4}{x[index][0]:>+8.3f}{x[index][1]:>+8.3f}"
              f"{mixture['K']:>4}{mixture['mixture_positive_mass']:>12.3f}"
              f"{ladder_mass:>12.3f}{drawn:>14.3f}{network:>12.3f}"
              f"{fidelity:>11.3f}{mixture['path_discrepancy']:>10.3f}")
        order = np.argsort(mixture["mean"])
        print(f"{'':>4}mean [{' '.join(f'{v:+.3f}' for v in mixture['mean'][order])}]"
              f"  sd [{' '.join(f'{v:.3f}' for v in np.sqrt(mixture['variance'][order]))}]"
              f"  w [{' '.join(f'{v:.3f}' for v in mixture['weight'][order])}]")

    composed, table = core.compose_gaussian_prior(mixtures, grid, prior_variance)
    print(f"\nmixture: {mixture_seconds:.1f}s, "
          f"{sum(m['K'] for m in mixtures)} components, "
          f"{table['components']}/{table['tuples']:.3g} tuples "
          f"({table['improper']} improper"
          f"{', beam-pruned' if table['pruned'] else ''}), "
          f"effective {table['effective_components']:.1f}")
    print(f"multi-start check at sigma={arguments.sigma_weights} "
          f"({arguments.validate_starts} starts): K = {report['validation']}")
    print(f"continuation                                     K = "
          f"{report['continuation']}")

    rows = {"mode_mixture": composed}
    if all("weight_branch" in m for m in mixtures):
        rows["mode_mixture_ladder_w"], _ = core.compose_gaussian_prior(
            mixtures, grid, prior_variance, "weight_branch")
    if single_draws is not None:
        rows["mode_mixture_drawn_w"], _ = core.compose_gaussian_prior(
            mixtures, grid, prior_variance, "weight_empirical")
        rows["network_product"] = grid_product(grid, per_observation_network,
                                               prior)
    for method in arguments.methods:
        samples, _, seconds = inference.draw(model, data.cpu(), method,
                                             arguments, arguments.device)
        shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
        rows[method] = kde_density(shared, grid, arguments.kde_bandwidth)
        print(f"  {method} sampled in {seconds:.0f}s")

    summary = {name: compare(grid, density, joint_reference, reference_moments)
               for name, density in rows.items()}
    print(f"\n{'curve':>22}{'mean':>9}{'sd':>8}{'mean err/sig':>14}"
          f"{'width':>8}{'W1/sig':>9}{'mass>0':>9}")
    print(f"{'joint ABC reference':>22}{reference_moments['mean']:>+9.4f}"
          f"{reference_moments['sd']:>8.4f}{0.0:>14.3f}{1.0:>8.3f}{0.0:>9.3f}"
          f"{reference_mass:>9.3f}")
    for name, entry in summary.items():
        print(f"{name:>22}{entry['mean']:>+9.4f}{entry['sd']:>8.4f}"
              f"{entry['mean_error_sigma']:>+14.3f}{entry['width_ratio']:>8.3f}"
              f"{entry['w1_over_sigma']:>9.3f}{entry['positive_mass']:>9.3f}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output / "curves.npz", grid=grid, prior=prior,
        joint_reference=joint_reference, observations=x,
        **{f"curve_{k}": v for k, v in rows.items()},
    )
    (arguments.output / "metrics.json").write_text(json.dumps({
        "checkpoint": str(arguments.checkpoint),
        "dataset": arguments.dataset,
        "observations": int(arguments.observations),
        "seed": int(arguments.seed),
        "sigma_weights": arguments.sigma_weights,
        "sigma_branch": arguments.sigma_branch,
        "component_source": arguments.component_source,
        "curvature_source": arguments.curvature_source,
        "score_assembly": arguments.score_assembly,
        "force_k1": bool(arguments.force_k1),
        "mixture_seconds": mixture_seconds,
        "reference": reference_moments,
        "reference_positive_mass": reference_mass,
        "curves": summary,
        "multistart_K": report["validation"],
        "continuation_K": report["continuation"],
        "antisymmetry": report["antisymmetry"],
        "coupling": report["coupling"],
        "mask_disagreement": report["mask_disagreement"],
        "trace": report["trace"],
        "tuples": {k: table[k] for k in
                   ("components", "tuples", "pruned", "improper",
                    "effective_components")},
        "per_observation": [
            {
                "observation": i, "x": x[i].tolist(), "K": m["K"],
                "mean": m["mean"].tolist(),
                "variance": m["variance"].tolist(),
                "weight": m["weight"].tolist(),
                "path_discrepancy": m["path_discrepancy"],
                "fidelity_w1": m.get("fidelity_w1"),
                "mixture_positive_mass": m.get("mixture_positive_mass"),
                "ladder_positive_mass": m.get("ladder_positive_mass"),
            }
            for i, m in enumerate(mixtures)
        ],
    }, indent=2, sort_keys=True))
    print(f"\nWrote {arguments.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
