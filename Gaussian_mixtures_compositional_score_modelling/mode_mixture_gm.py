#!/usr/bin/env python3
"""Mode-mixture composition against the *exact* diffused score.

On the arm, mode-mixture composition beats the Gaussian rules 1.3-2.6x but only
once the component widths are refit to the network's own draws; taken from the
score Jacobian they come out 1.5-1.9x too wide and the ``prior^(1-n)`` division
turns that into a catastrophe (``docs/mode_mixture.md``). The natural suspect is
the network -- a shrunk score is a flattened log density is an inflated variance,
and ``docs/score_calibration.md`` measures exactly that shrinkage.

This benchmark settles it, because here the score is not learned.
``oracle.OracleScore`` is the exact diffused score at every noise level, and the
tall posterior is a closed-form two-component Gaussian mixture at every ``n``. So:

* if the Laplace variant is accurate here, the arm's failure is the network's
  and the method is sound;
* if it fails here too, the failure is the method's and no amount of training
  fixes it.

The test is sharper than "an exact score instead of an approximate one". Every
diffused conditional in this model **is** a two-component Gaussian mixture,
exactly, so the representation the method assumes is not an approximation at all:
the modes are the component means, the Jacobian at a mode is the exact component
precision, and the score is an exact gradient, so the path integral is exactly
path-independent. Anything the method gets wrong here it gets wrong for reasons
that have nothing to do with score quality.

Usage:
    python Gaussian_mixtures_compositional_score_modelling/mode_mixture_gm.py \
        --regime asymmetric --observations 8 --branch-mass 0.75
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

# The core is shared with the arm rather than reimplemented, so the two
# benchmarks cannot drift apart; it is problem-independent and takes everything
# it needs to know through ``core.Problem``.
sys.path.insert(
    0, str(ROOT.parent / "inverse_kinematics" / "experiments"))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import datasets  # noqa: E402
import methods  # noqa: E402
import metrics  # noqa: E402
import mode_mixture_core as core  # noqa: E402
import model as M  # noqa: E402
import oracle  # noqa: E402
import train  # noqa: E402

LABELS = {
    "mode_mixture": "mode mixture (K discovered)",
    "marginal_product": "marginal product (network's own factors)",
    "mode_mixture_k1": "mode mixture, K = 1 control",
    "gauss_jacobian": "gauss_jacobian",
    "gauss_hierarchical": "gauss_hierarchical",
    "langevin_fnpse": "Langevin + F-NPSE",
}
COLOURS = {
    "mode_mixture": "#F4A300",
    "marginal_product": "#9C6644",
    "mode_mixture_k1": "#8D99AE",
    "gauss_jacobian": "#5E60CE",
    "gauss_hierarchical": "#E76F51",
    "langevin_fnpse": "#2A9D8F",
}
INK = "#22262B"
SOURCE_TITLE = {"oracle": "exact diffused score",
                "network": "learned score"}


def problem_for(mixture, dtype=torch.float64):
    """``core.Problem`` for this benchmark: latent ``(g, l)``, shared ``g``.

    The prior moments are Newton *start* moments only. The moment-matched
    ``global_prior_std`` is the right choice here precisely because it is wide
    (2.53 against a within-component 0.4): starts have to be able to land in
    either branch, and the composition prior is supplied separately and exactly.
    """
    return core.Problem(
        latent=len(M.LATENT_INDICES),
        shared=M.HIERARCHY[0],
        condition_mask=torch.tensor(
            [0.0] * len(M.LATENT_INDICES) + [1.0] * len(M.OBSERVED_INDICES),
            dtype=dtype),
        prior_mean=np.array([mixture.global_prior_mean,
                             mixture.local_prior_mean]),
        prior_std=np.array([mixture.global_prior_std,
                            mixture.local_prior_std]),
    )


def grid_w1(grid, left, right):
    """W1 between two densities tabulated on a shared grid."""
    def cdf(density):
        value = np.concatenate([[0.0], np.cumsum(
            0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
        return value / value[-1]
    return float(np.trapezoid(np.abs(cdf(left) - cdf(right)), grid))


def density_moments(grid, density):
    mean = float(np.trapezoid(grid * density, grid))
    variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    return mean, float(np.sqrt(max(variance, 0.0)))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--regime", default="asymmetric",
                        choices=datasets.REGIMES)
    result.add_argument("--observations", type=int, default=8)
    result.add_argument("--branch-mass", type=float, default=0.75)
    result.add_argument("--spread", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--score-source", default="oracle",
                        choices=("oracle", "network"),
                        help="'oracle' is the exact diffused score; 'network' "
                             "is the trained checkpoint named by the recipe "
                             "flags below. Everything else is held fixed, so "
                             "the difference between the two runs is the "
                             "network and nothing else.")
    result.add_argument("--config", default="h32d3wide")
    result.add_argument("--arch", default="tokenized")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--mask-menu", default="six_weighted")
    result.add_argument("--time-sampling", default="log_sigma")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    result.add_argument("--overlay", type=Path, default=None,
                        help="A curves.npz from the other score source; its "
                             "mode_mixture curve is drawn for reference.")

    result.add_argument("--sigma-top", type=float, default=6.0,
                        help="Top of the ladder. The prior modes are 2*mu_0 = 5 "
                             "apart, so this has to exceed that comfortably for "
                             "the K = 1 start to be correct; the recorded mode "
                             "count at the first level is the check.")
    result.add_argument("--sigma-weights", type=float, default=0.20)
    result.add_argument("--ladder", type=int, default=16)
    result.add_argument("--newton-steps", type=int, default=12)
    result.add_argument("--refine-steps", type=int, default=6)
    result.add_argument("--trust-radius", type=float, default=2.0)
    result.add_argument("--curvature-floor", type=float, default=1e-2)
    result.add_argument("--split-scale", type=float, default=3.0)
    result.add_argument("--split-directions", type=int, default=2)
    result.add_argument("--dedup-tolerance", type=float, default=1.5)
    result.add_argument("--max-modes", type=int, default=6)
    result.add_argument("--top-starts", type=int, default=12)
    result.add_argument("--level-starts", type=int, default=4)
    result.add_argument("--validate-starts", type=int, default=64)
    result.add_argument("--path-nodes", type=int, default=64)
    result.add_argument("--path-bend", type=float, default=1.0)
    result.add_argument("--deconvolve", action="store_true", default=True)
    result.add_argument("--no-deconvolve", dest="deconvolve",
                        action="store_false")
    result.add_argument("--deconvolve-floor", type=float, default=0.1)
    result.add_argument("--min-weight", type=float, default=0.0)
    result.add_argument("--prior-composition", default="mixture",
                        choices=("mixture", "grid"),
                        help="'mixture' composes within each prior component, "
                             "where the prior is a single Gaussian; 'grid' "
                             "divides by the mixture prior on the grid, which "
                             "puts a trap between the prior modes.")
    result.add_argument("--score-assembly", default="joint",
                        choices=("joint", "conditional"),
                        help="'joint' evaluates the all-latent mask; "
                             "'conditional' assembles each component from the "
                             "role mask where it is latent. Only the second "
                             "gives a score whose critical points are the "
                             "joint's rather than the Cartesian product of the "
                             "per-coordinate ones.")
    result.add_argument("--curvature-source", default="joint",
                        choices=("joint", "conditional"),
                        help="'joint' reads the Hessian from the all-latent "
                             "mask; 'conditional' assembles it from the two "
                             "role masks of SIX_CONDITIONALS, which is the only "
                             "route that recovers latent coupling under a "
                             "per-token-normalising trunk.")

    result.add_argument("--draws", type=int, default=4_000)
    result.add_argument("--pilot-draws", type=int, default=1_000)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--churn-eta", type=float, default=4.0)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--methods", nargs="*",
                        default=["gauss_jacobian", "gauss_hierarchical",
                                 "langevin_fnpse"])
    result.add_argument("--sigma-sweep", type=float, nargs="*",
                        default=[0.40, 0.20, 0.10, 0.05],
                        help="Rebuild the mixture at each working sigma. Branch "
                             "mass is invariant along the ladder and the "
                             "components are deconvolved, so a result that "
                             "moves with this is a defect, not a setting.")
    result.add_argument("--device", default="cpu",
                        help="The oracle is closed form and tiny; cpu is the "
                             "sane default and needs no GPU reservation.")
    result.add_argument("--output", type=Path, default=None,
                        help="Defaults to artifacts/mode_mixture/<source>.")
    return result


def settings_from(arguments, sigma_weights=None):
    return core.Settings(
        sigma_top=arguments.sigma_top,
        sigma_weights=(arguments.sigma_weights if sigma_weights is None
                       else sigma_weights),
        ladder=arguments.ladder, newton_steps=arguments.newton_steps,
        refine_steps=arguments.refine_steps, trust_radius=arguments.trust_radius,
        curvature_floor=arguments.curvature_floor,
        split_scale=arguments.split_scale,
        split_directions=arguments.split_directions,
        dedup_tolerance=arguments.dedup_tolerance,
        max_modes=arguments.max_modes, top_starts=arguments.top_starts,
        level_starts=arguments.level_starts, path_nodes=arguments.path_nodes,
        path_bend=arguments.path_bend, deconvolve=arguments.deconvolve,
        deconvolve_floor=arguments.deconvolve_floor,
        min_weight=arguments.min_weight,
        curvature_source=arguments.curvature_source,
    )


def run_mixture(score, data, problem, arguments, sigma_weights, grid, prior,
                force_k1=False, validate=0, weight_score=None):
    fields = dict(settings_from(arguments, sigma_weights).__dict__)
    fields["force_k1"] = force_k1
    settings = core.Settings(**fields)
    started = time.perf_counter()
    mixtures, report = core.build_mixtures(score, data, problem, settings,
                                           arguments.seed, validate,
                                           weight_score=weight_score)
    if arguments.prior_composition == "mixture":
        density, detail = core.compose_mixture_prior(
            mixtures, grid, M.DEFAULT.weights, M.DEFAULT.components,
            np.full(2, M.DEFAULT.tau_g ** 2))
        detail["tail_ratio"] = float("nan")
    else:
        density, detail = core.compose_grid_prior(mixtures, grid, prior)
    report["seconds"] = time.perf_counter() - started
    report["tail_ratio"] = detail["tail_ratio"]
    return density, mixtures, report


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.device.startswith("cuda") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly on this host.")

    mixture = M.DEFAULT
    if arguments.output is None:
        arguments.output = ROOT / "artifacts/mode_mixture" / arguments.score_source
    dtype = torch.float64 if arguments.score_source == "oracle" else torch.float32
    problem = problem_for(mixture, dtype)
    generator = np.random.default_rng(arguments.seed)
    dataset = datasets.make_dataset(arguments.regime, arguments.observations,
                                    generator, mixture, arguments.branch_mass,
                                    arguments.spread)
    x = dataset["x"]
    data = torch.as_tensor(x.reshape(-1, 1), dtype=dtype,
                           device=arguments.device)

    grid = mixture.default_grid(x, points=4001)
    exact = mixture.global_density(grid, x)
    exact_weights = mixture.branch_weights(x)
    prior = mixture.prior_density(grid)
    exact_mean, exact_sd = density_moments(grid, exact)

    print(f"{arguments.regime}, n = {arguments.observations}, "
          f"seed {arguments.seed}")
    print(f"exact p(g | x): mean {exact_mean:+.4f} sd {exact_sd:.4f}, "
          f"branch weights {exact_weights[0]:.3f} / {exact_weights[1]:.3f}")
    if arguments.score_source == "oracle":
        network = oracle.OracleModel(mixture, sigma=train.SDE_KWARGS["sigma"],
                                     device=arguments.device)
        print("score: oracle (exact diffused score at every noise level)\n")
    else:
        directory = train.model_directory(
            arguments.artifacts, arguments.config, arguments.arch,
            arguments.simulations, arguments.mask_menu,
            arguments.time_sampling)
        checkpoint = directory / "Model_checkpoint.pt"
        if not checkpoint.exists():
            raise SystemExit(f"No checkpoint at {checkpoint}; train one first.")
        network = train.load_to(checkpoint, arguments.device)
        parameters = sum(p.numel() for p in network.model.parameters())
        print(f"score: learned, {directory.name} ({parameters:,} parameters)\n")
    joint_score = core.score_function(network, problem, arguments.device)
    score = (core.assembled_score_function(network, problem, arguments.device)
             if arguments.score_assembly == "conditional" else joint_score)
    # Modes and curvature from the assembly, branch weights from the all-latent
    # mask: see build_mixtures for why the two cannot be the same field.
    weight_score = joint_score

    # --- the method, and the sweep that says whether it is a method ---------
    curves, records = {}, {}
    print(f"{'sigma_w':>9}{'K':>22}{'branch w':>18}{'path err':>10}"
          f"{'couple':>9}{'mask dis':>9}{'W1/sd':>8}{'sec':>7}")
    best = None
    for sigma_weights in arguments.sigma_sweep:
        density, mixtures, report = run_mixture(
            score, data, problem, arguments, sigma_weights, grid, prior,
            validate=(arguments.validate_starts
                      if sigma_weights == arguments.sigma_weights else 0),
            weight_score=weight_score)
        w1 = grid_w1(grid, density, exact) / exact_sd
        mass = float(np.trapezoid(density[grid >= 0], grid[grid >= 0]))
        path = max(m["path_discrepancy"] for m in mixtures)
        print(f"{sigma_weights:>9.2f}{str(report['continuation']):>22}"
              f"{f'{1 - mass:.3f} / {mass:.3f}':>18}{path:>10.2e}"
              f"{report['coupling']:>9.2e}{report['mask_disagreement']:>9.2e}"
              f"{w1:>8.3f}{report['seconds']:>7.1f}")
        records[f"sigma_{sigma_weights}"] = {
            "sigma_weights": sigma_weights, "w1_over_sd": w1,
            "positive_mass": mass, "path_discrepancy": path,
            "antisymmetry": report["antisymmetry"],
            "coupling": report["coupling"],
            "mask_disagreement": report["mask_disagreement"],
            "tail_ratio": report["tail_ratio"],
            "seconds": report["seconds"],
            "continuation_K": report["continuation"],
            "multistart_K": report["validation"],
            "components": [
                {"mean": m["mean"].tolist(),
                 "sd": np.sqrt(m["variance"]).tolist(),
                 "weight": m["weight"].tolist()} for m in mixtures
            ],
        }
        if sigma_weights == arguments.sigma_weights:
            curves["mode_mixture"] = density
            best = (mixtures, report)

    if "mode_mixture" not in curves:
        raise SystemExit("--sigma-weights must appear in --sigma-sweep")
    mixtures, report = best
    print(f"\nmulti-start check at sigma={arguments.sigma_weights} "
          f"({arguments.validate_starts} starts): K = {report['validation']}")
    print(f"continuation                                       "
          f"K = {report['continuation']}")

    # The internal control: one component per observation, everything else the
    # same. On the arm this reproduced the Gaussian rules' failure.
    curves["mode_mixture_k1"], _, k1_report = run_mixture(
        score, data, problem, arguments, arguments.sigma_weights, grid, prior,
        force_k1=True, weight_score=weight_score)

    # --- the rules ---------------------------------------------------------
    seconds = {"mode_mixture": report["seconds"],
               "mode_mixture_k1": k1_report["seconds"]}
    draws = {}
    for method in arguments.methods:
        samples, taken, _ = methods.draw(network, x, method, arguments,
                                         arguments.device, mixture)
        shared = np.asarray(samples[0, :, M.HIERARCHY[0]], dtype=np.float64)
        draws[method] = shared
        curves[method] = methods._normalise(
            grid, gaussian_kde(shared)(grid))
        seconds[method] = taken
        print(f"  {method} sampled in {taken:.0f}s")

    # W1 by quantile matching against the exact CDF, the grader this directory
    # uses, so the mixture is judged on the same footing as the samplers: its
    # composed density is turned into draws by CDF inversion.
    sample_rng = np.random.default_rng(arguments.seed + 1)
    for name in ("mode_mixture", "mode_mixture_k1"):
        draws[name] = methods.inverse_cdf_sample(grid, curves[name],
                                                 arguments.draws, sample_rng)

    print(f"\n{'curve':>28}{'mean':>9}{'sd':>8}{'W1/sd grid':>12}"
          f"{'W1 draws':>10}{'w_+':>8}{'sec':>8}")
    print(f"{'exact p(g | x_1..n)':>28}{exact_mean:>+9.4f}{exact_sd:>8.4f}"
          f"{0.0:>12.3f}{0.0:>10.3f}{exact_weights[1]:>8.3f}{0.0:>8.1f}")
    summary = {}
    for name, density in curves.items():
        mean, sd = density_moments(grid, density)
        entry = {
            "mean": mean, "sd": sd,
            "w1_over_sd_grid": grid_w1(grid, density, exact) / exact_sd,
            "w1_draws": metrics.wasserstein(draws[name], mixture, x),
            "positive_mass": float(np.trapezoid(density[grid >= 0],
                                                grid[grid >= 0])),
            "seconds": seconds[name],
        }
        summary[name] = entry
        print(f"{LABELS[name]:>28}{mean:>+9.4f}{sd:>8.4f}"
              f"{entry['w1_over_sd_grid']:>12.3f}{entry['w1_draws']:>10.3f}"
              f"{entry['positive_mass']:>8.3f}{entry['seconds']:>8.1f}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    overlay = None
    if arguments.overlay is not None and arguments.overlay.exists():
        stored = np.load(arguments.overlay)
        if not np.allclose(stored["grid"], grid):
            raise SystemExit("--overlay was computed on a different grid")
        overlay = stored["curve_mode_mixture"]
    figure = plot(grid, exact, curves, exact_weights, summary, dataset,
                  arguments, overlay)
    np.savez_compressed(arguments.output / "curves.npz", grid=grid,
                        exact=exact, prior=prior, x=x,
                        **{f"curve_{k}": v for k, v in curves.items()})
    (arguments.output / "metrics.json").write_text(json.dumps({
        "regime": arguments.regime,
        "observations": int(arguments.observations),
        "branch_mass": arguments.branch_mass,
        "seed": int(arguments.seed),
        "score_source": arguments.score_source,
        "curvature_source": arguments.curvature_source,
        "score_assembly": arguments.score_assembly,
        "prior_composition": arguments.prior_composition,
        "sigma_weights": arguments.sigma_weights,
        "exact": {"mean": exact_mean, "sd": exact_sd,
                  "branch_weights": exact_weights.tolist()},
        "curves": summary,
        "sigma_sweep": records,
    }, indent=2, sort_keys=True))
    print(f"\nWrote {arguments.output / 'metrics.json'}")
    print(f"Wrote {figure}")
    return 0


def plot(grid, exact, curves, exact_weights, summary, dataset, arguments,
         overlay=None):
    """The global parameter posterior, and the branch split that drives it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["mode_mixture", "mode_mixture_k1"] + [
        name for name in curves if name not in
        ("mode_mixture", "mode_mixture_k1")]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [2.05, 1.0]})

    # The exact curve is drawn as a wide pale band rather than a line: the
    # mode mixture lies on top of it to within the line width, and a thin black
    # line under a thin orange one just reads as "the orange one is missing".
    left.plot(grid, exact, color=INK, linewidth=7.0, alpha=0.22, zorder=1,
              solid_capstyle="round",
              label="exact  $p(g \\mid x_{1:n})$")
    ceiling = float(exact.max()) * 1.45
    for name in order:
        clipped = np.minimum(curves[name], ceiling * 1.02)
        left.plot(grid, clipped, color=COLOURS[name], linewidth=2.0,
                  linestyle="--" if name == "mode_mixture_k1" else "-",
                  alpha=0.95, zorder=6 if name == "mode_mixture" else 3,
                  label=f"{LABELS[name]}  (W1/sd {summary[name]['w1_over_sd_grid']:.3f})")
    if overlay is not None:
        other = "exact" if arguments.score_source == "network" else "learned"
        left.plot(grid, np.minimum(overlay, ceiling * 1.02), color="#B5179E",
                  linewidth=1.6, linestyle=(0, (5, 2)), alpha=0.9, zorder=4,
                  label=f"mode mixture, {other} score (reference)")
    left.plot(dataset["x"], np.full(len(dataset["x"]), -0.02 * ceiling), "|",
              color=INK, alpha=0.55, markersize=9, markeredgewidth=1.4,
              zorder=7, label="observations $x_i$")
    left.set_ylim(-0.05 * ceiling, ceiling)
    clipped = {name: float(curves[name].max()) for name in order
               if float(curves[name].max()) > ceiling}
    if clipped:
        # Say which curves are cut off, not just the loudest one: a reader who
        # cannot tell a clipped curve from a wall will read the wrong shape.
        listing = "\n".join(f"{LABELS[name]}: peaks at {height:.0f}"
                            for name, height in sorted(clipped.items(),
                                                       key=lambda e: -e[1]))
        left.annotate(
            "clipped at the top of the axis —\n" + listing,
            xy=(0.985, 0.035), xycoords="axes fraction", ha="right",
            va="bottom", fontsize=8.0, color="#6B7280", linespacing=1.35)
    left.set_xlabel("global parameter $g$")
    left.set_ylabel("posterior density")
    left.set_title(
        f"Global posterior, {SOURCE_TITLE[arguments.score_source]}  —  "
        f"{dataset['regime']}, n = {dataset['observations']}",
        loc="left", fontsize=11)
    left.legend(frameon=False, fontsize=8.5, loc="upper left")
    left.spines[["top", "right"]].set_visible(False)

    # The branch split is the quantity a single Gaussian per observation cannot
    # carry, so it gets its own panel rather than being read off the curves.
    names = ["exact"] + order
    values = [exact_weights[1]] + [summary[n]["positive_mass"] for n in order]
    colours = [INK] + [COLOURS[n] for n in order]
    positions = np.arange(len(names))
    right.barh(positions, values, color=colours, alpha=0.9, height=0.62)
    right.axvline(exact_weights[1], color=INK, linewidth=1.1, linestyle=":",
                  zorder=4)
    for position, value in zip(positions, values):
        right.text(value + 0.015, position, f"{value:.3f}", va="center",
                   fontsize=9, color=INK)
    right.set_yticks(positions)
    right.set_yticklabels(["exact"] + [LABELS[n] for n in order], fontsize=8.5)
    right.invert_yaxis()
    right.set_xlim(0, 1.16)
    right.set_xlabel("posterior mass on the $+$ branch")
    right.set_title("Branch weight", loc="left", fontsize=11)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    path = (arguments.output /
            f"global_posterior_{arguments.score_source}_{dataset['regime']}"
            f"_n{dataset['observations']}.png")
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
