#!/usr/bin/env python3
"""The exponential-tilt composition on all three noise twins of the hierarchy.

Why this is the cleanest test bed in the project. The shared posterior of each
twin is available in **closed form** (``module.shared_reference``), and the
score can be supplied **exactly** by ``row_scores.AnalyticModel`` rather than by
a trained network. So neither a Monte-Carlo referee nor network error is in the
loop, and whatever error a composition rule shows is its own. The three twins
then separate the two ways a problem can be non-Gaussian:

``gauss``        Normal noise. Symmetric, no wall, and Gaussian -- so the
                 Gaussian/Tweedie approximation every composition rule rests on
                 is *exact at every noise level*. The control: everything should
                 work, and the tilt at degree 2 contains the truth.
``laplace``      Laplace noise. Symmetric and no wall but non-Gaussian, so the
                 rules stay approximate. Isolates non-Gaussianity from skew.
``exponential``  The original: sharply skewed, hard wall at ``min_j x_j``,
                 posterior roughly ``1/N`` wide. Isolates the skew and the wall.

The tilt is run with the configuration recommended from the inverse-kinematics
study: stage-1 churn 0, 3,000 stage-1 draws, 40 diffusion steps, degree chosen
per observation by BIC. Against it, the two Gaussian rules and Langevin F-NPSE
at the same step count and draw count.

Metrics, all reference-free or exact:
    w1_over_sigma   W1 to the closed-form posterior, in its sd
    map_error       |argmax of the estimate - argmax of the exact posterior|
    truth_error     |argmax of the estimate - the generating g|
    mass_at_truth   density at the generating g, over the estimate's peak

Usage:
    python tilt_noise_comparison.py --problem gauss laplace exponential
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
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent.parent.parent
                       / "inverse_kinematics" / "experiments"))

import hierarchy  # noqa: E402
import hierarchy_gauss  # noqa: E402
import hierarchy_laplace  # noqa: E402
import symmetric_row_scores as row_scores  # noqa: E402
import tilt_fit  # noqa: E402
from compass import Sampler  # noqa: E402
from compass.MultiObsSampler import MultiObsSampler  # noqa: E402
from compass.SDE import VESDE  # noqa: E402

PROBLEMS = {
    "gauss": (hierarchy_gauss, row_scores.GaussianRowScoreNetwork,
              "Gaussian noise (symmetric, Gaussian: rules exact)"),
    "laplace": (hierarchy_laplace, row_scores.LaplaceRowScoreNetwork,
                "Laplace noise (symmetric, non-Gaussian)"),
    "exponential": (hierarchy, row_scores.ExponentialRowScoreNetwork,
                    "Exponential noise (skewed, hard wall)"),
}

# The two Gaussian rules and Langevin, corrector-free with churn, which is the
# corrector-free optimum this directory established.
RULES = {
    "gauss_hierarchical": {
        "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
        "precision_est_samples": 4096, "corrector_steps": 0,
        "final_corrector_steps": 0, "churn_eta": 4.0,
    },
    "gauss_jacobian": {
        "method": "dpm", "order": 2, "correction": "gauss_jacobian",
        "corrector_steps": 0, "final_corrector_steps": 0, "churn_eta": 4.0,
    },
    "langevin_fnpe": {
        "method": "langevin", "correction": "fnpe",
        "corrector_steps": 0, "final_corrector_steps": 0, "churn_eta": 0.0,
    },
}


@contextlib.contextmanager
def patched_problem(module):
    """Swap the problem module into every place that imported ``hierarchy``.

    ``MultiObsSampler`` never sees it, but ``row_scores`` and the reference
    helpers do, and a twin must not be graded against the original's support.
    """
    import symmetric_row_scores as rs

    saved = getattr(rs, "hierarchy", None)
    rs.hierarchy = module
    try:
        yield
    finally:
        if saved is not None:
            rs.hierarchy = saved


def statistics(grid, density, reference, truth):
    """Everything reported, all against exact objects."""
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        return None
    density = density / area
    reference = reference / np.trapezoid(reference, grid)
    step = np.diff(grid)
    left = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * step)])
    right = np.concatenate([[0.0], np.cumsum(0.5 * (reference[1:] + reference[:-1]) * step)])
    left /= left[-1]
    right /= right[-1]
    mean = float(np.trapezoid(grid * reference, grid))
    sigma = float(np.sqrt(max(float(np.trapezoid(grid ** 2 * reference, grid))
                              - mean ** 2, 1e-12)))
    peak = float(grid[int(np.argmax(density))])
    reference_peak = float(grid[int(np.argmax(reference))])
    own_mean = float(np.trapezoid(grid * density, grid))
    own_sd = float(np.sqrt(max(float(np.trapezoid(grid ** 2 * density, grid))
                               - own_mean ** 2, 0.0)))
    return {
        "w1_over_sigma": float(np.trapezoid(np.abs(left - right), grid)) / sigma,
        "map": peak,
        "map_error": abs(peak - reference_peak),
        "truth_error": abs(peak - float(truth)),
        "mass_at_truth": float(np.interp(truth, grid, density) / density.max()),
        "mean_error_sigma": abs(own_mean - mean) / sigma,
        "width_ratio": own_sd / sigma,
    }


def kde_density(draws, grid, bandwidth=None):
    from scipy.stats import gaussian_kde
    return gaussian_kde(np.asarray(draws, dtype=np.float64),
                        bw_method=bandwidth)(grid)


def run_tilt(model, module, x, grid, arguments, device, exact_stage1=False):
    """Stage 1 + tilt fit + compose. No stage 3: the shared posterior is the ask.

    ``exact_stage1`` takes the single-observation draws from the problem
    module's own validated ``sample_diffused`` at ``lam = 0`` instead of running
    the diffusion sampler on the analytic score. Both arms are worth having:
    with the sampler this is the deployed method on an exact score, and with
    exact draws it is the composition rule alone, with neither network nor
    sampler in the loop.
    """
    observations = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1, 1),
                                   dtype=torch.float32)
    n = observations.shape[0]

    def fit_domain(draws):
        """A quadrature grid covering the *factors*, then report on ``grid``.

        The reference grid is sized for the composed posterior, which is about
        sqrt(n) narrower than the single-observation posteriors the tilt fits.
        Fitting on it truncates every factor. So the fit uses its own grid --
        wide enough for the draws and the prior, fine enough to resolve a
        posterior sqrt(n) narrower than they are -- and the composed density is
        interpolated onto the reference grid afterwards.
        """
        spread = float(draws.std())
        low = min(float(draws.min()) - 2.0 * spread,
                  module.MU_G - 5.0 * module.SIGMA_G, float(grid[0]))
        high = max(float(draws.max()) + 2.0 * spread,
                   module.MU_G + 5.0 * module.SIGMA_G, float(grid[-1]))
        points = int(arguments.fit_grid_points)
        wide = np.linspace(low, high, points)
        prior = -0.5 * ((wide - module.MU_G) / module.SIGMA_G) ** 2
        prior -= np.log(np.trapezoid(np.exp(prior), wide))
        return wide, prior

    started = time.perf_counter()
    if exact_stage1:
        drawn = model.sampler.sample(
            data=observations, num_samples=int(arguments.single_draws),
            device=device)
        if isinstance(drawn, torch.Tensor):
            drawn = drawn.detach().cpu()
        single = np.asarray(drawn, dtype=np.float64)
        shared_draws = single[:, :, module.GLOBAL_INDEX]
        wide, prior = fit_domain(shared_draws)
        degrees = tuple(int(v) for v in str(arguments.tilt_degrees).split(","))
        thetas, chosen, centre, scale, _ = tilt_fit.fit_all(
            shared_draws, wide, prior, degrees=degrees)
        composed = tilt_fit.compose(thetas, wide, prior, centre, scale)
        density = np.interp(grid, wide, composed)
        return density, time.perf_counter() - started, [int(v) for v in chosen]
    sampler = Sampler(model)
    pieces = []
    block = int(arguments.stage1_chunk)
    for start in range(0, n, block):
        torch.manual_seed(int(arguments.seed) + 7919 * start)
        torch.cuda.manual_seed_all(int(arguments.seed) + 7919 * start)
        drawn = sampler.sample(
            world_size=1, data=observations[start:start + block],
            num_samples=int(arguments.single_draws),
            condition_mask=module.CONDITION_MASK,
            timesteps=int(arguments.timesteps), method="dpm", order=2,
            churn_eta=float(arguments.stage1_churn), corrector_steps=0,
            final_corrector_steps=0, terminal_denoise=True, device=device,
            verbose=False,
            # The analytic row-score modules take (x, t, c) only, and Sampler's
            # attention-capture branch calls the network with
            # return_attn_weights=True. That branch is opt-out, so declining it
            # is all that is needed -- no wrapper around the score module.
            capture_attention=False,
        )
        if isinstance(drawn, torch.Tensor):
            drawn = drawn.detach().cpu()
        pieces.append(np.asarray(drawn, dtype=np.float64))
    single = np.concatenate(pieces, axis=0)
    shared_draws = single[:, :, module.GLOBAL_INDEX].astype(np.float64)
    wide, prior = fit_domain(shared_draws)
    degrees = tuple(int(v) for v in str(arguments.tilt_degrees).split(","))
    thetas, chosen, centre, scale, _ = tilt_fit.fit_all(
        shared_draws, wide, prior, degrees=degrees)
    composed = tilt_fit.compose(thetas, wide, prior, centre, scale)
    density = np.interp(grid, wide, composed)
    return density, time.perf_counter() - started, [int(v) for v in chosen]


def run_rule(model, module, x, grid, name, arguments, device):
    data = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1, 1),
                           dtype=torch.float32)
    sampler = MultiObsSampler(model)
    torch.manual_seed(int(arguments.seed))
    torch.cuda.manual_seed_all(int(arguments.seed))
    started = time.perf_counter()
    samples = sampler.sample(
        world_size=1, data=data, condition_mask=module.CONDITION_MASK,
        timesteps=int(arguments.timesteps), num_samples=int(arguments.draws),
        hierarchy=list(module.HIERARCHY),
        prior=(torch.tensor([module.MU_G]), torch.tensor([module.SIGMA_G])),
        local_prior=(torch.tensor([module.LOCAL_PRIOR_MEAN]),
                     torch.tensor([module.LOCAL_PRIOR_STD])),
        denoise_clamp=float(arguments.denoise_clamp), device=device,
        verbose=False, **RULES[name],
    )
    seconds = time.perf_counter() - started
    draws = samples.detach().cpu().numpy()[0, :, module.GLOBAL_INDEX]
    return kde_density(draws, grid), seconds


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--problem", nargs="+", default=list(PROBLEMS),
                        choices=list(PROBLEMS))
    result.add_argument("--observations", type=int, default=30)
    result.add_argument("--draws", type=int, default=2000)
    result.add_argument("--single-draws", type=int, default=3000)
    result.add_argument("--stage1-churn", type=float, default=0.0)
    result.add_argument("--stage1-chunk", type=int, default=16)
    result.add_argument("--timesteps", type=int, default=40)
    result.add_argument("--tilt-degrees", default="2,4,6,8,10,12")
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--quadrature-nodes", type=int, default=513)
    result.add_argument("--reference-points", type=int, default=40001)
    result.add_argument("--fit-grid-points", type=int, default=120001,
                        help="Quadrature points for the tilt's own grid, which "
                             "must span the single-observation posteriors while "
                             "resolving a composed posterior sqrt(n) narrower.")
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--device", default="cuda")
    result.add_argument("--methods", nargs="+",
                        default=["tilt", "tilt_exact_stage1"] + list(RULES))
    result.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "tilt_noise")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows, stored = [], {}

    for name in arguments.problem:
        module, row_score, label = PROBLEMS[name]
        sde = VESDE(sigma=3.0)
        model = row_scores.AnalyticModel(
            sde, row_score, module, device=arguments.device,
            nodes=int(arguments.quadrature_nodes), seed=int(arguments.seed),
        )
        with patched_problem(module):
            truth, _, x = module.observations(int(arguments.observations),
                                              int(arguments.seed))
            grid, density, weights, mean, std = module.shared_reference(
                x, points=int(arguments.reference_points))
            stored[f"{name}_grid"] = grid
            stored[f"{name}_reference"] = density
            stored[f"{name}_x"] = np.asarray(x, dtype=np.float64)
            stored[f"{name}_truth"] = np.array([float(truth)])
            print(f"\n########## {name}: {label}")
            print(f"    true g = {float(truth):+.5f}   exact posterior mean "
                  f"{mean:+.5f} sd {std:.5f}   mode "
                  f"{float(grid[int(np.argmax(density))]):+.5f}", flush=True)
            print(f"    {'method':<22}{'secs':>8}{'W1/sig':>9}{'MAPerr':>9}"
                  f"{'truthErr':>10}{'p(g*)/pk':>10}{'width':>8}", flush=True)

            for method in arguments.methods:
                try:
                    if method in ("tilt", "tilt_exact_stage1"):
                        estimate, seconds, chosen = run_tilt(
                            model, module, x, grid, arguments, arguments.device,
                            exact_stage1=(method == "tilt_exact_stage1"))
                    else:
                        estimate, seconds = run_rule(
                            model, module, x, grid, method, arguments,
                            arguments.device)
                        chosen = None
                except Exception as error:  # noqa: BLE001
                    print(f"    {method:<22} FAILED: {error}", flush=True)
                    rows.append({"problem": name, "method": method,
                                 "failed": str(error)})
                    continue
                record = statistics(grid, estimate, density, float(truth))
                rows.append({"problem": name, "method": method,
                             "seconds": seconds, "degrees": chosen, **record})
                stored[f"{name}_{method}"] = estimate
                print(f"    {method:<22}{seconds:8.1f}"
                      f"{record['w1_over_sigma']:9.4f}{record['map_error']:9.4f}"
                      f"{record['truth_error']:10.4f}"
                      f"{record['mass_at_truth']:10.3f}"
                      f"{record['width_ratio']:8.3f}", flush=True)
                if chosen is not None:
                    print(f"        tilt degrees per observation: {chosen}",
                          flush=True)

    (arguments.output / "metrics.json").write_text(
        json.dumps(rows, indent=1, default=float))
    np.savez_compressed(arguments.output / "curves.npz", **stored)
    print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
