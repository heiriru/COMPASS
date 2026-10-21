#!/usr/bin/env python
"""Experiment 5 / Part VII -- profile error versus local-marginalization error.

The first-order IK diagnostic composes the global score at *profiled* locals,
``l*(g)``. That is not the marginal global score, and Part VII asks how much of
the remaining IK error is the profiling rather than the network. The exact
identity is

    d/dg log p(g | x) = E_{l ~ p(l | g, x)} [ d/dg log p(g, l | x) ] ,

so the marginal row contribution is the *average* of the joint row score over
the local conditional, not its value at the profile. This measures both.

Getting ``p_t(l | g, x_i)`` exactly
-----------------------------------
The arm's forward model is deterministic, so the clean row posterior
``p_0(theta | x_i)`` is supported on a measure-zero manifold and has no density
to sample by quadrature. But the diffused one is that manifold blurred by
``lambda``, and blurring is a convolution, so for draws ``theta_k`` from
``p_0(theta | x_i)``

    p_t(g, l | x_i) = (1/M) sum_k N(g; g_k, lambda^2) N(l; l_k, lambda^2)

exactly, and therefore

    p_t(l | g, x_i) = sum_k w_k N(l; l_k, lambda^2),
    w_k proportional to N(g; g_k, lambda^2).

So the conditional is an exact Gaussian mixture over rejection-sampling draws:
no quadrature over a 3-D local space, no Metropolis, and the only error is the
Monte-Carlo one of the draws themselves and the ABC tolerance that produced
them. Draws come from ``simulator.sample_consistent_with``, which is plain
rejection sampling against the prior -- no trained model anywhere in the
reference path.

Three arms on the same grid and the same data:

``profile``       the composition at ``l*(g)`` -- what the diagnostic does now
``marginal``      the composition at ``E_{l|g,x}``, the identity above
``reference``     the ABC tall posterior

Run:  python ModeTree/experiments/exp5_ik_marginalization.py --observations 8 16
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IK = ROOT / "inverse_kinematics"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(IK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim   # noqa: E402

from experiment_first_order_topology import (  # noqa: E402
    IKProblem, integrate_score, profile_locals, w1,
)
from modetree.diffusion import ComposedField  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ik_marginalization"
# node order is [angle_1, height_arm, angle_2, angle_3]; the simulator's is
# [height_arm, angle_1, angle_2, angle_3].
SIM_OF_NODE = [1, 0, 2, 3]


def row_posterior_draws(x_row, count, rng, tolerance, max_draws, minimum=400):
    """Rejection-sampling draws from ``p_0(theta | x_row)``, in node order.

    The same algebra as ``simulator.sample_consistent_with`` -- prior proposals,
    accept inside a ball of radius ``tolerance`` -- but budgeted rather than
    target-count driven. Acceptance rates here run to 3e-5 for targets far out
    in the reachable set, and demanding a fixed count from those makes the
    experiment's cost depend on the least likely observation in the dataset.
    Instead a fixed attempt budget is spent and whatever was accepted is
    returned, with the count reported so the Monte-Carlo quality of each row is
    visible rather than assumed.
    """
    target = np.asarray(x_row, dtype=np.float64).reshape(2)
    accepted, drawn, block = [], 0, 200_000
    while sum(len(item) for item in accepted) < count and drawn < max_draws:
        parameters = sim.sample_prior(block, rng)
        distance = np.linalg.norm(sim.forward(parameters) - target, axis=1)
        accepted.append(parameters[distance < tolerance])
        drawn += block
    draws = np.concatenate(accepted)[:count]
    if len(draws) < minimum:
        raise RuntimeError(
            f"only {len(draws)} draws within {tolerance} of {target} after "
            f"{drawn} attempts; the local conditional cannot be resolved")
    return np.asarray(draws, dtype=np.float64)[:, SIM_OF_NODE]


def marginal_row_scores(source, draws, grid, lam, samples, rng, chunk=16,
                        estimator="topk"):
    """``E_{l ~ p_t(l|g,x_i)}[ s_g(g, l, x_i) ]`` for every grid point and row.

    ``draws`` is a list of ``(M_i, 4)`` clean row-posterior draws, one per row;
    the local conditional is the exact mixture ``sum_k w_k N(l; l_k, lambda^2)``
    with ``w_k`` proportional to ``N(g; g_k, lambda^2)``.

    Two estimators, and the difference matters more than the sample count.

    ``sampled`` draws a fresh ``k`` and a fresh ``eps`` at every grid point.
    That is unbiased but its error is **white noise in g**, and the score is
    integrated to get a density -- integrating white noise is a random walk, so
    the reconstructed posterior picks up a drift of order
    ``eps_rms * sqrt(range * spacing)`` that grows as ``sqrt(n)`` over the rows.
    Measured on this problem it is what makes the "marginal" column worse than
    the profile at small lambda, which would otherwise be misread as
    marginalization not helping.

    ``topk`` (default) instead takes, at each ``g``, the ``samples`` draws of
    largest weight, renormalises over them, and perturbs each by a ``eps`` fixed
    once per draw index. The selection changes by about one element between
    neighbouring grid points and the noise is common, so the error is smooth in
    ``g`` and does not accumulate under integration. It pays a truncation bias
    instead, which is small whenever ``samples`` is a decent fraction of the
    effective sample size -- reported by ``--report-ess``.

    Both cost the same number of network evaluations.
    """
    n = len(draws)
    points = len(grid)
    total = np.zeros((points, n))
    noise = [rng.standard_normal(theta[:, 1:].shape) for theta in draws]

    weights = []
    for theta in draws:
        distance = (grid[:, None] - theta[None, :, 0]) ** 2
        log_weight = -0.5 * distance / lam ** 2
        log_weight -= log_weight.max(axis=1, keepdims=True)
        weight = np.exp(log_weight)
        weights.append(weight / weight.sum(axis=1, keepdims=True))

    for start in range(0, points, chunk):
        stop = min(start + chunk, points)
        size = stop - start
        locals_ = np.empty((size, samples, n, 3))
        coefficients = np.empty((size, samples, n))
        for index, theta in enumerate(draws):
            block = weights[index][start:stop]              # (size, M)
            take = min(samples, theta.shape[0])
            if estimator == "topk":
                picked = np.argpartition(-block, take - 1, axis=1)[:, :take]
                chosen = np.take_along_axis(block, picked, axis=1)
                chosen = chosen / chosen.sum(axis=1, keepdims=True)
            else:
                picked = np.array([rng.choice(theta.shape[0], size=take, p=row)
                                   for row in block])
                chosen = np.full((size, take), 1.0 / take)
            if take < samples:                              # pad, weight zero
                pad = np.zeros((size, samples - take), dtype=picked.dtype)
                picked = np.concatenate([picked, pad], axis=1)
                chosen = np.concatenate(
                    [chosen, np.zeros((size, samples - take))], axis=1)
            coefficients[:, :, index] = chosen
            perturbed = theta[picked, 1:] + lam * noise[index][picked]
            locals_[:, :, index, :] = perturbed
        g_block = np.broadcast_to(
            grid[start:stop, None, None, None], (size, samples, n, 1))
        state = np.concatenate([g_block, locals_], axis=-1)
        flat = torch.as_tensor(state.reshape(size * samples, n, 4),
                               dtype=torch.float64)
        scores = source.row_scores(flat, lam).numpy()[..., 0]
        scores = scores.reshape(size, samples, n)
        total[start:stop] = (coefficients * scores).sum(axis=1)
    return total


def effective_sample_size(draws, grid, lam, support):
    """Median ESS of the mixture weights, over the grid points that matter."""
    out = []
    for theta in draws:
        distance = (grid[support][:, None] - theta[None, :, 0]) ** 2
        log_weight = -0.5 * distance / lam ** 2
        log_weight -= log_weight.max(axis=1, keepdims=True)
        weight = np.exp(log_weight)
        weight /= weight.sum(axis=1, keepdims=True)
        out.append(1.0 / (weight ** 2).sum(axis=1))
    return float(np.median(np.concatenate(out)))


def single_row_source(problem, x_row):
    return problem.source("learned", np.asarray(x_row).reshape(1, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, nargs="+", default=(8, 16))
    parser.add_argument("--times", type=float, nargs="+", default=(1e-2,))
    parser.add_argument("--grid-points", type=int, default=401)
    parser.add_argument("--row-draws", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--estimator", default="topk",
                        choices=("topk", "sampled"))
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--max-draws", type=int, default=40_000_000)
    parser.add_argument("--profile-iterations", type=int, default=300)
    parser.add_argument("--regime", default="bimodal")
    parser.add_argument("--ik-checkpoint", default=None)
    parser.add_argument("--ik-wrapper", default="plain")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    problem = IKProblem(regime=args.regime, checkpoint=args.ik_checkpoint,
                        wrapper=args.ik_wrapper)
    rng = np.random.default_rng([args.seed, 5])
    grid = problem.grid(args.grid_points)
    rows, store = [], {}

    for n in args.observations:
        truth, x = problem.dataset(n)
        reference_grid, reference_density = problem.reference(x, n)
        reference_on_grid = np.interp(grid, reference_grid, reference_density)
        started = time.perf_counter()
        draws = [row_posterior_draws(x[i], args.row_draws, rng,
                                     args.tolerance, args.max_draws)
                 for i in range(n)]
        abc_seconds = time.perf_counter() - started
        counts = [len(d) for d in draws]
        print(f"n={n}: row draws min={min(counts)} median="
              f"{int(np.median(counts))} max={max(counts)} in "
              f"{abc_seconds:.0f}s", flush=True)

        source = problem.source("learned", x)
        prior_std = problem.prior_std
        for t in args.times:
            lam = source.schedule.lam(t)
            # -- A: the profile composition (what the diagnostic does today)
            field = ComposedField(source, "uncorrected")
            locals_, local_norms, _forwards = profile_locals(
                field, grid, lam, t, iterations=args.profile_iterations)
            state = torch.cat(
                [torch.as_tensor(grid, dtype=torch.float64).reshape(-1, 1),
                 locals_], dim=-1)
            with torch.no_grad():
                profile_score = field.score(state, t, lam).score[:, 0].numpy()

            # -- B: the marginal composition
            started = time.perf_counter()
            per_row = marginal_row_scores(source, draws, grid, lam,
                                          args.samples, rng,
                                          estimator=args.estimator)
            prior_score = -grid / (prior_std ** 2 + lam ** 2)
            marginal_score = (1 - n) * prior_score + per_row.sum(axis=1)
            marginal_seconds = time.perf_counter() - started

            profile_density = integrate_score(grid, profile_score)
            marginal_density = integrate_score(grid, marginal_score)
            row = {
                "n": int(n), "t": float(t), "lambda": float(lam),
                "regime": args.regime,
                "checkpoint": problem.checkpoint,
                "max_local_score_at_profile": float(local_norms.max()),
                "w1_profile_over_sigma": w1(grid, profile_density,
                                            reference_on_grid) / prior_std,
                "w1_marginal_over_sigma": w1(grid, marginal_density,
                                             reference_on_grid) / prior_std,
                "row_draws_requested": args.row_draws,
                "row_draws_min": int(min(counts)),
                "row_draws_median": float(np.median(counts)),
                "samples": args.samples, "estimator": args.estimator,
                "effective_sample_size": effective_sample_size(
                    draws, grid, lam,
                    reference_on_grid > 0.05 * reference_on_grid.max()),
                "abc_seconds": abc_seconds,
                "marginal_seconds": marginal_seconds,
            }
            rows.append(row)
            store[f"{n}|{t:g}"] = {
                "grid": grid.tolist(),
                "profile_score": profile_score.tolist(),
                "marginal_score": marginal_score.tolist(),
                "profile_density": profile_density.tolist(),
                "marginal_density": marginal_density.tolist(),
                "reference": reference_on_grid.tolist(),
            }
            print(f"  n={n:3d} t={t:<8g} lam={lam:.4f}  "
                  f"W1/sd profile={row['w1_profile_over_sigma']:.4f}  "
                  f"marginal={row['w1_marginal_over_sigma']:.4f}  "
                  f"ess={row['effective_sample_size']:.0f}  "
                  f"({marginal_seconds:.0f}s)", flush=True)
            with open(OUT / f"rows{args.out_suffix}.json", "w") as handle:
                json.dump(rows, handle, indent=1)
            with open(OUT / f"curves{args.out_suffix}.json", "w") as handle:
                json.dump(store, handle)
    print(f"\nwrote {OUT / ('rows' + args.out_suffix + '.json')}")


if __name__ == "__main__":
    main()
