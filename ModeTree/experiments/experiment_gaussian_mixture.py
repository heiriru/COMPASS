#!/usr/bin/env python
"""Section 27 -- ModeTree-JAC on the Gaussian-mixture benchmark, oracle vs learned.

This is the third rung of the error ladder, and the first experiment in this
project that uses a *real* COMPASS score network rather than an oracle built for
ModeTree. The benchmark is
``Gaussian_mixtures_compositional_score_modelling``: a hierarchical model

    C ~ {-,+};   g | C ~ N(C mu_0, tau_g^2);   l_i | g ~ N(g, tau_l^2);
    x_i | l_i ~ N(l_i, sigma_x^2)

whose tall posterior ``p(g, l_1..l_n | x_1..x_n)`` is an **exact two-component
Gaussian mixture at every n**, so every number below is graded against a closed
form with no reference noise.

Two score sources, identical everywhere else:

* ``oracle``  -- the benchmark's exact diffused score (``oracle.OracleModel``),
* ``learned`` -- the trained tokenized checkpoint.

Both are consumed through ``MultiObsSampler._raw_row_scores`` and
``._row_score_jacobian`` (see ``modetree.compass_adapter``), which is the same
path the sampler itself uses, so the only difference between the arms is what
sits behind ``SBIm.model``.

The ``asymmetric`` regime is used throughout: it holds the exact branch split at
``w_+ = 0.75`` for *every* n, so the target's shape does not drift along the
ladder and a change in a method's answer is the method's doing.

Run:  python ModeTree/experiments/experiment_gaussian_mixture.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCHMARK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse   # noqa: E402
import json       # noqa: E402
import time       # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

import datasets      # noqa: E402
import model as M    # noqa: E402

from compass.priors import GaussianMixturePrior            # noqa: E402
from modetree.compass_adapter import CompassRowSource      # noqa: E402
from modetree.diffusion import ComposedField               # noqa: E402
from modetree.laplace import assemble                      # noqa: E402
from modetree.tree import ModeTree, ModeTreeConfig, TimeComposer  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_LADDER = (8, 16, 32, 64, 128)
CHECKPOINT = (BENCHMARK / "artifacts" / "models"
              / "tokenized_h32d3wide_n200000_six_weighted_log_sigma"
              / "Model_checkpoint.pt")
BASELINES = ("gauss_jacobian", "gauss_jacobian_mixture_prior", "langevin_fnpse")
# The network's own terminal time. There is no exact t = 0 score for a network,
# so ModeTree's "clean" composition is the composed field at this time for both
# arms -- the oracle could go lower, and making it do so would compare the two
# arms at different places.
T_FINAL = 1e-3


def mixture_prior(mixture):
    """The benchmark's *exact* two-component prior on g.

    `MultiObsSampler`'s tuple interface can only express a Gaussian, and the
    composition subtracts ``(n-1)`` prior scores, so using a moment-matched
    Gaussian here is an n-fold amplified modelling error on precisely the
    structure under test. This is what `compass.priors` was added for.
    """
    return GaussianMixturePrior(
        [mixture.weights[0], mixture.weights[1]],
        torch.tensor([[-mixture.mu_0], [mixture.mu_0]], dtype=torch.float64),
        torch.tensor([[[mixture.tau_g ** 2]], [[mixture.tau_g ** 2]]],
                     dtype=torch.float64),
    )


def load_source(kind, mixture, device="cpu"):
    if kind == "oracle":
        import oracle
        import train
        return oracle.OracleModel(mixture, sigma=train.SDE_KWARGS["sigma"],
                                  device=device)
    import train
    if not CHECKPOINT.exists():
        raise SystemExit(f"no trained checkpoint at {CHECKPOINT}")
    return train.load_to(CHECKPOINT, device)


# ---------------------------------------------------------------------------
# metrics against the closed form
# ---------------------------------------------------------------------------

def reference(mixture, x, points=4001, pad=3.0):
    grid = mixture.default_grid(x, points=points, pad=pad)
    density = mixture.global_density(grid, x)
    return np.asarray(grid), np.asarray(density)


def w1_from_density(grid, density, reference_density):
    """``int |F - F_ref|`` between two densities on a shared grid."""
    def cdf(values):
        out = np.concatenate([[0.0], np.cumsum(
            0.5 * (values[1:] + values[:-1]) * np.diff(grid))])
        return out / max(out[-1], 1e-300)
    return float(np.trapezoid(np.abs(cdf(density) - cdf(reference_density)), grid))


def w1_from_samples(grid, draws, reference_density):
    order = np.sort(np.asarray(draws).reshape(-1))
    empirical = np.searchsorted(order, grid) / order.size
    out = np.concatenate([[0.0], np.cumsum(
        0.5 * (reference_density[1:] + reference_density[:-1]) * np.diff(grid))])
    out = out / max(out[-1], 1e-300)
    return float(np.trapezoid(np.abs(empirical - out), grid))


def mixture_density(grid, means, sds, weights):
    total = np.zeros_like(grid)
    for mean, sd, weight in zip(means, sds, weights):
        total += weight * np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (
            sd * np.sqrt(2.0 * np.pi))
    return total


# ---------------------------------------------------------------------------


def run_modetree(source, x, mixture, config=None):
    prior = mixture_prior(mixture)
    row_source = CompassRowSource(
        source, x, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES, prior,
    )
    # A lean configuration: this target has two well-separated modes and the
    # ablation on the synthetic benchmarks showed jacobian_stride = 8 and a
    # shorter clean refinement cost nothing in accuracy.  The JAC composition
    # rebuilds Lambda_j on *every* score evaluation, and a Newton line search
    # makes seven of those per iteration, so the stride is where the cost is.
    config = config or ModeTreeConfig(
        levels=16, probe_iterations=30, clean_refine_iterations=60,
        sentinel_checkpoints=2, sentinel_sobol=4, jacobian_stride=4,
        switch_to_clean_t=T_FINAL, lambda_max=8.0,
    )
    field = ComposedField(row_source, "gauss_jacobian",
                          jacobian_refresh=config.jacobian_stride)
    clean = TimeComposer(field, T_FINAL)
    started = time.perf_counter()
    tree = ModeTree(field, clean, config,
                    prior_std=torch.tensor([float(mixture.global_prior_std)]))
    branches = tree.run()
    if not branches:
        return None, time.perf_counter() - started, tree, row_source
    posterior = assemble(
        clean, [b.diagnostics["clean_result"] for b in branches],
        laplace_plus_samples=0,
    )
    return posterior, time.perf_counter() - started, tree, row_source


def run_baseline(source, x, method, draws, seed, device="cpu"):
    import methods
    arguments = argparse.Namespace(
        pilot_draws=500, jacobian_refresh=1, seed=seed, draws=draws,
        churn_eta=0.05, denoise_clamp=5.0,
    )
    samples, seconds = methods.draw_score_rule(
        source, x, method, arguments, device, mixture=M.DEFAULT,
    )
    return samples[0, :, M.HIERARCHY[0]], seconds


def run_cell(kind, n, mixture, draws, seed, replicate=0):
    rng = np.random.default_rng(
        [seed, datasets.REGIMES.index("asymmetric"), n, replicate])
    data = datasets.make_dataset("asymmetric", n, rng, mixture=mixture,
                                 branch_mass=0.75, spread=1.0)
    x = np.asarray(data["x"], dtype=np.float64)
    grid, exact = reference(mixture, x)
    exact_weights = np.asarray(mixture.branch_weights(x), dtype=np.float64)
    exact_means = np.asarray(mixture.branch_means(x), dtype=np.float64)
    exact_sd = float(np.sqrt(mixture.posterior_variance(n)))

    source = load_source(kind, mixture)
    rows = []
    curves = {}

    posterior, seconds, tree, row_source = run_modetree(source, x, mixture)
    if posterior is None:
        rows.append({"method": "modetree", "seconds": seconds,
                     "error": "no branches found"})
    else:
        order = np.argsort([float(b.global_state) for b in posterior.branches])
        means = np.array([float(posterior.branches[i].global_state) for i in order])
        sds = np.array([float(posterior.branches[i].global_covariance.sqrt())
                        for i in order])
        weights = np.array([float(posterior.weights[i]) for i in order])
        density = mixture_density(grid, means, sds, weights)
        curves["modetree"] = density
        aligned_w = np.zeros(2)
        for position, mean in enumerate(means):
            aligned_w[int(np.argmin(np.abs(exact_means - mean)))] += weights[position]
        rows.append({
            "method": "modetree", "seconds": seconds,
            "w1": w1_from_density(grid, density, exact),
            "branches": len(posterior.branches),
            "weights": aligned_w.tolist(),
            "weight_tv": 0.5 * float(np.abs(aligned_w - exact_weights).sum()),
            "means": means.tolist(),
            "mean_error": float(np.abs(np.sort(means) - np.sort(exact_means)).max())
            if len(means) == 2 else None,
            "sd": sds.tolist(),
            "sd_error": float(np.abs(sds - exact_sd).max()),
            "network_forwards": row_source.network_forwards,
            "score_calls": row_source.score_calls,
            "jacobian_calls": row_source.jacobian_calls,
            "spawn_probes": tree.counters["spawn_probes"],
            "sentinel_probes": tree.counters["sentinel_probes"],
        })

    for method in BASELINES:
        try:
            draws_g, seconds = run_baseline(source, x, method, draws, seed)
        except Exception as error:                        # pragma: no cover
            rows.append({"method": method, "error": repr(error)})
            continue
        counts, edges = np.histogram(draws_g, bins=200, range=(grid[0], grid[-1]),
                                     density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        curves[method] = np.interp(grid, centres, counts)
        split = 0.5 * (exact_means[0] + exact_means[1])
        below = float(np.mean(draws_g < split))
        estimated = np.array([below, 1.0 - below])
        rows.append({
            "method": method, "seconds": seconds,
            "w1": w1_from_samples(grid, draws_g, exact),
            "weights": estimated.tolist(),
            "weight_tv": 0.5 * float(np.abs(estimated - exact_weights).sum()),
            "draws": int(draws_g.size),
        })

    for row in rows:
        row.update({"score_source": kind, "n": n,
                    "exact_weights": exact_weights.tolist(),
                    "exact_means": exact_means.tolist(), "exact_sd": exact_sd})
    return rows, grid, exact, curves, exact_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--observations", type=int, nargs="+", default=N_LADDER)
    parser.add_argument("--sources", nargs="+", default=("oracle", "learned"))
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mixture = M.DEFAULT
    everything, panels = [], {}
    for kind in args.sources:
        for n in args.observations:
            rows, grid, exact, curves, exact_means = run_cell(
                kind, n, mixture, args.draws, args.seed)
            everything.extend(rows)
            panels[(kind, n)] = {
                "grid": grid.tolist(), "exact": exact.tolist(),
                "curves": {k: v.tolist() for k, v in curves.items()},
                "exact_means": exact_means.tolist(),
            }
            for row in rows:
                if "error" in row:
                    print(f"{kind:8s} n={n:4d} {row['method']:30s} FAILED "
                          f"{row['error']}", flush=True)
                    continue
                print(f"{kind:8s} n={n:4d} {row['method']:30s} "
                      f"W1={row['w1']:.4f} TV={row['weight_tv']:.4f} "
                      f"{row['seconds']:7.2f}s", flush=True)

    with open(ARTIFACTS / "gaussian_mixture_modetree.json", "w") as handle:
        json.dump(everything, handle, indent=1)
    with open(ARTIFACTS / "gaussian_mixture_panels.json", "w") as handle:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in panels.items()}, handle)
    print(f"\nwrote {ARTIFACTS / 'gaussian_mixture_modetree.json'}")


if __name__ == "__main__":
    main()
