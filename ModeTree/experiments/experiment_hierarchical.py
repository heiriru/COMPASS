#!/usr/bin/env python
"""Section A3 -- the hierarchical benchmark, including the COMPASS shape.

y_i = (g, l_i) with a multimodal global block.  Four rungs of the error ladder
are run at each n:

  oracle_known    exact row scores, true mode seeds        (energy + Laplace only)
  oracle_tree     exact row scores, automatic discovery    (adds discovery error)
  jac_sampler     gauss_jacobian reverse diffusion         (the current default)
  fnpe_sampler    F-NPSE annealed Langevin                 (the slow baseline)

The reference is the exact global marginal posterior on a grid, which the
Gaussian-mixture construction makes available at any n.

Run:  python ModeTree/experiments/experiment_hierarchical.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json   # noqa: E402
import time   # noqa: E402

import torch  # noqa: E402

from modetree import benchmarks as B                          # noqa: E402
from modetree.composition import make_composer                # noqa: E402
from modetree.diffusion import ComposedField, DiffusedOracle  # noqa: E402
from modetree.laplace import assemble                         # noqa: E402
from modetree.refine import refine_mode                       # noqa: E402
from modetree.samplers import annealed_langevin, reverse_diffusion  # noqa: E402
from modetree.tree import ModeTree, ModeTreeConfig            # noqa: E402

ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
SHAPES = ((1, 1), (2, 1), (7, 3))
N_LADDER = (1, 4, 16, 64, 100)
NUM_SAMPLES = 1000


def reference_1d_global(model, lo=-12.0, hi=12.0, points=100_001):
    grid = torch.linspace(lo, hi, points, dtype=torch.float64).unsqueeze(-1)
    chunks = [model.global_log_prob(grid[s:s + 20_000])
              for s in range(0, points, 20_000)]
    logp = torch.cat(chunks)
    step = float(grid[1, 0] - grid[0, 0])
    logp = logp - (torch.logsumexp(logp, 0) + torch.tensor(step).log())
    return grid.reshape(-1), logp


def w1_from_samples(samples, grid, logp):
    density = logp.exp()
    cdf = torch.cumulative_trapezoid(density, grid, dim=0)
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf])
    ordered = samples.reshape(-1).sort().values
    empirical = torch.searchsorted(ordered, grid.contiguous()).to(torch.float64)
    empirical = empirical / ordered.numel()
    return float(torch.trapz((empirical - cdf).abs(), grid))


def run(d_g, d_l, n, generator):
    model = B.hierarchical(n, d_g=d_g, d_l=d_l, modes=2, separation=2.0)
    clean = make_composer(model)
    rows = []
    reference = reference_1d_global(model) if d_g == 1 else None

    # The true global modes come from refining the *exact* locals-marginalised
    # composer, seeded at the row factors' own component centres.  A blind
    # multi-start sweep is not viable at d_g = 7: a line through one coordinate
    # misses every mode and a grid is exponential.  The component centres are a
    # property of the benchmark construction, not of the method under test, so
    # using them here costs no validity -- the *method* still has to find its
    # modes without them.
    marginal = make_composer(model, marginal=True)
    centres = torch.stack([f.means[:, :d_g] for f in model.factors]).mean(0)
    starts = torch.cat([centres, torch.zeros(1, d_g, dtype=torch.float64)])
    truth = []
    for start in starts:
        result = refine_mode(marginal, start, tol=1e-11, max_iter=200)
        if not result.diagnostics["is_local_maximum"]:
            continue
        if all(float((result.theta - t).abs().max()) > 1e-4 for t in truth):
            truth.append(result.theta)
    truth = torch.stack(truth) if truth else torch.zeros(0, d_g, dtype=torch.float64)

    for label in ("oracle_known", "oracle_tree"):
        started = time.perf_counter()
        try:
            if label == "oracle_known":
                template = torch.zeros(clean.dim, dtype=torch.float64)
                results = []
                for mode in truth:
                    seed = template.clone()
                    seed[:d_g] = mode
                    results.append(refine_mode(clean, seed, tol=1e-11, max_iter=300))
                counters = {}
            else:
                field = ComposedField(DiffusedOracle(model), "gauss_jacobian")
                tree = ModeTree(
                    field, clean,
                    ModeTreeConfig(levels=20, probe_iterations=40,
                                   sentinel_checkpoints=2, sentinel_sobol=6),
                    prior_std=model.prior_cov.diagonal().sqrt(),
                )
                branches = tree.run()
                counters = dict(tree.counters)
                results = [b.diagnostics["clean_result"] for b in branches]
            posterior = assemble(clean, results, laplace_plus_samples=0)
            samples = posterior.sample(NUM_SAMPLES, generator=generator)
            runtime = time.perf_counter() - started
        except Exception as error:
            rows.append({"method": label, "error": repr(error)})
            continue
        found = torch.stack([b.global_state for b in posterior.branches])
        matched = torch.zeros(max(truth.shape[0], 1), dtype=torch.bool)
        for position in found:
            if truth.shape[0]:
                distances = torch.linalg.vector_norm(truth - position, dim=-1)
                if float(distances.min()) < 0.25:
                    matched[int(torch.argmin(distances))] = True
        row = {
            "method": label, "seconds": runtime,
            "n_branches": len(posterior.branches),
            "n_true_modes": int(truth.shape[0]),
            "mode_recall": float(matched.to(torch.float64).mean()),
            "weights": posterior.weights.tolist(),
            "map": posterior.map_global.tolist(),
            "map_is_mass_branch": posterior.map_index == posterior.mass_index,
            "spawn_probes": counters.get("spawn_probes", 0),
            "sentinel_probes": counters.get("sentinel_probes", 0),
            "score_evaluations": posterior.diagnostics["score_evaluations"],
        }
        if reference is not None:
            row["w1"] = w1_from_samples(samples[:, 0], *reference)
        rows.append(row)

    for rule, sampler in (("gauss_jacobian", "diffusion"), ("fnpe", "langevin")):
        field = ComposedField(DiffusedOracle(model), rule)
        prior_std = model.prior_cov.diagonal().sqrt()
        if sampler == "diffusion":
            draws, info = reverse_diffusion(field, NUM_SAMPLES, levels=150,
                                            prior_std=prior_std, generator=generator)
        else:
            draws, info = annealed_langevin(field, NUM_SAMPLES, levels=75,
                                            steps_per_level=10, prior_std=prior_std,
                                            generator=generator)
        row = {"method": rule + "_sampler", "seconds": info["seconds"],
               "score_evaluations": info["score_field_calls"]}
        if reference is not None:
            row["w1"] = w1_from_samples(draws[:, 0], *reference)
        rows.append(row)

    for row in rows:
        row.update({"d_g": d_g, "d_l": d_l, "n": n})
    return rows


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    generator = torch.Generator().manual_seed(7)
    out = []
    for d_g, d_l in SHAPES:
        for n in N_LADDER:
            rows = run(d_g, d_l, n, generator)
            out.extend(rows)
            for row in rows:
                if "error" in row:
                    print(f"dg={d_g} dl={d_l} n={n:4d} {row['method']:16s} FAILED "
                          f"{row['error']}", flush=True)
                    continue
                print(f"dg={d_g} dl={d_l} n={n:4d} {row['method']:16s} "
                      f"W1={row.get('w1', float('nan')):.4f} "
                      f"B={row.get('n_branches', '-')}/"
                      f"{row.get('n_true_modes', '-')} "
                      f"recall={row.get('mode_recall', float('nan')):.2f} "
                      f"{row['seconds']:.2f}s", flush=True)
    path = os.path.join(ARTIFACTS, "experiment_hierarchical.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
