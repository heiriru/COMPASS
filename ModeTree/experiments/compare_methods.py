#!/usr/bin/env python
"""Sections 7-10, 13-17 -- the headline comparison, oracle row scores throughout.

Every method here sees the *same* exact per-observation scores, so the numbers
compare composition rules and inference procedures, not score networks:

  uncorrected      reverse diffusion on (1-n) s_prior + sum_j s_j
  gauss_jacobian   reverse diffusion on the JAC precision-weighted composition
  fnpe             annealed Langevin on the Geffner bridging score
  modetree         ModeTree-JAC: discovery + structured Laplace
  modetree_plus    ... with Laplace+
  modetree_known   ModeTree with the true mode seeds (isolates discovery error)

Run:  python ModeTree/experiments/compare_methods.py [--quick]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse   # noqa: E402
import json       # noqa: E402
import time       # noqa: E402

import torch      # noqa: E402

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
N_LADDER = (1, 2, 4, 8, 16, 32, 64, 100)
NUM_SAMPLES = 2000


def cases():
    yield "symmetric", lambda n: B.bimodal_1d(n, "symmetric"), (-12, 12)
    yield "map_vs_mass", B.map_vs_mass_1d, (-12, 12)
    yield "empty_valley", B.empty_valley_1d, (-20, 20)
    yield "nearly_tied", lambda n: B.bimodal_1d(n, "nearly_tied"), (-12, 12)
    yield "narrow_broad", lambda n: B.bimodal_1d(n, "narrow_broad"), (-12, 12)
    yield "vanishing", lambda n: B.bimodal_1d(n, "vanishing"), (-12, 12)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def reference_cdf(ref):
    density = ref.log_density.exp()
    cdf = torch.cumulative_trapezoid(density, ref.grid, dim=0)
    return torch.cat([torch.zeros(1, dtype=torch.float64), cdf])


def wasserstein1(samples, ref, cdf):
    """``int |F_emp - F_ref|``, exact on the reference grid."""
    samples = samples.reshape(-1).sort().values
    counts = torch.searchsorted(samples, ref.grid.contiguous())
    empirical = counts.to(torch.float64) / samples.numel()
    return float(torch.trapz((empirical - cdf).abs(), ref.grid))


def basin_weights(samples, ref):
    """Fraction of samples in each reference basin, in reference mode order."""
    edges = ref.separators.reshape(-1).sort().values
    index = torch.searchsorted(edges, samples.reshape(-1).contiguous())
    order = torch.argsort(ref.modes.reshape(-1))
    counts = torch.zeros(ref.modes.shape[0], dtype=torch.float64)
    for b in range(ref.modes.shape[0]):
        counts[order[b]] = float((index == b).sum())
    return counts / max(float(counts.sum()), 1.0)


def sample_map(samples, model):
    values = model.global_log_prob(samples.reshape(-1, 1))
    return samples.reshape(-1)[int(torch.argmax(values))]


def relevant_modes(ref, threshold=1e-3):
    keep = ref.mode_masses >= threshold
    masses = ref.mode_masses[keep]
    return ref.modes[keep], masses / masses.sum()


def map_error(map_estimate, ref):
    """MAP error, with exactly-tied modes reported as ambiguous.

    Several benchmarks are exactly symmetric, so which mode is "the" MAP is not
    defined; scoring against ``ref.map`` there measures a tie-break, not
    accuracy.  When the top two reference mode heights agree to 1e-9 the error
    is taken to the nearest tied mode and the row is flagged.
    """
    heights = ref.mode_log_heights
    ambiguous = bool(heights.numel() > 1 and float(heights[0] - heights[1]) < 1e-9)
    if ambiguous:
        error = float((ref.modes.reshape(-1) - map_estimate).abs().min())
    else:
        error = float((map_estimate - ref.map.reshape(())).abs())
    return error, ambiguous


def mixture_w1(posterior, ref, cdf):
    """Exact W1 of the returned Gaussian-mixture posterior, no sampling noise.

    ModeTree returns a *density*, so its sampling error is a property of the
    caller's draw count, not of the method.  Reporting both this and the
    sample-based W1 keeps the comparison against the samplers like-for-like
    while still saying what the method itself achieves.
    """
    grid = ref.grid
    total = torch.zeros_like(grid)
    weights = (posterior.weights_plus if posterior.weights_plus is not None
               else posterior.weights)
    for weight, branch in zip(weights, posterior.branches):
        mean = float(branch.global_state.reshape(()))
        sigma = float(branch.global_covariance.reshape(()).sqrt())
        z = (grid - mean) / (sigma * (2.0 ** 0.5))
        total = total + float(weight) * 0.5 * (1.0 + torch.erf(z))
    return float(torch.trapz((total - cdf).abs(), grid))


def metrics(name, samples, weights, map_estimate, ref, model, cdf, runtime,
            evaluations, extra=None):
    true_modes, true_masses = relevant_modes(ref)
    sigma = float(ref.covariance.diagonal().sum().sqrt())
    error, ambiguous = map_error(map_estimate, ref)
    row = {
        "method": name,
        "w1": wasserstein1(samples, ref, cdf),
        "mean_error": float((samples.mean() - ref.mean).abs()),
        "std_error": float((samples.std() - ref.covariance.diagonal().sqrt()).abs()),
        "e_map": error,
        "e_map_sigma": error / sigma,
        "map_ambiguous": ambiguous,
        "seconds": runtime,
        "score_field_calls": evaluations,
    }
    if weights is not None:
        w = weights.reshape(-1)[: true_masses.numel()]
        w = w / max(float(w.sum()), 1e-300)
        row["weight_tv"] = 0.5 * float((w - true_masses).abs().sum())
        row["weights"] = w.tolist()
    row["weights_true"] = true_masses.tolist()
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------


def run_modetree(model, ref, field, clean, known_modes=None, laplace_plus=0,
                 config=None, num_samples=NUM_SAMPLES, generator=None):
    started = time.perf_counter()
    counters = {"spawn_probes": 0, "sentinel_probes": 0}
    if known_modes is None:
        tree = ModeTree(field, clean, config or ModeTreeConfig(),
                        prior_std=model.prior_cov.diagonal().sqrt())
        branches = tree.run()
        counters = dict(tree.counters)
        results = [b.diagnostics["clean_result"] for b in branches]
    else:
        results = [refine_mode(clean, m, tol=1e-11, max_iter=200)
                   for m in known_modes]
    posterior = assemble(clean, results, laplace_plus_samples=laplace_plus)
    samples = posterior.sample(num_samples, generator=generator,
                               use_plus=bool(laplace_plus))
    runtime = time.perf_counter() - started
    weights = (posterior.weights_plus if laplace_plus else posterior.weights)
    # align branch order with the reference mode order (descending density)
    found = torch.stack([b.global_state for b in posterior.branches]).reshape(-1)
    true_modes, true_masses = relevant_modes(ref)
    aligned = torch.zeros(true_masses.numel(), dtype=torch.float64)
    matched = torch.zeros(true_masses.numel(), dtype=torch.bool)
    for b, position in enumerate(found):
        k = int(torch.argmin((true_modes.reshape(-1) - position).abs()))
        aligned[k] += float(weights[b])
        if float((true_modes.reshape(-1)[k] - position).abs()) < 0.1:
            matched[k] = True
    missing = float(true_masses[~matched].sum())
    extra = {
        "n_branches": len(posterior.branches),
        "missing_mass": missing,
        "mode_recall": float(matched.to(torch.float64).mean()),
        "mode_precision": float(matched.sum()) / max(len(posterior.branches), 1),
        "map_is_mass_branch": posterior.map_index == posterior.mass_index,
        "cycle_residual": (max(abs(v) for v in
                               posterior.energy_graph.cycle_residuals.values())
                           if posterior.energy_graph.cycle_residuals else None),
        "path_disagreement": (max(posterior.energy_graph.path_disagreement.values())
                              if posterior.energy_graph.path_disagreement else 0.0),
        "spawn_probes": counters.get("spawn_probes", 0),
        "sentinel_probes": counters.get("sentinel_probes", 0),
        "ess_min": min((b.ess for b in posterior.branches if b.ess is not None),
                       default=None),
    }
    return (samples, aligned, posterior.map_global.reshape(()), runtime, extra,
            posterior.diagnostics["score_evaluations"], posterior)


def run_case(name, builder, window, n, quick=False):
    model = builder(n)
    ref = B.reference_1d(model, window[0], window[1],
                         points=100_001 if quick else 200_001)
    cdf = reference_cdf(ref)
    clean = make_composer(model)
    oracle = DiffusedOracle(model)
    prior_std = model.prior_cov.diagonal().sqrt()
    generator = torch.Generator().manual_seed(12345)
    num_samples = 500 if quick else NUM_SAMPLES
    levels = 100 if quick else 200
    rows = []

    for rule, sampler in (("uncorrected", "diffusion"),
                          ("gauss_jacobian", "diffusion"),
                          ("fnpe", "langevin")):
        field = ComposedField(DiffusedOracle(model), rule)
        if sampler == "diffusion":
            samples, info = reverse_diffusion(
                field, num_samples, levels=levels, prior_std=prior_std,
                generator=generator,
            )
        else:
            samples, info = annealed_langevin(
                field, num_samples, levels=levels // 2,
                steps_per_level=10, prior_std=prior_std, generator=generator,
            )
        samples = samples[:, 0]
        rows.append(metrics(
            rule, samples, basin_weights(samples, ref),
            sample_map(samples, model), ref, model, cdf,
            info["seconds"], info["score_field_calls"],
        ))

    field = ComposedField(oracle, "gauss_jacobian")
    true_modes, _ = relevant_modes(ref)
    for label, known, plus in (("modetree_known", true_modes, 0),
                               ("modetree", None, 0),
                               ("modetree_plus", None, 32)):
        try:
            (samples, weights, map_estimate, runtime, extra, evaluations,
             posterior) = run_modetree(
                 model, ref,
                 ComposedField(DiffusedOracle(model), "gauss_jacobian"),
                 clean, known_modes=known, laplace_plus=plus,
                 num_samples=num_samples, generator=generator)
        except Exception as error:                       # pragma: no cover
            rows.append({"method": label, "error": repr(error)})
            continue
        extra["w1_analytic"] = mixture_w1(posterior, ref, cdf)
        rows.append(metrics(
            label, samples[:, 0] if samples.dim() > 1 else samples, weights,
            map_estimate, ref, model, cdf, runtime, evaluations, extra,
        ))

    for row in rows:
        row.update({"case": name, "n": n})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cases", default=None)
    args = parser.parse_args()
    ladder = (1, 8, 100) if args.quick else N_LADDER
    wanted = set(args.cases.split(",")) if args.cases else None

    os.makedirs(ARTIFACTS, exist_ok=True)
    out = []
    for name, builder, window in cases():
        if wanted and name not in wanted:
            continue
        for n in ladder:
            rows = run_case(name, builder, window, n, quick=args.quick)
            out.extend(rows)
            for row in rows:
                if "error" in row:
                    print(f"{name:13s} n={n:4d} {row['method']:15s} FAILED {row['error']}")
                    continue
                print(
                    f"{name:13s} n={n:4d} {row['method']:15s} "
                    f"W1={row['w1']:.4f}"
                    + (f"/{row['w1_analytic']:.4f}" if 'w1_analytic' in row else "")
                    + f" e_MAP/s={row['e_map_sigma']:.3f} "
                    f"TV={row.get('weight_tv', float('nan')):.4f} "
                    f"miss={row.get('missing_mass', 0.0):.4f} "
                    f"{row['seconds']:.2f}s calls={row['score_field_calls']}",
                    flush=True,
                )
    path = os.path.join(ARTIFACTS,
                        "compare_methods%s.json" % ("_quick" if args.quick else ""))
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nwrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
