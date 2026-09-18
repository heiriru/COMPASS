#!/usr/bin/env python
"""Section 22 -- the ablation table, plus Milestones M and N.

Rows:
  no-sentinel / no-curvature   which discovery mechanism is load-bearing
  one-path energies            is multi-path + cycle reconciliation worth it
  Laplace vs Laplace+          Milestone G
  jacobian_stride 1/2/4/8      Milestone M2 (sparse JAC)
  switch_to_clean_t            Milestone N (where to hand over to the exact
                               clean composition)
  predictor carry vs tangent   Milestone H (is the tangent predictor worth it)

Run:  python ModeTree/experiments/ablations.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import dataclasses  # noqa: E402
import json         # noqa: E402
import time         # noqa: E402

import torch        # noqa: E402

from modetree import benchmarks as B                          # noqa: E402
from modetree.composition import make_composer                # noqa: E402
from modetree.diffusion import ComposedField, DiffusedOracle  # noqa: E402
from modetree.laplace import assemble                         # noqa: E402
from modetree.tree import ModeTree, ModeTreeConfig            # noqa: E402

ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
N_LADDER = (8, 32, 100)
CASES = ("map_vs_mass", "empty_valley", "narrow_broad")


def build(case, n):
    if case == "map_vs_mass":
        return B.map_vs_mass_1d(n), (-12, 12)
    if case == "empty_valley":
        return B.empty_valley_1d(n), (-20, 20)
    return B.bimodal_1d(n, case), (-12, 12)


VARIANTS = {
    "full":            dict(),
    "no_sentinel":     dict(sentinel_enabled=False),
    "no_curvature":    dict(curvature_spawn=False),
    "one_path":        dict(energy_multiple_paths=False),
    "stride2":         dict(jacobian_stride=2),
    "stride4":         dict(jacobian_stride=4),
    "stride8":         dict(jacobian_stride=8),
    "switch_t0.05":    dict(switch_to_clean_t=0.05),
    "switch_t0.002":   dict(switch_to_clean_t=0.002),
    "tangent":         dict(predictor="tangent"),
}


def run(case, n, name, overrides, laplace_plus):
    model, window = build(case, n)
    ref = B.reference_1d(model, window[0], window[1], points=100_001)
    keep = ref.mode_masses >= 1e-3
    true_modes = ref.modes[keep]
    true_masses = ref.mode_masses[keep]
    true_masses = true_masses / true_masses.sum()

    config = ModeTreeConfig(**overrides)
    clean = make_composer(model)
    field = ComposedField(DiffusedOracle(model), "gauss_jacobian",
                          jacobian_refresh=config.jacobian_stride)
    started = time.perf_counter()
    tree = ModeTree(field, clean, config,
                    prior_std=model.prior_cov.diagonal().sqrt())
    branches = tree.run()
    posterior = assemble(
        clean, [b.diagnostics["clean_result"] for b in branches],
        multiple_paths=config.energy_multiple_paths,
        laplace_plus_samples=32 if laplace_plus else 0,
    )
    runtime = time.perf_counter() - started
    weights = posterior.weights_plus if laplace_plus else posterior.weights

    aligned = torch.zeros(true_masses.numel(), dtype=torch.float64)
    matched = torch.zeros(true_masses.numel(), dtype=torch.bool)
    for b, branch in enumerate(posterior.branches):
        distances = (true_modes.reshape(-1) - branch.global_state.reshape(())).abs()
        k = int(torch.argmin(distances))
        aligned[k] += float(weights[b])
        if float(distances[k]) < 0.1:
            matched[k] = True
    sigma = float(ref.covariance.diagonal().sum().sqrt())
    heights = ref.mode_log_heights[keep]
    ambiguous = bool(heights.numel() > 1 and float(heights[0] - heights[1]) < 1e-9)
    e_map = (float((true_modes.reshape(-1) - posterior.map_global.reshape(())).abs().min())
             if ambiguous
             else float((posterior.map_global.reshape(()) - ref.map.reshape(())).abs()))
    cycles = posterior.energy_graph.cycle_residuals
    return {
        "case": case, "n": n, "variant": name,
        "laplace_plus": laplace_plus,
        "e_map": e_map, "e_map_sigma": e_map / sigma,
        "map_ambiguous": ambiguous,
        "weight_tv": 0.5 * float((aligned - true_masses).abs().sum()),
        "missing_mass": float(true_masses[~matched].sum()),
        "mode_recall": float(matched.to(torch.float64).mean()),
        "n_branches": len(posterior.branches),
        "seconds": runtime,
        "spawn_probes": tree.counters["spawn_probes"],
        "sentinel_probes": tree.counters["sentinel_probes"],
        "jacobian_evaluations": tree.counters["jacobian_evaluations"],
        "path_disagreement": (max(posterior.energy_graph.path_disagreement.values())
                              if posterior.energy_graph.path_disagreement else 0.0),
        "cycle_residual": (max(abs(v) for v in cycles.values()) if cycles else None),
        "ess_min": min((b.ess for b in posterior.branches if b.ess is not None),
                       default=None),
    }


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    rows = []
    for case in CASES:
        for n in N_LADDER:
            for name, overrides in VARIANTS.items():
                for laplace_plus in ((False, True) if name == "full" else (False,)):
                    try:
                        row = run(case, n, name, overrides, laplace_plus)
                    except Exception as error:
                        row = {"case": case, "n": n, "variant": name,
                               "laplace_plus": laplace_plus, "error": repr(error)}
                    rows.append(row)
                    label = name + ("+laplace+" if laplace_plus else "")
                    if "error" in row:
                        print(f"{case:13s} n={n:4d} {label:16s} FAILED {row['error']}",
                              flush=True)
                        continue
                    print(f"{case:13s} n={n:4d} {label:16s} "
                          f"recall={row['mode_recall']:.2f} "
                          f"miss={row['missing_mass']:.4f} "
                          f"TV={row['weight_tv']:.4f} "
                          f"e_MAP/s={row['e_map_sigma']:.2e} "
                          f"{row['seconds']:.1f}s jac={row['jacobian_evaluations']}",
                          flush=True)
    path = os.path.join(ARTIFACTS, "ablations.json")
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
