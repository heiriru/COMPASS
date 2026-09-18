#!/usr/bin/env python
"""Sections A2 and 15 -- two-dimensional multimodality and the remote small mode.

The ``remote_small`` regime is the one that decides whether sentinel probes are
necessary: its second mode carries ~10% of the mass and is *not* reachable by
bifurcating the dominant mode, so curvature monitoring alone cannot find it.
The experiment therefore runs the discovery with and without sentinels.

Run:  python ModeTree/experiments/experiment_2d.py
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
from modetree.tree import ModeTree, ModeTreeConfig            # noqa: E402

ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
REGIMES = ("symmetric", "unequal_weights", "unequal_covariance",
           "narrow_broad", "remote_small")
N_LADDER = (4, 16, 64, 100)


def run(regime, n, sentinel, curvature, points=401):
    model = B.four_mode_2d(n, regime)
    span = 8.0
    ref = B.reference_2d(model, (-span, -span), (span, span), points=points)
    keep = ref.mode_masses >= 1e-2
    true_modes = ref.modes[keep]
    true_masses = ref.mode_masses[keep]
    true_masses = true_masses / true_masses.sum()

    clean = make_composer(model)
    field = ComposedField(DiffusedOracle(model), "gauss_jacobian")
    config = ModeTreeConfig(
        sentinel_enabled=sentinel, curvature_spawn=curvature,
        sentinel_sobol=16, max_branches=12,
    )
    started = time.perf_counter()
    tree = ModeTree(field, clean, config,
                    prior_std=model.prior_cov.diagonal().sqrt())
    branches = tree.run()
    if not branches:
        # a genuine outcome, not an error: with curvature spawning disabled the
        # discovery can come up empty, and that is exactly what the ablation is
        # asking about.
        return {
            "regime": regime, "n": n, "sentinel": sentinel,
            "curvature": curvature, "true_masses": true_masses.tolist(),
            "found_modes": [], "weights": [], "mode_recall": 0.0,
            "missing_mass": 1.0, "false_modes": 0, "weight_tv": 1.0,
            "cycle_residual": None, "path_disagreement": 0.0,
            "spawn_probes": tree.counters["spawn_probes"],
            "sentinel_probes": tree.counters["sentinel_probes"],
            "seconds": time.perf_counter() - started,
        }
    posterior = assemble(
        clean, [b.diagnostics["clean_result"] for b in branches],
        laplace_plus_samples=0,
    )
    runtime = time.perf_counter() - started

    found = torch.stack([b.global_state for b in posterior.branches])
    matched = torch.zeros(true_masses.numel(), dtype=torch.bool)
    aligned = torch.zeros(true_masses.numel(), dtype=torch.float64)
    for b, position in enumerate(found):
        distances = torch.linalg.vector_norm(true_modes - position, dim=-1)
        k = int(torch.argmin(distances))
        aligned[k] += float(posterior.weights[b])
        if float(distances[k]) < 0.4:
            matched[k] = True
    cycles = posterior.energy_graph.cycle_residuals
    return {
        "regime": regime, "n": n, "sentinel": sentinel, "curvature": curvature,
        "true_modes": true_modes.tolist(),
        "true_masses": true_masses.tolist(),
        "found_modes": found.tolist(),
        "weights": posterior.weights.tolist(),
        "mode_recall": float(matched.to(torch.float64).mean()),
        "missing_mass": float(true_masses[~matched].sum()),
        "false_modes": int(len(posterior.branches) - int(matched.sum())),
        "weight_tv": 0.5 * float((aligned - true_masses).abs().sum()),
        "cycle_residual": (max(abs(v) for v in cycles.values()) if cycles else None),
        "path_disagreement": (max(posterior.energy_graph.path_disagreement.values())
                              if posterior.energy_graph.path_disagreement else 0.0),
        "spawn_probes": tree.counters["spawn_probes"],
        "sentinel_probes": tree.counters["sentinel_probes"],
        "seconds": runtime,
    }


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    rows = []
    for regime in REGIMES:
        for n in N_LADDER:
            settings = [(True, True), (False, True), (True, False)]
            for sentinel, curvature in settings:
                row = run(regime, n, sentinel, curvature)
                rows.append(row)
                tag = ("both" if sentinel and curvature
                       else "curvature-only" if curvature else "sentinel-only")
                print(f"{regime:19s} n={n:4d} {tag:15s} "
                      f"recall={row['mode_recall']:.2f} "
                      f"miss={row['missing_mass']:.4f} "
                      f"false={row['false_modes']} "
                      f"TV={row['weight_tv']:.4f} "
                      f"cyc={row['cycle_residual']} {row['seconds']:.1f}s",
                      flush=True)
    path = os.path.join(ARTIFACTS, "experiment_2d.json")
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
