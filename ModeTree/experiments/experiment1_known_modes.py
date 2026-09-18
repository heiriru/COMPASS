#!/usr/bin/env python
"""Section 25 -- the gating experiment: known modes + oracle clean score.

Given the true mode locations as seeds and the *exact* compositional clean
score, can we recover

  * the MAP,
  * the relative mode log heights (score line integration + cycle consistency),
  * the mode masses (structured Laplace, then Laplace+),

for n = 1, 2, 4, 8, 16, 32, 64, 100?

Nothing about mode discovery is exercised here.  If this fails, ModeTree-JAC
cannot work and the branch-discovery milestones are not worth implementing.

Run:  python ModeTree/experiments/experiment1_known_modes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json      # noqa: E402
import math      # noqa: E402
import time      # noqa: E402

import torch     # noqa: E402

from modetree import benchmarks as B                       # noqa: E402
from modetree.composition import make_composer                 # noqa: E402
from modetree.laplace import assemble                      # noqa: E402
from modetree.refine import refine_mode                    # noqa: E402

N_LADDER = (1, 2, 4, 8, 16, 32, 64, 100)
ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "artifacts")


def cases():
    """(name, builder, grid window) -- every regime the task asks for."""
    yield "symmetric", lambda n: B.bimodal_1d(n, "symmetric"), (-12, 12)
    yield "nearly_tied", lambda n: B.bimodal_1d(n, "nearly_tied"), (-12, 12)
    yield "separated", lambda n: B.bimodal_1d(n, "separated"), (-16, 16)
    yield "vanishing", lambda n: B.bimodal_1d(n, "vanishing"), (-12, 12)
    yield "map_vs_mass", lambda n: B.map_vs_mass_1d(n), (-12, 12)
    yield "tilt_0.5", lambda n: B.tilted_two_mode_1d(n, delta=0.5), (-12, 12)
    yield "tilt_2.0", lambda n: B.tilted_two_mode_1d(n, delta=2.0), (-12, 12)
    yield "narrow_broad", lambda n: B.bimodal_1d(n, "narrow_broad"), (-12, 12)
    yield "empty_valley", lambda n: B.empty_valley_1d(n), (-20, 20)


def jensen_shannon(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * (a[mask] / b[mask]).log()).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_case(name, builder, window, n, laplace_plus_samples=32, seed_jitter=0.35):
    model = builder(n)
    ref = B.reference_1d(model, window[0], window[1])
    composer = make_composer(model)

    # only modes carrying >= 1e-3 of the mass count as "relevant"
    relevant = ref.mode_masses >= 1e-3
    true_modes = ref.modes[relevant]
    true_masses = ref.mode_masses[relevant]
    true_masses = true_masses / true_masses.sum()
    true_heights = ref.mode_log_heights[relevant]

    started = time.perf_counter()
    # seeds are the true modes *displaced*, so the refiner has to do real work
    results = []
    for k in range(true_modes.shape[0]):
        seed = true_modes[k] + seed_jitter * torch.sign(
            torch.tensor(1.0 - 2.0 * (k % 2))
        )
        results.append(refine_mode(composer, seed, tol=1e-10, max_iter=200))
    refine_time = time.perf_counter() - started

    started = time.perf_counter()
    posterior = assemble(
        composer, results, multiple_paths=True,
        laplace_plus_samples=laplace_plus_samples,
    )
    assemble_time = time.perf_counter() - started

    # ablation: Laplace+ without the basin guard (the estimator as specified)
    unguarded = assemble(
        composer, results, multiple_paths=False,
        laplace_plus_samples=laplace_plus_samples, laplace_plus_kwargs={
            "basin_guard": False},
    )

    found = torch.stack([b.global_state for b in posterior.branches]).reshape(-1)
    # align the found branches with the reference modes by position
    order = torch.tensor([
        int(torch.argmin((true_modes.reshape(-1) - f).abs())) for f in found
    ])
    sigma_total = float(ref.covariance.diagonal().sum().sqrt())

    # exactly-tied mode heights make the MAP identity genuinely ambiguous;
    # report it as such rather than letting a 1e-16 tie-break decide.
    energies_sorted = torch.sort(
        torch.tensor([b.energy for b in posterior.branches], dtype=torch.float64),
        descending=True).values
    gap = float(energies_sorted[0] - energies_sorted[1]) if len(energies_sorted) > 1 \
        else float("inf")
    map_ambiguous = gap < 1e-9
    ref_gap = (float(true_heights[0] - true_heights[1])
               if true_heights.numel() > 1 else float("inf"))
    if map_ambiguous:
        e_map = float(
            (true_modes.reshape(-1) - posterior.map_global.reshape(())).abs().min()
        )
    else:
        e_map = float((posterior.map_global.reshape(()) - ref.map.reshape(())).abs())
    mode_error = float((found - true_modes.reshape(-1)[order]).abs().max())

    # heights: compare reconstructed energy differences to the exact ones
    est_energy = torch.tensor([b.energy for b in posterior.branches], dtype=torch.float64)
    ref_energy = true_heights[order]
    height_error = float(
        ((est_energy - est_energy[0]) - (ref_energy - ref_energy[0])).abs().max()
    )

    w_hat = posterior.weights[torch.argsort(order)]
    w_plus = (posterior.weights_plus[torch.argsort(order)]
              if posterior.weights_plus is not None else None)
    weight_mae = float((w_hat - true_masses).abs().mean())
    tv = 0.5 * float((w_hat - true_masses).abs().sum())
    js = jensen_shannon(w_hat, true_masses)
    weight_mae_plus = tv_plus = js_plus = None
    if w_plus is not None:
        weight_mae_plus = float((w_plus - true_masses).abs().mean())
        tv_plus = 0.5 * float((w_plus - true_masses).abs().sum())
        js_plus = jensen_shannon(w_plus, true_masses)

    cyc = posterior.energy_graph.cycle_residuals
    return {
        "case": name, "n": n,
        "n_true_modes": int(true_modes.shape[0]),
        "n_branches": len(posterior.branches),
        "e_map": e_map,
        "e_map_sigma": e_map / sigma_total,
        "mode_position_error": mode_error,
        "score_norm_max": max(r.score_norm for r in results),
        "newton_iterations": sum(r.iterations for r in results),
        "height_error": height_error,
        "weight_mae": weight_mae, "tv": tv, "js": js,
        "weight_mae_plus": weight_mae_plus, "tv_plus": tv_plus, "js_plus": js_plus,
        "weights": w_hat.tolist(),
        "weights_plus": None if w_plus is None else w_plus.tolist(),
        "tv_plus_unguarded": 0.5 * float(
            (unguarded.weights_plus[torch.argsort(order)] - true_masses).abs().sum()),
        "accepted_min": min(
            b.diagnostics["laplace_plus"]["accepted"] for b in posterior.branches),
        "weights_true": true_masses.tolist(),
        "map_index": int(order[posterior.map_index]),
        "mass_index": int(order[posterior.mass_index]),
        "map_is_mass": int(posterior.map_index) == int(posterior.mass_index),
        "true_map_index": 0,
        "true_mass_index": int(torch.argmax(true_masses)),
        "map_branch_correct": bool(map_ambiguous or int(order[posterior.map_index]) == 0),
        "map_ambiguous": bool(map_ambiguous),
        "reference_height_gap": ref_gap,
        "path_disagreement": max(posterior.energy_graph.path_disagreement.values())
        if posterior.energy_graph.path_disagreement else 0.0,
        "reverse_residual": max(posterior.energy_graph.reverse_residual.values())
        if posterior.energy_graph.reverse_residual else 0.0,
        "cycle_residual": max(abs(v) for v in cyc.values()) if cyc else None,
        "ess_min": min((b.ess for b in posterior.branches if b.ess is not None),
                       default=None),
        "ess_fraction_min": min(
            (b.diagnostics["laplace_plus"]["ess_fraction"]
             for b in posterior.branches if b.ess is not None), default=None),
        "max_importance_weight": max(
            (b.max_weight for b in posterior.branches if b.max_weight is not None),
            default=None),
        "log_weight_variance": max(
            (b.log_weight_variance for b in posterior.branches
             if b.log_weight_variance is not None), default=None),
        "log_correction": [b.log_correction for b in posterior.branches],
        "refine_seconds": refine_time,
        "assemble_seconds": assemble_time,
        "score_evaluations": posterior.diagnostics["score_evaluations"],
    }


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    rows = []
    for name, builder, window in cases():
        for n in N_LADDER:
            row = run_case(name, builder, window, n)
            rows.append(row)
            print(
                f"{name:14s} n={n:4d}  modes={row['n_true_modes']}  "
                f"|S|={row['score_norm_max']:.1e}  "
                f"e_MAP/s={row['e_map_sigma']:.2e}  "
                f"dE={row['height_error']:.2e}  "
                f"TV={row['tv']:.4f}"
                + (f" TV+={row['tv_plus']:.4f}" if row['tv_plus'] is not None else "")
                + (f"  TV+ng={row['tv_plus_unguarded']:.4f}" if row['tv_plus_unguarded'] is not None else "")
                + (f"  ESS={row['ess_fraction_min']:.2f}"
                   if row['ess_fraction_min'] is not None else "")
            )
    path = os.path.join(ARTIFACTS, "experiment1_known_modes.json")
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nwrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
