#!/usr/bin/env python
"""Section 18 -- clean wall-clock and function-evaluation accounting.

Run on a quiet machine and with nothing else in flight: the comparison script
runs its methods back to back, so *ratios* there are sound, but absolute times
are inflated whenever other jobs share the host.  This script measures one case
at a time, repeats each method, and reports the per-stage split for ModeTree.

Run:  python ModeTree/experiments/timing.py
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
from modetree.samplers import annealed_langevin, reverse_diffusion  # noqa: E402
from modetree.tree import ModeTree, ModeTreeConfig            # noqa: E402

ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
N_LADDER = (8, 32, 100)
NUM_SAMPLES = 2000
REPEATS = 3


def timed(fn, repeats=REPEATS):
    best = float("inf")
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return best, result


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    rows = []
    for n in N_LADDER:
        model = B.map_vs_mass_1d(n)
        prior_std = model.prior_cov.diagonal().sqrt()
        clean = make_composer(model)

        def diffusion():
            field = ComposedField(DiffusedOracle(model), "gauss_jacobian")
            draws, info = reverse_diffusion(
                field, NUM_SAMPLES, levels=200, prior_std=prior_std,
                generator=torch.Generator().manual_seed(0))
            return info

        def langevin():
            field = ComposedField(DiffusedOracle(model), "fnpe")
            draws, info = annealed_langevin(
                field, NUM_SAMPLES, levels=100, steps_per_level=10,
                prior_std=prior_std, generator=torch.Generator().manual_seed(0))
            return info

        def discovery():
            field = ComposedField(DiffusedOracle(model), "gauss_jacobian")
            tree = ModeTree(field, clean, ModeTreeConfig(), prior_std=prior_std)
            return tree, tree.run()

        jac_time, jac_info = timed(diffusion)
        fnpe_time, fnpe_info = timed(langevin)
        tree_time, (tree, branches) = timed(discovery)
        results = [b.diagnostics["clean_result"] for b in branches]

        energy_time, posterior = timed(
            lambda: assemble(clean, results, laplace_plus_samples=0))
        plus_time, posterior_plus = timed(
            lambda: assemble(clean, results, laplace_plus_samples=32))
        sample_time, _ = timed(lambda: posterior_plus.sample(NUM_SAMPLES))

        total = tree_time + plus_time + sample_time
        row = {
            "n": n,
            "gauss_jacobian_seconds": jac_time,
            "gauss_jacobian_field_calls": jac_info["score_field_calls"],
            "gauss_jacobian_row_evaluations": jac_info["score_evaluations"],
            "fnpe_seconds": fnpe_time,
            "fnpe_field_calls": fnpe_info["score_field_calls"],
            "fnpe_row_evaluations": fnpe_info["score_evaluations"],
            "modetree_total_seconds": total,
            "modetree_discovery_seconds": tree_time,
            "modetree_energy_laplace_seconds": energy_time,
            "modetree_laplace_plus_seconds": plus_time,
            "modetree_sampling_seconds": sample_time,
            "modetree_field_calls": tree.field.oracle.score_calls,
            "modetree_jacobian_calls": tree.field.oracle.jacobian_calls,
            "modetree_newton_iterations": tree.counters["newton_iterations"],
            "modetree_over_fnpe": total / fnpe_time,
            "modetree_over_jac": total / jac_time,
            "branches": len(branches),
        }
        rows.append(row)
        print(f"n={n:4d}  JAC {jac_time:6.2f}s ({jac_info['score_field_calls']} calls)"
              f"  F-NPSE {fnpe_time:6.2f}s ({fnpe_info['score_field_calls']})"
              f"  ModeTree {total:6.2f}s"
              f"  [tree {tree_time:.2f} + energy/Laplace+ {plus_time:.2f}"
              f" + sample {sample_time:.2f}]"
              f"  ratio vs F-NPSE {row['modetree_over_fnpe']:.3f}", flush=True)
    path = os.path.join(ARTIFACTS, "timing.json")
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
