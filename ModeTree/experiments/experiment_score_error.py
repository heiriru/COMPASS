#!/usr/bin/env python
"""Section 12 / Q6 -- how does per-row score error accumulate with n?

With ``hat s_i = s_i + eps_i``,

    hat S_0 - S_0 = sum_i eps_i - (n-1) eps_prior

so a *systematic* row error grows like ``O(n)`` while a zero-mean one grows like
``O(sqrt(n))``.  That is arithmetic; what matters for the method is what the
growth does to the MAP, the mode weights and the mode count, because the
posterior itself contracts like ``1/sqrt(n)``.

No trained network is needed to answer that: the error is injected directly into
the oracle row scores, in three shapes,

  ``systematic``  a fixed direction, the same for every row (network bias)
  ``random``      i.i.d. per row, zero mean (estimation noise)
  ``state``       a fixed smooth function of the state, the same for every row
                  (the realistic case: a network's error is a function, not a
                  constant, so it is systematic *and* it deforms the field)

and the empirical exponent ``error ~ n^alpha`` is fitted for each.

Run:  python ModeTree/experiments/experiment_score_error.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json    # noqa: E402
import math    # noqa: E402

import torch   # noqa: E402

from modetree import benchmarks as B                 # noqa: E402
from modetree.composition import (                   # noqa: E402
    BatchedCleanComposer, BatchedOracleRowSource, make_composer,
)
from modetree.laplace import assemble                # noqa: E402
from modetree.refine import refine_mode              # noqa: E402

ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
N_LADDER = (1, 2, 4, 8, 16, 32, 64, 100, 128, 256)


class PerturbedRowSource(BatchedOracleRowSource):
    """Oracle row scores plus a controlled error, shared across the n ladder.

    The per-row draws are generated once from a fixed seed and then *sliced*, so
    the n = 100 model reuses the first 100 rows of the n = 256 draw.  Otherwise
    the scaling measured would be confounded with a fresh random realisation at
    every n.
    """

    def __init__(self, model, shape, magnitude, seed=0, max_rows=256):
        super().__init__(model)
        self.shape = shape
        self.magnitude = magnitude
        generator = torch.Generator().manual_seed(seed)
        self._row_draw = torch.randn(
            max_rows, self.d_g + self.d_l, generator=generator, dtype=torch.float64
        )
        self._direction = torch.ones(
            self.d_g + self.d_l, dtype=torch.float64
        ) / math.sqrt(self.d_g + self.d_l)

    def scores_all(self, rows_state):
        base = self.rows.score(rows_state)
        if self.shape == "systematic":
            return base + self.magnitude * self._direction
        if self.shape == "random":
            return base + self.magnitude * self._row_draw[: self.n]
        if self.shape == "state":
            # a smooth, bounded, row-independent deformation: the shape a
            # network's systematic error actually has
            return base + self.magnitude * torch.tanh(rows_state) * self._direction
        raise ValueError(self.shape)

    def hessians_all(self, rows_state):
        base = self.rows.hessian(rows_state)
        if self.shape == "state":
            derivative = self.magnitude * (1.0 - torch.tanh(rows_state) ** 2)
            return base + torch.diag_embed(derivative * self._direction)
        return base


def fit_exponent(ns, errors):
    x = torch.log(torch.tensor(ns, dtype=torch.float64))
    y = torch.log(torch.tensor(errors, dtype=torch.float64).clamp_min(1e-300))
    design = torch.stack([x, torch.ones_like(x)], dim=-1)
    solution = torch.linalg.lstsq(design, y.unsqueeze(-1)).solution.reshape(-1)
    return float(solution[0])


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    probe = torch.tensor([[1.2], [-0.4], [2.0], [-2.0]], dtype=torch.float64)
    rows = []
    for shape in ("systematic", "random", "state"):
        for magnitude in (0.01, 0.05):
            # The zero-mean case is a *statement about the distribution* of the
            # error, so one realisation says nothing: sum_i eps_i for a single
            # fixed draw is one sample of an O(sqrt(n)) quantity and can come
            # out anywhere.  Average the norm over independent draws.
            error_seeds = range(8) if shape == "random" else range(1)
            errors, map_errors, weight_errors, counts = [], [], [], []
            spread = []
            for n in N_LADDER:
                model = B.tilted_two_mode_1d(n, delta=0.5)
                exact = make_composer(model)
                realisations = []
                for seed in error_seeds:
                    candidate = BatchedCleanComposer(
                        PerturbedRowSource(model, shape, magnitude, seed=seed)
                    )
                    delta = candidate.score(probe) - exact.score(probe)
                    realisations.append(
                        float(torch.linalg.vector_norm(delta, dim=-1).mean())
                    )
                    if seed == 0:
                        perturbed = candidate
                realisations = torch.tensor(realisations, dtype=torch.float64)
                errors.append(float(realisations.mean()))
                spread.append(float(realisations.std(unbiased=False)))

                ref = B.reference_1d(model, -12, 12, points=100_001)
                keep = ref.mode_masses >= 1e-3
                true_masses = ref.mode_masses[keep]
                true_masses = true_masses / true_masses.sum()
                mode_seeds = ref.modes[keep]
                results, kept = [], []
                for k, seed in enumerate(mode_seeds):
                    result = refine_mode(perturbed, seed, tol=1e-11, max_iter=200)
                    if result.diagnostics["is_local_maximum"]:
                        results.append(result)
                        kept.append(k)
                counts.append(len(results))
                if not results:
                    map_errors.append(float("nan"))
                    weight_errors.append(float("nan"))
                    continue
                posterior = assemble(perturbed, results, laplace_plus_samples=0)
                map_errors.append(float(
                    (posterior.map_global.reshape(()) - ref.map.reshape(())).abs()
                ))
                aligned = torch.zeros(true_masses.numel(), dtype=torch.float64)
                for b, branch in enumerate(posterior.branches):
                    k = int(torch.argmin(
                        (mode_seeds.reshape(-1) - branch.global_state.reshape(())).abs()
                    ))
                    aligned[k] += float(posterior.weights[b])
                weight_errors.append(
                    0.5 * float((aligned - true_masses).abs().sum())
                )
            row = {
                "shape": shape, "magnitude": magnitude,
                "n": list(N_LADDER), "score_error": errors,
                "score_error_spread": spread,
                "alpha_score": fit_exponent(N_LADDER, errors),
                "map_error": map_errors, "weight_tv": weight_errors,
                "modes_found": counts,
            }
            rows.append(row)
            print(f"{shape:11s} mag={magnitude:.3f}  alpha={row['alpha_score']:+.3f}  "
                  f"|dS|(n=1)={errors[0]:.3e} (n=100)={errors[7]:.3e}  "
                  f"e_MAP(n=100)={map_errors[7]:.3e}  TV(n=100)={weight_errors[7]:.4f}",
                  flush=True)
    path = os.path.join(ARTIFACTS, "score_error_scaling.json")
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
