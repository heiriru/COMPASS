# ModeTree-JAC

Multimodal compositional inference for COMPASS: a deterministic, branch-based
posterior reconstruction that keeps the GAUSS/JAC cost profile and fixes its
failure on multimodal global posteriors.

* `TECHNICAL_NOTE.md` — Milestone 0: exactly what the existing COMPASS
  composition code does (formulas, shapes, files), read out of the source.
* `REPORT.md` — algorithm, derivation, results, failure modes, and the
  recommendation.
* `artifacts/TABLES.md` — every number, regenerated from the JSON artifacts.
* `artifacts/figures/` — figures 1-6.

## Layout

| module | milestone | contents |
|---|---|---|
| `modetree/benchmarks.py` | A | Gaussian-mixture factor models, exact tall posterior, 1-D/2-D grid references |
| `modetree/composition.py` | B | exact clean compositional score, arrowhead curvature, Schur elimination |
| `modetree/refine.py` | C | Newton mode refinement, batched probe refinement, mode-equivalence metric |
| `modetree/energy.py` | D, E | score line integration, multi-path estimates, cycle-consistent energies |
| `modetree/laplace.py` | F, G | structured Laplace evidence, Laplace+, posterior assembly |
| `modetree/diffusion.py` | L, M | diffusion-time composed fields, `row_jacobian_factor`, sparse JAC |
| `modetree/tree.py` | H, I, J, K, N | continuation, curvature spawning, sentinels, branch management |
| `modetree/samplers.py` | — | reverse-SDE and annealed-Langevin baselines |
| `src/compass/priors.py` | section 3 | prior-object interface for `MultiObsSampler` |

## Running

Every script caps itself at three threads before importing torch, per the
repository `CLAUDE.md`.

```bash
export PYTHONPATH=ModeTree:src

# the gate: known modes + oracle clean score, n = 1..100
python ModeTree/experiments/experiment1_known_modes.py

# the headline comparison against uncorrected / gauss_jacobian / F-NPSE
python ModeTree/experiments/compare_methods.py          # --quick for a smoke run

# discovery ablations, 2-D and the remote small mode, hierarchical, score error
python ModeTree/experiments/ablations.py
python ModeTree/experiments/experiment_2d.py
python ModeTree/experiments/experiment_hierarchical.py
python ModeTree/experiments/experiment_score_error.py

# figures and the markdown tables
python ModeTree/experiments/figures.py
python ModeTree/experiments/summarize.py

# tests
python -m pytest ModeTree/tests -q
```

## Using it

```python
from modetree.composition import make_composer
from modetree.diffusion import ComposedField, DiffusedOracle
from modetree.laplace import assemble
from modetree.tree import ModeTree, ModeTreeConfig

clean = make_composer(model)                       # exact clean composition
field = ComposedField(DiffusedOracle(model), "gauss_jacobian")
branches = ModeTree(field, clean, ModeTreeConfig()).run()
posterior = assemble(
    clean, [b.diagnostics["clean_result"] for b in branches],
    laplace_plus_samples=32,
)

posterior.map_global          # the highest-density mode
posterior.weights_plus        # mode masses
posterior.branches[i].global_covariance
posterior.sample(2000)        # draws from sum_b w_b N(g_b, P_b^-1)
```

A non-Gaussian prior goes into `MultiObsSampler.sample` directly; the
`(mean, scale)` tuple keeps working unchanged.

```python
from compass.priors import GaussianMixturePrior

sampler.sample(..., prior=GaussianMixturePrior(weights, means, covariances))
```
