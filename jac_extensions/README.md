# JAC extensions (branch `jac-extensions`, worktree `/export/data/rheinric/COMPASS-jacext`)

Five extensions of the Jacobian (JAC) composition, items 1.8–1.12 of the plan. Each one
is opt-in and removable on its own. With default arguments the production library is
bit-identical to the base commit `948bf37`: `python golden_all.py check` re-runs 9
correction rules × 2 observation counts on a randomized network and compares every
draw. (`golden_all.pt` is gitignored; regenerate it with `save` on `948bf37`.)

| item | what | where | switch | remove by deleting |
|---|---|---|---|---|
| 1.8 | N-level JAC/GH on nested trees | `src/compass/experimental/multilevel.py` | standalone sampler | module + `tests/test_experimental_multilevel.py` |
| 1.9 | covariance head instead of the Jacobian | `src/compass/experimental/covariance_head.py` | `sampler.set_covariance_head(head)` or env `COMPASS_COVARIANCE_HEAD=head.pt` | module, test, and the hook in `MultiObsSampler._jacobian_joint_precision` (3 lines) plus `set_covariance_head` / `_covariance_head` / `_head_joint_precision` |
| 1.10 | observation mini-batched JAC | `MultiObsSampler.sample(jacobian_row_batch=m)` | the argument (default None) | `_configure_row_batch`, `_row_batch_refresh`, `_row_batched_score`, the 2-line dispatch in `_get_score`, the `_row_batch_frozen` guard, `tests/test_jacobian_row_batch.py` |
| 1.11 | conservative score s = ∇E | `src/compass/experimental/conservative.py` | `enable_conservative(sbim)` before training | module + test |
| 1.12 | mode-aware JAC | `src/compass/experimental/mode_aware.py` | `mode_aware_sample(sampler, ...)` | module + test |

Experiment drivers live in the subfolders below. All of them claim a GPU through
`gpu.py`, which only takes a card that has none of the other users' processes and
enough free memory. Scripts that go through `experiments_arruda_schur/common.py` assert
that `compass` is imported from this worktree, because that module puts the main
checkout's `src` first on `sys.path`.

## 1.8 Multi-level JAC (`multilevel/`)

The rule is the N-level theorem from `docs/multilevel_composition/three_level_composition.tex`:

- **Upward pass, one per level.** Schur-eliminate each node's own block, then merge
  its children. The merge uses information form, `Q_v = P0_v + Σ_c (S_c − P0_v)`,
  and projects each child's information onto the PSD cone.
- **Downward pass.** A cascade of conditional means.

Cost is linear in the number of nodes. For the path priors, the closed-form Gaussian
`GaussianPathPrior` is implemented. A non-Gaussian path prior (e.g. a funnel) needs a
`terms(tree, levels, lam)` object; that is the open piece.

## 1.9 Covariance head (`covhead/`)

A post-hoc MLP, `K(y, x, λ) = L Lᵀ`, is fitted to a frozen score network by the Gaussian
NLL of the denoising residual. From it, `Λ_j = K + λ⁻² I`. Properties:

- symmetric and PSD by construction;
- no FD copies;
- for a Gaussian posterior, `K = C⁻¹` at every `(y, λ)`.

## 1.10 Mini-batched JAC (`large_n/`)

Each call refreshes `m` rows round-robin (score plus FD Jacobian). Every other row is
rebuilt from its cached λ-free natural parameters `K_j = Λ_j − λ⁻² I` and
`η_j = Λ_j D_j − λ⁻² y_j`. For a Gaussian single-observation posterior these are exact
constants. The cost per call drops from `n` rows to `m (D+2)` rows.

## 1.11 Conservative score (`conservative/`)

Uses the Salimans & Ho energy `E = −‖(1−c)(x − d)‖² / 2σ²` with `d = x + σ r_base`, so
the Jacobian is a Hessian. It equals the plain score where `J_d = 0`. Training uses the
math SDPA backend, because the fused kernels have no double-backward.

## 1.12 Mode-aware JAC (`mode_aware/`)

1. A discovery run, clustered by a BIC-selected Gaussian mixture.
2. One warm-started JAC run per mode, with `start_lambda = separation/6`.
3. Branch masses from the exact tall-data target, by PF-ODE (`certify_composition`)
   importance sampling against a factorized Gaussian proposal per mode.
   `log_prior_fn` supplies non-Gaussian priors.

## Results (2026-10-23)

Regression checks:
- `golden_all.py check`: ALL IDENTICAL after every library change.
- Full `tests/` (CPU): the same 13 failures as the base branch. These are the known
  tests for features missing from `src/`.

A pre-existing trap on the base branch: `pd_study/clamp_study.py` sets
`torch.set_default_dtype(float64)` at import, and
`test_conservative_clamp.py::test_on_a_mixture_only_...` imports it. Any test that
runs after it in the same process then runs in float64, which caused 68 spurious
failures. Run that test in its own process, or deselect it.

### 1.8 Multi-level JAC

Exact-score (linear-Gaussian oracle) results:
- The composed score is exact to ~1e-14 for 2-, 3- and 4-level trees.
- The N=2 case equals the production `gauss_jacobian`.
- The Heun + churn sampler recovers the joint 3- and 4-level posteriors near the MC
  floor. For dims [2,2,1], 3×4 groups, 4000 draws: max mean error 0.022 sd, sd
  ratio in [0.98, 1.05], max correlation error 0.041.
- Dropping the mid-level prior factors gives correlation error 0.43 and mean error
  1.0 sd. The checks have power.

Trained path network (`multilevel/bench3.py`; 200k path simulations, h64d4,
preconditioned):
- At 4 groups × 4 obs, JAC-FD matches the exact-score floor:
  - g/h/l mean RMSE 0.055/0.080/0.081 vs 0.051/0.079/0.081;
  - log-sd error 0.03/0.02/0.02.
  - GH-pilot is worse on h/l (0.118/0.115).
  - Refresh 9 costs a little (0.094/0.095) at 1/3 the time.
- Without a denoise clamp, every network arm diverges from 16 obs per group up,
  while the oracle arm does not. A bad tail score gets amplified (n_k − 1)-fold by
  the group factor. The production sampler's default `denoise_clamp=5`, applied to
  row and composed denoisers, is now in `MultiLevelSampler`. Clamped results:
  `multilevel/artifacts/bench3/results_clamp5_report.md`.

### 1.9 Covariance head

Head information error vs exact on the GM hierarchical net (median relative error
vs the exact Tweedie information):

| λ | FD-Jacobian | head |
|---|---|---|
| 0.03 | 0.47 | 0.07 |
| 0.1 | 0.27 | 0.11 |
| 0.5 | 0.20 | 0.06 |
| 1 | 0.20 | 0.03 |
| 2 | 0.69 | 0.02 |
| 5 | 3.45 | 0.09 |

At high noise FD fails through the cancellation in λ²(I + λ²J), not through the
Jacobian itself (6% error there).

GM composition, production hierarchical net, JAC (W1/sd and branch-mass error,
FD → head):
- Better in all 8 configs. Asymmetric n=8: 0.62/−0.22 → 0.36/−0.07; n=16:
  1.37/−0.48 → 1.09/−0.35; symmetric n=8: 0.39/−0.15 → 0.25/−0.05.
- 3.1–4.5× faster (CPU).
- Width ratio is sometimes slightly larger.

Training cost: ~20 min CPU for 20k steps.

### 1.10 Mini-batched JAC (`jacobian_row_batch`)

Exact for Gaussian rows (test). Arruda toy, J = 1000, 5 datasets:

| arm | KL | mean err | s/dataset | peak GPU |
|---|---|---|---|---|
| full JAC-FD r9 | 14.6 | 1.49 sd | 1564 | 6.1 GB |
| rb20 | 6.9 | 0.94 sd | 683 | 3.6 GB |
| rb100 | 1.7 | 0.50 sd | 889 | 3.9 GB |

The accuracy gain is most likely the stale Gaussian cache filtering the network's
low-noise error, which is harmless only because this toy is Gaussian. The robust
claim is speed and memory.

AR(1):

| grid | arm | mean err | width | s/dataset |
|---|---|---|---|---|
| 16×16 | rb20 | 1.03 sd | 1.43 | 27 |
| 16×16 | stored JAC-r9 | 1.55 sd | 1.42 | 28 |
| 16×16 | stored GH | 1.30 sd | — | 102 |
| 64×64 | rb20 | 8.2 sd | 3.4 | 649 |
| 64×64 | stored JAC-r9 | 13.0 sd | 3.0 | 446 |
| 64×64 | stored GH | 14.8 sd | — | 4248 |

At 64×64 every arm is network-limited, and with D=4 and a small network the
per-row composition linear algebra, not the network, dominates wall clock.
`rb100` degrades on AR(1): widths 3.4 (16×16) and 9.0 (64×64). Use ~J/20.

### 1.11 Conservative score

Exactly symmetric Jacobian (test).

GM, 2 seeds per arm, same recipe:
- At λ = 0.03, the score error is 0.38 vs 0.90–1.06 and the Jacobian error 0.34 vs
  0.66–0.80. From λ ≥ 0.1 the two arms are equal.
- JAC composition: mean |branch-mass error| over 8 configs is 0.18 and 0.10
  (conservative) vs 0.52 and 0.26 (plain). W1 is within seed spread (0.71/1.24 vs
  1.68/0.87).
- Caveat: the GM "none" trunk already has an exactly diagonal (hence symmetric)
  latent Jacobian, so the antisymmetry claim itself is untested here. The gain comes
  from better low-noise accuracy plus the g-l coupling the energy adds.
- Cost: training ~3× (double backward on the math SDPA backend); inference needs one
  extra backward pass.

### 1.12 Mode-aware JAC

GM, exact (oracle) scores, `artifacts_v2`, 2 replicates:
- Composing within each prior component (`mode_prior_fn`) cuts W1 by 1.9–3.5× in all
  8 configs. Examples: asymmetric n=8 0.95 → 0.33; symmetric n=16 1.11 → 0.35;
  asymmetric n=64 4.48 → 2.06.
- Mode-aware without the component prior is worse than plain JAC everywhere. The
  gain is the component prior, which removes the moment-matched-Gaussian prior
  interface error that pulls the branches outward (JAC put them at ±1.7 vs exact
  ±1.17).
- Masses from PF-ODE importance sampling are right at n ≤ 16 (e.g. 0.261/0.739 vs
  0.25/0.75) and noisy at n ≥ 32 (ESS 1–4%).

With the trained hierarchical net it fails:
- The PF-ODE masses come from network densities. Per-row density error summed over n
  rows makes them degenerate (1.0 vs 1e-33), and all draws land in one branch.
- The open piece is a branch-mass estimator that does not sum n network log
  densities.
