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

## Results

(filled in below as runs complete)
