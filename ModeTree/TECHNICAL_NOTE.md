# ModeTree-JAC — Milestone 0: what the existing COMPASS code actually does

Everything below was read out of the source on branch `modetree`
(HEAD = `4e68817 Finished compositional score modelling`), not assumed.
File/line references are to that state of the tree.

## 1. Diffusion formulation

`src/compass/SDE.py` defines `VESDE` and `VPSDE`, but *every* sampler and the
probability-flow ODE work in the rescaled state

    y = x / alpha(t),      lambda(t) = sigma(t) / alpha(t)

in which both SDEs are variance-exploding:

    y_t = y_0 + lambda_t * eps,      eps ~ N(0, I).

For the VESDE (the default, `sigma=25`) `alpha == 1`, so `lambda_t == sigma_t ==
sqrt((sigma^(2t) - 1) / (2 log sigma))`. The prompt's assumed
`y_t = y_0 + lambda_t eps` is therefore **confirmed**, and `var_t` in the
composition code is always `lambda_t^2`.

`MultiObsSampler._raw_row_scores` (line 1489) returns the score **in y
coordinates**: it maps `z -> x = z * alpha` on the latent block, calls the
network, applies `output_scale_function`, and multiplies by `alpha`.

## 2. Tensor shapes and the global/local partition

The multi-observation state is

    z : (n_rows, n_samples, n_features)

`n_rows` is the number of observations, `n_samples` the number of posterior
draws carried in parallel, `n_features` the full token/feature width (observed
+ latent). `condition_mask` has the same shape; 1 = observed, 0 = latent.

* `self.hierarchy` — a list of feature indices that are **shared/global**. The
  state is kept synchronized across rows on these indices, and `x[:1, :, h]`
  is used as *the* shared value (`_compositional_score`, line 3403).
* `self.local_latent_indices` — latent feature indices not in `hierarchy`; one
  independent copy per row. This is the `l_i` of the prompt.
* `self.full_gaussian_features` — `hierarchy + local_latent_indices`, the
  ordering in which the joint (H+L)x(H+L) blocks are built. Global block first,
  so the arrowhead partition `[[A, B], [B^T, C]]` is exactly
  `[:h, :h] / [:h, h:] / [h:, h:]`.

So the COMPASS row state is `y_i = (g, l_i)` as the prompt assumes, with
`d_g = len(hierarchy)` and `d_l = len(local_latent_indices)`.

## 3. Prior handling

`_resolve_prior` (1848) accepts **only** `prior=(mean, scale)` where `scale` is
either a per-coordinate std vector or a full covariance matrix. It produces
`self.prior_mean`, `self.prior_std`, `self.prior_covariance` and
`self.prior_precision_matrix`. There is no hook for a non-Gaussian prior — this
is the limitation section 3 of the task asks to lift.

`_diffused_gaussian_prior_score(theta_y, t)` (3293):

* for covariance-aware corrections: `-solve(Sigma_prior + lambda_t^2 I, y - mu)`
* otherwise the diagonal VP-aware form `alpha * (-(x - mu_x)/(alpha^2 s^2 + sigma^2))`.

Both are the score of the *diffused* prior `N(mu, Sigma_prior + lambda_t^2 I)`
in y coordinates. `_resolve_local_prior` (1889) handles an optional Gaussian
prior on the local block, used only for the denoise clamp
(`_clamp_local_scores`, 3313), not in the composition algebra.

## 4. The composition rules

`_compositional_score` (3357) is the single entry point; `VALID_CORRECTIONS`
lists 15 rules. The ones this project must beat:

* `uncorrected`: `composed = (1-n) * prior_score + sum_j s_j`.
* `fnpe`: `(1-n)(1-t) * prior_score + sum_j s_j` — the Geffner bridging score,
  valid only with `method="langevin"` (`_langevin_sampler`, 3903).
* `gauss_hierarchical` / `gauss_jacobian` (the `SCHUR_GAUSSIAN_CORRECTIONS`
  path): precision-weighted composition,

      Lambda(t)      = sum_j Lambda_j(t) + (1-n) Lambda_prior(t)
      composed       = Lambda(t)^-1 [ sum_j Lambda_j s_j + (1-n) Lambda_prior s_prior ]

  with `Lambda_prior(t) = Sigma_prior^-1 + lambda_t^-2 I`
  (`_prior_precision_at`, 2401) and the per-observation effective *global*
  precision from `_effective_global_factors` (2275).

### 4.1 `gauss_jacobian` — the exact chain, confirmed from source

1. `_row_score_jacobian` (1507): `len(feature_order)` forward-mode `torch.func.jvp`
   passes with a one-hot tangent in every row at once, under
   `sdpa_kernel([SDPBackend.MATH])` (the fused attention kernel has no forward
   AD rule). Returns `(n, samples, D, D)` in **float64**, `D = H + L`.
   Cost is `D` JVPs per refresh, independent of `n`.
2. `_jacobian_joint_precision` (2135): Tweedie second order in y coordinates,

       Sigma_t,j(y) = lambda^2 ( I + lambda^2 grad s_j(y) )
       Lambda_j(t)  = Sigma_t,j^-1

   implemented spectrally: `H_j = -grad s_j`, symmetrized
   (`0.5*(H + H^T)`), eigendecomposed (`_batched_eigh`, chunked to avoid the
   CUDA eigh workspace blow-up), eigenvalues clamped into
   `[0, (1 - floor)/lambda^2]` with `floor = jacobian_curvature_floor = 1e-3`,
   then `Lambda_j = 1/(lambda^2 (1 - lambda^2 * eig))`. The prompt's
   `Sigma_B,j = lambda^2 (I + lambda^2 J)` is **confirmed verbatim**.
3. `_effective_global_factors` (2275) caches the *lambda-free* information
   `Lambda_j(t) - lambda^-2 I` across `jacobian_refresh` calls, re-adds
   `lambda^-2 I` at the current time, adds `_global_block_adaptation`, then
   calls `_schur_factors`.
4. `_schur_factors` (2248): Cholesky of the joint precision, `cholesky_solve`
   to the joint covariance, **effective global precision = inverse of the
   global block of the covariance** (i.e. the marginal precision, which equals
   the Schur complement `A - B C^-1 B^T`), and the cross factor
   `R = Cov_lg * P_g` used for the local cross-correction
   `s_l <- s_l + R (s_composed - s_g,j)`.
5. `_project_information` (2409): projects each `I_j = Lambda_j - Lambda_prior`
   onto the PSD cone so the composed `Lambda(t) = Lambda_prior + sum_j I_j` is
   SPD by construction. Diagnostics counters `_pd_*`.
6. `_solve_composed_global` (2500): float64 solve with a last-resort eigenvalue
   floor at `pd_floor * scale`.

Note the natural-parameter form `Q_i = Lambda_i - lambda^-2 I`,
`q_i = Lambda_i (y_i + lambda^2 s_i) - lambda^-2 y_i` from the prompt is
*equivalent to*, but not literally the code's parameterisation: COMPASS keeps
`(Lambda_j, s_j)` and forms the weighted sum of scores rather than of means.
`Q_i` is exactly the cached `information` in step 3. ModeTree reuses the code's
parameterisation; it does not introduce a second one.

### 4.2 Existing MAP / Newton machinery

`newton_map_estimate` (1263) already does damped/trust-region Newton on the
arrowhead system: `_map_curvature_blocks` (1560) builds `(A, B_i, C_i)`,
`_project_map_information` (1606), `_arrow_newton_update` (1629) does the Schur
elimination `P_g = A - sum_i B_i C_i^-1 B_i^T` with `torch.linalg.solve`
(never an explicit inverse), `_map_merit` / `_trust_region_scale` handle
globalization. **ModeTree reuses this structure rather than reinventing it**;
the ModeTree refiner is the same arrowhead Newton run on the *clean* composed
score with multiple seeds.

### 4.3 Other reusable pieces

* `estimate_posterior_moments` / `_estimate_posterior_precision` (2584-2706).
* `certify_composition` (2796) — probability-flow-ODE importance weights and
  ESS against the exact tall target. This is the existing "is the composition
  right" instrument and the natural comparator for the Laplace+ ESS numbers.
* `composition_curl` (3001) — measures non-conservativity of the composed
  field. Directly relevant to Milestone E (cycle residuals).
* `PFODE.py` — exact log-density along the probability-flow ODE, available for
  the Milestone O evidence auditor.
* `composition_batch_size` (line 275) applies only to the true-minibatch rules
  (`damping`, `minibatch_damped`); it is *not* a general batching knob.

## 5. What the exact clean (t=0) composition looks like in this model

With `p(g)` the global prior, row factors `L_i(g, l_i)` and the row posteriors
`p(g, l_i | x_i) ∝ p(g) p(l_i | g) L_i`, the tall posterior is

    p(g, l_1..l_n | X) ∝ p(g)^{1-n} prod_i p(g, l_i | x_i)

so, exactly as the prompt states and as `uncorrected` implements at t=0,

    S_0,g   = sum_i s_i,g - (n-1) s_prior,g
    S_0,l_i = s_i,l_i

with `s_prior` the score of the **marginal** global prior `p(g)` — the
conditional prior `p(l_i|g)` appears once per row inside `s_i` and needs no
extra correction. This is verified numerically in
`tests/test_clean_composition.py` rather than assumed.

## 6. Testing conventions

`tests/` uses plain pytest, tiny CPU-only models, `torch.manual_seed`, and
analytic references (`test_multiobs_analytic.py`,
`test_gauss_jacobian_composition.py`). `tests/conftest.py` sets the CPU cap.
ModeTree follows the same style; its tests live in `ModeTree/tests/` and run
with `pytest ModeTree/tests`.

Per `CLAUDE.md`, every standalone script here calls
`modetree.runtime.configure_cpu_limit(3)` **before** importing torch.
