# Making hierarchical MAP exact — implementation and validation plan

> **Status.** Phases 0, 1 and most of 2 are implemented as a *separate*
> estimator, `MultiObsSampler.newton_map_estimate`, so it can be A/B'd against
> `map_estimate` (which is behaviourally unchanged). Gates 1-3 pass; see
> `README.md` for the numbers. Still open: multistart ranking (Phase 2),
> Gate 5 (score-error robustness), Gate 6 (GPU `Partial_Pooling`), and Phase 4
> (marginal MAP / Neyman-Scott).

## Context

`MultiObsSampler.map_estimate` (`src/compass/MultiObsSampler.py:671`) recovers the joint mode
of `(g, l_1..l_n)` by annealed score ascent. It is the primary point estimate reported by
`Partial_Pooling` and by several tutorials. It is currently inaccurate in a way that grows
with the number of observations, and the failure is **entirely in the optimizer** — it
reproduces with an exact analytic score and zero network error.

Measured, from `tutorials/output/annealed_score_ascent/comparison/all_metrics.csv`
(exact `ExactSharedLocalScore`, 20 repeats per row, mean over repeats):

| n  | global err /σ | local rmse /σ |
|----|---------------|---------------|
| 2  | 0.05          | 0.04          |
| 10 | 0.54          | 0.32          |
| 25 | 1.22          | 0.53          |
| 50 | 1.65          | 0.51          |

and from `tutorials/output/annealed_score_ascent/map_cache/exact_score_global_map_metrics.csv`
(prior `mu_g=-2.5, sigma_g=0.3`):

| n   | compass_map | analytic_map | err /σ |
|-----|-------------|--------------|--------|
| 5   | -2.5994     | -2.6005      | 0.007  |
| 50  | -2.5981     | -2.6430      | 0.73   |
| 200 | **-4.0000** | -2.6312      | **43.5** |

`-4.0` is exactly `mu_g - 5*sigma_g`, the `denoise_clamp=5.0` box edge. At n=200 the estimate
is pinned against the clamp.

Both `old_score` (flat `PFODE.map_estimate`) and `joint_score`
(`ModelTransfuser._shared_then_local_map`) fail near-identically, so the defect is not in the
block strategy — it is in the shared ascent step.

### Root cause

Two compounding mechanisms, both fixed by the same change.

**1. The step is preconditioned with the wrong curvature.** `MultiObsSampler.py:959` and `:967`
both take `z <- z + lambda^2 * s(z,t)`. That is a Newton step for a Gaussian of variance
`lambda^2`. The true curvature of the `lambda`-smoothed log-posterior is `(Sigma_0 + lambda^2)^-1`,
so the correct step is `(Sigma_0 + lambda^2) * s`. The undershoot factor is `1 + Sigma_0/lambda^2`.
With VESDE `sigma=25`, `eps=1e-3` we get `lambda_min ~ 0.0317`, `lambda^2 ~ 1e-3`; a local latent
with posterior sd 0.3 gives ~90x, sd 0.5 gives ~250x. Per-iteration contraction is 0.4–1%, so
`convergence_tol=1e-6` is unreachable and the loop silently exhausts
`max_iterations_per_level=50` at every one of 100 levels.

This mechanism is n-independent for the locals and *improves* with n for the globals, so it
cannot by itself explain the growth in the table above.

**2. Block Gauss–Seidel on an arrow system degrades with n.** The alternation at
`MultiObsSampler.py:948-976` relaxes locals, then globals. On an arrow-structured Hessian
(one shared block coupled to n independent local blocks) the spectral radius of block
Gauss–Seidel approaches 1 as n grows — the same centered-parameterization pathology that makes
Gibbs samplers mix badly in hierarchical models. The comment at `:949-953` already names the
symptom ("an observation-count dependent lag in tightly concentrated shared posteriors") and
tries to patch it by reordering the sweep.

The n=200 clamp pinning is the endgame of the two together: the iteration lags so far behind
the moving mode that the ascent runs to the box edge.

### Why the tests did not catch it

`tests/test_hierarchical_map.py:127` (and the other accuracy tests) do
`initial = make_joint(observations, expected[0], expected[1:])` — **they initialize the
optimizer at the analytic MAP**. They verify the iteration does not walk away from the answer;
they never test that it can find it. This is why `test_gaussian_corrected_joint_map_does_not_
shrink_at_high_observation_count` passes at n=50 while the tutorial fails at 1.65σ on the same n.

---

## Decisions taken

- **Optimizer first, marginal-MAP second.** At 1.65σ of optimizer error you cannot measure a
  Neyman–Scott bias. Phase 4 is gated on Phase 2 passing.
- **New behavior is the default**, with `step="tweedie"` to reproduce current results. Bump
  `Partial_Pooling/infer_partial_pooling.py:MAP_SETTINGS_VERSION` (currently 5) to invalidate
  MAP caches.
- **No GPU work without an explicit go-ahead** (shared host; `autocvd` policy in `AGENTS.md`).
  Phases 0–3 are CPU-only and take seconds.

---

## Phase 0 — Make the tests fail (no library changes)

Establishes the baseline and proves the mechanism before any fix.

1. In `tests/test_hierarchical_map.py`, add a `init_at_truth: bool` parametrization to the
   accuracy tests. The `False` variant initializes at the **prior mean** instead of `expected`.
   Reuse `MockSBIm`, `SharedLocalGaussianScore`, `analytic_shared_local_map`, `make_joint`
   unchanged.
2. Add `n in {2, 10, 50, 200}` to the sweep, and a narrow-prior variant
   (`sigma_global=0.3, sigma_local=0.4, sigma_x=0.2`, matching
   `tutorials/plot_local_vs_global_joint_map_validation.py:536`) so the tutorial's failure
   is reproduced inside the test suite.
3. Add an explicit clamp-pinning assertion: at n=200 the current code returns
   `mu_g - denoise_clamp*sigma_g` to within 1e-6.

**Exit criterion:** the new parametrizations fail at the tolerances recorded above, and the
existing `init_at_truth=True` cases still pass. If they do *not* fail, the diagnosis is wrong
and the rest of this plan is void — stop here.

---

## Phase 1 — Arrow-Newton step

Core change, in `MultiObsSampler.map_estimate`.

**Curvature.** Tweedie gives `H(z) = lambda^-2 I - lambda^-4 Sigma_t(z)` where `Sigma_t` is the
backward covariance. Two sources:

- **(A) From existing machinery.** `_effective_global_factors` (`:1382`) already computes
  `covariance_t = (posterior_precision_matrix + lambda^-2 I)^-1` at `:1426` — the per-observation
  joint `(g,l)` backward covariance. Available for the `COVARIANCE_GAUSSIAN_CORRECTIONS`.
- **(B) From `torch.func.jvp` at the iterate.** `Sigma_t = lambda^2 I + lambda^4 grad(s)`, exact
  and correction-agnostic. Costs `H+L` forward-mode passes; with `DEFAULT_MAP_STARTS=4`
  candidates this is negligible. Requires a non-`no_grad` variant of `_get_score` (`:1840`).

Implement (A) first (zero new autodiff, reuses everything), (B) behind
`curvature="jacobian"` as the accuracy upgrade for corrections without a covariance estimate.

**Assembly and solve.** Per observation `H_j = lambda^-2 I - lambda^-4 Sigma_{t,j}`; prior block
`H_prior = (Sigma_prior + lambda^2 I)^-1`. Then the arrow system, eliminated exactly as in
`tutorials/Compositional_Inference_Fix/New_Attempt/derivation.md` but applied to the Newton
system rather than the score composition:

```
S     = H_gg - sum_j H_gl,j H_ll,j^-1 H_lg,j        H_gg = sum_j [H_j]_gg + (1-n) H_prior
r     = gamma_g - sum_j H_gl,j H_ll,j^-1 gamma_l,j  gamma_g = composed score, gamma_l,j = s_j[l]
d_g   = S^-1 r
d_l,j = H_ll,j^-1 (gamma_l,j - H_lg,j d_g)
```

One `H x H` solve plus `n` tiny local solves per iteration — negligible against a network call.
This updates both blocks simultaneously and consistently, so the Gauss–Seidel lag disappears
by construction (mechanism 2) and the step carries the right curvature (mechanism 1).

**Safeguards.**

- `H` is PSD by construction when `Sigma_t <= lambda^2 I`. For estimated covariances that can
  fail; project the eigenvalues of `lambda^-2 Sigma_t` onto `[0, 1-delta]`. Blocks are small,
  so `eigh` is cheap. Follow the pattern of `_project_information` (`:1473`).
- **Merit function:** we are solving `s = 0`, so use `m(z) = ||s_composed(z)||^2`. Backtrack
  `tau <- tau/2` (cap 4) until `m` decreases. No log-density needed. Since `H >= 0` the Newton
  direction is always an ascent direction, so backtracking always terminates.
- **Trust region:** cap `||d||` at a multiple of `lambda`, the natural scale at that level.
  This is what prevents the runaway to the clamp boundary.

**Reporting.** Record per-level `converged: bool`, iterations used, backtracks, and trust-region
hits. `self.score_network_calls` (`:772`) is already tracked — surface it in the return path or
on the sampler so Gate 3 can read it.

---

## Phase 2 — MAP-specific defaults

- **`denoise_clamp`:** default to `None` for `map_estimate`. With a PSD-guaranteed Newton step,
  backtracking, and a trust region, the clamp is no longer load-bearing, and on a mode that
  legitimately sits in the prior tail it is a pure bias. Keep the parameter; when set, count
  activations as a diagnostic.
- **Richardson extrapolation.** The fixed point at level `lambda` is the mode of the *smoothed*
  posterior, biased by `O(lambda^2 * grad^3 log p)` — nonzero for any skewed posterior. From the
  last two levels: `z* = (lam_{K-1}^2 z_K - lam_K^2 z_{K-1}) / (lam_{K-1}^2 - lam_K^2)`.
  The gap `|z* - z_K|` is a free error bar on the smoothing bias. This costs nothing — both
  iterates already exist.
- **Multistart selection.** `map_estimate` accepts multiple candidates at `:745` but returns
  them all unranked. Rank with the existing `ModelTransfuser._hierarchical_candidate_scores`
  (already used by `Partial_Pooling._joint_map_dataset` and
  `tests/test_hierarchical_map.py:219`) rather than building a new objective.

---

## Phase 3 — Validation gates

All of Gates 1–5 are CPU-only and run in seconds to minutes.

**Gate 1 — analytic unit tests (decisive).** The Phase 0 tests must now pass at
`max|inferred - exact| / posterior_std < 0.05` for **all** `n in {2, 10, 50, 200}`, both prior
widths, and `init_at_truth in {True, False}`. Shared-coordinate synchronization `<= 1e-6` and
observed columns untouched, as today.

**Gate 2 — analytic sweep A/B (headline number).** Re-run
`tutorials/joint_vs_legacy_annealed_score_ascent.py` before and after; it uses an exact score
and needs no trained model. Compare `comparison/all_metrics.csv`.

- Accept: `global_error_analytic_sigma` and `local_rmse_analytic_sigma` below 0.05 for all
  `n in {2,5,10,25,50}`, **and flat in n** — the growth trend is the thing being fixed, so a
  uniform improvement that still grows with n is a fail.
- Also re-run `tutorials/plot_local_vs_global_joint_map_validation.py --map-timesteps 100` for
  `exact_score_global_map_metrics.csv`; accept `map_error_in_analytic_std < 0.05` at n=200
  (from 43.5).

**Gate 3 — cost.** Record `score_network_calls` and `map_runtime_seconds`. Expect the iteration
count per level to drop from ~50 (exhausted) to ~3 (converged). Accept: >= 5x fewer network
calls at equal or better accuracy. If accuracy improves but cost does not drop, the Newton step
is not actually converging and something is wrong.

**Gate 4 — regression.** Must still pass unchanged:
`tests/test_hierarchical_map.py` (all), `tests/test_damping_composition.py:141`
(`test_damped_score_ascent_recovers_global_gaussian_mode`), `tests/test_pfode_vpsde_analytic.py`,
`tests/test_variable_param_count.py:101`, `Partial_Pooling/tests/test_benchmark_contracts.py`.

**Gate 5 — robustness to imperfect and non-Gaussian scores.** The Newton step uses curvature, so
it is more sensitive to score error than a fixed-step method; this gate exists to check that it
does not destabilize.
- Inject bounded score error with `FrozenErrorNet` /
  `AnalyticPosteriorScore(epsilon=...)` from `Compositional_score_testing/Gauss_test/analytic.py`
  at `epsilon in {0, 1e-3, 1e-2, 1e-1}`.
- Run the bimodal `SymmetricMixtureScore` toy (`tests/test_hierarchical_map.py:191`) and confirm
  `test_multistart_avoids_low_density_posterior_mean` still holds.
- Accept: no worse than the `step="tweedie"` baseline at every epsilon.

**Gate 6 — end-to-end (GPU; requires explicit go-ahead and a free GPU via `autocvd`).**
`Partial_Pooling` recovery on the existing `partial_pooling-train-32768-test-100.pt` dataset,
which has `truth_globals` / `truth_locals` on disk.
- Primary: `joint_map_absolute_error` in `<method>-global_recovery.csv`, especially indices 3–5
  (`log_sigma_nu`, `log_sigma_log_alpha`, `log_sigma_log_t0`) — the population scales, and the
  Neyman–Scott canaries.
- Secondary, and a strong free signal: the heuristic defenses in `infer_partial_pooling.py`
  are already instrumented. Count how often `at_denoise_boundary`, `coherent=False`, the
  low-noise retry (`MAP_RETRY_SIGMA_START`), and `posterior_fallback` fire. **If the optimizer
  fix is real, these firing rates should drop sharply** — they exist to catch exactly this
  failure.

---

## Phase 4 — Marginal MAP (gated on Gate 2)

Separate work, higher research risk, biggest statistical payoff. Only meaningful once the
optimizer error is below ~0.05σ.

Joint MAP over `(g, l_1..l_n)` is the **Neyman–Scott problem**: nuisance parameters grow with n,
so the joint mode is inconsistent for the shared parameters. In `Partial_Pooling` the shared
block contains three population log-scales (`schema.py`, indices 3–5) generated as
`l_ik = mu_k + exp(log sigma_k) * z_ik`, so the expected symptom is a systematic underestimate
of population spread that **does not shrink with more subjects**.

Fix: profile the locals (the Newton step already does this exactly via the Schur elimination),
then add the Laplace correction to convert profile MAP into marginal MAP:

```
marginal ~= profile + 0.5 * log det H_ll(g)^-1
=> gradient picks up  -0.5 * grad_g log det H_ll(g)
```

`H_ll,j` is already assembled in Phase 1; its `g`-derivative can be finite-differenced along
`d_g` at no extra network cost, since scores are already evaluated there.

**Test:** no existing analytic toy has an *inferred* population scale (confirmed — every stub
fixes `sigma_global`/`sigma_local`). Build one: `l_i ~ N(mu, sigma^2)`, `x_i = l_i + N(0, s^2)`,
with `sigma` unknown and a prior on `log sigma`. Both the joint MAP (which provably collapses
`sigma` toward 0) and the marginal MAP (locals integrated out analytically, Gaussian) are closed
form. Accept: joint MAP reproduces the known downward bias, marginal MAP recovers `sigma`
within 0.05σ across `n in {5, 50, 200}` with the bias flat in n.

---

## Files touched

| File | Change |
|---|---|
| `src/compass/MultiObsSampler.py` | `map_estimate` (`:671`) — Newton step, safeguards, extrapolation, diagnostics; possibly a grad-enabled `_get_score` variant for curvature source (B) |
| `tests/test_hierarchical_map.py` | Phase 0 parametrizations; Phase 4 population-scale toy |
| `tutorials/joint_vs_legacy_annealed_score_ascent.py` | add `step` to the swept methods for A/B |
| `Partial_Pooling/infer_partial_pooling.py` | bump `MAP_SETTINGS_VERSION`; surface guard firing counts |

Reused unchanged: `MockSBIm`, `SharedLocalGaussianScore`, `analytic_shared_local_map`,
`make_joint`, `SymmetricMixtureScore` (`tests/test_hierarchical_map.py`);
`_effective_global_factors`, `_project_information`, `_schur_global_precision`, `score_network_calls`
(`src/compass/MultiObsSampler.py`); `ModelTransfuser._hierarchical_candidate_scores`;
`AnalyticSBIm`, `FrozenErrorNet`, `write_rows` (`Compositional_score_testing/Gauss_test/`).
