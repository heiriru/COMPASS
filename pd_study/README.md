# Keeping the composed precision positive definite, and clamping conservatively

Started 2026-10-21 on branch `lambda-s-pd-study`. Library changes are in
`src/compass/MultiObsSampler.py`. **Every default is unchanged**: `golden.py check`
reproduces the pre-change `gauss_jacobian` / `gauss_jacobian_FD` draws bit for bit at n = 1, 4
and 16. The full test suite has the same 13 failures and 1 CUDA-only collection error before
and after the change (all in features absent from `src/`; see
`tests_old.txt` and `tests_new.txt`). There are 25 new tests.

## 1. How Lambda_S is kept PD: `set_pd_strategy` (env `COMPASS_PD_STRATEGY`)

With `Lambda_j(t) = Lambda_prior(t) + I_j(t)` the composed precision is
`Lambda_S = Lambda_prior(t) + sum_j I_j`.

| strategy | what it does |
|---|---|
| `row_projection` (default) | Project each `I_j` onto the PSD cone: `I_j -> I_j^+ = I_j + Delta_j`. Weight the row's **unmodified** score with `Lambda_j + Delta_j`. |
| `composed_check` | Compose the raw `I_j` and check `Lambda_S` per sample. Samples where it is not PD are counted (`lambda_s_*` diagnostics). `COMPASS_PD_FALLBACK` then handles them: `repair` (the old composed-level spectral repair), `row_projection` (the default rule, on the failing samples only) or `gloeckler`. |
| `gloeckler` | Gloeckler et al. 2025, App. B.3, exactly as their code ships it (`markovsbi` `CorrectedScoreFn.ensure_valid_precission`). Every factor gets `V max(-A,0) V^T/(n-1) + eps*I`, where `Lambda_S = V A V^T` and `eps = COMPASS_PD_NUGGET` (default 0.1, their default). It is applied on **every** call with n > 1, PD or not. The scores are weighted by the adjusted precisions. |
| `row_projection_consistent` | The fix for "modified covariance, exact-covariance score" (section 1.1). |
| `row_projection_post` | The default rule applied after the Schur reduction. This is a check only: it equals the default to about 6e-6. |

Diagnostics in `covariance_diagnostics` and in the `COMPASS_PROJECTION_LOG` jsonl:
- `lambda_s_nonpd_fraction`: share of samples whose raw `Lambda_S` had an eigenvalue ≤ 0.
- `lambda_s_nonpd_call_fraction`: the same, counted over score calls.
- `lambda_s_min_relative`: the smallest eigenvalue times lambda².
- `lambda_s_trace`: per call, (lambda, non-PD share, minimum).

These are recorded under **every** strategy, including the default (where the raw
`Lambda_S` is reconstructed by subtracting the lifts).

### 1.1 Why the default rule is inconsistent, and the consistent version

GAUSS weights the row score with its backward precision:
`Lambda_j s_j = -lambda^-2 K_j (y - mu_j)` for a Gaussian row posterior `N(mu_j, K_j^-1)`.

The default rule lifts the precision to `K_j + Delta_j` but keeps `s_j`, which is the score of
the *unlifted* Gaussian. That adds `Delta_j s_j` to the numerator. In message form it adds a
pseudo-observation that pulls the global towards the row's **denoised** mean `m_j(y)` with
precision `Delta_j`. That attractor is a moving point, and it is not the row's posterior mean.

The consistent version uses the score of the lifted Gaussian with the same clean mean,
`mu_j = y + (K_j^-1 + lambda^2) s_j`:

    Lambda_j' s_j' = (K_j + Delta_j) K_j^-1 Lambda_j s_j,   K_j = Lambda_j - lambda^-2 I

So the extra numerator term is `Delta_j K_j^-1 Lambda_j s_j` instead of `Delta_j s_j`.
`tests/test_pd_strategies.py` checks it against GAUSS composed directly from the modified
Gaussians `N(mu_j, (K_j + Delta_j)^-1)`.

Caveat: `K_j^-1` is large where a row is almost uninformative. It is floored at 1e-6 of
the row's largest eigenvalue.

### 1.2 First observation (oracle, before the GPU runs)

Setup: lean toy, `oracle_hierarchical` (exact scores), n = 16, 200 draws, CPU.

- The **raw Lambda_S is not PD in 18% of samples and in 42% of score calls**.
- Its smallest eigenvalue is -295 lambda^-2.

So even with exact scores, "check Lambda_S and only inform" is not enough to sample: the
check fails at roughly every other call. That is why `composed_check` needs a fallback.

## 2. Conservative clamping: `set_clamp_mode("metric")` (env `COMPASS_CLAMP_MODE`)

The default is `coordinate`: `clip(m)` per coordinate, with `m = y + lambda^2 s`. Its score
has an antisymmetric Jacobian part `(D_ii - D_jj) (grad m)_ij / lambda^2` wherever one
coordinate is clipped and a coupled one is not (tested against this formula).

**The metric mode** is `m~ = argmin_{z in B} (z - m)^T A (z - m)`, where A is the backward
precision, i.e. the inverse denoiser Jacobian:
- for a row, its joint `Lambda_j(t)`. Globals and that row's locals are clamped together, and
  a local without a prior box moves with its global through `Lambda_ll^-1 Lambda_lg`.
- for the composed score, the composed `Lambda(t)`.
- after the cross-correction, the locals in the metric `Lambda_j,ll`.

Two properties follow:
- **It is the quadratic-model step of the conjugate construction.** That construction is
  `psi_B = (psi* + iota_B)*` with `psi = |y|^2/2 + lambda^2 log p_t`.
- **It equals `m` exactly where `m` is already inside B.**

The box QP is solved with an exact active-set method (`MultiObsSampler._box_qp`, tested
against brute-force enumeration).

**The exact construction**, for oracle fields, is `conservative_clamp.conjugate_clamp`.
- It runs proximal Newton on `psi(y') + sigma_B(y - y')` with an Armijo line search.
- Undamped Newton diverged and left the box. This version is in the box everywhere.
- Its antisymmetry is ≤ 1e-9 everywhere.

### 2.1 Clamp study (`clamp_study.py`, exact 2-D targets, box [-b, b]²)

**Non-conservativity** `||J - J^T|| / ||J||` of the clamped score where the clamp is active
(median):

| target, lambda | coordinate | metric | conjugate (exact) |
|---|---|---|---|
| Gaussian, 0.3 / 0.7 / 1.5 | 0.22 / 0.34 / 0.22 | 0 / 0 / 7e-12 | 0 / 6e-12 / 7e-12 |
| mixture, 0.3 / 0.7 / 1.5 | 0.10 / 0.11 / 0.11 | 5e-11 / 1.3e-3 / 2.0e-3 | ≤ 1e-9 |

- On the mixture the metric clamp's 95th percentile is about 0.1–0.5, the same as the
  coordinate clamp. Off a Gaussian it is only approximately conservative, and locally it can
  be as non-conservative as the coordinate clamp (`figures/clamp_antisymmetry_map.png`).
- Metric and exact differ by about 1e-3 of |s| (median).

**Sampling with the exact score** (reverse VE SDE, 400 levels). Clamping is pure bias here,
and the clamps are about equally biased; conservativity does not reduce it.

| sliced W1 (box b = 1.0 / 1.5 / 2.5) | none | coordinate | metric | conjugate |
|---|---|---|---|---|
| Gaussian | 0.009 | 0.25 / 0.10 / 0.015 | 0.26 / 0.11 / 0.016 | = metric |
| mixture | 0.011 | 0.26 / 0.079 / 0.011 | 0.27 / 0.083 / 0.011 | 0.27 / 0.082 / 0.011 |

**What the clamp is for: a corrupted, non-conservative tail score.** The error is a growing
rotation beyond radius 2.5, a stand-in for network extrapolation; box b = 3.

| | runaway (unclamped) | coordinate SW1 | metric SW1 | correlation: coordinate / metric (exact) |
|---|---|---|---|---|
| Gaussian | 99.7% | 0.132 | **0.071** | 0.82 / 0.80 (0.85) |
| mixture | 99.8% | 0.107 | **0.049** | 0.71 / 0.79 (0.82) |

Both clamps contain the runaway. The metric clamp leaves half the error of the coordinate
clamp.

## 3. Experiments (`experiments_arruda_schur/run_pd.py`, `compare_pd.py`)

**Scope.** All 30 jobs of `run_noproj.py` (gauss_jacobian_FD arms): PP baseline and Rg (30
datasets), IK |x1| (±R), IK flat fixed/global, IK reach sym/shift (± Bernoulli menu), GM
hierarchical/none/oracle, and the abs/square/lean/curved toys.

**Variants:** `default`, `lams`, `lamsrow`, `gloeckler`, `consistent`, `clampmetric`.

**Checks.** `default` must reproduce the stored draws bit for bit. That is our test that the
production `gauss_jacobian_FD` path is unchanged on the real experiments. It also logs the
raw Lambda_S and clamp activity on the published trajectories.

**Scoring.** W1 to the exact posterior in exact sd, versus the stored run. "Same" means within
15% + 0.02. PP is scored by mean r, RMSE and collapses.

**Status:** pending GPUs (see `artifacts/noproj/run_pd.out`).
