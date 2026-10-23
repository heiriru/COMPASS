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

**Status:** finished on 2026-10-23. All 180 jobs ran on GPUs no other user was on.
- Some jobs on the curved and lean_T_e4 toys ran out of memory at n = 128. Every one finished on a later retry, so no output is missing.
- Tables: `artifacts/noproj/comparison_pd.{txt,json}`. Figure: `artifacts/figures/17_pd_strategies.png`.

### 3.1 The default path is unchanged

`default` reproduces the stored draws **bit for bit**: max |diff| = 0 on all 112 rows, and PP mean r, RMSE and collapses are identical.

A few `default` diagnostics are missing (`nan`) for toy rows whose `n` was drawn during an aborted shared-GPU attempt. Those draws are bit-identical; their jsonl is in `diagnostics/killed_shared/`.

### 3.2 How often the raw Lambda_S is not PD (default trajectories)

| family | samples non-PD (max / median) | score calls with ≥1 non-PD sample (max / median) |
|---|---|---|
| GM (trained) | 0.3% / 0.1% | 18% / 8% |
| GM oracle | 0 / 0 | 0.3% / 0 |
| IK (all) | **80%** / 11% | 100% / 59% |
| toys (trained) | 15% / 10% | 67% / 51% |
| toys (oracle scores) | 20% / 20% | 50% / 50% |
| PP (baseline / Rg) | 22% / 23% | — |

The worst case is IK flat global, at 67–80% of samples.

So "only check Lambda_S" fails often, even with exact scores. Without per-row projection, Sum I_j is regularly indefinite: one row's negative curvature is not covered by the others.

### 3.3 Accuracy

W1 to the exact posterior, in exact sd, is compared with the stored row-projection run.
- **Geometric-mean ratio:** below 1 means the variant is better.
- **Tally:** better / same / worse over 112 rows, where "same" = within 15% + 0.02.
- **PP:** mean r, global RMSE and collapses over 30 datasets. Stored: baseline 0.869 / 0.628 / 5; Rg 0.868 / 0.608 / 3.

| variant | GM | GM oracle | IK | toys | toy oracle | tally | PP baseline | PP Rg |
|---|---|---|---|---|---|---|---|---|
| `lams` (check, then composed spectral repair) | 1.01 | 1.00 | 1.73 | 1.44 | 1.34 | 2/70/40 | 0.779 / 1.052 / 7 | 0.847 / 0.802 / 7 |
| `lamsrow` (check, row projection only if not PD) | 1.01 | 1.00 | 1.13 | 1.10 | 1.11 | 2/90/20 | 0.857 / 0.749 / 6 | 0.849 / 0.789 / 7 |
| `gloeckler` (nugget 0.1) | 1.01 | 1.00 | 1.24 | **0.93** | 0.98 | 17/76/19 | 0.878 / 0.566 / 4 | 0.888 / 0.492 / **1** |
| `consistent` | 1.51 | 1.00 | 1.73 | 4.60 | 1.65 | 5/22/85 | **0.895 / 0.478 / 0** | **0.909 / 0.476 / 0** |
| `clampmetric` | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1/111/0 | 0.869 / 0.628 / 5 | 0.868 / 0.607 / 3 |

**Reading:**
- **Check-only is worse than the current rule, and the fallback matters.**
  - `lams` repairs the composed Lambda_S on ~50% of calls. It loses badly on IK, the toys and PP: mean r 0.869 → 0.779.
  - `lamsrow` projects rows only where Lambda_S fails. It is closer to the current rule, but still worse everywhere (PP 0.857 / 0.849).
  - So projecting every row, even when the sum would be PD, is not what costs accuracy. Rows with negative curvature hurt even when the others compensate.
- **Gloeckler is the only drop-in that is never clearly worse on average.**
  - It is slightly better on the trained toys (0.93) and better on PP: mean r 0.878 / 0.888, global RMSE −10% / −19%, Rg collapses 3 → 1.
  - It is worse on IK (1.24, mostly IK flat global at n = 8–32: W1 0.19 → 0.93).
- **`consistent` splits by regime.**
  - It is the **best result on partial pooling we have**: mean r 0.895 / 0.909, global RMSE −24% / −22%, **zero collapses**. That is above the 0.904 of the TSM-trained nets (different training), here with the untouched nets.
  - It is also the best on IK flat global (n = 8: 0.19 → 0.04; n = 128: 3.27 → 1.68).
  - But it is much worse on the GM and toy problems (×1.5 and ×4.6), with W1 ≈ 0.5–0.7 sd on every curved and lean toy (0.1–0.45 on abs and square).
  - Likely cause: the `K_j^-1` amplification where a row is almost uninformative (section 1.1). On the toys most rows say little about g individually, so the floored `K_j^-1` turns a small projection into a large mean shift. In PP each row is informative about the globals, so the consistent mean is the better one.
  - Before using it outside PP, it needs a better treatment of uninformative rows (e.g. shrink `Delta_j` with `K_j`, or fall back to the default where `K_j` is near the floor).

### 3.4 The metric (conservative) clamp

On every experiment the clamp is almost never active: median 0%, max 0.4–0.9% of predictions. So `clampmetric` equals the default within noise (111/112 "same", PP identical).

The one exception is IK flat global n = 128: W1 3.27 → 2.53. There the clamp is active most often, and the conservative version helps.

This matches the 2-D study (section 2.1). The choice of clamp matters only where the network extrapolates, and there the metric clamp is the better one. It costs nothing elsewhere, so it is a safe default candidate. It just does not change results on the current benchmarks.
