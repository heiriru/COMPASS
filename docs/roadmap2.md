# Roadmap 2: library defaults (2.1-2.6) and three experiments (2.8-2.10)

Branch `library-roadmap-2x` (worktree `COMPASS-wt-2x`), based on `lambda-s-pd-study` @ 948bf37.
Experiment drivers and queues: `experiments_roadmap2/`. Logs and artifacts:
`/export/data/rheinric/roadmap2/`. Every experiment ran on a frozen copy of this branch's `src`
(`src_edm_test`, `src_gm`), so later edits could not leak into a running job.

## Decisions (2026-10-24)

**Standard (on by default):**

* `eps` forwarded to the samplers
* seeded validation
* patience 100
* final weights
* menu + EDM as a pair (`mask_menu="auto"`, `time_loss_weight="auto"`)
* `ScorePrior` for non-Gaussian priors
* latent attention as an `SBIm` option

**Opt-in, off by default:**

* `time_eps` (train to smaller λ): the score improves there, but nothing downstream benefits yet.
* Global-only TSM (`global_target_fn`) where the prior score is known.

**Experimental, not promoted:**

* the Fisher penalty (`compass.consistency`)
* the x0/c_skip parameterisation (`parameterization="edm"`)
* local TSM

**Recommended recipe.**

* Training:
  * tokenized trunk + `latent_attention="hierarchical"`
  * six or inference menu with the EDM loss (automatic when `mask_roles` is given)
  * `time_sampling="log_sigma"`
  * defaults for patience, restore and validation seed
  * global-only TSM when the prior score is known
* Inference:
  * `gauss_jacobian_FD` with `jacobian_refresh=9`
  * pass `eps` explicitly
  * pass a non-Gaussian prior as `ScorePrior`

The prior interface is where the next effort should go.

**Before announcing the menu + EDM default:**

* Run a calibration check on partial pooling: `log_sigma_nu` 90% coverage fell to 0.52.
* Run the GM composition at n = 32 on an exclusive GPU, to settle the Fisher penalty.

## Library changes

| # | change | default behaviour change? |
|---|---|---|
| 2.1 | `SBIm.sample(eps=...)` is forwarded to `Sampler.sample` and `MultiObsSampler.sample`. It was silently dropped: every sample stopped at 1e-3. | Only for callers who passed `eps`; the default stays 1e-3. |
| 2.2 | `Trainer.train(validation_noise_seed=20250719)`: every validation pass runs in a forked RNG seeded with this value (+ rank). Each epoch scores the same (t, noise) draw, and validation no longer consumes training draws. `None` is the old behaviour exactly. | **Yes**: new runs' validation curves are not comparable with old ones. Use `None` to reproduce an old run. |
| 2.3 | `early_stopping_patience` default 20 → **100** (`SBIm.train`, `Trainer.train`, `ModelTransfuser.train_models`). New `restore="final"` (default) or `"best"` chooses the weights the model holds and `{name}.pt` contains; `{name}_checkpoint.pt` (best-validation) is still written. | **Yes**: patience. Final weights were already what single-GPU training kept in memory. |
| 2.4 | `SBIm.train(mask_menu="auto", time_loss_weight="auto")`. The menu follows `mask_roles`: none given is Bernoulli, as before; an int is `"inference"`; a tuple or dict is `"six"`. The loss weight is `"edm"` whenever a menu is in use (named, or an explicit `mask_sampler`), otherwise the plain ε-loss. `Trainer.train` itself keeps `time_loss_weight=None`. | **Yes**, for anyone training with a menu or `mask_sampler`; pass `time_loss_weight=None` to get the old loss. Bernoulli training without roles is unchanged. |
| 2.5 | `compass.ScorePrior(score, mean, covariance, log_prob=None)` and `ScorePrior.from_samples(score, draws)`: any diffused prior score `score(theta_y, lambda)` (or `(theta_y, t, lambda)`) is accepted as `prior=`. A bare callable is refused with a pointer to `ScorePrior`: the moments are still needed, but only for the denoise clamp and the precision weighting, never for the score. `certify_composition` now uses an object prior's own `log_prob` instead of its Gaussian moment match. `GaussianPrior`, `GaussianMixturePrior` and `ScorePrior` are exported from `compass`. | No |
| 2.6 | Latent attention is a library feature: `SBIm(..., latent_attention="hierarchical", latent_roles=(global_size, local_size) or {"g","l","x"})`. The mode and roles live in the checkpoint (`arch_version` 2, written only by such models, so an older COMPASS refuses them instead of silently dropping the rule). The code moved to `compass.latent_attention`; `compass.experimental.latent_attention` is an alias, and sidecar checkpoints still load through `load_latent_attention`. | No |

Not changed: the tutorials that load `{name}_checkpoint.pt` explicitly (e.g.
`tutorials/Probability_Flow_validation.py`). Those caches predate the final-model save fix, so
switching them would break their skip-if-exists logic.

### Tests

* `tests/test_roadmap2_library.py` (23, including the 2.4 auto-default checks) and `tests/test_roadmap2_experimental.py` (9) are new.
* Full CPU suite (`CUDA_VISIBLE_DEVICES=-1`, without `test_multiobs_analytic.py`, which needs a GPU):
  the 77 failures on this branch are identical, test for test, to HEAD's. None are new.
* Those 77 were almost all one pre-existing leak. `pd_study/clamp_study.py` calls
  `torch.set_default_dtype(torch.float64)` at import, and `test_conservative_clamp.py` imports it
  inside a test. Every later test then ran in float64 and failed with Double/Float mismatches,
  while passing in isolation. `tests/conftest.py` now restores the default dtype after each test.
  With the fix: **13 failed, 448 passed, 8 skipped** (was 77 failed); after the 2.4 default change **13 failed, 454 passed, 8 skipped**, the same 13. Twelve of the thirteen are the
  known tests of features absent from `src` (`test_instantaneous_divergence.py` ×7, PF-ODE sampler
  tests in `test_pfode_vpsde_analytic.py` ×5). The thirteenth,
  `test_memory_bounded_attention.py::test_validation_uses_configured_training_batch_size`, also fails
  on HEAD: its stub `_prepare_data(data, batch_size, rank)` predates the `fixed_masks=` argument.

## 2.4: menu + EDM on partial pooling

Paired test: `n65536_edm_s{7,8,9}` against the existing `n65536_base_s{7,8,9}`. Same encoder, data,
seeds, menu (`six_mixed`, already the PP recipe for both arms) and inference (gauss_jacobian_FD, final
weights, the same 30 test datasets × 100 subjects; the truths are asserted to match). The only change
is `time_loss_weight="edm"`. Script: `experiments_roadmap2/analysis/pp_edm_per_global.py`
(→ `/export/data/rheinric/roadmap2/pp_edm_per_global.json`).

| seed | mean r base → edm | global RMSE base → edm | 90% coverage | z_rms |
|---|---|---|---|---|
| 7 | 0.905 → **0.926** | 0.419 → **0.340** | 0.80 → 0.75 | 1.30 → 1.47 |
| 8 | **0.892** → 0.878 | 0.628 → **0.449** | 0.82 → 0.76 | 1.20 → 1.29 |
| 9 | 0.815 → **0.905** | 0.950 → **0.513** | 0.74 → 0.78 | 1.52 → 1.32 |
| mean | 0.871 → **0.903** | 0.666 → **0.434** (−35%) | 0.79 → 0.76 | 1.34 → 1.36 |

Per global (mean over seeds; "closer" = share of datasets where the edm median is nearer the truth):

| global | r | RMSE | 90% coverage | edm closer |
|---|---|---|---|---|
| mu_nu | 0.963 → 0.963 | 0.113 → 0.112 | 0.80 → 0.72 | 44% |
| mu_log_alpha | 0.835 → 0.907 | 0.040 → 0.027 | 0.74 → 0.90 | 64% |
| mu_log_t0 | 0.984 → 0.987 | 0.057 → 0.047 | 0.77 → 0.74 | 53% |
| log_sigma_nu | 0.629 → 0.739 | 0.977 → 0.798 | **0.80 → 0.52** | 47% |
| log_sigma_log_alpha | 0.804 → 0.801 | 1.412 → 0.776 | 0.84 → 0.79 | 59% |
| log_sigma_log_t0 | 0.907 → 0.948 | 0.143 → 0.086 | 0.81 → 0.90 | 63% |
| beta_raw | 0.973 → 0.976 | 0.227 → 0.216 | 0.76 → 0.76 | 52% |

**Verdict:** accuracy is better. RMSE is lower in every seed, the largest gains are the worst
base seed and the two `log_sigma` globals the tokenized trunk struggles with, and mean r is better in
2/3 seeds (s8 −0.014, within the r standard error ≈ 0.05 at 30 datasets). The one cost is
`log_sigma_nu`: its intervals got narrower than its errors (coverage 0.52). Overall coverage moved by
−0.03 and z_rms is unchanged, so this is one global's calibration, not a general over-confidence.
The default was made conditional on a menu because the earlier likelihood study found EDM without a
menu cost the posterior a little; that combination is still untested on PP.

## 2.9: the `log_sigma_log_alpha` regression under TSM (PP)

From the existing `base` / `tsm2` runs (3 seeds × 30 datasets, final weights;
`analysis/pp_tsm_lsla.py`). The drop in r (0.81/0.77/0.83 → 0.74/0.72/0.80) is not a loss of
accuracy:

* RMSE falls on average (1.41 → 1.12, better in 2/3 seeds) and calibration improves (z_rms closer to
  1; 90% coverage 0.80/0.87/0.87 → 0.87/0.87/0.93).
* The base medians were over-dispersed (slopes 1.14 and 1.80 against the truth). TSM shrinks them
  toward the prior. That is what a calibrated posterior does when the data say little, and shrinkage
  lowers r.
* The data do say little. The mean posterior variance exceeds the variance of the test truths
  (truth sd 1.16), so the information ceiling `sqrt(1 − E[post var]/Var truth)` for a calibrated
  posterior is ≈ 0. r here is driven by a few datasets with wide, collapse-like posteriors.
* The standard error of r at 30 datasets is ≈ 0.067, larger than the mean drop (0.053). Paired per
  dataset, TSM is closer to the truth on 33/57/63% of datasets.

## 2.9: TSM on the locals of hierarchical two moons

`attn_hier_tsm_gl` (`experiments_roadmap2/two_moons.py`): `attn_hier_tsm` plus a local target
∇θ[log TN(θ | g) + log p(x | θ)], blended by inverse variance with the measured Var T_l ≈ 1550
(`analysis/two_moons_local_tsm_variance.py`: weight ½ at λ 0.025). Same data, seed and recipe; one
seed; smax 0.6; gauss_jacobian_FD; 10 datasets per N. Results are in the benchmark's
`artifacts_smax0.6/results/attn_hier_tsm_gl__*`.

| N | arm | local C2ST | crescent dev | off support | mu W1 | sigma W1 | global C2ST |
|---|---|---|---|---|---|---|---|
| 10 | attn_hier | 0.520 | 0.758 | 0.025 | 0.239 | 0.615 | 0.801 |
| 10 | attn_hier_tsm | 0.521 | 0.755 | 0.025 | 0.218 | 0.586 | 0.749 |
| 10 | +locals | 0.530 | 0.722 | 0.070 | 0.209 | 0.576 | 0.767 |
| 25 | attn_hier | 0.527 | 0.781 | 0.042 | 0.402 | 1.667 | 0.968 |
| 25 | attn_hier_tsm | 0.524 | 0.769 | 0.038 | 0.519 | 1.462 | 0.939 |
| 25 | +locals | 0.537 | 0.733 | 0.097 | 0.473 | 1.527 | 0.921 |
| 100 | attn_hier | 0.543 | 0.804 | 0.042 | 0.780 | 1.755 | 0.994 |
| 100 | attn_hier_tsm | 0.531 | 0.782 | 0.042 | 0.670 | 1.490 | 0.990 |
| 100 | +locals | 0.543 | 0.740 | 0.091 | 0.710 | 1.640 | 0.988 |

The per-group global score error (`checks/score_check.py`, rel. error vs oracle) is the same as
`attn_hier_tsm` within noise at every N and λ (e.g. N=100: 0.94/0.73/0.55/0.48 vs
0.83/0.71/0.63/0.53 at λ 0.02/0.05/0.1/0.2).

**Verdict: null, slightly negative. Not recommended.**

* The locals were already at the floor (local C2ST 0.52–0.54; 0.50 is perfect).
* Crescent deviation moves toward the exact 0.67, but the off-support mass doubles (0.04 → 0.09).
  That is the expected failure of a gradient target on a bounded support: the TN box and the
  crescent endpoints are boundaries, and the boundary term the target omits biases the score in a
  thin band next to them.
* The globals do not move beyond seed noise.

Global-only TSM stays the useful part.

## 2.8 / 2.9 / 2.10 on the Gaussian-mixture hierarchy

`experiments_roadmap2/gm.py` (frozen as `/export/data/rheinric/roadmap2/gm_driver.py`, library
`src_gm`). Recipe: h32d3wide, 200k simulations, six_weighted menu, log_sigma, σ = 15, 200 epochs,
patience 40; seeds 7 and 8. Arms:

* `base`
* `tsm_g`: TSM on g with the exact mixture prior score
* `tsm_gl`: TSM on g and l
* `fisher`: the 2.8 penalty, weight 1, 16 anchors × 64 resampled rows
* `lo`: `time_eps` 1e-5, training floor λ 0.0032 instead of 0.032
* `edm_lo`: the 2.10 x0/c_skip parameterisation, σ_d 2.5, plus `time_eps` 1e-5

Score error is measured against the exact oracle score: relative RMSE of the g score, mean of 2
seeds (half-range in brackets).

| λ | 0.003 | 0.006 | 0.01 | 0.02 | 0.03 | 0.06 | 0.1 | 0.3 |
|---|---|---|---|---|---|---|---|---|
| base | 11.8 (1.4) | 5.45 | 2.95 | 1.13 | 0.611 | 0.371 | 0.359 | 0.259 |
| tsm_g | 11.8 (0.3) | 5.48 | 2.96 | 1.13 | 0.604 | 0.369 | 0.358 | 0.259 |
| tsm_gl | 9.39 (0.1) | 4.27 | 2.25 | 0.823 | 0.471 | 0.370 | 0.357 | 0.258 |
| fisher | 10.0 (0.7) | 4.59 | 2.44 | 0.908 | 0.505 | 0.369 | 0.357 | 0.259 |
| lo | 3.41 (0.2) | 1.32 | 0.581 | **0.413** | **0.471** | 0.377 | 0.357 | 0.259 |
| edm_lo | **2.52** (0.0) | **0.912** | **0.432** | 0.460 | 0.487 | 0.373 | 0.357 | 0.259 |

**Caveat on stopping:**

* `base_s8` and `tsm_g_s8` early-stopped at epoch 95; `fisher` and `tsm_gl` ran 200 in both seeds.
* Seed 7 alone is the matched comparison, because base, tsm_g, tsm_gl and fisher all ran 200 epochs
  there. Relative RMSE at λ 0.003 / 0.01 / 0.03:
  * base 10.4 / 2.55 / 0.520
  * tsm_g 11.5 / 2.87 / 0.582
  * tsm_gl 9.26 / 2.21 / 0.452
  * fisher 9.30 / 2.23 / 0.475
* `edm_lo_s7` stopped at 138 and `lo_s7` at 180.

**Findings:**

* **2.10: the training floor is the lever, the parameterisation is second.**
  * Below λ 0.03 every 1e-3 arm is extrapolating. Training down to λ 0.003 (`lo`) cuts g-score
    error 3.5–5× there and costs nothing above λ 0.06.
  * The x0/c_skip output (`edm_lo`) takes another 26–31% off at λ ≤ 0.01. ε-prediction cannot
    resolve the needed output precision there, as 2.10 predicted.
  * It is 10–12% worse at λ 0.02–0.03 and neutral above. Its seed spread is the smallest of any arm.
  * Use both together, with sampling `eps` lowered to match. Neither is a default yet: whether the
    gain survives composition is what the compose rerun measures.
* **2.8 Fisher penalty: it does what it says, but what it controls is small.**
  * In the trained band its own target, the binned bias b(g_t), roughly halves: rms bias at λ 0.06
    falls 0.107 → 0.049 on the two-seed mean and 0.099 → 0.061 on seed 7.
  * The total score error there is ~10× the bias and does not move.
  * Below the floor it gives 10% on the matched seed, the same as tsm_gl.
  * It doubles training time (105–119 min vs 50). The penalty's ESS was never limiting (all 16
    anchors kept, ESS 6·10⁴–10⁵).
  * Its only plausible payoff is composition at large n, where a shared bias is summed n times while
    per-observation noise partly cancels.
* **2.9 TSM on GM:** g-only TSM does nothing here (the g score is already near the floor in the
  trained band). Adding the locals gives the same small gain below the floor as Fisher, at half the
  cost.

**Composition (exact-MC reference, 2 seeds × 2 replicates). n = 8, W1/σ / |branch-mass error|:**

| arm | asym GJ | asym GJ + mixture prior | sym GJ | sym GJ + mixture prior |
|---|---|---|---|---|
| base | 1.098 / 0.042 | 0.522 / 0.037 | 0.957 / 0.056 | 0.406 / 0.041 |
| tsm_g | 1.083 / 0.055 | 0.541 / 0.032 | 0.949 / 0.055 | 0.423 / 0.047 |
| tsm_gl | 1.084 / 0.049 | 0.542 / 0.033 | 0.962 / 0.070 | 0.457 / 0.057 |
| fisher | 1.050 / 0.043 | 0.473 / 0.024 | 0.994 / 0.096 | 0.442 / 0.080 |
| lo | 1.137 / 0.105 | 0.497 / 0.129 | 0.934 / 0.024 | 0.322 / 0.037 |
| edm_lo | 1.096 / 0.061 | 0.446 / 0.075 | 1.148 / 0.172 | 0.530 / 0.153 |

(`lo` and `edm_lo` sampled to eps 1e-5 give the same picture.)

**At n = 8 the composed error is set by the prior interface, not the network.**

* Passing the mixture prior's score instead of its Gaussian moments halves W1 in every arm.
* No training-side change moves W1 by more than the spread between seeds.
* `edm_lo` gets the symmetric branch mass wrong (0.17).
* The score gains below λ 0.03 do not reach the composed posterior here.

n = 32 still OOMs whenever another ~5 GB job shares the GPU; it needs a GPU to itself. It is the
decisive test for the Fisher penalty, whose case rests on bias being summed n times.
