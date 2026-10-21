# A4 (part) — WAIC from composed posteriors and PF-ODE likelihoods

Status: **in progress** (started 2026-10-17).

## What was built

`ModelTransfuser.compare(..., criterion="waic")`:

1. Draw `num_samples` joint samples from the **composed** posterior `p(g, l_1..l_N | x_1..x_N)`
   with `correction="gauss_jacobian_FD"`, `jacobian_refresh=9` and the churned DPM sampler
   (the fastest accurate rule, docs/jacobian_speedup_results.md).
2. Evaluate `log p(x_i | g_s, l_{i,s})` for every row i and draw s with the PF-ODE (`waic_timesteps=50`,
   exact divergence, isotropic endpoint: the trained-network optimum of validation/b5_b6).
3. `compass.information_criteria.waic`: `lppd_i = log mean_s p(x_i|θ_s)`,
   `p_waic,i = Var_s log p(x_i|θ_s)`, `WAIC = -2 Σ_i (lppd_i - p_waic,i)`, plus the SE over
   rows, a jackknife Monte-Carlo SE and the `p_waic,i > 0.4` count.
4. Model weights: `softmax(-WAIC/2)` (`model_prob`) and pseudo-BMA+ (`model_prob_pseudo_bma_plus`,
   Bayesian bootstrap over rows, Yao et al. 2018); `elpd_diff` and its paired SE to the best model.

`hierarchy` and `prior` may now be dicts per model, so nested models with different global
dims are compared in one call (a model with `hierarchy=[]` gets independent per-row posteriors).
WAIC refuses anything but `multi_obs_inference=True` with `gauss_jacobian_FD`.

This is the **conditional** WAIC: the locals are part of θ, so it scores the prediction of a
new measurement of the same unit.

## Design

Three levels (validation rule 2):

- (a) **exact references with the exact likelihood:** grid quadrature (exact for ≤ 1 global and
  1 local), emcee MCMC (DE + stretch moves) and dynesty nested sampling (evidence + posterior).
- (b) **exact-score oracle networks** through the full pipeline (offset family).
- (c) **trained networks** (legacy transformer 5.3 M, VESDE σ 25, `log_sigma`, inference mask
  menu + EDM weighting, final epoch; 2¹⁶ simulations, 2¹⁷ for population).

Families (`families.py`, `population.py`), N rows per dataset, data from the true model's prior:

| family | models | globals / locals / x per row | N |
|---|---|---|---|
| offset | M1: x = (l + νe, g + νe); M0: g = 0 (nested) | 1 / 1 / 2 | 16 |
| noise | x_j = g + l + ν e_j, e Gaussian (G) or Student-t₃ (T), equal variance | 1 / 1 / 3 | 16 |
| population | LV, logistic prey, satiated predator, Rosenzweig–MacArthur (tutorial); log-rates global, log initial state per site, Gaussian noise 0.05 | 4 / 2 / 20 | 8 |

The true model is chosen uniformly over datasets, so the data follow the prior predictive of the
equal model mixture: the setting in which exact posterior model probabilities are calibrated.

## Predictions (written before the results)

References (a):

1. Quadrature equals the closed form on the offset family (< 1e-3). *Checked: agrees to 4 decimals.*
2. MCMC and NS WAIC agree with quadrature within their Monte-Carlo error; NS log Z agrees with the
   exact log Z within its reported error (z-scores ≈ N(0, 1)).

Pipeline (b), oracle:

3. COMPASS WAIC − exact WAIC per model: a bias of about −2·N·0.027 ≈ −0.9 (the isotropic PF-ODE
   endpoint over-states log p by ≈ +0.027 per point at 50 steps; measured on the offset oracle) and a
   spread at the Monte-Carlo SE (≈ 0.35 at 1000 draws).
4. The bias is the same in both models (same likelihood covariance given θ), so ΔWAIC has no bias
   and a spread ≈ √2 × MC SE ≈ 0.5. The COMPASS decision equals the exact-WAIC decision on ≥ 95 %
   of datasets.

Trained (c):

5. Per-model WAIC error grows to the network's log p error (B6: 0.11–0.19 nats/row RMSE, biased low by
   0.03–0.05/row, i.e. WAIC +1…+1.6 on 16 rows); ΔWAIC error 1–2 units; decision agreement ≥ 90 %.
6. Population (20-D x, 6-D θ per row) is worse than the toy families.

Selection and calibration:

7. Recovery: WAIC behaves like AIC for the nested pair. With M0 true it picks M1 at a rate of roughly
   0.1–0.2; with M1 true the rate rises with |g|.
8. Exact posterior model probabilities (NS evidence) are calibrated (reliability on the diagonal
   within bootstrap bands).
9. WAIC weights are *not* posterior probabilities and are expected to be miscalibrated:
   under-confident for the simpler nested model (its weight is capped when M0 is true), and
   over-confident on hard non-nested pairs (G vs T, LP vs RM). Pseudo-BMA+ is less extreme.
10. COMPASS WAIC weights follow the exact-WAIC weights (same reliability curve within noise).

## Results

(pending)
