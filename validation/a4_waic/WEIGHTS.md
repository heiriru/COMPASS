# Honest model weights and elpd-difference error bars (2026-10-22)

Proposal 4 of the "make WAIC publishable" list: so far COMPASS reports softmax weights
exp(−WAIC/2), which are overconfident even with exact WAIC (noise family, exact level:
ECE 0.106 vs evidence 0.033, IMPROVEMENTS.md exp 4). Here the weighting scheme and the
reported uncertainty are changed using only the stored records (per-row elpd, MC-SE) --
no new runs. Script: `improvements.py exp5`.

## Methods (applied to the reference values and to COMPASS alike)

- softmax: w_k ∝ exp(elpd_k) (= exp(−WAIC_k/2)), the current default.
- pseudo-BMA+ (Yao et al. 2018): Bayesian bootstrap over rows (Dirichlet(1) weights α),
  w_k = E_α softmax(n Σ_i α_i elpd_ik); 1000 replicates.
- stacking (Yao et al. 2018): w = argmax over the simplex of Σ_i log Σ_k w_k exp(elpd_ik).
  Defined on pointwise LOO densities; the WAIC rows are used where LOO rows are missing.
- MC-aware softmax (COMPASS only): average the softmax over WAIC_k ~ N(ŴAIC_k, σ_k²),
  σ_k² = mc_se_k² (+ seed term 0.56²/2 per model, the between-seed sd of ΔWAIC from exp 2,
  measured on offset only).
- elpd difference ± se: Δ = elpd_best − elpd_k, se_diff = sqrt(n var_i(elpd_i,best −
  elpd_i,k)) (Vehtari et al. 2017), for COMPASS plus the MC term in quadrature.
  Decision rule: choose the best model if Δ > 2 se for every other model; otherwise "undecided".

Evaluation, per family and level (reference / COMPASS):
- weights against the true model: accuracy, ECE (null95 floor), Brier score and mean log
  weight of the true model (proper scores, so sharpness counts too);
- decision rule: fraction decided, accuracy among decided datasets, agreement with the
  reference rule;
- error-bar honesty: z = (ΔWAIC_COMPASS − ΔWAIC_ref) / σ_MC(+seed), fraction |z| < 2
  (should be ≈ 0.95).

## Hypotheses (written before running)

Already known before this analysis: COMPASS's stored pseudo-BMA+ weights have ECE 0.256
(noise), 0.092 (offset), 0.045 (population), vs softmax 0.11 / ~0.09 / 0.055.

- W1. Against the true model none of the predictive weights is calibrated: they estimate
  which model predicts best, not P(model | data). pseudo-BMA+ is underconfident (shrinks
  to equal weights when the per-row differences are noisy, as for G vs T), stacking is
  spiky (0/1 weights from 16–20 rows) and no better than softmax. Only evidence is
  calibrated (exact level).
- W2. MC-aware weights reduce COMPASS's overconfidence on offset/noise (errors there are
  MC-sized: z mostly within ±2 with the seed term), but not on the ODE families, where the
  errors are biases of tens to hundreds (|z| ≫ 2 for most datasets).
- W3. The Δ > 2 se rule: ≥ 90% correct among decided datasets at the reference level in
  every family; COMPASS matches that on offset/noise/population, not on the tutorial.

## Results (`improvements.py exp5`, results/improvements_exp5.json, figures/fig_exp5_weights.png)

Note on the "already known" line above: the stored 0.256 for noise was a different
measurement (P(first model) only, older 344-dataset subset). Recomputed here on the same
pooled (dataset, model) basis as everything else, pseudo-BMA+ is *not* underconfident;
my pseudo-BMA+ matches the library's stored weights (corr 0.999).

ECE (null95) / mean log weight of the true model; COMPASS level unless "ref":

| | softmax | pseudo-BMA+ | stacking | evidence (ref) |
|---|---|---|---|---|
| offset ref | 0.060 (0.051) / −0.59 | 0.047 / −0.58 | 0.241 / −2.33 | 0.025 / −0.59 |
| offset | 0.083 (0.048) / −0.61 | **0.060** / −0.61 | 0.256 / −2.80 | |
| offset exact scores | 0.058 / −0.60 | **0.035** / −0.59 | 0.252 / −2.49 | |
| noise ref | 0.106 (0.034) / −0.54 | **0.047** (0.043) / −0.45 | 0.064 / −0.58 | 0.033 / −0.32 |
| noise | 0.153 / −0.70 | **0.087** (0.044) / −0.53 | 0.137 / −1.16 | |
| population ref | 0.024 / −0.32 | 0.020 / −0.35 | 0.077 / −1.92 | |
| population | 0.055 (0.027) / −0.55 | **0.043** / −0.36 | 0.043 / −1.06 | |
| population_wide | 0.066 (0.037) / −0.79 | **0.053** / −0.67 | 0.090 / −2.44 | |
| tutorial | 0.112 / −4.59 | 0.099 / **−0.69** | 0.070 / −0.29 | |

MC-aware softmax (MC-SE, ± seed term) changes ECE by ≤ 0.005 everywhere.

Δelpd > 2 se rule: decided / accuracy among decided / decided-and-wrong:

| | reference | COMPASS | COMPASS + MC + seed |
|---|---|---|---|
| offset | 0.34 / 0.76 / 0.08 | 0.39 / 0.73 / 0.11 | 0.32 / 0.73 / 0.09 |
| offset exact scores | (same ref) | 0.26 / 0.77 / 0.06 | 0.14 / 0.85 / 0.02 |
| noise | 0.25 / 0.97 / 0.007 | 0.31 / 0.94 / 0.02 | 0.30 / 0.95 / 0.015 |
| population | 0.28 / 0.97 / 0.008 | 0.53 / 0.98 / 0.008 | same |
| population_wide | 0.10 / 1.00 / 0 | 0.44 / 0.91 / 0.04 | 0.42 / 0.94 / 0.025 |
| tutorial | 0.74 / 1.00 / 0 | 0.63 / 0.95 / 0.03 | same |

COMPASS ΔWAIC error within 2σ (target 0.95): MC-SE only / + seed term: offset exact
scores 0.92 / 0.99; offset 0.38 / 0.66; noise 0.46 / 0.60; population 0.04 / 0.05;
population_wide 0.11 / 0.12; tutorial 0.02 / 0.02.

- W1 half wrong. pseudo-BMA+ is the best predictive weight: it lowers COMPASS's ECE in
  every family (offset 0.083→0.060, noise 0.153→0.087, population 0.055→0.043, wide
  0.066→0.053) and raises the log score, most on the tutorial (−4.6 → −0.7: softmax put
  ~0 weight on the true model when COMPASS's bias flipped the decision). At the exact
  level it brings noise to its null floor (0.106 → 0.047). Stacking is spiky, as
  predicted: ECE 0.24–0.26 and log score −2.3 to −2.8 on the two-model families. Evidence
  is still the best-calibrated (noise 0.033, log −0.32).
- W2 rejected even for offset/noise: COMPASS's MC-SE covers the ΔWAIC error only with
  exact scores (0.92). With trained networks the error is mostly systematic per network
  (offset 38%, noise 46% within 2σ; 66%/60% with the seed term), and on the ODE families
  the MC-SE is meaningless as an error bar (2–12%). MC-aware weights therefore change
  nothing. An honest COMPASS error bar needs the network error -- the spread over a seed
  ensemble is the only label-free estimate we have (exp 2).
- W3 mostly holds: decided-and-wrong ≤ 4% for COMPASS in noise/population/wide/tutorial,
  and accuracy among decided 0.91–0.98. But COMPASS decides far more often than the
  reference on the ODE families (population 0.53 vs 0.28, wide 0.44 vs 0.10): its se is
  too small there, consistent with the coverage failure. Offset is the exception for both
  levels: even exact WAIC is wrong on 8% of datasets with Δ > 2 se (accuracy 0.76) --
  the known unreliability of se_diff for similar models with few rows (Sivula et al.).
- Verdict: switch the reported weights to pseudo-BMA+ (already computed by `compare()` as
  `model_prob_pseudo_bma_plus`; a default change, no cost) and report Δelpd ± se_diff, but
  do not present COMPASS's MC-SE as the uncertainty of the estimate: the network error is
  larger and must come from a seed ensemble.
