# WAIC improvements: four separate experiments (2026-10-18)

Follow-up to the A4 results (README.md). Each experiment changes one thing and is
graded against an exact or exact-likelihood reference. The predictions were written
before any of the runs below finished.

## Baseline (from README / A4)

| family | COMPASS vs reference pick | recovery COMPASS / exact | ECE COMPASS (null95) | main error |
|---|---|---|---|---|
| offset, trained | 91% | 0.677 / 0.710 | 0.091 (0.064) | ΔWAIC(M1−M0) shifted +1.20 (sd 0.73) |
| noise, trained | 93% | 0.703 / 0.79 | 0.27 (0.056) | outlier datasets, sd 6.7 |
| population (mine), trained | 86% | 0.825 / 0.833 | 0.055 (0.027) | misspecified WAIC +86..+1044 |
| tutorial (no noise, old nets) | — | 1/4 | — | likelihood not a density |

## Reference fix found on the way: MCMC walker restarts

The misspecified ODE fits have a rugged likelihood; emcee walkers get stranded in local
optima hundreds of nats below the best one. Their posterior mass is negligible, but they
inflate Var_s log p, i.e. p_waic. On three tutorial rows (RM fitted to LV data) the
plain run gave WAIC 8565 (p_waic 4327); with two restart rounds (walkers > 20 nats below
the best moved onto jittered copies of the good ones, then re-burned) two seeds agree,
115 and 120. The true model is unchanged (−352.07 vs −352.06, and −352.25 by prior
importance sampling). The population MCMC references of A4 were made without restarts,
so their misspecified-model WAICs are suspect; they are recomputed with restarts
(tag ``ref_r2``) before any population number below is quoted.

## 1. Tutorial with observation noise + likelihood recipe (family ``tutorial``)

``tutorial_family.py``: the tutorial's simulator exactly (Euler 0.01, clamp, t = 0..19,
/100, all four log-parameters per trajectory from N(MOCK, 0.1^2)) plus N(0, 0.05^2)
observation noise; 20 trajectories per dataset, no globals. Four networks trained with
the A4 recipe (legacy transformer 128x4, VESDE 25, log_sigma, inference menu, EDM,
131072 simulations, final epoch). 30 datasets per true model, 500 draws, Hutchinson
PF-ODE; reference: per-row MCMC with restarts (exact likelihood).

Predictions:
- 1a. The true model's COMPASS WAIC is within ±10 of MCMC (0.1–0.2 nats/row network
  error, B5/B6), and recovery is within 5 points of MCMC-WAIC recovery.
- 1b. Misspecified models' posteriors leave the narrow training prior (as in the
  population family), so their WAIC is overstated, but the winner is mostly preserved:
  decision agreement with MCMC ≥ 85%.
- 1c. The p_waic,i > 0.4 warning fires on every row of every model even when WAIC is
  right: each row has four own parameters, so p_waic,i ≈ 2 for a correct model. The
  threshold is for row-level LOO without local parameters and is uninformative here.

## 2. Network likelihood error: seed ensemble and budget (offset family)

Train seeds 1–4 at 65536 simulations and seed 0 at 262144 (4x) for M1 and M0. Run the
first 50 datasets of each true model (100) with every network; seed 0 at 65536 exists.
Ensemble WAIC = mean over the five seeds' WAICs per dataset and model.

Predictions:
- 2a. The +1.2 ΔWAIC shift is partly seed-specific: per-seed shifts scatter with an sd
  of 0.3–0.8, not all of the same size. The 5-seed ensemble cuts the shift to 0.5–0.9 and
  the per-dataset error sd by ~1.5x, but does not remove it (B5/B6: no cancellation).
- 2b. 4x budget reduces the shift by ~1.5–2x.
- 2c. Neither changes recovery by more than the gap to exact WAIC (0.677 → ≤ 0.710).

## 3. Training support: wider training prior, and an out-of-support flag

3a (flag, no new runs): from the posterior draws already recorded, the fraction of draws
outside 3 prior sd of the training prior. Prediction: it separates the models whose
WAIC error exceeds 10 nats from the rest with AUC > 0.9 on the population family.

3b (wider prior, family ``population_wide``): same models and test datasets as
``population`` (data still from the narrow prior), networks trained with all prior sds
x3. The analysis prior is then the wide one, for COMPASS and for the MCMC reference
alike. Predictions: misspecified-model WAIC error falls from +150..+1000 to under ~20;
decision agreement with MCMC rises from 86% to ≥ 95%; the true model's error grows
slightly (the same budget spread over 3^6 times the parameter volume).

## 4. PSIS-LOO instead of WAIC (all families, from the same draws)

``compass.information_criteria.psis_loo`` (Pareto-smoothed importance sampling, k
diagnostic) and ``compare(criterion="loo")``; every run above records both. Unit test:
conjugate Gaussian, elpd error −0.007 ± 0.048 nats over 50 datasets (MC-SE 0.048).
Exact reference: quadrature LOO (log Z(x) − log Z(x_−i)) for offset/noise, PSIS-LOO on
MCMC draws for population/tutorial.

Predictions:
- 4a. With locals, leaving out row i returns l_i to its prior, so the importance
  weights are as wide as the prior/posterior ratio of l_i. Offset/noise (local
  posterior sd ~ 0.7 of the prior): k mostly < 0.7, LOO matches the exact LOO within
  MC error at the oracle level.
- 4b. Population/tutorial (locals identified far more sharply than their prior): k > 0.7
  on most rows, so PSIS-LOO is unreliable exactly where WAIC was flagged; it does not fix
  those families.
- 4c. The network log p error enters LOO as it enters WAIC (a per-row offset passes
  straight through the harmonic-mean estimator), so the trained-network shifts stay.

## Results

### 1. Tutorial + noise (120 datasets, all done; `improvements.py exp1`, fig_exp1_tutorial.png)

| | COMPASS WAIC | MCMC WAIC |
|---|---|---|
| accuracy | 0.775 (LV, LP, SP 30/30; RM 3/30, LP picked 27/30) | 1.00 |
| decision agreement | 0.775 (0.80 when the MCMC gap > 4) | — |
| ECE (null95) | 0.112 (0.000) | 0.003 (0.006) |
| true-model WAIC − MCMC, median [10%, 90%] | +373 [+301, +837] (+18.6 per row) | — |
| p_waic of the true model, median | 69 | 30 |
| misspecified WAIC − MCMC, median [10%, 90%] | +1188 [−8578, +39458] | — |

- 1a fails: the true-model error is ~+370, not ±10. 1b fails narrowly (77.5% < 85%), and
  only on RM-true data: the exact RM−LP margin is only 3–18 nats, while COMPASS's RM
  and LP WAICs are each ~300–400 too high and differ by ~50 more for RM, so LP wins.
- Cause (diag/hutch_noise.py and the MCMC eigen-analysis): not Hutchinson noise (sd
  0.08–0.13 nats/row). The exact posterior per trajectory is a narrow ridge (LV
  eigen-sd ratio 0.086); the network posterior matches the marginal sds but not the
  correlation, so the exact log-lik varies 22 per row at its draws vs 1.75 at MCMC's
  (decorrelated MCMC draws: 701). The PF-ODE log p is also biased −2.5 (LV) to −8 (LP)
  nats/row. WAIC needs the posterior right in its narrowest direction; model choice
  does not, which is why the clear-cut cases are all correct.

### 2. Seed ensemble and 4x budget (offset, 100 datasets; `improvements.py exp2`, fig_exp2_ensemble_budget.png)

ΔWAIC(M1−M0) error, COMPASS − exact quadrature (exact divergence, 1000 draws):

| networks | shift ± se | per-dataset sd | RMSE | recovery (exact 0.71) | agreement |
|---|---|---|---|---|---|
| seed 0, 65536 | +1.13 ± 0.07 | 0.71 | 1.33 | 0.66 | 0.89 |
| seeds 1–4, 65536 | +0.33, −0.14, +0.07, −0.29 | 0.53–0.63 | 0.55–0.71 | 0.64–0.69 | 0.87–0.93 |
| 5-seed ensemble | +0.22 ± 0.05 | 0.51 | 0.55 | 0.72 | 0.95 |
| seed 0, 262144 (4x) | +0.03 ± 0.05 | 0.51 | 0.51 | 0.67 | 0.92 |
| exact scores (oracle, 400 datasets) | +0.09 | 0.53 | — | — | — |

- The +1.2 shift of the baseline was seed 0's bad luck: the seed-to-seed sd of the shift
  is 0.56, and seed 0 is the outlier (2a: shift is seed-specific, confirmed; the
  ensemble shift 0.22 is below the predicted 0.5–0.9 because seed 0 was the worst).
- Both fixes reach the floor: the remaining per-dataset sd 0.51 equals the exact-score
  sd 0.53, i.e. the Monte-Carlo noise of 1000 composed draws (recorded MC-SE of ΔWAIC
  0.47), not the network. Per-dataset errors of different seeds correlate 0.60 because
  they share that floor and the datasets.
- 2b exceeded: 4x budget removes the shift entirely (1.13 → 0.03), not 1.5–2x.
- 2c holds: recovery stays within the gap to exact (0.64–0.72 vs 0.71); only the
  ensemble's agreement with exact rises (0.89 → 0.95). Further gains need more draws
  (MC noise), not better networks.

### 3a. Out-of-support flag (population, 480 dataset×model; `improvements.py exp3a`, fig_exp3a_support_flag.png)

Re-baseline with the restart references (`ref_r2`): restarts changed no population
MCMC WAIC (median 0.0, 10–90% ±0.2) — the stuck-walker problem was specific to the
tutorial ODEs. Population numbers: COMPASS accuracy 0.825 vs MCMC 0.808, agreement 83%
(100% when the MCMC gap > 4), ECE 0.055 (null95 0.027) vs MCMC 0.024 (0.037).

Flag z_max = max_d |posterior mean of g_d − training prior mean| / prior sd:

| cut | flagged | recall of |err| > 10 | precision | true model flagged |
|---|---|---|---|---|
| 2 | 69% | 0.96 | 0.91 | 15% |
| 3 | 62% | 0.93 | 0.98 | 0.8% |
| 4 | 57% | 0.87 | 0.99 | 0% |

- Prediction AUC > 0.9 holds: AUC 0.972. At cut 3 the flag catches 93% of the bad WAICs
  and fires on the true model once in 120 datasets. Errors above 100 nats occur only
  with z > 3.
- It does not change decisions: dropping the flagged models and picking among the rest
  gives the same accuracy (0.825) and agreement (83%); flagged models were already
  losing. Its value is to say which WAIC values (and weights) not to trust; the
  remaining disagreements are between unflagged models within a few nats.

### 3b. Wider training prior (population_wide vs population, 120 datasets each; `improvements.py exp3b`, fig_exp3b_wide_prior.png)

Both graded against MCMC with restarts (wide-prior analysis prior for population_wide).

| | population (narrow nets) | population_wide (x3 nets) |
|---|---|---|
| COMPASS / MCMC accuracy | 0.825 / 0.808 | 0.742 / 0.758 |
| agreement with MCMC (gap > 4) | 0.83 (1.00) | 0.72 (0.93) |
| misspecified abs. error median / 90% | 155 / 776 | 85 / 388 |
| true-model abs. error median | 3.4 | 7.0 (LV +40, SP +20, LP +3.6, RM +2.6) |
| ECE COMPASS (null95) | 0.055 (0.027) | 0.066 (0.037) |
| misspecified posteriors > 3 training sd | 82% | 52% |

- Prediction fails. The wider prior halves the misspecified error but leaves it at
  tens to hundreds of nats: about half of the misspecified posteriors still lie more
  than 3 *wide* sds (9 narrow sds) out, so x3 is not wide enough for them.
- The true model gets worse, as predicted, but far more than "slightly": the LV net is
  +40 nats on every dataset (a network-level likelihood offset), SP +20. Together this
  lowers agreement 83% → 72%. Also, the analysis prior changes the task itself: MCMC
  accuracy drops 0.81 → 0.76 with the wide prior.
- Verdict: widening the training prior trades a support error on wrong models for a
  likelihood error on the right one at this budget. The support flag (3a) is the
  cheaper fix.

### 4. PSIS-LOO (`improvements.py exp4`, fig_exp4_psis_loo.png)

Offset, COMPASS PSIS-LOO vs exact quadrature LOO (400 exact-score datasets; 100 per trained net):

| networks | ΔLOO err mean ± sd | ΔWAIC err mean ± sd | per-model LOO bias | rows k > 0.7 | acc LOO / WAIC (exact 0.71) |
|---|---|---|---|---|---|
| exact scores | +0.08 ± 0.87 | +0.09 ± 0.53 | −2.1 | 22% | 0.67 / 0.69 |
| trained, 6 nets | −0.30..+1.07 ± 0.74–0.87 | −0.29..+1.13 ± 0.46–0.71 | −1.4..−2.2 | 15–19% | 0.64–0.66 / 0.64–0.69 |

Exact criteria (quadrature, 400 datasets each), calibration of the model weights:

| | evidence | WAIC | LOO |
|---|---|---|---|
| offset acc / ECE (null95 ≈ 0.049) | 0.69 / 0.025 | 0.71 / 0.060 | 0.71 / 0.060 |
| noise acc / ECE (null95 ≈ 0.037) | 0.835 / 0.033 | 0.79 / 0.106 | 0.84 / 0.028 |

ODE families, rows with Pareto k > 0.7: population (MCMC) 79%; population_wide 99% (MCMC)
/ 92% (COMPASS); tutorial 99% / 87%.

- 4a partly: on offset k > 0.7 on ~20% of rows (median k_max 0.82–0.88). The LOO
  *difference* matches exact LOO without bias at the exact-score level (+0.08), but it is
  noisier than ΔWAIC (sd 0.87 vs 0.53), and each model's LOO is biased by ~−2 (the
  truncated weights of the high-k rows), which cancels between the two models.
- 4b confirmed: on the ODE families PSIS is unreliable on 80–99% of rows, even with exact
  MCMC draws. LOO does not rescue them; COMPASS LOO equals COMPASS WAIC on the tutorial
  (accuracy 0.775 both, true-model error +357 vs +373).
- 4c confirmed: the trained networks' shifts pass through unchanged (seed 0: +1.07 LOO vs
  +1.13 WAIC).
- Unexpected, at the exact level: on the noise family (Gaussian vs Student-t) WAIC
  itself is miscalibrated (ECE 0.106) and exact LOO is not (0.028, as good as the
  evidence; accuracy 0.84 vs 0.79). WAIC is 4.2 nats off LOO in ΔIC there: the
  heavy-tailed model's rows have heavy-tailed log-lik, where WAIC's variance penalty
  fails. The noise COMPASS runs predate the LOO code, so trained-level LOO on noise is
  not measured.
