# Single-observation accuracy of the network posterior and PF-ODE likelihood (2026-10-19)

Question: before composing over many rows, is the network good enough on ONE row (the
setting it was trained on: one joint draw [g, l, x] per row)? If not, the WAIC errors of
the composed runs start here, and the networks must be improved before anything else.

## Design (``diag/single_obs.py``)

Families offset, noise, population, population_wide, tutorial; seed-0 networks of the
A4 runs (65536 / 131072 simulations). For every data model D, 20 single rows from D's
prior predictive (exactly the training distribution of D's network); every candidate
model M is fitted to every row, so D = M is in-distribution and D != M misspecified.

Per (row, M):
- exact posterior: single-row emcee with restarts (12 800 draws; 200 used for PF-ODE);
- network posterior: 200 draws from M's network, conditioned on the row alone
  (``multi_obs_inference=False``, no composition rule involved);
- exact log p(x | theta) and PF-ODE log p (exact divergence, 50 steps) at the MCMC
  draws, at the network draws, and at the true parameters (D = M only).

Metrics:
- likelihood: PF-ODE − exact at the true theta and at the MCMC draws (bias, sd), and the
  spread of that error across draws of one row (it enters p_waic);
- posterior: mean error in exact-posterior sds, sd ratio per dimension, the sd ratio along
  the exact posterior's narrowest direction, and the mean Mahalanobis d^2 of network draws
  under the exact posterior's covariance (≈ D for a correct posterior);
- what WAIC sees: the row's elpd_waic = lppd − p_waic in the four combinations
  {MCMC, network draws} x {exact, PF-ODE likelihood}, error vs the MCMC-exact value,
  split into posterior error (network draws, exact lik) and likelihood error (MCMC draws,
  PF-ODE lik).

Precision target. A per-row elpd error b adds N·b to a dataset's elpd; the composed
datasets have N = 8–20 rows and the model gaps that matter are a few nats, so
|b| ≲ 0.1 nats/row (N·b ≲ 2) is "precise enough"; 0.1–0.5 is marginal; above 0.5 must
be fixed.

## Hypotheses (written before running)

- H1 likelihood, in-distribution: offset/noise PF-ODE error |bias| ≤ 0.2, sd ≤ 0.3 nats
  per row (B5/B6: 0.11–0.19). ODE families: bias of several nats per row (exp 1 measured
  −2.5 / −8 on the tutorial at composed draws), so the likelihood fails the target there.
- H2 posterior, in-distribution: offset/noise matches the exact posterior (sd ratio
  0.9–1.1, d^2 ≈ D). Tutorial/population: marginal sds right, narrowest direction too
  wide by ≥ 2x (the ridge missed in exp 1), d^2 ≫ D.
- H3 WAIC split, in-distribution: offset/noise within target; on the ODE families the
  posterior term dominates the elpd error (off-ridge draws inflate p_waic).
- H4 misspecified rows: both errors grow by an order of magnitude (x outside the
  model's training support), most on the ODE families.

## Results

`diag/single_obs.py analyze` -> results/single_obs_summary.json, figures/fig_single_obs.png.
Medians over rows; elpd errors per row vs MCMC with the exact likelihood (12 800 draws).

| | offset in / mis | noise in / mis | tutorial in / mis |
|---|---|---|---|
| PF-ODE − exact at true θ (mean, sd) | +0.01, 0.06 | −0.02, 0.10 | −6.3, 2.7 |
| PF-ODE − exact at exact-posterior draws | +0.00 / +0.01 | 0.00 / −0.00 | −6.6 / −24 |
| elpd error: posterior term | −0.03 / −0.05 | +0.01 / −0.01 | −7.2 / −5100 |
| elpd error: likelihood term | +0.02 / +0.02 | +0.02 / +0.01 | −6.2 / −15 |
| elpd error: both (what COMPASS sees) | −0.01 / −0.02 | +0.02 / +0.00 | −8.4 / −58 |
| p_waic exact / at net draws | 0.34 / 0.35 | 0.53 / 0.53 | 1.5 / 7.9 (mis 40 / 1750) |
| sd ratio: marginal / narrowest direction | 1.04 / 1.02 | 1.03 / 1.00 | 0.98 / 1.7 (mis 1.2 / 5.2) |
| Mahalanobis d²/D | 1.08 | 1.05 | 1.9 (mis 690) |

- Offset and noise: everything within the ±0.1 target, at the level of the 200-draw MC
  noise; H1–H3 confirmed. Single-row networks are not the problem there; whatever is
  left in those families comes from composition and draw count (exp 2).
- Tutorial fails on one row already, in-distribution:
  - likelihood: −6.3 nats per row at the true θ (per model: LV −4.1, LP −8.4, SP −0.7,
    RM −8.7), i.e. 60x the target. Not the eps smoothing: σ(eps) = 0.032 against a
    smallest standardised noise sd of 0.036 predicts only −0.24. The 40-dim x | θ is very
    sharp (noise 0.05 against marginal sds up to 1.4), so the log-density comes from the
    scores at the smallest noise levels, where training is weakest
    (compass-score-calibration-small-lambda).
  - posterior: marginals right (0.98) but the narrowest direction 1.7x too wide; SP on
    its own data is 4.6x too wide there (elpd posterior term −357).
- Tutorial misspecified: the network posteriors are far from the exact ones (mean off by
  26 posterior sds, d²/D 690): the networks never saw such x. H4 confirmed.
- Hence the composed tutorial errors of exp 1 (+370 per 20-row dataset, ≈ 18 per row)
  are already present in single rows; composition is not the cause. Improve the
  networks before anything else for the ODE families.

Population families (added 2026-10-20):

| | population in / mis | population_wide in / mis |
|---|---|---|
| PF-ODE − exact at true θ (mean, sd) | +0.04, 0.30 | −0.14, 0.47 |
| elpd error: posterior term | −0.56 / −11.5 | −1.9 / −11.8 |
| elpd error: likelihood term | +0.17 / +0.25 | +0.12 / +0.24 |
| elpd error: both (COMPASS) | −0.25 / −2.8 | −1.1 / −2.6 |
| p_waic exact / at net draws | 1.7 / 2.2 | 2.3 / 4.0 |
| sd ratio marginal / narrowest direction | 1.04 / 1.12 | 1.02 / 1.30 |

- Population likelihood is close to target (bias 0.04, sd 0.3 per row; elpd term 0.17):
  much better than the tutorial, because the population noise is not sharp relative to
  x's spread.
- The posterior term dominates (−0.56 per row in-distribution, −11 misspecified): the
  network posterior is 12% (wide: 30%) too wide along the narrowest direction, and
  p_waic rises 1.7 → 2.2. On the wrong models the posteriors are off the ridge, which is
  what makes the misspecified WAIC errors large.
