# Is the gauss_jacobian_FD composed posterior accurate? (2026-10-20)

Follow-up to SINGLE_OBS.md: on one row the offset/noise networks are within target, so
any remaining error there comes from composing the rows. Here the true model is fitted to
the WAIC test datasets with the production composition (gauss_jacobian_FD,
jacobian_refresh 9, 50 steps, 1000 draws) and compared with the exact posterior, for the
global(s) and every row's locals; recovery is checked by the PIT of the true values.
(The tutorial has no globals, so nothing is composed there.)

## Phase A: from the stored WAIC records (global posterior mean / sd only)

Offset M1 (exact: quadrature), true model fitted; mean error in exact-posterior sds,
composed / exact sd, and coverage of the true global by mean ± 1 / 1.96 sd:

| networks | mean err RMS | sd ratio median [10–90%] | cover 68 / 95 composed | exact |
|---|---|---|---|---|
| exact scores (oracle), 200 | 0.03 | 1.01 [0.99, 1.04] | 0.64 / 0.97 | 0.64 / 0.96 |
| trained seed 0, 65536, 200 | 0.30 | 1.12 [1.06, 1.20] | 0.70 / 0.97 | 0.64 / 0.96 |
| trained seeds 1–4, 50 each | 0.12–0.26 | 0.97–1.11 | 0.54–0.64 / 0.92–0.98 | 0.52 / 0.96 |
| trained seed 0, 262144 (4x), 50 | 0.05 | 1.00 [0.97, 1.04] | 0.52 / 0.92 | 0.52 / 0.96 |
| noise, trained seed 0, 400 | 0.10 | 1.00 [0.97, 1.03] | 0.63 / 0.94 | 0.65 / 0.95 |

- The rule is exact with exact scores. With trained networks the global posterior is
  slightly off for some seeds (seed 0: 12% too wide, mean 0.3 sd off) and accurate at 4x
  budget; recovery coverage matches the exact posterior's everywhere.
- Population: the MCMC references did not store global summaries -> phase B.

## Phase B (`diag/composed_recovery.py`): full posteriors incl. locals

offset M1 and noise G/T (50 datasets each, quadrature) and population / population_wide
(4 x 30, emcee with restarts). Metrics for globals and locals: mean error / exact sd,
sd ratio, PIT coverage 50/90 vs the exact posterior's, corr(global, local).

Hypotheses:
- B1 offset/noise: globals as in phase A; locals within 0.1 sd and 5% width (each row's
  local is mostly determined by its own row).
- B2 population (4 globals from 8 rows, narrow prior): globals within 0.3 sd, width
  0.9–1.15; locals as accurate as on single rows (sd ratio ≈ 1), corr(g, l) reproduced
  to ±0.1; recovery coverage within 5 points of exact.
- B3 population_wide: larger errors than population (same budget over a wider prior),
  in line with the worse WAIC of exp 3b.

## Results (`composed_recovery.py analyze`, composed_recovery_summary.json, fig_composed_recovery.png)

Mean error = RMS (composed − exact mean) / exact sd; width = composed / exact sd (median
[10–90%]); coverage of the true value by the central 90% interval, composed / exact.

| | global: mean err, width, cover90 | locals: mean err, width, cover90 | corr(g, l) composed / exact |
|---|---|---|---|
| offset M1 | 0.25, 1.12 [1.07, 1.18], 0.86 / 0.88 | 0.04, 1.03, 0.90 / 0.90 | +0.01 / −0.00 |
| noise G | 0.07, 0.99, 0.94 / 0.94 | 0.05, 1.03, 0.93 / 0.92 | −0.39 / −0.39 |
| noise T | 0.11, 1.00, 0.78 / 0.80 | 0.07, 1.03, 0.87 / 0.86 | −0.52 / −0.52 |
| population LV | 0.18, 1.10, 0.95 / 0.91 | 0.14, 1.05, 0.92 / 0.88 | −0.11 / −0.11 |
| population LP | 0.23, 1.03, 0.93 / 0.94 | 0.08, 1.03, 0.92 / 0.90 | −0.06 / −0.05 |
| population SP | 0.20, 1.08, 0.90 / 0.87 | 0.16, 1.03, 0.90 / 0.89 | −0.22 / −0.24 |
| population RM | **0.60**, 0.95 [0.82, 1.13], 0.87 / 0.91 | 0.19, 1.03, 0.92 / 0.91 | −0.07 / −0.07 |
| population_wide LV | **0.45**, **0.85**, 0.92 / 0.97 | **0.44**, **1.23**, 0.94 / 0.89 | −0.11 / −0.09 |
| population_wide LP/SP/RM | 0.25–0.30, 0.98–1.05 | 0.15–0.23, 1.01–1.04 | reproduced (±0.04) |

- B1 holds: offset/noise locals 0.04–0.07 sd and 3% wide; noise globals exact; offset's
  global is the seed-0 network's 12% excess width seen in phase A.
- B2 holds except RM: population globals 0.18–0.23 sd off and ≤10% too wide, locals
  ≤0.16 sd and 3–5% wide, the global–local correlations reproduced to ±0.02, recovery
  coverage like the exact posterior's. RM's global is 0.6 sd off (dims 1 and 4: 0.8 /
  0.7 sd) with 50% coverage 0.38 vs 0.54 -- a real composed-posterior error.
- B3 only for LV: the wide LV network is 15% too narrow on the globals and 23% too wide on
  the locals (the same network whose WAIC was +40 in exp 3b); the other wide networks
  are as good as the narrow ones.
- Verdict: gauss_jacobian_FD itself is accurate; the remaining global errors (0.2 sd) are
  the trained scores summed over the rows, and are largest where the single-row network
  is weakest (RM, wide LV). Nothing here explains WAIC errors of hundreds of nats.
