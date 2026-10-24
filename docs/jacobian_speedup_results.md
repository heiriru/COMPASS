# gauss_jacobian_FD speed-ups: results

Plan: `docs/jacobian_speedup_plan.md`. Code: `src/compass/MultiObsSampler.py` (opt-in options,
defaults unchanged; the stored 0.928 run reproduces bit-exactly), tests
`tests/test_jacobian_speedup.py`, harness `jacobian_speedup/`.

Cost = network row-evaluations (observation rows x draws, finite-difference copies and the warm
start included), from the sampler's counters. Wall-clock below is from a shared GPU and only
comparable within one table.

## Variants

| name | meaning |
|---|---|
| `r<k>` | refresh the FD Jacobian every k-th composed-score call (r1 = every call) |
| `legacy` / `prod` | r1 above lambda 0.3, every 10th / 20th call below (prod = pp.py, the 0.928 run) |
| `hi<k>@<l>` / `lo<k>@<l>` | period k above / below lambda l |
| `band<a>-<b>[/k]` | refresh every call while a <= lambda^2 * median(min eig K) <= b, else never (or every k) |
| `...s`, `secant` | step 3: BFGS secant updates of the cached information between refreshes |
| `warm<l>` | step 2: Laplace warm start at lambda l (one Newton fit per dataset) |

## Toy: lean |g| hierarchy (asymmetric bimodal, exact posterior)

2,000 draws, 100 levels, churn 4, seeds 0 and 1. dW1 = W1/sd between a variant's draws and r1's
draws of the same seed; "floor" = W1/sd between the two r1 seeds. D = 2, so the FD Jacobian is
3 copies and ~75% of r1's rows.

**Where the information changes** (`trace_info.py`, n = 64). With exact scores, the cached
information K_j changes by > 1% per call only for lambda in ~[0.7, 4] (lambda^2 min eig K in ~[1, 15]);
below it < 0.01%, above it 2-5%. With the trained network it changes 80-97% per call at high
noise and 5-18% at low noise: finite-difference and network roughness, not state dependence.

**Step 1, n = 128** (trained hierarchical-attention net; floor dW1 0.037):

| variant | W1/sd exact | P+ (exact 0.722) | dW1 | cost |
|---|---|---|---|---|
| r1 | 0.109 | 0.723 | 0 | 1.00 |
| legacy | 0.118 | 0.723 | 0.010 | 0.72 |
| band1-15/5 | 0.111 | 0.720 | 0.009 | 0.66 |
| band1-15 | 0.125 | 0.720 | 0.025 | 0.58 |
| hi20@3 + lo10@0.3 | 0.135 | 0.738 | 0.042 | 0.61 |
| r2 | 0.124 | 0.735 | 0.031 | 0.62 |
| r3 | 0.126 | 0.737 | 0.033 | 0.50 |
| r5 | 0.117 | 0.732 | 0.022 | 0.40 |
| r9 | 0.109 | 0.728 | 0.030 | 0.33 |
| r15 | 0.106 | 0.724 | 0.027 | 0.30 |

With exact scores (oracle), every odd-period and band schedule is within 0.004 of r1 except
the long lags (r9/r15 differ by 0.05-0.06 but move *towards* the exact posterior, since r1 itself
over-concentrates on the majority branch). Even periods are worse everywhere (r2: P+ 0.924 vs 0.913 at
n = 16 with exact scores): DPM2 makes two calls per step, so an even period always refreshes at
the same one of the two.

**Step 2 (warm start) fails on this bimodal posterior.** The single-Gaussian Laplace start keeps
one branch: P(g>0) at n = 128 is 1.00 (lambda_s 0.5), 0.96 (1), 0.87 (2), 0.82 (3) against 0.72 exact
(trained net; the oracle is the same). As planned: the warm start is for unimodal posteriors only.

**Step 3 (secant) makes it worse.** r5s/r9s/r15s are further from r1 than r5/r9/r15 without
secant updates (n = 128 trained: dW1 0.11/0.18/0.21 vs 0.02/0.03/0.03; oracle 0.06/0.07/- vs
0.02/0.05/0.06), and r15s failed with a non-positive-definite composed covariance. The secant
equation is exact for a Gaussian row posterior (unit tests), but consecutive DPM2 calls are ~28%
apart in lambda (churn) and the state moves by ~lambda between them; on a truncated/bimodal row
posterior that pair does not describe the curvature at the new state. With the plain lag already
free (below), the secant update is not worth pursuing; it stays off by default.

## Partial pooling (hierarchical + Rg, 1.5x-wide prior; D = 10, n = 100, 10 datasets)

Reference: the stored 0.928 run (prod schedule, seed 7). dG/dL = mean |median - median_ref| / sd_ref
over the 7 globals / 300 locals; same seed as the reference (common random numbers), floor = prod
reseeded (seed 8). r and RMSE over these 10 datasets only.

| variant | dG | dL | sd ratio | r | RMSE | rows/dataset | x prod | s/dataset* |
|---|---|---|---|---|---|---|---|---|
| prod, seed 8 (floor) | 0.128 | 0.126 | 1.00 | 0.889 | 0.444 | 7.80M | 1.00 | 355 |
| r1 | 0.032 | 0.017 | 0.98 | 0.894 | 0.448 | 11.76M | 1.51 | 464 |
| r3 | 0.036 | 0.021 | 0.98 | 0.892 | 0.453 | 4.61M | 0.59 | 193 |
| r5 | 0.039 | 0.023 | 0.98 | 0.891 | 0.451 | 3.18M | 0.41 | 146 |

\* shared GPU, three to four jobs at once.

GH on this problem costs ~3.5M rows per dataset (pilot: 100 subjects x 256 draws x ~98 calls
= 2.5M, plus ~1.0M for the draws), so r5 is already at GH's cost with no measurable accuracy loss.

### Lane B: longer lags, warm starts, secant (same 10 datasets, same reference)

| variant | dG | dL | sd ratio | r | RMSE | rows/dataset | x prod |
|---|---|---|---|---|---|---|---|
| r9 | 0.047 | 0.028 | 0.99 | 0.896 | 0.437 | 2.19M | 0.28 |
| r15 | 0.062 | 0.038 | 1.01 | 0.890 | 0.434 | 1.75M | 0.22 |
| band1-15/5 | 0.035 | 0.021 | 0.99 | 0.894 | 0.446 | 5.82M | 0.75 |
| r5 + secant | 0.110 | 0.040 | 1.05 | 0.881 | 0.475 | 3.18M | 0.41 |
| warm 1.0 + prod | 0.118 | 0.127 | 0.99 | 0.882 | 0.463 | 3.30M | 0.42 |
| warm 0.5 + prod | 0.150 | 0.147 | 0.91 | 0.867 | 0.508 | 1.97M | 0.25 |
| warm 0.5 + r5 | 0.150 | 0.145 | 0.90 | 0.867 | 0.508 | 1.64M | 0.21 |
| warm 0.3 + r5 | 0.555 | 0.288 | 0.84 | 0.792 | 0.808 | 1.34M | 0.17 |

The warm starts draw their initial state from a different random stream, so their dG/dL compare
with the reseeded floor (0.128/0.126), not with the common-random-number rows. warm 1.0 is within
the floor but no cheaper than r5; warm 0.5 is 9-10% too narrow and loses 0.027 in r. The Laplace
fit itself is cheap (53k rows per dataset); the problem is that p_lambda(theta | X) is not Gaussian
enough at lambda 0.5 for the start to be unbiased. The secant update again moves the posterior
(dG 0.110 vs 0.039 for r5 without it, 5% too wide).

### Clean wall-clock (GPU 6 otherwise idle, datasets 0-2, one after another)

| method | s/dataset | vs GH |
|---|---|---|
| gauss_hierarchical (GH, incl. pilot) | 74 | 1.00 |
| gauss_jacobian_FD, prod (lo20@0.3) | 163 | 2.20 |
| gauss_jacobian_FD, r5 | 69 | 0.93 |
| gauss_jacobian_FD, r9 | 49 | 0.66 |

## Conclusions

1. **Step 1 is the win: refresh the FD Jacobian every 9th composed-score call (`jacobian_refresh=9`,
   no low-noise special case).** On partial pooling it is 3.3x faster than the production schedule
   and 1.5x faster than gauss_hierarchical, with no accuracy loss measurable against the seed floor
   (dG 0.047 vs floor 0.128; r 0.896, RMSE 0.437 on the 10 datasets). On the lean toy r9/r15 stay
   within the floor with the trained network. Use odd periods: with DPM2's two calls per step an
   even period always refreshes at the same call and is measurably worse (r2).
   Why the old advice ("lag hurts at high noise") no longer holds: the cache stores the
   lambda-free information Lambda_j - lambda^-2 I, which is exact across noise levels for a Gaussian
   row posterior; the measurement behind that advice predates or bypassed it.
2. **Step 2 (Laplace warm start) is not worth it.** It drops minor modes of multimodal posteriors
   (lean toy: P+ 1.00/0.96/0.87/0.82 at lambda_s 0.5/1/2/3 vs 0.72), and on the unimodal partial
   pooling it is either biased (lambda_s <= 0.5) or no cheaper than r5 (lambda_s 1). Kept as an
   opt-in API (`laplace_fit`, `laplace_draw`, `sample(start_lambda=, init_state=)`).
3. **Step 3 (secant BFGS updates) hurts** on every bench (toy oracle, toy trained, partial pooling)
   and crashed once (r15s, non-PD composed covariance). Exact for Gaussian rows (unit-tested), but
   consecutive calls differ by ~28% in lambda and ~lambda in state. Kept opt-in, off by default.

Not yet done: a second seed of r9 on all 30 datasets; IK / GM benches.

2026-10-23: `gauss_jacobian_FD_r9` is now pp.py's default `--method` (the library default
`jacobian_refresh` stays 1 until the IK / GM benches are run). Re-timed after the sampler refactor
(compact HierarchicalState etc.), datasets 0-4, 100 draws, GPU otherwise idle: production schedule
165 s/dataset (old and new sampler code identical, draws bit for bit), r9 44 s/dataset (3.75x).
