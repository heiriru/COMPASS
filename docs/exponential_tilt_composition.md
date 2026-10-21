# Composing likelihoods as exponential tilts of the prior

Measured 2026-09-15. Code: `inverse_kinematics/experiments/tilt_fit.py` (the
estimator), `Gaussian_mixtures_compositional_score_modelling/analytic_composition.py`
and `inverse_kinematics/experiments/ik_analytic_composition.py` (the two
rule-isolating harnesses), `tests/test_tilt.py` in the Gaussian-mixtures
benchmark.

## The failure this addresses

For one shared block with independent locals,

    log p(g | x_1..n) = sum_j log p(g | x_j) - (n - 1) log p(g) + const,

so the composed log-density is a **sum of n estimated curves**. A per-factor
error of `delta` nats becomes `n*delta` in the composed log-density while the
composed posterior's own width shrinks like `1/sqrt(n)`: in units of the
composed sd, a systematic per-factor error grows like `n^1.5` and a random one
like `n`. Every arm built on "estimate each posterior, then multiply" -- KDE
product, `mixture_hierarchical` with K by BIC, the likelihood form, either of
them bagged -- degrades monotonically in `n` for that reason, and no amount of
work on the estimator changes the exponent.

Two measurements pin where the damage comes from.

**Where the composed posterior sits.** On the Gaussian-mixtures benchmark the
prior is `0.5 N(-2.5, 0.4^2) + 0.5 N(+2.5, 0.4^2)`. One observation barely moves
it, so every `p(g | x_j)` has its mass at `g = +-2.19`. But `n` observations do
overcome it, and in the constructed regimes the data centre near zero, so the
composed posterior sits at `g ~ 0.2` -- five sd into the tail of *every* factor,
where the density is about `e^-13` and a 20,000-draw sample contains no draws at
all. The composition needs each of those `n` log-densities to ~0.01 nats in a
region no sample has ever visited. That is not an estimator that needs
improving; it is a question samples cannot answer.

**What the `(n-1)` power then does.** `-(n-1) log p(g)` reaches `+140` nats in
the prior's valley at `n = 8`. It multiplies pure extrapolation error exactly
where the factors are least known, and the product collapses onto a spike at
`g = 0` (visible as the magenta curve in `09_tilt_gm.png` from `n = 32` on).

## The change

The object whose tail we cannot estimate is the *posterior*. What the
composition actually needs is the **likelihood ratio** `L_j(g) ~ p(g|x_j)/p(g)`,
and a likelihood in the shared parameter carries none of the prior's structure.
On this benchmark `L_j(g) = N(x_j; g, v)` -- one broad Gaussian, log-quadratic
in `g`, fixed by two numbers -- against a sharply bimodal posterior.

So model the likelihood as a log-polynomial and fit it as an exponential family
whose **base measure is the known prior**:

    q_theta(g) = p(g) exp(theta . phi(g)) / Z(theta),   phi(g) = (g, g^2, ..., g^d).

Three consequences:

1. **No density estimate anywhere.** The MLE of an exponential family is moment
   matching: `E_theta[phi] = mean of phi over the draws`. Only the first `d`
   sample moments of each observation's draws are used. No KDE, no EM, no
   component count, no bandwidth, no beam, no pruning.
2. **The composition is exact and closed form.** A product of exponential tilts
   of one base measure is the tilt by the *sum* of the parameters, so
   `log p(g|x_1..n) = log p(g) + sum_j theta_j . phi(g) + const`. The
   `prior^(1-n)` power is **absent**, not clipped or masked, and with it the
   valley spike. Composing is adding `n` short vectors.
3. **The error stops growing with n.** Composed precision is `sum_j gamma_j` and
   composed mean is `sum_j beta_j / sum_j gamma_j`. With independent per-factor
   moment noise the mean's error is `~ sigma_beta / (sqrt(n) gamma)` while the
   composed sd is `~ 1/sqrt(n gamma)`, so **error in composed-sd units is
   `O(sigma_beta / sqrt(gamma))` -- independent of n** and `O(1/sqrt(S))` in the
   per-observation draw count. That is the whole point, and it is what the
   sweeps show.

`fit_all` picks `d` per observation by BIC over a supplied candidate list, so a
benchmark whose likelihood is quadratic is not charged for unused flexibility.

## Verification that it recovers the right thing

`tests/test_tilt.py`. Conditioned on the component, `x_j | g ~ N(g, v)`, so the
exact tilt is `beta_j = x_j/v`, `gamma_j = 1/v`. From draws alone:

| n | fitted `gamma` | truth | composed sd | exact |
|---|---|---|---|---|
| 4 | 0.8767 +- 0.0100 | 0.8850 | 0.3201 | 0.3196 |
| 16 | 0.8867 +- 0.0083 | 0.8850 | 0.2212 | 0.2214 |
| 100 | 0.8878 +- 0.0204 | 0.8850 | 0.1026 | 0.1027 |

BIC selects `d = 2` and never more when 2 is the truth.

## Gaussian mixtures, composition rule isolated

`analytic_composition.py`: single-observation draws taken **exactly** from the
closed-form two-component posterior, so no network and no sampler are involved
and every number is the rule's own error. Median W1/sd over 5 replicates,
20,000 draws per observation.

| regime | n=2 | 4 | 8 | 16 | 32 | 64 | 100 |
|---|---|---|---|---|---|---|---|
| natural, mixture | 0.009 | 0.009 | 0.016 | 0.020 | 0.042 | 0.016 | 0.015 |
| natural, **tilt** | 0.007 | 0.003 | 0.014 | 0.007 | 0.008 | 0.019 | 0.023 |
| asymmetric, mixture | 0.019 | 0.041 | 0.171 | 0.601 | 1.062 | 1.030 | 1.025 |
| asymmetric, **tilt** | 0.013 | 0.020 | 0.020 | 0.078 | 0.039 | 0.055 | **0.017** |
| symmetric, mixture | 0.022 | 0.045 | 0.493 | 0.356 | 0.891 | 0.841 | 0.813 |
| symmetric, **tilt** | 0.013 | 0.021 | 0.035 | 0.080 | 0.044 | 0.041 | 0.056 |
| antisymmetric, mixture | 0.948 | 0.960 | 0.956 | 0.943 | 0.917 | 0.876 | 0.846 |
| antisymmetric, **tilt** | 0.090 | 0.349 | 0.207 | 0.114 | 0.269 | 0.212 | 0.217 |

The tilt is flat in `n` in every regime, as predicted. `mixture_hierarchical`
rises to ~1.0 wherever the truth sits off the prior's modes. The `natural`
regime -- where `g` is drawn from the prior, so the composed posterior lies in
the factors' bulk -- is the one case where the old rule was already fine, which
is consistent with the diagnosis: its failure is about *where* the answer is,
not about `n` as such.

The antisymmetric regime (observations in `+-` pairs at the prior modes, exactly
symmetric bimodal posterior) is the tilt's weakest: 0.09-0.35. A degree-2 tilt
of a two-component prior can place and scale the branches but cannot reweight
them freely, and here the reference weights are exactly 0.5/0.5 with wide
branches.

## Gaussian mixtures through the real sampler

`run_scaling.py --oracle` -- exact diffused score, real DPM2 sampler, asymmetric
regime, 100 steps, 20,000 stage-1 draws, 2 replicates. So this adds the
sampler's discretisation to the table above but still no network error.

| n | tilt | mixture | Langevin + F-NPSE |
|---|---|---|---|
| 100 | **0.11** @ 166s | 0.97 @ 590s | 11.9 @ 7s |

### And through the trained network

Same sweep without `--oracle`, asymmetric regime, median of 2 replicates. This
is the configuration the benchmark actually reports, so it is the one to quote.

| n | tilt | mixture | Langevin + F-NPSE |
|---|---|---|---|
| 4 | **0.050** / 26s | 1.089 / 42s | 0.525 / 12s |
| 16 | **0.188** / 101s | 1.083 / 172s | 2.590 / 40s |
| 64 | **0.035** / 406s | 1.009 / 708s | 8.204 / 155s |

The branch mass tells the story more directly than `W1`: at `n = 64` the tilt
returns 0.740 and 0.750 against an exact 0.750, while `mixture_hierarchical`
returns 0.071 and 0.059 -- it has put essentially all the mass on the wrong
branch, which is the `prior^(1-n)` spike in the valley. The previous best on
this benchmark at `n = 4` was 0.170 (likelihood form plus bagging); the tilt is
0.050.

### Oracle score, continued

At `n = 100` the tilt is 9x more accurate **and 3.6x faster** than
`mixture_hierarchical`: the EM fit it replaces was most of that arm's wall
clock, and two moments cost nothing. F-NPSE is cheap and, in this regime, not
usable. (Wall clock here was measured on a shared card; the ratios are stable
but absolute seconds are contended.)

## The inverse-kinematics referee: read this before quoting any n >= 32 number

Every inverse-kinematics number here is a `W1` against one ABC reference, and
that reference is itself a Monte-Carlo estimate whose shared log-posterior is
`log prior + sum_j log p_h(x_j | a)` -- **a sum of n estimated curves, amplified
by n exactly the way a composition rule's error is**. So the referee degrades
with n for the same reason the methods do. Measured
(`reference_noise_floor.py`): `W1` between two references of the *same* data
differing only in Monte-Carlo seed, and the branch-mass gap between them:

| n | draws | W1(ref, ref) | abs branch-mass gap |
|---|---|---|---|
| 8 | 40,000 | 0.073 | 0.021 |
| 8 | 200,000 | **0.019** | 0.007 |
| 32 | 40,000 | 0.492 | 0.246 |
| 32 | **200,000** | **0.424** | **0.196** |
| 64 | 40,000 | 0.313 | 0.123 |

The graded bundles use 200,000 draws, so the 200k rows govern. A 5x draw
increase at n = 32 moved the floor only 0.492 -> 0.424, not the 2.2x pure
undersampling would give, so **more draws will not fix it**.

What that does and does not invalidate. The 0.424 is the spread between two
*independent* references, so it is not a floor on everything: grading two
methods against the *same* reference cancels much of the referee's error and
rankings survive better than the number suggests. But **absolute `W1` at n = 32
carries roughly +-0.2-0.3 of referee uncertainty**, and every rule and composed
arm measured at n = 32/64 sits in 0.48-0.79 -- those are not distinguishable from
one another. The "rules plateau near 0.5 whatever you spend" pattern is at least
partly the referee's own scatter.

Yet those posteriors are *qualitatively* different: dip ratio 0.00 for the
composed arms against 0.63-1.00 for the rules, sd 0.43 against 0.20-0.30. `W1`
against this reference is simply not sensitive to what separates them.

**So at n >= 32 on inverse kinematics, use the shape statistics and the endpoint
residual, not `W1`.** The endpoint residual contains no reference at all. It
ordered three networks cleanly and monotonically (0.0184 -> 0.0157 -> 0.0137)
where `W1` ordered them backwards. At n = 8 the floor is 0.019 and `W1` is fully
usable. Also: the n = 100 reference in this study was built at 60,000 draws, not
200,000, so it is the least trustworthy of all.

The Gaussian-mixtures results are untouched by any of this -- that benchmark is
graded against a closed-form posterior with an `exact_mc` floor of 0.017-0.036,
and the separations there are far outside it.

Convergence detail (`reference_convergence.py`): at n = 8 the reference's branch
mass is stable (0.221-0.250 over 40k-480k draws and two seeds). At n = 32 two
seeds at the same draw count give **0.574 and 0.329**. So a check at n = 8 says
nothing about n = 32, and an earlier version of this document drew exactly that
wrong inference.

## Inverse kinematics

Two different measurements, and they say different things.

**Rule isolated** (`ik_analytic_composition.py`): the per-observation factors are
ABC references built from the deterministic forward map, drawn from by CDF
inversion, and graded against the tall ABC reference built the same way. No
network. Median W1/sigma, 3 resampling replicates:

| n | exact product | mixture | mixture + bag | **tilt** | tilt + bag |
|---|---|---|---|---|---|
| 8 | 0.000 | 0.054 | 0.284 | 0.087 | 0.255 |
| 32 | 0.000 | **1.010** | 0.188 | **0.131** | 0.148 |

The `exact_product` row is 0.000 by construction and confirms the referee is the
product of the factors it is fed. At `n = 32` the tilt cuts the rule's own error
**7.7x**, and it does so without bagging -- which is what bagging had been
papering over.

**Through the trained network** (`run_mixture_ik.py`, 40 steps, 20,000 stage-1
draws, 2,000 composed draws):

| n | gauss_hier | gauss_jac | Langevin | mixture | **tilt** | tilt + bag |
|---|---|---|---|---|---|---|
| 8 | 0.518 / 7s | 0.494 / 7s | 0.442 / 5s | 0.063 / 19s | 0.063 / 16s | 0.148 |
| 16 | 0.566 / 9s | 0.475 / 10s | 0.488 / 10s | 0.185 / 34s | 0.171 / 30s | **0.086** |
| 32 | 0.677 / 9s | 0.529 / 14s | 0.575 / 15s | 0.232 / 55s | **0.161** / 46s | 0.234 |
| 64 | 0.662 / 10s | **0.516** / 15s | 0.713 / 24s | 0.936 / 115s | 0.978 / 103s | 0.774 |
| 100 | 0.729 / 10s | **0.578** / 12s | 0.825 / 19s | 1.285 / 62s | 1.039 / 52s | 0.874 |

(n = 100 at 1,000 composed draws rather than 2,000: the OOM at 2,000 is inside
`MultiObsSampler._raw_row_scores`, i.e. in the three *rules*, which push
`(n, draws) = 200k` rows through the transformer at once. The composed arms are
not what runs out of memory, but they are graded at the same draw count as the
baselines.)

Endpoint residual on the locals is unchanged (0.017-0.022 for both composed arms
against 0.078-0.087 for every rule), as it must be: stage 3 is untouched.

**At `n >= 64` inverse kinematics is not rule-limited, and the tilt does not
rescue it** -- both composed arms are worse there than `gauss_jacobian`. It is
not a resource problem either: 6,000 draws at 100 diffusion steps gives 0.998 at
`n = 64`, indistinguishable from 20,000 draws at 40 steps (0.978), and
`tilt + bagging` 0.759 against 0.774. In the rows where the rules win the shared
coordinate they are still 4-5x worse on the locals.

**At `n = 64`** Both composed arms
collapse onto a single over-narrow mode and disagree about *which* branch
(`10_tilt_ik.png`, bottom row: the tilt picks the negative branch, the mixture
the positive one, the reference the positive one). The branch decision is the
sum of `n` per-observation log-odds, so a small systematic per-observation bias
in the network moves it by `n * delta` and a better composition rule cannot
recover it. Improving `n >= 64` inverse kinematics means reducing
per-observation bias, not composing differently.

## Why it is not exact -- two different reasons, measured

The two benchmarks fail for structurally different reasons, and the distinction
matters because only one of them is curable by spending more compute.

### Gaussian mixtures: correctly specified, so what is left is Monte-Carlo noise

The true likelihood *is* log-quadratic here, so `d = 2` contains the truth and
BIC never asks for more. Against the closed form (`tilt_error_budget.py`):

| regime | n | draws | composed sd / exact | branch log-odds error | W1/sd |
|---|---|---|---|---|---|
| asymmetric | 4 | 20k | 1.0017 | 0.012 | 0.008 |
| asymmetric | 16 | 20k | 0.9993 | 0.080 | 0.035 |
| asymmetric | 100 | 20k | 0.9985 | 0.010 | 0.004 |
| antisymmetric | 4 | 20k | 0.9987 | -0.707 | 0.346 |
| antisymmetric | 16 | 20k | 0.9991 | -0.280 | 0.145 |
| antisymmetric | 100 | 20k | 1.0031 | +0.496 | 0.289 |
| antisymmetric | 4 | 200k | 0.9987 | -0.290 | 0.146 |
| antisymmetric | 16 | 200k | 0.9989 | -0.011 | 0.006 |
| antisymmetric | 100 | 200k | 1.0031 | +0.003 | 0.003 |

Two things to read off. The **width ratio is 1.000 everywhere**, and `W1/sd` is
consistently about half the branch log-odds error, so *the entire residual is
branch weight* -- the tilt places and scales both branches exactly. And **10x the
draws buys the expected `1/sqrt(S)`**: antisymmetric falls to 0.003-0.146. It is
noise, not a family limit.

Note it is the **antisymmetric** regime that is weak (0.09-0.35), not the
`symmetric` one (0.013-0.080, indistinguishable from asymmetric). Two conditions
have to coincide, and `tilt_identifiability.py` measures both at `n = 100`:

* **`beta` is weakly identified.** In the antisymmetric regime every observation
  sits *on* a prior mode, so its posterior is 99.999% on one branch and the other
  is unobserved. `dE[g]/dbeta` is **0.141** there against **0.963** in the
  asymmetric regime, so the same sampling noise in the first moment maps to ~7x
  the error in `beta` (0.019 vs 0.0028 per observation).
* **The composed answer is an exact cancellation.** `sum |beta_true| = 223`
  while `|sum beta_true| = 0` by construction, so `n` large individually-noisy
  tilts must cancel to precisely zero and accumulated noise has no signal to hide
  behind. Predicted log-odds error `0.33 * sqrt(100) * 0.145 = 0.48`; measured
  **0.496**.

`symmetric` has the cancellation but not the weak identification -- its
observations sit at zero, so `|beta|` is small and the sensitivity is high --
which is exactly why it does not suffer. It takes both.

Because moment matching **is** the MLE of this exponential family, it is
asymptotically efficient: the leftover antisymmetric error is the information
content of `n x S` draws in that configuration, not something to fix in the
estimator. The only lever is draws.

### Inverse kinematics: misspecified, so there is an irreducible floor

`tilt_misspecification.py` builds the per-observation ABC references and measures
the residual of the **best possible** degree-`d` polynomial fit to each exact
`log L_j(a)`, weighted by the tall reference. At `n = 32`, rms nats:

| | d=2 | 4 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| per factor (mean of 8) | 0.047 | 0.024 | 0.022 | **0.020** | 0.020 | 0.015 | 0.011 |
| x n = 32, composed | 1.51 | 0.78 | 0.71 | **0.65** | 0.63 | 0.49 | 0.35 |

The composed posterior's own log-density spans only **3.0 nats** across its
5%-of-peak support, so at the degree BIC selects the irreducible error is about a
fifth of the entire shape information. No draw count touches it. On the
Gaussian-mixtures benchmark this same quantity is **identically zero at d = 2**
-- that one number is the whole difference between the two benchmarks.

The curve is nearly flat from `d = 4` to `d = 10`, which is why widening the
candidate set to 16 left the measured `W1` unmoved: it was misspecification, not
under-parameterisation. One observation dominates the mean (0.107 -> 0.045, barely
improving with degree, against 0.022-0.055 -> 0.002-0.006 for the rest), so the
per-factor floor is set by the worst-behaved observation rather than the typical
one.

Figure: `13_why_not_exact.png`.

## What was tried and did not help

- **Bagging on top of the tilt.** On Gaussian mixtures it is strictly harmful
  (0.017 -> 0.490 at `n = 100`): bagging is a variance correction and the tilt is
  already near-unbiased, so averaging bootstrapped densities only inflates the
  width. On inverse kinematics it helps at `n = 16` (0.171 -> 0.086) and hurts at
  `n = 8` and 32, consistent with the residual error there being network noise
  rather than rule bias. **The rule is: remove the systematic term first, and
  then bagging usually has nothing left to do.**
- **Localised refit** (`tilt_fit.fit_all(log_weight=...)`, a consistent weighted
  MLE that attaches the first pass's composed density to both the base measure
  and the empirical distribution). Implemented and kept because it is cheap and
  principled, but on inverse kinematics it is a wash: 0.106/0.122/0.186 against
  0.063/0.171/0.161 at n = 8/16/32.
- **Local moment-deconvolution refinement** (`local_refine.py`: per-mode kernel-
  tilted moments with the kernel width deconvolved out). The premise -- that each
  factor is much wider than the composed posterior, so a narrow window localises
  -- is false here: the composed sd is only `2.84/sqrt(n)` times the factor sd, so
  a usable window is never narrow relative to a factor and the deconvolution is
  ill-conditioned. Kept in the tree as a negative result.
- **Raising the tilt's candidate degrees to 16.** BIC keeps choosing 8-10 on
  inverse kinematics and the numbers do not move.
- **More stage-1 draws at inverse-kinematics n = 64.** The hypothesis was that
  the branch-mass error there was the same Monte-Carlo mechanism as the
  Gaussian-mixtures antisymmetric regime. It is not: 20,000 draws gives 0.983 and
  80,000 gives **1.083** at 2.8x the cost. The n >= 64 error is systematic.
- **The time-embedding axis.** At h16d2 with all four widths trained under one
  protocol, n = 32 endpoint residual is 0.034-0.044 for te = 32/64/128/256
  against 0.0184 for the *same architecture* (h16d2, te = 256, 367,136
  parameters) trained longer under `_e10000`. **Training length dominates the
  embedding width entirely** at that trunk. On the h64d4wide trunk te = 32 and
  te = 256 are indistinguishable (n = 64: 0.153 vs 0.324, under a 0.31 floor,
  and the endpoint residual prefers te = 256). h64d4wide + te = 64 would be
  1,049,280 parameters and is untested rather than ruled out.
- **Correcting the composition for the fit's own uncertainty.** Two attempts,
  both dead. (a) *Analytic marginalisation*: `E[exp(theta . phi)] =
  exp(theta_hat . phi + 0.5 phi^T V phi)` is exact pointwise but **improper** --
  `phi` contains `u^d`, so the term grows like `u^(2d)` and the result does not
  integrate. Measured at the degrees BIC selects: 0.09-0.14 nats across the
  support and **1.1e7 nats** at the grid edge, which after exponentiation moves
  all mass to the tail and collapses the sd to zero. `compose()` now raises on
  it. (b) *The proper version* (`compose_uncertain`, averaging **normalised**
  densities over `theta ~ N(theta_hat, V)`, eigendecomposed square root so
  unidentified directions get no spread) is stable and correct -- and has **no
  effect**: sd ratio 1.000 at n = 32/64 and S = 6,000/20,000, MAP error unchanged
  at 0.065. The plug-in over-confidence is real but ~0.1 nats, negligible against
  the posterior's own log-density range. So the narrowness at n = 32 is *not*
  plug-in under-dispersion.
- **`legacy_h16d2_n200000_six_weighted_log_sigma` as a "more data" test.** Not a
  controlled comparison: the baseline carries `max_epochs_override: 10000` plus a
  resumed 3,284s, that checkpoint has neither. Its endpoint residual is 2.1x
  worse, so it is simply a less-optimised model and the row says nothing about
  data volume.

## Two free wins found while chasing the wall clock

**Stage 1 and stage 3 want opposite churn.** The composed arms had been
inheriting `churn_eta = 2` from the composition rules, and it had never been
tuned for them. At n = 32 with 6,000 stage-1 draws:

| churn eta | shared W1/sigma | endpoint r | seconds |
|---|---|---|---|
| **0** | **0.156** | 0.0246 | 23.6 |
| 2 (inherited) | 0.203 | 0.0188 | 24.6 |
| 4 | 0.277 | 0.0183 | 24.8 |

The shared coordinate gets monotonically *worse* with churn and the locals
monotonically *better*. Churn injects noise and re-denoises, which perturbs the
single-observation marginals the tilt reads its two moments from; stage 3 draws
locals conditioned on one shared value and benefits from the extra
stochasticity. A single value cannot express both, so `--stage1-churn` was split
out from `--churn-eta`. With `stage1_churn=0, churn_eta=2` and only 6,000 draws:
n = 32 gives 0.156 in 34.4s and n = 64 gives **0.773** where 20,000 draws at
eta = 2 gave 0.983 in 103s.

**Stage 1 chunked over observations.** It was one `sample` call over all
observations, so peak memory was `(n * draws)` rows of transformer activations
and n = 100 exceeded a 10.5 GB card regardless of what else was running. The
single-observation posteriors are independent given their own observation, so
chunking changes nothing computed (`--stage1-chunk`, default 16, each block
seeded from the run seed plus its index). This is what unblocked the n = 100
cells on both benchmarks.

## Draw budget

The tilt uses two moments, so it tolerates far fewer stage-1 draws than EM
(Gaussian mixtures, asymmetric, median W1/sd over 5 replicates):

| draws per observation | n=4 | 16 | 64 | 100 |
|---|---|---|---|---|
| 500 | 0.051 | 0.510 | 0.341 | 0.060 |
| 1,000 | 0.075 | 0.311 | 0.276 | 0.036 |
| 2,000 | 0.023 | 0.215 | 0.085 | 0.077 |
| 5,000 | 0.056 | 0.095 | 0.097 | 0.167 |
| 20,000 | 0.020 | 0.078 | 0.055 | 0.017 |

Roughly `1/sqrt(S)`, as the error analysis says. 5,000 draws is within a factor
of two of the floor at a quarter of the stage-1 cost, which is where the wall
clock actually is at large `n`.

On inverse kinematics, trading draws for diffusion steps at matched cost (6,000
draws at 100 steps versus 20,000 at 40) gives 0.072 / 0.100 / 0.162 at
n = 8/16/32 against 0.063 / 0.171 / 0.161 -- a clear win at `n = 16`, a wash
elsewhere.

## Network capacity: the one thing that moved inverse-kinematics n = 64

Composed arms only, identical settings (6,000 draws, split churn), so this is
controlled. Graded with the caveats above in force -- the endpoint residual is
the metric to trust here, and it agrees with `W1` on the trunk axis:

| n | checkpoint | params | mixture W1 | **tilt W1** | **endpoint r** |
|---|---|---|---|---|---|
| 32 | h16d2 te256 (`_e10000`) | 367k | 0.764 | 0.156 | 0.0184 |
| 32 | h64d4wide te32 | 697k | -- | 0.289 | 0.0157 |
| 32 | h64d4wide te256 | 3.16M | 0.290 | 0.696 | **0.0137** |
| 64 | h16d2 te256 (`_e10000`) | 367k | 0.793 | 0.773 | 0.0169 |
| 64 | h64d4wide te32 | 697k | -- | **0.153** | 0.0147 |
| 64 | h64d4wide te256 | 3.16M | 1.173 | 0.324 | **0.0136** |

**At n = 64 the wide trunk is the only thing that has ever moved this row**:
0.773 -> 0.153/0.324, a gap of 0.45-0.64 against a floor of 0.31, and the
endpoint residual agrees. Combined with the split-churn fix the row has gone
0.983 -> 0.773 -> 0.153. That confirms the earlier diagnosis -- n >= 64 inverse
kinematics is network-limited, and capacity is the lever.

Two details. `mixture_product` gets *worse* on the better trunk (0.793 -> 1.173)
while the tilt gets much better, consistent with its failure being the
`prior^(1-n)` spike, which no network can fix. And the n = 32 column runs
*backwards* against the reference-free endpoint residual across all three
networks -- which is the referee, not three successively worse posteriors.

## The metric changed: MAP distance to the true parameter

Adopted 2026-09-15 as the primary inverse-kinematics metric, for a concrete
reason: it contains **no reference at all**. The generating angle is an input to
`bimodality_study.selected_dataset`, so `| |argmax p(a)| - 0.6 |` is measurable
at any `n` without a referee whose own scatter is 0.424 (above). Both `+0.6`
and `-0.6` are the truth under the reflection, hence the absolute value.
`map_error.py` ranks every saved run by it.

**It vindicated itself immediately: at every n >= 16 the ABC reference is
further from the truth than the best method is.**

| n | reference's own error | best method | best error |
|---|---|---|---|
| 8 | 0.055 | tilt | **0.000** |
| 16 | 0.125 | tilt + bagging | 0.070 |
| 32 | 0.150 | tilt, h64d4wide/te32 | 0.065 |
| 64 | 0.120 | tilt, h64d4wide/te32 | **0.060** |
| 100 | 0.110 | tilt, h64d4wide/te32 | **0.060** |

`W1` and MAP-error disagree sharply -- at n = 64 the `net_wide` mixture arm has
MAP error 0.065 (third best) and `W1` **1.173** -- so they were measuring
different things all along. The Gaussian rules are decisively worse on this
metric and the gap is far outside any noise: at n = 100 `gauss_jacobian`'s MAP
sits at -0.295 (error **0.305**) and `gauss_hierarchical`'s at -0.135 (error
**0.465**), where their `W1` of 0.578 and 0.729 had looked competitive.

Grid spacing is 0.005, so errors quantise there and ties at n = 8 are ties.

## Recommended configuration, and it is now faster than Langevin

Exponential tilt on **h64d4wide/te32** (697k parameters), `--stage1-churn 0
--churn-eta 2`, `--single-draws 3000`, `--local-draws 200`, 40 steps.

The timing comparison had to be redone, because **Langevin's 12.4s at n = 32 was
measured on the small h16d2 trunk** while the tilt ran on h64d4wide, which costs
3.5-4.4x per network call (`gauss_hierarchical` 6.1s -> 20.7s at n = 32, 9.4s ->
41.3s at n = 64). On the same trunk:

| n | Langevin + F-NPSE | tilt S=6000 | tilt S=3000 | tilt S=1500 |
|---|---|---|---|---|
| 32 | 50.2s | 63.7s | **34.4s** | **20.3s** |
| 64 | 101.7s | 129.2s | **68.5s** | **39.5s** |

and the draw cut costs almost nothing on the reference-free metrics:

| n | S | seconds | \|MAP\|-0.6 | mass>0 | endpoint r | W1 |
|---|---|---|---|---|---|---|
| 32 | 6000 | 63.7 | 0.065 | 0.34 | 0.0157 | 0.289 |
| 32 | 3000 | **34.4** | 0.070 | 0.43 | 0.0155 | 0.451 |
| 32 | 1500 | **20.3** | 0.075 | 0.59 | 0.0153 | 0.775 |
| 64 | 6000 | 129.2 | 0.060 | 0.57 | 0.0147 | 0.153 |
| 64 | 3000 | **68.5** | 0.065 | 0.50 | 0.0149 | 0.292 |
| 64 | 1500 | **39.5** | 0.075 | 0.70 | 0.0145 | 0.336 |

Cutting draws 4x costs 0.005-0.015 of MAP error -- one to three grid cells --
and the endpoint residual *improves*. `W1` says the opposite (0.289 -> 0.775),
but `W1` at n = 32 is graded against a referee with 0.424 of self-disagreement,
so the two reference-free metrics are the ones to believe. **More draws being
worse is a repeated finding here** (20k -> 80k at n = 64 and 6k -> 20k at n = 32
both degraded): with a systematic per-observation network bias, extra draws
estimate the *bias* more precisely and the composition multiplies it by `n`.

Note the *cost attribution I got wrong first*: stage 1 is `n * S` rows and stage
3 is `n * draws`, so at 32 x 6000 against 32 x 1000 stage 1 is 6x the work.
`--local-draws 200` is a real but modest 15% (74.1s -> 63.4s, `W1` identical to
three decimals, which confirms the global posterior never touches stage 3).
**Stage-1 draws are the time lever**, not stage 3.

## Why the truth is not inside the posterior at n = 32, and why no method fixes it

`selection_tradeoff.py` sweeps the selection width, reporting the **reference's
own** numbers -- so this is a ceiling on any method, not a property of one:

| n | eps | mean abs x1 | reference MAP error | dip | p(0.6)/peak |
|---|---|---|---|---|---|
| 32 | 0.05 | 0.020 | 0.100 | **0.000** | 0.030 |
| 32 | **0.15** (used) | 0.090 | 0.155 | **0.000** | **0.004** |
| 32 | 0.30 | 0.160 | 0.185 | 1.000 | 0.006 |
| 32 | 0.60 | 0.314 | **0.045** | 1.000 | **0.493** |

At the selection actually used, **the density at the true angle is 0.4% of the
peak in the reference itself**. And there is **no eps at n = 32 that is both
bimodal and truth-covering**: every bimodal row has `p(0.6)/peak <= 0.03`, and
the only covering row (0.493) is unimodal. Bimodality here requires `|x1| -> 0`,
which requires atypical data, which pulls the posterior inward. The trade-off is
structural, and the other route to bimodality (widening the height prior) is
already marked in `bimodality_study` as the one that does not work.

Meanwhile the tilt's MAP error at n = 32 is **0.065 against the reference's
0.155** -- 2.4x closer to the truth than the best available estimate of the
posterior for that data. So the residual is the dataset, and further tuning
would optimise against a target further from the truth than we already are.

At n = 8 the picture differs and eps = 0.15 is well chosen: it gives the *best*
MAP error of the four (0.045) and the strongest bimodality (dip 0.004).
Tightening to eps = 0.05 makes MAP error **worse** (0.175), so "tighter
selection means more inward bias" is not monotone.

## What this method is, and is not

**It is not compositional score modelling.** That paradigm combines the
per-observation *scores* inside the reverse diffusion, at every noise level --
what `gauss_hierarchical`, `gauss_jacobian` and F-NPSE do. The tilt runs the
*ordinary* sampler once per observation, fits a parametric form to the resulting
**draws**, adds the parameters, and samples the shared coordinate by inverting a
CDF on a grid. There is no composed score and no modified SDE anywhere in it. It
belongs to the "sample each posterior, then combine" family that
`marginal_product` and `mixture_product` already occupied here. Comparison
tables that list it beside the rules are like-for-like on task, network and
reference but **not** on method family, and should say so.

It is a legitimate approximate-inference scheme with a countable ledger:
`p(g|x_1..n) ∝ p(g) prod_j L_j(g)` is an identity;
`p(g, l_1..n|x) = p(g|x_1..n) prod_j p(l_j|g,x_j)` is an identity; the *only*
approximation on the shared coordinate is `log L_j ≈ polynomial`, plus
Monte-Carlo error in `d` moments. But it needs three things the rules do not:
a **low-dimensional** shared block (1-D quadrature -- this is what blocks
`GLOBAL_DIM = 7`), the **prior in closed form** as base measure, and the network
to supply **`p(l_j|g,x_j)`**, which requires the right conditioning masks at
training time. So its wins read as "on a low-dimensional multimodal shared block
you need not compose scores", not "score composition is beaten".

**A version that would be genuine score composition exists for d = 2 only.** A
degree-2 tilt of a Gaussian-mixture prior *is* a Gaussian mixture, and mixtures
are closed under diffusion (`Sigma + sigma_t^2 I`), so the composed diffused
score is analytic at every noise level and could be handed to the sampler.
`diffused_mixture.py` walked part of this route: the shared coordinate worked and
the locals came out 7-17x worse. The restriction is the point -- a degree-6 tilt
is not closed under convolution, so exactly the flexibility that makes the tilt
win on inverse kinematics is what blocks converting it into a score composition.

## Why the Gaussian rules cannot reproduce a multimodal composed posterior

A counting argument, not a numerical one, and closed form on the
Gaussian-mixtures benchmark (`scratchpad/why_gauss_fails.py`, figure
`17_why_gaussian_rules_fail.png`). A GAUSS-type rule stores each observation's
contribution as a mean and a covariance -- two numbers in 1-D. A two-component
factor needs five (two means, two variances, one weight), and the **weight
direction has no image in a two-parameter family at all**. At n = 16:

* one factor is bimodal at -2.091/+2.289 with weights 0.044/0.956; its
  mean+covariance summary is `N(+2.098, 0.941)`, one hump;
* the exact composed posterior has modes -0.730/+0.801, weights 0.250/0.750;
* the rule, given **exact arithmetic on natural parameters** -- its best possible
  case, no sampler error at all -- returns `N(+0.460, 0.0487)`. The sd is right
  (0.221 against 0.221) and the shape is unreproducible.

`gauss_jacobian` uses the true local curvature and that genuinely helps where
non-Gaussianity is *smooth*: on the exponential twin 0.2707 against
`gauss_hierarchical`'s 0.6822, a 2.5x gain, width 1.136 against 1.403. But
curvature is a second-order *local* quantity and a mode weight is an *integral*,
so no Jacobian at any single state contains it. Measured directly: refreshing
every step instead of every 5th fixes the width exactly (0.420 -> 0.494 against a
reference 0.493) and halves the dip (0.669 -> 0.280), while **branch mass does
not move at all** -- 0.453/0.459/0.454 against a reference 0.243. The
implementation's `jacobian_curvature_floor` clamps precisely in the valley
between modes, where the true log-density is convex and the "precision" comes
out negative.

**So "non-Gaussian" splits in two.** Smooth departures (skew, tails, walls) are
handled well by curvature-based rules. Multimodality is not handled at all, by
any of them, at any solver setting.

## Transfer across benchmarks: stage-1 churn is the knob that does not travel

Applying the inverse-kinematics recipe wholesale to Gaussian mixtures degraded
it badly (natural n = 16: **3.06** against ~0.18 properly configured). I blamed
the draw budget, then the step count; a 2x2 over both exonerated **both** -- all
four cells agree to three decimals. The cause was `--stage1-churn 0` alone:

| stage-1 churn | asym n=4 | asym n=16 | natural n=4 | natural n=16 |
|---|---|---|---|---|
| 0 | 0.218/0.374 | 0.372/0.442 | 0.843/1.330 | **3.211/3.058** |
| 2 | 0.056/0.038 | 0.084/0.267 | 0.129/0.143 | **0.175/0.271** |
| 4 | 0.058/0.039 | 0.262/0.049 | 0.126/0.139 | 0.176/0.251 |

Its optimum is problem-specific and the spread is large: **0** on inverse
kinematics (0.203 -> 0.156), **2-4** on Gaussian mixtures (3.06 -> 0.18),
**2** on the Gaussian noise twin (0.173 -> 0.060), **0** on the Laplace and
exponential twins (exponential: 0.105 at churn 0 against 0.350 at churn 2, a
3.3x penalty for churning). Sweep it per problem; do not inherit it.

## Gaussian mixtures at the corrected configuration

Trained network, stage-1 churn 2, 3,000 stage-1 draws, 40 steps, locals capped at
200, medians over 3 replicates. Figures `18_gm_final_asymmetric.png` and
`18b_gm_final_natural.png`.

| n | tilt | mixture-hier. | Langevin+F-NPSE | gauss_hierarchical |
|---|---|---|---|---|
| 2 | **0.084** / 1s | 0.160 / 2s | 0.239 / 3s | 0.185 / 3s |
| 4 | **0.056** / 2s | 0.609 / 4s | 0.641 / 5s | 0.501 / 4s |
| 8 | **0.087** / 3s | 1.090 / 9s | 1.659 / 9s | 1.312 / 4s |
| 16 | **0.159** / 6s | 1.089 / 18s | 2.690 / 17s | 2.216 / 6s |
| 32 | **0.167** / 13s | 1.065 / 36s | 4.534 / 33s | 4.368 / 11s |
| 64 | **0.150** / 25s | 1.032 / 72s | 8.080 / 67s | 8.153 / 21s |
| 100 | **0.184** / 40s | 0.983 / 112s | 12.04 / 105s | 11.88 / 33s |

Flat at 0.06-0.18 over a 50x range in `n`, at 2.6-2.8x less wall clock than
Langevin or `mixture_product`. The figures separate the two rule failures that
`W1` compresses: Langevin and `gauss_hierarchical` drift *outward* with `n`
(±2.3 at n = 8 where the truth is ±1.2, off-panel by n = 16), while
`mixture_product` collapses onto a **spike at g ~ 0** in the prior's valley from
n = 8 on -- the `prior^(1-n)` cancellation, visible as a vertical line at zero.
The natural regime is weaker (0.11-0.46), so do not quote one figure of merit
across regimes.

**The tilt is also structurally cheaper than the rules**, which is
counterintuitive but mechanical: `gauss_hierarchical` needs a pilot covariance
from 4,096 draws per observation and `gauss_jacobian` a forward-mode JVP per
latent coordinate at every refresh step, where the tilt needs `d` sample moments
and one Newton solve on a 1-D grid, then adds `n` short vectors.

## The three noise twins: exact score, exact reference

`exponential_score_comparison/tilt_noise_comparison.py`. The problem modules give
the shared posterior in closed form and `symmetric_row_scores.AnalyticModel`
supplies the score exactly, so neither a Monte-Carlo referee nor network error is
in the loop. N = 30, 40 steps, 3,000 draws, best churn per twin.

| twin | tilt (churn) | exact-draw ceiling | gauss_jacobian |
|---|---|---|---|
| Gaussian | 0.0595 (2) / 23s | 0.0110 / 136s | 0.0571 / 7s |
| Laplace | 0.1367 (0) / 84s | 0.0979 / 130s | **0.0764** / 101s |
| Exponential | **0.1049** (0) / 64s | 0.2406 / 221s | 0.2707 / 67s |

**The tilt wins one, ties one, loses one, and which is predictable**: it pays
when the *posterior* is non-Gaussian (exponential: skew + hard wall, 2.6x over
the best rule), not when the *likelihood* merely is. The Laplace twin has a
kinked likelihood no polynomial represents and a near-Gaussian posterior at
N = 30 by CLT, so the curvature-based rule wins it 1.8x. On the Gaussian twin the
rules are exact by construction, so a tie is the ceiling -- and the exact-draw
arm's 0.0110 against the sampler arm's 0.0595 says **five sixths of the deployed
error there is stage-1 sampling, not composition**.

On the exponential twin the sampler arm beats its own exact-draw ceiling
(0.1049 against 0.2406). Not a contradiction -- the ceiling is limited by the
tilt's misspecification against a boundary, and churn-free sampling introduces a
shape error that partially cancels it. Treat as a cancellation, not a mechanism.

Capping degrees at 8 and the fit grid at 40,001 points made the tilt both
**faster and more accurate** here (23-84s against 131-254s; 0.0595 against the
0.0788 first measured at degree 12 on a 120,001-point grid). Degree-12 headroom
was actively harmful.

## Partial pooling: blocked, and the obstruction is measured

`GLOBAL_DIM = 7`, against a 1-D quadrature. `scratchpad/pp_diagonality.py` on the
saved posterior draws:

* **strongly non-Gaussian** -- excess kurtosis +37 to +253 (native_joint), +54 to
  +675 (ancestral). Exactly the regime where the tilt should win and a
  mean-plus-covariance summary must fail, so the motivation is real. (A kurtosis
  of 675 is more plausibly a few extreme draws than a smooth tail; check before
  designing around it.)
* **but not diagonal** -- max |off-diagonal correlation| **0.698**, on
  `mu_log_alpha` vs `log_sigma_log_alpha`: the classic hierarchical funnel. Mean
  |off-diagonal| 0.205.

So seven independent 1-D tilts would discard r = 0.70, and the degree-2
multivariate version reduces algebraically to `gauss_hierarchical`. This needs a
genuine multivariate extension (a low-rank higher-order term), not a
configuration change.

## Related

`docs/mixture_product.md` (the arm this replaces), `docs/nonsymmetric_composition.md`
(the exact factorisation and why score-space composition inherits the `(n-1)`
cancellation), `docs/score_calibration.md`.
