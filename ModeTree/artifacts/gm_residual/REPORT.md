# Improving the learned compositional first-order field

Experiments 1-5 of the improvement programme. Everything is graded against the
Gaussian mixture's exact diffused score and closed-form posterior, or against
inverse kinematics' ABC reference. Nothing below uses a score Jacobian, ModeTree
or a sampler.

**Scripts** `ModeTree/experiments/`: `exp1_gm_residual_error.py`,
`exp1b_gm_indistribution.py`, `gmresidual.py`, `residual_score.py`,
`exp2_gm_train.py`, `regrade_gm_arms.py`, `exp4_ik_transfer.py`,
`exp5_ik_marginalization.py`, plus figures and reports.
**Tests** `ModeTree/tests/test_gm_residual.py` (4), `test_ik_transfer.py` (3).
**Artifacts** `ModeTree/artifacts/{gm_residual,gm_arms,ik_transfer,ik_marginalization}/`.

## The ten questions

1. **Posterior score versus likelihood residual.** The residual is what
   composition consumes and it is 3-7x smaller than the posterior score, so the
   same absolute error is 3-7x worse on it. But cancellation is not the whole
   story: on its *own training distribution*, at the smallest noise scale it was
   ever trained at, the stock network's error on the residual is **twice the
   residual** (SNR 0.54, direction agreement 0.37).
2. **How the error scales with n.** `||e_n|| ~ n^1.00` with `r^2 = 1.000` at
   every noise level. Independent noise would give 0.5. The per-row errors add
   coherently: a systematic bias, not noise.
3. **Is prior pinning explained by the residual failure?** Yes. The stock
   network's composed modes sit at +-2.1 for every n from 1 to 128 while the
   exact posterior contracts from +-2.2 to +-0.1; the oracle-row arm, same rule
   and same states, tracks the reference to three decimals.
4. **Does direct residual prediction fix it?** Substantially. 3-6x lower
   composed error, and prior pinning is broken outright -- mode recall goes from
   0.00 to 1.00 at every n. Not sufficient on its own.
5. **Does compositional training add more?** Yes, ~2x further -- **but only at a
   weight that applies it.** Measured after the fact, the penalty at weight 1
   was 0.3% of the objective; at weight 200 it is worth 2.1x over the
   residual-only arm. My first pass under-weighted it by ~300x and the effect I
   initially attributed to it was mostly not it.
6. **Best noise/preconditioning scheme.** A log-`lambda` time coordinate
   rescaled onto the sinusoidal embedder's dynamic range, EDM input scaling, and
   a schedule floor lowered to the noise scales the composition is evaluated at.
   This is the largest single low-`lambda` lever, and it is the only change that
   makes the error flat in `lambda`.
7. **How much does the best variant improve things?** The composed **field**
   improves 10-31x, replicated across two seeds, with a stable ranking in every
   cell. The reconstructed **posterior** improves a lot but clears the Part IX
   gate reproducibly **only at n = 1**: its seed-to-seed variation is comparable
   to the effect of the interventions.
8. **Does the same intervention improve IK?** Yes. The transferable half
   (residual parameterization + preconditioning) converts the bimodal case from
   "topology not represented at all" -- the retrained stock control is unimodal
   with dip ratio 1.000 -- to two basins with a dip matching the ABC reference,
   and improves W1/sd by 4-6x. Ordinary unimodal inference is not damaged: it is
   3-4x better everywhere except the very highest noise level.
9. **How much IK error is local profiling?** In one cell, most of it: at n = 8,
   `lambda = 0.179`, exact local marginalization takes W1/sd from 0.777 to 0.101
   (unbiased estimator) or 0.274 (coupled estimator) -- a robust 3-8x. In the
   other nine cells it is worth nothing, or the two estimators disagree. So
   `B ~ A` over most of the regime, which by the task's own decision rule means
   **score training, not marginalization, is still the dominant IK bottleneck**.
10. **The dominant bottleneck now.** It depends on where you are, and the answer
    has changed:
    * **large n with `lambda >= 0.1`: the composition rule.** The best arm sits
      *at* the error floor a perfect network would reach under this rule
      (0.491 against 0.514 at n = 32).
    * **small `lambda`: still the score**, now 5-10x from its floor rather than
      100x.
    * **large `lambda` on IK: the local profiling**, worth 7.7x.
    * and cutting across all of them, **training-run variance**, which is as
      large as the interventions' effect on every posterior-level metric.

## What this says about the next step

The programme's premise was that a moment/covariance head and a better ModeTree
were the next things to build. The measurements do not support going there yet:

* a covariance head approximates the local marginalization measured in
  Experiment 5, and that was worth a large factor in **one cell out of ten** and
  nothing in the rest -- a far narrower case than assumed, and the low-`lambda`
  oracle is not yet reliable enough to see the regime that matters;
* at large n and `lambda >= 0.1` the score is no longer the binding constraint,
  so the *rule* is what needs work there, not the network;
* the compositional penalty that produced the largest gains **needs an oracle**
  and does not transfer: two oracle-free surrogates were measured and both were
  worse than doing nothing. Finding a simulation-only form of it is worth more
  than any of the above, because it is the difference between a 2x that works on
  one benchmark and one that works everywhere;
* and no arm selection should be made on a single training run again.

---

# Where the first-order composed field loses the data (Gaussian mixture)

Experiment 1 of the improvement programme: **no retraining**, everything graded
against the benchmark's exact diffused score and exact tall posterior.

Scripts: `ModeTree/experiments/exp1_gm_residual_error.py` (grid/ladder),
`exp1b_gm_indistribution.py` (state-distribution control),
`figures_gm_residual.py`. Artifacts in `ModeTree/artifacts/gm_residual/`.
Checkpoint under test: `tokenized_h32d3wide_n200000_six_weighted_log_sigma`
(268,545 parameters), the one every GM result in this project has used.

## 0. What was verified before measuring anything

* The composition rule was read out of the code, not assumed:
  `ComposedField(rule="uncorrected")` in `modetree/diffusion.py:257-260` is
  exactly `(1-n) s_prior,t + sum_i s_i,t`, with `s_prior` the **exact** diffused
  mixture prior score (`compass.priors.GaussianMixturePrior.score`), not a
  moment-matched Gaussian.
* Row scores are taken through `MultiObsSampler._raw_row_scores`, the sampler's
  own entry point, so learned and oracle arms differ only in what sits behind
  `SBIm.model`.
* Everything is evaluated on the slice `l_i(g) = a g + b x_i`, the exact
  conditional mode of `p(l_i | g, x_i)`. It is shared by both mixture branches
  (locals are conditionally independent of the component given `g`) and
  `Var(l_i | g, x_i) = q` does not depend on `g`, so the Occam factor is
  constant and the profile log posterior equals the marginal one up to a
  constant. Integrating the exact composed field along it therefore has to
  return the exact posterior -- and it does: the oracle-row control reaches
  **W1/sd = 0.0001 at n=1 and 0.006 at n=32** (lambda = 0.01). The pipeline adds
  no error of its own.

## 1. A precondition that reframes every low-noise number in this project

`Trainer.eps = 1e-3` is the smallest diffusion time ever drawn, so with
`sigma = 15` the network has **never seen a noise scale below lambda = 0.03167**.

`T_FINAL = 1e-3` in the ModeTree GM experiment sits exactly on that floor, and
the first-order topology experiment's `t = 1e-4` (lambda = 0.01) is a factor of
three *below* it. Rows at lambda < 0.03167 are labelled `extrapolated` in the
artifacts and are reported separately below; they are not evidence about
training, only about extrapolation.

## 2. Is the network asked to recover a small residual as a difference of two large terms? (Part VIII)

Yes, by a factor of about 3 on the grid and about 7 at the states a sampler
actually visits. Posterior-weighted medians over nine observations:

| lambda | \|s_post\|/\|ds\| | \|s_prior\|/\|ds\| |
|---|---|---|
| 2 | 1.9 | 1.1 |
| 0.5 | 17.4 | 17.6 |
| 0.2 | 4.2 | 4.0 |
| 0.1 | 3.3 | 3.1 |
| 0.05 | 3.1 | 2.8 |
| 0.0317 (floor) | 3.0 | 2.7 |

At states drawn from the exact tall posterior (n=16) the ratio `|s|/|ds|` is
**10.3 / 1.50 = 6.9**. So a 1% error on the trained target is a ~7% error on the
target composition consumes. That is real amplification, but it is *not* the
whole story, because the measured error on the trained target is nowhere near
1%.

## 3. How accurate is the posterior score versus the likelihood residual? (Question 1)

`exp1b` draws states from two exact closed forms and compares against the
oracle at the same states, with no grid and no slice: `training` is the law the
denoising objective actually scored, `posterior` is the law a correct sampler
visits. n=16, 8000 draws per cell.

| lambda | regime | \|s*\| | \|ds*\| | error | SNR(s) | SNR(ds) | cos(ds) |
|---|---|---|---|---|---|---|---|
| 2 | training | 0.49 | 0.33 | 0.04 | 11.6 | 7.8 | 0.99 |
| 0.5 | training | 1.62 | 0.45 | 0.28 | 5.7 | 1.6 | 0.77 |
| 0.1 | training | 2.78 | 1.31 | 1.04 | 2.7 | 1.3 | 0.61 |
| **0.0317** | training | 2.89 | 1.44 | 2.67 | 1.1 | **0.54** | 0.37 |
| 0.5 | posterior | 3.33 | 1.95 | 1.75 | 1.9 | 1.1 | 0.71 |
| 0.1 | posterior | 9.47 | 1.48 | 5.48 | 1.7 | **0.27** | 0.60 |
| **0.0317** | posterior | 10.27 | 1.50 | 6.50 | 1.6 | **0.23** | 0.26 |
| 0.01 *(extrap.)* | posterior | 10.28 | 1.51 | 32.0 | 0.3 | 0.05 | -0.15 |

Two readings, and the second is the important one:

* **Cancellation is real.** SNR on the likelihood residual is 3-6x worse than on
  the posterior score at every noise level.
* **Cancellation is not sufficient.** Even on its own training distribution, at
  the smallest noise scale it was ever trained at, the error on the residual is
  *twice the residual* (SNR 0.54) and the direction agreement is 0.37. This is
  not a states-off-distribution problem and not only a difference-of-large-
  numbers problem: the network is simply inaccurate at small lambda, and the
  residual parameterization has to be tested rather than assumed.

## 4. How does the error scale with n? (Question 2)

Fitting `||e_n|| ~ n^alpha` over n = 1,2,4,8,16,32,64,128:

| lambda | alpha (network error) | r^2 | alpha (composed signal) |
|---|---|---|---|
| 0.5 | 0.997 | 1.000 | 0.676 |
| 0.2 | 1.001 | 1.000 | 0.684 |
| 0.1 | 1.002 | 1.000 | 0.706 |
| 0.05 | 1.000 | 0.999 | 0.713 |

**alpha = 1.00 with r^2 = 1.000 at every noise level.** Independent per-row
noise would give 0.5. The per-row errors add coherently: the dominant error is a
systematic per-factor bias, exactly the hypothesis's second clause. Because the
signal only grows as `n^0.70`, the *relative* composed error grows as `n^0.30`,
from 0.29 at n=1 to 1.17 at n=128 (lambda = 0.05).

## 5. Is GM prior-pinning explained by failure to learn the residual? (Question 3)

Yes, and the failure mode is visible directly in the reconstructed posterior.
Mode locations at lambda = 0.05:

| n | exact modes | learned modes | exact sd |
|---|---|---|---|
| 1 | -2.155, +2.225 | -2.17, +2.32 | 1.93 |
| 8 | -1.137, +1.207 | -2.12, +2.26 | 1.05 |
| 32 | -0.417, +0.487 | -2.09, +2.25 | 0.43 |
| 128 | -0.096, +0.166 | -2.09, +2.24 | 0.15 |

The learned modes do not move at all across the ladder: they sit next to the
prior modes at +-2.5 while the exact posterior contracts by a factor of 13. The
oracle-row arm tracks the reference to three decimals at every n. Since both
arms use the same rule, the same prior score and the same states, the entire
contraction failure is the learned `Delta s`.

The reconstructed posterior is also far too narrow (width ratio 0.10-0.28 at
lambda = 0.05) and collapses onto one branch (recovered `w_+` = 0.00 against an
exact 0.75).

## 6. Rule error versus network error

Both are available in closed form here, so they are separated rather than
argued about. `rule error` is the gap between the exact composition `S_n^*` and
the true tall diffused score.

| lambda | n=8 net / rule | n=128 net / rule | W1/sd, oracle rows, n=128 |
|---|---|---|---|
| 0.5 | 7.3 / 3.2 | 116 / 102 | 0.94 |
| 0.1 | 38.9 / 2.5 | 620 / 242 | 0.81 |
| 0.05 | 29.9 / 0.8 | 453 / 147 | 0.39 |
| 0.02 *(extrap.)* | 175 / 0.14 | 2758 / 37 | 0.074 |

The first-order rule is *asymptotically exact as lambda falls* -- at lambda =
0.02 a perfect network would give W1/sd = 0.07 at n=128 -- and the network is
worst exactly there. The rule is not the binding constraint; the score is.

## 7. What this fixes about the plan

* The composition-relevant band is **lambda in [0.03, 0.2]**: below it the
  network is extrapolating, above it the rule's own error takes over.
* The target for intervention is the **coherent** component of the per-row
  error. A loss that penalizes mean squared per-row error cannot distinguish a
  shared bias from an independent wobble of the same size; the measured
  `alpha = 1.00` says the shared bias is what matters.
* Prior cancellation contributes a ~7x amplification and justifies testing the
  residual parameterization, but it cannot by itself explain an error that
  exceeds the signal on the training distribution.

---

# Experiments 2-3: retrained arms

Scripts: `ModeTree/experiments/gmresidual.py` (the three interventions),
`exp2_gm_train.py` (training + grading), `report_gm_arms.py`.
Artifacts in `ModeTree/artifacts/gm_arms/`. Correctness of the machinery is
pinned by `ModeTree/tests/test_gm_residual.py` (4 tests), including that the
compositional penalty is exactly zero when the oracle is substituted for the
network.

All arms share the backbone (`h32d3wide`, 268,545 parameters), the 200,000
simulations, the `six_weighted` mask menu, `log_sigma` time sampling, the
optimizer, the learning rate and the epoch/patience budget. Each differs from
the one before it by a single change.

| arm | change |
|---|---|
| A | the stock recipe, retrained here as a harness control |
| B | `s_phi = s_base + backbone/sigma` (residual parameterization) |
| C | B + the compositional penalty, oracle target |
| D | C + derived preconditioning + schedule floor lowered to lambda = 0.01 |
| Csim | C with the oracle-free surrogate target |

**Harness control.** Arm A is not the frozen checkpoint: retrained here it is
*better* (W1/sd ~1.08 against 1.9-4.2 at lambda = 0.05). Run-to-run variation on
this benchmark is therefore large, and no cross-arm claim below rests on a
single cell. The qualitative failure is unchanged in arm A: weight TV -> 0.75,
width ratio -> 0.03, modes pinned near the prior.

## Question 4/5/6: relative composed score error, `|S_phi - S^*| / |S^*|`

| lambda | n | A | Csim | B | C | D |
|---|---|---|---|---|---|---|
| 0.2 | 8 | 0.823 | 0.268 | 0.115 | 0.072 | 0.070 |
| 0.2 | 128 | 1.140 | 0.397 | 0.186 | 0.139 | **0.117** |
| 0.1 | 128 | 1.452 | 0.410 | 0.240 | 0.195 | **0.139** |
| 0.05 | 128 | 0.924 | 0.616 | 0.418 | 0.337 | **0.126** |
| 0.0317 | 128 | 1.771 | 0.907 | 0.638 | 0.513 | **0.134** |
| 0.02 *(was extrapolation)* | 128 | 4.522 | 1.394 | 0.997 | 0.799 | **0.114** |

The ordering `A > Csim > B > C > D` holds in **all 30** (lambda, n) cells
measured, which is what makes the smaller gaps credible against the run-to-run
variance above.

* **Direct residual prediction works (Question 4).** B cuts the composed error
  by 4-6x at every n and noise scale. It also breaks prior pinning outright: at
  lambda = 0.05 mode recall goes from 0.00 to **1.00 at every n**, weight TV
  from 0.75 to 0.012-0.40, width ratio from 0.035 to 0.445, and the modes track
  the contraction instead of sitting at +-2.2.
* **Compositional training adds a further, smaller gain (Question 5).** C
  improves on B by a consistent 15-30%. It is not a second 5x. It also *costs*
  branch-weight accuracy at small n (TV 0.33/0.69 against B's 0.014/0.027), so
  it is an improvement in the field and a mixed result in the posterior.
* **Preconditioning is the largest single remaining lever (Question 6).** D is
  the only arm whose error does not grow as lambda falls: 0.114-0.139 at n = 128
  across the whole band, against A's 1.14 -> 4.52. At the old trained floor it
  is **13x** better than A, and it is accurate *below* that floor, which is
  where ModeTree has been evaluating all along.
* **The oracle-free surrogate does not transfer (negative result).** Csim
  replaces `lambda s^*` with the unbiased `-eps`. It is consistently worse than
  B, i.e. the penalty *hurts* when its target is estimated this way: the shared
  noise term has unit variance while the signal it must resolve is `lambda *
  Delta s`, so the gradient signal-to-noise is poor. The compositional penalty
  as written needs a target the simulator cannot supply, which is a direct
  obstacle on the path to inverse kinematics.

## Question 7: does any arm clear the rescue gate?

**No.** The Part IX gate (W1/sd < 0.1 for n <= 32, < 0.15 for n = 64,128, full
mode recall, mode error < 0.1 sd, weight TV < 0.1, width ratio in [0.9, 1.1]) is
not met by any arm at any n > 8.

The two candidates fail in *different* ways, which is informative:

* **B** reconstructs a well-calibrated posterior over a useful range (at
  lambda = 0.05: W1/sd 0.034 at n=1, 0.069 at n=4, 0.132 at n=8, with TV < 0.03
  and width 0.87-1.0) and then degrades, reaching W1/sd 0.71 and width 0.445 at
  n = 128.
* **D** has by far the most accurate field but its reconstructed posterior at
  large n is *worse*: the mass collapses onto the dominant branch (weight TV
  pinned at exactly 0.250) and the width is 0.25-0.6.

The reason a 12% field error still fails at n = 128 is arithmetic, not mystery:
the exact modes there are 0.26 apart with a within-branch sd of 0.036, so
locating a root to 0.1 needs the composed field accurate to roughly 1% -- an
order of magnitude beyond where the best arm sits.

## Splitting arm D's two changes

`Deps` keeps the stock schedule floor and adds only the preconditioning;
`Dpre` keeps the stock conditioning and only lowers the floor. Relative
composed error at n = 128:

| lambda | C (neither) | Deps (precond. only) | D (both) |
|---|---|---|---|
| 0.1 | 0.195 | 0.181 | 0.139 |
| 0.05 | 0.337 | 0.262 | 0.126 |
| 0.0317 | 0.513 | 0.291 | 0.134 |
| 0.02 | 0.799 | 0.999 | 0.114 |

Preconditioning alone buys ~1.8x at the old floor; lowering the floor on top of
it buys a further ~2.2x. The last row is the clean separation: at lambda = 0.02
`Deps` is *below its own trained floor* and collapses to 0.999, while `D`,
trained down to 0.01, stays at 0.114. So the two changes do different jobs --
preconditioning improves accuracy **within** the trained band, the lower floor
extends the band -- and both are needed.

## A correction to how arm C was run

The task asks for the compositional weight to be set only after measuring the
natural scale of each term. That measurement was made *after* the arms had been
trained, and it says the penalty at `comp_weight = 1` is worth **0.3%** of the
DSM loss (`L_comp` 0.006 against a DSM per-example 2.43, on both arms B and C).

Arm C therefore barely applied the penalty at all. Its consistent 15-30%
advantage over B across all 30 cells is either a small real effect or
run-to-run variation, and the numbers above cannot separate the two. A properly
weighted arm (`Cbig`, weight 200) and seed replicates of B and D are running to
settle it; **no claim about the compositional loss should be quoted from arm C
until they land.**

## The result that changes the plan

Better field accuracy did **not** buy a better posterior. At lambda = 0.05:

| arm | rel. field error (n=128) | W1/sd n=1 | n=4 | n=8 | n=32 | n=128 |
|---|---|---|---|---|---|---|
| A | 0.924 | 0.428 | 1.084 | 1.115 | 1.083 | 1.029 |
| **B** | 0.418 | **0.034** | **0.069** | **0.131** | 0.665 | 0.711 |
| C | 0.337 | 0.755 | 1.633 | 1.701 | 0.720 | 0.434 |
| Deps | 0.262 | 0.287 | 1.442 | 1.677 | 1.790 | 2.470 |
| D | **0.126** | 0.331 | 0.616 | 0.733 | 1.348 | 3.450 |

B has the *worst* field of the four interventions and, at seed 7, by far the
best posterior.

**That posterior ranking does not survive a second seed, and the claim it
supported has to be withdrawn.** Retraining arm B unchanged at seed 13:

| n | field error s7 / s13 | W1/sd s7 / s13 | weight TV s7 / s13 |
|---|---|---|---|
| 1 | 0.086 / 0.094 | 0.034 / 0.070 | 0.014 / 0.028 |
| 4 | 0.206 / 0.235 | 0.069 / **0.520** | 0.027 / **0.225** |
| 8 | 0.280 / 0.319 | 0.131 / **0.944** | 0.012 / **0.413** |
| 32 | 0.394 / 0.461 | 0.665 / 1.396 | 0.183 / 0.608 |
| 128 | 0.418 / 0.504 | 0.711 / 1.397 | 0.397 / 0.663 |

The field error is reproducible to 10-20%. The posterior metrics move by 3-7x.
So:

* **the field-error ranking `A > Csim > B > C > D` is reliable** -- it is
  consistent across 30 cells and stable across seeds;
* **the posterior ranking between the interventions is not** -- single-seed
  W1, weight TV and width on this benchmark are dominated by training noise,
  and B's apparent pass of the Part IX gate at n = 1 and n = 4 was a property
  of seed 7, not of the recipe. Only n = 1 passes at both seeds.

The mechanism is not mysterious: the branch mass is set by the fine balance of
the composed score between two widely separated modes, so a small systematic
bias tips it wholesale, and W1 is then dominated by a quantity that a 10%
change in the field can flip.

**Consequently no arm clears the Part IX gate reproducibly at any n > 1**, and
the honest summary of Experiments 2-3 is: the interventions deliver a large and
repeatable improvement in the *accuracy of the composed first-order field*
(up to 13x), which has not yet converted into a reproducible improvement in the
*reconstructed posterior* beyond very small n.

This also re-frames what to measure next. RMS field error and the reconstructed
posterior disagree about the ranking of interventions; selecting on the former
is what the improvement programme has been implicitly doing, and it is not
sufficient.

## The missing cell, and what it exposes (`Bpre`)

`Bpre` is the residual parameterization with D's preconditioning and schedule
but **without** the compositional penalty -- the one combination the first pass
did not contain. It has the best-behaved posterior of any arm, and degrades
smoothly rather than collapsing.

More importantly, grading it against the *rule's own error floor* -- the W1 an
arm would get with exact oracle rows, measured in Experiment 1 -- shows where
the remaining error actually lives. `W1/sd`, oracle-row floor in the first
column:

| lambda = 0.1, n | RULE floor | A | B | **Bpre** | C | D |
|---|---|---|---|---|---|---|
| 4 | 0.025 | 1.100 | 0.568 | **0.228** | 1.137 | 0.575 |
| 8 | 0.086 | 1.111 | 0.829 | **0.248** | 1.290 | 0.666 |
| 16 | 0.239 | 1.099 | 0.946 | **0.310** | 0.701 | 0.752 |
| 32 | 0.514 | 1.077 | 0.946 | **0.491** | 0.714 | 0.901 |
| 128 | 0.814 | 1.013 | 0.937 | **0.775** | 0.836 | 1.566 |

At n = 32 and n = 128, `Bpre` sits **at or below the floor a perfect network
would reach under this rule** (0.491 against 0.514; 0.775 against 0.814). The
same holds at lambda = 0.2 (1.003 against 1.003 at n = 32). At those cells the
score network has stopped being the bottleneck: what remains is the first-order
composition's own error.

That splits the problem in two, quantitatively:

* **large n with lambda >= 0.1: rule-limited.** The best arm is already at the
  oracle floor. No amount of further score training helps here; only a better
  composition rule does.
* **small lambda, or small n: score-limited.** The rule floor there is tiny
  (0.002-0.14 at lambda = 0.05) and the best arm is 5-10x above it.

And it explains why the gate is still out of reach at large n. Meeting it needs
a *small rule floor*, which requires small lambda, **and** an accurate network
at small lambda. Before this experiment neither was available; now the second
exists in `D`/`Bpre` but is still roughly 5x short at lambda = 0.05, n = 32
(0.655 against a floor of 0.140).

## The compositional loss, tested properly (`Cbig`)

Re-run with `comp_weight = 200`, which the scale measurement says makes the two
terms comparable rather than leaving the penalty at 0.3% of the objective.
This is the arm that actually tests Part III, and it changes the answer to
Question 5.

Relative composed field error at lambda = 0.1 (rule floor for reference):

| n | rule floor (W1) | A | B | C (weight 1) | **Cbig (weight 200)** | D |
|---|---|---|---|---|---|---|
| 1 | 0.010 | 0.317 | 0.046 | 0.028 | **0.020** | 0.021 |
| 8 | 0.086 | 1.062 | 0.157 | 0.102 | **0.062** | 0.086 |
| 32 | 0.514 | 1.360 | 0.226 | 0.179 | **0.089** | 0.127 |
| 128 | 0.814 | 1.452 | 0.240 | 0.195 | **0.094** | 0.139 |

and the posterior it reconstructs at lambda = 0.1:

| n | W1/sd | mode recall | weight TV | width ratio | gate |
|---|---|---|---|---|---|
| 1 | 0.038 | 1.00 | 0.015 | 1.019 | **pass** |
| 4 | 0.037 | 1.00 | 0.018 | 0.973 | **pass** |
| 8 | 0.183 | 1.00 | 0.093 | 0.845 | near (W1, width) |
| 16 | 0.432 | 1.00 | 0.212 | 0.522 | no |
| 128 | 1.615 | 0.50 | 0.250 | 0.471 | no |

So the compositional penalty **is** worth a large factor -- 2.1x over `C` and
2.6x over `B` at n = 128 -- but only at a weight that applies it. My first pass
under-weighted it by roughly 300x, and the 15-30% effect reported for arm C
above was mostly not the penalty at all. Arm `Cbig` is the best arm measured:
it clears the full Part IX gate at n = 1 and n = 4, and at n = 8 it is within
2.1x of the floor a perfect network would reach under this rule (0.183 against
0.086).

It costs training time (42.6 minutes against ~18) and it does not help at
n = 128, where its W1 is worse than `B`'s despite a 2.6x better field -- the
same field/posterior divergence seen throughout.

`Dbig` (this penalty *plus* the preconditioning and extended schedule) and seed
replicates of `Cbig` are running; the single-seed caveat established above
applies to every number in this section until they land.

## Combining everything (`Dbig`), and the gate result that survives replication

`Dbig` = residual parameterization + compositional penalty at weight 200 +
preconditioning + extended schedule. It is the most accurate composed field
measured anywhere in this project:

| lambda | n | A | Bpre | Cbig | **Dbig** | Dbig vs A |
|---|---|---|---|---|---|---|
| 0.1 | 8 | 1.062 | 0.165 | 0.062 | **0.044** | 24x |
| 0.1 | 128 | 1.452 | 0.240 | 0.094 | **0.064** | 23x |
| 0.05 | 128 | 0.924 | 0.308 | 0.181 | **0.092** | 10x |
| 0.02 | 128 | 4.522 | 0.417 | 0.442 | **0.190** | 24x |

Its branch weights are also the best seen (weight TV 0.009-0.019 for n <= 32 at
lambda = 0.1, against A's 0.75).

**An honest qualification: the n-scaling exponent is unchanged.** Arm A's
relative error grows 4.6x from n = 1 to 128 and `Dbig`'s grows 4.9x, so both
still have an absolute error going as roughly `n^1.0`. The interventions bought
a 10-24x smaller *coefficient*, not a better *rate*. The coherent per-row bias
that Experiment 1 identified has been shrunk, not removed.

### Which cells actually clear the gate, at both seeds

Every arm was graded at seed 7 and seed 13. Listing every cell that clears the
full Part IX gate at *either* seed, and whether it clears at both:

| arm | lambda | n | seed 7 | seed 13 | W1/sd |
|---|---|---|---|---|---|
| B | 0.1 | 1 | PASS | PASS | 0.090 / 0.059 |
| B | 0.05 | 1 | PASS | PASS | 0.034 / 0.070 |
| Bpre | 0.2 | 1 | PASS | PASS | 0.027 / 0.096 |
| D | 0.2 | 1 | PASS | PASS | 0.092 / 0.057 |
| B | 0.1 | 2 | PASS | fail | 0.060 / 0.265 |
| Cbig | 0.1 | 4 | PASS | fail | 0.037 / 0.343 |
| Cbig | 0.1 | 1 | PASS | fail | 0.038 / 0.233 |
| Cbig | 0.2 | 2 | fail | PASS | 0.045 / 0.065 |
| *(five further single-seed Cbig passes)* | | | | | |

**The gate is met reproducibly only at n = 1.** No cell at n >= 2 passes at both
seeds. Every apparent multi-observation pass -- including `Cbig`'s at n = 4,
which I reported above before the replicate existed -- is a single-seed result
that the second seed does not confirm.

So the answer to Question 7 has two halves that must be stated together:

* the **composed first-order field** improves by a large, replicated factor
  (10-24x) and the ranking of interventions is stable across seeds;
* the **reconstructed posterior** improves substantially but does not reach the
  development gate reproducibly beyond a single observation, because its
  seed-to-seed variation is comparable to the effect of the interventions
  themselves.

That second point is itself a finding, and it was not on the list of things to
measure. Training-run variance on this benchmark is a first-order obstacle: any
future arm selected on a single training run's W1 will be selected on noise.
Whatever comes next needs replicate training as part of its protocol, and it
makes the cheap controls in Part IV.8 -- weight averaging in particular --
better motivated than they looked, since they attack variance directly.

## Replicated headline, and what can transfer

Mean of both seeds, relative composed field error at n = 128, with the factor
against the stock recipe in brackets:

| lambda | A | B | Bpre | D | Cbig | **Dbig** |
|---|---|---|---|---|---|---|
| 0.2 | 1.330 | 0.206 (6.4x) | 0.237 (5.6x) | 0.169 (7.9x) | 0.070 (19x) | **0.043 (31x)** |
| 0.1 | 1.373 | 0.273 (5.0x) | 0.290 (4.7x) | 0.223 (6.1x) | 0.139 (9.9x) | **0.083 (17x)** |
| 0.05 | 1.344 | 0.461 (2.9x) | 0.328 (4.1x) | 0.209 (6.4x) | 0.262 (5.1x) | **0.139 (9.7x)** |
| 0.0317 | 3.035 | 0.698 (4.3x) | 0.432 (7.0x) | 0.268 (11x) | 0.405 (7.5x) | **0.185 (16x)** |
| 0.02 | 6.661 | 1.085 (6.1x) | 0.477 (14x) | 0.276 (24x) | 0.635 (11x) | **0.265 (25x)** |

The ordering holds in every cell at both seeds.

### The compositional penalty does not transfer (Question 8's precondition)

`Cbig` and `Dbig` are the best arms, and both need the *exact composition* as a
training target. Inverse kinematics cannot supply one. Two attempts at an
oracle-free surrogate, replacing `lambda s^*_i` by its unbiased denoising
estimate `-eps_i`:

| arm | weight | field error, lambda=0.1, n=128 |
|---|---|---|
| B (no penalty) | - | 0.240 |
| Csim | 1 | 0.410 |
| **Csimbig** | 20 | **1.286** |
| A (stock) | - | 1.452 |
| Cbig (oracle target) | 200 | 0.094 |

The surrogate is worse than doing nothing at weight 1 and nearly as bad as the
stock recipe at weight 20. This is the behaviour its algebra predicts: the
surrogate's loss equals the oracle one plus an irreducible O(1) noise term that
does not depend on the parameters, so raising its weight amplifies gradient
*variance* and no signal. Larger weight therefore cannot rescue it, and the
2-3x that the penalty is worth on GM is, as implemented, an oracle-only gain.

**What transfers to inverse kinematics is therefore the residual
parameterization and the preconditioning, and not the compositional penalty.**
That is worth 4-14x on GM on its own (`Bpre`), which is the honest expectation
to carry into Experiment 4.

---

# Experiment 4: transfer to inverse kinematics (Question 8)

Scripts: `ModeTree/experiments/exp4_ik_transfer.py` (training),
`experiment_first_order_topology.py` (grading, unchanged except that the
checkpoint is now selectable), `ModeTree/tests/test_ik_transfer.py` (3 tests).

Only what *can* transfer was transferred: the residual parameterization and the
preconditioning. The compositional penalty was left behind for the reason
measured above -- its oracle-free surrogate is worse than doing nothing.

The IK prior is an independent zero-mean Gaussian on the four angles
(`hierarchical.PRIOR_STD`), so the diffused marginal score of any latent subset
of them is exactly `-y / (std^2 + lambda^2)`: closed form, no oracle. The two
observables are a nonlinear function of the angles and have no closed-form
marginal, so the baseline is zero there and the network predicts those
conditionals in full -- the intervention applies exactly to the conditionals the
composition consumes.

Arms, matched on backbone (`legacy/h16d2`, 367,136 parameters), 200,000
simulations, the `six_weighted` menu, optimizer and budget:

* `Ka` -- the stock recipe, retrained in this harness (57.1 minutes)
* `Kbpre` -- residual parameterization + preconditioning + schedule floor
  lowered to lambda = 0.01 (57.4 minutes)

Graded Jacobian-free against the existing ABC references, on the existing
bimodal and stock datasets. No ModeTree, no JAC, no F-NPSE.

## Bimodal regime -- the structure the benchmark exists to test

| n | t | lambda | Ka: W1/sd, modes, dip | Kbpre: W1/sd, modes, dip | ref dip |
|---|---|---|---|---|---|
| 8 | 0.03 | 0.179 | 1.091, 1, 1.000 | **0.925**, 3, 0.036 | 0.069 |
| 8 | 0.01 | 0.101 | 1.156, 1, 1.000 | **1.018**, 2, 0.084 | 0.069 |
| 16 | 0.03 | 0.179 | 1.150, 1, 1.000 | **0.284**, 2, 0.001 | 0.003 |
| 16 | 0.01 | 0.101 | 1.212, 1, 1.000 | **1.015**, 2, 0.005 | 0.003 |
| 32 | 0.03 | 0.179 | 1.546, 5, 0.840 | **0.247**, 2, 0.000 | 0.000 |
| 32 | 0.01 | 0.101 | 1.598, 2, 0.096 | **1.348**, 2, 0.000 | 0.000 |

The qualitative result is the important one: **`Ka` is unimodal at almost every
cell** -- one mode, dip ratio 1.000, against an ABC reference whose dip is 0.069
or smaller -- while **`Kbpre` finds two basins with a dip ratio that matches the
reference**: 0.001 against 0.003 at n = 16, 0.000 against 0.000 at n = 32. The
stock recipe does not represent the bimodality at all at these noise levels;
the retrained one does.

Quantitatively, at the noise level where both are best behaved (t = 0.03):
W1/sd falls 4.1x at n = 16 (1.150 -> 0.284) and 6.3x at n = 32 (1.546 -> 0.247).

## Stock regime -- the check that ordinary inference was not sacrificed

| n | t | Ka W1/sd | Kbpre W1/sd |
|---|---|---|---|
| 8 | 0.03 | **0.109** | 0.292 |
| 8 | 0.003 | 1.247 | **0.359** |
| 8 | 0.001 | 1.527 | **0.369** |
| 32 | 0.03 | **0.185** | 0.450 |
| 32 | 0.003 | 2.332 | **0.727** |
| 32 | 0.001 | 2.311 | **0.760** |

Both stay unimodal, as the reference does (dip ~1.0), so no spurious
multimodality was introduced. `Ka` is better at the *highest* noise level and
`Kbpre` is 3-4x better everywhere below it. That is the same signature as on
the Gaussian mixture: the intervention buys stability in lambda, trading a
little accuracy at large lambda for a lot at small lambda -- which is the half
of the schedule the composition actually runs in.

**Answer to Question 8: yes.** The transferable half of the intervention
reproduces on inverse kinematics the effect it had on the Gaussian mixture, and
it converts the bimodal case from "topology not represented" to "topology
represented with roughly the right valley depth".

Caveats, stated rather than buried: one training seed per IK arm (and the GM
work showed seed variance matters); and at t = 0.001 `Kbpre` reports 5-8 roots,
the local-profiling instability already documented for this diagnostic, not a
property of the score.

---

# Experiment 5 / Part VII: profile error versus marginalization error (Question 9)

Script: `ModeTree/experiments/exp5_ik_marginalization.py`. Artifacts in
`ModeTree/artifacts/ik_marginalization/`.

The first-order diagnostic composes the global score at *profiled* locals,
`l*(g)`, which is not the marginal global score. The exact identity is

    d/dg log p(g | x) = E_{l ~ p(l | g, x)} [ d/dg log p(g, l | x) ] ,

so the marginal row contribution is the **average** of the joint row score over
the local conditional, not its value at the profile.

**How the local conditional is obtained exactly.** The arm's forward model is
deterministic, so the clean row posterior has no density to sample by
quadrature. But the diffused one is that manifold blurred by `lambda`, and
blurring is a convolution, so for rejection-sampling draws `theta_k` from
`p_0(theta | x_i)`

    p_t(g, l | x_i) = (1/M) sum_k N(g; g_k, lambda^2) N(l; l_k, lambda^2)

exactly, hence `p_t(l | g, x_i) = sum_k w_k N(l; l_k, lambda^2)` with
`w_k` proportional to `N(g; g_k, lambda^2)`. The conditional is therefore an
exact Gaussian mixture over prior-rejection draws: no Metropolis, no quadrature
over the 3-D local space, and no trained model anywhere in the reference path.
4,000 draws per row (tolerance 0.05), 256 local samples per grid point, 801
grid points.

Two estimators were run, because a single one cannot be trusted here. The
`sampled` one redraws `k` and `eps` at every grid point -- unbiased, but its
error is white noise in `g`, which the integration turns into a random walk.
The `topk` one takes the `samples` largest-weight draws at each `g` with a
fixed `eps` per draw index -- coupled and smooth in `g`, but it truncates the
mixture and is biased when the effective sample size exceeds `samples`.
Reporting only the more favourable of the two would be cherry-picking.

`W1/sd` against the ABC tall reference, bimodal regime, improved network
(`Kbpre`), 4,000 draws per row, 256 local samples, 801 grid points:

| n | lambda | profile | marginal (sampled) | marginal (topk) | ESS |
|---|---|---|---|---|---|
| 8 | 0.179 | 0.777 | **0.101** | **0.274** | 1522 |
| 8 | 0.101 | 0.918 | 0.639 | 0.640 | 920 |
| 8 | 0.055 | 0.863 | 0.847 | 0.935 | 520 |
| 8 | 0.032 | 0.756 | 0.956 | 0.914 | 305 |
| 16 | 0.179 | 0.260 | 0.129 | 0.378 | 1507 |
| 16 | 0.101 | 0.919 | 0.838 | 0.978 | 961 |
| 16 | 0.032 | 0.843 | 1.067 | 1.119 | 322 |
| 32 | 0.179 | 0.222 | 0.577 | 0.440 | 1514 |
| 32 | 0.101 | 0.669 | 1.172 | 1.318 | 971 |
| 32 | 0.032 | 0.716 | 1.346 | 1.413 | 328 |

**One cell shows a robust gain.** At n = 8, `lambda = 0.179` both estimators
agree that marginalizing the locals is worth a large factor -- 7.7x by the
unbiased estimator, 2.8x by the coupled one. There, profiling rather than the
score is the dominant remaining error.

**Everywhere else the experiment does not support that conclusion.** At
`lambda <= 0.1` both estimators say marginalization is worth nothing. At
n = 16, `lambda = 0.179` they disagree in *direction* (0.129 against 0.378). At
n = 32 both say the marginal reconstruction is worse than the profile at every
noise level, which cannot be true of the exact quantity they approximate and so
marks the oracle as unreliable there.

Two explanations for the small-`lambda` behaviour were proposed and both tested
and rejected: *weight collapse* (ESS falls only 5x, from 1522 to 305 out of
4000, which is not enough) and *estimator noise* (the coupled estimator was
written specifically to remove it and gives the same answer, 0.640 against
0.639 at n = 8, `lambda = 0.101`).

**Answer to Question 9, in the task's own terms: `B ≈ A` over most of the
regime, with one clear exception.** By the decision rule stated in Part VII
that means **score training, not local marginalization, remains the dominant
bottleneck** for inverse kinematics at the noise scales the composition runs
in. The exception -- small n at large `lambda` -- is real and worth 3-8x, but
it is one corner of the ladder.

This weakens rather than strengthens the case for a moment/covariance head: a
head approximates exactly the quantity measured here, and that quantity is
worth a large factor in one cell and nothing in nine. Before building one, the
low-`lambda` oracle should be made reliable -- more draws, a variance-reduced
estimator, and agreement between two independent estimators as the acceptance
criterion -- because as it stands the measurement cannot see the regime that
matters.

Caveats: single training seed per IK arm; bimodal regime only; and the n = 32
rows bottomed out at 1,840 accepted draws for the hardest row against 4,000
requested.
