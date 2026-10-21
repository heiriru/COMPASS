# Composing a mixture instead of a Gaussian, with K discovered

Measured 2026-09-10 on `inverse_kinematics`, frozen checkpoint
`legacy_h16d2_n10000_six_weighted_log_sigma_e10000`, the same dataset and the
same ABC referee as `docs/nonsymmetric_composition.md`. New code, all outside
`src/compass/`:

| file | what |
|---|---|
| `inverse_kinematics/experiments/mode_mixture_core.py` | the machinery, problem-independent and batched |
| `inverse_kinematics/experiments/mode_mixture.py` | the arm: dataset, referee, controls |
| `inverse_kinematics/experiments/run_mode_mixture.sh` | the arm runs |
| `Gaussian_mixtures_compositional_score_modelling/mode_mixture_gm.py` | the exact-score benchmark of section 7 |

## 1. Why a mixture

`docs/nonsymmetric_composition.md` attributes the non-symmetric composition
error to the composition rule rather than the network: the Gaussian rules
summarise each observation by a mean and a covariance, which cannot carry
unequal branch weights, so they return a near-symmetric answer (mass 0.467
against a reference 0.243).

The family is what fails, not the machinery. Gaussians are not closed under the
operations composition needs; Gaussian mixtures are — the product of GMMs is a
GMM, a GMM convolved with the perturbation kernel is a GMM, and a GMM divided by
a Gaussian is a GMM wherever the composed precision stays positive. So replace
the one-Gaussian summary of `p(theta | x_i)` by a K-component mixture and
compose those, with the existing arrow elimination applied once per component
tuple. At K = 1 it degenerates to the existing rule, which is the internal
control below.

Three things have to be supplied that the Gaussian rules never form. All three
consume only the network's raw joint score.

**K, by continuation.** At the top of the noise ladder the kernel dominates and
each factor is unimodal, so K = 1 is known to be right there for free. Walking
sigma down, each tracked mode is Newton-refined and then probed for a split
along its flattest curvature directions; whatever survives deduplication is the
mode set. K is discovered per observation per level, never supplied.

**Curvature, from the Jacobian.** Tweedie's second-order identity gives the
Hessian from the score Jacobian. Rows do not mix inside the network, so four
forward-mode JVPs give the whole `4 x 4` block for every row and every mode,
whatever the count.

**Weights, by score path integration.** The quantity every score-space rule is
missing is the log-partition of each factor, and relative log density between
two modes is a line integral of the score along any path joining them. That is a
1-D quadrature — one batched call for all `K-1` paths — not an ODE solve, and no
maximisation: the endpoints are the modes already in hand. Branch mass is
invariant under convolution, so the integral can be taken where the network is
calibrated rather than at the terminal noise where it is worst.

## 2. What works

`|x1| < 0.15`, n = 8, seed 99. Reference mass on the positive branch **0.243**.

**K is recovered exactly, twice.** Continuation returns K = 2 for all eight
observations, and an independent 64-start multi-start at the working sigma
returns K = 2 for all eight. The bifurcation is visible in the trace: for
observation 0 the top Hessian eigenvalue rises monotonically from -0.31 to
-0.02 in units of `1/sigma^2` while K = 1, and the split lands at sigma = 0.78.

**The path integral is good enough at moderate sigma, and only there.** The
straight-versus-bent path discrepancy — the direct measurement of how far the
learned score is from being a gradient — is 0.05-0.97 nats at sigma = 0.40 and
0.29-1.89 at sigma = 0.25, on a quantity whose whole dynamic range is a few
nats. The relative antisymmetry of the score Jacobian moves the same way,
0.10-0.20 against 0.14-0.25. Composed with path-integral weights the answer is
**better** than with weights taken from the network's own draws (0.189 against
0.218), so at sigma = 0.40 the weights are not the binding constraint.

**The composition itself works.** With the components refit to the network's own
draws (row `em` below), 256 tuples enumerate to an effective 2.0 components and
land at **W1/sigma 0.189 with branch mass 0.292** — against `gauss_jacobian` at
0.497 / 0.467. That is 2.6x better than the rule it replaces, and the branch
mass matches the target `network_product` (0.288) rather than the symmetric
answer.

**The control confirms K is what buys it.** Forcing K = 1 through the identical
pipeline gives per-observation branch masses of 0.087-0.916 — each observation
committing entirely to one branch, which is exactly what one Gaussian must do —
and composes to W1/sigma **0.512**, indistinguishable in magnitude from
`gauss_jacobian`'s 0.497. Note the direction differs: `gauss_jacobian`
symmetrises (mass 0.467, too high), K = 1 over-commits (0.036, too low). Same
size of error, opposite sign, same cause — one component cannot hold two
branches.

## 3. What fails: the Laplace curvature

The method as specified — components from the score Jacobian — does **not**
work. It returns W1/sigma **4.33**, worse than doing nothing.

| curve | mean err/sigma | width | W1/sigma | mass > 0 |
|---|---:|---:|---:|---:|
| joint ABC reference | 0 | 1.000 | 0 | 0.243 |
| `network_product` (target) | +0.065 | 1.059 | 0.107 | 0.288 |
| **`mode_mixture`, EM components** | +0.159 | 0.953 | **0.189** | **0.292** |
| `mode_mixture`, EM, weights from draws | +0.195 | 0.970 | 0.218 | 0.310 |
| `gauss_jacobian` | +0.492 | 0.992 | 0.497 | 0.467 |
| `gauss_hierarchical` | +0.485 | 0.888 | 0.522 | 0.468 |
| `mode_mixture`, K = 1 control | -0.489 | 0.564 | 0.512 | 0.036 |
| **`mode_mixture`, Laplace components** | +4.330 | 0.351 | **4.333** | 0.998 |

The cause is measured, not inferred. Per observation, the Laplace mixture is a
poor fit to the factor it summarises — W1 against the network's own
`p(a | x_i)` of **0.18-0.60** in units of that posterior's own sd — while the
same K components refit by EM sit at **0.025-0.039**. The Laplace widths are
systematically **1.5-1.9x too wide**: observation 0 gives sd `[0.444, 0.488]`
against EM's `[0.304, 0.273]`.

That over-widening is the small-lambda score deficit of
`docs/score_calibration.md` in curvature form — a shrunk score is a flattened
log density is an inflated variance — and the `(n-1)` prior division turns it
into a catastrophe. The composed precision is

    tau* = sum_i tau_i - (n-1) / v_prior

With prior sd 0.5 and n = 8 the subtracted term is 28. EM components at sd 0.27
contribute `sum_i tau_i` around 110 and the cancellation is harmless. Laplace
components at sd 0.47 contribute about 36, leaving `tau*` around 8 — a 78%
cancellation — and the composed mean, `sum_i tau_i m_i / tau*`, is then the
component means amplified roughly fourfold. The answer lands at +1.875 with the
reference at -0.260.

This is the same `(n-1)` cancellation that killed score-space composition in
section 7 of `docs/nonsymmetric_composition.md`, reached from a different
direction. It is not specific to Laplace: **any** representation whose
components are as wide as the prior will be destroyed by it at n = 8. The
enumeration makes the condition explicit (a tuple with `tau* <= 0` is improper
and is dropped) but the failure here is quantitative, not a sign flip — all 256
tuples were proper.

Reading the weights off higher up the ladder helps in the right direction and is
not enough on its own: at sigma = 0.25 components with sigma = 0.40 weights give
1.948 against 2.348, and branch mass 0.117 against 0.000. Widths, not weights,
are the binding constraint.

## 4. Where this leaves the idea

Three of the four pieces work and are independently verified: K is discovered
correctly, the split cascade is visible in the eigenvalue trace, the path
integral carries branch weight at moderate noise, and the tuple enumeration
composes correctly and beats `gauss_jacobian` 2.6x. The piece that fails is the
one that reads component *widths* off the score Jacobian.

The working configuration is therefore not yet pilot-free: it takes K, the
component locations and the branch weights from the score, and the widths from
a sampler call. That is still materially cheaper and more scalable than
`marginal_product`, which needs a density tabulated on a grid and binds at about
three shared dimensions — fitting `K` Gaussians in `d` dimensions does not. But
it is not the grid-free method the derivation promised, and on this benchmark it
does not beat `marginal_product` or `langevin_fnpse` (section 5).

Cost: **16.9s** against 9s for `gauss_jacobian`, down from 257s. The original
number was an implementation artifact, and the profile said so: one observation
took **1746 score calls at a mean batch of 6.3 rows**, while a batch of 512 rows
costs exactly what a batch of 1 does (2.83 ms against 2.83 ms). The run was
launch-bound and the batch dimension was free. Three changes, worth 15x
together:

* every network call is now batched across **all** observations at once, on a
  flat row set with an `owner` vector. Rows do not mix inside the network --
  the premise the four-JVP Hessian already relies on -- and that holds across
  observations exactly as it does within one;
* `score_and_hessian` is no longer recomputed three or four times per level for
  the split probe, the deduplication, the saddle filter and the trace;
  `newton_refine` returns the curvature its last iteration computed;
* the Newton budget after the first level dropped from 12 steps to 4
  (`--refine-steps`), since continuation means every later level starts next to
  its answer.

Verified numerically identical: main 4.314, K = 1 control 0.512, EM 0.189 before
and after the refactor, at 16.5s instead of 257s.

The remaining floor is forward-mode AD: one JVP costs 19.3 ms against 3.4 ms for
a plain forward pass, so the four-JVP Hessian is 72 ms -- 21x a score
evaluation, where the arithmetic says 8x. The attention backend is not the cause
(3.38 ms under `MATH` against 3.36 ms fused). A central finite difference would
cost four to eight plain passes instead, at the price of having to justify a
step size. Not done.

## 5. Replicates

Working configuration (`--component-source em --sigma-weights 0.40`), same
checkpoint, datasets and ABC referee as `docs/nonsymmetric_composition.md`.
W1/sigma against the joint reference:

| case | `gauss_jacobian` | `gauss_hier` | `langevin_fnpse`* | `marginal_product`* | **`mode_mixture`** |
|---|---:|---:|---:|---:|---:|
| selected n=8 seed 99 (mass 0.243) | 0.497 | 0.522 | 0.237 | **0.115** | 0.189 |
| selected n=8 seed 101 (mass 0.467) | 0.251 | 0.347 | 0.184 | **0.036** | 0.194 |
| selected n=16 seed 99 (mass 0.273) | 0.436 | 0.564 | 0.261 | **0.159** | 0.272 |
| exact-symmetric n=8 (mass 0.500) | **0.288** | 0.333 | 0.230 | *0.541* | *0.430* |

\* quoted from `docs/nonsymmetric_composition.md`, not re-run here. The two rows
that were re-run reproduce it exactly (`gauss_jacobian` 0.497, reference mass
0.243), so the pipeline is the same one.

Branch mass, the quantity the Gaussian summary cannot carry:

| case | reference | `gauss_jacobian` | `network_product` | **`mode_mixture`** |
|---|---:|---:|---:|---:|
| n=8 seed 99 | 0.243 | 0.467 | 0.288 | **0.292** |
| n=8 seed 101 | 0.467 | 0.458 | 0.440 | 0.405 |
| n=16 seed 99 | 0.273 | 0.428 | 0.214 | 0.170 |
| exact-symmetric | 0.500 | 0.582 | 0.767 | 0.710 |

**It beats both Gaussian rules on all three asymmetric cases** — 2.6x, 1.3x and
1.6x — and it does so by getting the branch mass approximately right instead of
symmetrising. It is over-confident too, though less so than the rules (width
0.87 / 0.76 against `gauss_hierarchical`'s 0.72 / 0.68 at n = 8 / 16), and like
every method here it gets worse with n.

**It is roughly a tie with `langevin_fnpse`** (0.189/0.194/0.272 against
0.237/0.184/0.261) and it loses to `marginal_product` everywhere. So the mixture
is a real improvement on the Gaussian rules and is not currently the best method
on this benchmark.

**On exactly symmetric data it loses to the rules, for the documented reason.**
It tracks `network_product` (mass 0.710 against that target's 0.767) and
`network_product` is itself wrong here: the network's +0.028 per-observation
bias toward the true branch is faithfully amplified by any method that composes
honestly, and the rules' symmetrising bias accidentally hides it. A method that
carries branch weight correctly must lose this case as long as that bias exists.

**K was recovered exactly in every run**, by continuation and by an independent
64-start multi-start, including the one observation at n = 16 that is genuinely
unimodal — both methods returned K = 1 for it and K = 2 for the other fifteen.

## 6. The exact-score benchmark: the method is right, the network is the problem

Section 3 attributes the Laplace failure to component widths, and names the
network as the suspect via `docs/score_calibration.md`. That is a hypothesis
until the same method is run with a score that has no deficit.
`Gaussian_mixtures_compositional_score_modelling` supplies one: `oracle.py` is
the **exact** diffused score at every noise level, and the tall posterior is a
closed-form two-component Gaussian mixture at every `n`, so there is no ABC
referee and no KDE anywhere in the comparison.

The test is sharper than "exact instead of approximate". Every diffused
conditional in that model *is* a two-component Gaussian mixture, so the
representation the method assumes is not an approximation at all: the modes are
the component means, the Jacobian at a mode is the exact component precision,
and the score is a true gradient, so the path integral is exactly path
independent. Anything the method gets wrong there, it gets wrong for reasons
that have nothing to do with score quality.

`asymmetric`, n = 8, exact branch weights 0.250 / 0.750:

| curve | mean | sd | W1/sd | branch w_+ | seconds |
|---|---:|---:|---:|---:|---:|
| exact `p(g | x_1..n)` | +0.6213 | 1.0515 | 0 | 0.750 | — |
| **`mode_mixture`, Laplace components** | **+0.6254** | **1.0492** | **0.004** | **0.752** | 3.8 |
| `mode_mixture`, K = 1 control | +0.0003 | 0.0259 | 1.116 | 0.483 | 1.4 |
| `gauss_jacobian` | +0.0750 | 2.2722 | 1.391 | 0.515 | 30.1 |
| `gauss_hierarchical` | +0.5688 | 2.1967 | 1.176 | 0.626 | 22.9 |
| `langevin_fnpse` | +1.4037 | 1.7841 | 1.092 | 0.811 | 80.4 |

**With an exact score the Laplace variant is exact** -- W1/sd 0.004, branch
weight 0.752 against 0.750, and 8x faster than the rule it replaces. The
diagnostics behave exactly as the derivation says they should: the relative
antisymmetry of the score Jacobian is **3e-17**, machine zero, and the
straight-versus-bent path discrepancy is **3e-5**, which is quadrature error and
nothing else. The working-sigma sweep is flat where it should be -- W1/sd
0.000 / 0.004 / 0.019 at sigma 0.40 / 0.20 / 0.10 -- which is the claim that
branch mass is invariant along the ladder, measured rather than asserted.

So section 3's attribution holds: **the arm's failure is the network's, not the
method's.** The Gaussian rules, meanwhile, fail here for the same two reasons
they fail on the arm and more visibly, because there is no reference noise to
hide behind: they symmetrise the branch weight (0.515 and 0.626 against 0.750)
*and* they are over-dispersed by more than a factor of two (sd 2.27 and 2.20
against 1.05). The K = 1 control fails in the way one Gaussian per observation
must: it collapses to a spike of width 0.026 at g = 0, between the branches.

Figure: `artifacts/mode_mixture/global_posterior_asymmetric_n8.png`.

## 7. The same benchmark with a learned score

Swapping `--score-source oracle` for `--score-source network` changes the score
and nothing else -- same dataset, same exact target, same method, same
referee -- on the one trained checkpoint this directory has
(`tokenized_h32d3wide_n200000_six_weighted_log_sigma`, 268,545 parameters).

| curve | mean | sd | W1/sd | branch w_+ | seconds |
|---|---:|---:|---:|---:|---:|
| exact `p(g | x_1..n)` | +0.6213 | 1.0515 | 0 | 0.750 | — |
| `mode_mixture`, **exact** score | +0.6254 | 1.0492 | **0.004** | 0.752 | 3.9 |
| `mode_mixture`, **learned** score | -0.3137 | 0.2137 | **1.138** | 0.002 | 13.6 |
| `marginal_product` (the network's own factors) | -0.5590 | 0.8366 | 1.159 | 0.183 | 65.9 |
| `gauss_jacobian` | -0.3374 | 2.2565 | 1.615 | 0.414 | 12.2 |
| `gauss_hierarchical` | +0.6227 | 2.2118 | 1.200 | 0.627 | 14.5 |
| `langevin_fnpse` | +1.3207 | 1.8818 | 1.071 | 0.784 | 39.7 |
| `mode_mixture`, K = 1 control | +0.0010 | 0.0259 | 1.116 | 0.494 | 5.6 |

**The method goes from 0.004 to 1.138, and that is the network, not the rule.**
`marginal_product` composes the network's own single-observation posteriors
through the exact factorisation -- no composed score, no Gaussian summary, no
correction rule -- and lands at **1.159**. That is the ceiling any method built
on this checkpoint can be held to, and `mode_mixture` is at it. The branch weight
tells the same story from the other side: the truth is 0.750, the network's own
factors composed exactly give 0.183, and the mixture gives 0.002. Both are on the
wrong branch; the mixture is more confident about it because it is not smearing
the answer the way the sampler-based rules do.

(`langevin_fnpse`'s 1.071 slips just under that ceiling. A ceiling from a
*consistent* estimator is not a bound on an inconsistent one: its sd is 1.88
against a true 1.05, and errors that large can cancel in W1.)

**Two measurements about the network fall out, both only visible because the
truth is known here.**

*The learned score field has spurious modes.* Continuation and the 64-start
multi-start **agree** with each other -- both return K = [2, 4, 2, 2, 2, 4, 4, 4]
-- and both are wrong: the true diffused conditional is a two-component mixture
for every observation, at every noise level. Four of eight observations have four
genuine maxima in the learned field that do not exist in the exact one. That
independently confirms the arm's K = 3-8 in section 8 as real structure in
learned score fields rather than a Newton artifact, which could not be
established on the arm because the arm has no known K.

*This checkpoint's score cannot couple the two latent coordinates at all.*
`d(score_g)/dl` is **exactly zero**: shifting `l` by 5.0 leaves `score_g`
unchanged to every printed digit, and the Jacobian's off-diagonal entries are
0.0000000 at every point tested, by forward-mode AD and by central differences
independently (the two agree to 7e-5 on the diagonal, so the AD path is sound).
The oracle at the same points gives `J01 = J10 = 1.345`. The true posterior has
`Cov(g, l) = a V` with `a = 0.62`, so a strictly diagonal curvature cannot
represent it. The arm's legacy checkpoint does **not** have this property -- its
Hessian off-diagonals are order 1 -- and both architectures build the same
`key_padding_mask = 1 - c`, so the mechanism is not established and is not
claimed here. It is a measurement, and it is enough to explain why every
curvature-based rule is over-dispersed by a factor of two on this benchmark.

Figures: `artifacts/mode_mixture/{oracle,network}/global_posterior_*.png`. The
learned-score figure carries the exact-score mixture as a dashed reference, so
the two runs can be read against one target.

## 7b. Why the learned score is decoupled, and what fixing it buys

The exact zero in section 7 is **structural, not a training failure**, and two
facts combine to produce it:

* COMPASS masks latent tokens out of attention (`key_padding_mask = 1 - c` in
  both trunks), so no token can attend *to* a latent;
* `TokenizedTransformer` normalises **per token**, `LayerNorm(hidden_size)`.

With both, there is no path at all from `l` to `score_g` when the two are
latent. The network learns `p(l | g, x)` perfectly well -- there `g` is observed
and therefore a valid attention key -- but it **cannot represent the correlation
in `p(g, l | x)`**, which is the object composition consumes.
`ConditionTransformer` (`arch="legacy"`, what the arm runs) normalises across
tokens *and* hidden, `LayerNorm((nodes_size, hidden_size))`. That joint
normalisation is a cross-token operation and, attention being blocked, the only
one available. Hence order-1 off-diagonals on the arm and exactly zero here.

Tested by a controlled swap -- same 200k simulations, same `h32d3wide`, same
`six_weighted`, same `log_sigma`, same method, same referee, only the trunk:

| | tokenized | legacy |
|---|---:|---:|
| parameters | 268,545 | 597,664 |
| **latent coupling** (mean abs off-diagonal / diagonal) | **0.0** | **0.11-0.16** |
| K found (truth is 2 everywhere) | [2,**4**,2,2,2,**4**,**4**,**4**] | [2,2,2,2,2,2,2,2] |
| path-integral discrepancy (nats) | 2e-5 | 0.67-0.80 |
| branch w_+ across sigma 0.40/0.20/0.10 | 0.316/0.002/0.001 | 0.976/0.543/0.016 |
| `marginal_product` (the network's ceiling) | 1.159 | **0.827** |
| **`mode_mixture` W1/sd** | **1.138** | **1.097** |

**The mechanism claim is confirmed; the accuracy claim is refuted.** Coupling
goes from an exact zero to 0.11-0.16 and the spurious modes disappear -- K is
now 2 for every observation, matching both the truth and the multi-start check,
where the tokenized net had four genuine maxima on half the observations. The
network's own ceiling improves from 1.159 to 0.827. And `mode_mixture` barely
moves: **1.138 to 1.097**, against 0.004 with the exact score.

So latent coupling was necessary and is not sufficient. The legacy trunk trades
a structural blindness for a different defect: its score is **no longer
conservative** -- the straight-versus-bent path discrepancy jumps from 2e-5 to
0.67-0.80 nats, on a quantity whose whole range is a few nats -- so the branch
weights stop being stable in the working sigma, swinging 0.976 -> 0.543 -> 0.016
as sigma goes 0.40 -> 0.20 -> 0.10. With the exact score that sweep is flat to
three digits. Unlike the tokenized run, `mode_mixture` is now **not** at the
network's ceiling (1.097 against 0.827), and the path integral is where the
remaining gap lives.

Caveat: legacy carries 597k parameters against tokenized's 268k, so this is not
a capacity-matched comparison. It does not affect the coupling conclusion -- a
parameter count does not turn an exact zero into a nonzero derivative -- but the
0.827-against-1.159 ceiling difference could be capacity as easily as trunk.

Figure: `artifacts/mode_mixture/legacy/global_posterior_network_asymmetric_n8.png`.

## 7c. Recovering the coupling from conditional masks

Section 7b fixed the decoupling by changing the trunk. There is a cheaper fix
that needs no retraining and no architecture change, and it follows from the
same identity the benchmark's `marginal_product` uses. At sigma = 0,

    log p(theta | x) = log p(theta_A | theta_B, x) + log p(theta_B | x)

and the second term does not contain `theta_A`, so **every second derivative
with at least one `theta_A` index is carried by the conditional alone**. Two
role masks therefore span the whole Hessian: the shared-latent mask gives the
shared rows, the locals-latent mask gives the local rows, and each supplies its
own estimate of the off-diagonal block. Both are entries of `SIX_CONDITIONALS`
(`p(g|l,x)` and `p(l|g,x)`), so any network trained on that menu has seen them.

This works where the all-latent mask cannot, for a specific reason: under a
per-token-normalising trunk a *latent* coordinate is masked out of attention and
can influence nothing, but a *conditioned* one is a valid attention key.
Measured on the tokenized checkpoint at sigma = 0.2:

    joint mask c=[0,0,1]:  d s_g/d l = 0.0        d s_l/d g = 0.0
    cond  mask c=[1,0,1]:  d s_l/d g = 1.723      d s_l/d l = -1.958

The sigma correction is exact per block rather than approximate. The conditioned
coordinates enter clean, so only the latent block carries the kernel:

    C      = -d s_A/d theta_A = (Sigma_A|B + sigma^2 I)^-1
    D      =  d s_A/d theta_B = C  dm/d theta_B
    Lambda_AA = (C^-1 - sigma^2 I)^-1
    Lambda_AB = -Lambda_AA C^-1 D

Implemented as `--curvature-source conditional`. It costs the same `D` JVPs as
the joint route, on a batch twice the size, and the batch is free.

**Validated against the exact score**: identical to the joint route to three
digits (W1/sd 0.000 / 0.004 / 0.019 at sigma 0.40 / 0.20 / 0.10) and the two
masks' estimates of the off-diagonal agree to **1e-8**, machine precision, which
is what the identity says must happen.

**On the tokenized network** (`marginal_product`, the network's ceiling, is
1.159 with branch weight 0.183; the truth is 0.750):

| | joint curvature | conditional curvature |
|---|---:|---:|
| latent coupling | **0.0** | **0.16-0.23** |
| branch weight w_+ | 0.002 | **0.176** (ceiling 0.183) |
| cross-mask disagreement | — | **0.32-0.44** |
| composed sd | 0.2137 | 0.0288 |
| **W1/sd** | 1.138 | **1.125** |

**Correction.** The first conditional-curvature run reported W1/sd 1.125 and
branch weight 0.176. Those numbers came from a run containing **NaN** component
widths and should be discarded. Each block is floored positive on its own, but a
matrix assembled from separately-estimated blocks need not be positive definite:
a too-large off-diagonal makes the Schur complement negative, the inverse gets a
negative diagonal entry, and the width comes out NaN. The assembled matrix is now
eigen-floored, which is the only place that constraint can be imposed.

## 7d. The spurious modes were a Cartesian product, and the same identity fixes them

The composed density under the tokenized net was a spike indistinguishable from
the K = 1 control, which is what a decoupled score has to produce. Reading the
mode set off directly says why:

    obs1:  g in {+2.2211360931, -2.1649563313}
           l in {+1.4244123697, -0.9622133374}
           modes = all four combinations, identical to the last digit

A **perfect Cartesian product**. With `s_g` a function of `g` alone and `s_l` of
`l` alone, the implied density is `exp(u(g)) exp(v(l))` and its critical points
are *every pairing* of the zeros of `u'` with the zeros of `v'`. The network's
`l`-marginal is bimodal -- it marginalises over both `g` branches -- so two
branches times two gives four modes, three of them spurious cross-terms pairing
`g` from one branch with `l` from the other. Deduplication was right to keep
them: they are genuinely 2.8-10.8 apart in Mahalanobis distance and they are
genuine maxima of the density the network implies.

The fix is the same identity as section 7c, applied to the **score** rather than
only to the Hessian:

    d/d theta_A  log p(theta | x)  =  d/d theta_A  log p(theta_A | theta_B, x)

so the whole joint score is assembled component-wise from the role masks
(`--score-assembly conditional`). Then `s_g` genuinely depends on `l` and the
product structure cannot arise.

**The assembly must not drive the weights.** It is exact only at `sigma = 0`:
the masked network returns `p_sigma(theta_A | theta_B clean, x)`, so at
`sigma > 0` the assembled field is not the gradient of anything. On the exact
score its path discrepancy rises from 8e-3 to 0.27 nats and W1/sd from 0.004 to
0.035. Modes and curvature come from the assembly; the branch weights stay on
the all-latent mask, which is self-consistent for its own density. With that
split the oracle returns to 0.000 / 0.004 exactly.

**On the tokenized network**, at the working sigma 0.40:

| | joint score | assembled score |
|---|---:|---:|
| K found (truth 2) | [2,**4**,2,2,2,**4**,**4**,**4**] | **[2,2,2,2,2,2,2,2]** |
| composed sd (truth 1.0515) | 0.2137 | **0.6240** |
| branch weight w_+ (truth 0.750) | 0.002 | **0.463** |
| **W1/sd** | 1.138 | **0.900** |

The spike is gone, K is right, and the answer is now **below the network's own
`marginal_product` ceiling of 1.159** -- the first time anything in this file
has beaten that, which is possible because the mixture uses a two-component
parametric fit where `marginal_product` uses a KDE.

**One new caution.** The 64-start multi-start now returns K = [2,2,3,3,3,4,4,3]
where the continuation returns [2]*8. Under the joint score the two agreed; they
no longer do. The continuation's answer matches the truth, but a disagreement
between two discovery methods is exactly the signal that called the earlier
result into question, and it is unresolved.

## 7e. The remaining spike was the mixture prior, not the network

Even with K correct and the score assembled, the composed density was still a
spike at ``g = 0`` indistinguishable from the K = 1 control. Evaluating the two
halves of the composition separately says why, and the component widths were
*correct* to 5% (fitted median 0.357 against a true 0.374), so this is not a
width error:

                        at g = 0      at g = +1.2
    sum log f_i           -152.2          -63.6      numerator prefers +1.2 by 88.6
    -(n-1) log prior      +136.7          +41.8      prior division prefers 0 by 94.9
    total                  -15.5          -21.7      g = 0 wins by 6.2

The composed density peaked at ``g = 0.000``, exactly between the two prior
modes, where the true posterior has no mass whatsoever. The prior there is
``~e^-19.5``, so its inverse seventh power is ``e^+136``: the composition becomes
the difference of two ~150-nat logarithms and a 6-nat residue picks the answer.

**A Gaussian prior can be divided out anywhere; a mixture prior cannot.**
Conditioned on a prior component the prior *is* a single Gaussian with no
interior low-density region, and

    p(g | x_1..n, c)  ~  p(g | c)^(1-n)  prod_i p(g | x_i, c)

is ordinary Gaussian algebra that never evaluates anything between the modes. So
every factor component is assigned to the prior component that best explains it,
tuples are enumerated **within** a component, and the branch weights fall out of
the same normalisation (`--prior-composition mixture`, now the default here).
Tuples that mix components across observations are not merely improbable: the
component index is shared by construction, exactly like the shared coordinate,
so they are not part of the factorisation at all. A factor with no component of
its own on a branch lends its nearest one rather than making the branch
impossible -- without that fallback the K = 1 control has no consistent branch
assignment and the composition is undefined.

**Tokenized network, sigma_w = 0.40, all three fixes:**

| curve | mean | sd | W1/sd | w_+ |
|---|---:|---:|---:|---:|
| exact | +0.6213 | 1.0515 | 0 | 0.750 |
| **mode mixture** | +0.2009 | 1.5674 | **0.704** | 0.541 |
| mode mixture, K = 1 control | -4.4805 | 0.0195 | 4.852 | 0.000 |
| `marginal_product` (network ceiling) | -0.5590 | 0.8366 | 1.159 | 0.183 |
| `gauss_jacobian` | -0.3374 | 2.2565 | 1.615 | 0.414 |
| `gauss_hierarchical` | +0.6227 | 2.2118 | 1.200 | 0.627 |

Cumulatively **1.138 -> 0.704**, well under the network's own ceiling, and the
density is bimodal in roughly the right places instead of a spike. The K = 1
control is now 4.852 and visibly a different object -- which is the point: a
control that tracks the method is a control that is measuring nothing.

**Still open.** The working-sigma sweep is unstable -- 0.704 / 1.452 / 0.536 /
2.040 at sigma 0.40 / 0.20 / 0.10 / 0.05, against 0.000 / 0.004 / 0.019 / 0.366
with the exact score -- so the branch weight is still not being determined
reliably, and 0.704 should be read as one point of a scattered sweep rather than
a converged number. The multi-start/continuation disagreement of section 7d is
also unresolved.

## 7f. The conditional assembly does not transfer to the arm

Run on `inverse_kinematics`, everything else held fixed:

| arm, sigma_w = 0.40 | joint | assembled + conditional |
|---|---:|---:|
| joint-mask coupling | **0.127** | 0.315 |
| cross-mask disagreement | — | **1.02** |
| K, continuation | [3,5,8,5,3,8,8,8] | [8,8,8,8,8,8,8,8] (all at the cap) |
| K, 64-start multi-start | [2,2,2,2,2,2,2,2] | **[33,27,14,35,40,4,1,7]** |
| W1/sigma | 4.314 | 3.280 |

**It makes the arm worse, and the reason is that the arm never had the defect.**
The whole point of the assembly is that under a per-token-normalising trunk a
latent coordinate cannot influence another's score, so the joint-mask Hessian is
*exactly* diagonal. The arm runs the legacy trunk, whose joint-mask coupling is
already **0.127** -- there is nothing to recover. What the assembly adds instead
is its cost: above ``sigma = 0`` the assembled field is not a gradient, and on a
four-dimensional latent block with a learned score that field has dozens of
zeros. The multi-start count goes from a clean [2]*8 to between 1 and 40 modes
per observation, and the cross-mask disagreement is **1.02** -- the two masks
disagree about the off-diagonal by as much as the off-diagonal itself.

**The rule this gives.** Use `--score-assembly conditional` only where the
joint-mask coupling diagnostic is ~0, i.e. on a per-token-normalising trunk with
a fully latent block. The diagnostic is already reported, so the decision needs
no extra run: 0.0 on the tokenized Gaussian-mixture checkpoint, 0.127 on the
arm's legacy one. It is a targeted repair for a specific architectural blindness,
not a general improvement.

The arm's own defect is still the one in section 3 -- Laplace component widths
1.5-1.9x too wide -- and the best arm configuration remains
`--component-source em` at W1/sigma 0.095.

## 8. Two machinery defects the exact score exposed

Neither would have been findable on the arm, because on the arm a wrong answer
is always attributable to the network.

**The trust radius was tied to sigma, and must be tied to the local width.** A
step cap of `trust_radius * sigma` shrinks down the ladder while the distances
Newton has to travel do not, so split probes cannot walk back into the parent
they came from and survive deduplication as modes that are really unconverged
iterates. Measured: K = 6 at sigma = 0.05 where the truth is 2, and raising the
Newton budget from 6 steps to 40 pulled it back to 2-3. Capping by the widest
local direction instead fixes it outright -- K = 2 at every sigma in the sweep,
and the sweep only then became flat. The local width already contains sigma (the
noise kernel contributes `1/sigma^2` to the precision), so the two rules agree
high up the ladder and differ only where the old one failed.

**Dropping low-weight components is wrong, however tempting.** With the trust
radius fixed, mode discovery on the *arm* began finding K = 3-8 against a
64-start multi-start's 2 -- shallow local maxima of the **learned** score field
that the exact field does not have, and that the saddle test cannot remove
because they are genuine maxima. Cutting components below a share of their own
factor's mass removes them and improves the arm. It is still wrong: composition
is a product over `n` factors, so a factor that says `1e-4` at a point is not
interchangeable with one that says `0` there. On the exact-score benchmark a
1e-3 threshold dropped a single legitimate component and took the composed
branch weight from **0.752 to 0.000**. The knob exists (`--min-weight`) and
defaults to off. A spurious mode needs a filter that tests whether it is
spurious, not how little mass it carries; that filter is not written.

**Composition is now an incremental fold with a beam**, because `prod_i K_i`
reaches 9.2e5 on the arm once mode discovery converges properly. Components are
carried as unnormalised quadratics, the `n` factors multiplied first and the
`n-1` prior divisions applied once at the end so that `tau` increases
monotonically and a partial mass is a sound ranking. `effective_components` is
reported so a run where the beam binds is visible.

## 9. Not done

* Four arm datasets and one exact-score dataset, one checkpoint, no SBC or
  coverage run. This is a pilot, not calibration evidence.
* The exact-score benchmark was run at one regime (`asymmetric`) and one `n`.
  Its `symmetric` / `antisymmetric` pair is the sharper test -- same target
  density, opposite evidence -- and has not been run.
* No attempt to fix the Laplace width directly. The obvious candidates are
  measuring curvature at a larger sigma and deconvolving (the deconvolution is
  implemented and on by default; it is not sufficient), or fitting the width
  from a short score-only line search along each mode's principal axes instead
  of from the Jacobian at the mode.
* The beam fold replaced full enumeration, but the pruning is a plain beam, not
  the split-tree pruning the design calls for; a tuple that looks weak partway
  cannot recover.
* **Mode discovery on the arm now disagrees with itself**: continuation returns
  K = 3-8 where a 64-start multi-start returns 2, and the extra components carry
  enough mass to push `effective_components` to 569. They are genuine maxima of
  the learned score field. Section 7 says why the obvious filter is wrong; the
  right one is not written, and until it is, the arm numbers in sections 3 and 5
  are from the configuration that predates the trust-radius fix. Section 7 shows
  the same spurious modes with a learned score where the true K is known.
* The learned-score comparison is two checkpoints, one regime, one `n`. Section
  7b answers "does a coupling architecture close the gap" with no, and leaves
  the next question open: the legacy trunk's score is not conservative, and
  whether that is trainable away (more capacity, more data, a conservative
  parameterisation) or intrinsic has not been tested.
* No capacity-matched trunk comparison; legacy is 2.2x the parameters.
* `--curvature-source conditional` is not tried on the arm. The arm's roles are
  1 shared + 3 locals, so the same two role masks span its Hessian and the code
  is dimension-general, but it has not been run.
* The cross-mask disagreement of section 7c is a network diagnostic with no
  calibration behind it yet: 32-44% is clearly bad and 1e-8 is clearly good, and
  nothing establishes where the line is.
* Not tried on `Partial_Pooling`.
* Not wired into `run_inference.METHODS` or any grader.
